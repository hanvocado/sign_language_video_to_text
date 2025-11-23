"""
Web Application for Vietnamese Sign Language Recognition (Real-time)
Using Flask + Flask-SocketIO for WebSocket communication
Receives frames from client, processes with MediaPipe, predicts with LSTM model

IMPORTANT: Follows EXACT preprocessing from training data_loader.py
- Extract keypoints from MediaPipe
- Normalize: center at wrist midpoint, scale by bbox diagonal
- Model inference

REAL-TIME APPROACH (like ASL demo):
- Continuous frame capture at 25 FPS
- Sliding window: every 25 frames triggers prediction
- Background processing doesn't block main thread
- Predictions sent back immediately when ready
"""

import os
import sys
import base64
import io
import json
import threading
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Import from project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.train import build_model, BiLSTM
from src.utils.utils import load_label_map, load_checkpoint
from src.config.config import DEVICE

# Suppress warnings
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Flask app setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    engineio_logger=False,
    logger=False,
    ping_timeout=60,
    ping_interval=25,
)

# Global variables
model = None
label_map = None
CONFIDENCE_THRESHOLD = 0.30
NUM_FRAMES = 25
FEATURE_DIM = 225
MIN_PREDICTION_CONFIDENCE = 0.75  # STRICT (75%): Only accept high-confidence predictions

# Real-time processing state
processing_lock = threading.Lock()
is_processing = False

# Motion detection state - prevent repeated predictions of same gesture
last_predicted_keypoints = None
last_prediction = None
motion_threshold = 0.15  # Sensitivity: minimum distance for new gesture detection

# IMPORTANT: Initialize with "NONE" (no gesture detected yet)
last_prediction = {
    'label': 'NONE',
    'confidence': 0.0
}

# Logging
from src.utils.logger import ProjectLogger
logger = ProjectLogger.get_logger("web_app", console_output=True)


def extract_keypoints(results):
    """
    Extract 225-dim keypoints from MediaPipe results.
    Format: [pose(99) + left_hand(63) + right_hand(63)]
    
    Args:
        results: MediaPipe Holistic results
    
    Returns:
        np.array of shape (225,)
    """
    pose = []
    lh = []
    rh = []

    # Pose: 33 landmarks × 3 = 99 dims
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose.extend([lm.x, lm.y, lm.z])
    else:
        pose = [0.0] * 99

    # Left hand: 21 landmarks × 3 = 63 dims
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else:
        lh = [0.0] * 63

    # Right hand: 21 landmarks × 3 = 63 dims
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else:
        rh = [0.0] * 63

    return np.array(pose + lh + rh, dtype=np.float32)


def detect_motion(prev_keypoints, current_keypoints, motion_threshold=0.15):
    """
    Detect if there's significant motion between two frames.
    Avoids predicting the same static gesture multiple times.
    
    IMPORTANT: Skip frames with NO hands detected (all keypoints = 0)
    
    Args:
        prev_keypoints: np.array of shape (225,) - previous frame keypoints
        current_keypoints: np.array of shape (225,) - current frame keypoints
        motion_threshold: Minimum distance to consider as motion (0-1 range)
    
    Returns:
        tuple: (has_motion: bool, distance: float)
    """
    # Check if current frame has NO hands (all zeros)
    if np.allclose(current_keypoints, 0):
        logger.debug("⏭️ [BG] Skipping: No hands detected in frame")
        return False, 0.0
    
    # Check if previous frame also has no hands
    if prev_keypoints is None or np.allclose(prev_keypoints, 0):
        # First real frame with hands detected
        return True, 1.0
    
    # Calculate L2 distance between frames
    diff = np.linalg.norm(current_keypoints - prev_keypoints)
    
    # If distance > threshold, it's a new gesture
    has_motion = diff > motion_threshold
    
    if has_motion:
        logger.debug(f"✅ Motion detected: distance={diff:.4f} (threshold={motion_threshold})")
    else:
        logger.debug(f"⏭️ No motion: distance={diff:.4f} (threshold={motion_threshold})")
    
    return has_motion, diff


def normalize_keypoints(seq, left_wrist_idx=15, right_wrist_idx=16):
    """
    CRITICAL: Normalize keypoints EXACTLY as done in training.
    
    - Center at midpoint between wrists (left_wrist_idx=15, right_wrist_idx=16)
    - Scale by bounding box diagonal
    
    Args:
        seq: np.array of shape (N, 225) - raw keypoints
        left_wrist_idx: Index of left wrist in pose landmarks (15)
        right_wrist_idx: Index of right wrist in pose landmarks (16)
    
    Returns:
        np.array of shape (N, 225) - normalized keypoints
    """
    # Reshape to (N, 75, 3) for easier manipulation
    num_landmarks = seq.shape[1] // 3  # 225 / 3 = 75
    seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)

    # Get wrist positions
    lw = seq3d[:, left_wrist_idx, :2]   # (N, 2) - left wrist x,y
    rw = seq3d[:, right_wrist_idx, :2]  # (N, 2) - right wrist x,y
    
    # Detect if wrists are missing (all zeros)
    lw_missing = np.all(lw == 0, axis=1)
    rw_missing = np.all(rw == 0, axis=1)
    both_missing = lw_missing & rw_missing
    
    # Calculate reference point (center between wrists)
    ref = (lw + rw) / 2  # (N, 2)
    
    # For frames where both wrists missing, use mean of all keypoints
    if np.any(both_missing):
        mean_all = np.mean(seq3d[:, :, :2], axis=1)
        ref[both_missing] = mean_all[both_missing]
    
    # CENTER: Subtract reference point from all x,y coordinates
    seq3d[:, :, 0] -= ref[:, 0].reshape(-1, 1)
    seq3d[:, :, 1] -= ref[:, 1].reshape(-1, 1)

    # SCALE: Normalize by bounding box diagonal
    min_c = np.min(seq3d[:, :, :2], axis=1)  # (N, 2)
    max_c = np.max(seq3d[:, :, :2], axis=1)  # (N, 2)
    scale = np.linalg.norm(max_c - min_c, axis=1)  # (N,) - diagonal length
    scale[scale == 0] = 1  # Avoid division by zero
    seq3d[:, :, :2] /= scale.reshape(-1, 1, 1)

    # Reshape back to (N, 225)
    return seq3d.reshape(seq.shape[0], -1)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Serve the main page"""
    return render_template('index.html', num_frames=NUM_FRAMES)


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("✅ Client connected")
    emit('connect_response', {'data': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("❌ Client disconnected")


@socketio.on('process_frames')
def process_image(data_images):
    """
    Real-time frame processing with background threading.
    
    REAL-TIME APPROACH (from ASL demo):
    - Client sends frames continuously (no button click needed)
    - Server processes in background thread
    - Predictions streamed back immediately when ready
    - Doesn't block main thread - can handle continuous streams
    
    Pipeline:
    1. Decode base64 frames
    2. Extract keypoints using MediaPipe (225-dim each)
    3. Stack to (25, 225)
    4. NORMALIZE keypoints (center + scale)
    5. Model inference
    6. Return prediction to client
    
    Args:
        data_images: List of 25 base64 encoded JPEG images
    """
    global is_processing
    
    # Skip if already processing (avoid queue buildup)
    if is_processing:
        logger.debug("⏭️ Skipping: Already processing previous batch")
        return
    
    # Start background processing thread
    thread = threading.Thread(
        target=_process_frames_background,
        args=(data_images,)
    )
    thread.daemon = True
    thread.start()
    emit('response_back', {'label': 'PROCESSING', 'confidence': 0.0})


def _process_frames_background(data_images):
    """
    Background thread for frame processing with MOTION DETECTION.
    
    Key changes from continuous sliding-window:
    - Detect if gesture is NEW (motion > threshold)
    - If same gesture (no motion), DON'T predict again
    - Only emit prediction when significant motion detected
    
    This prevents: 1 gesture → 1 prediction (instead of 25+ predictions)
    """
    global is_processing, last_predicted_keypoints, last_prediction
    
    with processing_lock:
        is_processing = True
        try:
            if not isinstance(data_images, list) or len(data_images) == 0:
                logger.error(f"❌ Invalid input: expected list of frames, got {type(data_images)}")
                socketio.emit('response_back', {'label': 'ERROR', 'confidence': 0.0})
                return

            frame_count = len(data_images)
            logger.info(f"🔄 [BG] Processing {frame_count} frames in background...")

            # Step 1: Create MediaPipe processor (static_image_mode=True for batch)
            holistic = mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            frame_keypoints = []

            # Step 2: Extract keypoints from each frame
            for i, data in enumerate(data_images):
                try:
                    # Decode base64 to PIL Image
                    img_data = base64.b64decode(data[23:])  # Remove "data:image/jpeg;base64," prefix
                    img = Image.open(io.BytesIO(img_data))
                    frame = np.array(img)  # Already RGB from PIL

                    # Process with MediaPipe (frame is already RGB)
                    results = holistic.process(frame)
                    keypoints = extract_keypoints(results)
                    frame_keypoints.append(keypoints)

                except Exception as e:
                    logger.error(f"[BG] Frame {i}: {e}")
                    # Add zero keypoints for failed frame
                    frame_keypoints.append(np.zeros(225, dtype=np.float32))

            holistic.close()

            # Check we have frames
            if len(frame_keypoints) == 0:
                socketio.emit('response_back', {'label': 'ERROR', 'confidence': 0.0})
                return

            # Step 3: Stack to (N, 225)
            arr = np.array(frame_keypoints, dtype=np.float32)
            logger.debug(f"[BG] Stacked keypoints shape: {arr.shape}")

            # Pad to NUM_FRAMES if needed
            if arr.shape[0] < NUM_FRAMES:
                pad = np.zeros((NUM_FRAMES - arr.shape[0], FEATURE_DIM), dtype=np.float32)
                arr = np.vstack([arr, pad])
            elif arr.shape[0] > NUM_FRAMES:
                arr = arr[:NUM_FRAMES]

            # *** SIMPLIFIED: No complex motion detection ***
            # Just check if we have hands in buffer
            current_keypoints = arr[-1]  # Last frame in buffer
            
            # Check if we have hands in ANY frame of the buffer
            max_keypoints = np.max(np.abs(arr), axis=0)
            has_hands = np.max(max_keypoints) > 0.01
            
            if not has_hands:
                logger.info(f"⏭️ [BG] NO HANDS detected - skipping prediction")
                return

            # Step 4: NORMALIZE (CRITICAL!)
            arr_norm = normalize_keypoints(arr)

            # Step 5: Model inference
            X = torch.from_numpy(arr_norm).unsqueeze(0).float().to(DEVICE)
            
            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Step 6: Get prediction
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            pred_label = label_map[pred_idx]

            # STRICT confidence check (≥75% only)
            # Low confidence = ignore (like ASL demo with threshold)
            if confidence < MIN_PREDICTION_CONFIDENCE:
                logger.info(f"⏭️ [BG] Low confidence ({confidence:.4f} < {MIN_PREDICTION_CONFIDENCE}%) - ignoring")
                return
            
            logger.info(f"✅ [BG] Prediction: {pred_label} (confidence: {confidence:.4f})")
            prediction_result = {
                'label': pred_label,
                'confidence': confidence
            }

            # Update state
            last_predicted_keypoints = current_keypoints.copy()
            last_prediction = prediction_result

            # Emit result back to client
            socketio.emit('response_back', prediction_result)

        except Exception as e:
            logger.error(f"❌ [BG] Error: {e}", exc_info=True)
            socketio.emit('response_back', {'label': 'ERROR', 'confidence': 0.0})
        finally:
            is_processing = False


def load_model_and_weights(model_path, label_map_path, device='cpu'):
    """Load trained model and label map"""
    try:
        # Load label map
        label_list = load_label_map(label_map_path)
        num_classes = len(label_list)
        logger.info(f"Loaded {num_classes} classes: {label_list}")

        # Load checkpoint directly to see what model was saved
        ck = load_checkpoint(model_path, device=device)
        
        # Infer model structure from checkpoint keys
        model_keys = set(ck['model_state'].keys())
        
        if 'rnn.' in ' '.join(model_keys):
            # GRU or basic RNN model
            logger.info("Detected GRU/RNN model from checkpoint")
            model = build_model(
                num_classes=num_classes,
                input_dim=FEATURE_DIM,
                hidden_dim=256,
                num_layers=1,
                dropout=0.3,
                model_type='gru'
            ).to(device)
        elif 'lstm.' in ' '.join(model_keys):
            # LSTM or BiLSTM model
            logger.info("Detected LSTM model from checkpoint")
            model = build_model(
                num_classes=num_classes,
                input_dim=FEATURE_DIM,
                hidden_dim=256,
                num_layers=2,
                dropout=0.3,
                model_type='bilstm'
            ).to(device)
        else:
            logger.warning("Could not detect model type, trying BiLSTM")
            model = build_model(
                num_classes=num_classes,
                input_dim=FEATURE_DIM,
                hidden_dim=256,
                num_layers=2,
                dropout=0.3,
                model_type='bilstm'
            ).to(device)

        # Load weights with strict=False to ignore mismatches
        model.load_state_dict(ck['model_state'], strict=False)
        model.eval()

        logger.info(f"✅ Model loaded from {model_path}")
        return model, label_list

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


if __name__ == '__main__':
    # Paths
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'models',
        'checkpoints',
        'best.pth'
    )
    LABEL_MAP_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'models',
        'checkpoints',
        'label_map.json'
    )

    # Verify
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(LABEL_MAP_PATH):
        logger.error(f"Label map not found: {LABEL_MAP_PATH}")
        sys.exit(1)

    # Load
    model, label_map = load_model_and_weights(MODEL_PATH, LABEL_MAP_PATH, device=DEVICE)

    logger.info(f"🚀 Starting server at http://127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
