"""
Web Application for Vietnamese Sign Language Recognition (Real-time)
Using Flask + Flask-SocketIO for WebSocket communication
Receives frames from client, processes with MediaPipe, predicts with LSTM model
"""

import os
import sys
import base64
import io
import json
import numpy as np
import cv2
import torch
import mediapipe as mp
from pathlib import Path
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Import from project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import build_model
from src.utils.utils import load_label_map, load_checkpoint
from src.config.config import DEVICE

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
    manage_acks=False,
    packet_namespace=True
)

# Global variables
model = None
label_map = None
holistic = None

# Suppress MediaPipe warnings
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)
CONFIDENCE_THRESHOLD = 0.30
NUM_FRAMES = 25  # Variable for sequence length (changeable)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CHANNELS = 3
FEATURE_DIM = 225

# Logging
from src.utils.logger import ProjectLogger
logger = ProjectLogger.get_logger("web_app", console_output=True)


def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe results.
    Returns: (225,) numpy array
    """
    pose, lh, rh = [], [], []

    # Pose landmarks: 33 × 3 = 99
    has_pose = results.pose_landmarks is not None
    if has_pose:
        for lm in results.pose_landmarks.landmark:
            pose.extend([lm.x, lm.y, lm.z])
    else:
        pose = [0.0] * 33 * 3

    # Left hand: 21 × 3 = 63
    has_lh = results.left_hand_landmarks is not None
    if has_lh:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else:
        lh = [0.0] * 21 * 3

    # Right hand: 21 × 3 = 63
    has_rh = results.right_hand_landmarks is not None
    if has_rh:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else:
        rh = [0.0] * 21 * 3

    keypoints = np.array(pose + lh + rh, dtype=np.float32)
    
    # Log detection status
    if not (has_pose or has_lh or has_rh):
        logger.debug("⚠️ No landmarks detected by MediaPipe!")
    
    return keypoints


def normalize_keypoints(seq, left_wrist_idx=15, right_wrist_idx=16):
    """
    Normalize keypoints (CRITICAL FOR MATCHING TRAINING DATA):
    - Center at midpoint between wrists
    - Scale by bounding box diagonal
    
    Args:
        seq: (N, 225) array of keypoints
        left_wrist_idx: Index of left wrist in pose landmarks (15)
        right_wrist_idx: Index of right wrist in pose landmarks (16)
    
    Returns:
        (N, 225) normalized keypoints
    """
    # Reshape to (N, 75, 3) for easier manipulation
    num_landmarks = seq.shape[1] // 3
    seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)

    # Get wrist positions (left wrist at pose idx 15, right at 16)
    lw = seq3d[:, left_wrist_idx, :2]  # (N, 2)
    rw = seq3d[:, right_wrist_idx, :2]  # (N, 2)
    
    # Check if both wrists are missing (all zeros)
    lw_missing = np.all(lw == 0, axis=1)
    rw_missing = np.all(rw == 0, axis=1)
    both_missing = lw_missing & rw_missing
    
    # Calculate reference point (center between wrists)
    ref = (lw + rw) / 2  # (N, 2)
    
    # For frames with missing wrists, use mean of all keypoints
    if np.any(both_missing):
        mean_all = np.mean(seq3d[:, :, :2], axis=1)
        ref[both_missing] = mean_all[both_missing]
    
    # Center at reference point
    seq3d[:, :, 0] -= ref[:, 0].reshape(-1, 1)
    seq3d[:, :, 1] -= ref[:, 1].reshape(-1, 1)

    # Scale by bounding box diagonal
    min_c = np.min(seq3d[:, :, :2], axis=1)  # (N, 2)
    max_c = np.max(seq3d[:, :, :2], axis=1)  # (N, 2)
    scale = np.linalg.norm(max_c - min_c, axis=1)  # (N,)
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


@socketio.on('frame_count')
def handle_frame_count(data):
    """Handle frame count announcement"""
    count = data.get('count', 0)
    logger.info(f"📬 Client sending {count} frames...")


@socketio.on('process_frames')
def process_image(data_images, *args):
    """
    Process frames received from client.
    
    Args:
        data_images: List of base64 encoded images
    
    Workflow:
        1. Decode frames from base64
        2. Extract keypoints using MediaPipe
        3. Stack into sequence
        4. Run model prediction
        5. Emit result back to client
    """
    try:
        logger.info(f"🔔 EVENT RECEIVED: process_image() called with {len(data_images) if isinstance(data_images, list) else '?'} frames")
        
        if not isinstance(data_images, list):
            logger.error(f"❌ data_images is not a list! Type: {type(data_images)}")
            emit('response_back', {'label': 'ERROR', 'confidence': 0.0, 'message': 'Invalid data format'})
            return
        
        # Create fresh MediaPipe instance for each batch to avoid timestamp errors
        holistic_processor = mp_holistic.Holistic(
            static_image_mode=True,  # CRITICAL: Set to True for batch processing without timestamp tracking
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize buffer for keypoints
        frame_keypoints = []

        # Process each frame
        logger.info(f"Processing {len(data_images)} frames...")
        for i, data in enumerate(data_images):
            try:
                # Decode base64 image
                img_data = base64.b64decode(data[23:])  # Remove "data:image/jpeg;base64," prefix
                img = Image.open(io.BytesIO(img_data))

                # Convert to OpenCV format (BGR)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # Extract keypoints using MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic_processor.process(image)
                keypoints = extract_keypoints(results)
                frame_keypoints.append(keypoints)
                logger.debug(f"  Frame {i+1}: extracted {np.count_nonzero(keypoints)} non-zero keypoints")

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue
        
        # Close the holistic processor
        holistic_processor.close()

        # Check if we have keypoints
        if len(frame_keypoints) == 0:
            emit('response_back', {
                'label': 'ERROR',
                'confidence': 0.0,
                'message': 'Could not extract keypoints'
            })
            return

        # Stack keypoints into sequence: (NUM_FRAMES, 225)
        arr = np.array(frame_keypoints, dtype=np.float32)

        # DEBUG: Check if all keypoints are zero
        non_zero_count = np.count_nonzero(arr)
        logger.info(f"Raw keypoints - Shape: {arr.shape}, Non-zero elements: {non_zero_count}/{arr.size}, Min: {arr.min():.6f}, Max: {arr.max():.6f}")
        
        if non_zero_count < 10:  # Very few non-zero elements = no detection
            logger.warning(f"⚠️ WARNING: Very few non-zero keypoints! Detection might be failing.")

        # Pad if necessary
        if arr.shape[0] < NUM_FRAMES:
            pad = np.zeros((NUM_FRAMES - arr.shape[0], FEATURE_DIM), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif arr.shape[0] > NUM_FRAMES:
            arr = arr[:NUM_FRAMES]

        # ✅ CRITICAL: Normalize keypoints (must match training preprocessing!)
        arr = normalize_keypoints(arr)
        
        logger.info(f"After normalize - Shape: {arr.shape}, Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")

        # Convert to tensor: (1, NUM_FRAMES, 225)
        X = torch.from_numpy(arr).unsqueeze(0).float().to(DEVICE)

        # Model inference
        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Get prediction
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        pred_label = label_map[pred_idx]
        
        # Log all predictions for debugging
        logger.info(f"🔍 Model outputs - Best: {pred_label} ({confidence:.4f})")
        logger.debug(f"   All probabilities: {[(label_map[i], p) for i, p in enumerate(probs)]}")
        # Check confidence threshold
        if confidence > CONFIDENCE_THRESHOLD:
            result_label = pred_label
        else:
            result_label = 'NONE'

        # Log prediction
        logger.info(f"Prediction: {result_label} (confidence: {confidence:.4f})")

        # Emit result to client
        emit('response_back', {
            'label': result_label,
            'confidence': confidence,
            'all_probs': {label_map[i]: float(probs[i]) for i in range(len(label_map))}
        })

    except Exception as e:
        logger.error(f"Error in image processing: {e}", exc_info=True)
        emit('response_back', {
            'label': 'ERROR',
            'confidence': 0.0,
            'message': str(e)
        })


@socketio.on('config_update')
def update_config(data):
    """
    Update configuration parameters from client.
    
    Args:
        data: Dictionary with 'num_frames' and/or 'confidence_threshold'
    """
    global NUM_FRAMES, CONFIDENCE_THRESHOLD
    
    if 'num_frames' in data:
        NUM_FRAMES = int(data['num_frames'])
        logger.info(f"Updated NUM_FRAMES to {NUM_FRAMES}")
    
    if 'confidence_threshold' in data:
        CONFIDENCE_THRESHOLD = float(data['confidence_threshold'])
        logger.info(f"Updated CONFIDENCE_THRESHOLD to {CONFIDENCE_THRESHOLD}")
    
    emit('config_updated', {
        'num_frames': NUM_FRAMES,
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })


def load_model_and_weights(model_path, label_map_path, device='cpu'):
    """
    Load model and label map.
    
    Args:
        model_path: Path to .pth checkpoint
        label_map_path: Path to label_map.json
        device: 'cuda' or 'cpu'
    
    Returns:
        model, label_map
    """
    try:
        # Load label map
        label_list = load_label_map(label_map_path)
        num_classes = len(label_list)

        # Build model
        model = build_model(
            num_classes=num_classes,
            input_dim=FEATURE_DIM,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3
        ).to(device)

        # Load checkpoint
        ck = load_checkpoint(model_path, device=device)
        model.load_state_dict(ck['model_state'])
        model.eval()

        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Label map: {label_list}")

        return model, label_list

    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Configuration
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

    # Verify paths exist
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Please train the model first using src.model.train")
        sys.exit(1)

    if not os.path.exists(LABEL_MAP_PATH):
        logger.error(f"Label map not found at {LABEL_MAP_PATH}")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model from {MODEL_PATH}")
    model, label_map = load_model_and_weights(MODEL_PATH, LABEL_MAP_PATH, device=DEVICE)

    # Initialize MediaPipe
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Start server
    logger.info(f"Starting server on http://127.0.0.1:5000")
    logger.info(f"Configuration: NUM_FRAMES={NUM_FRAMES}, DEVICE={DEVICE}")

    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
