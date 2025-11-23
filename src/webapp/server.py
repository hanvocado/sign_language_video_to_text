"""
Web Application for Vietnamese Sign Language Recognition (Real-time)
Using Flask + Flask-SocketIO for WebSocket communication

LOGIC: Follows infer_realtime.py approach
- State Machine (FSM): waiting -> recording -> predicting
- Motion detection: triggers recording start/stop
- Frame sampling: remove idle frames, then sample uniformly
- Keypoint normalization: using common_functions.py
"""

import os
import sys
import base64
import io
import json
import threading
import time
import numpy as np
import torch
import cv2
import mediapipe as mp
from pathlib import Path
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from collections import deque

# Import from project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.train import build_model
from src.utils.utils import load_label_map, load_checkpoint
from src.utils.common_functions import (
    extract_keypoints,
    normalize_keypoints,
    sample_frames,
    is_pose_detected,
)
from src.config.config import (
    DEVICE,
    MODEL_TYPE,
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    BIDIRECTIONAL,
    FEATURE_DIM,
    SEQ_LEN,
)
from src.webapp.config import ModelConfig

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

# Model inference
FEATURE_DIM = 225
SEQ_LEN = 64  # Match infer_realtime.py default
MIN_PREDICTION_CONFIDENCE = 0.35  # Require at least 35% confidence to make prediction

# FSM States
STATE_WAITING = "waiting"
STATE_RECORDING = "recording"

# Motion detection parameters
MOTION_THRESHOLD = 3
STILL_FRAMES_REQUIRED = 8  # Match infer_realtime.py
DISPLAY_DURATION = 1.5

# Real-time state
frame_state = {
    'state': STATE_WAITING,
    'segment': [],
    'still_count': 0,
    'prev_gray': None,
    'last_pred_label': None,
    'last_pred_conf': None,
    'last_pred_time': 0,
}

# Thread-safe lock
state_lock = threading.Lock()
logger = logging.getLogger(__name__)


def setup_logger():
    """Setup logger for debugging"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    handler = logging.FileHandler(os.path.join(log_dir, "webapp.log"))
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def load_model_and_weights(model_path, label_map_path, device='cpu'):
    """Load model and label map"""
    logger.info(f"Loading model from {model_path}")
    label_list = load_label_map(label_map_path)
    
    # Build model with config
    model = build_model(
        MODEL_TYPE,
        INPUT_DIM,
        HIDDEN_DIM,
        len(label_list),
        NUM_LAYERS,
        DROPOUT,
        BIDIRECTIONAL,
    ).to(device)
    
    # Load checkpoint
    ck = load_checkpoint(model_path, device=device)
    model.load_state_dict(ck['model_state'], strict=False)
    model.eval()
    
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"Classes: {label_list}")
    
    return model, label_list


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    logger.info("✅ Client connected")
    emit('connect_response', {'data': 'Connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")


@socketio.on('frame')
def handle_frame(data):
    """
    Process individual frame from client
    - Detect motion
    - Extract keypoints
    - Update FSM state
    - Send predictions when ready
    """
    global model, label_list, holistic, frame_state
    
    if model is None or label_list is None:
        return
    
    try:
        # Decode image
        img_data = base64.b64decode(data['image'].split(',')[1])
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        with state_lock:
            # =====================================================
            # MOTION DETECTION
            # =====================================================
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if frame_state['prev_gray'] is None:
                frame_state['prev_gray'] = gray
                return
            
            motion = cv2.absdiff(frame_state['prev_gray'], gray)
            _, th = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)
            motion_level = th.mean()
            frame_state['prev_gray'] = gray
            
            movement = motion_level > MOTION_THRESHOLD
            
            # =====================================================
            # STATE MACHINE
            # =====================================================
            if frame_state['state'] == STATE_WAITING:
                if movement:
                    frame_state['state'] = STATE_RECORDING
                    frame_state['segment'] = []
                    frame_state['still_count'] = 0
                    logger.info(f"🎬 Recording started (motion level: {motion_level:.2f})")
            
            elif frame_state['state'] == STATE_RECORDING:
                # Extract keypoints from this frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                # Only process if pose was actually detected (check visibility scores)
                if is_pose_detected(results, visibility_threshold=0.5):
                    vec = extract_keypoints(results)  # from common_functions.py
                    frame_state['segment'].append(vec)
                    
                    # Update still counter
                    if movement:
                        frame_state['still_count'] = 0
                    else:
                        frame_state['still_count'] += 1
                else:
                    # No pose detected, reset still counter
                    frame_state['still_count'] = 0
                
                # Check if sign is complete (stopped moving)
                if frame_state['still_count'] >= STILL_FRAMES_REQUIRED:
                    # Only infer if we have enough frames (avoid padding artifacts)
                    if len(frame_state['segment']) >= 20:  # Minimum frames for reliable inference
                        # =====================================================
                        # INFERENCE
                        # =====================================================
                        raw_arr = np.array(frame_state['segment'], dtype=np.float32)
                        total_frames = len(raw_arr)
                        
                        logger.info(f"📊 Total frames in segment: {total_frames}")
                        logger.info(f"📊 Raw array shape: {raw_arr.shape}")
                        
                        # Frame sampling: remove idle frames
                        indices = sample_frames(total_frames, SEQ_LEN, mode="2")
                        sampled = raw_arr[indices]
                        
                        logger.info(f"📊 Sampled frames: {len(sampled)} | Sampled shape: {sampled.shape}")
                        
                        # Normalize keypoints using common_functions.py
                        sampled = normalize_keypoints(sampled)
                        
                        # Convert to tensor
                        X = torch.from_numpy(sampled).unsqueeze(0).to(DEVICE)
                        
                        logger.info(f"📊 Tensor shape: {X.shape}")
                        
                        # Model inference
                        with torch.no_grad():
                            logits = model(X)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        
                        pred = int(probs.argmax())
                        label = label_list[pred]
                        conf = float(probs[pred])
                        
                        # Log detailed probabilities for all classes
                        probs_str = " | ".join([f"{label_list[i]}: {probs[i]:.4f}" for i in range(len(label_list))])
                        logger.info(f"Full probs: {probs_str}")
                        
                        # Apply confidence threshold
                        if conf < MIN_PREDICTION_CONFIDENCE:
                            logger.info(f"⚠️  Low confidence ({conf:.4f} < {MIN_PREDICTION_CONFIDENCE}), not making prediction")
                        else:
                            frame_state['last_pred_label'] = label
                            frame_state['last_pred_conf'] = conf
                            frame_state['last_pred_time'] = time.time()
                            
                            logger.info(f"✅ Prediction: {label} ({conf:.4f}) | Frames: {total_frames}")
                        
                        # Emit prediction to client
                        emit('prediction', {
                            'label': label,
                            'confidence': float(conf),
                            'frames': total_frames,
                        }, broadcast=False)
                    
                    # Reset FSM
                    frame_state['state'] = STATE_WAITING
                    frame_state['segment'] = []
                    frame_state['still_count'] = 0
    
    except Exception as e:
        logger.error(f"❌ Error processing frame: {str(e)}")
        import traceback
        traceback.print_exc()


@socketio.on('status')
def handle_status():
    """Send current state to client"""
    with state_lock:
        current_state = {
            'state': frame_state['state'],
            'segment_size': len(frame_state['segment']),
            'still_count': frame_state['still_count'],
        }
    
    emit('status_response', current_state, broadcast=False)


def initialize_app(model_path, label_map_path):
    """Initialize the app with model and label map"""
    global model, label_list, holistic
    
    setup_logger()
    logger.info("=" * 60)
    logger.info("WEBAPP STARTING - Vietnamese Sign Language Recognition")
    logger.info("=" * 60)
    
    # Load model
    model, label_list = load_model_and_weights(model_path, label_map_path, device=DEVICE)
    
    # Initialize MediaPipe
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    logger.info(f"MediaPipe initialized")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"SEQ_LEN: {SEQ_LEN}, FEATURE_DIM: {FEATURE_DIM}")


if __name__ == '__main__':
    # Use ModelConfig for paths (vsl_v1 model with 97.77% val_acc)
    model_path = str(ModelConfig.MODEL_PATH)
    label_map_path = str(ModelConfig.LABEL_MAP_PATH)
    
    initialize_app(model_path, label_map_path)
    
    logger.info(f"🚀 Starting server at http://127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
