"""
Ứng dụng Streamlit đơn giản - Nhận diện kí hiệu bằy tay từ camera
Không dùng streamlit-webrtc, dùng OpenCV trực tiếp
FIXED: - use_column_width → use_container_width
       - Session state cho is_running flag
       - Vòng lặp camera đơn giản
       - Cho phép chọn model từ UI
"""

import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
CHECKPOINT_DIR = "models/checkpoints"
LABEL_MAP_PATH = "models/checkpoints/label_map.json"

SEQUENCE_LENGTH = 64
NUM_FEATURES = 225
CONFIDENCE_THRESHOLD = 0.30  # Min confidence to add prediction (lowered from 0.60 because threshold doesn't help with model's "người" weakness)
FRAME_SKIP = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_available_checkpoints():
    """Get list of available checkpoints sorted by accuracy (HIGHEST FIRST)."""
    checkpoints = []
    checkpoint_dir = Path(CHECKPOINT_DIR)
    
    if checkpoint_dir.exists():
        pth_files = sorted(checkpoint_dir.glob("best_epoch*.pth"), reverse=True)
        for pth_file in pth_files:
            # Extract accuracy from filename (e.g., best_epoch25_acc0.7778.pth)
            try:
                parts = pth_file.stem.split('_')
                acc_str = parts[-1].replace('acc', '')
                acc = float(acc_str)
                checkpoints.append({
                    'path': str(pth_file),
                    'name': pth_file.name,
                    'accuracy': acc
                })
            except:
                pass
    
    # ✅ CRITICAL: Sort by accuracy (HIGHEST FIRST)
    checkpoints.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return checkpoints

# ============================================================================
# MODEL
# ============================================================================
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3, num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        factor = 2 if bidirectional else 1
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * factor, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        out, (hn, cn) = self.rnn(x)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits

# ============================================================================
# KEYPOINT EXTRACTOR
# ============================================================================
class KeyPointExtractor:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=False,  # ❌ DISABLED smoothing for testing
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.last_results = None
    
    def extract(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        self.last_results = results  # Store for visualization
        
        keypoints = []
        
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)
        
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)
        
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)
        
        return np.array(keypoints, dtype=np.float32), len(keypoints) == NUM_FEATURES
    
    def draw_landmarks(self, frame):
        """Draw pose and hand landmarks on frame."""
        if self.last_results is None:
            return frame
        
        h, w, c = frame.shape
        
        # Draw pose landmarks
        if self.last_results.pose_landmarks:
            for connection in mp.solutions.holistic.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_lm = self.last_results.pose_landmarks.landmark[start_idx]
                end_lm = self.last_results.pose_landmarks.landmark[end_idx]
                
                start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
            
            # Draw keypoints
            for lm in self.last_results.pose_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        # Draw left hand
        if self.last_results.left_hand_landmarks:
            for connection in mp.solutions.holistic.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_lm = self.last_results.left_hand_landmarks.landmark[start_idx]
                end_lm = self.last_results.left_hand_landmarks.landmark[end_idx]
                
                start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                
                cv2.line(frame, start_pos, end_pos, (255, 0, 0), 2)
            
            # Draw keypoints
            for lm in self.last_results.left_hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        # Draw right hand
        if self.last_results.right_hand_landmarks:
            for connection in mp.solutions.holistic.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_lm = self.last_results.right_hand_landmarks.landmark[start_idx]
                end_lm = self.last_results.right_hand_landmarks.landmark[end_idx]
                
                start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                
                cv2.line(frame, start_pos, end_pos, (0, 0, 255), 2)
            
            # Draw keypoints
            for lm in self.last_results.right_hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        return frame

# ============================================================================
# NORMALIZATION (MATCHES TRAINING PREPROCESSING FROM video2npy.py)
# ============================================================================
def normalize_sequence(sequence, wrist_left_idx=15, wrist_right_idx=16):
    """
    Normalize a full sequence (64, 225) to match EXACTLY training preprocessing from video2npy.py.
    This must be called AFTER buffer is full, on the entire sequence at once.
    
    sequence: np.array of shape (64, 225) = 64 frames × (pose(99) + left_hand(63) + right_hand(63))
    wrist_left_idx: Pose landmark index for left wrist (15)
    wrist_right_idx: Pose landmark index for right wrist (16)
    """
    try:
        sequence = np.array(sequence, dtype=np.float32).copy()
        
        # Reshape to (seq_len, num_landmarks, 3)
        seq_len = sequence.shape[0]
        num_landmarks = sequence.shape[1] // 3
        seq3d = sequence.reshape(seq_len, num_landmarks, 3)  # (64, 75, 3)
        
        # Find reference points (wrist joints) - SAME AS TRAINING
        wrist_left = seq3d[:, wrist_left_idx, :2]  # (64, 2)
        wrist_right = seq3d[:, wrist_right_idx, :2]  # (64, 2)
        
        # Check if valid - fallback to center
        if np.all(seq3d[:, wrist_left_idx, :2] == 0) and np.all(seq3d[:, wrist_right_idx, :2] == 0):
            center = np.mean(seq3d[:, :, :2], axis=1, keepdims=True)  # (64, 1, 2)
            ref_point = center
        else:
            # Use average of two wrists
            ref_point = (wrist_left + wrist_right) / 2  # (64, 2)
            ref_point = ref_point.reshape(-1, 1, 2)  # (64, 1, 2)
        
        # Translation: shift all landmarks to origin using wrist reference
        seq3d[:, :, 0] -= ref_point[:, 0, 0].reshape(-1, 1)
        seq3d[:, :, 1] -= ref_point[:, 0, 1].reshape(-1, 1)
        
        # Scale normalization: find min/max per frame
        min_coords = np.min(seq3d[:, :, :2], axis=1)  # (64, 2)
        max_coords = np.max(seq3d[:, :, :2], axis=1)  # (64, 2)
        scale = np.linalg.norm(max_coords - min_coords, axis=1)  # (64,)
        
        # Avoid division by zero
        scale[scale == 0] = 1.0
        scale = scale.reshape(-1, 1, 1)  # (64, 1, 1)
        
        # Apply scale normalization to x, y, z coordinates
        seq3d[:, :, :2] /= scale  # Normalize x, y
        seq3d[:, :, 2] /= scale[:, 0, 0].reshape(-1, 1)  # Normalize z to handle distance variation
        
        return seq3d.reshape(seq_len, -1).astype(np.float32)
    except Exception as e:
        logger.error(f"Sequence normalization error: {e}")
        import traceback
        traceback.print_exc()
        return sequence.astype(np.float32)

# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    if "model" not in st.session_state:
        st.session_state.model = None
    if "selected_checkpoint" not in st.session_state:
        # Default to best checkpoint
        checkpoints = get_available_checkpoints()
        if checkpoints:
            st.session_state.selected_checkpoint = checkpoints[0]['path']
        else:
            st.session_state.selected_checkpoint = None
    if "label_map" not in st.session_state:
        st.session_state.label_map = None
    if "keypoint_extractor" not in st.session_state:
        st.session_state.keypoint_extractor = None
    if "sequence_buffer" not in st.session_state:
        st.session_state.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    if "recognized_text" not in st.session_state:
        st.session_state.recognized_text = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = deque(maxlen=5)  # Last 5 predictions for smoothing
    if "last_recognition_time" not in st.session_state:
        st.session_state.last_recognition_time = None  # Timestamp for buffer reset delay
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "gesture_locked" not in st.session_state:
        st.session_state.gesture_locked = False  # Flag to lock gesture once recognized
    if "stable_prediction_count" not in st.session_state:
        st.session_state.stable_prediction_count = 0  # Count consecutive stable frames

# ============================================================================
# LOAD RESOURCES
# ============================================================================
def load_model(checkpoint_path):
    """Load model from specified checkpoint path."""
    try:
        logger.info(f"Loading model from {checkpoint_path}...")
        model = LSTMClassifier(input_dim=NUM_FEATURES, hidden_dim=256, num_classes=3, num_layers=2, dropout=0.3)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        logger.info(f"✅ Model loaded from {Path(checkpoint_path).name}")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

@st.cache_resource
def load_label_map():
    try:
        with open(LABEL_MAP_PATH) as f:
            data = json.load(f)
        if isinstance(data, list):
            label_map = {i: label for i, label in enumerate(data)}
        else:
            label_map = data
        logger.info(f"✅ Labels loaded: {list(label_map.values())}")
        return label_map
    except Exception as e:
        logger.error(f"❌ Error loading labels: {e}")
        return None

@st.cache_resource
def create_extractor():
    return KeyPointExtractor(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ============================================================================
# PREDICTION
# ============================================================================
def predict(sequence_buffer, model, label_map):
    if len(sequence_buffer) < SEQUENCE_LENGTH:
        return None, 0.0
    
    try:
        sequence = np.array(list(sequence_buffer), dtype=np.float32)
        
        # NORMALIZE sequence AFTER buffer is full (CRITICAL - matches training preprocessing)
        sequence = normalize_sequence(sequence)
        
        # NO scaler - data is already normalized to match training
        X = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(X)  # Raw logits
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            pred_idx = pred_idx.item()
            confidence = confidence.item()
        
        label_list = [label_map[i] for i in range(len(label_map))]
        prediction = label_list[pred_idx]
        
        return prediction, confidence
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def apply_temporal_smoothing(prediction, confidence, prediction_history, smooth_threshold=0.6):
    """
    Apply temporal smoothing to predictions using voting window.
    
    Args:
        prediction: Current prediction (str)
        confidence: Current confidence score (float)
        prediction_history: deque of past 5 predictions
        smooth_threshold: Minimum confidence + voting consistency needed
    
    Returns:
        smoothed_prediction: Final prediction after voting
        should_add: Boolean indicating if this should be added to recognized text
    """
    prediction_history.append((prediction, confidence))
    
    # Need at least 3 predictions for voting
    if len(prediction_history) < 3:
        return prediction, False
    
    # Get majority vote from last 5 predictions
    predictions_list = [p[0] for p in prediction_history if p[0] is not None]
    confidences_list = [p[1] for p in prediction_history if p[0] is not None]
    
    if not predictions_list:
        return None, False
    
    # Count predictions
    from collections import Counter
    pred_counts = Counter(predictions_list)
    most_common_pred, count = pred_counts.most_common(1)[0]
    
    # Only accept if:
    # 1. Appears in at least 3 out of last 5 predictions (60% consistency)
    # 2. Average confidence is reasonable
    consistency_ratio = count / len(predictions_list)
    avg_confidence = np.mean(confidences_list)
    
    if consistency_ratio >= 0.6 and avg_confidence >= smooth_threshold:
        return most_common_pred, True
    
    return prediction, False

# ============================================================================
# MAIN UI
# ============================================================================
def main():
    st.set_page_config(page_title="Sign Language Recognition", layout="wide")
    st.title("🤟 Nhận Diện Kí Hiệu Tay")
    st.markdown("Dùng camera để nhận diện các kí hiệu tay")
    
    init_session_state()
    
    # ========== SIDEBAR: Model selection ==========
    st.sidebar.header("⚙️ Cấu Hình")
    
    checkpoints = get_available_checkpoints()
    if checkpoints:
        checkpoint_options = {f"{cp['name']} (Acc: {cp['accuracy']:.2%})": cp['path'] for cp in checkpoints}
        default_idx = 0
        selected_name = st.sidebar.selectbox(
            "📦 Chọn Model:",
            list(checkpoint_options.keys()),
            index=default_idx,
            help="Chọn checkpoint model để sử dụng"
        )
        selected_path = checkpoint_options[selected_name]
        
        # Load model nếu thay đổi
        if st.session_state.selected_checkpoint != selected_path:
            st.session_state.selected_checkpoint = selected_path
            st.session_state.model = None  # Force reload
            st.rerun()
    else:
        st.sidebar.error("❌ Không tìm thấy model nào!")
        return
    
    confidence_threshold = st.sidebar.slider(
        "Ngưỡng tin cậy (Confidence Threshold)", 
        min_value=0.0, max_value=1.0, value=CONFIDENCE_THRESHOLD, step=0.05,
        help="Recommend 0.20-0.40. Model's 'người' class has lower accuracy (78%), threshold can't fix this."
    )
    
    show_landmarks = st.sidebar.checkbox(
        "🔷 Vẽ Keypoints",
        value=False,
        help="Hiển thị pose và hand landmarks trên video"
    )
    
    # Load resources
    if st.session_state.model is None:
        st.session_state.model = load_model(st.session_state.selected_checkpoint)
    
    st.session_state.label_map = load_label_map()
    st.session_state.keypoint_extractor = create_extractor()
    
    if st.session_state.model is None or st.session_state.label_map is None:
        st.error("❌ Không thể load model")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera")
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
    
    with col2:
        st.subheader("🎯 Dự Đoán")
        pred_placeholder = st.empty()
    
    st.subheader("📝 Kết Quả")
    result_placeholder = st.empty()
    
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
    
    with col_btn1:
        if st.button("▶️ BẮT ĐẦU"):
            st.session_state.is_running = True
            st.rerun()
    
    with col_btn2:
        if st.button("⏹️ DỪNG"):
            st.session_state.is_running = False
            st.session_state.gesture_locked = False
            st.session_state.stable_prediction_count = 0
            st.session_state.last_recognition_time = None
            st.rerun()
    
    with col_btn3:
        if st.button("➡️ TIẾP TỤC"):
            # 🔓 UNLOCK: Ready for next gesture
            st.session_state.gesture_locked = False
            st.session_state.sequence_buffer.clear()
            st.session_state.prediction_history.clear()
            st.session_state.last_prediction = None
            st.session_state.stable_prediction_count = 0
            st.session_state.last_recognition_time = None
            st.rerun()  # Refresh UI immediately
    
    with col_btn4:
        if st.button("📋 SAO CHÉP"):
            text = " ".join(st.session_state.recognized_text)
            st.info(f"Kết quả: {text}")
    
    with col_btn5:
        if st.button("🗑️ XÓA"):
            st.session_state.recognized_text = []
            st.session_state.sequence_buffer.clear()
            st.session_state.prediction_history.clear()
            st.session_state.last_prediction = None
            st.session_state.last_recognition_time = None
            st.session_state.gesture_locked = False
            st.session_state.stable_prediction_count = 0
            st.rerun()
    
    result_text = " ".join(st.session_state.recognized_text) if st.session_state.recognized_text else "(Chưa có)"
    result_placeholder.text_area("Kết quả:", value=result_text, height=100, disabled=True)
    
    if not st.session_state.is_running:
        st.info("📍 Nhấn **▶️ BẮT ĐẦU**")
        return
    
    st.warning("📹 Camera đang chạy...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Không thể mở camera")
        st.session_state.is_running = False
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_skip_counter = 0
    frames_processed = 0
    
    while st.session_state.is_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_skip_counter += 1
        
        if frame_skip_counter < FRAME_SKIP:
            continue
        
        frame_skip_counter = 0
        frames_processed += 1
        
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        keypoints, success = st.session_state.keypoint_extractor.extract(frame)
        
        # Draw landmarks nếu checkbox được check
        if show_landmarks:
            display_frame = st.session_state.keypoint_extractor.draw_landmarks(display_frame)
        
        if success:
            # Add RAW keypoints to buffer (normalization happens when buffer is full)
            st.session_state.sequence_buffer.append(keypoints)
            
            # ✅ SIMPLE LOGIC: Only predict if not yet locked
            if not st.session_state.gesture_locked and len(st.session_state.sequence_buffer) == SEQUENCE_LENGTH:
                prediction, confidence = predict(
                    st.session_state.sequence_buffer,
                    st.session_state.model,
                    st.session_state.label_map
                )
                
                if prediction and confidence > confidence_threshold:
                    # ✅ Count consecutive frames with SAME prediction
                    if st.session_state.last_prediction == prediction:
                        st.session_state.stable_prediction_count += 1
                    else:
                        st.session_state.stable_prediction_count = 1
                    
                    st.session_state.last_prediction = prediction
                    
                    # 🔒 LOCK when same prediction for 30+ consecutive frames
                    # (30 frames = 1 second stable = confident enough)
                    if st.session_state.stable_prediction_count >= 30:
                        st.session_state.recognized_text.append(prediction)
                        st.session_state.gesture_locked = True  # 🔒 LOCK!
                        pred_placeholder.success(
                            f"✅ {prediction} ({confidence:.1%}) [LOCKED] - "
                            f"Bấm TIẾP TỤC để tiếp"
                        )
                    else:
                        # Show current prediction (no progress bar to avoid confusion)
                        pred_placeholder.info(
                            f"🔍 {prediction} ({confidence:.1%})"
                        )
                else:
                    st.session_state.stable_prediction_count = 0
                    if prediction:
                        pred_placeholder.warning(f"⚠️ Confidence: {confidence:.1%} < {confidence_threshold}")
            
            cv2.putText(
                display_frame, 
                f"Buffer: {len(st.session_state.sequence_buffer)}/{SEQUENCE_LENGTH}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                display_frame,
                "No detection",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        
        stats_text = f"""
**Frames:** {frames_processed}
**Buffer:** {len(st.session_state.sequence_buffer)}/{SEQUENCE_LENGTH}
**Results:** {len(st.session_state.recognized_text)}
        """
        stats_placeholder.markdown(stats_text)
        
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(display_frame_rgb, use_container_width=True)
    
    cap.release()
    st.session_state.is_running = False
    st.success("✅ Dừng camera")

if __name__ == "__main__":
    main()
