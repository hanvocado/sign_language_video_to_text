"""
Realtime sign language preprocessing module.
Designed to match training preprocessing for consistency.
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, List, Optional
from collections import deque
import threading
import time

mp_holistic = mp.solutions.holistic


class RealtimeKeyPointExtractor:
    """Extract MediaPipe keypoints from frames in realtime."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract keypoints from frame.
        Returns vector: pose(99) + left_hand(63) + right_hand(63) = 225
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        
        pose, lh, rh = [], [], []
        
        # Extract pose landmarks (33 landmarks * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                pose.extend([lm.x, lm.y, lm.z])
        else:
            pose = [0.0] * 33 * 3
        
        # Extract left hand (21 landmarks * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                lh.extend([lm.x, lm.y, lm.z])
        else:
            lh = [0.0] * 21 * 3
        
        # Extract right hand (21 landmarks * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                rh.extend([lm.x, lm.y, lm.z])
        else:
            rh = [0.0] * 21 * 3
        
        return np.array(pose + lh + rh, dtype=np.float32)
    
    def close(self):
        """Clean up resources."""
        self.holistic.close()


class KeyPointNormalizer:
    """Normalize keypoints to match training data."""
    
    @staticmethod
    def normalize_keypoints(seq: np.ndarray, wrist_left_idx=15, wrist_right_idx=16) -> np.ndarray:
        """
        Normalize keypoints using wrist joints as reference.
        
        Formula: L̂_t = (L_t - L_ref) / ||L_max - L_min||
        
        Args:
            seq: Shape (seq_len, 225) - pose(99) + left_hand(63) + right_hand(63)
            wrist_left_idx: Pose landmark index for left wrist
            wrist_right_idx: Pose landmark index for right wrist
        
        Returns:
            Normalized sequence with same shape
        """
        num_landmarks = seq.shape[1] // 3
        seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)  # (seq, N, 3)
        
        # Find reference points
        if np.all(seq3d[:, wrist_left_idx, :2] == 0) and np.all(seq3d[:, wrist_right_idx, :2] == 0):
            # Fallback: use center of all landmarks
            center = np.mean(seq3d[:, :, :2], axis=1, keepdims=True)
            ref_point = center
        else:
            # Use average of wrist points
            wrist_left = seq3d[:, wrist_left_idx, :2]
            wrist_right = seq3d[:, wrist_right_idx, :2]
            ref_point = (wrist_left + wrist_right) / 2
            ref_point = ref_point.reshape(-1, 1, 2)
        
        # Translation: shift to origin
        seq3d[:, :, 0] -= ref_point[:, 0, 0].reshape(-1, 1)
        seq3d[:, :, 1] -= ref_point[:, 0, 1].reshape(-1, 1)
        
        # Scale normalization
        min_coords = np.min(seq3d[:, :, :2], axis=1)  # (seq, 2)
        max_coords = np.max(seq3d[:, :, :2], axis=1)  # (seq, 2)
        scale = np.linalg.norm(max_coords - min_coords, axis=1)  # (seq,)
        
        # Avoid division by zero
        scale[scale == 0] = 1.0
        scale = scale.reshape(-1, 1, 1)
        
        # Apply scale
        seq3d[:, :, :2] /= scale
        
        return seq3d.reshape(seq.shape[0], -1).astype(np.float32)


class RealtimeSequenceBuffer:
    """
    Sliding window buffer for sequence preprocessing.
    Accumulates frames and provides preprocessed sequences.
    """
    
    def __init__(self, seq_len=64, normalize=True, use_scaler=None):
        """
        Args:
            seq_len: Target sequence length (default: 64)
            normalize: Apply keypoint normalization
            use_scaler: Optional sklearn StandardScaler for feature scaling
        """
        self.seq_len = seq_len
        self.normalize = normalize
        self.scaler = use_scaler
        self.buffer = deque(maxlen=seq_len)
        self.lock = threading.Lock()
    
    def add_frame(self, keypoints: np.ndarray):
        """Add a frame's keypoints to buffer."""
        with self.lock:
            self.buffer.append(keypoints)
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get preprocessed sequence if buffer is full.
        
        Returns:
            Array of shape (seq_len, 225) or None if buffer not full
        """
        with self.lock:
            if len(self.buffer) < self.seq_len:
                return None
            
            # Stack frames
            arr = np.stack(list(self.buffer), axis=0).astype(np.float32)
            
            # Normalize if needed
            if self.normalize:
                arr = KeyPointNormalizer.normalize_keypoints(arr)
            
            # Apply scaler if provided
            if self.scaler is not None:
                frames = [self.scaler.transform(f.reshape(1, -1)).reshape(-1) for f in arr]
                arr = np.stack(frames, axis=0).astype(np.float32)
            
            return arr
    
    def is_ready(self) -> bool:
        """Check if buffer has enough frames."""
        with self.lock:
            return len(self.buffer) >= self.seq_len
    
    def reset(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
    
    def buffer_size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)


class FrameProcessor:
    """Process frames for display with landmark visualization."""
    
    @staticmethod
    def draw_landmarks(frame: np.ndarray, results) -> np.ndarray:
        """Draw pose and hand landmarks on frame."""
        # Draw pose
        if results.pose_landmarks:
            for connection in mp_holistic.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_pos = results.pose_landmarks.landmark[start_idx]
                end_pos = results.pose_landmarks.landmark[end_idx]
                
                start = (int(start_pos.x * frame.shape[1]), int(start_pos.y * frame.shape[0]))
                end = (int(end_pos.x * frame.shape[1]), int(end_pos.y * frame.shape[0]))
                
                cv2.line(frame, start, end, (0, 255, 0), 2)
            
            # Draw keypoints
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw left hand
        if results.left_hand_landmarks:
            for connection in mp_holistic.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_pos = results.left_hand_landmarks.landmark[start_idx]
                end_pos = results.left_hand_landmarks.landmark[end_idx]
                
                start = (int(start_pos.x * frame.shape[1]), int(start_pos.y * frame.shape[0]))
                end = (int(end_pos.x * frame.shape[1]), int(end_pos.y * frame.shape[0]))
                
                cv2.line(frame, start, end, (255, 0, 0), 2)
        
        # Draw right hand
        if results.right_hand_landmarks:
            for connection in mp_holistic.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_pos = results.right_hand_landmarks.landmark[start_idx]
                end_pos = results.right_hand_landmarks.landmark[end_idx]
                
                start = (int(start_pos.x * frame.shape[1]), int(start_pos.y * frame.shape[0]))
                end = (int(end_pos.x * frame.shape[1]), int(end_pos.y * frame.shape[0]))
                
                cv2.line(frame, start, end, (0, 0, 255), 2)
        
        return frame
