"""
PyTorch Dataset for Sign Language Recognition

Supports two modes:
1. Load from pre-extracted .npy files
2. Load directly from videos with on-the-fly extraction

Includes runtime augmentation for training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import mediapipe as mp
import joblib
from src.utils.common_functions import *

mp_holistic = mp.solutions.holistic

def extract_keypoints_from_video(video_path, seq_len, sampling_mode="2"):
    """Extract keypoints sequence from video file"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((seq_len, 225), dtype=np.float32)
    
    indices = sample_frames(total_frames, seq_len, mode=sampling_mode)
    
    seq = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                seq.append(np.zeros(225, dtype=np.float32))
                continue
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            seq.append(extract_keypoints(results))
    
    cap.release()
    return np.stack(seq).astype(np.float32)


# =====================================================
# Normalization
# =====================================================

def normalize_keypoints(seq, left_wrist_idx=15, right_wrist_idx=16):
    """
    Normalize keypoints:
    - Center at midpoint between wrists
    - Scale by bounding box diagonal
    """
    num_landmarks = seq.shape[1] // 3
    seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)

    # Get reference point (center of wrists)
    lw = seq3d[:, left_wrist_idx, :2]
    rw = seq3d[:, right_wrist_idx, :2]
    
    # Check if both wrists are missing (all zeros)
    lw_missing = np.all(lw == 0, axis=1)
    rw_missing = np.all(rw == 0, axis=1)
    both_missing = lw_missing & rw_missing
    
    # Calculate reference point
    ref = (lw + rw) / 2
    
    # For frames with missing wrists, use mean of all keypoints
    if np.any(both_missing):
        mean_all = np.mean(seq3d[:, :, :2], axis=1)
        ref[both_missing] = mean_all[both_missing]
    
    # Center
    seq3d[:, :, 0] -= ref[:, 0].reshape(-1, 1)
    seq3d[:, :, 1] -= ref[:, 1].reshape(-1, 1)

    # Scale by bounding box diagonal
    min_c = np.min(seq3d[:, :, :2], axis=1)
    max_c = np.max(seq3d[:, :, :2], axis=1)
    scale = np.linalg.norm(max_c - min_c, axis=1)
    scale[scale == 0] = 1  # Avoid division by zero
    seq3d[:, :, :2] /= scale.reshape(-1, 1, 1)

    return seq3d.reshape(seq.shape[0], -1)


# =====================================================
# Augmentation (Runtime)
# =====================================================

def augment_keypoints(seq, config=None):
    """
    Apply random augmentations to keypoint sequence.
    
    Args:
        seq: (seq_len, 225) keypoint sequence
        config: dict with augmentation parameters
    
    Returns:
        Augmented sequence
    """
    if config is None:
        config = {
            'rotation_range': 15,      # degrees
            'scale_range': (0.85, 1.15),
            'shift_range': 0.08,
            'flip_prob': 0.5,
            'time_mask_prob': 0.2,
            'time_mask_max': 3,
        }
    
    seq = seq.copy()
    seq3d = seq.reshape(seq.shape[0], -1, 3)
    
    # 1. Random rotation
    if config.get('rotation_range', 0) > 0:
        angle = np.random.uniform(
            -config['rotation_range'], 
            config['rotation_range']
        ) * np.pi / 180
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ], dtype=np.float32)
        seq3d[:, :, :2] = seq3d[:, :, :2] @ R.T
    
    # 2. Random scaling
    if 'scale_range' in config:
        scale = np.random.uniform(*config['scale_range'])
        seq3d[:, :, :2] *= scale
    
    # 3. Random translation
    if config.get('shift_range', 0) > 0:
        shift = np.random.uniform(
            -config['shift_range'], 
            config['shift_range'], 
            size=(2,)
        )
        seq3d[:, :, 0] += shift[0]
        seq3d[:, :, 1] += shift[1]
    
    # 4. Horizontal flip (with hand swap)
    if np.random.random() < config.get('flip_prob', 0):
        # Flip x-coordinates
        seq3d[:, :, 0] = -seq3d[:, :, 0]
        
        # Swap left and right hands
        # Pose: 0-32 (33 landmarks), Left hand: 33-53 (21), Right hand: 54-74 (21)
        left_hand = seq3d[:, 33:54, :].copy()
        right_hand = seq3d[:, 54:75, :].copy()
        seq3d[:, 33:54, :] = right_hand
        seq3d[:, 54:75, :] = left_hand
    
    # 5. Time masking (zero out some frames)
    if np.random.random() < config.get('time_mask_prob', 0):
        mask_len = np.random.randint(1, config.get('time_mask_max', 3) + 1)
        start = np.random.randint(0, max(1, seq.shape[0] - mask_len))
        seq3d[start:start + mask_len, :, :] = 0
    
    return seq3d.reshape(seq.shape)


# =====================================================
# Dataset Classes
# =====================================================

class SignLanguageDataset(Dataset):
    """
    Dataset for sign language recognition.
    
    Can load from:
    - Pre-extracted .npy files (set source='npy')
    - Video files directly (set source='video')
    
    Supports runtime augmentation for training.
    """
    
    def __init__(
        self,
        data_dir,
        seq_len=30,
        source='npy',           # 'npy' or 'video'
        split='train',          # 'train', 'val', 'test'
        normalize=True,
        augment=False,
        augment_config=None,
        sampling_mode='2',
        label_map=None,         # Optional: predefined label->idx mapping
        scaler_path=None,       # Optional: sklearn scaler
    ):
        """
        Args:
            data_dir: Root directory containing train/val/test subdirs
            seq_len: Fixed sequence length
            source: 'npy' to load .npy files, 'video' to extract from videos
            split: Which split to load ('train', 'val', 'test')
            normalize: Whether to normalize keypoints
            augment: Whether to apply augmentation (usually True for train)
            augment_config: Dict of augmentation parameters
            sampling_mode: '1' or '2' for frame sampling (video mode only)
            label_map: Predefined label mapping (for consistency across splits)
            scaler_path: Path to saved sklearn scaler
        """
        self.seq_len = seq_len
        self.source = source
        self.normalize = normalize
        self.augment = augment
        self.augment_config = augment_config
        self.sampling_mode = sampling_mode
        
        # Load scaler if provided
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Build file list
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        self.samples = []  # List of (file_path, label)
        
        # Determine file extension based on source
        if source == 'npy':
            extensions = ('.npy',)
        else:  # video
            extensions = ('.mp4', '.avi', '.mov', '.mkv')
        
        # Scan directory
        labels_found = set()
        for label in sorted(os.listdir(split_dir)):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            
            labels_found.add(label)
            
            for filename in sorted(os.listdir(label_dir)):
                if filename.lower().endswith(extensions):
                    file_path = os.path.join(label_dir, filename)
                    self.samples.append((file_path, label))
        
        # Build label mapping
        if label_map is None:
            labels = sorted(labels_found)
            self.label_to_idx = {l: i for i, l in enumerate(labels)}
            self.idx_to_label = labels
        else:
            if isinstance(label_map, dict):
                self.label_to_idx = label_map
                self.idx_to_label = [None] * (max(label_map.values()) + 1)
                for k, v in label_map.items():
                    self.idx_to_label[v] = k
            elif isinstance(label_map, list):
                self.idx_to_label = label_map
                self.label_to_idx = {l: i for i, l in enumerate(label_map)}
            else:
                raise ValueError("label_map must be dict or list")
        
        print(f"Loaded {len(self.samples)} samples from {split_dir}")
        print(f"Classes: {len(self.idx_to_label)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load keypoints
        if self.source == 'npy':
            arr = np.load(file_path)
        else:  # video
            arr = extract_keypoints_from_video(
                file_path, 
                self.seq_len, 
                self.sampling_mode
            )
        
        # Ensure correct sequence length (for npy files that might differ)
        if arr.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif arr.shape[0] > self.seq_len:
            arr = arr[:self.seq_len]
        
        # Augment BEFORE normalize (important!)
        if self.augment:
            arr = augment_keypoints(arr, self.augment_config)
        
        # Normalize
        if self.normalize:
            arr = normalize_keypoints(arr)
        
        # Apply scaler if provided
        if self.scaler is not None:
            frames = [self.scaler.transform(f.reshape(1, -1)).reshape(-1) for f in arr]
            arr = np.stack(frames, axis=0).astype(np.float32)
        
        # Get label index
        label_idx = self.label_to_idx[label]
        
        return (
            torch.from_numpy(arr).float(),
            torch.tensor(label_idx, dtype=torch.long)
        )
    
    def get_label_map(self):
        """Return label mapping for saving/loading"""
        return self.idx_to_label

# =====================================================
# Utility Functions
# =====================================================

def create_data_loaders(
    data_dir,
    seq_len=30,
    batch_size=16,
    source='npy',
    normalize=True,
    augment_train=True,
    num_workers=0,
):
    """
    Create train, val, test data loaders.
    
    Args:
        data_dir: Root directory with train/val/test subdirs
        seq_len: Sequence length
        batch_size: Batch size
        source: 'npy' or 'video'
        normalize: Whether to normalize
        augment_train: Whether to augment training data
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader, label_map
    """
    from torch.utils.data import DataLoader
    
    # Create train dataset first to get label mapping
    train_ds = SignLanguageDataset(
        data_dir,
        seq_len=seq_len,
        source=source,
        split='train',
        normalize=normalize,
        augment=augment_train,
    )
    
    label_map = train_ds.get_label_map()
    
    # Create val and test with same label mapping
    val_ds = SignLanguageDataset(
        data_dir,
        seq_len=seq_len,
        source=source,
        split='val',
        normalize=normalize,
        augment=False,  # No augmentation for validation
        label_map=label_map,
    )
    
    # Test set might not exist
    test_loader = None
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        test_ds = SignLanguageDataset(
            data_dir,
            seq_len=seq_len,
            source=source,
            split='test',
            normalize=normalize,
            augment=False,
            label_map=label_map,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader, label_map


if __name__ == "__main__":
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/wlasl100")
    parser.add_argument("--source", choices=['npy', 'video'], default='video')
    parser.add_argument("--seq_len", type=int, default=30)
    args = parser.parse_args()
    
    print("Testing SignLanguageDataset...")
    
    ds = SignLanguageDataset(
        args.data_dir,
        seq_len=args.seq_len,
        source=args.source,
        split='train',
        normalize=True,
        augment=True,
    )
    
    print(f"\nDataset size: {len(ds)}")
    print(f"Labels: {ds.idx_to_label}")
    
    # Test loading a sample
    X, y = ds[0]
    print(f"\nSample shape: {X.shape}")
    print(f"Sample label: {y.item()} ({ds.idx_to_label[y.item()]})")
    print(f"Value range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Mean: {X.mean():.3f}, Std: {X.std():.3f}")