import argparse, os
from pathlib import Path
import numpy as np
from src.config.config import SEQ_LEN

def normalize_keypoints(seq, wrist_left_idx=15, wrist_right_idx=16):
    """
    Chuẩn hóa keypoints:
    - Sử dụng wrist joints làm reference point (pose landmarks 15, 16)
    - Áp dụng công thức: L̂_t = (L_t - L_ref) / ||L_max - L_min||
    
    seq: numpy (seq_len, 225) - pose(99) + left_hand(63) + right_hand(63)
    """
    num_landmarks = seq.shape[1] // 3
    seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)  # (seq, N, 3)
    
    # Tìm wrist reference points (pose landmarks 15, 16)
    # Nếu không có pose landmarks, sử dụng center của bounding box
    if np.all(seq3d[:, wrist_left_idx, :2] == 0) and np.all(seq3d[:, wrist_right_idx, :2] == 0):
        # Fallback: sử dụng center của tất cả landmarks
        center = np.mean(seq3d[:, :, :2], axis=1, keepdims=True)
        ref_point = center
    else:
        # Sử dụng trung bình của 2 wrist points
        wrist_left = seq3d[:, wrist_left_idx, :2]
        wrist_right = seq3d[:, wrist_right_idx, :2]
        ref_point = (wrist_left + wrist_right) / 2
        ref_point = ref_point.reshape(-1, 1, 2)
    
    # Translation normalization: dịch về gốc tọa độ
    seq3d[:, :, 0] -= ref_point[:, 0, 0].reshape(-1, 1)
    seq3d[:, :, 1] -= ref_point[:, 0, 1].reshape(-1, 1)
    
    # Scale normalization: tính khoảng cách cực đại
    # Tìm min/max coordinates trong mỗi frame
    min_coords = np.min(seq3d[:, :, :2], axis=1)  # (seq, 2)
    max_coords = np.max(seq3d[:, :, :2], axis=1)  # (seq, 2)
    scale = np.linalg.norm(max_coords - min_coords, axis=1)  # (seq,)
    
    # Tránh chia cho 0
    scale[scale == 0] = 1.0
    scale = scale.reshape(-1, 1, 1)
    
    # Áp dụng scale normalization
    seq3d[:, :, :2] /= scale
    
    return seq3d.reshape(seq.shape[0], -1).astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/npy_raw")
    parser.add_argument("--output_dir", default="data/npy_normalized")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        out_label_dir = os.path.join(output_dir, label)
        Path(out_label_dir).mkdir(parents=True, exist_ok=True)
        for f in sorted(os.listdir(label_dir)):
            if f.lower().endswith((".npy")):
                raw_path = os.path.join(label_dir, f)
                raw_keypoints = np.load(raw_path)
                normalized_keypoints = normalize_keypoints(raw_keypoints)
                out_name = os.path.splitext(f)[0] + "_normalized.npy"
                out_path = os.path.join(out_label_dir, out_name)
                np.save(out_path, normalized_keypoints)
                print(f"Saved normalized keypoints to {out_path}")