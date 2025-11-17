"""
Convert videos to .npy sequences of keypoints (face + hands) with normalization.

Chuẩn hóa:
- Dịch toàn bộ keypoints sao cho mũi (nose) nằm tại gốc (0,0).
- Chia scale theo khoảng cách giữa hai vai (left_shoulder, right_shoulder).
"""

import os, cv2, numpy as np, argparse, sys
from pathlib import Path
import mediapipe as mp
from src.config.config import SEQ_LEN

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    """
    Trả về vector gồm pose + left hand + right hand (không bao gồm face).
    Nếu part không có thì điền 0.
    """
    # Comment face extraction - chỉ lấy pose và hands
    # face, lh, rh = [], [], []
    # if results.face_landmarks:
    #     for lm in results.face_landmarks.landmark:
    #         face.extend([lm.x, lm.y, lm.z])
    # else:
    #     face = [0.0] * 468 * 3
    
    pose, lh, rh = [], [], []
    
    # Extract pose landmarks (33 landmarks * 3 coordinates = 99 features)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose.extend([lm.x, lm.y, lm.z])
    else:
        pose = [0.0] * 33 * 3

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else:
        lh = [0.0] * 21 * 3

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else:
        rh = [0.0] * 21 * 3

    return np.array(pose + lh + rh, dtype=np.float32)


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


def convert_video_to_npy(video_path, output_path, seq_len=SEQ_LEN, normalize=True, skip_existing=False):
    if skip_existing and os.path.exists(output_path):
        print(f"⏭️  Skip {video_path}: found existing {output_path}")
        return

    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        kp = extract_keypoints(results)
        seq.append(kp)
    cap.release()
    holistic.close()

    if len(seq) == 0:
        print(f"⚠️ Warning: No keypoints extracted from {video_path}")
        # Updated feature dimension: pose(99) + left_hand(63) + right_hand(63) = 225
        seq = [np.zeros(225, dtype=np.float32) for _ in range(seq_len)]

    arr = np.stack(seq, axis=0).astype(np.float32)

    # padding / truncate
    if arr.shape[0] < seq_len:
        pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad])
    elif arr.shape[0] > seq_len:
        arr = arr[:seq_len]

    # normalize
    if normalize:
        arr = normalize_keypoints(arr)

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    print(f"Saved {output_path}, shape={arr.shape}")


def batch_convert(input_dir, output_dir, seq_len=SEQ_LEN, normalize=False, skip_existing=False):
    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        out_label_dir = os.path.join(output_dir, label)
        Path(out_label_dir).mkdir(parents=True, exist_ok=True)
        for f in sorted(os.listdir(label_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                vpath = os.path.join(label_dir, f)
                out_name = os.path.splitext(f)[0] + ".npy"
                out_path = os.path.join(out_label_dir, out_name)
                convert_video_to_npy(vpath, out_path, seq_len=seq_len, normalize=normalize, skip_existing=skip_existing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/npy")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--normalize", action="store_true", help="Chuẩn hóa keypoints")
    parser.add_argument("--skip_existing", action="store_true", help="Bỏ qua video nếu file .npy đã tồn tại")
    args = parser.parse_args()

    batch_convert(
        args.input_dir, args.output_dir, seq_len=args.seq_len, normalize=args.normalize, skip_existing=args.skip_existing
    )
