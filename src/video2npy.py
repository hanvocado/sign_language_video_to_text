"""
Convert videos to .npy sequences of keypoints (face + hands) with normalization.

Chuẩn hóa:
- Dịch toàn bộ keypoints sao cho mũi (nose) nằm tại gốc (0,0).
- Chia scale theo khoảng cách giữa hai vai (left_shoulder, right_shoulder).
"""

import os, cv2, numpy as np, argparse, sys
from pathlib import Path

# ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import mediapipe as mp
from src.config import SEQ_LEN

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    """
    Trả về vector (1530,) gồm face + left hand + right hand.
    Nếu part không có thì điền 0.
    """
    face, lh, rh = [], [], []
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            face.extend([lm.x, lm.y, lm.z])
    else:
        face = [0.0] * 468 * 3

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

    return np.array(face + lh + rh, dtype=np.float32)


def normalize_keypoints(seq, anchor_idx=1, scale_idx=(11, 12)):
    """
    Chuẩn hóa keypoints trong một sequence.
    - anchor_idx: landmark làm gốc (1 = nose trong pose landmarks của MediaPipe).
    - scale_idx: cặp landmark để tính scale (11 = left_shoulder, 12 = right_shoulder).

    seq: numpy (seq_len, 1530)
    """
    num_landmarks = seq.shape[1] // 3
    seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)  # (seq, N, 3)

    # Nếu anchor (mũi) không có (toàn 0) thì bỏ qua normalize
    if np.all(seq3d[:, anchor_idx, :2] == 0):
        return seq

    # Dịch toàn bộ keypoints sao cho anchor (nose) = (0,0)
    anchor = seq3d[:, anchor_idx, :2]  # (seq,2)
    seq3d[:, :, 0] -= anchor[:, 0].reshape(-1, 1)
    seq3d[:, :, 1] -= anchor[:, 1].reshape(-1, 1)

    # Tính scale = khoảng cách giữa 2 vai
    p1, p2 = seq3d[:, scale_idx[0], :2], seq3d[:, scale_idx[1], :2]
    scale = np.linalg.norm(p1 - p2, axis=1).reshape(-1, 1)
    scale[scale == 0] = 1.0
    seq3d[:, :, :2] /= scale

    return seq3d.reshape(seq.shape[0], -1).astype(np.float32)


def convert_video_to_npy(video_path, output_path, seq_len=SEQ_LEN, normalize=True):
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
        seq = [np.zeros(1530, dtype=np.float32) for _ in range(seq_len)]

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


def batch_convert(input_dir, output_dir, seq_len=SEQ_LEN, normalize=True):
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
                convert_video_to_npy(vpath, out_path, seq_len=seq_len, normalize=normalize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/npy")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--no_normalize", action="store_true", help="Không chuẩn hóa keypoints")
    args = parser.parse_args()

    batch_convert(
        args.input_dir, args.output_dir, seq_len=args.seq_len, normalize=not args.no_normalize
    )
