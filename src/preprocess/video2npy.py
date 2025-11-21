"""
Convert videos to .npy sequences of keypoints (pose + hands) with normalization
and frame sampling (no padding/truncate).

Sampling:
- Apply frame sampling, ensure uniform coverage of the signing motion, regardless of video length.
- Always returns exactly seq_len frames.

Normalization:
- Translate so wrist-center = (0,0)
- Scale by bounding box size.

Augmentation (optional):
- Random rotation
- Random scaling
- Random translation
"""

import os, cv2, numpy as np, argparse
from pathlib import Path
import itertools
import mediapipe as mp
from src.config.config import SEQ_LEN
from src.utils.logger import *

mp_holistic = mp.solutions.holistic


# =====================================================
# Frame Sampling (same as VideoFrameGenerator)
# =====================================================

def get_chunks(l, n):
    k, m = divmod(len(l), n)
    return (l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def sampling_mode_1(chunks):
    sampling = []
    for i, chunk in enumerate(chunks):
        if i == 0 or i == 1:
            sampling.append(chunk[-1])
        elif i == len(chunks)-1 or i == len(chunks)-2:
            sampling.append(chunk[0])
        else:
            sampling.append(chunk[len(chunk)//2])
    return sampling


def sampling_mode_2(frames, n_sequence):
    chunks = list(get_chunks(frames, 12))
    sub_chunks = chunks[1:-1]
    sub_frame_list = list(itertools.chain.from_iterable(sub_chunks))
    new_chunks = list(get_chunks(sub_frame_list, n_sequence))
    return sampling_mode_1(new_chunks)


def sample_frames(total_frames, seq_len, mode="2"):
    frames = list(range(total_frames))
    if mode == "1":
        return sampling_mode_1(list(get_chunks(frames, seq_len)))
    if mode == "2":
        return sampling_mode_2(frames, seq_len)
    raise ValueError("Invalid sampling mode")


# =====================================================
# Keypoint extraction
# =====================================================

def extract_keypoints(results):
    pose, lh, rh = [], [], []

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


# =====================================================
# Normalize keypoints
# =====================================================

def normalize_keypoints(seq, left_wrist=15, right_wrist=16):
    num_lm = seq.shape[1] // 3
    seq3d = seq.reshape(seq.shape[0], num_lm, 3)

    # get reference point
    if np.all(seq3d[:, left_wrist, :2] == 0) and np.all(seq3d[:, right_wrist, :2] == 0):
        ref = np.mean(seq3d[:, :, :2], axis=1, keepdims=True)
    else:
        lw = seq3d[:, left_wrist, :2]
        rw = seq3d[:, right_wrist, :2]
        ref = ((lw + rw) / 2).reshape(-1, 1, 2)

    seq3d[:, :, 0] -= ref[:, 0, 0].reshape(-1, 1)
    seq3d[:, :, 1] -= ref[:, 0, 1].reshape(-1, 1)

    # scale
    min_c = np.min(seq3d[:, :, :2], axis=1)
    max_c = np.max(seq3d[:, :, :2], axis=1)
    scale = np.linalg.norm(max_c - min_c, axis=1)
    scale[scale == 0] = 1
    seq3d[:, :, :2] /= scale.reshape(-1, 1, 1)

    return seq3d.reshape(seq.shape[0], -1)


# =====================================================
# Keypoint augmentation (for training only)
# =====================================================

def augment_keypoints(seq, type=1):
    """Apply rotation, scaling, translation jitter to keypoints"""
    seq3d = seq.reshape(seq.shape[0], -1, 3)

    # rotation
    angle = np.random.uniform(-8, 8) * np.pi / 180
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]], np.float32)
    seq3d[:, :, :2] = seq3d[:, :, :2] @ R.T

    # scale jitter
    scale = np.random.uniform(0.9, 1.1)
    seq3d[:, :, :2] *= scale

    # translation jitter
    shift = np.random.uniform(-0.05, 0.05, size=(2,))
    seq3d[:, :, 0] += shift[0]
    seq3d[:, :, 1] += shift[1]

    return seq3d.reshape(seq.shape)


# =====================================================
# Main conversion function
# =====================================================

def convert_video_to_npy(video_path, output_path, seq_len=SEQ_LEN, augment=False,
                         sampling_mode="2", skip_existing=False):

    if skip_existing and os.path.exists(output_path):
        logger.info(f"⏭️ Skip {video_path}: already exists")
        return

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # sample frame indices
    indices = sample_frames(total, seq_len, mode=sampling_mode)

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
            res = holistic.process(image)
            seq.append(extract_keypoints(res))

    cap.release()
    arr = np.stack(seq).astype(np.float32)        

    if augment:              
        # random choice of number of transformations
        random_transforms = np.random.randint(2, 4)  # min 2 - max 3
        transformations = [1, 2, 3]
        # random choice of transformations
        transforms_idxs = np.random.choice(len(transformations), random_transforms, replace=False)
        
        for idx in transforms_idxs:
            arr_aug = arr.copy()
            arr_aug = augment_keypoints(arr_aug)
            arr_aug = normalize_keypoints(arr_aug)
            
            # Save augmented version with _aug suffix
            aug_path = output_path.replace('.npy', f'_aug_{idx}.npy')
            Path(os.path.dirname(aug_path)).mkdir(parents=True, exist_ok=True)
            np.save(aug_path, arr_aug)
            logger.info(f"Saved {aug_path}, shape={arr_aug.shape}")

    arr = normalize_keypoints(arr)
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    logger.info(f"Saved {output_path}, shape={arr.shape}")


def batch_convert(input_dir, output_dir, seq_len=SEQ_LEN, augment=False,
                  sampling_mode="2", skip_existing=False):

    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue

        out_label = os.path.join(output_dir, label)
        Path(out_label).mkdir(parents=True, exist_ok=True)

        for f in sorted(os.listdir(label_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                in_path = os.path.join(label_dir, f)
                out_path = os.path.join(out_label, f.replace('.mp4', '.npy'))
                convert_video_to_npy(
                    in_path, out_path,
                    seq_len=seq_len,
                    augment=augment,
                    sampling_mode=sampling_mode,
                    skip_existing=skip_existing
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/npy")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--sampling_mode", default="2", choices=["1", "2"])
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("video2npy")
    log_arguments(logger, args)

    batch_convert(
        args.input_dir,
        args.output_dir,
        seq_len=args.seq_len,
        augment=args.augment,
        sampling_mode=args.sampling_mode,
        skip_existing=args.skip_existing
    )
