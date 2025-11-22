"""
Convert videos to .npy sequences of keypoints (pose + hands) with 
frame sampling (no padding/truncate).

Sampling:
- Apply frame sampling, ensure uniform coverage of the signing motion, regardless of video length.
- Always returns exactly seq_len frames.
"""

import os, cv2, numpy as np, argparse
from pathlib import Path
import itertools
import mediapipe as mp
from src.config.config import SEQ_LEN
from src.utils.logger import *

mp_holistic = mp.solutions.holistic


# =====================================================
# Frame Sampling
# =====================================================

def get_chunks(l, n):
    """
    Divide list `l` into `n` chunks as evenly as possible.
    Guarantees: never returns empty chunks.
    """
    if len(l) == 0:
        return [[] for _ in range(n)]

    if len(l) < n:
        # pad by repeating the last element
        l = l + [l[-1]] * (n - len(l))

    k, m = divmod(len(l), n)
    chunks = [
        l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(n)
    ]

    # Final safety: ensure no empty chunk
    for i in range(n):
        if len(chunks[i]) == 0:
            chunks[i] = [l[-1]]

    return chunks


def safe_pick(chunk, pick_index):
    """Pick element safely from chunk."""
    if len(chunk) == 0:
        return 0
    pick_index = min(max(pick_index, 0), len(chunk) - 1)
    return chunk[pick_index]


def sampling_mode_1(chunks):
    """
    Your original logic but safe for short videos.
    """
    sampling = []
    L = len(chunks)

    for i, chunk in enumerate(chunks):
        if i == 0 or i == 1:
            sampling.append(safe_pick(chunk, -1))
        elif i == L - 1 or i == L - 2:
            sampling.append(safe_pick(chunk, 0))
        else:
            sampling.append(safe_pick(chunk, len(chunk) // 2))

    return sampling


def sampling_mode_2(frames, n_sequence):
    """
    Remove idle frames (first+last chunk's worth), then sample n_sequence frames.
    Works even if frames < 12.
    """

    L = len(frames)

    # If the video is too short for the full pre-sampling logic
    if L < 12:
        # fallback to uniform sampling
        chunks = get_chunks(frames, n_sequence)
        return sampling_mode_1(chunks)

    # Normal mode:
    chunks_12 = get_chunks(frames, 12)

    # drop first + last chunk
    middle = chunks_12[1:-1]

    # flatten
    sub_frame_list = [x for c in middle for x in c]

    # If sub_frame_list becomes too small
    if len(sub_frame_list) < n_sequence:
        chunks = get_chunks(sub_frame_list, n_sequence)
    else:
        chunks = get_chunks(sub_frame_list, n_sequence)

    return sampling_mode_1(chunks)


def sample_frames(total_frames, seq_len, mode="2"):
    frames = list(range(total_frames))
    if mode == "1":
        return sampling_mode_1(get_chunks(frames, seq_len))
    elif mode == "2":
        return sampling_mode_2(frames, seq_len)
    else:
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
# Main conversion function
# =====================================================

def convert_video_to_npy(video_path, output_path, seq_len=SEQ_LEN,
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

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    logger.info(f"Saved {output_path}, shape={arr.shape}")


def batch_convert(input_dir, output_dir, seq_len=SEQ_LEN, sampling_mode="2", skip_existing=False):

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
                    sampling_mode=sampling_mode,
                    skip_existing=skip_existing
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/npy")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--sampling_mode", default="2", choices=["1", "2"])
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("video2npy")
    log_arguments(logger, args)

    batch_convert(
        args.input_dir,
        args.output_dir,
        seq_len=args.seq_len,
        sampling_mode=args.sampling_mode,
        skip_existing=args.skip_existing
    )
