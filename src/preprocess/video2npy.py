"""
Convert videos to .npy sequences of keypoints (pose + hands) with 
frame sampling (no padding/truncate).

Sampling:
- Apply frame sampling, ensure uniform coverage of the signing motion, regardless of video length.
- Always returns exactly seq_len frames.
"""

import os, cv2, numpy as np, argparse
from pathlib import Path
import mediapipe as mp
from src.config.config import SEQ_LEN
from src.utils.logger import *
from src.utils.common_functions import *

mp_holistic = mp.solutions.holistic

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
