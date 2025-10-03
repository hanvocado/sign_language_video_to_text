"""
Preprocess raw sign language videos:
- Convert fps -> 30
- Resize to 1280x720 (16:9)
- Optionally split long videos into smaller clips (2-5s)
"""

import os
import cv2
from pathlib import Path
import argparse

def preprocess_video(input_path, output_path, fps=30, width=1280, height=720, max_duration=5):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {input_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    # Nếu max_duration > 0, cắt thành nhiều đoạn
    segment_len = int(fps * max_duration) if max_duration > 0 else total_frames
    seg_idx, frame_idx = 0, 0
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize
        frame = cv2.resize(frame, (width, height))

        # init writer cho segment mới
        if frame_idx % segment_len == 0:
            if out is not None:
                out.release()
            seg_name = os.path.splitext(os.path.basename(output_path))[0] + f"_{seg_idx}.mp4"
            seg_path = os.path.join(os.path.dirname(output_path), seg_name)
            out = cv2.VideoWriter(seg_path, fourcc, fps, (width, height))
            seg_idx += 1

        # ghi frame
        out.write(frame)
        frame_idx += 1

    if out is not None:
        out.release()
    cap.release()
    print(f"✅ Preprocessed {input_path} -> {os.path.dirname(output_path)}")


def batch_preprocess(input_dir, output_dir, fps=30, width=1280, height=720, max_duration=5):
    for label in sorted(os.listdir(input_dir)):
        in_label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(in_label_dir):
            continue
        out_label_dir = os.path.join(output_dir, label)
        Path(out_label_dir).mkdir(parents=True, exist_ok=True)
        for f in sorted(os.listdir(in_label_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                in_path = os.path.join(in_label_dir, f)
                out_path = os.path.join(out_label_dir, f)
                preprocess_video(in_path, out_path, fps=fps, width=width, height=height, max_duration=max_duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw_unprocessed", help="Thư mục chứa video gốc")
    parser.add_argument("--output_dir", default="data/raw", help="Thư mục lưu video đã chuẩn hóa")
    parser.add_argument("--fps", type=int, default=30, help="Tần số khung hình sau chuẩn hóa")
    parser.add_argument("--width", type=int, default=1280, help="Chiều rộng video output")
    parser.add_argument("--height", type=int, default=720, help="Chiều cao video output")
    parser.add_argument("--max_duration", type=int, default=5, help="Thời lượng tối đa mỗi clip (giây), 0 = không cắt")
    args = parser.parse_args()

    batch_preprocess(args.input_dir, args.output_dir,
                     fps=args.fps, width=args.width, height=args.height,
                     max_duration=args.max_duration)
