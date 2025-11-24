import cv2
import os
import argparse
import numpy as np
from logger import *

logger = setup_logger("find_video_duration") 
durations = []

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="data/raw")
args = parser.parse_args()
dir = args.dir
log_arguments(logger, args)

for root, _, files in os.walk(dir):
    for file in files:
        if file.endswith(".mp4"):
            path = os.path.join(root, file)
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frames / fps if fps > 0 else 0
            cap.release()
            durations.append((path, duration))

if durations:
    durations.sort(key=lambda x: x[1])
    min_file, min_duration = durations[0]
    max_file, max_duration = durations[-1]
    avg_duration = np.mean([d for _, d in durations])

    logger.info(f"Total videos: {len(durations)}")
    logger.info(f"Shortest video: {min_file} — {min_duration:.2f} seconds")
    logger.info(f"Longest video:  {max_file} — {max_duration:.2f} seconds")
    logger.info(f"Average duration: {avg_duration:.2f} seconds")
else:
    logger.info("No .mp4 files found in data/raw/")
