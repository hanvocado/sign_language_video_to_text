import cv2
import os
import numpy as np

raw_dir = "data/raw"
durations = []

for root, _, files in os.walk(raw_dir):
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

    print(f"Total videos: {len(durations)}")
    print(f"Shortest video: {min_file} — {min_duration:.2f} seconds")
    print(f"Longest video:  {max_file} — {max_duration:.2f} seconds")
    print(f"Average duration: {avg_duration:.2f} seconds")
else:
    print("No .mp4 files found in data/raw/")
