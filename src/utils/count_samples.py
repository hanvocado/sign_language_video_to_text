import os
from collections import defaultdict
import argparse
from src.utils.logger import *

def main(dir):
    counts = defaultdict(int)

    # Walk through label folders
    for label in sorted(os.listdir(dir)):
        label_dir = os.path.join(dir, label)

        if not os.path.isdir(label_dir):
            continue  # skip non-folders

        # Count .mp4 files
        num_videos = len([f for f in os.listdir(label_dir) if f.endswith(".mp4")])
        counts[label] = num_videos

    # Print results
    print("Number of videos per label:\n")
    for lbl, num in counts.items():
        print(f"{lbl}: {num}")

    print("\nTotal labels:", len(counts))
    print("Total videos:", sum(counts.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/raw")
    args = parser.parse_args()

    logger = setup_logger("video2npy")
    log_arguments(logger=logger, args=args)

    dir = args.dir
    main(dir)
