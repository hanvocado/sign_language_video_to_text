"""
Create a WLASL subset and organize videos into:
train/<gloss>/, val/<gloss>/, test/<gloss>/

Uses split information from WLASL_v0.3.json:
    'split' ∈ ['train', 'val', 'test']

Usage:
    python split_and_organize_wlasl.py --subset 100 \
        --json data/wlasl/WLASL_v0.3.json \
        --src_dir data/wlasl/videos \
        --dst_dir data/wlasl/wlasl100
"""

import os
import json
import shutil
from tqdm import tqdm
import argparse


def create_split_folders(dst_root, glosses):
    """Create train/val/test and gloss subfolders."""
    for split in ["train", "val", "test"]:
        for gloss in glosses:
            os.makedirs(os.path.join(dst_root, split, gloss), exist_ok=True)


def main(subset, json_path, src_dir, dst_dir):
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Only keep first `subset` glosses
    selected_entries = data[:subset]
    gloss_names = [entry["gloss"] for entry in selected_entries]

    # Prepare directory tree
    create_split_folders(dst_dir, gloss_names)

    print(f"\nOrganizing WLASL subset of {subset} glosses...")
    missing = 0
    copied = 0

    for entry in tqdm(selected_entries):
        gloss = entry["gloss"]
        instances = entry["instances"]

        for inst in instances:
            vid = inst["video_id"]
            split_dir = inst["split"]  # 'train', 'val', or 'test'
            src = os.path.join(src_dir, f"{vid}.mp4")
            dst = os.path.join(dst_dir, split_dir, gloss, f"{vid}.mp4")

            if not os.path.exists(src):
                print(f"[WARNING] Missing video: {src}")
                missing += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

    print("\n[SUMMARY]")
    print(f"Glosses processed: {subset}")
    print(f"Videos copied:     {copied}")
    print(f"Missing videos:    {missing}")
    print(f"Output structure:  {dst_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100, help="Number of glosses to include")
    parser.add_argument("--json", default="data/wlasl/WLASL_v0.3.json", help="Path to WLASL json")
    parser.add_argument("--src_dir", default="data/wlasl/videos", help="Directory with all WLASL .mp4 videos")

    args = parser.parse_args()

    dst_dir = f"data/wlasl/wlasl{args.subset}"
    
    main(args.subset, args.json, args.src_dir, dst_dir)
