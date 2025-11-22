"""
Split a dataset of sign language videos into train/val/test
with balanced class distribution AND move actual video files accordingly.

Input structure (two modes):

1) Nested mode (default):
    data_dir/
        GLOSS1/*.mp4
        GLOSS2/*.mp4

2) Flat mode (use --flat):
    data_dir/*.mp4
    + WLASL JSON required for mapping video_id -> gloss
"""

import os, glob, argparse, shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.utils.logger import *


def collect_nested(data_dir):
    """Collect files from gloss folders: data_dir/GLOSS/*.mp4."""
    rows = []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for p in glob.glob(os.path.join(label_dir, "*.mp4")):
            rows.append({"path": p, "label": label})
    return rows


def collect_flat(data_dir, json_path):
    """Collect from data_dir/*.mp4 using WLASL JSON to assign gloss labels."""
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create mapping video_id -> gloss
    vid2gloss = {}
    for entry in data:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            vid = inst["video_id"]
            vid2gloss[vid] = gloss

    rows = []
    for p in glob.glob(os.path.join(data_dir, "*.mp4")):
        vid = Path(p).stem  # video_id.mp4 → video_id
        if vid not in vid2gloss:
            logger.warning(f"[WARNING] Cannot find gloss for {vid}")
            continue
        rows.append({"path": p, "label": vid2gloss[vid]})
    return rows


def move_files(df, split_name, output_dir):
    """Move actual video files into output_dir/split_name/label/."""
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Moving {split_name}"):
        src = row["path"]
        label = row["label"]

        dst_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)


def split_dataset(data_dir, output_dir, json_path=None, flat=False,
                  train_ratio=0.7, val_ratio=0.15, seed=42):

    # Collect files
    if flat:
        if json_path is None:
            raise ValueError("Flat mode requires --json mapping file")
        rows = collect_flat(data_dir, json_path)
    else:
        rows = collect_nested(data_dir)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(f"No video files found under {data_dir}")

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    # Check for classes too small
    if df["label"].value_counts().min() < 3:
        logger.warning("Some classes have <3 samples → stratification may fail")

    test_ratio = 1.0 - train_ratio - val_ratio

    # 1) Split test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df["label"]
    )

    # 2) Split train/val
    val_adj = val_ratio / (train_ratio + val_ratio)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_adj,
        random_state=seed,
        stratify=train_val_df["label"]
    )

    # Create output tree
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Move videos
    move_files(train_df, "train", output_dir)
    move_files(val_df, "val", output_dir)
    move_files(test_df, "test", output_dir)

    summary = f"""
[SUMMARY]
Saved videos to: {output_dir}

Total:  {len(df)}
Train:  {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)
Val:    {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)
Test:   {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)

Train distribution:
{train_df['label'].value_counts().to_string()}

Val distribution:
{val_df['label'].value_counts().to_string()}

Test distribution:
{test_df['label'].value_counts().to_string()}
"""
    logger.info(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split dataset and MOVE actual videos into folders.")
    p.add_argument("--data_dir", default="data/raw", help="Directory containing videos")
    p.add_argument("--output_dir", default="data/split", help="Output split dataset directory")
    p.add_argument("--json", default=None, help="JSON mapping path (required for flat mode)")
    p.add_argument("--flat", action="store_true", help="Use flat mode (videos directly inside data_dir)")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logger = setup_logger("split_dataset")
    log_arguments(logger=logger, args=args)

    split_dataset(
        args.data_dir,
        args.output_dir,
        json_path=args.json,
        flat=args.flat,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
