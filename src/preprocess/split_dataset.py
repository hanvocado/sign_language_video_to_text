"""Split dataset of .npy files into train/val/test csv indexes with balanced class distribution."""
import os, glob, argparse, pandas as pd, random, sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.utils.logger import *

def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    # Collect all .npy files with their labels
    rows = []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for p in glob.glob(os.path.join(label_dir, '*.npy')):
            rows.append({'path': p, 'label': label})
    
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(f"No .npy files found under {data_dir}")
    
    # Log initial distribution
    logger.info(f"Total samples found: {len(df)}")
    logger.info(f"Class distribution:\n{df['label'].value_counts().to_string()}")
    
    # Check for classes with insufficient samples
    min_samples_per_class = df['label'].value_counts().min()
    if min_samples_per_class < 3:
        warning_msg = f"WARNING: Some classes have fewer than 3 samples (min={min_samples_per_class}). Stratified split may fail."
        logger.warning(warning_msg)
    
    # Calculate test ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # First split: separate train+val from test (stratified)
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=seed, 
        stratify=df['label']
    )
    
    # Second split: separate train from val (stratified)
    # Adjust val_ratio relative to train+val size
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_ratio_adjusted, 
        random_state=seed, 
        stratify=train_val_df['label']
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save splits to CSV
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Log results
    summary = f"""
Saved splits to {output_dir}:
Total samples: {len(df)}s
Train samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)
Val samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)
Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)

Train class distribution:
{train_df['label'].value_counts().to_string()}

Val class distribution:
{val_df['label'].value_counts().to_string()}

Test class distribution:
{test_df['label'].value_counts().to_string()}
"""
    logger.info(summary)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Split dataset with stratified sampling to ensure balanced class distribution')
    p.add_argument('--data_dir', default='data/npy', help='Directory containing .npy files organized by label')
    p.add_argument('--output_dir', default='data/splits', help='Output directory for train/val/test CSV files')
    p.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training samples (default: 0.7)')
    p.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation samples (default: 0.15)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    args = p.parse_args()

    logger = setup_logger("split_dataset")
    log_arguments(logger=logger, args=args)
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) must be < 1.0")
    
    split_dataset(args.data_dir, args.output_dir, args.train_ratio, args.val_ratio, args.seed)