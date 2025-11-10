"""Split dataset of .npy files into train/val/test csv indexes."""
import os, glob, argparse, pandas as pd, random, sys
from pathlib import Path

def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
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
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved splits to {output_dir}: total={n}, train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/npy')
    p.add_argument('--output_dir', default='data/splits')
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    split_dataset(args.data_dir, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
