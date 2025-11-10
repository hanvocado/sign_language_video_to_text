"""Utilities to gather dataset index and compute scaler for normalization."""
import os, pandas as pd, numpy as np, joblib, argparse, sys
from pathlib import Path
from src.config.config import NPY_DIR
from sklearn.preprocessing import StandardScaler

def gather_index(npy_dir=NPY_DIR, out_csv='data/splits/dataset_index.csv'):
    rows = []
    for label in sorted(os.listdir(npy_dir)):
        label_dir = os.path.join(npy_dir, label)
        if not os.path.isdir(label_dir): continue
        for p in sorted(os.listdir(label_dir)):
            if p.endswith('.npy'):
                rows.append({'path': os.path.join(label_dir, p), 'label': label})
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote index with {len(df)} samples -> {out_csv}")
    return out_csv

def compute_scaler(index_csv='data/splits/train.csv', scaler_path='models/checkpoints/scaler.joblib', sample_limit=1000):
    df = pd.read_csv(index_csv)
    sample_paths = df['path'].sample(min(sample_limit, len(df)), random_state=42).tolist()
    Xs = []
    for p in sample_paths:
        a = np.load(p)
        # use mean frame vector as a representative per sample
        Xs.append(a.mean(axis=0))
    X = np.vstack(Xs)
    scaler = StandardScaler().fit(X)
    Path = os.path
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler -> {scaler_path}")
    return scaler_path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index_csv', default='data/splits/train.csv')
    p.add_argument('--scaler_path', default='models/checkpoints/scaler.joblib')
    args = p.parse_args()
    compute_scaler(args.index_csv, args.scaler_path)
