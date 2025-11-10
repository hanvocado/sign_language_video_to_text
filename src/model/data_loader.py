"""PyTorch Dataset that reads .npy sequences and label mapping from csv index."""
import os, sys, pandas as pd, numpy as np
import torch
from torch.utils.data import Dataset
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from src.config.config import SEQ_LEN
from sklearn.preprocessing import LabelEncoder
import joblib

class SignLanguageDataset(Dataset):
    def __init__(self, index_csv, seq_len=SEQ_LEN, scaler_path=None, label_map=None):
        self.df = pd.read_csv(index_csv)
        self.seq_len = seq_len
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        # label_map: dict label->idx OR list index->label
        if label_map is None:
            # derive mapping from the CSV (use sorted unique to keep deterministic order)
            labels = sorted(self.df['label'].unique().tolist())
            self.label_to_idx = {l:i for i,l in enumerate(labels)}
            self.idx_to_label = labels
        else:
            if isinstance(label_map, dict):
                self.label_to_idx = label_map
                # build inverse
                self.idx_to_label = [None] * (max(label_map.values())+1)
                for k,v in label_map.items():
                    self.idx_to_label[v] = k
            elif isinstance(label_map, list):
                self.idx_to_label = label_map
                self.label_to_idx = {l:i for i,l in enumerate(label_map)}
            else:
                raise ValueError('label_map must be dict or list')

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        label = row['label']
        arr = np.load(path)  # shape (seq_len, feat)
        # pad/truncate
        if arr.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif arr.shape[0] > self.seq_len:
            arr = arr[:self.seq_len]
        # scaling if provided (scaler expects 2D)
        if self.scaler is not None:
            # transform each frame: scaler.transform expects shape (n_samples, n_features)
            frames = [self.scaler.transform(f.reshape(1,-1)).reshape(-1) for f in arr]
            arr = np.stack(frames, axis=0).astype(np.float32)
        label_idx = self.label_to_idx[label]
        return torch.from_numpy(arr).float(), torch.tensor(label_idx, dtype=torch.long)
