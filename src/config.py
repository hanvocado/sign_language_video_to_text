import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
NPY_DIR = os.path.join(DATA_DIR, "npy")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# ensure directories exist
for p in [DATA_DIR, RAW_DIR, NPY_DIR, SPLIT_DIR, MODEL_DIR, CKPT_DIR]:
    os.makedirs(p, exist_ok=True)

# Feature dimensions after extracting Face(468*3) + LHand(21*3) + RHand(21*3)
FEATURE_DIM = 1530
SEQ_LEN = 64
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 40
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
