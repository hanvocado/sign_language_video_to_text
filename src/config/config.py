import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
# FEATURE_DIM = 1530

# Feature dimensions after extracting Pose(33*3) + LHand(21*3) + RHand(21*3)
# Updated: removed face landmarks, added pose landmarks
FEATURE_DIM = 225  # 99 + 63 + 63 = 225
SEQ_LEN = 30
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_TYPE = 'gru'
INPUT_DIM = FEATURE_DIM
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.6
BIDIRECTIONAL = False