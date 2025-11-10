import os, json, joblib, torch
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, path, extra=None):
    """Save model + optimizer state + epoch. extra is a dict for other info."""
    ensure_dir(os.path.dirname(path))
    payload = {'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'epoch': epoch}
    if extra:
        payload.update(extra)
    torch.save(payload, path)

def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    return ck

def save_label_map(label_list, path):
    """Save ordered list of labels (index->label)"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(label_list, f, ensure_ascii=False, indent=2)

def load_label_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_scaler(scaler, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)
