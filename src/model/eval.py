"""Evaluate a trained checkpoint on a CSV index file (test set)."""
import os, sys, argparse
import torch
from src.model.data_loader import SignLanguageDataset
from torch.utils.data import DataLoader
from src.model.model import build_model
from src.utils.utils import load_label_map, load_checkpoint
from src.config.config import DEVICE
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(args):
    label_list = load_label_map(args.label_map)
    ds = SignLanguageDataset(args.index_csv, seq_len=args.seq_len, scaler_path=args.scaler, label_map=label_list)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model = build_model(num_classes=len(label_list), input_dim=args.input_dim).to(DEVICE)
    ck = load_checkpoint(args.ckpt, device=DEVICE)
    model.load_state_dict(ck['model_state'])
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for X,y in loader:
            X = X.to(DEVICE)
            logits = model(X)
            p = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(p)
            ys.extend(y.numpy().tolist())
    print(classification_report(ys, preds, target_names=label_list))
    print('Confusion matrix:')
    print(confusion_matrix(ys, preds))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index_csv', default='data/splits/test.csv')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--label_map', required=True)
    p.add_argument('--scaler', default=None)
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=32)
    # p.add_argument('--input_dim', type=int, default=1530)
    p.add_argument('--input_dim', type=int, default=225)
    args = p.parse_args()
    evaluate(args)
