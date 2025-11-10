"""Training script for LSTM-based sign recognition."""
import os, sys, argparse, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from src.config.config import DEVICE, CKPT_DIR, BATCH_SIZE, LR, EPOCHS, SEQ_LEN
from src.model.data_loader import SignLanguageDataset
from src.model.model import build_model
from src.utils.utils import save_checkpoint, save_label_map, ensure_dir
import numpy as np
from sklearn.metrics import accuracy_score
import json

def train(args):
    ensure_dir(CKPT_DIR)
    # Build training dataset and label map from training csv (deterministic order)
    train_df = None
    # Create train dataset to infer label order
    train_ds = SignLanguageDataset(args.train_csv, seq_len=args.seq_len, scaler_path=args.scaler)
    # label list in consistent order
    label_list = train_ds.idx_to_label
    label_map = {l:i for i,l in enumerate(label_list)}
    # save label map to checkpoint dir for inference later
    label_map_path = os.path.join(CKPT_DIR, 'label_map.json')
    save_label_map(label_list, label_map_path)
    print(f"Label map saved -> {label_map_path}")

    # Recreate train/val datasets with explicit mapping so labels align
    train_ds = SignLanguageDataset(args.train_csv, seq_len=args.seq_len, scaler_path=args.scaler, label_map=label_list)
    val_ds = SignLanguageDataset(args.val_csv, seq_len=args.seq_len, scaler_path=args.scaler, label_map=label_list)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(label_list)
    model = build_model(num_classes=num_classes, input_dim=args.input_dim,
                        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                        dropout=args.dropout, bidirectional=args.bidirectional).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        t0 = time.time()
        for X,y in train_loader:
            X = X.to(DEVICE); y = y.to(DEVICE)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # validation
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for X,y in val_loader:
                X = X.to(DEVICE); y = y.to(DEVICE)
                logits = model(X)
                p = logits.argmax(dim=1).cpu().numpy().tolist()
                preds.extend(p)
                ys.extend(y.cpu().numpy().tolist())
        val_acc = accuracy_score(ys, preds) if ys else 0.0
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  time={elapsed:.1f}s")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CKPT_DIR, f"best_epoch{epoch}_acc{val_acc:.4f}.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path, extra={'val_acc': val_acc, 'label_list': label_list})
            print(f"Saved checkpoint -> {ckpt_path}")

    print(f"Training finished. Best val acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', default='data/splits/train.csv')
    p.add_argument('--val_csv', default='data/splits/val.csv')
    p.add_argument('--scaler', default=None, help='Optional scaler.joblib to normalize frames')
    p.add_argument('--seq_len', type=int, default=SEQ_LEN)
    p.add_argument('--input_dim', type=int, default=225)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    p.add_argument('--lr', type=float, default=LR)
    p.add_argument('--epochs', type=int, default=EPOCHS)
    args = p.parse_args()
    train(args)
