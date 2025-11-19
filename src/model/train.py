"""
Optimized training script for small sign language datasets
Includes debugging, proper regularization, and small-dataset best practices
"""
import os, sys, argparse, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import DEVICE, CKPT_DIR, SEQ_LEN
from src.model.data_loader import SignLanguageDataset
from src.utils.utils import save_checkpoint, save_label_map, ensure_dir


class SimpleLSTM(nn.Module):
    """Simplified LSTM for small datasets"""
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=10, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Use last hidden state
        last = out[:, -1, :]
        return self.classifier(last)


class SimpleGRU(nn.Module):
    """Even simpler GRU for very small datasets"""
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=10, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        out, hn = self.gru(x)
        return self.classifier(out[:, -1, :])


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        return self.early_stop


def train(args):
    ensure_dir(CKPT_DIR)
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    # Create train dataset to infer label order
    train_ds = SignLanguageDataset(args.train_csv, seq_len=args.seq_len, scaler_path=args.scaler)
    label_list = train_ds.idx_to_label
    
    # Save label map
    label_map_path = os.path.join(CKPT_DIR, 'label_map.json')
    save_label_map(label_list, label_map_path)
    print(f"Label map saved -> {label_map_path}")
    print(f"Classes ({len(label_list)}): {label_list}\n")
    
    # Recreate datasets with explicit mapping
    train_ds = SignLanguageDataset(args.train_csv, seq_len=args.seq_len, 
                                   scaler_path=args.scaler, label_map=label_list)
    val_ds = SignLanguageDataset(args.val_csv, seq_len=args.seq_len, 
                                 scaler_path=args.scaler, label_map=label_list)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # =========================================================================
    # DEBUG: Check data
    # =========================================================================
    
    print("\n" + "="*70)
    print("DATA SANITY CHECK")
    print("="*70)
    
    # Check a few samples
    for i in range(min(3, len(train_ds))):
        X, y = train_ds[i]
        print(f"Sample {i}: shape={X.shape}, label={y.item()}, "
              f"range=[{X.min():.3f}, {X.max():.3f}], "
              f"mean={X.mean():.3f}, std={X.std():.3f}")
        
        # Check for issues
        if torch.all(X == 0):
            print(f"  ⚠️  WARNING: Sample {i} is all zeros!")
        if torch.any(torch.isnan(X)):
            print(f"  ❌ ERROR: Sample {i} contains NaN!")
        if X.std() < 0.001:
            print(f"  ⚠️  WARNING: Sample {i} has very low variance!")
    
    # =========================================================================
    # DATA LOADERS
    # =========================================================================
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for debugging
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # =========================================================================
    # MODEL
    # =========================================================================
    
    num_classes = len(label_list)
    
    if args.model_type == 'lstm':
        model = SimpleLSTM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        )
    elif args.model_type == 'gru':
        model = SimpleGRU(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        )
    else:
        # Use original model
        from src.model.model import build_model
        model = build_model(
            num_classes=num_classes, 
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers,
            dropout=args.dropout, 
            bidirectional=args.bidirectional
        )
    
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model_type.upper()}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # =========================================================================
    # LOSS AND OPTIMIZER
    # =========================================================================
    
    # Use label smoothing for small datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Use AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # ----- Training -----
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        t0 = time.time()
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            
            # Forward pass
            logits = model(X)
            loss = criterion(logits, y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"❌ NaN loss at epoch {epoch}, batch {batch_idx}")
                print(f"   X range: [{X.min():.3f}, {X.max():.3f}]")
                print(f"   Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                return
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record
            train_losses.append(loss.item())
            train_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            train_labels.extend(y.cpu().numpy().tolist())
        
        avg_train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # ----- Validation -----
        model.eval()
        val_preds = []
        val_labels = []
        val_losses = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                
                logits = model(X)
                loss = criterion(logits, y)
                
                val_losses.append(loss.item())
                val_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                val_labels.extend(y.cpu().numpy().tolist())
        
        avg_val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_labels, val_preds)
        
        elapsed = time.time() - t0
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            ckpt_path = os.path.join(CKPT_DIR, "best.pth")
            save_checkpoint(
                model, optimizer, epoch, ckpt_path,
                extra={'val_acc': val_acc, 'label_list': label_list}
            )
            print(f"  ✓ New best! Saved -> {ckpt_path}")
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        # Debug: Print predictions distribution every 10 epochs
        if epoch % 10 == 0:
            print(f"  Predictions distribution: {np.bincount(val_preds, minlength=num_classes)}")
            print(f"  Ground truth distribution: {np.bincount(val_labels, minlength=num_classes)}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {os.path.join(CKPT_DIR, 'best.pth')}")
    print(f"Label map saved to: {label_map_path}")
    
    # Final check
    if best_val_acc < 0.1:
        print("\n⚠️  WARNING: Accuracy is very low!")
        print("Please run the diagnostic script:")
        print("  python diagnose_pipeline.py --train_csv", args.train_csv)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train sign language recognition model (optimized for small datasets)')
    
    # Data
    p.add_argument('--train_csv', default='data/splits/train.csv')
    p.add_argument('--val_csv', default='data/splits/val.csv')
    p.add_argument('--scaler', default=None, help='Optional scaler.joblib')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--input_dim', type=int, default=225)
    
    # Model (optimized for small datasets)
    p.add_argument('--model_type', choices=['lstm', 'gru', 'original'], default='lstm',
                   help='Model type: lstm (recommended), gru (simpler), original (your model)')
    p.add_argument('--hidden_dim', type=int, default=128,
                   help='Hidden dimension (128 for small datasets)')
    p.add_argument('--num_layers', type=int, default=1,
                   help='Number of layers (1 for small datasets)')
    p.add_argument('--dropout', type=float, default=0.5,
                   help='Dropout rate (0.5 for small datasets)')
    p.add_argument('--bidirectional', action='store_true')
    
    # Training (optimized for small datasets)
    p.add_argument('--batch_size', type=int, default=8,
                   help='Batch size (8-16 for small datasets)')
    p.add_argument('--lr', type=float, default=5e-4,
                   help='Learning rate (5e-4 for small datasets)')
    p.add_argument('--weight_decay', type=float, default=1e-3,
                   help='Weight decay for L2 regularization')
    p.add_argument('--label_smoothing', type=float, default=0.1,
                   help='Label smoothing (0.1 for small datasets)')
    p.add_argument('--epochs', type=int, default=100,
                   help='Maximum epochs')
    p.add_argument('--patience', type=int, default=20,
                   help='Early stopping patience')
    
    args = p.parse_args()
    train(args)