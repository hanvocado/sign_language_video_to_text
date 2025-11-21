"""
Diagnostic script for Sign Language Recognition
Run this BEFORE training to identify data issues
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

def check_npy_files(data_dir):
    """Check all .npy files for issues"""
    print("\n" + "="*70)
    print("1. CHECKING NPY FILES")
    print("="*70)
    
    issues = []
    stats = {
        'total_files': 0,
        'empty_files': 0,
        'all_zeros': 0,
        'nan_values': 0,
        'inf_values': 0,
        'shapes': set(),
        'value_ranges': []
    }
    
    labels = []
    samples_per_label = {}
    
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        
        labels.append(label)
        npy_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        samples_per_label[label] = len(npy_files)
        
        for f in npy_files:
            path = os.path.join(label_dir, f)
            stats['total_files'] += 1
            
            try:
                arr = np.load(path)
                
                # Check shape
                stats['shapes'].add(arr.shape)
                
                # Check for issues
                if arr.size == 0:
                    stats['empty_files'] += 1
                    issues.append(f"EMPTY: {path}")
                
                if np.all(arr == 0):
                    stats['all_zeros'] += 1
                    issues.append(f"ALL ZEROS: {path}")
                
                if np.any(np.isnan(arr)):
                    stats['nan_values'] += 1
                    issues.append(f"HAS NaN: {path}")
                
                if np.any(np.isinf(arr)):
                    stats['inf_values'] += 1
                    issues.append(f"HAS Inf: {path}")
                
                # Value range
                stats['value_ranges'].append((arr.min(), arr.max(), arr.mean(), arr.std()))
                
            except Exception as e:
                issues.append(f"ERROR loading {path}: {e}")
    
    # Print results
    print(f"\nTotal files: {stats['total_files']}")
    print(f"Number of labels: {len(labels)}")
    print(f"Labels: {labels}")
    print(f"\nSamples per label:")
    for label, count in samples_per_label.items():
        print(f"  {label}: {count}")
    
    print(f"\nUnique shapes found: {stats['shapes']}")
    
    if stats['value_ranges']:
        mins = [r[0] for r in stats['value_ranges']]
        maxs = [r[1] for r in stats['value_ranges']]
        means = [r[2] for r in stats['value_ranges']]
        stds = [r[3] for r in stats['value_ranges']]
        
        print(f"\nValue statistics across all files:")
        print(f"  Min range: [{np.min(mins):.4f}, {np.max(mins):.4f}]")
        print(f"  Max range: [{np.min(maxs):.4f}, {np.max(maxs):.4f}]")
        print(f"  Mean range: [{np.min(means):.4f}, {np.max(means):.4f}]")
        print(f"  Std range: [{np.min(stds):.4f}, {np.max(stds):.4f}]")
    
    # Issues summary
    print(f"\n⚠️  ISSUES FOUND:")
    print(f"  Empty files: {stats['empty_files']}")
    print(f"  All zeros: {stats['all_zeros']}")
    print(f"  Contains NaN: {stats['nan_values']}")
    print(f"  Contains Inf: {stats['inf_values']}")
    
    if issues:
        print(f"\nFirst 10 issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    
    return len(issues) == 0


def check_csv_files(splits_dir):
    """Check train/val/test CSV files"""
    print("\n" + "="*70)
    print("2. CHECKING CSV SPLIT FILES")
    print("="*70)
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(splits_dir, f'{split}.csv')
        
        if not os.path.exists(csv_path):
            print(f"❌ Missing: {csv_path}")
            issues.append(f"Missing {split}.csv")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"\n{split}.csv:")
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for required columns
        if 'path' not in df.columns:
            issues.append(f"{split}.csv missing 'path' column")
        if 'label' not in df.columns:
            issues.append(f"{split}.csv missing 'label' column")
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"  ⚠️  NaN values: {nan_counts.to_dict()}")
            issues.append(f"{split}.csv has NaN values")
        
        # Check label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            print(f"  Labels: {len(label_counts)} unique")
            print(f"  Distribution:")
            for label, count in label_counts.items():
                print(f"    {label}: {count}")
        
        # Check if files exist
        if 'path' in df.columns:
            missing_files = 0
            for path in df['path']:
                if not os.path.exists(path):
                    missing_files += 1
            if missing_files > 0:
                print(f"  ❌ Missing files: {missing_files}")
                issues.append(f"{split}.csv has {missing_files} missing files")
    
    return len(issues) == 0


def check_label_consistency(splits_dir):
    """Check if labels are consistent across splits"""
    print("\n" + "="*70)
    print("3. CHECKING LABEL CONSISTENCY")
    print("="*70)
    
    all_labels = {}
    
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(splits_dir, f'{split}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'label' in df.columns:
                all_labels[split] = set(df['label'].dropna().unique())
    
    if len(all_labels) < 2:
        print("❌ Not enough split files to compare")
        return False
    
    # Compare labels
    train_labels = all_labels.get('train', set())
    val_labels = all_labels.get('val', set())
    test_labels = all_labels.get('test', set())
    
    print(f"Train labels: {sorted(train_labels)}")
    print(f"Val labels: {sorted(val_labels)}")
    print(f"Test labels: {sorted(test_labels)}")
    
    # Check consistency
    if train_labels != val_labels:
        diff = train_labels.symmetric_difference(val_labels)
        print(f"⚠️  Train/Val label mismatch: {diff}")
    
    if train_labels != test_labels:
        diff = train_labels.symmetric_difference(test_labels)
        print(f"⚠️  Train/Test label mismatch: {diff}")
    
    return train_labels == val_labels == test_labels


def test_data_loading(train_csv, seq_len=64):
    """Test actual data loading"""
    print("\n" + "="*70)
    print("4. TESTING DATA LOADING")
    print("="*70)
    
    try:
        # Import your dataset class
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.model.data_loader import SignLanguageDataset
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = SignLanguageDataset(train_csv, seq_len=seq_len)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {len(dataset.idx_to_label)}")
        print(f"Classes: {dataset.idx_to_label}")
        
        # Test loading a few samples
        print("\nTesting sample loading:")
        for i in range(min(5, len(dataset))):
            X, y = dataset[i]
            print(f"  Sample {i}: X shape={X.shape}, y={y.item()}, "
                  f"X range=[{X.min():.3f}, {X.max():.3f}]")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        batch_X, batch_y = next(iter(loader))
        
        print(f"\nBatch test:")
        print(f"  Batch X shape: {batch_X.shape}")
        print(f"  Batch y shape: {batch_y.shape}")
        print(f"  Batch y values: {batch_y.tolist()}")
        print(f"  Unique labels in batch: {torch.unique(batch_y).tolist()}")
        
        # Check for issues
        if torch.all(batch_X == 0):
            print("  ❌ ERROR: All batch values are zero!")
            return False
        
        if torch.any(torch.isnan(batch_X)):
            print("  ❌ ERROR: Batch contains NaN!")
            return False
        
        if len(torch.unique(batch_y)) == 1 and len(dataset) > 8:
            print("  ⚠️  WARNING: All samples in batch have same label")
        
        print("  ✅ Data loading OK")
        return True
        
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward(train_csv, seq_len=64, input_dim=225):
    """Test model forward pass"""
    print("\n" + "="*70)
    print("5. TESTING MODEL FORWARD PASS")
    print("="*70)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.model.data_loader import SignLanguageDataset
        from src.model.model import build_model
        from torch.utils.data import DataLoader
        
        # Load data
        dataset = SignLanguageDataset(train_csv, seq_len=seq_len)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        batch_X, batch_y = next(iter(loader))
        
        # Build model
        num_classes = len(dataset.idx_to_label)
        model = build_model(num_classes=num_classes, input_dim=input_dim)
        model.eval()
        
        print(f"Model: {model.__class__.__name__}")
        print(f"Input dim: {input_dim}")
        print(f"Num classes: {num_classes}")
        
        # Forward pass
        with torch.no_grad():
            output = model(batch_X)
        
        print(f"\nForward pass:")
        print(f"  Input shape: {batch_X.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check predictions
        probs = torch.softmax(output, dim=1)
        preds = output.argmax(dim=1)
        
        print(f"\nPredictions:")
        print(f"  Predicted classes: {preds.tolist()}")
        print(f"  True classes: {batch_y.tolist()}")
        print(f"  Max probabilities: {probs.max(dim=1).values.tolist()}")
        
        # Check for issues
        if torch.any(torch.isnan(output)):
            print("  ❌ ERROR: Model output contains NaN!")
            return False
        
        if torch.all(output == output[0]):
            print("  ❌ ERROR: Model outputs same values for all samples!")
            return False
        
        print("  ✅ Model forward pass OK")
        return True
        
    except Exception as e:
        print(f"❌ Error during model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gradient_flow(train_csv, seq_len=64, input_dim=225):
    """Test if gradients flow properly"""
    print("\n" + "="*70)
    print("6. TESTING GRADIENT FLOW")
    print("="*70)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.model.data_loader import SignLanguageDataset
        from src.model.model import build_model
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim
        
        # Load data
        dataset = SignLanguageDataset(train_csv, seq_len=seq_len)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        batch_X, batch_y = next(iter(loader))
        
        # Build model
        num_classes = len(dataset.idx_to_label)
        model = build_model(num_classes=num_classes, input_dim=input_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        model.train()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        
        print(f"Initial loss: {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norms = []
        zero_grads = 0
        nan_grads = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm == 0:
                    zero_grads += 1
                if np.isnan(grad_norm):
                    nan_grads += 1
        
        print(f"\nGradient statistics:")
        print(f"  Total parameters with grads: {len(grad_norms)}")
        print(f"  Zero gradients: {zero_grads}")
        print(f"  NaN gradients: {nan_grads}")
        print(f"  Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
        print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
        
        if nan_grads > 0:
            print("  ❌ ERROR: NaN gradients detected!")
            return False
        
        if zero_grads == len(grad_norms):
            print("  ❌ ERROR: All gradients are zero!")
            return False
        
        # Test one optimization step
        optimizer.step()
        output2 = model(batch_X)
        loss2 = criterion(output2, batch_y)
        
        print(f"\nAfter 1 step:")
        print(f"  New loss: {loss2.item():.4f}")
        print(f"  Loss change: {loss.item() - loss2.item():.4f}")
        
        if loss2.item() >= loss.item():
            print("  ⚠️  WARNING: Loss did not decrease after 1 step")
        else:
            print("  ✅ Loss decreased - training is working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during gradient test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose sign language recognition pipeline")
    parser.add_argument('--data_dir', default='data/npy', help='Directory with .npy files')
    parser.add_argument('--splits_dir', default='data/splits', help='Directory with CSV splits')
    parser.add_argument('--train_csv', default='data/splits/train.csv', help='Training CSV path')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--input_dim', type=int, default=225, help='Input feature dimension')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SIGN LANGUAGE RECOGNITION - DIAGNOSTIC TOOL")
    print("="*70)
    
    all_passed = True
    
    # Run all checks
    if os.path.exists(args.data_dir):
        if not check_npy_files(args.data_dir):
            all_passed = False
    else:
        print(f"❌ Data directory not found: {args.data_dir}")
        all_passed = False
    
    if os.path.exists(args.splits_dir):
        if not check_csv_files(args.splits_dir):
            all_passed = False
        if not check_label_consistency(args.splits_dir):
            all_passed = False
    else:
        print(f"❌ Splits directory not found: {args.splits_dir}")
        all_passed = False
    
    if os.path.exists(args.train_csv):
        if not test_data_loading(args.train_csv, args.seq_len):
            all_passed = False
        if not test_model_forward(args.train_csv, args.seq_len, args.input_dim):
            all_passed = False
        if not check_gradient_flow(args.train_csv, args.seq_len, args.input_dim):
            all_passed = False
    else:
        print(f"❌ Training CSV not found: {args.train_csv}")
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if all_passed:
        print("✅ All checks passed! The pipeline should be working.")
        print("\nIf accuracy is still low, try:")
        print("  1. Increase augmentation")
        print("  2. Reduce model complexity")
        print("  3. Check if normalization is correct")
        print("  4. Increase training epochs")
    else:
        print("❌ Some checks failed! Please fix the issues above.")
    
    return all_passed


if __name__ == "__main__":
    main()