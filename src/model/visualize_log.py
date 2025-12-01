"""
Training Log Visualizer
========================
Parse training log files and generate learning curve visualizations.
"""

import os
import re
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class EpochMetrics:
    """Store metrics for a single epoch"""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    time: float
    is_best: bool = False


@dataclass
class TrainingConfig:
    """Store training configuration"""
    model_type: str = "unknown"
    hidden_dim: int = 0
    num_layers: int = 0
    dropout: float = 0.0
    lr: float = 0.0
    batch_size: int = 0
    seq_len: int = 0
    num_classes: int = 0
    train_samples: int = 0
    val_samples: int = 0
    data_dir: str = ""


@dataclass
class TrainingResult:
    """Store final training results"""
    best_epoch: int = 0
    best_val_acc: float = 0.0
    test_acc: float = 0.0
    early_stopped: bool = False
    final_epoch: int = 0


def parse_log_file(log_path: str) -> Tuple[TrainingConfig, List[EpochMetrics], TrainingResult]:
    """
    Parse a training log file and extract metrics.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Tuple of (config, epoch_metrics, results)
    """
    config = TrainingConfig()
    epochs: List[EpochMetrics] = []
    results = TrainingResult()
    
    # Regex patterns
    epoch_pattern = re.compile(
        r'Epoch\s+(\d+)/\d+\s+\|\s+'
        r'Train Loss:\s+([\d.]+)\s+\|\s+'
        r'Train Acc:\s+([\d.]+)\s+\|\s+'
        r'Val Loss:\s+([\d.]+)\s+\|\s+'
        r'Val Acc:\s+([\d.]+)\s+\|\s+'
        r'LR:\s+([\d.]+)\s+\|\s+'
        r'Time:\s+([\d.]+)s'
    )
    
    best_pattern = re.compile(r'New best!')
    config_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}.*\|\s+(\w+):\s+(.+)$')
    train_samples_pattern = re.compile(r'Training samples:\s+(\d+)')
    val_samples_pattern = re.compile(r'Validation samples:\s+(\d+)')
    num_classes_pattern = re.compile(r'Number of classes:\s+(\d+)')
    best_val_pattern = re.compile(r'Best validation accuracy:\s+([\d.]+)\s+at epoch\s+(\d+)')
    test_acc_pattern = re.compile(r'Test Accuracy:\s+([\d.]+)')
    early_stop_pattern = re.compile(r'Early stopping at epoch\s+(\d+)')
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    is_next_best = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Parse epoch metrics
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch = EpochMetrics(
                epoch=int(epoch_match.group(1)),
                train_loss=float(epoch_match.group(2)),
                train_acc=float(epoch_match.group(3)),
                val_loss=float(epoch_match.group(4)),
                val_acc=float(epoch_match.group(5)),
                lr=float(epoch_match.group(6)),
                time=float(epoch_match.group(7)),
                is_best=False
            )
            epochs.append(epoch)
            continue
        
        # Check if this epoch was best
        if best_pattern.search(line) and epochs:
            epochs[-1].is_best = True
            continue
        
        # Parse configuration
        config_match = config_pattern.match(line)
        if config_match:
            key, value = config_match.group(1), config_match.group(2).strip()
            if key == 'model_type':
                config.model_type = value
            elif key == 'hidden_dim':
                config.hidden_dim = int(value)
            elif key == 'num_layers':
                config.num_layers = int(value)
            elif key == 'dropout':
                config.dropout = float(value)
            elif key == 'lr':
                config.lr = float(value)
            elif key == 'batch_size':
                config.batch_size = int(value)
            elif key == 'seq_len':
                config.seq_len = int(value)
            elif key == 'data_dir':
                config.data_dir = value
            continue
        
        # Parse sample counts
        train_match = train_samples_pattern.search(line)
        if train_match:
            config.train_samples = int(train_match.group(1))
            continue
            
        val_match = val_samples_pattern.search(line)
        if val_match:
            config.val_samples = int(val_match.group(1))
            continue
            
        classes_match = num_classes_pattern.search(line)
        if classes_match:
            config.num_classes = int(classes_match.group(1))
            continue
        
        # Parse results
        best_val_match = best_val_pattern.search(line)
        if best_val_match:
            results.best_val_acc = float(best_val_match.group(1))
            results.best_epoch = int(best_val_match.group(2))
            continue
            
        test_match = test_acc_pattern.search(line)
        if test_match:
            results.test_acc = float(test_match.group(1))
            continue
            
        early_match = early_stop_pattern.search(line)
        if early_match:
            results.early_stopped = True
            results.final_epoch = int(early_match.group(1))
            continue
    
    if epochs and not results.final_epoch:
        results.final_epoch = epochs[-1].epoch
    
    return config, epochs, results


def plot_learning_curves(
    epochs: List[EpochMetrics],
    config: TrainingConfig,
    results: TrainingResult,
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot learning curves (loss and accuracy).
    
    Args:
        epochs: List of epoch metrics
        config: Training configuration
        results: Training results
        output_path: Path to save the plot
        show: Whether to display the plot
    """
    if not epochs:
        print("No epoch data to plot!")
        return
    
    # Extract data
    epoch_nums = [e.epoch for e in epochs]
    train_loss = [e.train_loss for e in epochs]
    val_loss = [e.val_loss for e in epochs]
    train_acc = [e.train_acc for e in epochs]
    val_acc = [e.val_acc for e in epochs]
    lrs = [e.lr for e in epochs]
    best_epochs = [e.epoch for e in epochs if e.is_best]
    best_val_accs = [e.val_acc for e in epochs if e.is_best]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Training Curves - {config.model_type.upper()} '
        f'(hidden={config.hidden_dim}, layers={config.num_layers}, dropout={config.dropout})',
        fontsize=14, fontweight='bold'
    )
    
    # Colors
    TRAIN_COLOR = '#2563eb'  # Blue
    VAL_COLOR = '#dc2626'    # Red
    BEST_COLOR = '#16a34a'   # Green
    LR_COLOR = '#9333ea'     # Purple
    
    # ===== Plot 1: Loss =====
    ax1 = axes[0, 0]
    ax1.plot(epoch_nums, train_loss, color=TRAIN_COLOR, linewidth=2, label='Train Loss', alpha=0.8)
    ax1.plot(epoch_nums, val_loss, color=VAL_COLOR, linewidth=2, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(epoch_nums) + 1])
    
    # Add min loss markers
    min_val_loss_idx = np.argmin(val_loss)
    ax1.axvline(x=epoch_nums[min_val_loss_idx], color=BEST_COLOR, linestyle='--', alpha=0.5, linewidth=1)
    ax1.scatter([epoch_nums[min_val_loss_idx]], [val_loss[min_val_loss_idx]], 
                color=BEST_COLOR, s=100, zorder=5, marker='*', label=f'Min Val Loss: {val_loss[min_val_loss_idx]:.4f}')
    
    # ===== Plot 2: Accuracy =====
    ax2 = axes[0, 1]
    ax2.plot(epoch_nums, train_acc, color=TRAIN_COLOR, linewidth=2, label='Train Acc', alpha=0.8)
    ax2.plot(epoch_nums, val_acc, color=VAL_COLOR, linewidth=2, label='Val Acc', alpha=0.8)
    
    # Mark best epochs
    if best_epochs:
        ax2.scatter(best_epochs, best_val_accs, color=BEST_COLOR, s=50, zorder=5, 
                   marker='^', label='New Best', alpha=0.7)
    
    # Mark the overall best
    if results.best_epoch:
        best_idx = next((i for i, e in enumerate(epochs) if e.epoch == results.best_epoch), None)
        if best_idx is not None:
            ax2.axvline(x=results.best_epoch, color=BEST_COLOR, linestyle='--', alpha=0.5, linewidth=1)
            ax2.scatter([results.best_epoch], [epochs[best_idx].val_acc], 
                       color=BEST_COLOR, s=150, zorder=6, marker='*')
            ax2.annotate(f'Best: {results.best_val_acc:.2%}\n(epoch {results.best_epoch})',
                        xy=(results.best_epoch, epochs[best_idx].val_acc),
                        xytext=(results.best_epoch + 5, epochs[best_idx].val_acc + 0.05),
                        fontsize=9, color=BEST_COLOR,
                        arrowprops=dict(arrowstyle='->', color=BEST_COLOR, alpha=0.7))
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(epoch_nums) + 1])
    ax2.set_ylim([0, min(1.0, max(max(train_acc), max(val_acc)) + 0.1)])
    
    # ===== Plot 3: Learning Rate =====
    ax3 = axes[1, 0]
    ax3.plot(epoch_nums, lrs, color=LR_COLOR, linewidth=2, marker='o', markersize=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Learning Rate', fontsize=11)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, max(epoch_nums) + 1])
    ax3.set_yscale('log')
    
    # Add LR change annotations
    lr_changes = []
    for i in range(1, len(lrs)):
        if lrs[i] != lrs[i-1]:
            lr_changes.append((epoch_nums[i], lrs[i]))
    
    for epoch, lr in lr_changes[:5]:  # Show first 5 changes
        ax3.axvline(x=epoch, color=LR_COLOR, linestyle=':', alpha=0.5)
        ax3.annotate(f'{lr:.1e}', xy=(epoch, lr), xytext=(epoch+2, lr*1.5),
                    fontsize=8, color=LR_COLOR)
    
    # ===== Plot 4: Train-Val Gap (Overfitting Indicator) =====
    ax4 = axes[1, 1]
    
    # Calculate gaps
    loss_gap = [t - v for t, v in zip(val_loss, train_loss)]
    acc_gap = [t - v for t, v in zip(train_acc, val_acc)]
    
    ax4_twin = ax4.twinx()
    
    line1, = ax4.plot(epoch_nums, loss_gap, color='#f97316', linewidth=2, label='Loss Gap (Val - Train)')
    line2, = ax4_twin.plot(epoch_nums, acc_gap, color='#06b6d4', linewidth=2, label='Acc Gap (Train - Val)')
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss Gap (Val - Train)', fontsize=11, color='#f97316')
    ax4_twin.set_ylabel('Acc Gap (Train - Val)', fontsize=11, color='#06b6d4')
    ax4.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, max(epoch_nums) + 1])
    
    # Legend
    ax4.legend([line1, line2], ['Loss Gap (Val - Train)', 'Acc Gap (Train - Val)'], 
              loc='upper left', fontsize=9)
    
    # Add text box with summary
    summary_text = (
        f"Training Summary\n"
        f"{'─' * 25}\n"
        f"Model: {config.model_type.upper()}\n"
        f"Classes: {config.num_classes}\n"
        f"Train/Val: {config.train_samples}/{config.val_samples}\n"
        f"{'─' * 25}\n"
        f"Best Val Acc: {results.best_val_acc:.2%}\n"
        f"Best Epoch: {results.best_epoch}\n"
        f"Test Acc: {results.test_acc:.2%}\n"
        f"{'─' * 25}\n"
        f"Final Epoch: {results.final_epoch}\n"
        f"Early Stop: {'Yes' if results.early_stopped else 'No'}"
    )
    
    # Add text box
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    fig.text(0.98, 0.02, summary_text, fontsize=9, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=props)
    
    plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to: {output_path}")
    
    # Show
    if show:
        plt.show()
    
    plt.close()


def plot_accuracy_comparison(
    epochs: List[EpochMetrics],
    config: TrainingConfig,
    results: TrainingResult,
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create a simple, clean accuracy plot for reports.
    """
    if not epochs:
        return
    
    epoch_nums = [e.epoch for e in epochs]
    train_acc = [e.train_acc * 100 for e in epochs]  # Convert to percentage
    val_acc = [e.val_acc * 100 for e in epochs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot
    ax.plot(epoch_nums, train_acc, color='#2563eb', linewidth=2.5, label='Train Accuracy')
    ax.plot(epoch_nums, val_acc, color='#dc2626', linewidth=2.5, label='Validation Accuracy')
    
    # Mark best epoch
    if results.best_epoch:
        best_idx = next((i for i, e in enumerate(epochs) if e.epoch == results.best_epoch), None)
        if best_idx is not None:
            ax.axvline(x=results.best_epoch, color='#16a34a', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.scatter([results.best_epoch], [val_acc[best_idx]], 
                      color='#16a34a', s=100, zorder=5, marker='*')
            ax.annotate(f'Best: {results.best_val_acc*100:.1f}%',
                       xy=(results.best_epoch, val_acc[best_idx]),
                       xytext=(results.best_epoch + 3, val_acc[best_idx] + 3),
                       fontsize=10, color='#16a34a', fontweight='bold')
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Training Progress - {config.model_type.upper()} Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(epoch_nums) + 1])
    ax.set_ylim([0, 100])
    
    # Add final results text
    ax.text(0.02, 0.98, 
            f"Best Val: {results.best_val_acc*100:.1f}% (epoch {results.best_epoch})\n"
            f"Test: {results.test_acc*100:.1f}%",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        # Save with different name
        base, ext = os.path.splitext(output_path)
        simple_path = f"{base}_simple{ext}"
        plt.savefig(simple_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved simple plot to: {simple_path}")
    
    if show:
        plt.show()
    
    plt.close()


def print_summary(config: TrainingConfig, epochs: List[EpochMetrics], results: TrainingResult):
    """Print a summary of the training"""
    print("\n" + "=" * 60)
    print("TRAINING LOG SUMMARY")
    print("=" * 60)
    
    print("\nConfiguration:")
    print(f"   Model: {config.model_type.upper()}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Num layers: {config.num_layers}")
    print(f"   Dropout: {config.dropout}")
    print(f"   Learning rate: {config.lr}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Sequence length: {config.seq_len}")
    
    print(f"\nDataset:")
    print(f"   Data dir: {config.data_dir}")
    print(f"   Classes: {config.num_classes}")
    print(f"   Train samples: {config.train_samples}")
    print(f"   Val samples: {config.val_samples}")
    
    print(f"\nTraining Progress:")
    print(f"   Total epochs: {len(epochs)}")
    print(f"   Final epoch: {results.final_epoch}")
    print(f"   Early stopped: {'Yes' if results.early_stopped else 'No'}")
    
    print(f"\nResults:")
    print(f"   Best val accuracy: {results.best_val_acc:.2%} (epoch {results.best_epoch})")
    print(f"   Test accuracy: {results.test_acc:.2%}")
    
    # Calculate overfitting
    if epochs:
        final_train_acc = epochs[-1].train_acc
        final_val_acc = epochs[-1].val_acc
        overfit_gap = final_train_acc - final_val_acc
        print(f"\nOverfitting Analysis:")
        print(f"   Final train acc: {final_train_acc:.2%}")
        print(f"   Final val acc: {final_val_acc:.2%}")
        print(f"   Gap: {overfit_gap:.2%}")
        if overfit_gap > 0.15:
            print(f"   Status: Significant overfitting detected")
        elif overfit_gap > 0.05:
            print(f"   Status: Mild overfitting")
        else:
            print(f"   Status: Good generalization")
    
    print("\n" + "=" * 60)


def export_metrics(epochs: List[EpochMetrics], output_path: str):
    """Export metrics to JSON"""
    data = {
        'epochs': [
            {
                'epoch': e.epoch,
                'train_loss': e.train_loss,
                'train_acc': e.train_acc,
                'val_loss': e.val_loss,
                'val_acc': e.val_acc,
                'lr': e.lr,
                'time': e.time,
                'is_best': e.is_best
            }
            for e in epochs
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported metrics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_log.py --log_file logs/train.log
  python visualize_log.py --log_file logs/train.log --output_dir plots --show
  python visualize_log.py --log_file logs/train.log --export_json metrics.json
        """
    )
    parser.add_argument('--log_file', required=True, help='Path to training log file')
    parser.add_argument('--output_dir', default=None, help='Directory to save plots')
    parser.add_argument('--show', action='store_true', help='Display plots')
    parser.add_argument('--export_json', default=None, help='Export metrics to JSON file')
    parser.add_argument('--simple', action='store_true', help='Generate simple plot only')
    
    args = parser.parse_args()
    
    # Check log file exists
    if not os.path.exists(args.log_file):
        print(f"Log file not found: {args.log_file}")
        return
    
    # Parse log
    print(f"Parsing log file: {args.log_file}")
    config, epochs, results = parse_log_file(args.log_file)
    
    if not epochs:
        print("No epoch data found in log file!")
        return
    
    print(f"   Found {len(epochs)} epochs")
    
    # Print summary
    print_summary(config, epochs, results)
    
    # Determine output path
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        log_name = Path(args.log_file).stem
        output_path = os.path.join(args.output_dir, f"{log_name}_curves.png")
    else:
        output_path = args.log_file.replace('.log', '_curves.png')
    
    # Generate plots
    if args.simple:
        plot_accuracy_comparison(epochs, config, results, output_path, args.show)
    else:
        plot_learning_curves(epochs, config, results, output_path, args.show)
        plot_accuracy_comparison(epochs, config, results, output_path, args.show)
    
    # Export JSON if requested
    if args.export_json:
        export_metrics(epochs, args.export_json)


if __name__ == '__main__':
    main()