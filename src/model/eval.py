"""
Comprehensive Evaluation Script for Sign Language Recognition

Features:
- Per-class accuracy analysis
- Confusion matrix visualization
- Error analysis with confidence scores
- Class difficulty ranking
- Prediction bias detection
- Detailed metrics export
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix,
    top_k_accuracy_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import from project
from src.model.data_loader import SignLanguageDataset
from src.utils.logger import *
from src.config.config import *
from src.model.train import build_model

# =====================================================
# Model Loading
# =====================================================

def load_model_and_config(ckpt_path, device='cpu'):
    """Load model from checkpoint with all metadata"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get label map
    label_map = checkpoint.get('label_map', None)
    if label_map is None:
        label_map_path = os.path.join(os.path.dirname(ckpt_path), 'label_map.json')
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
        else:
            raise ValueError("Could not find label_map")
    
    # Get training args
    train_args = checkpoint.get('args', {})

    # Build model
    model = build_model(
        model_type=train_args.get('model_type', 'lstm'),
        input_dim=train_args.get('input_dim', 225),
        hidden_dim=train_args.get('hidden_dim', 128),
        num_classes=len(label_map),
        num_layers=train_args.get('num_layers', 1),
        dropout=train_args.get('dropout', 0.5),
        bidirectional=train_args.get('bidirectional', False)
    )
    
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {ckpt_path}")
    logger.info(f"  Model type: {train_args.get('model_type', 'N/A')}")
    logger.info(f"  Hidden dim: {train_args.get('hidden_dim', 'N/A')}")
    logger.info(f"  Num classes: {len(label_map)}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'val_acc' in checkpoint:
        logger.info(f"  Val accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}\n")
    
    return model, label_map, train_args


# =====================================================
# Evaluation Functions
# =====================================================

def evaluate_model(model, loader, device, label_map):
    """
    Run model on data loader and collect predictions with probabilities
    
    Returns:
        predictions: list of predicted class indices
        true_labels: list of true class indices
        probabilities: numpy array of shape (n_samples, n_classes)
        confidences: list of confidence scores for predictions
    """
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    confidences = [all_probs[i, pred] for i, pred in enumerate(all_preds)]
    
    return all_preds, all_labels, all_probs, confidences


# =====================================================
# Analysis Functions
# =====================================================

def analyze_per_class_performance(true_labels, predictions, label_map):
    """
    Analyze performance for each class
    
    Returns DataFrame with per-class metrics
    """
    num_classes = len(label_map)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, 
        labels=range(num_classes),
        zero_division=0
    )
    
    # Build results DataFrame
    results = []
    for i in range(num_classes):
        class_name = label_map[i]
        
        # Get samples for this class
        class_mask = np.array(true_labels) == i
        class_preds = np.array(predictions)[class_mask]
        
        # Calculate accuracy for this class
        if support[i] > 0:
            class_acc = (class_preds == i).sum() / support[i]
        else:
            class_acc = 0.0
        
        results.append({
            'class_id': i,
            'class_name': class_name,
            'support': int(support[i]),
            'accuracy': class_acc,
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('accuracy', ascending=True)  # Worst first
    
    return df


def analyze_confusion_patterns(confusion_matrix, label_map, top_k=10):
    """
    Identify most common confusion patterns
    
    Returns list of confusion pairs sorted by frequency
    """
    confusions = []
    
    for true_idx in range(len(label_map)):
        for pred_idx in range(len(label_map)):
            if true_idx != pred_idx:  # Not on diagonal
                count = confusion_matrix[true_idx, pred_idx]
                if count > 0:
                    confusions.append({
                        'true_class': label_map[true_idx],
                        'predicted_class': label_map[pred_idx],
                        'count': int(count),
                        'true_idx': true_idx,
                        'pred_idx': pred_idx
                    })
    
    confusions = sorted(confusions, key=lambda x: x['count'], reverse=True)
    return confusions[:top_k]


def analyze_prediction_bias(predictions, label_map):
    """
    Analyze if model is biased toward certain classes
    
    Returns DataFrame with prediction distribution
    """
    pred_counts = np.bincount(predictions, minlength=len(label_map))
    
    results = []
    for i in range(len(label_map)):
        results.append({
            'class_id': i,
            'class_name': label_map[i],
            'prediction_count': int(pred_counts[i]),
            'prediction_percentage': pred_counts[i] / len(predictions) * 100
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('prediction_count', ascending=False)
    return df


def analyze_confident_errors(true_labels, predictions, probabilities, label_map, top_k=20):
    """
    Find most confident wrong predictions
    
    Returns list of confident errors
    """
    errors = []
    
    for i, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
        if true_label != pred_label:
            confidence = probabilities[i, pred_label]
            errors.append({
                'sample_idx': i,
                'true_class': label_map[true_label],
                'predicted_class': label_map[pred_label],
                'confidence': float(confidence),
                'true_idx': true_label,
                'pred_idx': pred_label
            })
    
    errors = sorted(errors, key=lambda x: x['confidence'], reverse=True)
    return errors[:top_k]


def analyze_class_difficulty(per_class_df):
    """
    Categorize classes by difficulty
    
    Returns dict with classes grouped by performance
    """
    perfect = per_class_df[per_class_df['accuracy'] == 1.0]
    good = per_class_df[(per_class_df['accuracy'] >= 0.7) & (per_class_df['accuracy'] < 1.0)]
    medium = per_class_df[(per_class_df['accuracy'] >= 0.3) & (per_class_df['accuracy'] < 0.7)]
    poor = per_class_df[(per_class_df['accuracy'] > 0) & (per_class_df['accuracy'] < 0.3)]
    failed = per_class_df[per_class_df['accuracy'] == 0.0]
    
    return {
        'perfect': perfect,
        'good': good, 
        'medium': medium,
        'poor': poor,
        'failed': failed
    }


# =====================================================
# Visualization Functions
# =====================================================

def plot_confusion_matrix(cm, label_map, output_path, normalize=True):
    """Plot and save confusion matrix"""
    if normalize:
        cm_plot = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        cm_plot = cm
        title = 'Confusion Matrix'
        fmt = 'd'
    
    num_classes = len(label_map)
    figsize = max(12, num_classes * 0.3)
    
    plt.figure(figsize=(figsize, figsize))
    
    # Show annotations only if few classes
    annot = num_classes <= 20
    
    sns.heatmap(
        cm_plot,
        annot=annot,
        fmt=fmt,
        cmap='Blues',
        xticklabels=label_map,
        yticklabels=label_map,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_per_class_accuracy(per_class_df, output_path):
    """Plot per-class accuracy bar chart"""
    plt.figure(figsize=(16, 8))
    
    df_sorted = per_class_df.sort_values('accuracy', ascending=True)
    
    colors = []
    for acc in df_sorted['accuracy']:
        if acc == 0:
            colors.append('red')
        elif acc < 0.3:
            colors.append('orange')
        elif acc < 0.7:
            colors.append('yellow')
        else:
            colors.append('green')
    
    plt.barh(range(len(df_sorted)), df_sorted['accuracy'], color=colors)
    plt.yticks(range(len(df_sorted)), df_sorted['class_name'], fontsize=8)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14)
    plt.xlim([0, 1])
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Per-class accuracy plot saved to {output_path}")


def plot_confidence_distribution(predictions, true_labels, probabilities, output_path):
    """Plot confidence distribution for correct vs incorrect predictions"""
    correct_confidences = []
    incorrect_confidences = []
    
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        confidence = probabilities[i, pred]
        if pred == true:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(correct_confidences, bins=50, alpha=0.7, label='Correct', color='green')
    plt.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', color='red')
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confidence distribution plot saved to {output_path}")


# =====================================================
# Reporting Functions
# =====================================================

def generate_text_report(args, results, output_path):
    """Generate comprehensive text report (Windows-compatible encoding)"""
    # Use UTF-8 encoding explicitly
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SIGN LANGUAGE RECOGNITION - DETAILED EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Accuracy:    {results['overall']['accuracy']:.4f} ({results['overall']['accuracy']*100:.2f}%)\n")
        f.write(f"Top-3 Accuracy:   {results['overall']['top3_accuracy']:.4f} ({results['overall']['top3_accuracy']*100:.2f}%)\n")
        f.write(f"Top-5 Accuracy:   {results['overall']['top5_accuracy']:.4f} ({results['overall']['top5_accuracy']*100:.2f}%)\n")
        f.write(f"Number of samples: {results['overall']['num_samples']}\n")
        f.write(f"Number of classes: {results['overall']['num_classes']}\n\n")
        
        # Class difficulty breakdown
        f.write("CLASS DIFFICULTY BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        difficulty = results['class_difficulty']
        f.write(f"Perfect (100%):     {len(difficulty['perfect'])} classes ({len(difficulty['perfect'])/results['overall']['num_classes']*100:.1f}%)\n")
        f.write(f"Good (70-99%):      {len(difficulty['good'])} classes ({len(difficulty['good'])/results['overall']['num_classes']*100:.1f}%)\n")
        f.write(f"Medium (30-69%):    {len(difficulty['medium'])} classes ({len(difficulty['medium'])/results['overall']['num_classes']*100:.1f}%)\n")
        f.write(f"Poor (1-29%):       {len(difficulty['poor'])} classes ({len(difficulty['poor'])/results['overall']['num_classes']*100:.1f}%)\n")
        f.write(f"Failed (0%):        {len(difficulty['failed'])} classes ({len(difficulty['failed'])/results['overall']['num_classes']*100:.1f}%)\n\n")
        
        # Best performing classes
        f.write("BEST PERFORMING CLASSES (Top 10)\n")
        f.write("-" * 80 + "\n")
        per_class_df = results['per_class_performance']
        best_classes = per_class_df.nlargest(10, 'accuracy')
        for _, row in best_classes.iterrows():
            f.write(f"  {row['class_name']:20s}  Acc: {row['accuracy']:.2%}  Support: {row['support']:3d}  F1: {row['f1_score']:.3f}\n")
        f.write("\n")
        
        # Worst performing classes
        f.write("WORST PERFORMING CLASSES (Bottom 10)\n")
        f.write("-" * 80 + "\n")
        worst_classes = per_class_df.nsmallest(10, 'accuracy')
        for _, row in worst_classes.iterrows():
            f.write(f"  {row['class_name']:20s}  Acc: {row['accuracy']:.2%}  Support: {row['support']:3d}  F1: {row['f1_score']:.3f}\n")
        f.write("\n")
        
        # Common confusions
        f.write("MOST COMMON CONFUSIONS (Top 10)\n")
        f.write("-" * 80 + "\n")
        for conf in results['confusions'][:10]:
            f.write(f"  '{conf['true_class']}' predicted as '{conf['predicted_class']}': {conf['count']} times\n")
        f.write("\n")
        
        # Prediction bias
        f.write("PREDICTION BIAS (Most Predicted Classes)\n")
        f.write("-" * 80 + "\n")
        bias_df = results['prediction_bias']
        for _, row in bias_df.head(10).iterrows():
            f.write(f"  {row['class_name']:20s}  Predicted {row['prediction_count']:3d} times ({row['prediction_percentage']:.1f}%)\n")
        f.write("\n")
        
        # Confident errors
        f.write("MOST CONFIDENT ERRORS (Top 10)\n")
        f.write("-" * 80 + "\n")
        for err in results['confident_errors'][:10]:
            f.write(f"  Sample {err['sample_idx']:4d}: '{err['true_class']}' -> '{err['predicted_class']}' (confidence: {err['confidence']:.3f})\n")
        f.write("\n")
        
        # Failed classes
        if len(difficulty['failed']) > 0:
            f.write("FAILED CLASSES (0% Accuracy)\n")
            f.write("-" * 80 + "\n")
            for _, row in difficulty['failed'].iterrows():
                f.write(f"  {row['class_name']:20s}  Support: {row['support']:3d}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Text report saved to {output_path}")


def export_results_json(results, output_path):
    """Export all results to JSON"""
    export_data = {
        'overall': results['overall'],
        'per_class_performance': results['per_class_performance'].to_dict('records'),
        'confusions': results['confusions'],
        'prediction_bias': results['prediction_bias'].to_dict('records'),
        'confident_errors': results['confident_errors'],
        'class_difficulty': {
            'perfect': results['class_difficulty']['perfect']['class_name'].tolist(),
            'good': results['class_difficulty']['good']['class_name'].tolist(),
            'medium': results['class_difficulty']['medium']['class_name'].tolist(),
            'poor': results['class_difficulty']['poor']['class_name'].tolist(),
            'failed': results['class_difficulty']['failed']['class_name'].tolist(),
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results exported to JSON: {output_path}")


# =====================================================
# Main Evaluation
# =====================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")
    
    model, label_map, train_args = load_model_and_config(args.ckpt, device)
    num_classes = len(label_map)
    
    seq_len = args.seq_len or train_args.get('seq_len', 30)
    
    test_ds = SignLanguageDataset(
        args.data_dir,
        seq_len=seq_len,
        source=args.source,
        split=args.split,
        normalize=True,
        augment=False,
        label_map=label_map,
    )
    
    logger.info(f"Evaluating on {args.split} set")
    logger.info(f"Number of samples: {len(test_ds)}")
    logger.info(f"Number of classes: {num_classes}\n")
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    logger.info("Running evaluation...")
    predictions, true_labels, probabilities, confidences = evaluate_model(
        model, test_loader, device, label_map
    )
    
    accuracy = accuracy_score(true_labels, predictions)
    top3_acc = top_k_accuracy_score(true_labels, probabilities, k=min(3, num_classes))
    top5_acc = top_k_accuracy_score(true_labels, probabilities, k=min(5, num_classes))
    
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL METRICS")
    logger.info("=" * 80)
    logger.info(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    logger.info(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    
    cm = confusion_matrix(true_labels, predictions)
    
    logger.info("\n" + "=" * 80)
    logger.info("PER-CLASS ANALYSIS")
    logger.info("=" * 80)
    
    per_class_df = analyze_per_class_performance(true_labels, predictions, label_map)
    
    logger.info("\nBest performing classes:")
    for _, row in per_class_df.nlargest(5, 'accuracy').iterrows():
        logger.info(f"  {row['class_name']:20s} - Accuracy: {row['accuracy']:.2%} (support: {row['support']})")
    
    logger.info("\nWorst performing classes:")
    for _, row in per_class_df.nsmallest(5, 'accuracy').iterrows():
        logger.info(f"  {row['class_name']:20s} - Accuracy: {row['accuracy']:.2%} (support: {row['support']})")
    
    class_difficulty = analyze_class_difficulty(per_class_df)
    
    logger.info("\n" + "=" * 80)
    logger.info("CLASS DIFFICULTY DISTRIBUTION")
    logger.info("=" * 80)
    logger.info(f"Perfect (100%):      {len(class_difficulty['perfect'])} classes")
    logger.info(f"Good (70-99%):       {len(class_difficulty['good'])} classes")
    logger.info(f"Medium (30-69%):     {len(class_difficulty['medium'])} classes")
    logger.info(f"Poor (1-29%):        {len(class_difficulty['poor'])} classes")
    logger.info(f"Failed (0%):         {len(class_difficulty['failed'])} classes")
    
    if len(class_difficulty['failed']) > 0:
        logger.info(f"\nFailed classes: {', '.join(class_difficulty['failed']['class_name'].tolist())}")
    
    confusions = analyze_confusion_patterns(cm, label_map, top_k=10)
    
    logger.info("\n" + "=" * 80)
    logger.info("MOST COMMON CONFUSIONS")
    logger.info("=" * 80)
    for i, conf in enumerate(confusions[:10], 1):
        logger.info(f"{i:2d}. '{conf['true_class']}' -> '{conf['predicted_class']}': {conf['count']} times")
    
    prediction_bias = analyze_prediction_bias(predictions, label_map)
    
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION BIAS (Over-predicted classes)")
    logger.info("=" * 80)
    for _, row in prediction_bias.head(10).iterrows():
        logger.info(f"  {row['class_name']:20s}: {row['prediction_count']:3d} predictions ({row['prediction_percentage']:.1f}%)")
    
    confident_errors = analyze_confident_errors(true_labels, predictions, probabilities, label_map, top_k=10)
    
    logger.info("\n" + "=" * 80)
    logger.info("MOST CONFIDENT ERRORS")
    logger.info("=" * 80)
    for i, err in enumerate(confident_errors, 1):
        logger.info(f"{i:2d}. Sample {err['sample_idx']:4d}: '{err['true_class']}' -> '{err['predicted_class']}' (conf: {err['confidence']:.3f})")
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        results = {
            'overall': {
                'accuracy': float(accuracy),
                'top3_accuracy': float(top3_acc),
                'top5_accuracy': float(top5_acc),
                'num_samples': len(test_ds),
                'num_classes': num_classes,
            },
            'per_class_performance': per_class_df,
            'confusions': confusions,
            'prediction_bias': prediction_bias,
            'confident_errors': confident_errors,
            'class_difficulty': class_difficulty,
        }
        
        report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
        generate_text_report(args, results, report_path)
        
        json_path = os.path.join(args.output_dir, 'evaluation_results.json')
        export_results_json(results, json_path)
        
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, label_map, cm_path, normalize=True)
        
        acc_path = os.path.join(args.output_dir, 'per_class_accuracy.png')
        plot_per_class_accuracy(per_class_df, acc_path)
        
        conf_path = os.path.join(args.output_dir, 'confidence_distribution.png')
        plot_confidence_distribution(predictions, true_labels, probabilities, conf_path)
        
        csv_path = os.path.join(args.output_dir, 'per_class_metrics.csv')
        per_class_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Per-class metrics saved to {csv_path}")
        
        cm_npy_path = os.path.join(args.output_dir, 'confusion_matrix.npy')
        np.save(cm_npy_path, cm)
        logger.info(f"Confusion matrix saved to {cm_npy_path}")
        
        logger.info(f"\nAll results saved to {args.output_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comprehensive Evaluation for Sign Language Recognition'
    )
    
    # Required
    parser.add_argument('--ckpt', required=True,
                        help='Path to model checkpoint')
    
    # Data
    parser.add_argument('--data_dir', default='data/wlasl/wlasl100',
                        help='Root directory with test subdir')
    parser.add_argument('--source', choices=['npy', 'video'], default='npy',
                        help='Load from .npy or extract from videos')
    parser.add_argument('--split', default='test',
                        help='Which split to evaluate (test, val, train)')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Sequence length (default: from checkpoint)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Data loading workers')
    
    # Output
    parser.add_argument('--output_dir', default='eval_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    logger = setup_logger("eval")
    main(args)