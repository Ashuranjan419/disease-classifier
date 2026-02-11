"""
Evaluation Metrics and Visualization

Provides metrics computation, confusion matrix, and ROC curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_NAMES, NUM_CLASSES, RESULTS_DIR


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = (np.array(y_true) == i)
        if class_mask.sum() > 0:
            class_pred = (np.array(y_pred) == i)
            metrics[f'precision_{class_name}'] = precision_score(
                class_mask, class_pred, zero_division=0
            )
            metrics[f'recall_{class_name}'] = recall_score(
                class_mask, class_pred, zero_division=0
            )
    
    # ROC-AUC if probabilities available
    if y_proba is not None:
        try:
            # Multi-class ROC-AUC
            metrics['roc_auc_macro'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )
            metrics['roc_auc_weighted'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='weighted'
            )
        except ValueError:
            # May fail if not all classes present in y_true
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None, title='Confusion Matrix'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Get unique classes actually present in the data
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    n_classes = len(unique_classes)
    class_labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}" for i in unique_classes]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_labels,
        yticklabels=class_labels,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=title
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black',
                   fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()
    
    return cm


def plot_roc_curves(y_true, y_proba, save_path=None, title='ROC Curves'):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique classes and number of classes from probabilities
    n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else len(np.unique(y_true))
    unique_classes = sorted(np.unique(y_true))
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    # Compute ROC curve for each class that exists in the data
    for idx, class_id in enumerate(unique_classes):
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
        color = colors[idx % len(colors)]
        
        # Binary labels for this class
        y_true_binary = (np.array(y_true) == class_id).astype(int)
        
        # Get probability for this class
        if len(y_proba.shape) > 1 and y_proba.shape[1] > class_id:
            y_score = y_proba[:, class_id]
        else:
            continue
        
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        except Exception as e:
            print(f"Warning: Could not compute ROC for class {class_name}: {e}")
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.close()


def print_classification_report(y_true, y_pred, class_names=None):
    """Print sklearn classification report."""
    print("\nClassification Report:")
    print("=" * 60)
    # Determine number of classes from data
    n_classes = len(np.unique(y_true))
    if class_names is None:
        if n_classes <= len(CLASS_NAMES):
            class_names = CLASS_NAMES[:n_classes]
        else:
            class_names = [f"Class {i}" for i in range(n_classes)]
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def evaluate_model(model, dataloader, device='cpu'):
    """
    Evaluate model on a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to use
        
    Returns:
        y_true, y_pred, y_proba
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_proba = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labs = batch['lab'].to(device)
            labels = batch['label']
            
            outputs = model(images, labs)
            proba = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_proba)


if __name__ == "__main__":
    # Test metrics with dummy data
    print("Testing Evaluation Metrics...")
    
    np.random.seed(42)
    n_samples = 100
    
    # Simulate predictions
    y_true = np.random.randint(0, NUM_CLASSES, n_samples)
    y_pred = np.random.randint(0, NUM_CLASSES, n_samples)
    y_proba = np.random.rand(n_samples, NUM_CLASSES)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(RESULTS_DIR, 'test_confusion_matrix.png')
    )
    
    # Plot ROC curves
    plot_roc_curves(
        y_true, y_proba,
        save_path=os.path.join(RESULTS_DIR, 'test_roc_curves.png')
    )
    
    print("\nTest complete!")
