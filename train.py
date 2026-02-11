"""
Training Script for Multimodal Disease Classification

Supports training fusion model and baselines with various configurations.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    SCHEDULER_STEP, SCHEDULER_GAMMA, EARLY_STOPPING_PATIENCE,
    SEED, MODELS_DIR, RESULTS_DIR, NUM_CLASSES
)

# Import all loaders
from data.dataset import (
    create_data_loaders, create_yolo_ct_loaders, create_covid_ct_classfolder_loaders, 
    LIDCIDRIDicomDataset, create_unified_loaders
)
from models.fusion_model import create_model
from utils.metrics import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_training_history, print_classification_report, evaluate_model
)
from utils.logger import TrainingLogger


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        labs = batch['lab'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images, labs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    return total_loss / total, correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labs = batch['lab'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, labs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / total, correct / total


def train(model, train_loader, val_loader, config, device='cpu'):
    """
    Full training loop.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Training configuration dict
        device: Device to train on
        
    Returns:
        Trained model, training history
    """
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', LEARNING_RATE),
        weight_decay=config.get('weight_decay', WEIGHT_DECAY)
    )
    scheduler = StepLR(
        optimizer,
        step_size=config.get('scheduler_step', SCHEDULER_STEP),
        gamma=config.get('scheduler_gamma', SCHEDULER_GAMMA)
    )
    
    # Logger
    experiment_name = config.get('experiment_name', 'experiment')
    logger = TrainingLogger(experiment_name)
    logger.log_config(config)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    num_epochs = config.get('num_epochs', NUM_EPOCHS)
    patience = config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
    
    model.to(device)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs(MODELS_DIR, exist_ok=True)
            checkpoint_path = os.path.join(MODELS_DIR, f'{experiment_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
            logger.log_info(f"Saved best model with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.log_info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    checkpoint = torch.load(os.path.join(MODELS_DIR, f'{experiment_name}_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, logger.get_history()


def run_experiment(config):
    """
    Run a complete experiment: train and evaluate.
    
    Args:
        config: Experiment configuration
    """
    print("=" * 60)
    print(f"Running Experiment: {config['experiment_name']}")
    print("=" * 60)
    
    # Set seed
    set_seed(config.get('seed', SEED))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    # Data selection logic
    print("\nLoading data...")
    dataset_type = config.get('dataset', 'synthetic')
    batch_size = config.get('batch_size', BATCH_SIZE)
    # Dataset is inside disease_classifier folder
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    
    if dataset_type == 'unified':
        # Unified dataset combining COVID, Kidney, and optionally Lungs
        base_dir = os.path.join(workspace_root, 'dataset')
        train_loader, val_loader, test_loader, lab_normalizer = create_unified_loaders(
            dataset_root=base_dir,
            batch_size=batch_size,
            include_lungs=config.get('include_lungs', False),
            max_samples_per_class=config.get('max_samples_per_class', None),
            use_real_labs=config.get('use_real_labs', True)  # Use real NHANES data by default
        )
        num_classes = NUM_CLASSES  # 4 classes: Normal, Tumor, Infection, Inflammatory
    elif dataset_type == 'kidney':
        base_dir = os.path.join(workspace_root, 'dataset', 'ctscans_kidney')
        train_loader = create_yolo_ct_loaders(base_dir, split='train', batch_size=batch_size)
        val_loader = create_yolo_ct_loaders(base_dir, split='valid', batch_size=batch_size)
        test_loader = create_yolo_ct_loaders(base_dir, split='test', batch_size=batch_size)
        lab_normalizer = None
        num_classes = 2
    elif dataset_type == 'covid':
        base_dir = os.path.join(workspace_root, 'dataset', 'ctscan_covid')
        train_loader, val_loader, test_loader, lab_normalizer = create_covid_ct_classfolder_loaders(base_dir, batch_size=batch_size)
        num_classes = 2  # COVID vs non-COVID
    elif dataset_type == 'lungs':
        nodule_dir = os.path.join(workspace_root, 'dataset', 'ctscans_lungs', 'nodules', 'LIDC-IDRI')
        non_nodule_dir = os.path.join(workspace_root, 'dataset', 'ctscans_lungs', 'no_nodules', 'manifest-1769675599632', 'LIDC-IDRI')
        dataset = LIDCIDRIDicomDataset([nodule_dir, non_nodule_dir])
        total_size = len(dataset)
        indices = np.arange(total_size)
        np.random.seed(SEED)
        np.random.shuffle(indices)
        train_end = int(0.7 * total_size)
        val_end = int(0.85 * total_size)
        from torch.utils.data import Subset, DataLoader
        train_loader = DataLoader(Subset(dataset, indices[:train_end]), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, indices[train_end:val_end]), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, indices[val_end:]), batch_size=batch_size, shuffle=False)
        lab_normalizer = None
        num_classes = 2  # nodule vs non-nodule
    else:
        # Default: synthetic
        train_loader, val_loader, test_loader, lab_normalizer = create_data_loaders(
            use_synthetic=True,
            n_synthetic_samples=config.get('n_synthetic_samples', 200),
            batch_size=batch_size
        )
        num_classes = NUM_CLASSES  # Default 4 classes
    
    # Model
    print("\nCreating model...")
    model = create_model(
        model_type=config.get('model_type', 'fusion'),
        fusion_method=config.get('fusion_method', 'concat'),
        use_simple_cnn=config.get('use_simple_cnn', True),
        use_lab_attention=config.get('use_lab_attention', False),
        num_classes=num_classes
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    print("\nTraining...")
    model, history = train(model, train_loader, val_loader, config, device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_true, y_pred, y_proba = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    print("\nTest Results:")
    print("-" * 40)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    if 'roc_auc_macro' in metrics:
        print(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    experiment_name = config['experiment_name']
    
    # Plots
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(RESULTS_DIR, f'{experiment_name}_confusion_matrix.png'),
        title=f'Confusion Matrix - {experiment_name}'
    )
    
    plot_roc_curves(
        y_true, y_proba,
        save_path=os.path.join(RESULTS_DIR, f'{experiment_name}_roc_curves.png'),
        title=f'ROC Curves - {experiment_name}'
    )
    
    plot_training_history(
        history,
        save_path=os.path.join(RESULTS_DIR, f'{experiment_name}_training_history.png')
    )
    
    # Save metrics to JSON
    results = {
        'config': config,
        'metrics': metrics,
        'history': history
    }
    
    with open(os.path.join(RESULTS_DIR, f'{experiment_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    return model, metrics, history


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description='Train Multimodal Disease Classifier')
    parser.add_argument('--model_type', type=str, default='fusion',
                       choices=['fusion', 'image_only', 'lab_only'],
                       help='Model type to train')
    parser.add_argument('--fusion_method', type=str, default='concat',
                       choices=['concat', 'gated', 'attention'],
                       help='Fusion method for multimodal model')
    parser.add_argument('--use_simple_cnn', action='store_true', default=True,
                       help='Use simple CNN instead of pretrained')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--n_samples', type=int, default=200,
                       help='Synthetic samples per class')
    parser.add_argument('--dataset', type=str, default='unified',
                       choices=['synthetic', 'kidney', 'covid', 'lungs', 'unified'],
                       help='Dataset to use: synthetic, kidney, covid, lungs, unified (all combined)')
    parser.add_argument('--include_lungs', action='store_true', default=False,
                       help='Include DICOM lung scans in unified dataset (slower)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples per class for balanced training')
    parser.add_argument('--use_real_labs', action='store_true', default=True,
                       help='Use real NHANES blood report data (default: True)')
    parser.add_argument('--use_synthetic_labs', action='store_true', default=False,
                       help='Use synthetic lab values instead of real NHANES data')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')

    args = parser.parse_args()

    # Build config
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        lab_type = 'synthetic' if args.use_synthetic_labs else 'nhanes'
        args.experiment_name = f"{args.model_type}_{args.fusion_method}_{args.dataset}_{lab_type}_{timestamp}"

    config = {
        'experiment_name': args.experiment_name,
        'model_type': args.model_type,
        'fusion_method': args.fusion_method,
        'use_simple_cnn': args.use_simple_cnn,
        'use_synthetic': args.dataset == 'synthetic',
        'n_synthetic_samples': args.n_samples,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': WEIGHT_DECAY,
        'seed': SEED,
        'dataset': args.dataset,
        'include_lungs': args.include_lungs,
        'max_samples_per_class': args.max_samples,
        'use_real_labs': not args.use_synthetic_labs  # Use real NHANES data unless synthetic specified
    }

    # Run experiment
    run_experiment(config)


if __name__ == '__main__':
    main()
