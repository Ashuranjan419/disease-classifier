"""
Main Runner Script for Multimodal Disease Classification Experiments

This script runs all experiments:
1. Baseline: CT-only model
2. Baseline: Lab-only model  
3. Proposed: CT + Labs Fusion Model (concat)
4. Proposed: CT + Labs Fusion Model (gated)
5. Ablation: Remove each lab value one at a time
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SEED, RESULTS_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
from train import run_experiment, set_seed


def run_all_experiments():
    """Run all experiments for the paper."""
    
    print("=" * 70)
    print("MULTIMODAL DISEASE CLASSIFICATION - FULL EXPERIMENT SUITE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results_summary = {}
    
    # =========================================================================
    # Experiment 1: Image-Only Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Image-Only Baseline (CT images without lab values)")
    print("=" * 70)
    
    config_image_only = {
        'experiment_name': 'baseline_image_only',
        'model_type': 'image_only',
        'fusion_method': 'concat',
        'use_simple_cnn': True,
        'use_synthetic': True,
        'n_synthetic_samples': 200,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'seed': SEED
    }
    
    _, metrics, _ = run_experiment(config_image_only)
    results_summary['image_only'] = metrics
    
    # =========================================================================
    # Experiment 2: Lab-Only Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Lab-Only Baseline (CRP, WBC, Hb without images)")
    print("=" * 70)
    
    config_lab_only = {
        'experiment_name': 'baseline_lab_only',
        'model_type': 'lab_only',
        'fusion_method': 'concat',
        'use_simple_cnn': True,
        'use_synthetic': True,
        'n_synthetic_samples': 200,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'seed': SEED
    }
    
    _, metrics, _ = run_experiment(config_lab_only)
    results_summary['lab_only'] = metrics
    
    # =========================================================================
    # Experiment 3: Fusion Model (Concatenation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Multimodal Fusion (Concatenation)")
    print("=" * 70)
    
    config_fusion_concat = {
        'experiment_name': 'fusion_concat',
        'model_type': 'fusion',
        'fusion_method': 'concat',
        'use_simple_cnn': True,
        'use_synthetic': True,
        'n_synthetic_samples': 200,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'seed': SEED
    }
    
    _, metrics, _ = run_experiment(config_fusion_concat)
    results_summary['fusion_concat'] = metrics
    
    # =========================================================================
    # Experiment 4: Fusion Model (Gated)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Multimodal Fusion (Gated)")
    print("=" * 70)
    
    config_fusion_gated = {
        'experiment_name': 'fusion_gated',
        'model_type': 'fusion',
        'fusion_method': 'gated',
        'use_simple_cnn': True,
        'use_synthetic': True,
        'n_synthetic_samples': 200,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'seed': SEED
    }
    
    _, metrics, _ = run_experiment(config_fusion_gated)
    results_summary['fusion_gated'] = metrics
    
    # =========================================================================
    # Print Summary Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1-Score"
    ))
    print("-" * 60)
    
    for model_name, metrics in results_summary.items():
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            model_name,
            metrics['accuracy'],
            metrics['precision_macro'],
            metrics['recall_macro'],
            metrics['f1_macro']
        ))
    
    # Save summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, 'experiments_summary.json')
    
    # Convert numpy types to Python types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    with open(summary_path, 'w') as f:
        json.dump(convert_to_serializable(results_summary), f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    # =========================================================================
    # Key Findings
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Find best model
    best_model = max(results_summary.keys(), key=lambda x: results_summary[x]['accuracy'])
    best_acc = results_summary[best_model]['accuracy']
    
    print(f"\n✓ Best performing model: {best_model} (Accuracy: {best_acc:.4f})")
    
    # Compare fusion vs baselines
    if 'fusion_concat' in results_summary and 'image_only' in results_summary:
        improvement = results_summary['fusion_concat']['accuracy'] - results_summary['image_only']['accuracy']
        print(f"✓ Fusion vs Image-only improvement: {improvement*100:+.2f}%")
    
    if 'fusion_concat' in results_summary and 'lab_only' in results_summary:
        improvement = results_summary['fusion_concat']['accuracy'] - results_summary['lab_only']['accuracy']
        print(f"✓ Fusion vs Lab-only improvement: {improvement*100:+.2f}%")
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    
    return results_summary


def run_ablation_study():
    """
    Run ablation study: remove one lab value at a time.
    
    This demonstrates the importance of each lab marker.
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY: Lab Value Importance")
    print("=" * 70)
    
    # Note: This would require modifying the data pipeline to mask lab values
    # For now, we'll describe what would be done
    
    print("""
    Ablation experiments to run:
    1. Full model (CRP + WBC + Hb)
    2. Remove CRP (WBC + Hb only)
    3. Remove WBC (CRP + Hb only)  
    4. Remove Hb (CRP + WBC only)
    
    This shows which lab marker contributes most to classification.
    
    Expected findings based on medical knowledge:
    - CRP: Most important for Infection vs Inflammatory distinction
    - WBC: Important for Infection detection
    - Hb: Important for Tumor detection (anemia)
    """)
    
    # To implement: Modify LabEncoder to accept a mask parameter
    # and set masked values to 0 (after normalization)


def quick_test():
    """Quick test with minimal epochs to verify everything works."""
    print("\n" + "=" * 70)
    print("QUICK TEST - Verifying Pipeline")
    print("=" * 70)
    
    config = {
        'experiment_name': 'quick_test',
        'model_type': 'fusion',
        'fusion_method': 'concat',
        'use_simple_cnn': True,
        'use_synthetic': True,
        'n_synthetic_samples': 50,  # Small dataset
        'num_epochs': 3,  # Few epochs
        'batch_size': 16,
        'learning_rate': LEARNING_RATE,
        'seed': SEED
    }
    
    model, metrics, history = run_experiment(config)
    
    print("\n✓ Quick test completed successfully!")
    print(f"  Final accuracy: {metrics['accuracy']:.4f}")
    
    return model, metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Multimodal Classification Experiments')
    parser.add_argument('--mode', type=str, default='quick_test',
                       choices=['quick_test', 'full', 'ablation'],
                       help='Experiment mode')
    
    args = parser.parse_args()
    
    if args.mode == 'quick_test':
        quick_test()
    elif args.mode == 'full':
        run_all_experiments()
    elif args.mode == 'ablation':
        run_ablation_study()
