# Multimodal Disease Classification Framework

A disease-agnostic multimodal deep learning framework that predicts diseases from CT images enhanced with minimal lab values (CRP, WBC, Hb).

## 🎯 Project Overview

This framework demonstrates that combining CT imaging with just **three key lab biomarkers** can effectively classify multiple disease categories:

| Class | Description |
|-------|-------------|
| 0 | Normal |
| 1 | Tumor |
| 2 | Infection |
| 3 | Inflammatory |

### Why This Approach?

Traditional systems require:
- One model per disease
- Heavy clinical inputs
- Not scalable

**Our system:**
- One unified model
- Multiple disease classes  
- Minimal lab data (only 3 values!)
- CT-based imaging

## 📁 Project Structure

```
disease_classifier/
├── config.py                 # Configuration settings
├── main.py                   # Main experiment runner
├── train.py                  # Training script
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Dataset classes
│   └── lab_generator.py     # Synthetic lab value generator
├── models/
│   ├── __init__.py
│   ├── cnn_encoder.py       # CNN for CT images
│   ├── lab_encoder.py       # MLP for lab values
│   └── fusion_model.py      # Multimodal fusion
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Logging utilities
│   └── metrics.py           # Evaluation metrics
├── saved_models/            # Trained model checkpoints
├── results/                 # Experiment results
└── logs/                    # Training logs
```

## 🔬 Lab Values and Disease Correlation

| Disease Type | CRP | WBC | Hb |
|-------------|-----|-----|-----|
| Normal | Low | Normal | Normal |
| Tumor | Moderate ↑ | Slight ↑ | ↓ |
| Infection | High ↑↑ | High ↑↑ | Slight ↓ |
| Inflammatory | High ↑ | Normal/↑ | Normal |

**Key insight:** Same CT patterns may overlap, but labs provide contextual disambiguation.

## 🏗️ Architecture

```
CT Image
   │
CNN Encoder (ResNet18/Simple CNN)
   │
Image Feature Vector (512-dim)
   │
   ├───────────────────────┐
   │                       │
CRP, WBC, Hb               │
   │                       │
MLP Encoder                │
   │                       │
Lab Feature Vector (64-dim)│
   └────── Fusion Layer ───┘
               │
        Fully Connected
               │
        Softmax (4 classes)
```

### Fusion Methods
- **Concat**: Simple concatenation
- **Gated**: Learned weighting of modalities
- **Attention**: Cross-attention mechanism

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm pillow
```

### 2. Run Quick Test

```bash
cd disease_classifier
python main.py --mode quick_test
```

### 3. Run Full Experiments

```bash
python main.py --mode full
```

This runs:
1. **Baseline 1**: CT-only model
2. **Baseline 2**: Lab-only model
3. **Proposed**: Fusion model (concat)
4. **Proposed**: Fusion model (gated)

### 4. Train Custom Model

```bash
python train.py --model_type fusion --fusion_method concat --epochs 50
```

**Options:**
- `--model_type`: `fusion`, `image_only`, `lab_only`
- `--fusion_method`: `concat`, `gated`, `attention`
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate

## 📊 Experiments

### Baselines
- **CT-only**: Uses only image features
- **Lab-only**: Uses only CRP, WBC, Hb

### Proposed Models
- **Fusion (concat)**: Concatenates image and lab features
- **Fusion (gated)**: Learns to weight modalities

### Metrics Computed
- Accuracy
- Precision (per-class and macro)
- Recall (per-class and macro)
- F1-Score
- ROC-AUC (multi-class)
- Confusion Matrix

## 📈 Expected Results

With synthetic data, the fusion model should outperform single-modality baselines, demonstrating the complementary value of lab values.

## 🔧 Using Real Data

To use real CT images instead of synthetic data:

1. Organize images in folders:
```
data/
├── Normal/
│   ├── img001.png
│   └── ...
├── Tumor/
│   └── ...
├── Infection/
│   └── ...
└── Inflammatory/
    └── ...
```

2. Modify the data loading:
```python
train_loader, val_loader, test_loader, normalizer = create_data_loaders(
    data_dir='path/to/data',
    use_synthetic=False
)
```

## 📝 Citation

If you use this framework, please cite:

```
@article{multimodal_disease_2024,
  title={Disease-Agnostic Multimodal Framework for CT-based Disease Classification with Minimal Lab Values},
  author={Ashutosh Ranjan},
  year={2026}
}
```

## 📄 License

MIT License
