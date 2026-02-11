"""
Configuration file for Multimodal Disease Classification Framework
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =============================================================================
# DISEASE CLASSES
# =============================================================================
NUM_CLASSES = 4
CLASS_NAMES = ["Normal", "Tumor", "Infection", "Inflammatory"]
CLASS_MAPPING = {
    0: "Normal",
    1: "Tumor", 
    2: "Infection",
    3: "Inflammatory"
}

# =============================================================================
# LAB VALUE CONFIGURATIONS (Based on medical literature)
# =============================================================================
# Format: (mean, std) for each class
# CRP: C-Reactive Protein (mg/L) - Normal < 10
# WBC: White Blood Cell Count (×10^9/L) - Normal 4-11
# Hb: Hemoglobin (g/dL) - Normal 12-17

LAB_RANGES = {
    "Normal": {
        "CRP": (3.0, 2.0),      # Low
        "WBC": (7.5, 1.5),      # Normal
        "Hb": (14.0, 1.0)       # Normal
    },
    "Tumor": {
        "CRP": (25.0, 10.0),    # Moderate elevation
        "WBC": (9.0, 2.0),      # Slight elevation
        "Hb": (10.5, 1.5)       # Decreased (anemia)
    },
    "Infection": {
        "CRP": (80.0, 30.0),    # High elevation
        "WBC": (15.0, 4.0),     # High elevation
        "Hb": (12.0, 1.5)       # Slight decrease
    },
    "Inflammatory": {
        "CRP": (50.0, 20.0),    # High elevation
        "WBC": (9.5, 2.5),      # Normal to slight elevation
        "Hb": (13.0, 1.2)       # Normal
    }
}

# =============================================================================
# IMAGE CONFIGURATION
# =============================================================================
IMAGE_SIZE = 224
IMAGE_CHANNELS = 1  # Grayscale CT
MEAN = 0.5
STD = 0.5

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# CNN Encoder
CNN_BACKBONE = "resnet18"  # Options: resnet18, resnet34, resnet50, efficientnet
CNN_PRETRAINED = True
CNN_FEATURE_DIM = 512

# Lab MLP Encoder
LAB_INPUT_DIM = 3  # CRP, WBC, Hb
LAB_HIDDEN_DIMS = [32, 64]
LAB_FEATURE_DIM = 64

# Fusion
FUSION_METHOD = "concat"  # Options: concat, attention, gated
FUSION_DIM = CNN_FEATURE_DIM + LAB_FEATURE_DIM

# Classifier
CLASSIFIER_HIDDEN_DIM = 256
DROPOUT_RATE = 0.5

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SCHEDULER_STEP = 10
SCHEDULER_GAMMA = 0.5
EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# DATA SPLIT
# =============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# RANDOM SEED
# =============================================================================
SEED = 42

# =============================================================================
# GPU/DEVICE CONFIGURATION
# =============================================================================
import torch

# Auto-detect GPU
USE_GPU = torch.cuda.is_available()
GPU_ID = 0  # Use first GPU (RTX 3060)
DEVICE = torch.device(f"cuda:{GPU_ID}" if USE_GPU else "cpu")

# GPU Memory optimization
GPU_MEMORY_FRACTION = 0.8  # Use max 80% of GPU memory
CUDA_BENCHMARK = True  # Enable CUDA benchmarking for faster training
CUDA_DETERMINISTIC = False  # Allow non-deterministic for speed (set True if reproducibility needed)

# Batch size adjustments for GPU
GPU_BATCH_SIZE = 64  # Larger batch size when using GPU
CPU_BATCH_SIZE = 32  # Smaller batch size for CPU

# Use GPU batch size if available
if USE_GPU:
    BATCH_SIZE = GPU_BATCH_SIZE
    # Set CUDA optimization flags
    torch.backends.cudnn.benchmark = CUDA_BENCHMARK
    if CUDA_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True

print(f"[INFO] Device: {DEVICE}")
if USE_GPU:
    print(f"[INFO] GPU: {torch.cuda.get_device_name(GPU_ID)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(GPU_ID).total_memory / 1024**3:.2f} GB")
    print(f"[INFO] Batch Size (GPU optimized): {BATCH_SIZE}")

