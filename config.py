"""
config.py
=========
Central configuration for CSCD618/DSCD604 Assignment 2 - Image Classification.
All hyperparameters and paths are defined here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR      = os.path.join(BASE_DIR, "train", "train")
TEST_DIR       = os.path.join(BASE_DIR, "test",  "test")
CLASS_CSV      = os.path.join(BASE_DIR, "class_names.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
SUBMISSION_CSV = os.path.join(BASE_DIR, "submission.csv")
LOG_DIR        = os.path.join(BASE_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Classes ──────────────────────────────────────────────────────────────────
# The 16 categories, ordered alphabetically with exact casing from the dataset.
CLASS_NAMES = [
    "Keyboard", "Knife",
    "airplane", "bear", "bicycle", "bird", "boat",
    "bottle", "car", "cat", "chair", "clock",
    "dog", "elephant", "oven", "truck",
]
NUM_CLASSES = len(CLASS_NAMES)            # 16

# ─── Image Settings ───────────────────────────────────────────────────────────
IMAGE_SIZE  = 384      # INCREASED for max detail
MEAN        = [0.485, 0.456, 0.406]      # ImageNet mean
STD         = [0.229, 0.224, 0.225]      # ImageNet std

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE       = 64       # Single GPU RTX 3060 12GB
NUM_EPOCHS       = 60
VAL_SPLIT        = 0.15      # 15% validation split for local evaluation
RANDOM_SEED      = 42
NUM_WORKERS      = 4        # GPU-enabled, use multiple workers
PIN_MEMORY       = True     # enable pin_memory for GPU

# ─── Optimiser ────────────────────────────────────────────────────────────────
OPTIMIZER        = "sgd"    # SGD generalizes better than Adam for fine-tuning
LEARNING_RATE    = 0.01     # initial LR for SGD
MOMENTUM         = 0.9      # SGD momentum
WEIGHT_DECAY     = 1e-3     # STRONG regularization to prevent overfitting
GRADIENT_CLIP    = 1.0      # max-norm for gradient clipping

# ─── Scheduler ────────────────────────────────────────────────────────────────
SCHEDULER        = "cosine"  # cosine annealing warm restarts
T0               = 20        # longer initial restart period
T_MULT           = 2         # period multiplier

# ─── Regularisation (Anti-Overfitting) ───────────────────────────────────────
LABEL_SMOOTHING  = 0.1
MIXUP_ALPHA      = 0.2       # Soft Mixup for MobileNetV3-Large
CUTMIX_ALPHA     = 0.0       # Disabled for tiny model
DROPOUT          = 0.2       # Reduced to let model properly fit

# ─── Early Stopping ───────────────────────────────────────────────────────────
PATIENCE         = 40        # disabled early stopping

# ─── Data Augmentation ────────────────────────────────────────────────────────
AUG_HFLIP_P     = 0.5
AUG_ROTATION    = 15         # degrees
AUG_BRIGHTNESS  = 0.3
AUG_CONTRAST    = 0.3
AUG_SATURATION  = 0.3
AUG_HUE         = 0.1
AUG_CROP_SCALE  = (0.8, 1.0)   # tighter crop
AUG_CROP_RATIO  = (0.85, 1.15) # tighter ratio

# ─── Two-Phase Fine-tuning ───────────────────────────────────────────────────
FREEZE_EPOCHS    = 5         # Phase 1: train only classifier for this many epochs
FREEZE_LR        = 0.01      # higher LR for Phase 1 (only classifier weights)
FINETUNE_LR      = 0.00005   # LOWER LR for Phase 2 to prevent Catastrophic Forgetting

# ─── Test-Time Augmentation ───────────────────────────────────────────────────
TTA_STEPS        = 16        # INCREASED for more robust predictions

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")
PRETRAINED       = True      # use ImageNet pretrained weights
