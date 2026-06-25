# CSCD618/DSCD604 Assignment 2 – Image Classification
## Custom MobileNet-like CNN for 16-Category Object Recognition

---

## Project Structure

```
assignment2/
├── train.py            # Training script
├── inference.py        # Inference & submission generation
├── model.py            # MobileLiteNet architecture
├── data_loader.py      # Dataset, transforms, Mixup/CutMix
├── utils.py            # Metrics, checkpointing, logger
├── config.py           # All hyperparameters & paths
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── class_names.csv     # 16 class labels
├── sample_submission.csv  # Template submission file
├── train/train/        # Training images (16 subdirs)
├── test/test/          # Unlabelled test images
├── checkpoints/        # Saved model checkpoints (auto-created)
└── logs/               # Training logs (auto-created)
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (highly recommended), install PyTorch with CUDA from https://pytorch.org/get-started/locally/

Example (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Verify Dataset Structure

Ensure the following structure exists:
```
train/
  train/
    airplane/   (400 images)
    bear/       (400 images)
    ...         (16 classes, 400 images each)
test/
  test/
    test_000001.JPEG
    test_000002.JPEG
    ...         (1600 unlabelled test images)
```

---

## Running the Code

### Step 1: Train the Model

```bash
# Default training (Adam optimiser, 80 epochs)
python train.py

# With SGD + more epochs
python train.py --optimizer sgd --epochs 120

# Resume from checkpoint
python train.py --resume

# Disable Mixup/CutMix
python train.py --no-mixup
```

Training will:
- Use 80% of train data for training, 20% for validation (stratified)
- Apply advanced augmentation (flips, rotations, colour jitter, random erasing)
- Apply Mixup + CutMix augmentation (50/50 alternation)
- Use Adam + Cosine Annealing Warm Restarts scheduler
- Save best model to `checkpoints/best_model.pth`
- Apply early stopping (patience=20 epochs)
- Log training metrics to `logs/training_log.jsonl`

### Step 2: Generate Submission

```bash
# Standard inference
python inference.py

# With Test-Time Augmentation (TTA) — recommended for best accuracy
python inference.py --tta

# TTA with more steps (slower but better)
python inference.py --tta --tta-steps 10
```

This generates `submission.csv` in the required format.

---

## Model Architecture: MobileLiteNet

| Feature | Detail |
|---------|--------|
| Base design | MobileNetV2 inverted residuals |
| Attention | Squeeze-and-Excitation (SE) in each block |
| Activation | HardSwish / ReLU |
| Parameters | ~2.3 M (budget: < 4.0 M) |
| Input size | 224 × 224 RGB |
| Output | 16-class softmax |

### Architecture Block Diagram

```
Input (224×224×3)
       │
  Stem Conv (3→32, stride 2)
       │
  7 Inverted Residual Stages:
  ┌─────────────────────────────────┐
  │ Expand → Depthwise → SE → Project│
  │ + skip connection (stride=1)    │
  └─────────────────────────────────┘
       │
  Last Conv (→1280)
       │
  Global Average Pooling
       │
  Dropout (0.3)
       │
  FC Layer (→16)
       │
   Predictions
```

---

## Training Strategy

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (LR=0.001) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| Batch size | 32 |
| Epochs | 80 (+ early stopping) |
| Weight decay | 1e-4 |
| Gradient clip | max_norm=1.0 |
| Label smoothing | 0.1 |
| Mixup α | 0.2 |
| CutMix α | 1.0 |
| Dropout | 0.3 |

### Data Augmentation Pipeline

| Augmentation | Parameters |
|-------------|------------|
| RandomResizedCrop | scale=(0.7, 1.0) |
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.05 |
| RandomRotation | ±15° |
| ColorJitter | brightness/contrast/saturation=0.2, hue=0.1 |
| RandomGrayscale | p=0.05 |
| RandomErasing | p=0.25 |
| Mixup + CutMix | 50/50 alternate per batch |
| Normalisation | ImageNet stats (mean/std) |

---

## Submission Format

The output `submission.csv` follows this exact format:

```
Id,Prediction
test_000001.JPEG,airplane
test_000002.JPEG,bear
...
```

- `Id` column: exact test image filename (e.g., `test_000001.JPEG`)
- `Prediction` column: one of the 16 valid class labels from `class_names.csv`

---

## Expected Performance

| Metric | Expected Value |
|--------|---------------|
| Validation accuracy | ≥ 85% |
| Test accuracy (Kaggle) | ≥ 80–90% |
| With TTA (8 steps) | +1–3% over standard |

---

## Key Design Decisions

1. **Depthwise Separable Convolutions**: Reduces parameters by ~8–9× vs standard convolutions while maintaining representational power.
2. **Squeeze-and-Excitation (SE)**: Adds channel-wise attention at minimal parameter cost, shown to improve accuracy by 2–3%.
3. **Cosine Annealing Warm Restarts**: Prevents local minima trapping; warm restarts allow the model to explore and escape bad optima.
4. **Mixup + CutMix**: Improves generalisation on small-medium datasets by creating interpolated training samples.
5. **Label Smoothing**: Reduces overconfidence and improves calibration.
6. **Test-Time Augmentation**: Multiple random augmented passes + probability averaging for more robust predictions.

---

## Troubleshooting

**Out of memory (CUDA OOM)**:
- Reduce batch size: `python train.py --batch 16`

**Slow training (no GPU)**:
- Set `NUM_WORKERS = 0` in `config.py` if on Windows

**Windows multiprocessing errors**:
- Set `NUM_WORKERS = 0` in `config.py` or use `python train.py --workers 0`

**Checkpoint not found for inference**:
- Train the model first: `python train.py`

---

## Academic Integrity

This code was written independently as part of CSCD618/DSCD604 Assignment 2.
All design choices, implementations, and experiments are original work.
