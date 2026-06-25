"""
inference.py
============
Inference script for CSCD618/DSCD604 Assignment 2.

Loads the best model checkpoint and generates predictions for all test images.
Supports Test-Time Augmentation (TTA) for improved accuracy.

Usage
-----
  python inference.py                    # standard inference
  python inference.py --tta             # with TTA (recommended)
  python inference.py --tta --tta-steps 10   # more TTA steps
  python inference.py --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import config as cfg
from data_loader import get_test_loader, get_val_transform, get_tta_transform, TestDataset
from model import build_model
from utils import get_device, load_checkpoint, progress_bar


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_standard(
    model:    torch.nn.Module,
    loader:   torch.utils.data.DataLoader,
    device:   torch.device,
) -> tuple[list[str], list[str]]:
    """Single-pass inference. Returns (filenames, predicted_labels)."""
    model.eval()
    all_fnames:      list[str] = []
    all_predictions: list[str] = []

    total = len(loader)
    for i, (images, fnames) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().tolist()
        all_fnames.extend(fnames)
        all_predictions.extend(preds)
        progress_bar(i + 1, total, batch=i + 1)

    return all_fnames, all_predictions


@torch.no_grad()
def predict_tta(
    model:     torch.nn.Module,
    test_dir:  str,
    idx_to_class: dict[int, str],
    device:    torch.device,
    tta_steps: int = cfg.TTA_STEPS,
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
) -> tuple[list[str], list[str]]:
    """
    Test-Time Augmentation: run `tta_steps` random augmented passes and
    average the softmax probabilities before taking argmax.
    """
    model.eval()

    # First pass: standard transform to collect filenames
    std_loader = get_test_loader(test_dir, batch_size=batch_size,
                                 num_workers=num_workers, tta=False)
    all_fnames = []
    for _, fnames in std_loader:
        all_fnames.extend(fnames)

    num_images = len(all_fnames)
    num_classes = len(idx_to_class)
    accumulated = torch.zeros(num_images, num_classes, device=device)

    for step in range(tta_steps):
        print(f"  TTA step {step+1}/{tta_steps} …")
        use_aug = step > 0   # step 0 = standard, rest = random augmented
        loader  = get_test_loader(test_dir, batch_size=batch_size,
                                  num_workers=num_workers, tta=use_aug)
        offset = 0
        for images, _ in loader:
            images  = images.to(device, non_blocking=True)
            outputs = model(images)
            probs   = F.softmax(outputs, dim=1)
            bs      = probs.size(0)
            accumulated[offset:offset + bs] += probs
            offset += bs

    preds = accumulated.argmax(dim=1).cpu().tolist()
    return all_fnames, preds


# ──────────────────────────────────────────────────────────────────────────────
# CSV generation
# ──────────────────────────────────────────────────────────────────────────────

def build_submission(
    fnames:        list[str],
    pred_indices:  list[int],
    idx_to_class:  dict[int, str],
    sample_sub_csv: str = cfg.SAMPLE_SUB_CSV,
    output_csv:    str  = cfg.SUBMISSION_CSV,
) -> pd.DataFrame:
    """
    Build the submission DataFrame with exactly the filenames present in
    sample_submission.csv (same order), filling any missing predictions
    with the most common class as a fallback.
    """
    # Map predictions to class names
    fname_to_pred = {
        fname: idx_to_class[idx]
        for fname, idx in zip(fnames, pred_indices)
    }

    # Load the canonical filename order
    if os.path.exists(sample_sub_csv):
        sample_df = pd.read_csv(sample_sub_csv)
        ordered_fnames = sample_df.iloc[:, 0].tolist()
    else:
        # Fall back to sorted order if no sample_submission.csv
        ordered_fnames = sorted(fnames)

    # Fallback class (most common prediction)
    from collections import Counter
    fallback = Counter(fname_to_pred.values()).most_common(1)[0][0]

    rows = [
        {"Id": fname, "Prediction": fname_to_pred.get(fname, fallback)}
        for fname in ordered_fnames
    ]
    df = pd.DataFrame(rows, columns=["Id", "Prediction"])
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Submission saved to: {output_csv}")
    print(f"  Total rows : {len(df)}")
    print(f"  Sample:\n{df.head(5).to_string(index=False)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Assignment 2")
    parser.add_argument("--checkpoint",  default=cfg.MODEL_CHECKPOINT)
    parser.add_argument("--test-dir",    default=cfg.TEST_DIR)
    parser.add_argument("--output",      default=cfg.SUBMISSION_CSV)
    parser.add_argument("--tta",         action="store_true",
                        help="Use Test-Time Augmentation")
    parser.add_argument("--tta-steps",   type=int, default=cfg.TTA_STEPS)
    parser.add_argument("--batch",       type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--workers",     type=int, default=cfg.NUM_WORKERS)
    return parser.parse_args()


def main() -> None:
    # Force single GPU (RTX 3060 is GPU 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args   = parse_args()
    device = get_device()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\n[1/3] Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run train.py first."
        )

    model = build_model(num_classes=cfg.NUM_CLASSES, dropout=0.0)
    model = model.to(device)

    ckpt  = load_checkpoint(args.checkpoint, model, device=device)

    class_to_idx  = ckpt.get("class_to_idx", {})
    idx_to_class  = {v: k for k, v in class_to_idx.items()}
    best_val_acc  = ckpt.get("val_acc", "N/A")
    epoch_trained = ckpt.get("epoch",   "N/A")
    print(f"  Model trained for {epoch_trained} epochs, best val acc = {best_val_acc}")
    print(f"  Classes: {list(class_to_idx.keys())}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\n[2/3] Running inference on {args.test_dir} …")
    t0 = time.time()

    if args.tta:
        print(f"  Mode: TTA ({args.tta_steps} steps)")
        fnames, pred_indices = predict_tta(
            model, args.test_dir, idx_to_class, device,
            tta_steps   = args.tta_steps,
            batch_size  = args.batch,
            num_workers = args.workers,
        )
    else:
        print("  Mode: Standard (single pass)")
        loader = get_test_loader(args.test_dir, batch_size=args.batch,
                                 num_workers=args.workers, tta=False)
        fnames, pred_indices = predict_standard(model, loader, device)

    elapsed = time.time() - t0
    print(f"\n  Inference completed in {elapsed:.1f}s")

    # ── Build & save CSV ──────────────────────────────────────────────────────
    print("\n[3/3] Building submission CSV …")
    df = build_submission(
        fnames        = fnames,
        pred_indices  = pred_indices,
        idx_to_class  = idx_to_class,
        sample_sub_csv = cfg.SAMPLE_SUB_CSV,
        output_csv     = args.output,
    )

    # Class distribution summary
    print("\nPrediction distribution:")
    dist = df["Prediction"].value_counts().to_dict()
    for cls_name, count in sorted(dist.items()):
        bar = "#" * (count // 5)
        print(f"  {cls_name:12s}: {count:4d}  {bar}")


if __name__ == "__main__":
    main()

