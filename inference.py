"""
inference.py
============
Inference script for CSCD618/DSCD604 Assignment 2.

Loads the Large and Small checkpoints, performs Test-Time Augmentation (TTA),
averages their logits, and generates the final submission.csv.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import pandas as pd

import config as cfg
from data_loader import get_test_loader
from model import build_model
from utils import get_device, load_checkpoint, progress_bar


@torch.no_grad()
def predict_tta_ensemble(
    model_large: torch.nn.Module,
    model_small: torch.nn.Module,
    test_dir: str,
    idx_to_class: dict[int, str],
    device: torch.device,
    tta_steps: int = cfg.TTA_STEPS,
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
) -> tuple[list[str], list[str]]:
    
    model_large.eval()
    model_small.eval()

    # Get filenames
    std_loader = get_test_loader(test_dir, batch_size=batch_size, num_workers=num_workers, tta=False)
    all_fnames = []
    for _, fnames in std_loader:
        all_fnames.extend(fnames)

    num_images = len(all_fnames)
    num_classes = len(idx_to_class)
    accumulated = torch.zeros(num_images, num_classes, device=device)

    for step in range(tta_steps):
        print(f"  TTA step {step+1}/{tta_steps} …")
        use_aug = step > 0
        loader  = get_test_loader(test_dir, batch_size=batch_size, num_workers=num_workers, tta=use_aug)
        offset = 0
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            
            # Predict with both models
            out_large = model_large(images)
            out_small = model_small(images)
            
            # Average probabilities
            prob_large = F.softmax(out_large, dim=1)
            prob_small = F.softmax(out_small, dim=1)
            avg_prob = (prob_large + prob_small) / 2.0
            
            bs = avg_prob.size(0)
            accumulated[offset:offset + bs] += avg_prob
            offset += bs

    preds = accumulated.argmax(dim=1).cpu().tolist()
    return all_fnames, preds

def build_submission(fnames: list[str], pred_indices: list[int], idx_to_class: dict[int, str]) -> None:
    fname_to_pred = {fname: idx_to_class[idx] for fname, idx in zip(fnames, pred_indices)}
    if os.path.exists(cfg.SAMPLE_SUB_CSV):
        sample_df = pd.read_csv(cfg.SAMPLE_SUB_CSV)
        ordered_fnames = sample_df.iloc[:, 0].tolist()
    else:
        ordered_fnames = sorted(fnames)

    from collections import Counter
    fallback = Counter(fname_to_pred.values()).most_common(1)[0][0]

    rows = [{"Id": fname, "Prediction": fname_to_pred.get(fname, fallback)} for fname in ordered_fnames]
    df = pd.DataFrame(rows, columns=["Id", "Prediction"])
    df.to_csv(cfg.SUBMISSION_CSV, index=False)
    print(f"\n[OK] Submission saved to: {cfg.SUBMISSION_CSV}")

def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = get_device()

    ckpt_large_path = cfg.MODEL_CHECKPOINT.replace(".pth", "_large.pth")
    ckpt_small_path = cfg.MODEL_CHECKPOINT.replace(".pth", "_small.pth")

    if not (os.path.exists(ckpt_large_path) and os.path.exists(ckpt_small_path)):
        print("Error: Could not find both _large.pth and _small.pth checkpoints.")
        return

    print("=== Loading Large Model ===")
    model_large = build_model(num_classes=cfg.NUM_CLASSES, pretrained=False, model_type="large").to(device)
    ckpt_large = load_checkpoint(ckpt_large_path, model_large, device=device)

    print("=== Loading Small Model ===")
    model_small = build_model(num_classes=cfg.NUM_CLASSES, pretrained=False, model_type="small").to(device)
    ckpt_small = load_checkpoint(ckpt_small_path, model_small, device=device)

    idx_to_class = {v: k for k, v in ckpt_large["class_to_idx"].items()}

    print("\n=== Running Ensemble Inference ===")
    t0 = time.time()
    fnames, preds = predict_tta_ensemble(model_large, model_small, cfg.TEST_DIR, idx_to_class, device)
    
    print(f"\n  Inference completed in {time.time()-t0:.1f}s")
    build_submission(fnames, preds, idx_to_class)

if __name__ == "__main__":
    main()
