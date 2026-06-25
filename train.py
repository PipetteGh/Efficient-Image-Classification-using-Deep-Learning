"""
train.py
========
Training script for CSCD618/DSCD604 Assignment 2 – Image Classification.

Features:
- Pretrained MobileNetV2 with two-phase fine-tuning
- Multi-GPU Support via DataParallel
- Mixup & Cutmix Regularization (to prevent overfitting)

Usage
-----
  python train.py                    # train with defaults in config.py
  python train.py --epochs 100       # override epochs
  python train.py --resume           # resume from last checkpoint
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import config as cfg
from data_loader import (
    get_loaders,
    mixup_criterion,
    mixup_data,
    cutmix_data,
)
from model import build_model, count_parameters, count_all_parameters
from utils import (
    AverageMeter,
    EarlyStopping,
    Logger,
    get_device,
    load_checkpoint,
    progress_bar,
    save_checkpoint,
    top1_accuracy,
)


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    scheduler  = None,
    use_mixup_cutmix: bool = True,
    mixup_prob: float = 0.5,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter  = AverageMeter("acc")

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        lam = 1.0
        labels_a = labels
        labels_b = labels
        if use_mixup_cutmix and random.random() < mixup_prob:
            # Randomly choose between Mixup and Cutmix
            if cfg.CUTMIX_ALPHA > 0 and random.random() < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels)
            elif cfg.MIXUP_ALPHA > 0:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
            labels = labels_a

        optimizer.zero_grad()
        outputs = model(images)

        if lam < 1.0:
            loss = mixup_criterion(criterion, outputs, labels, labels_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
        optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch + i / len(loader))

        acc = top1_accuracy(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc,         images.size(0))

        progress_bar(i + 1, len(loader),
                     loss=loss_meter.avg, acc=acc_meter.avg)

    return loss_meter.avg, acc_meter.avg


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter("loss")
    acc_meter  = AverageMeter("acc")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc  = top1_accuracy(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc,          images.size(0))

    return loss_meter.avg, acc_meter.avg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 classifier")
    parser.add_argument("--epochs",    type=int,   default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch",     type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--no-mixup",  action="store_true",
                        help="Disable Mixup/CutMix augmentation")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume training from best_model.pth")
    parser.add_argument("--workers",   type=int,   default=cfg.NUM_WORKERS)
    parser.add_argument("--freeze-epochs", type=int, default=cfg.FREEZE_EPOCHS,
                        help="Number of epochs to train with frozen backbone")
    parser.add_argument("--model-type", type=str, default="large", choices=["large", "small"])
    return parser.parse_args()


def main() -> None:
    # Force single GPU (RTX 3060 is GPU 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args   = parse_args()
    set_seed(cfg.RANDOM_SEED)
    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    train_loader, val_loader, class_to_idx = get_loaders(
        batch_size  = args.batch,
        num_workers = args.workers,
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"      Classes  : {list(class_to_idx.keys())}")
    print(f"      Train batches : {len(train_loader)}")
    print(f"      Val   batches : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Building model …")
    model = build_model(num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT,
                        pretrained=cfg.PRETRAINED, model_type=args.model_type)
    
    total_params = count_all_parameters(model)
    print(f"      Total parameters: {total_params:,}  (budget: < 4,000,000)")
    assert total_params < 4_000_000, "Parameter budget exceeded!"

    # ── Phase 1: Freeze backbone ──────────────────────────────────────────────
    freeze_epochs = args.freeze_epochs
    if cfg.PRETRAINED and freeze_epochs > 0:
        print(f"\n[*] Phase 1: Freezing backbone for {freeze_epochs} epochs")
        model.freeze_backbone()
        trainable = count_parameters(model)
        print(f"      Trainable params (classifier only): {trainable:,}")

    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    # ── Optimizer (initially for frozen backbone) ─────────────────────────────
    print("\n[3/5] Configuring optimiser …")
    lr = cfg.FREEZE_LR if (cfg.PRETRAINED and freeze_epochs > 0) else args.lr
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.T0, T_mult=cfg.T_MULT, eta_min=1e-6
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = cfg.MODEL_CHECKPOINT.replace(".pth", f"_{args.model_type}.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"\n[*] Resuming from {checkpoint_path}")
        # Unfreeze underlying model before loading
        base_model = model
        base_model.unfreeze_backbone()   
        ckpt = load_checkpoint(checkpoint_path, model,
                               optimizer, scheduler, device)
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt["val_acc"]
        freeze_epochs = 0  # skip freeze phase when resuming
        print(f"    Resumed at epoch {start_epoch}, best val acc = {best_val_acc:.4f}")

    # ── Logger ────────────────────────────────────────────────────────────────
    log_path = os.path.join(cfg.LOG_DIR, "training_log.jsonl")
    logger   = Logger(log_path)
    early_stopper = EarlyStopping(patience=cfg.PATIENCE)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {args.epochs} epochs …\n")
    for epoch in range(start_epoch, args.epochs):

        # ── Phase transition: Unfreeze backbone ──────────────────────────────
        if cfg.PRETRAINED and epoch == freeze_epochs and freeze_epochs > 0:
            print(f"\n[*] Phase 2: Unfreezing backbone (epoch {epoch+1})")
            base_model = model
            base_model.unfreeze_backbone()
            trainable = count_parameters(model)
            print(f"      Trainable params (all): {trainable:,}")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.FINETUNE_LR,
                weight_decay=cfg.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T0, T_mult=cfg.T_MULT, eta_min=1e-6
            )

        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        phase = "Phase1-frozen" if epoch < freeze_epochs else "Phase2-finetune"
        print(f"Epoch [{epoch+1:03d}/{args.epochs}]  lr={current_lr:.6f}  ({phase})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch,
            scheduler=scheduler,
            use_mixup_cutmix=not args.no_mixup,
            mixup_prob=0.5, # increased probability
        )
        if len(val_loader) > 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0
            
        elapsed = time.time() - t0

        logger.log({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
            "lr":         round(current_lr, 8),
            "elapsed_s":  round(elapsed,    1),
            "phase":      phase,
        })

        # Save best checkpoint (or every epoch if no validation set)
        if len(val_loader) == 0 or val_acc > best_val_acc:
            best_val_acc = val_acc if len(val_loader) > 0 else train_acc
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_val_acc, class_to_idx, checkpoint_path)
            print(f"  [OK] New checkpoint saved")

        if len(val_loader) > 0 and early_stopper.step(val_acc):
            print(f"\n[*] Early stopping triggered at epoch {epoch+1}.")
            break

    logger.close()
    print(f"\n[5/5] Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"      Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
