"""
utils.py
========
Utility helpers: logging, metrics, early stopping, checkpoint I/O.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def top1_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of correct top-1 predictions (0–1 range)."""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation accuracy stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        """Returns True when training should stop."""
        if val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    scheduler,
    epoch:        int,
    val_acc:      float,
    class_to_idx: dict[str, int],
    path:         str,
) -> None:
    """Save model + training state to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "val_acc":      val_acc,
        "class_to_idx": class_to_idx,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "sched_state":  scheduler.state_dict() if scheduler else None,
    }, path)


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler  = None,
    device:    torch.device = torch.device("cpu"),
) -> dict:
    """Load checkpoint and return the metadata dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler and ckpt.get("sched_state"):
        scheduler.load_state_dict(ckpt["sched_state"])
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

class Logger:
    """Minimal JSON-line logger that writes to file and prints to console."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._file = open(path, "a", encoding="utf-8")

    def log(self, data: dict) -> None:
        data.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        line = json.dumps(data)
        self._file.write(line + "\n")
        self._file.flush()
        # Pretty console print
        parts = [f"{k}={v}" for k, v in data.items() if k != "ts"]
        print("  ".join(parts))

    def close(self) -> None:
        self._file.close()

    def __del__(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[device] CPU")
    return dev


# ──────────────────────────────────────────────────────────────────────────────
# Progress bar (simple, no external library needed)
# ──────────────────────────────────────────────────────────────────────────────

def progress_bar(current: int, total: int, width: int = 40, **kwargs) -> None:
    filled = int(width * current / total)
    bar    = "=" * filled + "-" * (width - filled)
    info   = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in kwargs.items())
    end    = "\n" if current >= total else "\r"
    print(f"\r  [{bar}] {current}/{total}  {info}", end=end, flush=True)
