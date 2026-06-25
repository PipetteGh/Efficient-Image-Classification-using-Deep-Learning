"""
data_loader.py
==============
Dataset and DataLoader factories for CSCD618/DSCD604 Assignment 2.

Key features
------------
* Reads class labels directly from the train/ directory structure.
* Stratified train/validation split.
* Rich training augmentation pipeline.
* Lightweight test loader (optionally with TTA).
* Mixup / CutMix collate functions.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True   # handle truncated JPEGs gracefully

import config as cfg


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=cfg.AUG_CROP_SCALE,
            ratio=cfg.AUG_CROP_RATIO,
        ),
        transforms.RandomHorizontalFlip(p=cfg.AUG_HFLIP_P),
        transforms.RandomRotation(degrees=cfg.AUG_ROTATION),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(
            brightness=cfg.AUG_BRIGHTNESS,
            contrast=cfg.AUG_CONTRAST,
            saturation=cfg.AUG_SATURATION,
            hue=cfg.AUG_HUE,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20)),
    ])


def get_val_transform(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


def get_tta_transform(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    """Transform for test-time augmentation (random crop + flip) with preserved scale."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

class ImageFolderCustom(Dataset):
    """
    Lightweight replacement for torchvision.datasets.ImageFolder that
    supports an externally supplied class-to-index mapping (so validation
    and test sets share the same label encoding as training).
    """

    def __init__(
        self,
        root: str,
        class_to_idx: dict[str, int],
        transform: Optional[transforms.Compose] = None,
        extensions: Tuple[str, ...] = (".jpeg", ".jpg", ".png", ".bmp", ".webp"),
    ):
        self.root        = root
        self.transform   = transform
        self.class_to_idx = class_to_idx
        self.classes     = sorted(class_to_idx.keys())

        self.samples: list[Tuple[str, int]] = []
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(extensions):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label)
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = _load_image(path)
        if self.transform:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    """Dataset for the unlabelled test directory."""

    def __init__(
        self,
        test_dir: str,
        transform: Optional[transforms.Compose] = None,
        extensions: Tuple[str, ...] = (".jpeg", ".jpg", ".png", ".bmp"),
    ):
        self.test_dir  = test_dir
        self.transform = transform
        self.filenames = sorted(
            f for f in os.listdir(test_dir)
            if f.lower().endswith(extensions)
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fname = self.filenames[idx]
        path  = os.path.join(self.test_dir, fname)
        img   = _load_image(path)
        if self.transform:
            img = self.transform(img)
        return img, fname


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_image(path: str) -> Image.Image:
    """Open an image, converting to RGB (handles palette / RGBA / grayscale)."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARNING] Could not open {path}: {e}. Returning blank image.")
        img = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), (128, 128, 128))
    return img


def get_class_to_idx(train_dir: str = cfg.TRAIN_DIR) -> dict[str, int]:
    """Build a deterministic class → integer mapping from the train folder."""
    classes = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    return {c: i for i, c in enumerate(classes)}


def make_train_val_split(
    dataset: ImageFolderCustom,
    val_split: float = cfg.VAL_SPLIT,
    seed: int = cfg.RANDOM_SEED,
) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
    """
    Stratified split: returns (train_sampler, val_sampler).
    Each class contributes `val_split` fraction to validation.
    """
    from collections import defaultdict
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for label in sorted(label_to_indices):
        idxs = label_to_indices[label][:]
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_split)
        if n_val > 0:
            val_indices.extend(idxs[:n_val])
            train_indices.extend(idxs[n_val:])
        else:
            train_indices.extend(idxs)

    return (
        SubsetRandomSampler(train_indices),
        SubsetRandomSampler(val_indices),
    )


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ──────────────────────────────────────────────────────────────────────────────

def get_loaders(
    train_dir:   str = cfg.TRAIN_DIR,
    val_split:   float = cfg.VAL_SPLIT,
    batch_size:  int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
    seed:        int = cfg.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Returns (train_loader, val_loader, class_to_idx).
    Both loaders share the same underlying dataset object;
    only the samplers differ.
    """
    class_to_idx = get_class_to_idx(train_dir)

    # A single dataset with TRAIN transforms; val samples get val transform on-the-fly
    # via a wrapper (see _TransformSubset below).
    full_train = ImageFolderCustom(train_dir, class_to_idx,
                                   transform=get_train_transform())
    full_val   = ImageFolderCustom(train_dir, class_to_idx,
                                   transform=get_val_transform())

    train_sampler, val_sampler = make_train_val_split(full_train,
                                                      val_split, seed)

    train_loader = DataLoader(
        full_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        full_val,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, class_to_idx


def get_test_loader(
    test_dir:    str = cfg.TEST_DIR,
    batch_size:  int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
    tta:         bool = False,
) -> DataLoader:
    """Returns a DataLoader over the unlabelled test set."""
    transform = get_tta_transform() if tta else get_val_transform()
    ds = TestDataset(test_dir, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.PIN_MEMORY,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mixup / CutMix
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = cfg.MIXUP_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Returns mixed inputs, pairs of targets, and mixing coefficient λ."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = cfg.CUTMIX_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix augmentation."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size, _, H, W = x.shape
    idx = torch.randperm(batch_size, device=x.device)

    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, y, y[idx], lam


def mixup_criterion(
    criterion: torch.nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
