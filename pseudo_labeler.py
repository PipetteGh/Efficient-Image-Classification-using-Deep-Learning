import os
import torch
import shutil
import pandas as pd
from PIL import Image
import torch.nn.functional as F

import config as cfg
from model import build_model
from data_loader import get_tta_transform

print("=== Generating High-Confidence Pseudo Labels ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = build_model(num_classes=cfg.NUM_CLASSES, pretrained=False)
checkpoint = torch.load(cfg.MODEL_CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

transform = get_tta_transform(cfg.IMAGE_SIZE)

# Paths
test_dir = cfg.TEST_DIR
pseudo_dir = os.path.join(cfg.BASE_DIR, "train", "train_pseudo")

# Create pseudo-label folders
if os.path.exists(pseudo_dir):
    shutil.rmtree(pseudo_dir)
os.makedirs(pseudo_dir)
for cls in cfg.CLASS_NAMES:
    os.makedirs(os.path.join(pseudo_dir, cls), exist_ok=True)

image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".JPEG")])
high_conf_count = 0

print("Processing test images...")
with torch.no_grad():
    for fname in image_files:
        path = os.path.join(test_dir, fname)
        img = Image.open(path).convert("RGB")
        
        # We use standard TTA-like inference to be extremely certain
        probs_sum = torch.zeros(1, cfg.NUM_CLASSES).to(device)
        for _ in range(8):
            t_img = transform(img).unsqueeze(0).to(device)
            logits = model(t_img)
            probs = F.softmax(logits, dim=1)
            probs_sum += probs
        
        avg_probs = probs_sum / 8.0
        max_prob, pred_idx = torch.max(avg_probs, 1)
        
        if max_prob.item() > 0.95:  # Extremely confident threshold
            pred_class = idx_to_class[pred_idx.item()]
            dst_path = os.path.join(pseudo_dir, pred_class, fname)
            shutil.copy(path, dst_path)
            high_conf_count += 1

print(f"\nSuccessfully generated {high_conf_count} high-confidence pseudo-labels from {len(image_files)} test images.")
print(f"They have been saved to {pseudo_dir} for training!")
