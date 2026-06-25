"""
describe_dataset.py
===================
Quick script to describe and validate the dataset.
Does NOT require PyTorch. Uses only built-in Python.
Run: python describe_dataset.py
"""

import os
import csv
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train", "train")
TEST_DIR  = os.path.join(BASE_DIR, "test",  "test")
CLASS_CSV = os.path.join(BASE_DIR, "class_names.csv")
SAMPLE_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

IMG_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".gif", ".webp"}

def count_images(directory):
    return sum(
        1 for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )

print("=" * 55)
print("  Dataset Description – CSCD618/DSCD604 Assignment 2")
print("=" * 55)

# Classes
classes = sorted(
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
)
print(f"\nNumber of classes : {len(classes)}")
print("\nPer-class image counts (training):")
total_train = 0
for cls in classes:
    n = count_images(os.path.join(TRAIN_DIR, cls))
    total_train += n
    bar = "#" * (n // 20)
    print(f"  {cls:12s}: {n:4d}  {bar}")

print(f"\nTotal training images : {total_train}")

# Test images
test_images = sorted(os.listdir(TEST_DIR))
test_images = [f for f in test_images if os.path.splitext(f)[1].lower() in IMG_EXTS]
print(f"Total test images     : {len(test_images)}")

# Class names CSV
print(f"\nclass_names.csv exists : {os.path.exists(CLASS_CSV)}")
print(f"sample_submission.csv  : {os.path.exists(SAMPLE_CSV)}")

if os.path.exists(SAMPLE_CSV):
    with open(SAMPLE_CSV) as f:
        rows = list(csv.reader(f))
    print(f"  Rows in sample_submission.csv : {len(rows) - 1}")

print("\nOK Dataset description complete.")
print("  Run 'python train.py' to start training.\n")
