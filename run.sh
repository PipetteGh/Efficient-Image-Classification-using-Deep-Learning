#!/usr/bin/env bash
# run.sh
# ======
# End-to-end pipeline for CSCD618/DSCD604 Assignment 2.
# Usage: bash run.sh [train|infer|all]

set -e

MODE=${1:-all}

echo "============================================"
echo " CSCD618/DSCD604 Assignment 2 Pipeline"
echo "============================================"

# ── Install dependencies ─────────────────────────────────────────────────────
echo ""
echo "[Step 0] Installing dependencies …"
pip install -r requirements.txt -q

# ── Train ────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
    echo ""
    echo "[Step 1] Training the model …"
    python train.py
fi

# ── Inference ────────────────────────────────────────────────────────────────
if [[ "$MODE" == "infer" || "$MODE" == "all" ]]; then
    echo ""
    echo "[Step 2] Running inference with TTA …"
    python inference.py --tta --tta-steps 8
fi

echo ""
echo "============================================"
echo " Done! Submission file: submission.csv"
echo "============================================"
