#!/bin/bash
# Download pretrained LSS (Lift-Splat-Shoot) checkpoint
# Usage: bash scripts/download_model.sh [checkpoints_dir]
set -e

CKPT_DIR="${1:-/Users/enron/Documents/calvis/checkpoints}"
mkdir -p "$CKPT_DIR"

LSS_FILE="$CKPT_DIR/lss_model.pt"
GDRIVE_ID="1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ"

echo "=== LSS Model Download Script ==="

if [ -f "$LSS_FILE" ]; then
    SIZE=$(stat -f%z "$LSS_FILE" 2>/dev/null || stat -c%s "$LSS_FILE" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 50000000 ]; then
        echo "Checkpoint already exists: $LSS_FILE ($(echo "scale=1; $SIZE/1048576" | bc)MB)"
        exit 0
    fi
fi

if command -v gdown &>/dev/null; then
    echo "Downloading via gdown..."
    gdown "$GDRIVE_ID" -O "$LSS_FILE"
else
    echo "Install gdown first: pip install gdown"
    exit 1
fi

echo "Done! Checkpoint: $LSS_FILE"
