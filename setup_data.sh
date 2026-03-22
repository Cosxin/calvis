#!/usr/bin/env bash
# setup_data.sh — Download nuScenes mini, checkpoints, and install dependencies
# Run from the repo root: bash setup_data.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Calvis Setup ==="
echo "Working directory: $SCRIPT_DIR"

# ─── 1. Python dependencies ─────────────────────────────────────────────────
echo ""
echo "--- [1/3] Checking Python dependencies ---"

check_pkg() { python3 -c "import $1" 2>/dev/null; }

if ! check_pkg torch; then
    echo "ERROR: PyTorch not found. Install PyTorch first:"
    echo "  pip install torch torchvision torchaudio"
    exit 1
fi

# Detect CUDA version for mmcv
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda or '')" 2>/dev/null)
TORCH_VER=$(python3 -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))" 2>/dev/null)
echo "PyTorch $TORCH_VER, CUDA $CUDA_VER"

if ! check_pkg mmcv; then
    echo "Installing mmcv..."
    if [ -n "$CUDA_VER" ]; then
        CUDA_SHORT=$(echo "$CUDA_VER" | tr -d '.')
        pip install mmcv==2.1.0 -f "https://download.openmmlab.com/mmcv/dist/cu${CUDA_SHORT}/torch${TORCH_VER}/index.html"
    else
        pip install mmcv==2.1.0
    fi
fi

PKGS="mmengine==0.10.7 mmsegmentation==1.2.2 mmdet==3.3.0 mmdet3d==1.4.0"
PKGS="$PKGS nuscenes-devkit pyquaternion fastapi uvicorn pillow"
echo "Installing remaining packages..."
pip install --ignore-installed blinker $PKGS 2>&1 | tail -3

echo "Deps OK."

# ─── 2. nuScenes mini dataset ───────────────────────────────────────────────
echo ""
echo "--- [2/3] nuScenes mini dataset ---"

DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/v1.0-mini" ] && [ -d "$DATA_DIR/samples" ]; then
    echo "nuScenes mini already present."
else
    echo "Downloading nuScenes mini (~400MB)..."
    cd "$DATA_DIR"
    wget -q --show-progress "https://www.nuscenes.org/data/v1.0-mini.tgz" -O v1.0-mini.tgz \
        || curl -L -o v1.0-mini.tgz "https://www.nuscenes.org/data/v1.0-mini.tgz"
    echo "Extracting..."
    tar xzf v1.0-mini.tgz
    rm -f v1.0-mini.tgz
    echo "nuScenes mini ready."
    cd "$SCRIPT_DIR"
fi

# ─── 3. Model checkpoints ───────────────────────────────────────────────────
echo ""
echo "--- [3/3] Model checkpoints ---"

CKPT_DIR="$SCRIPT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# --- TPVFormer (tpv04_occupancy, ~241MB) ---
TPV_CKPT="$CKPT_DIR/tpv04_occupancy_v2.pth"
if [ -f "$TPV_CKPT" ] && [ "$(stat -f%z "$TPV_CKPT" 2>/dev/null || stat -c%s "$TPV_CKPT" 2>/dev/null)" -gt 1000000 ]; then
    echo "TPVFormer checkpoint OK."
else
    echo "Downloading TPVFormer checkpoint (~241MB)..."
    # Official: https://github.com/wzzheng/TPVFormer — Google Drive
    # Using the direct link from the TPVFormer repo
    GDRIVE_ID="1OMPv9u0KH2sHvKv6w4e9iGEP-goAF9NH"
    wget -q --show-progress --no-check-certificate \
        "https://docs.google.com/uc?export=download&id=${GDRIVE_ID}&confirm=t" \
        -O "$TPV_CKPT" 2>/dev/null \
    || python3 -c "
import gdown
gdown.download('https://drive.google.com/uc?id=${GDRIVE_ID}', '$TPV_CKPT', quiet=False)
" 2>/dev/null \
    || echo "WARNING: Could not auto-download TPVFormer checkpoint."
    echo "  Download manually from: https://drive.google.com/file/d/${GDRIVE_ID}"
    echo "  Place at: $TPV_CKPT"
fi

# --- GaussianFormer (NonEmpty, 25600 Gaussians, ~218MB) ---
GF_CKPT="$CKPT_DIR/gaussianformer.pth"
if [ -f "$GF_CKPT" ] && [ "$(stat -f%z "$GF_CKPT" 2>/dev/null || stat -c%s "$GF_CKPT" 2>/dev/null)" -gt 1000000 ]; then
    echo "GaussianFormer checkpoint OK."
else
    echo "Downloading GaussianFormer checkpoint (~218MB)..."
    wget -q --show-progress \
        "https://cloud.tsinghua.edu.cn/f/d1766fff8ad74756920b/?dl=1" \
        -O "$GF_CKPT" 2>/dev/null \
    || curl -L -o "$GF_CKPT" \
        "https://cloud.tsinghua.edu.cn/f/d1766fff8ad74756920b/?dl=1"
    echo "GaussianFormer checkpoint downloaded."
fi

# --- LSS (Lift-Splat-Shoot, ~55MB) ---
LSS_CKPT="$CKPT_DIR/lss_model.pt"
if [ -f "$LSS_CKPT" ] && [ "$(stat -f%z "$LSS_CKPT" 2>/dev/null || stat -c%s "$LSS_CKPT" 2>/dev/null)" -gt 1000000 ]; then
    echo "LSS checkpoint OK."
else
    echo "WARNING: LSS checkpoint not found at $LSS_CKPT"
    echo "  The LSS model requires manual download."
fi

# --- Verify ---
echo ""
echo "=== Verification ==="
for f in "$TPV_CKPT" "$GF_CKPT" "$LSS_CKPT"; do
    if [ -f "$f" ]; then
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        echo "  ✓ $(basename "$f") ($SIZE)"
    else
        echo "  ✗ $(basename "$f") MISSING"
    fi
done

echo ""
echo "=== Setup complete ==="
echo "Start the app:  python3 app.py"
echo "Open browser:   http://localhost:7860"
