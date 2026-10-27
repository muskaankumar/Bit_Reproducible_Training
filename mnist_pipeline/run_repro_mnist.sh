#!/bin/bash
# Run bit-reproducible MNIST training twice and verify hashes match.
#
# Usage:
#   ./run_repro_mnist.sh              # default: 2 runs, 3 epochs
#   ./run_repro_mnist.sh 5            # 5 epochs
#   GPUS=0,1 ./run_repro_mnist.sh     # run each on a different GPU
#
set -euo pipefail
cd "$(dirname "$0")"

EPOCHS=${1:-3}
SEED=42
SLICES=3
BATCH=100
LR=0.001

# Parse GPU list
IFS=',' read -ra GPU_LIST <<< "${GPUS:-0,0}"
GPU_A=${GPU_LIST[0]:-0}
GPU_B=${GPU_LIST[1]:-$GPU_A}

echo "=== Bit-Reproducible MNIST Verification ==="
echo "Epochs: $EPOCHS, Seed: $SEED, Slices: $SLICES"
echo "GPU A: $GPU_A, GPU B: $GPU_B"
echo

# Build extension if needed
if ! python -c "import repro_gemm_ext" 2>/dev/null; then
    echo "Building repro_gemm extension..."
    pip install -e . 2>&1 | tail -3
    echo
fi

mkdir -p results

echo "--- Run 1 (GPU $GPU_A) ---"
CUDA_VISIBLE_DEVICES=$GPU_A python train_mnist.py \
    --epochs "$EPOCHS" --seed "$SEED" --slices "$SLICES" \
    --batch-size "$BATCH" --lr "$LR" \
    --save-hashes results/run1_hashes.txt

echo
echo "--- Run 2 (GPU $GPU_B) ---"
CUDA_VISIBLE_DEVICES=$GPU_B python train_mnist.py \
    --epochs "$EPOCHS" --seed "$SEED" --slices "$SLICES" \
    --batch-size "$BATCH" --lr "$LR" \
    --save-hashes results/run2_hashes.txt

echo
echo "=== Comparing hashes ==="
if diff results/run1_hashes.txt results/run2_hashes.txt; then
    echo "PASS: All epoch hashes are bit-identical across the two runs."
else
    echo "FAIL: Hashes differ! Training is NOT bit-reproducible."
    exit 1
fi
