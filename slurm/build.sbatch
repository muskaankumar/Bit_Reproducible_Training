#!/bin/bash
#SBATCH --account=csci_ga_3033_025-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --job-name=repro-build
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

# ============================================================================
# BUILD repro_torch CUDA kernels for the detected GPU architecture.
#
# Usage:
#   sbatch --partition=g2-standard-12   slurm/build.sbatch   # L4 (sm_89)
#   sbatch --partition=c12m85-a100-1    slurm/build.sbatch   # A100 (sm_80)
#
# BUDGET NOTE: This uses GPU time! Build takes ~10-15 min.
#   L4 partition is cheaper to use for builds.
# ============================================================================

set -euo pipefail
mkdir -p logs

OVERLAY_DIR=$SCRATCH/repro_training
OVERLAY_FILE=$OVERLAY_DIR/overlay-15GB-500K.ext3

# Try both image locations
SINGULARITY_IMAGE=/share/apps/images/cuda12.4.1-cudnn9.1.0-devel-ubuntu22.04.4.sif
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.4.1-cudnn9.1.0-devel-ubuntu22.04.4.sif
fi

# Detect GPU type
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr ' ' '_')
echo "Building for GPU: $GPU_NAME on $(hostname)"

# Map GPU name to CUDA arch
case "$GPU_NAME" in
    *A100*)       ARCH=sm_80 ;;
    *L4*)         ARCH=sm_89 ;;
    *V100*)       ARCH=sm_70 ;;
    *RTX_8000*)   ARCH=sm_75 ;;
    *RTX_4070*)   ARCH=sm_89 ;;
    *RTX_2080*)   ARCH=sm_75 ;;
    *)            ARCH=native; echo "WARNING: Unknown GPU '$GPU_NAME', using native arch" ;;
esac

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO_DIR=$(cd "$PROJECT_DIR/../Bit_Reproducible_Training" 2>/dev/null && pwd || echo "")
BUILD_DIR=$SCRATCH/repro_training/builds/$ARCH

echo "ARCH=$ARCH BUILD_DIR=$BUILD_DIR"

singularity exec --nv \
    --overlay $OVERLAY_FILE:rw \
    $SINGULARITY_IMAGE /bin/bash <<BUILD_EOF
set -euo pipefail
source /ext3/miniconda3/bin/activate repro

# Build the base repro_torch (from the original repo) if available
if [ -n "$REPO_DIR" ] && [ -d "$REPO_DIR/repro_torch" ]; then
    echo "Building base repro_torch from $REPO_DIR..."
    cd $REPO_DIR/repro_torch
    mkdir -p $BUILD_DIR
    REPRO_ARCH=$ARCH python setup.py build_ext --inplace
    cp -r repro_torch/_backend* $BUILD_DIR/ 2>/dev/null || true
    cp -r repro_torch/*.py $BUILD_DIR/ 2>/dev/null || true
fi

# Build the new GPT-2 kernels
echo "Building GPT-2 reproducible kernels..."
cd $PROJECT_DIR
mkdir -p $BUILD_DIR
REPRO_ARCH=$ARCH python setup.py build_ext --inplace 2>&1

echo "=== Build complete for $ARCH ==="
echo "Build output: $BUILD_DIR"
BUILD_EOF
