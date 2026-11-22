#!/bin/bash
#SBATCH --account=csci_ga_3033_025-2026fa
#SBATCH --job-name=gpt2-repro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================================
# TRAIN GPT-2 with reproducible kernels on one GPU.
#
# Submit with the right partition:
#   sbatch --partition=g2-standard-12   slurm/train_gpt2.sbatch   # L4
#   sbatch --partition=c12m85-a100-1    slurm/train_gpt2.sbatch   # A100
#
# Environment variables (optional overrides):
#   MODE=repro|stock     (default: repro)
#   SEED=42              (default: 42)
#   MAX_STEPS=1000       (default: 1000)
#   BLOCK_SIZE=256       (default: 256)
#   BATCH_SIZE=4         (default: 4)
#
# BUDGET: 300 GPU-hours total. This job requests 4h max.
#   1000 steps ≈ 1-2h on A100, ~2-3h on L4.
#   Monitor with: squeue --account=csci_ga_3033_025-2026fa
# ============================================================================

set -euo pipefail
mkdir -p logs results/gpt2

# Defaults
MODE=${MODE:-repro}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-1000}
BLOCK_SIZE=${BLOCK_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-4}

OVERLAY_DIR=$SCRATCH/repro_training
OVERLAY_FILE=$OVERLAY_DIR/overlay-15GB-500K.ext3

SINGULARITY_IMAGE=/share/apps/images/cuda12.4.1-cudnn9.1.0-devel-ubuntu22.04.4.sif
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.4.1-cudnn9.1.0-devel-ubuntu22.04.4.sif
fi

# Detect GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr ' ' '_')
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+' || echo "unknown")
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)

# Map GPU to arch
case "$GPU_NAME" in
    *A100*)       ARCH=sm_80 ;;
    *L4*)         ARCH=sm_89 ;;
    *V100*)       ARCH=sm_70 ;;
    *)            ARCH=native ;;
esac

# Tag for result files
TAG="${GPU_NAME}_cuda${CUDA_VER}_${MODE}_seed${SEED}"
RESULT_FILE="results/gpt2/${TAG}"

echo "================================================================"
echo "GPT-2 Reproducible Training"
echo "  GPU:       $GPU_NAME"
echo "  CUDA:      $CUDA_VER"
echo "  Driver:    $DRIVER"
echo "  Arch:      $ARCH"
echo "  Mode:      $MODE"
echo "  Seed:      $SEED"
echo "  Steps:     $MAX_STEPS"
echo "  Block:     $BLOCK_SIZE"
echo "  Batch:     $BATCH_SIZE"
echo "  Output:    $RESULT_FILE"
echo "  Account:   csci_ga_3033_025-2026fa"
echo "================================================================"

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=$SCRATCH/repro_training/builds/$ARCH

singularity exec --nv \
    --overlay $OVERLAY_FILE:rw \
    $SINGULARITY_IMAGE /bin/bash <<TRAIN_EOF
set -euo pipefail
source /ext3/miniconda3/bin/activate repro

export REPRO_BUILD=$BUILD_DIR
export RESULT_TAG=$TAG

# Disable TF32 globally (critical for reproducibility on A100/L4)
export NVIDIA_TF32_OVERRIDE=0

cd $PROJECT_DIR

python gpt2_repro/train.py \
    --mode $MODE \
    --device cuda \
    --seed $SEED \
    --max-steps $MAX_STEPS \
    --block-size $BLOCK_SIZE \
    --batch-size $BATCH_SIZE \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --eval-interval 100 \
    --tag $TAG \
    --out $RESULT_FILE.txt

echo ""
echo "=== Training complete ==="
echo "Result: $RESULT_FILE.txt / .json"
echo "Final weight hash:"
grep "hash final" $RESULT_FILE.txt || true

TRAIN_EOF
