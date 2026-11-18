#!/bin/bash
#SBATCH --account=csci_ga_3033_025-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --job-name=repro-setup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# ============================================================================
# ONE-TIME ENVIRONMENT SETUP for NYU HPC (Cloud Bursting via OOD)
#
# Creates a Singularity overlay with conda, PyTorch, and dependencies.
# Run once: sbatch slurm/setup_env.sbatch
#
# GPU partitions available:
#   g2-standard-12      1× NVIDIA L4        (Ada Lovelace, sm_89)
#   g2-standard-24      2× NVIDIA L4
#   c12m85-a100-1       1× NVIDIA A100 40GB (Ampere, sm_80)
#   c24m170-a100-2      2× NVIDIA A100 40GB
#
# BUDGET: 300 GPU-hours total — be very careful!
# ============================================================================

set -euo pipefail
mkdir -p logs

echo "=== Setting up reproducible training environment ==="
echo "Host: $(hostname), Date: $(date)"

# --- Overlay filesystem ---
OVERLAY_DIR=$SCRATCH/repro_training
OVERLAY_FILE=$OVERLAY_DIR/overlay-15GB-500K.ext3
SINGULARITY_IMAGE=/share/apps/images/cuda12.4.1-cudnn9.1.0-devel-ubuntu22.04.4.sif

# Fallback image paths (varies by HPC setup)
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.4.1-cudnn9.1.0-devel-ubuntu22.04.4.sif
fi
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: Cannot find Singularity image. Available images:"
    ls /share/apps/images/*.sif 2>/dev/null || ls /scratch/work/public/singularity/*.sif 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "Using image: $SINGULARITY_IMAGE"

mkdir -p $OVERLAY_DIR

if [ ! -f "$OVERLAY_FILE" ]; then
    echo "Creating overlay filesystem..."
    # Try cloud bursting path first, then Greene path
    OVERLAY_TEMPLATE=""
    for path in /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz \
                /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz; do
        if [ -f "$path" ]; then
            OVERLAY_TEMPLATE="$path"
            break
        fi
    done
    if [ -z "$OVERLAY_TEMPLATE" ]; then
        echo "ERROR: Cannot find overlay template. Available:"
        ls /share/apps/overlay-fs-ext3/ 2>/dev/null || echo "  (none at /share/apps)"
        ls /scratch/work/public/overlay-fs-ext3/ 2>/dev/null || echo "  (none at /scratch/work)"
        exit 1
    fi
    cp -rp "$OVERLAY_TEMPLATE" $OVERLAY_DIR/
    gunzip $OVERLAY_DIR/overlay-15GB-500K.ext3.gz
fi

# --- Install conda + packages inside the overlay ---
singularity exec --overlay $OVERLAY_FILE:rw $SINGULARITY_IMAGE /bin/bash <<'INSTALL_EOF'
set -euo pipefail

if [ ! -d /ext3/miniconda3 ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /ext3/miniconda3
    rm /tmp/miniconda.sh
fi

source /ext3/miniconda3/bin/activate

# Create environment
if ! conda info --envs | grep -q repro; then
    conda create -n repro python=3.11 -y
fi
conda activate repro

# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Other dependencies
pip install tiktoken datasets numpy

echo "=== Environment setup complete ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

INSTALL_EOF

echo "Done. Overlay at: $OVERLAY_FILE"
