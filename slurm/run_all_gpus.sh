#!/bin/bash
# ============================================================================
# Submit GPT-2 training on both GPU types (L4 + A100) for comparison.
# Run from project root:
#   bash slurm/run_all_gpus.sh
#
# After all jobs complete:
#   python experiments/compare_gpt2_results.py results/gpt2/*.json
#
# BUDGET WARNING: 300 GPU-hours total for the semester!
#   Each 1000-step run uses ~1-3 GPU-hours depending on GPU.
#   This script submits 3 jobs = ~6-8 GPU-hours.
# ============================================================================

set -euo pipefail

ACCOUNT=csci_ga_3033_025-2026fa
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-1000}

echo "============================================"
echo "  GPT-2 Bit-Reproducible Training Suite"
echo "  Account: $ACCOUNT"
echo "  Seed: $SEED, Steps: $MAX_STEPS"
echo "============================================"
echo ""
echo "⚠  BUDGET: ~6-8 GPU-hours for 3 jobs"
echo ""

# --- Reproducible mode on L4 (Ada Lovelace, sm_89) ---
JOB_L4=$(sbatch --partition=g2-standard-12 \
    --export=ALL,MODE=repro,SEED=$SEED,MAX_STEPS=$MAX_STEPS \
    --job-name="gpt2-repro-l4" \
    slurm/train_gpt2.sbatch | awk '{print $4}')
echo "  ✓ Submitted repro on L4:   job $JOB_L4  (partition: g2-standard-12)"

# --- Reproducible mode on A100 (Ampere, sm_80) ---
JOB_A100=$(sbatch --partition=c12m85-a100-1 \
    --export=ALL,MODE=repro,SEED=$SEED,MAX_STEPS=$MAX_STEPS \
    --job-name="gpt2-repro-a100" \
    slurm/train_gpt2.sbatch | awk '{print $4}')
echo "  ✓ Submitted repro on A100: job $JOB_A100 (partition: c12m85-a100-1)"

# --- Stock PyTorch baseline on L4 (cheapest GPU) ---
JOB_STOCK=$(sbatch --partition=g2-standard-12 \
    --export=ALL,MODE=stock,SEED=$SEED,MAX_STEPS=$MAX_STEPS \
    --job-name="gpt2-stock-l4" \
    slurm/train_gpt2.sbatch | awk '{print $4}')
echo "  ✓ Submitted stock on L4:   job $JOB_STOCK (partition: g2-standard-12)"

echo ""
echo "Monitor:  squeue --account=$ACCOUNT"
echo "Cancel:   scancel <jobid>"
echo "After:    python experiments/compare_gpt2_results.py results/gpt2/*.json"
