# Bit-Reproducible GPT-2 Training Across GPU Architectures

Extends [Bit_Reproducible_Training](https://github.com/riya-mahesh/Bit_Reproducible_Training)
from LeNet-5/MNIST to GPT-2 (small, 124M parameters), with bit-identical final weights
across GPU architectures (L4, A100), CUDA versions, and CPUs.

## What's New (over the MNIST baseline)

| Component | What it does | Status |
|---|---|---|
| `repro_layernorm/` | LayerNorm using superaccumulator for mean/variance, correctly rounded rsqrt | Implemented |
| `repro_gelu/` | Correctly rounded GELU (tanh approx via repro_exp), bit-identical across GPUs | Implemented |
| `repro_embedding/` | Embedding forward (gather) + backward (deterministic scatter-add via counting sort) | Implemented |
| `repro_attention/` | Batched GEMM for multi-head attention, causal mask, reproducible softmax | Implemented |
| `repro_dropout/` | Philox-4x32-10 counter-based RNG, mask recomputed in backward (no storage) | Implemented |
| `gpt2_repro/` | GPT-2 small model + training loop on WikiText-103 | Implemented |

## Target Hardware (NYU HPC Cloud Bursting)

| GPU | Architecture | Compute Capability | Partition |
|---|---|---|---|
| NVIDIA L4 | Ada Lovelace | sm_89 | `g2-standard-12` (1×) or `g2-standard-24` (2×) |
| NVIDIA A100 40GB | Ampere | sm_80 | `c12m85-a100-1` (1×) or `c24m170-a100-2` (2×) |

Two different architectures = cross-architecture reproducibility claim.

**Budget: 300 GPU-hours total for the semester — be careful!**

## Quick Start (NYU HPC via Open OnDemand)

```bash
# 1. Clone and enter
git clone <this-repo> && cd gpt2_repro_project

# 2. Set up environment (one-time, runs on CPU partition — no GPU hours)
sbatch slurm/setup_env.sbatch

# 3. Build CUDA kernels (needs GPU — use L4 to save budget)
sbatch --partition=g2-standard-12 slurm/build.sbatch    # ~15 min on L4

# 4. Build for A100 too
sbatch --partition=c12m85-a100-1 slurm/build.sbatch     # ~10 min on A100

# 5. Run training on both GPUs + stock baseline (~6-8 GPU-hours total)
bash slurm/run_all_gpus.sh

# 6. Compare results
python experiments/compare_gpt2_results.py results/gpt2/*.json
```

### Submitting Individual Jobs

```bash
# Repro mode on L4
sbatch --partition=g2-standard-12 \
    --export=ALL,MODE=repro,SEED=42,MAX_STEPS=500 \
    slurm/train_gpt2.sbatch

# Repro mode on A100
sbatch --partition=c12m85-a100-1 \
    --export=ALL,MODE=repro,SEED=42,MAX_STEPS=500 \
    slurm/train_gpt2.sbatch

# Stock PyTorch baseline
sbatch --partition=g2-standard-12 \
    --export=ALL,MODE=stock,SEED=42,MAX_STEPS=500 \
    slurm/train_gpt2.sbatch

# Monitor
squeue --account=csci_ga_3033_025-2026fa
```

## Slurm Account

```
Account:    csci_ga_3033_025-2026fa
Budget:     300 GPU-hours (18,000 minutes)
VPN:        Required from off-campus
Access:     Open OnDemand — https://ood.hpc.nyu.edu
```

## Requirements

- CUDA Toolkit 12.x
- PyTorch 2.x with matching CUDA
- The base `Bit_Reproducible_Training` repo (cloned as a sibling or submodule)

## Project Structure

```
gpt2_repro_project/
├── repro_gelu/           # GELU kernel (.cu + .cuh)
├── repro_dropout/        # Philox dropout kernel
├── repro_layernorm/      # LayerNorm with superaccumulator
├── repro_embedding/      # Embedding with sort-based backward
├── repro_attention/      # Softmax + causal mask kernels
├── gpt2_repro/           # Python package
│   ├── layers.py         # Autograd functions wrapping CUDA kernels
│   ├── model.py          # GPT-2 small (124M params)
│   └── train.py          # Training script
├── slurm/                # SLURM job scripts for NYU HPC
├── experiments/          # Result comparison scripts
├── tests/                # Unit tests (27 tests)
└── setup.py              # Build system
```
