# Bit-Reproducible GPT-2 Training Across GPU Architectures

Bit-exact reproducible training of GPT-2 (124M parameters) on NYU HPC, producing **identical weight hashes** across different NVIDIA GPU architectures (A100, L4, V100).

## Overview

Standard deep learning training is non-deterministic across GPU architectures due to floating-point non-associativity in reductions, fused multiply-add differences, and non-deterministic atomics. This project replaces every non-deterministic operation in GPT-2 training with custom CUDA kernels that guarantee bit-identical results regardless of GPU type.

The project builds on top of [`repro_torch`](https://github.com/riya-mahesh/Bit_Reproducible_Training/tree/mnist-results) (the MNIST reproducible training package), extending it with GPT-2-specific layers.

## Architecture

```
repro_torch (base package — from mnist-results branch)
├── Ozaki GEMM (integer-exact matrix multiplication)
├── Superaccumulator (exact floating-point summation)
├── Adam with AdamClock (reproducible optimizer)
├── FiniteGuard (async NaN/Inf scanning)
├── SplitMix64 (portable deterministic PRNG)
├── CrossEntropyLoss (superaccumulator-based)
└── reproducible_init (SplitMix64 weight initialization)

gpt2_repro (this package — extends repro_torch)
├── ReproLayerNorm (superaccumulator mean/variance)
├── ReproGELU (correctly rounded tanh via --fmad=false)
├── ReproDropout (Philox counter-based RNG)
├── ReproEmbedding (sort + superaccumulator backward)
├── ReproCausalSelfAttention (reproducible softmax + causal mask)
└── GPT2 model (all layers use reproducible primitives)
```

## Key Techniques

| Technique | Purpose |
|-----------|---------|
| **Ozaki GEMM** | Integer-exact matrix multiplication with configurable slices (default 3 = 21-bit mantissa) |
| **Superaccumulator** | 320-bit exact, order-independent floating-point sums (LayerNorm, embedding backward, softmax, cross-entropy) |
| **`--fmad=false`** | Prevents CUDA compiler from fusing multiply-add operations (different rounding on different architectures) |
| **Philox-4x32-10 RNG** | Stateless, deterministic dropout masks independent of thread scheduling; regenerated in backward (no storage) |
| **Sort + superaccumulator** | Embedding backward uses counting sort + superaccumulator to avoid non-deterministic `atomicAdd` |
| **SplitMix64** | Portable 64-bit PRNG for deterministic data shuffling and weight initialization |
| **AdamClock** | Superaccumulator-based optimizer updates for bit-exact parameter updates |

## Custom CUDA Kernels

| Kernel | File | What it replaces |
|--------|------|-----------------|
| LayerNorm | `repro_layernorm/repro_layernorm.cu` | `torch.nn.LayerNorm` |
| GELU | `repro_gelu/repro_gelu.cu` | `torch.nn.functional.gelu` |
| Dropout | `repro_dropout/repro_dropout.cu` | `torch.nn.Dropout` |
| Embedding | `repro_embedding/repro_embedding.cu` | `torch.nn.Embedding` |
| Softmax + Causal Mask | `repro_attention/repro_attention.cu` | `torch.softmax` + manual masking |

Base kernels (from `repro_torch`): GEMM, summation, softmax, convolution, neural network ops.

## Project Structure

```
bit_reproducible_training/
├── setup.py                    # Build system for GPT-2 CUDA extensions
├── bindings.cu                 # PyTorch C++ bindings for GPT-2 CUDA kernels
├── gpt2_repro/
│   ├── __init__.py             # Package init — imports from repro_torch + GPT-2 layers
│   ├── model.py                # GPT-2 model using reproducible layers
│   ├── layers.py               # Autograd functions wrapping CUDA kernels
│   └── train.py                # Training script (repro and stock modes)
├── repro_gelu/                 # GELU CUDA kernel
├── repro_dropout/              # Philox dropout CUDA kernel
├── repro_layernorm/            # Superaccumulator LayerNorm CUDA kernel
├── repro_embedding/            # Sort+superaccumulator embedding CUDA kernel
├── repro_attention/            # Reproducible softmax + causal mask kernel
├── experiments/
│   └── compare_gpt2_results.py # Cross-GPU result comparison script
├── slurm/
│   ├── setup_env.sbatch        # One-time environment setup
│   ├── train_gpt2.sbatch       # Training job submission script
│   ├── build.sbatch            # Build-only job
│   └── run_all_gpus.sh         # Submit repro + stock experiments
├── results/                    # Training output (JSON + text)
└── tests/
    └── test_kernels.py         # Kernel correctness tests
```

## Quick Start on NYU HPC

### Prerequisites

- NYU HPC account with access to Cloud Bursting nodes
- SLURM account: `csci_ga_3033_025-2026fa` (or your own)
- Singularity overlay with conda + PyTorch (see Step 2)

### Step 1: Clone the Repository

```bash
cd ~
git clone -b gpt2 https://github.com/muskaankumar/Bit_Reproducible_Training.git gpt2_repro_project
cd gpt2_repro_project
```

### Step 2: Set Up the Environment (One-Time)

This creates a Singularity overlay with conda, PyTorch, and all dependencies.

```bash
# Edit slurm/setup_env.sbatch if your SLURM account differs
sbatch slurm/setup_env.sbatch

# Monitor progress
squeue -u $USER
cat logs/setup_*.out
```

This takes ~10 minutes. It creates an overlay at `$SCRATCH/repro_training/overlay-15GB-500K.ext3` with Miniconda, PyTorch (CUDA 12.6), tiktoken, datasets, and numpy.

If you already have an overlay (e.g., at `/scratch/<netid>/overlay-15GB-500K.ext3`), use that instead — just update the path in Step 3.

### Step 3: Get a GPU Node and Enter Singularity

```bash
# Request an A100 interactive session
srun --account=csci_ga_3033_025-2026fa --partition=c12m85-a100-1 \
     --gres=gpu:a100:1 --time=04:00:00 --mem=32G --pty /bin/bash

# Enter the Singularity container
singularity exec --nv \
    --overlay /scratch/$USER/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash

# Activate conda
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate base

# Disable TF32 (critical for reproducibility on A100/L4)
export NVIDIA_TF32_OVERRIDE=0
```

### Step 4: Install `repro_torch` (One-Time)

The base reproducible training package must be installed first:

```bash
cd ~/gpt2_repro_project

# Clone the MNIST branch which contains repro_torch
git clone -b mnist-results https://github.com/riya-mahesh/Bit_Reproducible_Training.git repro_torch_repo

# Install it (fix C++20 if needed)
cd repro_torch_repo/repro_torch
sed -i 's/-std=c++17/-std=c++20/g' setup.py
pip install -e . --no-build-isolation

cd ~/gpt2_repro_project
```

### Step 5: Build GPT-2 CUDA Kernels

```bash
cd ~/gpt2_repro_project
rm -rf build

# Set architecture for your GPU:
#   A100 → sm_80
#   L4   → sm_89
#   V100 → sm_70
REPRO_ARCH=sm_80 python setup.py build_ext --inplace
```

You should see all 5 `.cu` files plus `bindings.cu` compile and link into `gpt2_repro/_gpt2_backend.*.so`.

### Step 6: Run Training

```bash
mkdir -p results/gpt2

# REPRO mode — bit-reproducible training with memory profiling
python gpt2_repro/train.py \
    --mode repro --device cuda --seed 42 \
    --max-steps 1000 --block-size 256 --batch-size 4 \
    --lr 3e-4 --weight-decay 0.1 --grad-clip 1.0 \
    --eval-interval 100 --profile \
    --out results/gpt2/a100_repro.txt

# STOCK mode — standard PyTorch baseline (for speed/memory comparison)
python gpt2_repro/train.py \
    --mode stock --device cuda --seed 42 \
    --max-steps 1000 --block-size 256 --batch-size 4 \
    --lr 3e-4 --weight-decay 0.1 --grad-clip 1.0 \
    --eval-interval 100 --profile \
    --out results/gpt2/a100_stock.txt
```

### Step 7: Compare Results

```bash
python experiments/compare_gpt2_results.py results/gpt2/*.json --verbose
```

This shows:
- **Cross-GPU bit-identity check** — confirms repro runs produce identical weight hashes
- **Stock non-reproducibility** — shows that stock PyTorch produces different hashes
- **Performance table** — ms/step and peak memory for each GPU × mode
- **Overhead calculation** — repro vs stock time and memory ratios

### Step 8: Run on a Different GPU Architecture

Exit Singularity and the `srun` session, then request the other GPU:

```bash
# For L4:
srun --account=csci_ga_3033_025-2026fa --partition=g2-standard-12 \
     --gres=gpu:1 --time=04:00:00 --mem=32G --pty /bin/bash
```

Enter Singularity (Step 3), then rebuild for the new architecture and run training:

```bash
cd ~/gpt2_repro_project
rm -rf build
REPRO_ARCH=sm_89 python setup.py build_ext --inplace

# Run training (same commands as Step 6)
```

The comparison script picks up all JSON files, so after running on both GPUs:

```bash
python experiments/compare_gpt2_results.py results/gpt2/*.json --verbose
```

## Submitting Batch Jobs (Alternative to Interactive)

For unattended runs that survive disconnects:

```bash
# A100 — repro + stock with profiling
bash slurm/run_all_gpus.sh

# Or individually:
sbatch --partition=c12m85-a100-1 slurm/train_gpt2.sbatch
MODE=stock sbatch --partition=c12m85-a100-1 slurm/train_gpt2.sbatch

# With profiling:
MODE=repro PROFILE=1 sbatch --partition=c12m85-a100-1 slurm/train_gpt2.sbatch

# Check job status
squeue --account=csci_ga_3033_025-2026fa

# View output
cat logs/train_<jobid>.out
```

## Verifying Bit-Reproducibility

The training script prints a weight hash at every evaluation step and at the end:

```
hash init b3d20175f217
  step    10 loss 10.8171 (bits 412d12f3)
  step    20 loss 10.7838 (bits 412c8a4b)
  ...
  step   100 loss 8.6808  (bits 410ae49b)
hash final 32cd6f59f666
```

**Bit-reproducibility means the `hash final` value is identical across:**

- Multiple runs on the same GPU — determinism
- Different GPU architectures (A100 vs L4 vs V100) — cross-architecture reproducibility

The `bits` column shows the IEEE 754 hex representation of each step's loss, making even single-bit differences visible.

## Memory Profiling

Use `--profile` to get a detailed GPU memory breakdown:

```
=== MEMORY PROFILE ===
  Peak allocated:  2847 MiB
  Reserved (pool): 3120 MiB
  Parameters:      474.8 MiB
  Gradients:       474.8 MiB
  Optimizer (est): 949.6 MiB
  Activations+tmp: 947.8 MiB (est)
```

For more detailed profiling:

```python
# Add to train.py for a visual memory timeline:
torch.cuda.memory._record_memory_history()
# ... training loop ...
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
# View at: https://pytorch.org/memory_viz
```

## Training Modes

| Mode | What it does | Use case |
|------|-------------|----------|
| `repro` | Uses Ozaki GEMM, superaccumulator Adam, FiniteGuard, Philox dropout, reproducible init | Bit-identical results across GPUs |
| `stock` | Same model architecture, same initial weights, standard PyTorch ops | Speed/memory baseline for comparison |

Both modes use the same starting weights (`reproducible_init`) for a fair comparison.

## Available GPU Partitions on NYU HPC

| Partition | GPU | Compute Capability | GRES flag |
|-----------|-----|-------------------|-----------|
| `c12m85-a100-1` | A100 40GB (Ampere) | `sm_80` | `--gres=gpu:a100:1` |
| `g2-standard-12` | L4 (Ada Lovelace) | `sm_89` | `--gres=gpu:1` |
| `n1s8-v100-1` | V100 (Volta) | `sm_70` | `--gres=gpu:v100:1` |
| `n1s8-t4-1` | T4 (Turing) | `sm_75` | `--gres=gpu:t4:1` |

**Note:** L4 partitions require `--gres=gpu:1` without the type specifier. Other partitions use `--gres=gpu:<type>:1`.

## Critical Reproducibility Flags

| Flag | Where | Purpose |
|------|-------|---------|
| `NVIDIA_TF32_OVERRIDE=0` | Shell env variable | Disables TF32 on Ampere/Ada GPUs |
| `--fmad=false` | nvcc (setup.py) | Prevents FMA fusion across architectures |
| `--ftz=false` | nvcc (setup.py) | Keeps denormal numbers (no flush-to-zero) |
| `--prec-div=true` | nvcc (setup.py) | IEEE-compliant division |
| `--prec-sqrt=true` | nvcc (setup.py) | IEEE-compliant square root |

All flags are set automatically by `setup.py` — no manual configuration needed. You only need to set `NVIDIA_TF32_OVERRIDE=0` in your shell.

## GPU-Hour Budget

The course SLURM account has **300 GPU-hours**. Approximate costs per experiment:

| Experiment | GPU | Steps | Wall Time | GPU-hours |
|-----------|-----|-------|-----------|-----------|
| Quick test | A100 | 100 | ~17s | 0.005 |
| Quick test | L4 | 100 | ~1min | 0.02 |
| Full run | A100 | 1000 | ~3min | 0.05 |
| Full run | L4 | 1000 | ~8min | 0.13 |
| Full comparison (2 jobs) | A100 | 1000 | ~6min | 0.10 |

## Troubleshooting

### C++20 compilation error

```
error: #error C++20 or later compatible compiler is required to use PyTorch.
```

Your PyTorch requires C++20. Fix by changing `-std=c++17` to `-std=c++20` in `setup.py`:

```bash
sed -i 's/-std=c++17/-std=c++20/g' setup.py
```

Do the same in `repro_torch`'s `setup.py` if building that from source.

### Overlay locked error

```
FATAL: failed to open overlay image ... currently in use for writing by another process
```

Only one process can mount the overlay `:rw` at a time. Exit any other Singularity sessions using it, or mount as `:ro` if you only need read access.

### GRES specification error

```
srun: error: Invalid GRES specification
```

Use `--gres=gpu:1` (without type) for L4 partitions (`g2-standard-*`). Use `--gres=gpu:a100:1` or `--gres=gpu:v100:1` for other partitions.

### Module not found: repro_torch

You need to install the base package first. See Step 4.

### Dataset download fails

The WikiText dataset requires the namespaced name `Salesforce/wikitext` on recent versions of HuggingFace datasets. This is already set in `train.py`. If download fails on the compute node, download it on a login node first (which has internet access).

## References

- Ozaki et al., "Error-free transformations of matrix multiplication by using fast routines of matrix multiplication and its applications" (2012)
- Collange et al., "Reproducible and Accurate Matrix Multiplication for High-Performance Computing" (2015)
- Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (Philox RNG, 2011)

## License

MIT
