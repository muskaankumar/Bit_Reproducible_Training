Bit-Reproducible GPT-2 Training Across GPU Architectures

Bit-exact reproducible training of GPT-2 (124M parameters) on NYU HPC, producing identical weight hashes across different NVIDIA GPU architectures (A100, L4, V100).

Overview

Standard deep learning training is non-deterministic across GPU architectures due to floating-point non-associativity in reductions, fused multiply-add differences, and non-deterministic atomics. This project replaces every non-deterministic operation in GPT-2 training with custom CUDA kernels that guarantee bit-identical results regardless of GPU type.

Key Techniques
Superaccumulator summation: 10 signed 64-bit limbs (320 bits) for exact, order-independent floating-point sums (used in LayerNorm, embedding backward, softmax)
--fmad=false compilation: Prevents the CUDA compiler from fusing multiply-add operations, which produce different rounding on different architectures
Philox-4x32-10 counter-based RNG: Stateless, deterministic dropout masks independent of thread scheduling
Sort-then-superaccumulator: Embedding backward uses counting sort + superaccumulator to avoid non-deterministic atomicAdd
Correctly rounded transcendentals: GELU activation uses --ftz=false --prec-div=true --prec-sqrt=true for identical results across architectures
Custom CUDA Kernels
Kernel	File	What it replaces
LayerNorm	repro_layernorm/repro_layernorm.cu	torch.nn.LayerNorm
GELU	repro_gelu/repro_gelu.cu	torch.nn.functional.gelu
Dropout	repro_dropout/repro_dropout.cu	torch.nn.Dropout
Embedding	repro_embedding/repro_embedding.cu	torch.nn.Embedding
Softmax + Causal Mask	repro_attention/repro_attention.cu	torch.softmax + manual masking
Project Structure
gpt2_repro_project/
├── setup.py                    # Build system for CUDA extensions
├── bindings.cu                 # PyTorch C++ bindings for all CUDA kernels
├── gpt2_repro/
│   ├── __init__.py             # Package init, SplitMix64 RNG, deterministic Adam
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
│   └── run_all_gpus.sh         # Run experiments on all GPU types
├── results/                    # Training output (JSON + text)
└── tests/
    └── test_kernels.py         # Kernel correctness tests
Quick Start on NYU HPC
Prerequisites
NYU HPC account with access to Cloud Bursting nodes
SLURM account: csci_ga_3033_025-2026fa (or your own)
Familiarity with Singularity containers
Step 1: Clone the Repository
bash
cd ~
git clone https://github.com/riya-mahesh/Bit_Reproducible_Training.git gpt2_repro_project
cd gpt2_repro_project
Step 2: Set Up the Environment (One-Time)

This creates a Singularity overlay with conda, PyTorch, and all dependencies.

bash
# Edit slurm/setup_env.sbatch if your SLURM account differs
sbatch slurm/setup_env.sbatch

# Monitor progress
squeue -u $USER
cat logs/setup_*.out

This takes ~10 minutes. It creates an overlay at $SCRATCH/repro_training/overlay-15GB-500K.ext3 with Miniconda, PyTorch (CUDA 12.6), tiktoken, datasets, and numpy.

If the overlay already exists (e.g., at /scratch/<netid>/overlay-15GB-500K.ext3), you can use that instead — just update the path in the scripts.

Step 3: Build the CUDA Kernels

Start an interactive session on a GPU node:

bash
# For A100:
srun --account=csci_ga_3033_025-2026fa --partition=c12m85-a100-1 \
     --gres=gpu:a100:1 --time=01:00:00 --mem=32G --pty /bin/bash

# For L4 (note: use gpu:1, not gpu:l4:1):
srun --account=csci_ga_3033_025-2026fa --partition=g2-standard-12 \
     --gres=gpu:1 --time=01:00:00 --mem=32G --pty /bin/bash

Then enter the Singularity container and build:

bash
singularity exec --nv \
    --overlay /scratch/$USER/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate base

cd ~/gpt2_repro_project
rm -rf build

# Set architecture for your GPU:
#   A100 → sm_80
#   L4   → sm_89
#   V100 → sm_70
REPRO_ARCH=sm_80 python setup.py build_ext --inplace

You should see all 5 .cu files plus bindings.cu compile and link into gpt2_repro/_gpt2_backend.*.so.

Step 4: Run Training
bash
# CRITICAL: Disable TF32 for reproducibility on A100/L4
export NVIDIA_TF32_OVERRIDE=0

# Reproducible mode
python gpt2_repro/train.py \
    --mode repro \
    --device cuda \
    --seed 42 \
    --max-steps 100 \
    --block-size 256 \
    --batch-size 4 \
    --lr 3e-4 \
    --tag A100_repro_seed42

# Stock PyTorch baseline (for comparison — results WILL differ across GPUs)
python gpt2_repro/train.py \
    --mode stock \
    --device cuda \
    --seed 42 \
    --max-steps 100 \
    --block-size 256 \
    --batch-size 4 \
    --lr 3e-4 \
    --tag A100_stock_seed42
Step 5: Run on a Different GPU Architecture

Repeat Steps 3–4 on a different GPU partition (e.g., L4 after A100). You must rebuild for each architecture:

bash
rm -rf build
REPRO_ARCH=sm_89 python setup.py build_ext --inplace
Step 6: Compare Results
bash
python experiments/compare_gpt2_results.py results/gpt2/*.json --verbose
Submitting Batch Jobs

For unattended runs, use the SLURM batch scripts:

bash
# A100
sbatch --partition=c12m85-a100-1 slurm/train_gpt2.sbatch

# L4
sbatch --partition=g2-standard-12 slurm/train_gpt2.sbatch

# Override defaults
MODE=stock SEED=123 MAX_STEPS=500 \
    sbatch --partition=c12m85-a100-1 slurm/train_gpt2.sbatch

# Check job status
squeue --account=csci_ga_3033_025-2026fa

# View output
cat logs/train_<jobid>.out
Verifying Bit-Reproducibility

The training script prints a weight hash at every evaluation step and at the end:

hash init b3d20175f217
  step    10 loss 10.8171 (bits 412d12f3)
  step    20 loss 10.7838 (bits 412c8a4b)
  ...
  step   100 loss 8.6808  (bits 410ae49b)
hash final 32cd6f59f666

Bit-reproducibility means the hash final value is identical across:

Multiple runs on the same GPU — determinism
Different GPU architectures (A100 vs L4 vs V100) — cross-architecture reproducibility

The bits column shows the IEEE 754 hex representation of each step's loss, making even single-bit differences visible.

Available GPU Partitions on NYU HPC
Partition	GPU	Compute Capability	GRES flag
c12m85-a100-1	A100 40GB (Ampere)	sm_80	--gres=gpu:a100:1
g2-standard-12	L4 (Ada Lovelace)	sm_89	--gres=gpu:1
n1s8-v100-1	V100 (Volta)	sm_70	--gres=gpu:v100:1
n1s8-t4-1	T4 (Turing)	sm_75	--gres=gpu:t4:1

Note: L4 partitions require --gres=gpu:1 without the type specifier. Other partitions use --gres=gpu:<type>:1.

Critical Reproducibility Flags
Flag	Where	Purpose
NVIDIA_TF32_OVERRIDE=0	Shell env variable	Disables TF32 on Ampere/Ada GPUs
--fmad=false	nvcc (setup.py)	Prevents FMA fusion across architectures
--ftz=false	nvcc (setup.py)	Keeps denormal numbers (no flush-to-zero)
--prec-div=true	nvcc (setup.py)	IEEE-compliant division
--prec-sqrt=true	nvcc (setup.py)	IEEE-compliant square root

All flags are set automatically by setup.py — no manual configuration needed.

GPU-Hour Budget

The course SLURM account has 300 GPU-hours. Approximate costs per experiment:

Experiment	GPU	Steps	Wall Time	GPU-hours
Quick test	A100	100	~17s	0.005
Quick test	L4	100	~1min	0.02
Full run	A100	1000	~3min	0.05
Full run	L4	1000	~8min	0.13
Full run	V100	1000	~10min	0.17

You have plenty of budget for dozens of runs across all GPU types.

Troubleshooting
Overlay locked error
FATAL: failed to open overlay image ... currently in use for writing by another process

Only one process can mount the overlay :rw at a time. Exit any other Singularity sessions using it. If you only need read access, mount as :ro.

GRES specification error
srun: error: Invalid GRES specification (with and without type identification)

Use --gres=gpu:1 (without type) for L4 partitions (g2-standard-*). Use --gres=gpu:a100:1 or --gres=gpu:v100:1 for other partitions.

--use_fast_math=false nvcc error

Never use --use_fast_math=false — it's a boolean toggle with no =false form. The individual flags --ftz=false, --prec-div=true, --prec-sqrt=true are the correct replacements and are already set in setup.py.

C++20 compilation errors

The bindings file must be bindings.cu (not .cpp) so nvcc compiles it with C++20 support. The host g++ on HPC is too old for PyTorch's C++20 headers.

getCurrentCUDAStream not found

If you see namespace "at::cuda" has no member "getCurrentCUDAStream", ensure bindings.cu uses c10::cuda::getCurrentCUDAStream() and includes <c10/cuda/CUDAStream.h>. This is already fixed in the current code.

Dataset download fails

The WikiText dataset requires the namespaced name Salesforce/wikitext on recent versions of HuggingFace datasets. This is already set in train.py.

References
Ozaki et al., "Error-free transformations of matrix multiplication by using fast routines of matrix multiplication and its applications" (2012)
Collange et al., "Reproducible and Accurate Matrix Multiplication for High-Performance Computing" (2015)
Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (Philox RNG, 2011)
License

MIT
