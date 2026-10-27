# Bit-Reproducible Training

GPU building blocks for neural-network training whose results are **bit-identical**: the same inputs give the same output bits on every run, process, kernel configuration, and (by design) GPU model. The goal is verifiable learning: a verifier can re-execute a training step and check the result exactly.

Ordinary GPU kernels are not bit-reproducible because they add floating-point numbers in an order that depends on tiling, parallel splits, library versions, and hardware, and each addition rounds. The components here avoid order-dependent floating-point arithmetic, so the result is defined by the math alone.

## Components

| Folder | What it is | Status |
|---|---|---|
| [`repro_gemm_cuda/`](repro_gemm_cuda/) | Bit-reproducible matrix multiplication (GEMM) for FP32, using exact integer arithmetic (an integer-exact Ozaki-style scheme with S = 1–3 int8 slices). Covers the forward and backward GEMMs of linear layers. | S = 1 verified bit-identical on 2× RTX 2080 Ti; S = 1–3 verified against a CPU reference on the host, GPU run pending |

Each folder has its own README with the method, guarantees, build steps, and tests.

## Still needed for a fully reproducible training step

- Reductions outside GEMM (LayerNorm/RMSNorm statistics, softmax sums, loss averaging)
- Batched/strided GEMM for attention
- Deterministic multi-GPU gradient all-reduce
- Replacements for atomics-based ops (embedding-gradient scatter-add, `index_add`)
- Seeded, recorded randomness (dropout, data order)

## Requirements

CUDA Toolkit 11.6+ (tested with 12.4) and an NVIDIA GPU. See each component's README.
