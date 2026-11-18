"""
Build script for GPT-2 reproducible training kernels.

This extends the base repro_torch package with new CUDA kernels for:
  - LayerNorm (superaccumulator mean/variance)
  - GELU (correctly rounded tanh via exp)
  - Dropout (Philox counter-based RNG)
  - Embedding (sort-then-superaccumulator backward)
  - Attention (reproducible softmax, causal mask)

Usage:
    # Auto-detect GPU arch
    python setup.py install

    # Specify arch explicitly (for cross-compilation)
    REPRO_ARCH=sm_80 python setup.py install

    # CPU-only (no CUDA kernels)
    REPRO_CPU_ONLY=1 python setup.py install
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages

# Check if we can use CUDA
USE_CUDA = not os.environ.get("REPRO_CPU_ONLY", "")
cuda_ext = []

if USE_CUDA:
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        # Determine compute capability
        arch = os.environ.get("REPRO_ARCH", "")
        if not arch:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap",
                     "--format=csv,noheader"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    cc = result.stdout.strip().split('\n')[0].replace('.', '')
                    arch = f"sm_{cc}"
                else:
                    arch = "sm_70"  # default to V100
            except FileNotFoundError:
                arch = "sm_70"

        print(f"Building for arch: {arch}")

        # CUDA source files
        cuda_sources = [
            "repro_gelu/repro_gelu.cu",
            "repro_dropout/repro_dropout.cu",
            "repro_layernorm/repro_layernorm.cu",
            "repro_embedding/repro_embedding.cu",
            "repro_attention/repro_attention.cu",
        ]

        # Check which files exist
        existing_sources = [s for s in cuda_sources if os.path.exists(s)]
        if not existing_sources:
            print("WARNING: No CUDA source files found. Building CPU-only.")
            USE_CUDA = False
        else:
            print(f"CUDA sources: {existing_sources}")

            nvcc_flags = [
                f"-arch={arch}",
                "--fmad=false",        # CRITICAL: no FMA fusion
                "-O2",
                "--ftz=false",             # no flush-to-zero
                "--prec-div=true",         # precise division
                "--prec-sqrt=true",        # precise sqrt
                "-DREPRO_ARCH=" + arch,
            ]

            cuda_ext = [
                CUDAExtension(
                    name="gpt2_repro._gpt2_backend",
                    sources=existing_sources + ["bindings.cu"],
                    extra_compile_args={
                        "cxx": ["-O2", "-std=c++17"],
                        "nvcc": nvcc_flags,
                    },
                    include_dirs=[
                        "repro_gelu",
                        "repro_dropout",
                        "repro_layernorm",
                        "repro_embedding",
                        "repro_attention",
                    ],
                )
            ]
    except ImportError:
        print("PyTorch not found or no CUDA support. Building CPU-only.")
        USE_CUDA = False

# Determine build command class
cmdclass = {}
if USE_CUDA and cuda_ext:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass["build_ext"] = BuildExtension

setup(
    name="gpt2_repro",
    version="0.1.0",
    description="Bit-reproducible GPT-2 training across GPU architectures",
    packages=find_packages(),
    ext_modules=cuda_ext,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "tiktoken",
    ],
)