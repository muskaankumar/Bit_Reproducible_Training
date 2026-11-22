"""
Build script for GPT-2 reproducible training kernels.

This builds on top of repro_torch (the MNIST package), which must be installed
first.  repro_torch provides the base GEMM, sum, softmax, conv, and nn kernels;
this package adds GPT-2-specific CUDA kernels for:
  - LayerNorm  (superaccumulator mean/variance)
  - GELU       (correctly rounded tanh via exp, --fmad=false)
  - Dropout    (Philox counter-based RNG)
  - Embedding  (gather forward, sort+superacc backward)
  - Attention  (reproducible softmax, causal mask)

Usage:
    # First install the base package (from the mnist-results branch)
    cd repro_torch && python setup.py build_ext --inplace && pip install -e .

    # Then install GPT-2 extensions
    python setup.py install

    # Specify arch explicitly (for cross-compilation)
    REPRO_ARCH=sm_80 python setup.py install

    # CPU-only (no CUDA kernels — CPU fallbacks will be used)
    REPRO_CPU_ONLY=1 python setup.py install
"""

import os
import subprocess
import shutil

from setuptools import setup, find_packages

USE_CUDA = not os.environ.get("REPRO_CPU_ONLY", "")
cuda_ext = []

if USE_CUDA:
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        # Determine compute capability (same logic as repro_torch/setup.py)
        arch = os.environ.get("REPRO_ARCH", "")
        if not arch:
            if shutil.which("nvcc") is not None:
                arch = "native"
            else:
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
                        arch = "sm_70"
                except FileNotFoundError:
                    arch = "sm_70"

        print(f"Building GPT-2 kernels for arch: {arch}")

        # GPT-2-specific CUDA sources
        cuda_sources = []
        for name in ("repro_gelu", "repro_dropout", "repro_layernorm",
                     "repro_embedding", "repro_attention"):
            cu_path = os.path.join(name, name + ".cu")
            if os.path.exists(cu_path):
                cuda_sources.append(cu_path)

        if not cuda_sources:
            print("WARNING: No GPT-2 CUDA source files found. "
                  "Building CPU-only.")
            USE_CUDA = False
        else:
            print(f"GPT-2 CUDA sources: {cuda_sources}")

            # Same nvcc flags as repro_torch — critically --fmad=false
            nvcc_flags = [
                f"-arch={arch}",
                "--fmad=false",        # CRITICAL: no FMA fusion
                "-O2",
                "--ftz=false",         # no flush-to-zero
                "--prec-div=true",     # precise division
                "--prec-sqrt=true",    # precise sqrt
                "-std=c++20",
                "-lineinfo",
            ]

            # Include dirs for the GPT-2 kernel headers
            include_dirs = [
                name for name in ("repro_gelu", "repro_dropout",
                                  "repro_layernorm", "repro_embedding",
                                  "repro_attention")
                if os.path.isdir(name)
            ]

            cuda_ext = [
                CUDAExtension(
                    name="gpt2_repro._gpt2_backend",
                    sources=cuda_sources + ["bindings.cu"],
                    extra_compile_args={
                        "cxx": ["-O2", "-std=c++20", "-fno-fast-math",
                                "-ffp-contract=off"],
                        "nvcc": nvcc_flags,
                    },
                    include_dirs=include_dirs,
                )
            ]
    except ImportError:
        print("PyTorch not found or no CUDA support. Building CPU-only.")
        USE_CUDA = False

cmdclass = {}
if USE_CUDA and cuda_ext:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass["build_ext"] = BuildExtension

setup(
    name="gpt2_repro",
    version="0.1.0",
    description="Bit-reproducible GPT-2 training across GPU architectures",
    long_description="Extends repro_torch with GPT-2-specific reproducible "
                     "layers (LayerNorm, GELU, Dropout, Embedding, Attention). "
                     "Requires repro_torch to be installed first.",
    packages=find_packages(),
    ext_modules=cuda_ext,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "repro_torch",   # base package from mnist-results branch
        "tiktoken",
    ],
)