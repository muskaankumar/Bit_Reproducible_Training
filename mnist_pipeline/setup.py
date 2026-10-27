"""
Build the repro_gemm PyTorch extension.

    cd mnist_pipeline && pip install -e .

Or:
    python setup.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

here = os.path.dirname(os.path.abspath(__file__))
gemm_dir = os.path.join(here, '..', 'repro_gemm_cuda')

setup(
    name='repro_gemm_ext',
    ext_modules=[
        CUDAExtension(
            'repro_gemm_ext',
            sources=[
                os.path.join(here, 'repro_gemm_ext.cu'),
                os.path.join(gemm_dir, 'repro_gemm.cu'),
            ],
            include_dirs=[gemm_dir],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-lineinfo',
                    # DO NOT use --use_fast_math: it changes rounding behavior
                    # and breaks reproducibility of element-wise ops.
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
