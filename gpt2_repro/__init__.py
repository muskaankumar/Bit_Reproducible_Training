"""GPT-2 Bit-Reproducible Training Package.

Extends repro_torch with GPT-2-specific reproducible layers (LayerNorm, GELU,
Dropout, Embedding, Attention) built on the same superaccumulator arithmetic,
Ozaki GEMM, and Philox RNG used in the MNIST pipeline.

The base primitives (Linear, cross_entropy, Adam, FiniteGuard, softmax) come
from repro_torch so that GEMM, bias-gradient, and optimizer arithmetic is
bit-identical to the MNIST path.
"""

import torch

# ---------------------------------------------------------------------------
# Import base reproducible primitives from repro_torch (the MNIST package).
# These provide: Ozaki GEMM Linear, superaccumulator cross-entropy and Adam,
# FiniteGuard with async NaN/Inf scanning, SplitMix64, reproducible_init, etc.
# ---------------------------------------------------------------------------
from repro_torch import (
    Adam,
    CrossEntropyLoss,
    FiniteGuard,
    Linear,
    SplitMix64,
    cross_entropy,
    get_slices,
    has_cuda,
    model_hash,
    name_layers,
    reproducible_init,
    set_slices,
    shuffled,
    softmax,
    tensor_hash,
)

# Import the guard singleton from repro_torch.layers
from repro_torch.layers import guard

# ---------------------------------------------------------------------------
# GPT-2-specific reproducible layers
# ---------------------------------------------------------------------------
from .layers import (
    ReproCausalSelfAttention,
    ReproDropout,
    ReproEmbedding,
    ReproGELU,
    ReproLayerNorm,
    repro_causal_mask_scale,
    repro_softmax,
)

try:
    from .model import GPT2, GPT2Config
except ImportError:
    pass  # model.py has additional dependencies

__version__ = "0.1.0"

__all__ = [
    # From repro_torch (MNIST base)
    "Adam", "CrossEntropyLoss", "FiniteGuard", "Linear", "SplitMix64",
    "cross_entropy", "get_slices", "guard", "has_cuda", "model_hash",
    "name_layers", "reproducible_init", "set_slices", "shuffled",
    "softmax", "tensor_hash",
    # GPT-2-specific
    "ReproCausalSelfAttention", "ReproDropout", "ReproEmbedding",
    "ReproGELU", "ReproLayerNorm",
    "repro_causal_mask_scale", "repro_softmax",
]
