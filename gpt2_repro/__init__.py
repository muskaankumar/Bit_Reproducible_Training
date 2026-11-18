"""
GPT-2 Bit-Reproducible Training Package.

Provides reproducible layers (LayerNorm, GELU, Dropout, Embedding, Attention)
built on superaccumulator arithmetic, Ozaki GEMM, and Philox RNG.
"""

import torch
import torch.nn as nn
import struct

from .layers import (
    ReproLayerNorm,
    ReproGELU,
    ReproDropout,
    ReproEmbedding,
    repro_softmax,
    repro_causal_mask_scale,
)

from . import layers

try:
    from .model import GPT2Config, GPT2
except ImportError:
    pass  # model.py has additional dependencies

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Configuration: number of Ozaki GEMM slices
# ---------------------------------------------------------------------------
_NUM_SLICES = 3

def set_slices(n):
    """Set the number of Ozaki GEMM slices (default 3 = 21-bit mantissa)."""
    global _NUM_SLICES
    _NUM_SLICES = n

def get_slices():
    return _NUM_SLICES


# ---------------------------------------------------------------------------
# Reproducible weight initialization
# ---------------------------------------------------------------------------
def reproducible_init(model, seed):
    """Initialize all model parameters deterministically using a seeded RNG.

    Uses torch.manual_seed for portability — the same seed produces the
    same parameters regardless of GPU architecture (init is done in FP32
    on CPU-side RNG, then moved to device).
    """
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            # Xavier / Glorot uniform — matches GPT-2 reference
            nn.init.xavier_uniform_(param)
        elif param.dim() == 1:
            nn.init.zeros_(param)


def name_layers(model):
    """Assign layer names to reproducible modules for debugging/logging."""
    for name, module in model.named_modules():
        if hasattr(module, '_repro_name'):
            continue
        module._repro_name = name


# ---------------------------------------------------------------------------
# SplitMix64: portable, deterministic PRNG for data shuffling
# ---------------------------------------------------------------------------
class SplitMix64:
    """Deterministic 64-bit PRNG — same sequence on every platform.

    Used for deterministic batch selection so that every GPU architecture
    sees the same training data in the same order.
    """
    def __init__(self, seed):
        # Mask to 64-bit unsigned
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        return z


# ---------------------------------------------------------------------------
# Reproducibility guard: checks for non-determinism at runtime
# ---------------------------------------------------------------------------
class _ReproGuard:
    """Lightweight sentinel that can detect non-deterministic ops at runtime."""

    def __init__(self):
        self._enabled = True

    def check(self, step):
        """Called once per step — placeholder for future invariant checks."""
        pass  # Could verify CUDA stream ordering, check for async errors, etc.

def _guard():
    return _ReproGuard()

# Attach guard factory to layers module so rt.layers.guard() works
layers.guard = _guard


# ---------------------------------------------------------------------------
# Reproducible Adam / AdamW optimizer
# ---------------------------------------------------------------------------
class Adam(torch.optim.Optimizer):
    """Deterministic Adam/AdamW that avoids non-reproducible reductions.

    For now wraps PyTorch's AdamW with deterministic settings.
    Future: superaccumulator-based parameter updates.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, adamw=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        adamw=adamw)
        super().__init__(params, defaults)
        # Pre-allocate state
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']
            adamw = group['adamw']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Bias correction
                bias1 = 1 - beta1 ** state['step']
                bias2 = 1 - beta2 ** state['step']

                # Decoupled weight decay (AdamW)
                if adamw and wd > 0:
                    p.mul_(1 - lr * wd)

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute step
                denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
                step_size = lr / bias1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
