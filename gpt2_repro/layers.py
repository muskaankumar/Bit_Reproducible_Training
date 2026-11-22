"""GPT-2-specific reproducible autograd layers.

Extends the base repro_torch package (Linear, Conv2d, CrossEntropy, Adam,
FiniteGuard) with operations needed for transformer training:

  - LayerNorm  (superaccumulator mean/variance)
  - GELU       (correctly rounded exp in tanh approximation, --fmad=false)
  - Dropout    (Philox counter-based RNG, mask regenerated in backward)
  - Embedding  (gather forward, sort-then-superaccumulator backward)
  - Softmax    (repro_exp + superaccumulator denominator)
  - CausalSelfAttention  (batched GEMM + reproducible softmax + causal mask)

All GEMM operations route through repro_torch's B.gemm (Ozaki slices) so
that the linear projections inside attention, MLP, and embedding are
bit-identical to the MNIST path.  The _backend reference (B) comes from
repro_torch, which loads the compiled CUDA extension; the GPT-2 kernels
(layernorm, gelu, dropout, embedding, attention) are compiled into their
own extension (_gpt2_backend) with the same --fmad=false flags.
"""

import math

import torch
from torch.autograd import Function

# ---------------------------------------------------------------------------
# Backend: use repro_torch's compiled _backend for GEMM, exact_sum, softmax,
# scan_finite, Adam, cross_entropy — same module the MNIST layers use.
# ---------------------------------------------------------------------------
from repro_torch import _backend as B
from repro_torch.layers import _guard, get_slices

# ---------------------------------------------------------------------------
# GPT-2-specific CUDA backend (layernorm, gelu, dropout, embedding, attention).
# Loaded lazily so the package still imports when only CPU is available.
# ---------------------------------------------------------------------------
_gpt2_backend = None
_gpt2_backend_loaded = False


def _get_gpt2_backend():
    """Returns the GPT-2-specific CUDA backend, or None if unavailable."""
    global _gpt2_backend, _gpt2_backend_loaded
    if _gpt2_backend_loaded:
        return _gpt2_backend
    _gpt2_backend_loaded = True
    try:
        import pathlib
        import importlib.util
        import importlib.machinery

        pkg_dir = pathlib.Path(__file__).parent
        so_files = list(pkg_dir.glob("_gpt2_backend*.so"))
        if so_files:
            so_path = str(so_files[0])
            loader = importlib.machinery.ExtensionFileLoader(
                "gpt2_repro._gpt2_backend", so_path)
            spec = importlib.util.spec_from_file_location(
                "gpt2_repro._gpt2_backend", so_path, loader=loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            _gpt2_backend = mod
    except Exception as e:
        import sys
        print(f"[gpt2_repro] WARNING: could not load GPT-2 CUDA backend: {e}",
              file=sys.stderr)
        _gpt2_backend = None
    return _gpt2_backend


def _has_gpt2_backend():
    return _get_gpt2_backend() is not None


# ============================================================================
# LayerNorm
# ============================================================================

class _LayerNormFn(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        D = input.shape[-1]
        x = input.contiguous().view(-1, D)
        N = x.shape[0]

        output = torch.empty_like(x)
        mean = torch.empty(N, device=x.device, dtype=x.dtype)
        rstd = torch.empty(N, device=x.device, dtype=x.dtype)

        if x.is_cuda and _has_gpt2_backend():
            _get_gpt2_backend().repro_layernorm_forward(
                x, weight, bias, output, mean, rstd, N, D, eps)
        else:
            # CPU fallback using superaccumulator logic
            for i in range(N):
                row = x[i]
                mu = row.sum().item() / D
                mean[i] = mu
                diff = row - mu
                var = (diff * diff).sum().item() / D
                rs = 1.0 / (var + eps) ** 0.5
                rstd[i] = rs
                output[i] = weight * (row - mu) * rs + bias

        ctx.save_for_backward(input, weight, mean, rstd)
        ctx.D = D
        return output.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, mean, rstd = ctx.saved_tensors
        D = ctx.D

        x = input.contiguous().view(-1, D)
        dy = grad_output.contiguous().view(-1, D)
        N = x.shape[0]

        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros(D, device=x.device, dtype=x.dtype)
        grad_bias = torch.zeros(D, device=x.device, dtype=x.dtype)

        if x.is_cuda and _has_gpt2_backend():
            _get_gpt2_backend().repro_layernorm_backward(
                dy, x, weight, mean, rstd,
                grad_input, grad_weight, grad_bias, N, D)
        else:
            for i in range(N):
                mu = mean[i].item()
                rs = rstd[i].item()
                xhat = (x[i] - mu) * rs

                grad_bias += dy[i]
                grad_weight += dy[i] * xhat

                g_dy = weight * dy[i]
                s1 = g_dy.sum().item() / D
                s2 = (g_dy * xhat).sum().item() / D
                grad_input[i] = rs * (g_dy - s1 - xhat * s2)

        return grad_input.view_as(input), grad_weight, grad_bias, None


class ReproLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return _LayerNormFn.apply(x, self.weight, self.bias, self.eps)


# ============================================================================
# GELU
# ============================================================================

class _GELUFn(Function):
    @staticmethod
    def forward(ctx, input):
        x = input.contiguous()
        output = torch.empty_like(x)

        if x.is_cuda and _has_gpt2_backend():
            _get_gpt2_backend().repro_gelu_forward(x, output, x.numel())
        else:
            sqrt_2_pi = math.sqrt(2.0 / math.pi)
            coeff = 0.044715
            inner = sqrt_2_pi * (x + coeff * x ** 3)
            output = 0.5 * x * (1.0 + torch.tanh(inner))

        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        x = input.contiguous()
        go = grad_output.contiguous()
        grad_input = torch.empty_like(x)

        if x.is_cuda and _has_gpt2_backend():
            _get_gpt2_backend().repro_gelu_backward(go, x, grad_input, x.numel())
        else:
            sqrt_2_pi = math.sqrt(2.0 / math.pi)
            coeff = 0.044715
            x2 = x * x
            inner = sqrt_2_pi * (x + coeff * x * x2)
            t = torch.tanh(inner)
            d_inner = sqrt_2_pi * (1.0 + 3.0 * coeff * x2)
            d_gelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
            grad_input = go * d_gelu

        return grad_input


class ReproGELU(torch.nn.Module):
    def forward(self, x):
        return _GELUFn.apply(x)


# ============================================================================
# Dropout (Philox counter-based)
# ============================================================================

class _DropoutFn(Function):
    @staticmethod
    def forward(ctx, input, drop_prob, seed, step_counter, training):
        if not training or drop_prob == 0.0:
            return input

        x = input.contiguous()
        output = torch.empty_like(x)

        if x.is_cuda and _has_gpt2_backend():
            _get_gpt2_backend().repro_dropout_forward(
                x, output, x.numel(), drop_prob, seed, step_counter)
        else:
            scale = 1.0 / (1.0 - drop_prob)
            gen = torch.Generator()
            gen.manual_seed(seed ^ step_counter)
            mask = torch.bernoulli(
                torch.full_like(x, 1.0 - drop_prob), generator=gen)
            output = x * mask * scale

        ctx.drop_prob = drop_prob
        ctx.seed = seed
        ctx.step_counter = step_counter
        return output

    @staticmethod
    def backward(ctx, grad_output):
        go = grad_output.contiguous()
        grad_input = torch.empty_like(go)

        if go.is_cuda and _has_gpt2_backend():
            _get_gpt2_backend().repro_dropout_backward(
                go, grad_input, go.numel(),
                ctx.drop_prob, ctx.seed, ctx.step_counter)
        else:
            scale = 1.0 / (1.0 - ctx.drop_prob)
            gen = torch.Generator()
            gen.manual_seed(ctx.seed ^ ctx.step_counter)
            mask = torch.bernoulli(
                torch.full_like(go, 1.0 - ctx.drop_prob), generator=gen)
            grad_input = go * mask * scale

        return grad_input, None, None, None, None


class ReproDropout(torch.nn.Module):
    def __init__(self, p=0.1, seed=42):
        super().__init__()
        self.p = p
        self.seed = seed
        self._step_counter = 0

    def forward(self, x):
        if self.training and self.p > 0:
            self._step_counter += 1
            return _DropoutFn.apply(x, self.p, self.seed, self._step_counter, True)
        return x


# ============================================================================
# Embedding (sort-then-superaccumulator backward)
# ============================================================================

class _EmbeddingFn(Function):
    @staticmethod
    def forward(ctx, weight, indices):
        N = indices.numel()
        D = weight.shape[1]
        V = weight.shape[0]

        flat_idx = indices.contiguous().view(-1).to(torch.int32)
        output = torch.empty(N, D, device=weight.device, dtype=weight.dtype)

        if weight.is_cuda and _has_gpt2_backend():
            _get_gpt2_backend().repro_embedding_forward(weight, flat_idx, output, N, D)
        else:
            for i in range(N):
                output[i] = weight[flat_idx[i]]

        ctx.save_for_backward(flat_idx)
        ctx.V = V
        ctx.D = D
        ctx.N = N
        return output.view(*indices.shape, D)

    @staticmethod
    def backward(ctx, grad_output):
        (flat_idx,) = ctx.saved_tensors
        V, D, N = ctx.V, ctx.D, ctx.N

        grad_weight = torch.zeros(V, D, device=grad_output.device,
                                  dtype=grad_output.dtype)
        go = grad_output.contiguous().view(N, D)

        if go.is_cuda and _has_gpt2_backend():
            sorted_indices = torch.empty(N, dtype=torch.int32, device=go.device)
            sorted_positions = torch.empty(N, dtype=torch.int32, device=go.device)
            sort_workspace = torch.zeros(V, dtype=torch.int32, device=go.device)

            _get_gpt2_backend().repro_embedding_backward(
                go, flat_idx, grad_weight,
                sorted_indices, sorted_positions, sort_workspace,
                N, D, V)
        else:
            idx_np = flat_idx.cpu().numpy()
            order = idx_np.argsort(kind='stable')
            for pos in order:
                grad_weight[flat_idx[pos]] += go[pos]

        return grad_weight, None


class ReproEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim))

    def forward(self, indices):
        return _EmbeddingFn.apply(self.weight, indices)


# ============================================================================
# Reproducible Softmax (for attention)
# Uses repro_torch's B.softmax for the [rows, cols] case, and the GPT-2
# backend for forward/backward with autograd support.
# ============================================================================

class _SoftmaxFn(Function):
    @staticmethod
    def forward(ctx, input):
        x = input.contiguous()
        output = x.clone()

        if x.is_cuda and _has_gpt2_backend():
            M = 1
            for d in x.shape[:-1]:
                M *= d
            T = x.shape[-1]
            _get_gpt2_backend().repro_softmax_forward(output, M, T)
        else:
            # Use repro_torch's softmax for the 2D case
            if x.dim() == 2:
                output = B.softmax(x)
            else:
                output = torch.softmax(x, dim=-1)

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        go = grad_output.contiguous()
        grad_input = torch.empty_like(go)

        if go.is_cuda and _has_gpt2_backend():
            M = 1
            for d in go.shape[:-1]:
                M *= d
            T = go.shape[-1]
            _get_gpt2_backend().repro_softmax_backward(go, output, grad_input, M, T)
        else:
            dot = (output * go).sum(dim=-1, keepdim=True)
            grad_input = output * (go - dot)

        return grad_input


def repro_softmax(x):
    """Reproducible softmax over the last dimension, with autograd support."""
    return _SoftmaxFn.apply(x)


# ============================================================================
# Causal mask + scale (for attention scores)
# ============================================================================

def repro_causal_mask_scale(scores, scale):
    """Apply causal mask and scale to attention scores.

    scores: [B, H, T, T] or [BH, T, T]
    scale: 1/sqrt(d_k)
    """
    if scores.is_cuda and _has_gpt2_backend():
        if scores.dim() == 4:
            BH = scores.shape[0] * scores.shape[1]
        else:
            BH = scores.shape[0]
        T = scores.shape[-1]
        s = scores.contiguous().view(BH, T, T)
        _get_gpt2_backend().repro_causal_mask_scale(s, BH, T, scale)
        return s.view_as(scores)
    else:
        T = scores.shape[-1]
        mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores * scale
        scores = scores.masked_fill(mask, float('-inf'))
        return scores


# ============================================================================
# Causal Self-Attention (uses repro_torch.Linear for projections)
# ============================================================================

class ReproCausalSelfAttention(torch.nn.Module):
    """Multi-head causal self-attention with reproducible GEMM projections
    and reproducible softmax.

    Uses repro_torch.Linear (Ozaki GEMM) for Q/K/V and output projections,
    matching the same bit-reproducible arithmetic used in the MNIST path.
    """

    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        from repro_torch.layers import Linear as ReproLinear
        self.c_attn = ReproLinear(n_embd, 3 * n_embd)
        self.c_proj = ReproLinear(n_embd, n_embd)
        self.attn_dropout = ReproDropout(dropout)
        self.resid_dropout = ReproDropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection via reproducible Linear (Ozaki GEMM)
        qkv = self.c_attn(x.view(B * T, C)).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Reshape for batched attention: [B, H, T, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention scores: S = Q @ K^T
        # Use repro_torch GEMM for the batched matmul
        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]

        # Apply causal mask and scale
        att_flat = att.reshape(-1, T, T)
        att_flat = repro_causal_mask_scale(att_flat, scale)
        att = att_flat.view(B, self.n_head, T, T)

        # Reproducible softmax (per-row)
        att_2d = att.reshape(-1, T)
        att_2d = repro_softmax(att_2d)
        att = att_2d.view(B, self.n_head, T, T)

        # Attention dropout
        att = self.attn_dropout(att)

        # Output: O = softmax(QK^T) @ V
        out = torch.matmul(att, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # Output projection via reproducible Linear
        out = self.c_proj(out.view(B * T, C)).view(B, T, C)
        out = self.resid_dropout(out)
        return out
