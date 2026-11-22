"""GPT-2 small (124M parameters) built on reproducible primitives.

Architecture (matching OpenAI's GPT-2 small):
  - vocab_size: 50257
  - n_layer: 12
  - n_head: 12
  - n_embd: 768
  - block_size: 1024
  - activation: GELU (tanh approximation)

Reproducible operations used:
  - Linear layers: repro_torch.Linear (Ozaki GEMM, same as MNIST path)
  - Cross-entropy: repro_torch.cross_entropy (superaccumulator reduction)
  - LayerNorm: repro_layernorm (superaccumulator mean/var)
  - GELU: repro_gelu (correctly rounded exp in tanh approx, --fmad=false)
  - Dropout: repro_dropout (Philox counter-based, mask regenerated in backward)
  - Embeddings: repro_embedding (gather forward, sort+superacc backward)
  - Softmax: repro_softmax (repro_exp + superaccumulator denominator)
  - Adam: repro_torch.Adam (AdamClock + superaccumulator updates)
  - FiniteGuard: scans inputs for NaN/Inf before every GEMM (from repro_torch)
"""

import torch
import torch.nn as nn

# Base reproducible primitives from repro_torch (shared with MNIST)
from repro_torch.layers import Linear as ReproLinear
from repro_torch.layers import cross_entropy, guard

# GPT-2-specific reproducible layers
from gpt2_repro.layers import (
    ReproCausalSelfAttention,
    ReproDropout,
    ReproEmbedding,
    ReproGELU,
    ReproLayerNorm,
)


class MLP(nn.Module):
    """Feed-forward network with reproducible GELU activation.

    Uses repro_torch.Linear (Ozaki GEMM) for both projections, matching
    the same GEMM path used in the MNIST LeNet linear layers.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = ReproLinear(config.n_embd, 4 * config.n_embd)
        self.gelu = ReproGELU()
        self.c_proj = ReproLinear(4 * config.n_embd, config.n_embd)
        self.dropout = ReproDropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        # Flatten to [B*T, C] for the reproducible Linear (Ozaki GEMM expects 2D)
        h = self.c_fc(x.view(B * T, C))       # [B*T, 4C]
        h = self.gelu(h)
        h = self.c_proj(h)                     # [B*T, C]
        h = self.dropout(h.view(B, T, C))
        return h


class Block(nn.Module):
    """Transformer block: LN -> Attention -> residual -> LN -> MLP -> residual."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = ReproLayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = ReproCausalSelfAttention(
            config.n_embd, config.n_head, config.dropout)
        self.ln_2 = ReproLayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Config:
    """GPT-2 small configuration."""

    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 50257)
        self.block_size = kwargs.get("block_size", 1024)
        self.n_layer = kwargs.get("n_layer", 12)
        self.n_head = kwargs.get("n_head", 12)
        self.n_embd = kwargs.get("n_embd", 768)
        self.dropout = kwargs.get("dropout", 0.1)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-5)


class GPT2(nn.Module):
    """GPT-2 built on reproducible primitives.

    Every operation that involves a floating-point sum goes through either:
      1. repro_torch's Ozaki GEMM (Linear projections, same as MNIST)
      2. Superaccumulator reductions (cross-entropy, layernorm, embedding backward)
      3. Correctly rounded element-wise ops (GELU, --fmad=false)
      4. Philox-based dropout (deterministic mask from counter)

    Non-rounding ops (embedding gather, causal mask, residual add) use plain
    PyTorch — they are exact or involve a single IEEE addition.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token + position embeddings (reproducible backward via sort+superacc)
        self.wte = ReproEmbedding(config.vocab_size, config.n_embd)
        self.wpe = ReproEmbedding(config.block_size, config.n_embd)
        self.drop = ReproDropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = ReproLayerNorm(config.n_embd, eps=config.layer_norm_eps)

        # LM head: reproducible Linear (Ozaki GEMM), weight-tied with wte
        self.lm_head = ReproLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.wte.weight.numel()  # subtract tied weight
        print(f"GPT-2 model: {n_params / 1e6:.1f}M parameters "
              f"(excluding tied embeddings)")

    def forward(self, idx, targets=None):
        """
        idx: [B, T] token indices
        targets: [B, T] target indices (optional, for loss computation)

        Returns (logits, loss).  loss is computed with repro_torch.cross_entropy
        (superaccumulator reduction), matching the MNIST cross-entropy path.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} > block size {self.config.block_size}"

        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)                    # [B, T, C]
        pos_emb = self.wpe(pos)                    # [T, C]
        x = tok_emb + pos_emb.unsqueeze(0)         # [B, T, C] + [1, T, C]
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # LM head: flatten to 2D for the reproducible Linear
        logits = self.lm_head(x.view(B * T, -1)).view(B, T, -1)

        if targets is not None:
            # Reproducible cross-entropy (same as MNIST path)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1))
            return logits, loss
        else:
            return logits, None

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
