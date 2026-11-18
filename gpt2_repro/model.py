"""GPT-2 small (124M parameters) built on reproducible primitives.

Architecture (matching OpenAI's GPT-2 small):
  - vocab_size: 50257
  - n_layer: 12
  - n_head: 12
  - n_embd: 768
  - block_size: 1024
  - activation: GELU (tanh approximation)

Reproducible operations used:
  - LayerNorm: repro_layernorm (superaccumulator mean/var)
  - GELU: repro_gelu (correctly rounded exp in tanh approx)
  - Dropout: repro_dropout (Philox counter-based, mask regenerated in backward)
  - Embeddings: repro_embedding (gather forward, sort+superacc backward)
  - Softmax: repro_softmax (repro_exp + superaccumulator denominator)
  - Linear layers: standard nn.Linear (future: Ozaki GEMM)
  - Cross-entropy: standard (future: reproducible reduction)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2_repro.layers import (
    ReproLayerNorm,
    ReproGELU,
    ReproDropout,
    ReproEmbedding,
    repro_softmax,
    repro_causal_mask_scale,
)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with reproducible softmax."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = ReproDropout(config.dropout)
        self.resid_dropout = ReproDropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)  # [B, T, 3*C]
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each [B, T, H, D]

        # Reshape for batched attention: [B, H, T, D]
        q = q.permute(0, 2, 1, 3)  # [B, H, T, D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention scores: S = Q @ K^T / sqrt(d_k)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]

        # Apply causal mask and scale using our reproducible kernel
        att_flat = att.reshape(-1, T, T)  # [B*H, T, T]
        att_flat = repro_causal_mask_scale(att_flat, scale)
        att = att_flat.view(B, self.n_head, T, T)

        # Reproducible softmax (per-row)
        att_2d = att.reshape(-1, T)  # [B*H*T, T]
        att_2d = repro_softmax(att_2d)
        att = att_2d.view(B, self.n_head, T, T)

        # Attention dropout
        att = self.attn_dropout(att)

        # Output: O = softmax(QK^T) @ V
        out = torch.matmul(att, v)  # [B, H, T, D]

        # Reshape back: [B, T, C]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # Output projection
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out


class MLP(nn.Module):
    """Feed-forward network with reproducible GELU activation."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = ReproGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = ReproDropout(config.dropout)

    def forward(self, x):
        h = self.c_fc(x)       # [B, T, 4C]
        h = self.gelu(h)
        h = self.c_proj(h)     # [B, T, C]
        h = self.dropout(h)
        return h


class Block(nn.Module):
    """Transformer block: LN -> Attention -> residual -> LN -> MLP -> residual."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = ReproLayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
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
    """GPT-2 built on reproducible primitives."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token + position embeddings (reproducible backward via sort+superacc)
        self.wte = ReproEmbedding(config.vocab_size, config.n_embd)
        self.wpe = ReproEmbedding(config.block_size, config.n_embd)
        self.drop = ReproDropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = ReproLayerNorm(config.n_embd, eps=config.layer_norm_eps)

        # LM head (weight-tied with wte)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.wte.weight.numel()  # subtract tied weight
        print(f"GPT-2 model: {n_params/1e6:.1f}M parameters (excluding tied embeddings)")

    def forward(self, idx, targets=None):
        """
        idx: [B, T] token indices
        targets: [B, T] target indices (optional, for loss computation)
        """
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} > block size {self.config.block_size}"

        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)       # [B, T, C]
        pos_emb = self.wpe(pos)       # [T, C]
        x = tok_emb + pos_emb.unsqueeze(0)  # [B, T, C] + [1, T, C]
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # LM head
        logits = self.lm_head(x)  # [B, T, vocab_size]

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
            return logits, loss
        else:
            return logits, None

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
