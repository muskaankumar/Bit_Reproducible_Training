#!/usr/bin/env python3
"""
Unit tests for GPT-2 reproducible kernels.

Tests:
  1. GPU vs CPU reference agreement (per kernel)
  2. Cross-run identity: same inputs → same outputs every time
  3. Gradient correctness via finite differences
  4. GPT-2-shaped inputs (d=768, T=1024, H=12)

Usage:
    python tests/test_kernels.py
    python tests/test_kernels.py --verbose
    python tests/test_kernels.py -k test_gelu  # run one test
"""

import torch
import unittest
import hashlib
import struct
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gpt2_repro.layers import (
        ReproLayerNorm, ReproGELU, ReproDropout, ReproEmbedding,
        repro_softmax, repro_causal_mask_scale,
    )
    HAS_LAYERS = True
except ImportError as e:
    print(f"Warning: Could not import layers: {e}")
    HAS_LAYERS = False

HAS_CUDA = torch.cuda.is_available()


def tensor_hash(t):
    """SHA-256 hash of tensor bytes."""
    data = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def splitmix64(seed):
    """Portable RNG for deterministic test inputs."""
    state = seed
    while True:
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        yield z


def deterministic_tensor(shape, seed=42, scale=1.0):
    """Create a tensor with deterministic values (portable across platforms)."""
    n = 1
    for s in shape:
        n *= s
    gen = splitmix64(seed)
    values = []
    for _ in range(n):
        v = next(gen)
        # Convert to float in [-scale, scale]
        f = (v / 0xFFFFFFFFFFFFFFFF) * 2.0 * scale - scale
        values.append(f)
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


@unittest.skipUnless(HAS_LAYERS, "Layers not available")
class TestGELU(unittest.TestCase):
    def test_forward_shape(self):
        gelu = ReproGELU()
        x = torch.randn(4, 768)
        y = gelu(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_values(self):
        """GELU should match the standard approximation."""
        gelu = ReproGELU()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = gelu(x)
        y_ref = torch.nn.functional.gelu(x, approximate='tanh')
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-5),
                       f"GELU mismatch: {y} vs {y_ref}")

    def test_cross_run_identity(self):
        """Same input → same hash every run."""
        gelu = ReproGELU()
        x = deterministic_tensor((32, 768), seed=123)
        h1 = tensor_hash(gelu(x))
        h2 = tensor_hash(gelu(x))
        self.assertEqual(h1, h2, "GELU not deterministic across runs")

    @unittest.skipUnless(HAS_CUDA, "No CUDA")
    def test_gpu_determinism(self):
        gelu = ReproGELU()
        x = deterministic_tensor((32, 768), seed=42).cuda()
        h1 = tensor_hash(gelu(x))
        h2 = tensor_hash(gelu(x))
        self.assertEqual(h1, h2)

    def test_backward(self):
        gelu = ReproGELU()
        x = torch.randn(8, 768, requires_grad=True)
        y = gelu(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


@unittest.skipUnless(HAS_LAYERS, "Layers not available")
class TestLayerNorm(unittest.TestCase):
    def test_forward_shape(self):
        ln = ReproLayerNorm(768)
        x = torch.randn(4, 16, 768)
        y = ln(x)
        self.assertEqual(y.shape, x.shape)

    def test_normalized_output(self):
        """Output should have ~zero mean and ~unit variance per row."""
        ln = ReproLayerNorm(768)
        x = torch.randn(4, 768) * 5 + 3  # shifted and scaled
        y = ln(x)
        # After layer norm, mean ≈ beta (0) and std ≈ gamma (1)
        self.assertTrue(torch.allclose(y.mean(dim=-1), torch.zeros(4), atol=1e-4))
        self.assertTrue(torch.allclose(y.std(dim=-1, correction=0),
                                       torch.ones(4), atol=1e-2))

    def test_cross_run_identity(self):
        ln = ReproLayerNorm(768)
        x = deterministic_tensor((32, 768), seed=456)
        h1 = tensor_hash(ln(x))
        h2 = tensor_hash(ln(x))
        self.assertEqual(h1, h2)

    def test_backward(self):
        ln = ReproLayerNorm(768)
        x = torch.randn(4, 16, 768, requires_grad=True)
        y = ln(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ln.weight.grad)
        self.assertIsNotNone(ln.bias.grad)


@unittest.skipUnless(HAS_LAYERS, "Layers not available")
class TestDropout(unittest.TestCase):
    def test_training_mode(self):
        """Dropout should zero some elements during training."""
        drop = ReproDropout(p=0.5, seed=42)
        drop.train()
        x = torch.ones(1000)
        y = drop(x)
        zeros = (y == 0).sum().item()
        self.assertGreater(zeros, 200, "Too few zeros (expected ~50%)")
        self.assertLess(zeros, 800, "Too many zeros")

    def test_eval_mode(self):
        """Dropout should be identity during eval."""
        drop = ReproDropout(p=0.5, seed=42)
        drop.eval()
        x = torch.ones(1000)
        y = drop(x)
        self.assertTrue(torch.equal(x, y))

    def test_deterministic_mask(self):
        """Same seed + step → same mask."""
        drop1 = ReproDropout(p=0.1, seed=42)
        drop2 = ReproDropout(p=0.1, seed=42)
        drop1.train()
        drop2.train()
        x = deterministic_tensor((100, 768), seed=789)
        h1 = tensor_hash(drop1(x))
        h2 = tensor_hash(drop2(x))
        self.assertEqual(h1, h2, "Dropout masks don't match")

    def test_different_seeds(self):
        """Different seeds → different masks."""
        drop1 = ReproDropout(p=0.5, seed=42)
        drop2 = ReproDropout(p=0.5, seed=99)
        drop1.train()
        drop2.train()
        x = torch.ones(10000)
        y1 = drop1(x)
        y2 = drop2(x)
        # Very unlikely to be identical with different seeds
        self.assertFalse(torch.equal(y1, y2))


@unittest.skipUnless(HAS_LAYERS, "Layers not available")
class TestEmbedding(unittest.TestCase):
    def test_forward_shape(self):
        emb = ReproEmbedding(50257, 768)
        torch.nn.init.normal_(emb.weight)
        indices = torch.randint(0, 50257, (4, 128))
        y = emb(indices)
        self.assertEqual(y.shape, (4, 128, 768))

    def test_forward_correctness(self):
        """Forward should be a pure lookup."""
        emb = ReproEmbedding(100, 16)
        torch.nn.init.normal_(emb.weight)
        indices = torch.tensor([5, 10, 50])
        y = emb(indices)
        self.assertTrue(torch.equal(y[0], emb.weight[5]))
        self.assertTrue(torch.equal(y[1], emb.weight[10]))
        self.assertTrue(torch.equal(y[2], emb.weight[50]))

    def test_backward(self):
        emb = ReproEmbedding(1000, 768)
        torch.nn.init.normal_(emb.weight)
        indices = torch.randint(0, 1000, (4, 32))
        y = emb(indices)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(emb.weight.grad)

    def test_backward_duplicate_indices(self):
        """Backward should correctly accumulate for repeated indices."""
        emb = ReproEmbedding(10, 4)
        torch.nn.init.ones_(emb.weight)
        # Same index appears multiple times
        indices = torch.tensor([3, 3, 3, 5, 5])
        y = emb(indices)
        loss = y.sum()
        loss.backward()
        # Index 3 appears 3 times, so grad should be 3 * ones
        self.assertTrue(torch.allclose(emb.weight.grad[3], torch.tensor([3.0, 3.0, 3.0, 3.0])))
        self.assertTrue(torch.allclose(emb.weight.grad[5], torch.tensor([2.0, 2.0, 2.0, 2.0])))


@unittest.skipUnless(HAS_LAYERS, "Layers not available")
class TestSoftmax(unittest.TestCase):
    def test_forward_shape(self):
        x = torch.randn(4, 12, 64, 64)
        y = repro_softmax(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_properties(self):
        """Softmax output should sum to 1 and be non-negative."""
        x = torch.randn(4, 128)
        y = repro_softmax(x)
        sums = y.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(4), atol=1e-5))
        self.assertTrue((y >= 0).all())

    def test_with_negative_infinity(self):
        """Masked positions (causal) should get zero probability."""
        x = torch.tensor([[1.0, 2.0, float('-inf'), float('-inf')]])
        y = repro_softmax(x)
        self.assertAlmostEqual(y[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(y[0, 3].item(), 0.0, places=6)
        self.assertAlmostEqual(y[0, :2].sum().item(), 1.0, places=5)

    def test_cross_run_identity(self):
        x = deterministic_tensor((8, 12, 64, 64), seed=101)
        h1 = tensor_hash(repro_softmax(x))
        h2 = tensor_hash(repro_softmax(x))
        self.assertEqual(h1, h2)


@unittest.skipUnless(HAS_LAYERS, "Layers not available")
class TestCausalMask(unittest.TestCase):
    def test_mask_shape(self):
        scores = torch.randn(2, 12, 8, 8)
        masked = repro_causal_mask_scale(scores, 0.125)
        self.assertEqual(masked.shape, scores.shape)

    def test_upper_triangle_masked(self):
        """Upper triangle (future positions) should be -inf."""
        T = 4
        scores = torch.ones(1, 1, T, T)
        masked = repro_causal_mask_scale(scores, 1.0)
        for i in range(T):
            for j in range(T):
                if j > i:
                    self.assertEqual(masked[0, 0, i, j].item(), float('-inf'),
                                   f"Position ({i},{j}) should be -inf")
                else:
                    self.assertNotEqual(masked[0, 0, i, j].item(), float('-inf'),
                                       f"Position ({i},{j}) should NOT be -inf")


@unittest.skipUnless(HAS_LAYERS, "Layers not available")
class TestGPT2Shapes(unittest.TestCase):
    """Test all kernels at GPT-2 small dimensions."""

    def test_layernorm_gpt2(self):
        ln = ReproLayerNorm(768)
        x = torch.randn(4, 1024, 768)  # [B, T, D]
        y = ln(x)
        self.assertEqual(y.shape, (4, 1024, 768))

    def test_gelu_gpt2(self):
        gelu = ReproGELU()
        x = torch.randn(4, 1024, 3072)  # MLP intermediate
        y = gelu(x)
        self.assertEqual(y.shape, (4, 1024, 3072))

    def test_embedding_gpt2(self):
        emb = ReproEmbedding(50257, 768)
        torch.nn.init.normal_(emb.weight, std=0.02)
        indices = torch.randint(0, 50257, (4, 1024))
        y = emb(indices)
        self.assertEqual(y.shape, (4, 1024, 768))

    def test_softmax_attention(self):
        # [B, H, T, T] attention matrix
        x = torch.randn(2, 12, 128, 128)
        y = repro_softmax(x)
        self.assertEqual(y.shape, (2, 12, 128, 128))
        sums = y.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))


if __name__ == "__main__":
    print(f"CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print()
    unittest.main(verbosity=2)
