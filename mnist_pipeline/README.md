# Bit-Reproducible MNIST Training Pipeline

Full LeNet-5 training on MNIST where **every weight is bit-identical** across runs, GPUs, and split-K values (same CUDA version required).

## Architecture

| Component | Method | Reproducibility mechanism |
|-----------|--------|--------------------------|
| **Conv2d** (forward + backward) | im2col + `repro_gemm` | Integer-exact Ozaki GEMM |
| **Linear** (forward + backward) | `repro_gemm` with transpose flags | Integer-exact Ozaki GEMM |
| **Cross-entropy loss** | Double-precision log-softmax + NLL | `(float)exp((double)x)` eliminates GPU/CPU differences |
| **Adam optimizer** | Double-precision sqrt, division, bias correction | Element-wise ops through FP64; no reductions |
| **ReLU, MaxPool, Flatten** | Plain PyTorch | Exact (comparisons/indexing only) |
| **Bias gradients** | Double-precision batch sum | Exact accumulation, single round to FP32 |

### Why this is faster than RepDL

RepDL computes each dot product as a sequential chain (`sum = fma(x[0], w[0], sum); sum = fma(x[1], w[1], sum); ...`). This is reproducible because there's only one possible grouping, but it can't parallelize *within* a single sum.

Our `repro_gemm` uses **integer arithmetic** (Ozaki scheme): inputs are quantized to fixed-point, multiplied as int8 on Tensor Cores, and accumulated exactly in int64. Since integer addition is associative, the result doesn't depend on grouping—so we get full parallelism (split-K, tiling, any thread assignment) **and** bit-reproducibility.

## Quick start

```bash
# On a machine with CUDA GPU (CIMS cuda2, etc.)
cd mnist_pipeline

# Build the extension (one-time)
pip install -e .

# Train once
python train_mnist.py --epochs 10

# Verify reproducibility: train twice and diff hashes
./run_repro_mnist.sh 10

# Cross-GPU verification
GPUS=0,1 ./run_repro_mnist.sh 10
```

## Output

Each run writes `results/runN_hashes.txt` with SHA-256 hashes of all parameters and optimizer state after each epoch:

```
epoch=init params=a1b2c3d4...
epoch=1 params=e5f6a7b8... optim=1234abcd... train_loss=0.4321 test_acc=95.20
epoch=2 params=deadbeef... optim=cafebabe... train_loss=0.1234 test_acc=97.80
...
```

If two runs produce identical hash files, training is bit-reproducible.

## What's covered vs. not yet

- ✅ All Linear-layer GEMMs (forward, input gradient, weight gradient)
- ✅ All Conv2d GEMMs via im2col (forward, input gradient, weight gradient)
- ✅ Cross-entropy loss (log-softmax + NLL in double precision)
- ✅ Adam optimizer (all ops through double or exact integer arithmetic)
- ✅ Bias gradients (double-precision batch sum)
- ✅ Deterministic data loading (seeded shuffle, no multiprocess)
- ⏳ Attention QK^T and softmax·V (needs batched GEMM — the kernel supports it)
- ⏳ BatchNorm (not used in this LeNet, but would need reproducible running stats)

## Files

| File | Purpose |
|------|---------|
| `repro_gemm_ext.cu` | PyTorch C++ extension: wraps `repro_gemm`, adds reproducible element-wise ops, im2col/col2im |
| `repro_nn.py` | `ReproLinear`, `ReproConv2d`, `ReproCrossEntropyLoss`, `ReproAdam` with autograd |
| `train_mnist.py` | Full training script with per-epoch hashing |
| `run_repro_mnist.sh` | Convenience: run twice and diff hashes |
| `setup.py` | Build script for the CUDA extension |
