# Bit-Reproducible MNIST Training Pipeline

A LeNet-5 training pipeline on MNIST where every weight is made **bit-identical**
across runs, by routing every floating-point summation through either an
integer-exact GEMM (the Ozaki scheme from `repro_gemm_cuda/`) or a
double-precision intermediate. This README documents how reproducibility is
achieved, what was actually tested, and an honest account of a speed
comparison against Microsoft's RepDL that did **not** show the expected
speedup — plus a diagnosis of why.

---

## 1. How bit-reproducibility is achieved, component by component

Floating-point addition is not associative: `(a+b)+c` can differ in the last
bit from `a+(b+c)`. On a GPU, the order threads add partial sums together
depends on block size, scheduling, and hardware — which is the standard
reason GPU training isn't bit-reproducible run to run. This pipeline removes
every place that ambiguity could enter.

### Conv2d and Linear layers (the GEMMs)

All matrix multiplies — forward pass, input gradient, and weight gradient —
go through `repro_gemm` (`repro_gemm_cuda/repro_gemm.cu`), which:

1. Scales each row of A and column of B by a power of two based on its max
   absolute value.
2. Rounds every input **once** to a fixed-point integer (7·S bits, S = 1–3
   slices).
3. Multiplies and accumulates in **int8 → int64**. Integer addition is
   associative, so the result does not depend on how the sum is split across
   threads, split-K value, or tiling strategy.
4. Rounds the exact integer result to FP32 **once**, at the very end.

Because step 3 is exact regardless of grouping, this is what lets the GEMM be
parallelized arbitrarily (unlike a literal sequential FMA chain) while
staying bit-identical.

`Conv2d` is implemented as `im2col` (unfold the image into a matrix) followed
by exactly this GEMM, then `col2im` for the input gradient (accumulated in
double precision, since the number of overlapping contributions per pixel is
small and fixed by the kernel size).

### Cross-entropy loss

`log_softmax` and the per-example negative log-likelihood are computed with
`(float)exp((double)x)`-style intermediates: the max-subtraction, `exp`, and
`log` all happen in double precision inside the kernel, and only the final
answer is rounded to float once. This matches the RepDL paper's strategy —
CUDA's double-precision `exp`/`log` are correctly rounded and identical
across GPU generations for a fixed CUDA version, so routing through double
precision removes the GPU-to-GPU differences that exist in the *single*precision
approximate math library functions (`expf`, `logf`, etc., which are
not correctly rounded and do vary GPU to GPU in their low bits).

### Adam optimizer

- `sqrt` (for the second-moment normalization) and the final division both go
  through double-precision casts (`(a.to(float64) / b.to(float64)).to(float32)`).
- The momentum updates themselves (`m = β₁·m + (1−β₁)·grad`) are plain
  element-wise float operations with a fixed operand order — no reduction
  happens here, so they are already exact/reproducible without any special
  handling.

### ReLU, MaxPool, Flatten

Unchanged, plain PyTorch. These only do comparisons and index rearrangement,
never float addition, so they're reproducible for free.

### Bias gradients

Summed over the batch dimension in double precision
(`grad_output.to(torch.float64).sum(...)`) before rounding back to float, to
avoid order-dependent partial-sum rounding across the batch.

---

## 2. Experiments run

### 2.1 Same-GPU, repeated-run reproducibility

Ran `train_mnist.py --epochs 10 --seed 42` **twice** on the same A100 (NYU
HPC, `c12m85-a100-1` partition, CUDA 12.6), and diffed the per-epoch SHA-256
hashes of all parameters and optimizer state:

```
diff run1_hashes.txt run2_hashes.txt
```

**Result: PASS.** `diff` produced no output — every epoch's parameter hash
and optimizer-state hash matched exactly between the two runs, confirming
bit-identical training across repeated runs on the same GPU.

Not yet run: the stronger cross-GPU version of this check (e.g. A100 vs.
L4), which would confirm reproducibility across GPU architectures rather
than just across repeated runs on the same one.

### 2.2 Timing comparison vs. RepDL

**Setup:** Both pipelines trained the identical LeNet-5 architecture
(`Conv2d(1,6,5,pad=2) → ReLU → MaxPool → Conv2d(6,16,5) → ReLU → MaxPool →
FC(400,120) → ReLU → FC(120,84) → ReLU → FC(84,10)`) on MNIST, batch size
100, 10 epochs, Adam optimizer (lr adjusted to match: 0.001), on the same
NYU HPC A100 node, same CUDA 12.6 container.

**Result:**

| | Avg epoch time |
|---|---|
| RepDL (`examples/mnist_trainig.py`, patched for matching hyperparameters) | **9.04 s/epoch** |
| This pipeline (`train_mnist.py`) | **30.97 s/epoch** |

**Speedup: 0.29×ꟷ i.e. this pipeline is currently ~3.4× *slower* than RepDL,
not faster.** The original hypothesis (integer-GEMM split-K parallelism
should beat RepDL's sequential-chain GEMM) was not borne out by this
end-to-end measurement.

---

## 3. Why the expected speedup didn't show up (diagnosis)

The algorithmic argument — that integer addition's associativity permits
splitting a dot product across many threads/kernels, which a literal
sequential FMA chain cannot do — is still correct at the level of a single
large GEMM (this is exactly what Experiment 1 in the top-level project
README measured directly: `repro_gemm` **is** faster than cuBLAS FP32 on
large square matrices, and bit-reproducible). The slowdown shows up at the
**training pipeline level**, for reasons that are about implementation, not
the underlying math:

1. **Per-image Python loop in `ReproConv2dFn`.** The current
   implementation (`repro_nn.py`) loops `for b in range(B): ...`, launching a
   separate `im2col` + `repro_gemm` call *per image* in the batch. With batch
   size 100, that's 100 small GEMM calls per conv layer, each paying
   kernel-launch overhead and the quantization-kernel fixed cost, instead of
   one batched GEMM. RepDL's `from_torch_module` presumably handles the batch
   dimension without this per-sample Python-level loop.

2. **GEMM shapes are small relative to per-call overhead.** LeNet is tiny:
   `repro_gemm`'s quantization, scale-exponent, and digit-splitting kernels
   have a largely fixed per-call cost that only pays for itself at larger M,
   N, K. For a 100-sample MNIST batch through a 6-, then 16-channel conv and
   three small FC layers, the relative overhead per call is much higher than
   in the 1024³/4096³ GEMMs the base `repro_gemm_cuda` benchmarks were run
   at.

3. **Workspace/plan recomputation.** `make_plan` and the workspace lookup run
   on every call; for 100 tiny per-image calls per layer per batch, this adds
   up as pure overhead that a single batched call would pay once.

### Proposed fix (not yet implemented/tested)

Batch the `im2col` across the whole batch dimension so each conv layer does
**one** GEMM shaped `[Cout, Cin·kH·kW] @ [Cin·kH·kW, B·outH·outW]` instead of
`B` separate `[Cout, Cin·kH·kW] @ [Cin·kH·kW, outH·outW]` calls, then reshape
back. This should let a single, larger `repro_gemm` call amortize the
per-call overhead and let split-K parallelism actually engage — matching the
conditions under which the base GEMM benchmarks showed their improvement.
This fix has not yet been implemented or re-benchmarked.

---

## 4. Honest status

- ✅ Bit-reproducibility mechanism is implemented for every relevant op
  (GEMM, conv, cross-entropy, Adam) as described in Section 1.
- ✅ Same-GPU repeat-run hash match: **PASS** — see Section 2.1.
- ❌ Cross-GPU hash match (e.g. A100 vs L4): not yet run.
- ❌ Speedup over RepDL: **not achieved** in the current implementation — see
  Section 2.2/3 for the measured 3.4× slowdown and diagnosis.

## 5. Files

| File | Purpose |
|------|---------|
| `repro_gemm_ext.cu` | PyTorch C++ extension wrapping `repro_gemm`; reproducible element-wise ops (exp/log/sqrt/tanh/div, softmax, cross-entropy); im2col/col2im |
| `repro_nn.py` | `ReproLinear`, `ReproConv2d` (currently per-image loop — see Section 3), `ReproCrossEntropyLoss`, `ReproAdam` |
| `train_mnist.py` | Training script with per-epoch SHA-256 parameter/optimizer hashing |
| `run_repro_mnist.sh` | Convenience script: run twice, diff hashes |
| `setup.py` | Build script for the CUDA extension |
