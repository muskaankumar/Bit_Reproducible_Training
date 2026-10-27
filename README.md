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
precision removes the GPU-to-GPU differences that exist in the single-precision
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
