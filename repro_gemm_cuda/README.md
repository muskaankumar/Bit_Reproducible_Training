# Reproducible GEMM with exact integer slices (Ozaki-style) — CUDA

`C = op(A) · op(B)` for FP32 row-major matrices, with output bits that depend only on the input bits and the number of slices `S`, not on the GPU, split-K, kernel, or run.

## What it computes

For each row `i` of `op(A)` and column `j` of `op(B)`:

```
eA[i] = smallest e with 2^e >= max_k |A[i,k]| / 127          (power-of-two scale; 0 if the row is zero)
X[i,k] = round_to_nearest_even(A[i,k] / 2^(eA[i] - 7(S-1)))   |X| <= 127 * 2^(7(S-1))   (7*S-bit fixed point)
Y[k,j] = same for B with eB[j]
T[i,j] = sum_k X[i,k] * Y[k,j]                                 exact integer
C[i,j] = round_to_float( T[i,j] * 2^(eA[i] + eB[j] - 14(S-1)) )  one correctly rounded conversion
```

The functions are in [`repro_gemm_math.cuh`](repro_gemm_math.cuh) (`scale_exponent`, `fixed_point`, `round_to_float`) and are shared by the GPU kernels and the CPU reference in the tests.

| S | Bits per input | int8 GEMMs | Notes |
|---|---|---|---|
| 1 | 7 | 1 | plain int8 quantization; identical bits to the original single-slice version |
| 2 | 14 | 4 | |
| 3 (default) | 21 | 9 | close to FP32's 24-bit inputs; the dot product itself has no rounding error |

Inputs are rounded relative to their row/column maximum, so values much smaller than the max keep fewer than `7*S` significant bits. Different `S` values give different results; pick one per training run.

## How the GPU computes it

1. **Quantize.** Each operand is converted to `X` and split exactly into `S` int8 digits with integer operations: `X = d0·128^(S-1) + … + d(S-1)`, `|d0| <= 127`, other digits in `[-64, 63]`. Digits are stored as `S` planes in zero-padded, k-contiguous buffers (`op(A) -> [S][Mp][Kp]`, `op(B) -> [S][Np][Kp]`). Transposed inputs only change how the quantizer reads memory.
2. **Integer GEMM.** For each digit pair `(p, q)` the kernel multiplies digit planes with int8 dp4a or Tensor Core instructions. Pairs with the same level `p+q` share one INT32 accumulator.
3. **Combine.** `T = Σ_level acc_level · 2^(7(2S-2-level))` in INT64 (exactly `Σ X·Y`), split-K partials summed in INT64, then `round_to_float` once. With `split_k == 1` this happens in the GEMM epilogue (no extra kernel).

### Overflow bounds (enforced by `make_plan`)

- **INT32 level accumulators:** a level has at most `S` pairs, each digit product is at most `127² = 16129`, and one split covers at most `131072/S` values of k, so a level sum is at most `131072 · 16129 = 2,114,060,288 < 2^31 − 1`. `make_plan` raises `split_k` automatically if K is larger.
- **INT64 totals:** every partial and final sum is at most `K · max|X|²`. `make_plan` requires this to be `<= 2^62`. For S=3 that allows K up to about 1,000,000 (it returns `cudaErrorInvalidValue` beyond).

## Bit-identical: what is guaranteed

Every step is exact or a single IEEE rounding with a fixed mode: `fmaxf` (order-independent), `frexpf`/`ldexpf` (exact), `m/127.0f` (correctly rounded), `__float2int_rn` of an exactly scaled value, integer digit split, integer products and sums, and one integer-implemented round-to-nearest-even to FP32. There is no floating-point addition anywhere. Therefore, for a fixed `S` and finite inputs:

- identical across runs, processes, `split_k`, and algorithms (`Reference`, `Tiled`, `TensorCore`);
- identical across GPU models and architectures;
- identical to the CPU implementation in [`repro_test_util.h`](repro_test_util.h) (`cpu_reference`), which the tests check element by element.

Conditions: finite inputs (NaN/Inf unsupported); do not compile with `--use_fast_math`, `-ftz=true`, or `-prec-div=false`. This does not make the rest of training deterministic (gradient all-reduce order, optimizer, data order, dropout RNG, atomics).

## Algorithms

| `Config::algo` | Kernel | Needs |
|---|---|---|
| `Reference` | one thread per output, `Σ X·Y` in INT64 directly (no digits). Slow; for verification | any GPU |
| `Tiled` | 64x64x32 shared-memory tiles, `dp4a` (4 int8 products per instruction) | sm_61+ for `dp4a` (slower fallback below) |
| `TensorCore` | 64x64x32 shared-memory tiles, INT8 WMMA 16x16x16 | sm_72+ |
| `Auto` (default) | `TensorCore` if available, otherwise `Tiled` | |

## Using it from training code

Compile `repro_gemm.cu` into your project (or link the `repro_gemm` CMake target) and include `repro_gemm.cuh`.

```cpp
#include "repro_gemm.cuh"

repro_gemm::Config cfg;      // defaults: S=3, algo=Auto, split_k=1, no transposes
cfg.slices = 3;

// Once per shape: workspace size depends only on (M, N, K, cfg).
repro_gemm::Plan plan;
repro_gemm::make_plan(M, N, K, cfg, &plan);
void* ws;  cudaMalloc(&ws, plan.workspace_bytes);

// Every step: asynchronous on `stream`, no allocation, no synchronization.
repro_gemm::repro_gemm_fp32_ws(A, B, C, M, N, K, cfg, ws, plan.workspace_bytes, stream);
```

A linear layer `Y = X·Wᵀ` (PyTorch weight layout `W[out, in]`) needs three GEMMs:

| GEMM | Shapes | Call |
|---|---|---|
| forward `Y = X·Wᵀ` | X `[B, in]`, W `[out, in]` | `M=B, N=out, K=in`, `trans_b = true` |
| input grad `dX = dY·W` | dY `[B, out]`, W `[out, in]` | `M=B, N=in, K=out` |
| weight grad `dW = dYᵀ·X` | dY `[B, out]`, X `[B, in]` | `M=out, N=in, K=B`, `trans_a = true` |

Each call returns a `cudaError_t`; check it. `repro_gemm_fp32` is a convenience wrapper that allocates, runs, synchronizes, and frees (fine for tests, slow in a training loop).

## Build and test

Requires the CUDA Toolkit and an NVIDIA GPU. `run_repro.sh` does everything below (and builds with `nvcc` directly if CMake 3.24+ is missing).

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/test_repro_gemm                                        # correctness suite
./build/hash_repro_gemm 256 256 1024 5 1,2,7 12345 normal ref,tiled,tc 3
./build/bench_repro_gemm 4096 4096 4096 1 30 tiled,tc 3
```

- `test_repro_gemm`: for S = 1..3 and every algorithm on this GPU, checks the output against the CPU definition bit for bit, across split_k values, repeated runs, and all four transpose combinations.
- `hash_repro_gemm [M N K runs splits seed dist algos S]`: hashes the output (FNV-1a 64) for every algorithm, split, and run; checks it against the CPU definition (for M·N·K ≤ 2^28); reports accuracy vs FP64. Input distributions: `uniform`, `normal`, `weights`, `wide`, `stress`, or `file:A.bin,B.bin` (raw float32). Inputs come from a portable generator, so a seed gives the same bits on every machine.
- `bench_repro_gemm [M N K split iters algos S]`: workspace allocated once, warm-up, median [min, p90] ms per stage, effective TOPS (`2·M·N·K / time`, comparable to cuBLAS), cuBLAS FP32 SGEMM for comparison, and a bit check against `Reference`.

## Limitations and next optimizations

- GEMM tiles are 64x64 with register prefetch only; 128x128 tiles, double-buffered shared memory, and `ldmatrix`/`mma.sync` would raise Tensor Core utilization.
- Quantization reads each operand twice (max, then quantize).
- `S` up to 3 (INT64). S=4 (28-bit inputs) needs 128-bit accumulation.
- No arbitrary strides (leading dimensions must equal the logical width) or batched GEMM.
