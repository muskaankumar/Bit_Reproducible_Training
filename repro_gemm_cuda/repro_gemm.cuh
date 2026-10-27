#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace repro_gemm {

// Reproducible GEMM: C[M,N] = op(A)[M,K] * op(B)[K,N], FP32 in and out, row-major.
//
// Method (see README.md): each row of op(A) and column of op(B) gets a power-of-two
// scale from its max |value|. Every input is rounded once to a fixed-point integer
// with 7*S bits ("S slices"), split exactly into S int8 digits, and the digit
// products are computed with int8 GEMMs and combined exactly in INT64. The exact
// integer result is rounded once to FP32.
//
// The output is defined by that math alone, so it is bit-identical across runs,
// split_k values, algorithms, and GPUs, and equal to a CPU implementation of the
// same definition. Larger S is more accurate and slower:
//   S=1: 7-bit inputs (plain int8 quantization), 1 int8 GEMM
//   S=2: 14-bit inputs, 4 int8 GEMMs
//   S=3: 21-bit inputs, 9 int8 GEMMs (default; close to FP32 input precision)
// Different S give different (not bit-identical) results; pick one per training run.
// Inputs must be finite. Do not build with --use_fast_math / -ftz=true / -prec-div=false.

enum class Algo {
    Auto,        // TensorCore on sm_72+, otherwise Tiled
    Reference,   // simple one-thread-per-output INT64 kernel; slow, for verification
    Tiled,       // shared-memory tiles, dp4a (sm_61+), INT32 per-level accumulation
    TensorCore,  // shared-memory tiles, INT8 WMMA Tensor Cores (sm_72+)
};

struct Config {
    int split_k = 1;          // >= 1. May be raised automatically to keep INT32 sums exact.
    int tile_m = 16;          // ignored; kept for source compatibility
    int tile_n = 16;          // ignored
    int tile_k = 32;          // ignored
    Algo algo = Algo::Auto;
    int slices = 3;           // S, 1..MAX_SLICES
    bool trans_a = false;     // false: A is MxK row-major.  true: A is KxM row-major (use A^T)
    bool trans_b = false;     // false: B is KxN row-major.  true: B is NxK row-major (use B^T)
};

// What will actually run for a given problem on the current device.
struct Plan {
    Algo algo;               // resolved (never Auto)
    int split_k;             // effective split count
    size_t workspace_bytes;  // size needed by repro_gemm_fp32_ws
};

// cudaErrorNotSupported: cfg.algo cannot run on the current device.
// cudaErrorInvalidValue: bad sizes/slices, or K too large for exact INT64 at this S
//                        (S=3 allows K up to about 1,000,000).
cudaError_t make_plan(int M, int N, int K, Config cfg, Plan* plan);

// Asynchronous, allocation-free version: use this in training loops.
// workspace must be at least plan.workspace_bytes (cudaMalloc alignment).
// Workspace size depends only on (M, N, K, cfg), so allocate it once per shape.
// If stage_events is non-null it must hold 4 events; they are recorded on `stream`
// before quantization, after quantization, after the GEMM, and after the final step.
cudaError_t repro_gemm_fp32_ws(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    Config cfg,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream = nullptr,
    const cudaEvent_t* stage_events = nullptr);

// Convenience version: allocates the workspace, runs, synchronizes, frees.
cudaError_t repro_gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    Config cfg = Config(),
    cudaStream_t stream = nullptr);

const char* algo_name(Algo a);

} // namespace repro_gemm
