// Bit-reproducible LayerNorm for FP32.
//
// Given x[rows, cols], gamma[cols], beta[cols]:
//   mean[r]  = exact_sum(x[r,:]) / cols           (superaccumulator, one rounding)
//   var[r]   = exact_sum((x[r,:] - mean[r])^2) / cols  (superaccumulator, one rounding)
//   y[r,c]   = gamma[c] * (x[r,c] - mean[r]) * rsqrt(var[r] + eps) + beta[c]
//
// The key insight: mean and variance use the superaccumulator from repro_sum,
// so they are order-independent and correctly rounded. The per-element operations
// (subtract mean, multiply by rsqrt, scale by gamma, add beta) are each a single
// IEEE FP32 operation, so they are deterministic by construction.
//
// For the backward pass:
//   dx, dgamma, dbeta all use superaccumulator reductions where needed.
//
// Depends on: repro_sum_math.cuh (superaccumulator)
//
// NOTE on rsqrt: CUDA's rsqrtf is not correctly rounded on all architectures.
// We use __fdiv_rn(1.0f, sqrtf(x)) which gives correctly rounded 1/sqrt(x)
// when sqrtf is correctly rounded (IEEE 754 guarantees sqrt is correctly rounded).
// __fdiv_rn is round-to-nearest-even division, also correctly rounded.
// So the composition is at most 1 ulp from the true value, and — crucially —
// gives identical bits on every GPU because both operations are IEEE-mandated.

#pragma once

#include "repro_sum_math.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace repro_layernorm {

// Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Also saves mean and rstd for backward.
//
// x:      [rows, cols]  input
// gamma:  [cols]        scale
// beta:   [cols]        bias
// y:      [rows, cols]  output
// mean:   [rows]        saved for backward
// rstd:   [rows]        saved for backward (1/sqrt(var+eps))
cudaError_t layernorm_forward(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* rstd,
    int rows, int cols, float eps, cudaStream_t stream);

// Backward: given dy, compute dx, dgamma, dbeta.
//
// dgamma[c] = sum_r dy[r,c] * (x[r,c] - mean[r]) * rstd[r]   (superaccumulator sum over rows)
// dbeta[c]  = sum_r dy[r,c]                                    (superaccumulator sum over rows)
// dx[r,c]   = rstd[r] * (dy[r,c] - mean_dy[r] - (x[r,c]-mean[r])*rstd[r]*mean_dyxhat[r])
//   where mean_dy[r]     = sum_c dy[r,c] / cols                (superaccumulator)
//         mean_dyxhat[r] = sum_c dy[r,c]*(x[r,c]-mean[r])*rstd[r] / cols  (superaccumulator)
cudaError_t layernorm_backward(
    const float* dy, const float* x, const float* gamma,
    const float* mean, const float* rstd,
    float* dx, float* dgamma, float* dbeta,
    int rows, int cols, cudaStream_t stream);

// CPU reference (same definition, for verification)
void cpu_layernorm_forward(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* rstd,
    int rows, int cols, float eps);

void cpu_layernorm_backward(
    const float* dy, const float* x, const float* gamma,
    const float* mean, const float* rstd,
    float* dx, float* dgamma, float* dbeta,
    int rows, int cols);

} // namespace repro_layernorm
