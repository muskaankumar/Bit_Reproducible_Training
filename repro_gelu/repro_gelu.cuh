// Bit-reproducible GELU activation for FP32.
//
// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// The challenge: erff() is not guaranteed to be correctly rounded across GPUs,
// and different GPU architectures may use different polynomial approximations.
//
// Strategy options (implement and benchmark both):
//
// Option A: Tanh approximation (used by PyTorch's "gelu_tanh"):
//   GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   Then implement tanh via our correctly rounded exp:
//     tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)  = 1 - 2/(exp(2x) + 1)
//   Each step uses a single IEEE operation or our correctly rounded exp.
//   This is what GPT-2 actually uses (the "new GELU").
//
// Option B: Lookup table + polynomial, like we did for exp/log.
//   More work, but gives the true erf-based GELU.
//
// For GPT-2, Option A is the right choice: it matches the original model's definition
// and every step can be made bit-reproducible with our existing correctly rounded exp.
//
// Backward:
//   d/dx GELU(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
//   where z = sqrt(2/pi) * (x + 0.044715 * x^3)
//   Again, each step is a single IEEE FP32 op or our correctly rounded exp.

#pragma once

#include <cuda_runtime.h>

namespace repro_gelu {

// Forward: y[i] = gelu(x[i]) for n elements.
// Uses the tanh approximation with correctly rounded exp.
cudaError_t gelu_forward(const float* x, float* y, size_t n, cudaStream_t stream);

// Backward: dx[i] = dy[i] * gelu'(x[i]).
// x is the original input (saved from forward).
cudaError_t gelu_backward(const float* dy, const float* x, float* dx, size_t n, cudaStream_t stream);

// CPU reference
void cpu_gelu_forward(const float* x, float* y, size_t n);
void cpu_gelu_backward(const float* dy, const float* x, float* dx, size_t n);

} // namespace repro_gelu
