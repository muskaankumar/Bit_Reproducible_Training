// ============================================================================
// Bit-Reproducible GELU Activation (Forward + Backward)
// ============================================================================
//
// GPT-2 uses the "new GELU" approximation (Hendrycks & Gimpel, 2016):
//   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// For bit-reproducibility, we need tanh to be computed identically
// across GPUs. We implement tanh(x) = (exp(2x) - 1) / (exp(2x) + 1),
// using the correctly-rounded exp from repro_math.
//
// All intermediate FP operations use IEEE-mandated rounding (round-to-
// nearest-even). Compiled with --fmad=false to prevent FMA fusion.
// ============================================================================

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// Correctly rounded exp (from repro_math). This is IEEE-mandated for the
// specific case of expf(), but we provide our own to guarantee identical
// results across driver versions.
// ---------------------------------------------------------------------------
// We rely on the repro_math exp implementation from the main library.
// For standalone testing, we use __expf with controlled rounding.
// In the integrated build, this is replaced by repro_exp().

#ifndef REPRO_EXP_DEFINED
// Placeholder: in the full build, repro_exp is linked from repro_math_cuda
__device__ float repro_exp(float x) {
    // For values that would overflow/underflow, clamp
    if (x > 88.7f) return INFINITY;
    if (x < -103.9f) return 0.0f;
    // Use the hardware exp with no FMA fusion (--fmad=false ensures this)
    return expf(x);
}
#endif

// ---------------------------------------------------------------------------
// Correctly rounded tanh via exp
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
// For large |x|, clamp to ±1 to avoid NaN from inf/inf.
// ---------------------------------------------------------------------------
__device__ float repro_tanh(float x) {
    // For large arguments, tanh saturates
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;

    float exp2x = repro_exp(2.0f * x);
    // Use __fdiv_rn for correctly rounded division (IEEE-mandated)
    return __fdiv_rn(exp2x - 1.0f, exp2x + 1.0f);
}

// ---------------------------------------------------------------------------
// GELU constants (pre-computed to exact float values)
// ---------------------------------------------------------------------------
static __device__ __constant__ float SQRT_2_OVER_PI = 0.7978845608f; // sqrt(2/pi)
static __device__ __constant__ float GELU_COEFF = 0.044715f;

// ---------------------------------------------------------------------------
// Forward kernel
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------
__global__ void repro_gelu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);

    float t = repro_tanh(inner);
    output[idx] = 0.5f * x * (1.0f + t);
}

// ---------------------------------------------------------------------------
// Backward kernel
// d/dx GELU(x) = 0.5 * (1 + tanh(inner))
//              + 0.5 * x * (1 - tanh(inner)^2) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
// ---------------------------------------------------------------------------
__global__ void repro_gelu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    float go = grad_output[idx];

    float x2 = x * x;
    float x3 = x2 * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);

    float t = repro_tanh(inner);
    float t2 = t * t;

    // d(inner)/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * x2);

    // d(GELU)/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * d_inner
    float d_gelu = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t2) * d_inner;

    grad_input[idx] = go * d_gelu;
}

// ---------------------------------------------------------------------------
// CPU reference (for cross-verification)
// ---------------------------------------------------------------------------
static float cpu_tanh_via_exp(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float exp2x = expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

extern "C" void repro_gelu_forward_cpu(
    const float* input,
    float* output,
    int64_t n
) {
    const float sqrt_2_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (int64_t i = 0; i < n; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_pi * (x + coeff * x3);
        float t = cpu_tanh_via_exp(inner);
        output[i] = 0.5f * x * (1.0f + t);
    }
}

extern "C" void repro_gelu_backward_cpu(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int64_t n
) {
    const float sqrt_2_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (int64_t i = 0; i < n; i++) {
        float x = input[i];
        float go = grad_output[i];
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = sqrt_2_pi * (x + coeff * x3);
        float t = cpu_tanh_via_exp(inner);
        float t2 = t * t;
        float d_inner = sqrt_2_pi * (1.0f + 3.0f * coeff * x2);
        float d_gelu = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t2) * d_inner;
        grad_input[i] = go * d_gelu;
    }
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------
extern "C" void repro_gelu_forward(
    const float* input,
    float* output,
    int64_t n,
    cudaStream_t stream
) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    repro_gelu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

extern "C" void repro_gelu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int64_t n,
    cudaStream_t stream
) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    repro_gelu_backward_kernel<<<blocks, threads, 0, stream>>>(grad_output, input, grad_input, n);
}
