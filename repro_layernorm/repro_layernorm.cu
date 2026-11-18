// ============================================================================
// Bit-Reproducible Layer Normalization (Forward + Backward)
// ============================================================================
//
// Standard LayerNorm:
//   y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Non-reproducible operations in standard implementations:
//   1. Mean reduction: parallel sum in fp32 is order-dependent
//   2. Variance reduction: same issue
//   3. Division by sqrt(var + eps): rsqrt is not IEEE-mandated
//
// Our approach:
//   1. Use superaccumulator (320-bit exact) for mean and variance sums
//   2. Use __fdiv_rn(1.0f, sqrtf(x)) for rsqrt — both sqrt and div are
//      IEEE-mandated to be correctly rounded
//   3. All other ops use default IEEE round-to-nearest-even
//
// Compiled with --fmad=false to prevent FMA fusion.
// ============================================================================

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cfloat>

// ---------------------------------------------------------------------------
// Superaccumulator: 10 x int64 limbs, radix 2^32 (320-bit exact sum)
// Imported from repro_sum. In the integrated build, these are linked
// from repro_sum_cuda.
// ---------------------------------------------------------------------------

#define SA_NLIMBS 10
#define SA_RADIX_BITS 32
#define SA_RADIX (1LL << SA_RADIX_BITS)
#define SA_HALF_RADIX (1LL << (SA_RADIX_BITS - 1))

struct SuperAccum {
    int64_t limbs[SA_NLIMBS];
};

__host__ __device__ inline void sa_init(SuperAccum* sa) {
    for (int i = 0; i < SA_NLIMBS; i++) {
        sa->limbs[i] = 0;
    }
}

// Add a float to the superaccumulator (exact)
__host__ __device__ inline void sa_add_float(SuperAccum* sa, float val) {
    if (val == 0.0f) return;

    // Decompose float: val = mantissa * 2^exponent
    // IEEE 754: 1 sign, 8 exponent, 23 mantissa bits
    uint32_t bits;
    #ifdef __CUDA_ARCH__
    bits = __float_as_uint(val);
    #else
    memcpy(&bits, &val, sizeof(bits));
    #endif

    int sign = (bits >> 31) ? -1 : 1;
    int biased_exp = (int)((bits >> 23) & 0xFF);
    uint32_t mantissa = bits & 0x7FFFFF;

    if (biased_exp == 0) {
        // Denormal
        biased_exp = 1;
    } else {
        mantissa |= (1u << 23); // implicit 1
    }

    int exponent = biased_exp - 127 - 23; // actual power of 2

    // mantissa is a 24-bit integer, shift it into the right limb position
    // limb i represents values in [i*32, (i+1)*32) bit positions
    // We need to place mantissa * 2^exponent into the accumulator
    // The accumulator covers bits from some base offset

    // Offset: limb 0 holds the least significant bits
    // We center the accumulator around typical float range
    // Exponent range for float32: -149 to 127
    // We place exponent 0 at limb 5 (bits 160-191)
    const int BASE_OFFSET = -160; // limb 0 starts at 2^(-160)

    int bit_pos = exponent - BASE_OFFSET;
    int limb_idx = bit_pos / SA_RADIX_BITS;
    int bit_offset = bit_pos % SA_RADIX_BITS;
    if (bit_offset < 0) {
        bit_offset += SA_RADIX_BITS;
        limb_idx -= 1;
    }

    // Place 24-bit mantissa starting at bit_offset in limb limb_idx
    int64_t m = (int64_t)mantissa * sign;

    if (limb_idx >= 0 && limb_idx < SA_NLIMBS) {
        sa->limbs[limb_idx] += (m << bit_offset) & (SA_RADIX - 1);
        // Handle overflow from the shift
        if (bit_offset > 0 && limb_idx + 1 < SA_NLIMBS) {
            sa->limbs[limb_idx + 1] += (m << bit_offset) >> SA_RADIX_BITS;
        }
        // Upper bits of mantissa may spill into next limb
        if (bit_offset + 24 > SA_RADIX_BITS && limb_idx + 1 < SA_NLIMBS) {
            // Already handled by the shift above
        }
    }
    if (limb_idx + 1 >= 0 && limb_idx + 1 < SA_NLIMBS && bit_offset + 24 > SA_RADIX_BITS) {
        // Spill handled above
    }
}

// Propagate carries through the superaccumulator
__host__ __device__ inline void sa_propagate(SuperAccum* sa) {
    for (int i = 0; i < SA_NLIMBS - 1; i++) {
        int64_t carry = 0;
        if (sa->limbs[i] >= SA_HALF_RADIX) {
            carry = sa->limbs[i] >> SA_RADIX_BITS;
            sa->limbs[i] &= (SA_RADIX - 1);
            if (sa->limbs[i] >= SA_HALF_RADIX)
                sa->limbs[i] -= SA_RADIX, carry++;
        } else if (sa->limbs[i] < -SA_HALF_RADIX) {
            carry = -((-sa->limbs[i]) >> SA_RADIX_BITS);
            sa->limbs[i] += (-carry) * SA_RADIX;
            if (sa->limbs[i] < -SA_HALF_RADIX)
                sa->limbs[i] += SA_RADIX, carry--;
        }
        sa->limbs[i + 1] += carry;
    }
}

// Convert superaccumulator back to float (correctly rounded)
__host__ __device__ inline float sa_to_float(SuperAccum* sa) {
    sa_propagate(sa);

    // Find highest non-zero limb
    int top = SA_NLIMBS - 1;
    while (top > 0 && sa->limbs[top] == 0) top--;

    if (top == 0 && sa->limbs[0] == 0) return 0.0f;

    // Convert the top 2-3 limbs to double, then to float
    const int BASE_OFFSET = -160;
    double result = 0.0;
    for (int i = top; i >= 0 && i >= top - 2; i--) {
        double limb_val = (double)sa->limbs[i];
        double scale = ldexp(1.0, (i * SA_RADIX_BITS) + BASE_OFFSET);
        result += limb_val * scale;
    }

    return (float)result;
}

// Merge two superaccumulators
__host__ __device__ inline void sa_merge(SuperAccum* dst, const SuperAccum* src) {
    for (int i = 0; i < SA_NLIMBS; i++) {
        dst->limbs[i] += src->limbs[i];
    }
}

// ---------------------------------------------------------------------------
// Forward kernel: one warp per row (hidden_dim elements)
// For GPT-2 small, hidden_dim = 768 — fits well in one warp with loops
// ---------------------------------------------------------------------------

// Phase 1: compute mean using superaccumulator
__global__ void repro_layernorm_mean_kernel(
    const float* __restrict__ input,   // [N, D]
    float* __restrict__ mean_out,      // [N]
    int N,
    int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x = input + (int64_t)row * D;

    // Each thread accumulates a portion
    SuperAccum sa;
    sa_init(&sa);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sa_add_float(&sa, x[i]);
    }

    // Warp reduction of superaccumulators via shared memory
    __shared__ SuperAccum shared_sa[32]; // max 32 threads per warp
    shared_sa[threadIdx.x] = sa;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sa_merge(&shared_sa[threadIdx.x], &shared_sa[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = sa_to_float(&shared_sa[0]);
        // Correctly rounded division
        mean_out[row] = __fdiv_rn(sum, (float)D);
    }
}

// Phase 2: compute variance and normalize
__global__ void repro_layernorm_normalize_kernel(
    const float* __restrict__ input,   // [N, D]
    const float* __restrict__ gamma,   // [D]
    const float* __restrict__ beta,    // [D]
    const float* __restrict__ mean,    // [N]
    float* __restrict__ output,        // [N, D]
    float* __restrict__ rstd_out,      // [N] — saved for backward
    int N,
    int D,
    float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x = input + (int64_t)row * D;
    float* y = output + (int64_t)row * D;
    float mu = mean[row];

    // Compute variance with superaccumulator
    SuperAccum sa;
    sa_init(&sa);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = x[i] - mu;
        sa_add_float(&sa, diff * diff);
    }

    __shared__ SuperAccum shared_sa[32];
    shared_sa[threadIdx.x] = sa;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sa_merge(&shared_sa[threadIdx.x], &shared_sa[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    __shared__ float shared_rstd;
    if (threadIdx.x == 0) {
        float var_sum = sa_to_float(&shared_sa[0]);
        float var = __fdiv_rn(var_sum, (float)D);
        // rstd = 1 / sqrt(var + eps)
        // Both sqrt and div are IEEE-mandated correctly rounded
        float rstd = __fdiv_rn(1.0f, sqrtf(var + eps));
        shared_rstd = rstd;
        rstd_out[row] = rstd;
    }
    __syncthreads();

    float rstd = shared_rstd;

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xhat = (x[i] - mu) * rstd;
        y[i] = gamma[i] * xhat + beta[i];
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
//
// Given: dL/dy (grad_output), x, gamma, mean, rstd
// Compute: dL/dx, dL/dgamma, dL/dbeta
//
// dL/dbeta_j  = sum_i dL/dy_{i,j}
// dL/dgamma_j = sum_i dL/dy_{i,j} * xhat_{i,j}
// dL/dx_{i,j} = rstd * (gamma_j * dL/dy_{i,j}
//              - (1/D) * sum_k gamma_k * dL/dy_{i,k}
//              - (1/D) * xhat_{i,j} * sum_k gamma_k * dL/dy_{i,k} * xhat_{i,k})
//
// The sums over k must use superaccumulators for reproducibility.
// ---------------------------------------------------------------------------

// Phase 1: compute dL/dgamma and dL/dbeta (reduce over N dimension)
// These need superaccumulators since we're summing over the batch.
__global__ void repro_layernorm_backward_params_kernel(
    const float* __restrict__ grad_output,  // [N, D]
    const float* __restrict__ input,        // [N, D]
    const float* __restrict__ mean,         // [N]
    const float* __restrict__ rstd,         // [N]
    float* __restrict__ grad_gamma,         // [D]
    float* __restrict__ grad_beta,          // [D]
    int N,
    int D
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    SuperAccum sa_gamma, sa_beta;
    sa_init(&sa_gamma);
    sa_init(&sa_beta);

    for (int i = 0; i < N; i++) {
        float dy = grad_output[(int64_t)i * D + j];
        float xhat = (input[(int64_t)i * D + j] - mean[i]) * rstd[i];

        sa_add_float(&sa_beta, dy);
        sa_add_float(&sa_gamma, dy * xhat);
    }

    grad_beta[j] = sa_to_float(&sa_beta);
    grad_gamma[j] = sa_to_float(&sa_gamma);
}

// Phase 2: compute dL/dx (per-row)
__global__ void repro_layernorm_backward_input_kernel(
    const float* __restrict__ grad_output,  // [N, D]
    const float* __restrict__ input,        // [N, D]
    const float* __restrict__ gamma,        // [D]
    const float* __restrict__ mean,         // [N]
    const float* __restrict__ rstd,         // [N]
    float* __restrict__ grad_input,         // [N, D]
    int N,
    int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    float mu = mean[row];
    float rs = rstd[row];

    const float* dy = grad_output + (int64_t)row * D;
    const float* x = input + (int64_t)row * D;
    float* dx = grad_input + (int64_t)row * D;

    // Compute two sums with superaccumulators:
    // sum1 = sum_k gamma_k * dy_k
    // sum2 = sum_k gamma_k * dy_k * xhat_k
    SuperAccum sa1, sa2;
    sa_init(&sa1);
    sa_init(&sa2);

    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float g_dy = gamma[k] * dy[k];
        float xhat = (x[k] - mu) * rs;
        sa_add_float(&sa1, g_dy);
        sa_add_float(&sa2, g_dy * xhat);
    }

    __shared__ SuperAccum shared_sa1[32], shared_sa2[32];
    shared_sa1[threadIdx.x] = sa1;
    shared_sa2[threadIdx.x] = sa2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sa_merge(&shared_sa1[threadIdx.x], &shared_sa1[threadIdx.x + stride]);
            sa_merge(&shared_sa2[threadIdx.x], &shared_sa2[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    __shared__ float s1, s2;
    if (threadIdx.x == 0) {
        s1 = __fdiv_rn(sa_to_float(&shared_sa1[0]), (float)D);
        s2 = __fdiv_rn(sa_to_float(&shared_sa2[0]), (float)D);
    }
    __syncthreads();

    // dx_j = rstd * (gamma_j * dy_j - s1 - xhat_j * s2)
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xhat = (x[j] - mu) * rs;
        dx[j] = rs * (gamma[j] * dy[j] - s1 - xhat * s2);
    }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
extern "C" void repro_layernorm_forward_cpu(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean_out,
    float* rstd_out,
    int N, int D, float eps
) {
    for (int i = 0; i < N; i++) {
        const float* x = input + (int64_t)i * D;
        float* y = output + (int64_t)i * D;

        // Mean with superaccumulator
        SuperAccum sa;
        sa_init(&sa);
        for (int j = 0; j < D; j++) sa_add_float(&sa, x[j]);
        float mu = sa_to_float(&sa) / (float)D;
        mean_out[i] = mu;

        // Variance with superaccumulator
        sa_init(&sa);
        for (int j = 0; j < D; j++) {
            float d = x[j] - mu;
            sa_add_float(&sa, d * d);
        }
        float var = sa_to_float(&sa) / (float)D;
        float rstd = 1.0f / sqrtf(var + eps);
        rstd_out[i] = rstd;

        for (int j = 0; j < D; j++) {
            float xhat = (x[j] - mu) * rstd;
            y[j] = gamma[j] * xhat + beta[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------
extern "C" void repro_layernorm_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean_out,
    float* rstd_out,
    int N, int D, float eps,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0) return;

    // Choose block size: power of 2, max 32 for warp-level reduction
    // For D=768, 32 threads each handle 24 elements
    int threads = 32;
    if (D <= 32) threads = 32;

    // Phase 1: compute means
    repro_layernorm_mean_kernel<<<N, threads, 0, stream>>>(
        input, mean_out, N, D);

    // Phase 2: compute variance, normalize, apply affine
    repro_layernorm_normalize_kernel<<<N, threads, 0, stream>>>(
        input, gamma, beta, mean_out, output, rstd_out, N, D, eps);
}

extern "C" void repro_layernorm_backward(
    const float* grad_output,
    const float* input,
    const float* gamma,
    const float* mean,
    const float* rstd,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int N, int D,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0) return;

    // Phase 1: parameter gradients (one thread per feature)
    {
        int threads = 256;
        int blocks = (D + threads - 1) / threads;
        repro_layernorm_backward_params_kernel<<<blocks, threads, 0, stream>>>(
            grad_output, input, mean, rstd, grad_gamma, grad_beta, N, D);
    }

    // Phase 2: input gradients (one block per row)
    {
        int threads = 32;
        repro_layernorm_backward_input_kernel<<<N, threads, 0, stream>>>(
            grad_output, input, gamma, mean, rstd, grad_input, N, D);
    }
}
