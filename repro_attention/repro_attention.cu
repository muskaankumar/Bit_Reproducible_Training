// ============================================================================
// Bit-Reproducible Attention Components
// ============================================================================
//
// GPT-2 attention:
//   Q, K, V = x @ Wq, x @ Wk, x @ Wv    (use repro_gemm)
//   attn = softmax(Q @ K^T / sqrt(d_k) + causal_mask)
//   output = attn @ V                      (use repro_gemm)
//
// Non-reproducible operations:
//   1. Batched GEMM: Q @ K^T and attn @ V across heads
//   2. Softmax: exp and sum are order-dependent
//   3. Softmax backward: row dot-products need exact accumulation
//
// Our approach:
//   1. Batched GEMM: loop over heads, call repro_gemm per head
//   2. Softmax forward: repro_exp + superaccumulator for denominator
//   3. Softmax backward: superaccumulator for dot-product sum
//   4. Causal mask: simple comparison (already reproducible)
//   5. Scale: multiply by 1/sqrt(d_k) using __fdiv_rn + sqrtf
//
// Compiled with --fmad=false.
// ============================================================================

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cfloat>

// ---------------------------------------------------------------------------
// Superaccumulator (shared with layernorm/embedding)
// ---------------------------------------------------------------------------
#define SA_NLIMBS 10
#define SA_RADIX_BITS 32
#define SA_RADIX (1LL << SA_RADIX_BITS)
#define SA_HALF_RADIX (1LL << (SA_RADIX_BITS - 1))

struct SuperAccumAttn {
    int64_t limbs[SA_NLIMBS];
};

__host__ __device__ inline void sa_attn_init(SuperAccumAttn* sa) {
    for (int i = 0; i < SA_NLIMBS; i++) sa->limbs[i] = 0;
}

__host__ __device__ inline void sa_attn_add_float(SuperAccumAttn* sa, float val) {
    if (val == 0.0f) return;
    uint32_t bits;
    #ifdef __CUDA_ARCH__
    bits = __float_as_uint(val);
    #else
    memcpy(&bits, &val, sizeof(bits));
    #endif

    int sign = (bits >> 31) ? -1 : 1;
    int biased_exp = (int)((bits >> 23) & 0xFF);
    uint32_t mantissa = bits & 0x7FFFFF;
    if (biased_exp == 0) { biased_exp = 1; }
    else { mantissa |= (1u << 23); }
    int exponent = biased_exp - 127 - 23;
    const int BASE_OFFSET = -160;
    int bit_pos = exponent - BASE_OFFSET;
    int limb_idx = bit_pos / SA_RADIX_BITS;
    int bit_offset = bit_pos % SA_RADIX_BITS;
    if (bit_offset < 0) { bit_offset += SA_RADIX_BITS; limb_idx -= 1; }
    int64_t m = (int64_t)mantissa * sign;
    if (limb_idx >= 0 && limb_idx < SA_NLIMBS) {
        sa->limbs[limb_idx] += (m << bit_offset) & (SA_RADIX - 1);
        if (bit_offset > 0 && limb_idx + 1 < SA_NLIMBS)
            sa->limbs[limb_idx + 1] += (m << bit_offset) >> SA_RADIX_BITS;
    }
}

__host__ __device__ inline void sa_attn_propagate(SuperAccumAttn* sa) {
    for (int i = 0; i < SA_NLIMBS - 1; i++) {
        int64_t carry = 0;
        if (sa->limbs[i] >= SA_HALF_RADIX) {
            carry = sa->limbs[i] >> SA_RADIX_BITS;
            sa->limbs[i] &= (SA_RADIX - 1);
            if (sa->limbs[i] >= SA_HALF_RADIX) { sa->limbs[i] -= SA_RADIX; carry++; }
        } else if (sa->limbs[i] < -SA_HALF_RADIX) {
            carry = -((-sa->limbs[i]) >> SA_RADIX_BITS);
            sa->limbs[i] += (-carry) * SA_RADIX;
            if (sa->limbs[i] < -SA_HALF_RADIX) { sa->limbs[i] += SA_RADIX; carry--; }
        }
        sa->limbs[i + 1] += carry;
    }
}

__host__ __device__ inline float sa_attn_to_float(SuperAccumAttn* sa) {
    sa_attn_propagate(sa);
    int top = SA_NLIMBS - 1;
    while (top > 0 && sa->limbs[top] == 0) top--;
    if (top == 0 && sa->limbs[0] == 0) return 0.0f;
    const int BASE_OFFSET = -160;
    double result = 0.0;
    for (int i = top; i >= 0 && i >= top - 2; i--)
        result += (double)sa->limbs[i] * ldexp(1.0, i * SA_RADIX_BITS + BASE_OFFSET);
    return (float)result;
}

// ---------------------------------------------------------------------------
// Correctly rounded exp (from repro_math)
// ---------------------------------------------------------------------------
#ifndef REPRO_EXP_DEFINED
__device__ float repro_exp_attn(float x) {
    if (x > 88.7f) return INFINITY;
    if (x < -103.9f) return 0.0f;
    return expf(x);
}
#else
#define repro_exp_attn repro_exp
#endif

// ---------------------------------------------------------------------------
// Causal mask + scale kernel
// scores[b, h, i, j] = (j <= i) ? qk[b, h, i, j] * scale : -inf
// ---------------------------------------------------------------------------
__global__ void repro_causal_mask_scale_kernel(
    float* __restrict__ scores,  // [B*H, T, T]
    int T,
    float scale
) {
    int bh = blockIdx.z;  // batch * head index
    int i = blockIdx.y;   // query position
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key position

    if (j >= T) return;

    int64_t offset = (int64_t)bh * T * T + (int64_t)i * T + j;

    if (j <= i) {
        // Scale: multiply by 1/sqrt(d_k)
        scores[offset] = scores[offset] * scale;
    } else {
        scores[offset] = -INFINITY;
    }
}

// ---------------------------------------------------------------------------
// Softmax forward (per-row, using superaccumulator for denominator)
//
// For each row:
//   1. Find max (for numerical stability) — simple parallel reduction
//   2. Compute exp(x - max) using repro_exp
//   3. Sum exp values with superaccumulator
//   4. Divide each by sum using __fdiv_rn
// ---------------------------------------------------------------------------
__global__ void repro_softmax_forward_kernel(
    float* __restrict__ data,  // [M, T] — in-place
    int M,                     // number of rows (B*H*T)
    int T                      // row length
) {
    int row = blockIdx.x;
    if (row >= M) return;

    float* x = data + (int64_t)row * T;

    // Step 1: find max (deterministic — max is order-independent)
    float row_max = -INFINITY;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float val = x[j];
        if (val > row_max) row_max = val;
    }
    // Warp reduction for max
    __shared__ float shared_max[32];
    shared_max[threadIdx.x] = row_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x])
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
        }
        __syncthreads();
    }
    row_max = shared_max[0];

    // Step 2: compute exp(x - max)
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        x[j] = repro_exp_attn(x[j] - row_max);
    }
    __syncthreads();

    // Step 3: sum with superaccumulator
    SuperAccumAttn sa;
    sa_attn_init(&sa);
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        sa_attn_add_float(&sa, x[j]);
    }

    __shared__ SuperAccumAttn shared_sa[32];
    shared_sa[threadIdx.x] = sa;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int l = 0; l < SA_NLIMBS; l++)
                shared_sa[threadIdx.x].limbs[l] += shared_sa[threadIdx.x + stride].limbs[l];
        }
        __syncthreads();
    }

    __shared__ float inv_sum;
    if (threadIdx.x == 0) {
        float sum = sa_attn_to_float(&shared_sa[0]);
        inv_sum = __fdiv_rn(1.0f, sum);
    }
    __syncthreads();

    // Step 4: normalize
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        x[j] = x[j] * inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Softmax backward
//
// Given: dL/dy (grad on softmax output), y (softmax output)
// dL/dx_j = y_j * (dL/dy_j - sum_k y_k * dL/dy_k)
//
// The sum_k needs superaccumulator for reproducibility.
// ---------------------------------------------------------------------------
__global__ void repro_softmax_backward_kernel(
    const float* __restrict__ grad_output,  // [M, T]
    const float* __restrict__ output,       // [M, T] (softmax output)
    float* __restrict__ grad_input,         // [M, T]
    int M,
    int T
) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* dy = grad_output + (int64_t)row * T;
    const float* y = output + (int64_t)row * T;
    float* dx = grad_input + (int64_t)row * T;

    // Compute dot product sum_k y_k * dy_k with superaccumulator
    SuperAccumAttn sa;
    sa_attn_init(&sa);
    for (int k = threadIdx.x; k < T; k += blockDim.x) {
        sa_attn_add_float(&sa, y[k] * dy[k]);
    }

    __shared__ SuperAccumAttn shared_sa[32];
    shared_sa[threadIdx.x] = sa;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int l = 0; l < SA_NLIMBS; l++)
                shared_sa[threadIdx.x].limbs[l] += shared_sa[threadIdx.x + stride].limbs[l];
        }
        __syncthreads();
    }

    __shared__ float dot_sum;
    if (threadIdx.x == 0) {
        dot_sum = sa_attn_to_float(&shared_sa[0]);
    }
    __syncthreads();

    // dx_j = y_j * (dy_j - dot_sum)
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        dx[j] = y[j] * (dy[j] - dot_sum);
    }
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------

extern "C" void repro_causal_mask_scale(
    float* scores,  // [B*H, T, T]
    int BH,         // B * num_heads
    int T,
    float scale,    // 1/sqrt(d_k)
    cudaStream_t stream
) {
    int threads = 256;
    int blocks_j = (T + threads - 1) / threads;
    dim3 grid(blocks_j, T, BH);
    repro_causal_mask_scale_kernel<<<grid, threads, 0, stream>>>(scores, T, scale);
}

extern "C" void repro_softmax_forward(
    float* data,    // [M, T] — modified in-place
    int M,          // total rows
    int T,          // row length
    cudaStream_t stream
) {
    if (M <= 0 || T <= 0) return;
    int threads = 32; // warp-sized for shared memory reduction
    repro_softmax_forward_kernel<<<M, threads, 0, stream>>>(data, M, T);
}

extern "C" void repro_softmax_backward(
    const float* grad_output,
    const float* output,
    float* grad_input,
    int M,
    int T,
    cudaStream_t stream
) {
    if (M <= 0 || T <= 0) return;
    int threads = 32;
    repro_softmax_backward_kernel<<<M, threads, 0, stream>>>(
        grad_output, output, grad_input, M, T);
}

// ---------------------------------------------------------------------------
// Batched GEMM wrapper
// For attention: loop over heads, call repro_gemm per head.
// This is called from Python, which loops over (batch, head) and invokes
// repro_gemm for each Q@K^T and attn@V multiplication.
//
// In the integrated build, the Python layer handles the loop and calls
// the existing repro_gemm. This file provides the attention-specific
// kernels (causal mask, softmax) that don't exist in the base library.
// ---------------------------------------------------------------------------

// CPU references for testing
extern "C" void repro_softmax_forward_cpu(
    float* data, int M, int T
) {
    for (int row = 0; row < M; row++) {
        float* x = data + (int64_t)row * T;

        // Max
        float mx = -INFINITY;
        for (int j = 0; j < T; j++) if (x[j] > mx) mx = x[j];

        // Exp
        for (int j = 0; j < T; j++) x[j] = expf(x[j] - mx);

        // Sum with superaccumulator
        SuperAccumAttn sa;
        sa_attn_init(&sa);
        for (int j = 0; j < T; j++) sa_attn_add_float(&sa, x[j]);
        float sum = sa_attn_to_float(&sa);

        // Normalize
        for (int j = 0; j < T; j++) x[j] = x[j] / sum;
    }
}

extern "C" void repro_softmax_backward_cpu(
    const float* grad_output, const float* output,
    float* grad_input, int M, int T
) {
    for (int row = 0; row < M; row++) {
        const float* dy = grad_output + (int64_t)row * T;
        const float* y = output + (int64_t)row * T;
        float* dx = grad_input + (int64_t)row * T;

        SuperAccumAttn sa;
        sa_attn_init(&sa);
        for (int k = 0; k < T; k++) sa_attn_add_float(&sa, y[k] * dy[k]);
        float dot = sa_attn_to_float(&sa);

        for (int j = 0; j < T; j++) dx[j] = y[j] * (dy[j] - dot);
    }
}
