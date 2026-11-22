// ============================================================================
// Bit-Reproducible Embedding (Forward + Backward)
// ============================================================================
//
// Forward: pure gather operation — already bit-reproducible.
//   output[i] = embedding_table[indices[i]]
//
// Backward: the non-reproducible part. Standard scatter-add (atomicAdd or
// index_add_) accumulates gradients for each token embedding, but the
// order depends on thread scheduling.
//
// Our approach for backward:
//   1. Sort (index, position) pairs by index using a deterministic sort
//   2. For each unique index, sum gradients using superaccumulator
//   3. This makes the gradient accumulation order-independent
//
// For GPT-2: vocab_size=50257, d_model=768, seq_len=1024
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Superaccumulator (same as LayerNorm — in integrated build, from repro_sum)
// ---------------------------------------------------------------------------
#define SA_NLIMBS 10
#define SA_RADIX_BITS 32
#define SA_RADIX (1LL << SA_RADIX_BITS)
#define SA_HALF_RADIX (1LL << (SA_RADIX_BITS - 1))

struct SuperAccumEmbed {
    int64_t limbs[SA_NLIMBS];
};

__host__ __device__ inline void sa_embed_init(SuperAccumEmbed* sa) {
    for (int i = 0; i < SA_NLIMBS; i++) sa->limbs[i] = 0;
}

__host__ __device__ inline void sa_embed_add_float(SuperAccumEmbed* sa, float val) {
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

    if (biased_exp == 0) {
        biased_exp = 1;
    } else {
        mantissa |= (1u << 23);
    }

    int exponent = biased_exp - 127 - 23;
    const int BASE_OFFSET = -160;
    int bit_pos = exponent - BASE_OFFSET;
    int limb_idx = bit_pos / SA_RADIX_BITS;
    int bit_offset = bit_pos % SA_RADIX_BITS;
    if (bit_offset < 0) { bit_offset += SA_RADIX_BITS; limb_idx -= 1; }

    int64_t m = (int64_t)mantissa * sign;
    if (limb_idx >= 0 && limb_idx < SA_NLIMBS) {
        sa->limbs[limb_idx] += (m << bit_offset) & (SA_RADIX - 1);
        if (bit_offset > 0 && limb_idx + 1 < SA_NLIMBS) {
            sa->limbs[limb_idx + 1] += (m << bit_offset) >> SA_RADIX_BITS;
        }
    }
}

__host__ __device__ inline void sa_embed_propagate(SuperAccumEmbed* sa) {
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

__host__ __device__ inline float sa_embed_to_float(SuperAccumEmbed* sa) {
    sa_embed_propagate(sa);
    int top = SA_NLIMBS - 1;
    while (top > 0 && sa->limbs[top] == 0) top--;
    if (top == 0 && sa->limbs[0] == 0) return 0.0f;
    const int BASE_OFFSET = -160;
    double result = 0.0;
    for (int i = top; i >= 0 && i >= top - 2; i--) {
        result += (double)sa->limbs[i] * ldexp(1.0, i * SA_RADIX_BITS + BASE_OFFSET);
    }
    return (float)result;
}

// ---------------------------------------------------------------------------
// Forward kernel: pure gather (already reproducible)
// output[b, t, d] = weight[indices[b, t], d]
// ---------------------------------------------------------------------------
__global__ void repro_embedding_forward_kernel(
    const float* __restrict__ weight,     // [V, D]
    const int32_t* __restrict__ indices,  // [N] (flattened B*T)
    float* __restrict__ output,           // [N, D]
    int N,
    int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int row = (int)(idx / D);
    int col = (int)(idx % D);

    if (row >= N || col >= D) return;

    int token_id = indices[row];
    output[(int64_t)row * D + col] = weight[(int64_t)token_id * D + col];
}

// ---------------------------------------------------------------------------
// Backward: sort-then-superaccumulator approach
//
// Step 1: Build (index, position) pairs
// Step 2: Sort by index (radix sort for determinism)
// Step 3: For each unique index, sum grad_output rows using superaccumulator
//
// This is slower than atomicAdd but guarantees bit-reproducibility.
// ---------------------------------------------------------------------------

// Simple bitonic sort for the (index, position) pairs
// We use a radix sort approach for full determinism.

// For simplicity and correctness, we do the backward on one dimension at
// a time: for each embedding dimension d, accumulate all gradients going
// to the same vocabulary index.
__global__ void repro_embedding_backward_kernel(
    const float* __restrict__ grad_output,  // [N, D]
    const int32_t* __restrict__ indices,    // [N]
    float* __restrict__ grad_weight,        // [V, D]
    const int32_t* __restrict__ sorted_indices,  // [N] — pre-sorted token IDs
    const int32_t* __restrict__ sorted_positions, // [N] — original positions
    int N,
    int D,
    int V
) {
    // One thread per (unique_index, dimension) pair
    // We process one embedding dimension at a time
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    // Walk through sorted indices, accumulating with superaccumulator
    int i = 0;
    while (i < N) {
        int current_idx = sorted_indices[i];

        SuperAccumEmbed sa;
        sa_embed_init(&sa);

        // Sum all grad_output rows that map to this index
        while (i < N && sorted_indices[i] == current_idx) {
            int pos = sorted_positions[i];
            sa_embed_add_float(&sa, grad_output[(int64_t)pos * D + d]);
            i++;
        }

        grad_weight[(int64_t)current_idx * D + d] += sa_embed_to_float(&sa);
    }
}

// ---------------------------------------------------------------------------
// Deterministic sort of indices (host-side preparation)
// Uses counting sort for integer indices — perfectly deterministic.
// ---------------------------------------------------------------------------
extern "C" void sort_indices_by_value(
    const int32_t* indices,
    int32_t* sorted_indices,
    int32_t* sorted_positions,
    int N,
    int V
) {
    // Counting sort: O(N + V)
    // Count occurrences
    int* counts = (int*)calloc(V, sizeof(int));
    for (int i = 0; i < N; i++) {
        counts[indices[i]]++;
    }

    // Prefix sum
    int* offsets = (int*)calloc(V, sizeof(int));
    offsets[0] = 0;
    for (int v = 1; v < V; v++) {
        offsets[v] = offsets[v-1] + counts[v-1];
    }

    // Fill sorted arrays (stable: preserves order within same index)
    int* current = (int*)calloc(V, sizeof(int));
    for (int i = 0; i < N; i++) {
        int idx = indices[i];
        int pos = offsets[idx] + current[idx];
        sorted_indices[pos] = idx;
        sorted_positions[pos] = i;
        current[idx]++;
    }

    free(counts);
    free(offsets);
    free(current);
}

// GPU-side counting sort
__global__ void counting_sort_count_kernel(
    const int32_t* __restrict__ indices,
    int* __restrict__ counts,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    atomicAdd(&counts[indices[idx]], 1);
}

__global__ void counting_sort_scatter_kernel(
    const int32_t* __restrict__ indices,
    const int* __restrict__ offsets,
    int32_t* __restrict__ sorted_indices,
    int32_t* __restrict__ sorted_positions,
    int* __restrict__ current_offsets,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int token = indices[idx];
    int pos = atomicAdd(&current_offsets[token], 1);
    sorted_indices[offsets[token] + pos] = token;
    sorted_positions[offsets[token] + pos] = idx;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
extern "C" void repro_embedding_forward_cpu(
    const float* weight,
    const int32_t* indices,
    float* output,
    int N, int D
) {
    for (int i = 0; i < N; i++) {
        int tok = indices[i];
        for (int d = 0; d < D; d++) {
            output[(int64_t)i * D + d] = weight[(int64_t)tok * D + d];
        }
    }
}

extern "C" void repro_embedding_backward_cpu(
    const float* grad_output,
    const int32_t* indices,
    float* grad_weight,
    int N, int D, int V
) {
    // Sort indices
    int32_t* sorted_idx = (int32_t*)malloc(N * sizeof(int32_t));
    int32_t* sorted_pos = (int32_t*)malloc(N * sizeof(int32_t));
    sort_indices_by_value(indices, sorted_idx, sorted_pos, N, V);

    // For each dimension, accumulate with superaccumulator
    for (int d = 0; d < D; d++) {
        int i = 0;
        while (i < N) {
            int current_idx = sorted_idx[i];
            SuperAccumEmbed sa;
            sa_embed_init(&sa);

            while (i < N && sorted_idx[i] == current_idx) {
                int pos = sorted_pos[i];
                sa_embed_add_float(&sa, grad_output[(int64_t)pos * D + d]);
                i++;
            }

            grad_weight[(int64_t)current_idx * D + d] += sa_embed_to_float(&sa);
        }
    }

    free(sorted_idx);
    free(sorted_pos);
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------
extern "C" void repro_embedding_forward(
    const float* weight,
    const int32_t* indices,
    float* output,
    int N, int D,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0) return;
    int threads = 256;
    int64_t total = (int64_t)N * D;
    int blocks = (int)((total + threads - 1) / threads);
    repro_embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
        weight, indices, output, N, D);
}

extern "C" void repro_embedding_backward(
    const float* grad_output,
    const int32_t* indices,
    float* grad_weight,
    int32_t* sorted_indices,  // pre-allocated [N]
    int32_t* sorted_positions, // pre-allocated [N]
    int* sort_workspace,       // pre-allocated [V] for counting sort
    int N, int D, int V,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0) return;

    // Step 1: Zero the sort workspace
    cudaMemsetAsync(sort_workspace, 0, V * sizeof(int), stream);

    // Step 2: Count occurrences of each index
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        counting_sort_count_kernel<<<blocks, threads, 0, stream>>>(
            indices, sort_workspace, N);
    }

    // Step 3: Prefix sum on CPU (small enough for V=50257)
    // For full GPU pipeline, use CUB scan. For now, sync and do on CPU.
    cudaStreamSynchronize(stream);

    int* h_counts = (int*)malloc(V * sizeof(int));
    int* h_offsets = (int*)malloc(V * sizeof(int));
    cudaMemcpy(h_counts, sort_workspace, V * sizeof(int), cudaMemcpyDeviceToHost);

    h_offsets[0] = 0;
    for (int v = 1; v < V; v++) {
        h_offsets[v] = h_offsets[v-1] + h_counts[v-1];
    }

    int* d_offsets;
    cudaMalloc(&d_offsets, V * sizeof(int));
    cudaMemcpyAsync(d_offsets, h_offsets, V * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Zero current offsets for scatter
    int* d_current;
    cudaMalloc(&d_current, V * sizeof(int));
    cudaMemsetAsync(d_current, 0, V * sizeof(int), stream);

    // Step 4: Scatter into sorted order
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        counting_sort_scatter_kernel<<<blocks, threads, 0, stream>>>(
            indices, d_offsets, sorted_indices, sorted_positions, d_current, N);
    }

    // Step 5: Accumulate gradients with superaccumulator
    {
        int threads = 256;
        int blocks = (D + threads - 1) / threads;
        repro_embedding_backward_kernel<<<blocks, threads, 0, stream>>>(
            grad_output, indices, grad_weight,
            sorted_indices, sorted_positions,
            N, D, V);
    }

    cudaFree(d_offsets);
    cudaFree(d_current);
    free(h_counts);
    free(h_offsets);
}
