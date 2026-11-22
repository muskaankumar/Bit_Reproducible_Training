// Bit-reproducible Embedding layer for FP32.
//
// Forward: y[b, t, d] = weight[input[b, t], d]
//   This is a pure gather — already bit-reproducible, no floating-point arithmetic.
//
// Backward: dW[v, d] = sum over all (b,t) where input[b,t]==v of dy[b, t, d]
//   The standard approach uses atomicAdd (scatter-add), which is non-deterministic
//   because the order of atomic additions depends on thread scheduling.
//
// Our approach: SORT-THEN-REDUCE
//   1. Create (index, position) pairs for all tokens in the batch
//   2. Sort by index (vocabulary ID) using a deterministic radix sort
//   3. Segment-reduce: for each unique vocabulary ID, sum its gradients
//      using our superaccumulator (order-independent, correctly rounded)
//
// This gives identical bits regardless of thread scheduling, GPU, or batch layout.
// The sort is O(n log n) but n = batch_size * seq_len, which is small relative to
// the embedding dimension, so the extra cost is modest.
//
// Alternative considered: deterministic index_add.
//   PyTorch's index_add has a deterministic CUDA kernel (used by TAO/NVIDIA).
//   But its reduction order depends on the input layout, which means it's
//   run-to-run deterministic but NOT cross-GPU identical if the kernel's tiling
//   differs. Our sort-then-superaccumulator approach is truly order-independent.

#pragma once

#include <cuda_runtime.h>

namespace repro_embedding {

// Forward: out[b*T + t, :] = weight[input[b*T + t], :]
// Pure gather, always bit-identical.
cudaError_t embedding_forward(
    const float* weight,   // [vocab_size, embed_dim]
    const int* input,      // [batch * seq_len]
    float* out,            // [batch * seq_len, embed_dim]
    int n_tokens,          // batch * seq_len
    int embed_dim,
    cudaStream_t stream);

// Backward: dW[v, d] = sum_{i: input[i]==v} dy[i, d]
// Sort-then-superaccumulator-reduce for bit-reproducibility.
//
// workspace: temp buffer for sort keys/values. Call embedding_backward_workspace()
// to get the required size.
size_t embedding_backward_workspace(int n_tokens);

cudaError_t embedding_backward(
    const float* dy,       // [n_tokens, embed_dim]
    const int* input,      // [n_tokens]
    float* dW,             // [vocab_size, embed_dim] — zeroed by caller
    int n_tokens,
    int vocab_size,
    int embed_dim,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

// CPU reference
void cpu_embedding_forward(
    const float* weight, const int* input, float* out,
    int n_tokens, int embed_dim);

void cpu_embedding_backward(
    const float* dy, const int* input, float* dW,
    int n_tokens, int vocab_size, int embed_dim);

} // namespace repro_embedding
