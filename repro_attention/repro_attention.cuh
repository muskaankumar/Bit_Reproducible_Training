// Bit-reproducible multi-head causal self-attention for FP32.
//
// GPT-2 attention for one head:
//   Q, K, V = linear projections (handled by repro_gemm Linear layers)
//   S = Q @ K^T / sqrt(d_k)          batched GEMM + scalar multiply
//   S = causal_mask(S)                set future positions to -inf (exact)
//   P = softmax(S, dim=-1)            reproducible softmax (existing)
//   P = dropout(P)                    seeded dropout (repro_dropout)
//   O = P @ V                         batched GEMM
//
// The key new primitive is BATCHED GEMM: the existing repro_gemm handles a
// single (M, N, K) multiplication. For attention we need B*H independent
// GEMMs of shape (T, T, d_k) and (T, d_k, T).
//
// Approach: loop over heads and call repro_gemm for each. This is simple and
// correct, though not optimal. A fused batched kernel is a later optimization.
// Since each call is independently bit-reproducible, the loop is too.
//
// For the backward pass, the six GEMMs per head are:
//   dO -> dP = dO @ V^T              (T, T, d_k)
//   dO -> dV = P^T @ dO              (d_k, d_k_unused... actually T, d_k, T)
//   dS from dP through softmax backward
//   dS -> dQ = dS @ K                (T, d_k, T)
//   dS -> dK = dS^T @ Q              (T, d_k, T)
//
// Softmax backward:
//   dS[r,c] = P[r,c] * (dP[r,c] - dot(P[r,:], dP[r,:]))
//   The dot product uses superaccumulator for the row reduction.

#pragma once

#include <cuda_runtime.h>

namespace repro_attention {

// Batched GEMM: C[b, :, :] = op(A[b, :, :]) @ op(B[b, :, :])
// For attention: batch = num_heads (or batch_size * num_heads).
// Uses repro_gemm internally in a loop (each head independently reproducible).
cudaError_t batched_gemm(
    const float* A,    // [batch, M, K] or [batch, K, M] if trans_a
    const float* B,    // [batch, K, N] or [batch, N, K] if trans_b
    float* C,          // [batch, M, N]
    int batch, int M, int N, int K,
    bool trans_a, bool trans_b,
    int slices,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream);

size_t batched_gemm_workspace(int M, int N, int K, int slices);

// Softmax backward (per-row):
//   dS[r,c] = P[r,c] * (dP[r,c] - sum_c'(P[r,c'] * dP[r,c']))
// The row dot product uses superaccumulator.
cudaError_t softmax_backward(
    const float* dP,   // [rows, cols]
    const float* P,    // [rows, cols] (softmax output from forward)
    float* dS,         // [rows, cols]
    int rows, int cols,
    cudaStream_t stream);

// Causal mask: set S[r, c] = -inf for c > r (upper triangle).
// Exact integer comparison, no floating-point ambiguity.
cudaError_t causal_mask(float* S, int T, int batch_heads, cudaStream_t stream);

// Scale: S[i] = S[i] * scale. Single IEEE multiply, always reproducible.
cudaError_t scale(float* S, float scale_val, size_t n, cudaStream_t stream);

// CPU references
void cpu_batched_gemm(const float* A, const float* B, float* C,
                      int batch, int M, int N, int K, bool ta, bool tb, int slices);
void cpu_softmax_backward(const float* dP, const float* P, float* dS, int rows, int cols);

} // namespace repro_attention
