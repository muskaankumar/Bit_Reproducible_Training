// ============================================================================
// Bit-Reproducible Dropout via Philox Counter-Based RNG
// ============================================================================
//
// Standard dropout uses cuRAND, whose state depends on launch config and
// GPU architecture. We use Philox-4x32-10, a counter-based RNG where the
// mask is a pure function of (seed, step_counter, element_offset).
//
// Key properties:
//   - Deterministic: same inputs → same mask, regardless of GPU
//   - Stateless: no curandState to allocate or manage
//   - Recomputable: backward regenerates the mask from the same counters
//     instead of storing it (saves memory for large models)
//
// Philox-4x32-10 produces 4 uint32 outputs per call. We use one bit per
// output to decide keep/drop, giving us 4 elements per Philox invocation.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// Philox-4x32-10 implementation (minimal, self-contained)
// ---------------------------------------------------------------------------

struct Philox4x32 {
    uint32_t ctr[4];
    uint32_t key[2];
};

// Philox round function
__host__ __device__ inline void philox_round(uint32_t* ctr, const uint32_t* key) {
    // Philox S-box constants
    const uint32_t M0 = 0xD2511F53u;
    const uint32_t M1 = 0xCD9E8D57u;

    uint64_t p0 = (uint64_t)M0 * ctr[0];
    uint64_t p1 = (uint64_t)M1 * ctr[2];

    uint32_t hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo0 = (uint32_t)p0;
    uint32_t hi1 = (uint32_t)(p1 >> 32);
    uint32_t lo1 = (uint32_t)p1;

    uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t new1 = lo1;
    uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t new3 = lo0;

    ctr[0] = new0;
    ctr[1] = new1;
    ctr[2] = new2;
    ctr[3] = new3;
}

// Key schedule: bump key each round
__host__ __device__ inline void philox_bump_key(uint32_t* key) {
    key[0] += 0x9E3779B9u; // golden ratio
    key[1] += 0xBB67AE85u;
}

// Full Philox-4x32-10: 10 rounds
__host__ __device__ inline void philox4x32_10(
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t k0, uint32_t k1,
    uint32_t* out
) {
    uint32_t ctr[4] = {c0, c1, c2, c3};
    uint32_t key[2] = {k0, k1};

    for (int i = 0; i < 10; i++) {
        philox_round(ctr, key);
        philox_bump_key(key);
    }

    out[0] = ctr[0];
    out[1] = ctr[1];
    out[2] = ctr[2];
    out[3] = ctr[3];
}

// Convert uint32 to float in [0, 1)
__host__ __device__ inline float uint32_to_uniform(uint32_t x) {
    // Divide by 2^32, giving [0, 1)
    return (float)x * (1.0f / 4294967296.0f);
}

// ---------------------------------------------------------------------------
// Forward: apply dropout mask
// output[i] = input[i] * mask[i] / (1 - p)   if keep
//           = 0                                 if drop
//
// mask[i] = (philox_output > p) ? 1 : 0
// ---------------------------------------------------------------------------
__global__ void repro_dropout_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t n,
    float drop_prob,
    float scale,       // 1.0 / (1.0 - drop_prob)
    uint64_t seed,
    uint64_t step_counter
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes 4 elements (one Philox call)
    int64_t base = idx * 4;

    if (base >= n) return;

    // Counter: (element_offset_lo, element_offset_hi, step_lo, step_hi)
    uint32_t c0 = (uint32_t)(base & 0xFFFFFFFF);
    uint32_t c1 = (uint32_t)(base >> 32);
    uint32_t c2 = (uint32_t)(step_counter & 0xFFFFFFFF);
    uint32_t c3 = (uint32_t)(step_counter >> 32);
    uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFF);
    uint32_t k1 = (uint32_t)(seed >> 32);

    uint32_t rng[4];
    philox4x32_10(c0, c1, c2, c3, k0, k1, rng);

    for (int j = 0; j < 4; j++) {
        int64_t elem = base + j;
        if (elem >= n) break;

        float u = uint32_to_uniform(rng[j]);
        if (u >= drop_prob) {
            output[elem] = input[elem] * scale;
        } else {
            output[elem] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Backward: regenerate mask from same counters (no mask storage needed)
// grad_input[i] = grad_output[i] * scale   if kept in forward
//               = 0                          if dropped
// ---------------------------------------------------------------------------
__global__ void repro_dropout_backward_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int64_t n,
    float drop_prob,
    float scale,
    uint64_t seed,
    uint64_t step_counter
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t base = idx * 4;

    if (base >= n) return;

    uint32_t c0 = (uint32_t)(base & 0xFFFFFFFF);
    uint32_t c1 = (uint32_t)(base >> 32);
    uint32_t c2 = (uint32_t)(step_counter & 0xFFFFFFFF);
    uint32_t c3 = (uint32_t)(step_counter >> 32);
    uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFF);
    uint32_t k1 = (uint32_t)(seed >> 32);

    uint32_t rng[4];
    philox4x32_10(c0, c1, c2, c3, k0, k1, rng);

    for (int j = 0; j < 4; j++) {
        int64_t elem = base + j;
        if (elem >= n) break;

        float u = uint32_to_uniform(rng[j]);
        if (u >= drop_prob) {
            grad_input[elem] = grad_output[elem] * scale;
        } else {
            grad_input[elem] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
extern "C" void repro_dropout_forward_cpu(
    const float* input,
    float* output,
    int64_t n,
    float drop_prob,
    uint64_t seed,
    uint64_t step_counter
) {
    float scale = 1.0f / (1.0f - drop_prob);

    for (int64_t base = 0; base < n; base += 4) {
        uint32_t c0 = (uint32_t)(base & 0xFFFFFFFF);
        uint32_t c1 = (uint32_t)(base >> 32);
        uint32_t c2 = (uint32_t)(step_counter & 0xFFFFFFFF);
        uint32_t c3 = (uint32_t)(step_counter >> 32);
        uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFF);
        uint32_t k1 = (uint32_t)(seed >> 32);

        uint32_t rng[4];
        philox4x32_10(c0, c1, c2, c3, k0, k1, rng);

        for (int j = 0; j < 4 && (base + j) < n; j++) {
            float u = uint32_to_uniform(rng[j]);
            if (u >= drop_prob) {
                output[base + j] = input[base + j] * scale;
            } else {
                output[base + j] = 0.0f;
            }
        }
    }
}

extern "C" void repro_dropout_backward_cpu(
    const float* grad_output,
    float* grad_input,
    int64_t n,
    float drop_prob,
    uint64_t seed,
    uint64_t step_counter
) {
    float scale = 1.0f / (1.0f - drop_prob);

    for (int64_t base = 0; base < n; base += 4) {
        uint32_t c0 = (uint32_t)(base & 0xFFFFFFFF);
        uint32_t c1 = (uint32_t)(base >> 32);
        uint32_t c2 = (uint32_t)(step_counter & 0xFFFFFFFF);
        uint32_t c3 = (uint32_t)(step_counter >> 32);
        uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFF);
        uint32_t k1 = (uint32_t)(seed >> 32);

        uint32_t rng[4];
        philox4x32_10(c0, c1, c2, c3, k0, k1, rng);

        for (int j = 0; j < 4 && (base + j) < n; j++) {
            float u = uint32_to_uniform(rng[j]);
            if (u >= drop_prob) {
                grad_input[base + j] = grad_output[base + j] * scale;
            } else {
                grad_input[base + j] = 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------
extern "C" void repro_dropout_forward(
    const float* input,
    float* output,
    int64_t n,
    float drop_prob,
    uint64_t seed,
    uint64_t step_counter,
    cudaStream_t stream
) {
    if (n <= 0) return;
    if (drop_prob <= 0.0f) {
        // No dropout — just copy
        cudaMemcpyAsync(output, input, n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }
    if (drop_prob >= 1.0f) {
        cudaMemsetAsync(output, 0, n * sizeof(float), stream);
        return;
    }

    float scale = 1.0f / (1.0f - drop_prob);
    int threads = 256;
    // Each thread handles 4 elements
    int64_t num_threads = (n + 3) / 4;
    int blocks = (int)((num_threads + threads - 1) / threads);

    repro_dropout_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, n, drop_prob, scale, seed, step_counter);
}

extern "C" void repro_dropout_backward(
    const float* grad_output,
    float* grad_input,
    int64_t n,
    float drop_prob,
    uint64_t seed,
    uint64_t step_counter,
    cudaStream_t stream
) {
    if (n <= 0) return;
    if (drop_prob <= 0.0f) {
        cudaMemcpyAsync(grad_input, grad_output, n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }
    if (drop_prob >= 1.0f) {
        cudaMemsetAsync(grad_input, 0, n * sizeof(float), stream);
        return;
    }

    float scale = 1.0f / (1.0f - drop_prob);
    int threads = 256;
    int64_t num_threads = (n + 3) / 4;
    int blocks = (int)((num_threads + threads - 1) / threads);

    repro_dropout_backward_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, grad_input, n, drop_prob, scale, seed, step_counter);
}
