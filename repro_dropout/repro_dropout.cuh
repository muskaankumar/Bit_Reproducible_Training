// Bit-reproducible dropout for FP32.
//
// Dropout must be reproducible in two senses:
//   1. The MASK must be identical across GPUs (same positions zeroed out).
//   2. The SCALING (multiply by 1/(1-p)) must be a single IEEE FP32 operation.
//
// For the mask: we use a counter-based RNG (Philox 4x32, same as PyTorch's
// default CUDA generator). The key insight: Philox is a pure function of
// (counter, key) -> output, with no state beyond the counter. So given:
//   - A base seed (set once per training run)
//   - A step counter (incremented each forward pass)
//   - An element offset
// the mask is fully determined and identical on every machine.
//
// We do NOT use cuRAND or PyTorch's generator, because their internal state
// and launch configuration can differ across GPUs. Instead we implement
// Philox directly in the kernel.
//
// For the scaling: x * (1.0f / (1.0f - p)) is two IEEE operations, both
// correctly rounded, so identical everywhere.
//
// The dropout mask is regenerated (not stored) in the backward pass using
// the same (seed, step, offset), which also saves memory.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace repro_dropout {

// Forward: y[i] = (mask[i] ? x[i] * scale : 0), where scale = 1/(1-p).
// Mask is determined by (seed, step_counter, i) via Philox.
cudaError_t dropout_forward(
    const float* x, float* y, size_t n,
    float p,               // drop probability
    uint64_t seed,         // base seed for the entire training run
    uint64_t step_counter, // incremented each call (layer_id * max_steps + step)
    cudaStream_t stream);

// Backward: dx[i] = (mask[i] ? dy[i] * scale : 0).
// Regenerates the same mask from (seed, step_counter, i).
cudaError_t dropout_backward(
    const float* dy, float* dx, size_t n,
    float p,
    uint64_t seed,
    uint64_t step_counter,
    cudaStream_t stream);

// CPU reference
void cpu_dropout_forward(const float* x, float* y, size_t n,
                         float p, uint64_t seed, uint64_t step_counter);
void cpu_dropout_backward(const float* dy, float* dx, size_t n,
                          float p, uint64_t seed, uint64_t step_counter);

} // namespace repro_dropout
