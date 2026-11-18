// PyTorch C++ bindings for reproducible CUDA kernels
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Forward declarations matching the actual extern "C" signatures in .cu files

// GELU
extern "C" void repro_gelu_forward(const float*, float*, int64_t, cudaStream_t);
extern "C" void repro_gelu_backward(const float*, const float*, float*, int64_t, cudaStream_t);

// Dropout
extern "C" void repro_dropout_forward(const float*, float*, int64_t, float, uint64_t, uint64_t, cudaStream_t);
extern "C" void repro_dropout_backward(const float*, float*, int64_t, float, uint64_t, uint64_t, cudaStream_t);

// LayerNorm (forward saves mean and rstd for backward)
extern "C" void repro_layernorm_forward(const float*, const float*, const float*,
                                         float*, float*, float*,
                                         int, int, float, cudaStream_t);
extern "C" void repro_layernorm_backward(const float*, const float*, const float*,
                                          const float*, const float*,
                                          float*, float*, float*,
                                          int, int, cudaStream_t);

// Embedding (uses int32 indices)
extern "C" void repro_embedding_forward(const float*, const int32_t*, float*, int, int, cudaStream_t);
extern "C" void repro_embedding_backward(const float*, const int32_t*, float*,
                                          int32_t*, int32_t*, int*,
                                          int, int, int, cudaStream_t);

// Softmax
extern "C" void repro_softmax_forward(float*, int, int, cudaStream_t);
extern "C" void repro_softmax_backward(const float*, const float*, float*, int, int, cudaStream_t);

// Causal mask + scale
extern "C" void repro_causal_mask_scale(float*, int, int, float, cudaStream_t);


// ============================================================================
// Torch wrapper functions
// ============================================================================

static cudaStream_t get_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// --- GELU ---

void torch_repro_gelu_forward(torch::Tensor input, torch::Tensor output, int64_t n) {
    repro_gelu_forward(input.data_ptr<float>(), output.data_ptr<float>(), n, get_stream());
}

void torch_repro_gelu_backward(torch::Tensor grad_output, torch::Tensor input,
                                torch::Tensor grad_input, int64_t n) {
    repro_gelu_backward(grad_output.data_ptr<float>(), input.data_ptr<float>(),
                        grad_input.data_ptr<float>(), n, get_stream());
}

// --- Dropout ---

void torch_repro_dropout_forward(torch::Tensor input, torch::Tensor output,
                                  int64_t n, float drop_prob,
                                  int64_t seed, int64_t step_counter) {
    repro_dropout_forward(input.data_ptr<float>(), output.data_ptr<float>(),
                          n, drop_prob, (uint64_t)seed, (uint64_t)step_counter,
                          get_stream());
}

void torch_repro_dropout_backward(torch::Tensor grad_output, torch::Tensor grad_input,
                                   int64_t n, float drop_prob,
                                   int64_t seed, int64_t step_counter) {
    repro_dropout_backward(grad_output.data_ptr<float>(), grad_input.data_ptr<float>(),
                           n, drop_prob, (uint64_t)seed, (uint64_t)step_counter,
                           get_stream());
}

// --- LayerNorm ---
// Forward: input[N,D], gamma[D], beta[D] -> output[N,D], mean[N], rstd[N]

void torch_repro_layernorm_forward(torch::Tensor input, torch::Tensor gamma,
                                    torch::Tensor beta, torch::Tensor output,
                                    torch::Tensor mean_out, torch::Tensor rstd_out,
                                    int N, int D, float eps) {
    repro_layernorm_forward(input.data_ptr<float>(), gamma.data_ptr<float>(),
                            beta.data_ptr<float>(), output.data_ptr<float>(),
                            mean_out.data_ptr<float>(), rstd_out.data_ptr<float>(),
                            N, D, eps, get_stream());
}

// Backward: needs saved mean[N] and rstd[N]
void torch_repro_layernorm_backward(torch::Tensor grad_output, torch::Tensor input,
                                     torch::Tensor gamma, torch::Tensor mean,
                                     torch::Tensor rstd,
                                     torch::Tensor grad_input, torch::Tensor grad_gamma,
                                     torch::Tensor grad_beta, int N, int D) {
    repro_layernorm_backward(grad_output.data_ptr<float>(), input.data_ptr<float>(),
                             gamma.data_ptr<float>(), mean.data_ptr<float>(),
                             rstd.data_ptr<float>(),
                             grad_input.data_ptr<float>(), grad_gamma.data_ptr<float>(),
                             grad_beta.data_ptr<float>(), N, D, get_stream());
}

// --- Embedding ---
// Forward: weight[V,D], indices[N] (int32!) -> output[N,D]

// indices already int32 from Python side
void torch_repro_embedding_forward(torch::Tensor weight, torch::Tensor indices,
                                    torch::Tensor output, int N, int D) {
    repro_embedding_forward(weight.data_ptr<float>(), indices.data_ptr<int32_t>(),
                            output.data_ptr<float>(), N, D, get_stream());
}

// Backward: workspace buffers passed from Python
void torch_repro_embedding_backward(torch::Tensor grad_output, torch::Tensor indices,
                                     torch::Tensor grad_weight,
                                     torch::Tensor sorted_indices,
                                     torch::Tensor sorted_positions,
                                     torch::Tensor sort_workspace,
                                     int N, int D, int V) {
    repro_embedding_backward(grad_output.data_ptr<float>(), indices.data_ptr<int32_t>(),
                             grad_weight.data_ptr<float>(),
                             sorted_indices.data_ptr<int32_t>(),
                             sorted_positions.data_ptr<int32_t>(),
                             sort_workspace.data_ptr<int32_t>(),
                             N, D, V, get_stream());
}

// --- Softmax ---

void torch_repro_softmax_forward(torch::Tensor data, int M, int T) {
    repro_softmax_forward(data.data_ptr<float>(), M, T, get_stream());
}

void torch_repro_softmax_backward(torch::Tensor grad_output, torch::Tensor output,
                                   torch::Tensor grad_input, int M, int T) {
    repro_softmax_backward(grad_output.data_ptr<float>(), output.data_ptr<float>(),
                           grad_input.data_ptr<float>(), M, T, get_stream());
}

// --- Causal Mask + Scale ---

void torch_repro_causal_mask_scale(torch::Tensor scores, int BH, int T, float scale) {
    repro_causal_mask_scale(scores.data_ptr<float>(), BH, T, scale, get_stream());
}


// ============================================================================
// Python module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repro_gelu_forward", &torch_repro_gelu_forward, "Reproducible GELU forward");
    m.def("repro_gelu_backward", &torch_repro_gelu_backward, "Reproducible GELU backward");
    m.def("repro_dropout_forward", &torch_repro_dropout_forward, "Reproducible Dropout forward");
    m.def("repro_dropout_backward", &torch_repro_dropout_backward, "Reproducible Dropout backward");
    m.def("repro_layernorm_forward", &torch_repro_layernorm_forward, "Reproducible LayerNorm forward");
    m.def("repro_layernorm_backward", &torch_repro_layernorm_backward, "Reproducible LayerNorm backward");
    m.def("repro_embedding_forward", &torch_repro_embedding_forward, "Reproducible Embedding forward");
    m.def("repro_embedding_backward", &torch_repro_embedding_backward, "Reproducible Embedding backward");
    m.def("repro_softmax_forward", &torch_repro_softmax_forward, "Reproducible Softmax forward");
    m.def("repro_softmax_backward", &torch_repro_softmax_backward, "Reproducible Softmax backward");
    m.def("repro_causal_mask_scale", &torch_repro_causal_mask_scale, "Causal mask + scale");
}
