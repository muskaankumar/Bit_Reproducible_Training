// PyTorch C++ extension wrapping repro_gemm for use as a custom autograd op.
// Exposes: repro_mm(A, B, trans_a, trans_b, slices) -> C
//
// A is [M, K] (or [K, M] if trans_a), B is [K, N] (or [N, K] if trans_b).
// Returns C [M, N] = op(A) @ op(B), bit-reproducible.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "../repro_gemm_cuda/repro_gemm.cuh"

// ---- workspace cache (one per shape to avoid malloc in the training loop) ----
#include <unordered_map>
#include <mutex>

struct WsKey {
    int M, N, K, S;
    bool operator==(const WsKey& o) const { return M==o.M && N==o.N && K==o.K && S==o.S; }
};
struct WsHash {
    size_t operator()(const WsKey& k) const {
        size_t h = 17;
        h = h * 31 + k.M; h = h * 31 + k.N; h = h * 31 + k.K; h = h * 31 + k.S;
        return h;
    }
};

static std::unordered_map<WsKey, std::pair<void*, size_t>, WsHash> ws_cache;
static std::mutex ws_mutex;

static void* get_workspace(int M, int N, int K, int S, const repro_gemm::Plan& plan) {
    WsKey key{M, N, K, S};
    std::lock_guard<std::mutex> lock(ws_mutex);
    auto it = ws_cache.find(key);
    if (it != ws_cache.end() && it->second.second >= plan.workspace_bytes) {
        return it->second.first;
    }
    if (it != ws_cache.end()) {
        cudaFree(it->second.first);
        ws_cache.erase(it);
    }
    void* ws = nullptr;
    cudaError_t e = cudaMalloc(&ws, plan.workspace_bytes);
    TORCH_CHECK(e == cudaSuccess, "cudaMalloc workspace: ", cudaGetErrorString(e));
    ws_cache[key] = {ws, plan.workspace_bytes};
    return ws;
}

// Core matrix multiply: C[M,N] = op(A) @ op(B)
// A stored row-major: [M,K] if !trans_a, [K,M] if trans_a
// B stored row-major: [K,N] if !trans_b, [N,K] if trans_b
torch::Tensor repro_mm(torch::Tensor A, torch::Tensor B,
                       bool trans_a, bool trans_b, int slices) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "inputs must be float32");
    A = A.contiguous();
    B = B.contiguous();

    int M, N, K;
    if (!trans_a) {
        TORCH_CHECK(A.dim() == 2);
        M = A.size(0); K = A.size(1);
    } else {
        TORCH_CHECK(A.dim() == 2);
        K = A.size(0); M = A.size(1);
    }
    if (!trans_b) {
        TORCH_CHECK(B.dim() == 2);
        TORCH_CHECK(B.size(0) == K, "dimension mismatch");
        N = B.size(1);
    } else {
        TORCH_CHECK(B.dim() == 2);
        N = B.size(0);
        TORCH_CHECK(B.size(1) == K, "dimension mismatch");
    }

    auto C = torch::empty({M, N}, A.options());

    repro_gemm::Config cfg;
    cfg.slices = slices;
    cfg.trans_a = trans_a;
    cfg.trans_b = trans_b;

    repro_gemm::Plan plan;
    cudaError_t e = repro_gemm::make_plan(M, N, K, cfg, &plan);
    TORCH_CHECK(e == cudaSuccess, "make_plan: ", cudaGetErrorString(e));

    void* ws = get_workspace(M, N, K, slices, plan);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    e = repro_gemm::repro_gemm_fp32_ws(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, cfg, ws, plan.workspace_bytes, stream);
    TORCH_CHECK(e == cudaSuccess, "repro_gemm: ", cudaGetErrorString(e));

    return C;
}

// ---- Reproducible element-wise ops via double precision ----
// These match the RepDL strategy: cast to double, compute, cast back.
// This is bit-reproducible across GPUs with the same CUDA version,
// and across CPU/GPU for IEEE-exact ops.

__global__ void repro_exp_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)exp((double)x[i]);
}

__global__ void repro_log_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)log((double)x[i]);
}

__global__ void repro_sqrt_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)sqrt((double)x[i]);
}

__global__ void repro_rsqrt_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)(1.0 / sqrt((double)x[i]));
}

__global__ void repro_tanh_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)tanh((double)x[i]);
}

// Division by a scalar: x / d. d is passed as a float to avoid x*0.1 pitfall.
__global__ void repro_div_scalar_kernel(const float* __restrict__ x, float d,
                                        float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)((double)x[i] / (double)d);
}

// Softmax: for reproducibility, we use double-precision intermediate.
// max subtraction -> exp in double -> sum in double -> divide in double -> cast to float
__global__ void repro_softmax_kernel(const float* __restrict__ x, float* __restrict__ y,
                                     int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * cols;
    float* yr = y + (size_t)row * cols;

    // Find max (exact, no rounding)
    double mx = (double)xr[0];
    for (int j = 1; j < cols; ++j) {
        double v = (double)xr[j];
        if (v > mx) mx = v;
    }

    // exp and sum in double
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
        double e = exp((double)xr[j] - mx);
        yr[j] = __double2float_rn(e);  // store exp temporarily
        sum += e;
    }

    // divide in double, round to float
    for (int j = 0; j < cols; ++j) {
        // Re-compute exp to stay in double (avoid float round-trip)
        double e = exp((double)xr[j] - mx);
        yr[j] = __double2float_rn(e / sum);
    }
}

// Log-softmax in double precision for cross-entropy
__global__ void repro_log_softmax_kernel(const float* __restrict__ x, float* __restrict__ y,
                                         int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * cols;
    float* yr = y + (size_t)row * cols;

    double mx = (double)xr[0];
    for (int j = 1; j < cols; ++j) {
        double v = (double)xr[j];
        if (v > mx) mx = v;
    }

    double sum = 0.0;
    for (int j = 0; j < cols; ++j)
        sum += exp((double)xr[j] - mx);
    double lse = log(sum);  // log(sum(exp(x - mx)))

    for (int j = 0; j < cols; ++j)
        yr[j] = __double2float_rn((double)xr[j] - mx - lse);
}

// Cross-entropy loss: -mean(log_softmax[i, target[i]])
__global__ void repro_nll_kernel(const float* __restrict__ log_probs,
                                 const int64_t* __restrict__ targets,
                                 float* __restrict__ loss,
                                 int batch_size, int num_classes) {
    // Single thread kernel for exact summation of the loss in double
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double total = 0.0;
    for (int i = 0; i < batch_size; ++i) {
        int t = (int)targets[i];
        total -= (double)log_probs[(size_t)i * num_classes + t];
    }
    // IEEE division by batch_size: pass as double to get correct rounding
    *loss = __double2float_rn(total / (double)batch_size);
}

// ---- Python-visible wrappers ----

torch::Tensor repro_exp(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    repro_exp_kernel<<<(n+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor repro_log(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    repro_log_kernel<<<(n+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor repro_sqrt(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    repro_sqrt_kernel<<<(n+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor repro_rsqrt(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    repro_rsqrt_kernel<<<(n+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor repro_tanh(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    repro_tanh_kernel<<<(n+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor repro_div_scalar(torch::Tensor x, float d) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    repro_div_scalar_kernel<<<(n+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), d, y.data_ptr<float>(), n);
    return y;
}

torch::Tensor repro_log_softmax(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "expected 2D input");
    auto y = torch::empty_like(x);
    int rows = x.size(0), cols = x.size(1);
    repro_log_softmax_kernel<<<rows, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    return y;
}

torch::Tensor repro_softmax(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "expected 2D input");
    auto y = torch::empty_like(x);
    int rows = x.size(0), cols = x.size(1);
    repro_softmax_kernel<<<rows, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    return y;
}

// Cross-entropy: returns scalar loss
torch::Tensor repro_cross_entropy(torch::Tensor logits, torch::Tensor targets) {
    TORCH_CHECK(logits.dim() == 2);
    TORCH_CHECK(targets.dim() == 1 && targets.dtype() == torch::kInt64);
    int B = logits.size(0), C = logits.size(1);

    // Step 1: log-softmax
    auto log_probs = torch::empty_like(logits);
    repro_log_softmax_kernel<<<B, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), log_probs.data_ptr<float>(), B, C);

    // Step 2: NLL
    auto loss = torch::empty({1}, logits.options());
    repro_nll_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        log_probs.data_ptr<float>(), targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(), B, C);

    return loss.squeeze(0);
}

// ---- im2col / col2im for reproducible Conv2d ----

__global__ void im2col_kernel_v2(const float* __restrict__ data_im,
                                  float* __restrict__ data_col,
                                  int C, int H, int W,
                                  int kH, int kW, int padH, int padW,
                                  int strideH, int strideW, int outH, int outW) {
    // Each thread writes one element of the column matrix.
    // col matrix is [C*kH*kW, outH*outW], row-major.
    int col_size = outH * outW;
    int col_rows = C * kH * kW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= col_rows * col_size) return;

    int col_col = idx % col_size;
    int col_row = idx / col_size;

    int ow = col_col % outW;
    int oh = col_col / outW;
    int kw = col_row % kW;
    int kh = (col_row / kW) % kH;
    int c  = col_row / (kH * kW);

    int ih = oh * strideH - padH + kh;
    int iw = ow * strideW - padW + kw;

    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        val = data_im[((size_t)c * H + ih) * W + iw];

    data_col[idx] = val;
}

__global__ void col2im_kernel(const float* __restrict__ data_col,
                              float* __restrict__ data_im,
                              int C, int H, int W,
                              int kH, int kW, int padH, int padW,
                              int strideH, int strideW, int outH, int outW) {
    // Each thread handles one pixel of the input image.
    // Accumulates all contributions from the column matrix.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * H * W) return;

    int iw = idx % W;
    int ih = (idx / W) % H;
    int c  = idx / (W * H);

    // Use double for exact accumulation (number of contributions is small: kH*kW)
    double val = 0.0;
    int col_size = outH * outW;
    for (int kh = 0; kh < kH; ++kh) {
        for (int kw_i = 0; kw_i < kW; ++kw_i) {
            int oh_num = ih + padH - kh;
            int ow_num = iw + padW - kw_i;
            if (oh_num % strideH != 0 || ow_num % strideW != 0) continue;
            int oh = oh_num / strideH;
            int ow = ow_num / strideW;
            if (oh < 0 || oh >= outH || ow < 0 || ow >= outW) continue;
            int col_row = (c * kH + kh) * kW + kw_i;
            int col_col = oh * outW + ow;
            val += (double)data_col[(size_t)col_row * col_size + col_col];
        }
    }
    data_im[idx] = (float)val;
}

torch::Tensor repro_im2col(torch::Tensor input, int kH, int kW,
                           int padH, int padW, int strideH, int strideW) {
    TORCH_CHECK(input.dim() == 3, "expected [C, H, W]");
    int C = input.size(0), H = input.size(1), W = input.size(2);
    int outH = (H + 2*padH - kH) / strideH + 1;
    int outW = (W + 2*padW - kW) / strideW + 1;
    int col_rows = C * kH * kW;
    int col_cols = outH * outW;
    auto col = torch::empty({col_rows, col_cols}, input.options());
    int total = col_rows * col_cols;
    im2col_kernel_v2<<<(total+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), col.data_ptr<float>(),
        C, H, W, kH, kW, padH, padW, strideH, strideW, outH, outW);
    return col;
}

torch::Tensor repro_col2im(torch::Tensor col, int C, int H, int W,
                           int kH, int kW, int padH, int padW,
                           int strideH, int strideW) {
    int outH = (H + 2*padH - kH) / strideH + 1;
    int outW = (W + 2*padW - kW) / strideW + 1;
    auto im = torch::zeros({C, H, W}, col.options());
    int total = C * H * W;
    col2im_kernel<<<(total+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        col.data_ptr<float>(), im.data_ptr<float>(),
        C, H, W, kH, kW, padH, padW, strideH, strideW, outH, outW);
    return im;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repro_mm", &repro_mm, "Reproducible matrix multiply",
          py::arg("A"), py::arg("B"),
          py::arg("trans_a") = false, py::arg("trans_b") = false,
          py::arg("slices") = 3);
    m.def("repro_exp", &repro_exp);
    m.def("repro_log", &repro_log);
    m.def("repro_sqrt", &repro_sqrt);
    m.def("repro_rsqrt", &repro_rsqrt);
    m.def("repro_tanh", &repro_tanh);
    m.def("repro_div_scalar", &repro_div_scalar);
    m.def("repro_softmax", &repro_softmax);
    m.def("repro_log_softmax", &repro_log_softmax);
    m.def("repro_cross_entropy", &repro_cross_entropy);
    m.def("repro_im2col", &repro_im2col);
    m.def("repro_col2im", &repro_col2im);
}
