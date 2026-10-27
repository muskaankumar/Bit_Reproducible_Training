#include "repro_gemm.cuh"
#include "repro_gemm_math.cuh"
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>

namespace repro_gemm {

// ---------------------------------------------------------------------------
// Layout
//
// Both operands are quantized into k-contiguous, zero-padded buffers:
//   op(A) -> [Mp][Kp], op(B) -> [Np][Kp]   (op(B) stored transposed)
// with Mp, Np multiples of 64 and Kp a multiple of 32. Padding is zero, so it adds
// nothing to any dot product and the GEMM kernels need no bounds checks.
// Fast paths store S int8 digit planes [S][Rp][Kp]; the Reference path stores
// the fixed-point integer X itself as int32 [Rp][Kp].
//
// Digits: X = sum_p d_p * 128^(S-1-p), p = 0 is most significant, |d_0| <= 127,
// d_p in [-64, 63] for p > 0 (balanced base-128, computed with exact integer ops).
//
// INT32 accumulation: pairs (i, j) with the same level L = i + j share one INT32
// accumulator. A level has at most S pairs and each digit product is <= 127^2, so
// over n values of k a level sum is <= n * S * 16129. make_plan limits one split to
// 4096/S k-tiles (n <= 131072/S), giving <= 2,114,060,288 < 2^31 - 1.
// Levels and splits are then combined exactly in INT64:
//   T = sum_L level_L * 2^(7 * (2S - 2 - L)) = sum_k X[i,k] * Y[j,k].
// ---------------------------------------------------------------------------

constexpr int TBM = 64, TBN = 64, TBK = 32;
constexpr int MAX_TILES_PER_SPLIT_S1 = 4096;  // 131072 / TBK
static_assert((int64_t)MAX_TILES_PER_SPLIT_S1 * TBK * 127 * 127 <= INT32_MAX, "INT32 level bound");

// Writes one quantized value: S digit planes, or the int32 fixed-point value.
template<int S, bool DIGITS>
__device__ __forceinline__ void emit(int X, void* out, size_t plane, size_t idx) {
    if (DIGITS) {
        int8_t* o = static_cast<int8_t*>(out);
#pragma unroll
        for (int p = S - 1; p >= 1; --p) {
            int d = ((X + 64) & 127) - 64;  // balanced digit in [-64, 63]
            o[p * plane + idx] = (int8_t)d;
            X = (X - d) / 128;              // exact: X - d is a multiple of 128
        }
        o[idx] = (int8_t)X;                 // |X| <= 127 here
    } else {
        static_cast<int32_t*>(out)[idx] = X;
    }
}

// Operand stored k-contiguous: element (r, k) at src[r*K + k]. Used for A (no
// transpose) and for B^T. One block per padded row; rows >= R are written as zero.
template<int S, bool DIGITS>
__global__ void quantize_kmajor(const float* __restrict__ src, void* __restrict__ out,
                                int* __restrict__ expo, int R, int K, int Rp, int Kp) {
    int row = blockIdx.x;
    size_t plane = (size_t)Rp * Kp;
    size_t base = (size_t)row * Kp;
    if (row >= R) {
        for (int k = threadIdx.x; k < Kp; k += blockDim.x) emit<S, DIGITS>(0, out, plane, base + k);
        if (threadIdx.x == 0) expo[row] = 0;
        return;
    }
    const float* a = src + (size_t)row * K;
    float mx = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(a[k]));
    __shared__ float sh[256];
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
        __syncthreads();
    }
    int se = scale_exponent(sh[0]);
    if (threadIdx.x == 0) expo[row] = se;
    for (int k = threadIdx.x; k < Kp; k += blockDim.x)
        emit<S, DIGITS>(k < K ? fixed_point(a[k], se, S) : 0, out, plane, base + k);
}

// Operand stored k-strided: element (r, k) at src[k*R + r]. Used for B (no
// transpose) and for A^T. One 32x8 block per 32 padded rows; reads are coalesced
// along r, and the transposed write goes through shared memory so it is coalesced along k.
template<int S, bool DIGITS>
__global__ void quantize_kstrided(const float* __restrict__ src, void* __restrict__ out,
                                  int* __restrict__ expo, int R, int K, int Rp, int Kp) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int r0 = blockIdx.x * 32;
    int r = r0 + tx;
    size_t plane = (size_t)Rp * Kp;
    __shared__ float red[8][33];
    __shared__ int se_sh[32];
    __shared__ float tile[32][33];

    float mx = 0.0f;
    if (r < R)
        for (int k = ty; k < K; k += 8) mx = fmaxf(mx, fabsf(src[(size_t)k * R + r]));
    red[ty][tx] = mx;
    __syncthreads();
    if (ty == 0) {
        for (int i = 1; i < 8; ++i) mx = fmaxf(mx, red[i][tx]);
        int se = scale_exponent(mx);  // rows >= R have mx == 0, so se == 0
        se_sh[tx] = se;
        expo[r] = se;
    }
    __syncthreads();

    for (int k0 = 0; k0 < Kp; k0 += 32) {
        for (int j = 0; j < 4; ++j) {
            int kr = ty + 8 * j, k = k0 + kr;
            tile[kr][tx] = (k < K && r < R) ? src[(size_t)k * R + r] : 0.0f;
        }
        __syncthreads();
        for (int j = 0; j < 4; ++j) {
            int rr = ty + 8 * j, k = k0 + tx;
            int X = (k < K && r0 + rr < R) ? fixed_point(tile[tx][rr], se_sh[rr], S) : 0;
            emit<S, DIGITS>(X, out, plane, (size_t)(r0 + rr) * Kp + k);
        }
        __syncthreads();
    }
}

// Final value for one output: exact integer T times 2^(eA + eB - 14(S-1)), rounded once.
__device__ __forceinline__ void write_output(int64_t t, int r, int c, int part, int M, int N, int Mp, int Np,
                                             int splitK, int e0, const int* eA, const int* eB,
                                             int64_t* partial, float* C) {
    if (splitK == 1) {
        if (r < M && c < N) C[(size_t)r * N + c] = round_to_float(t, eA[r] + eB[c] + e0);
    } else {
        partial[((size_t)part * Mp + r) * Np + c] = t;
    }
}

// ---------------------------------------------------------------------------
// Reference: the definition, written as simply as possible. One thread per
// output, T = sum_k X[r,k] * Y[c,k] in INT64 (|X*Y| < 2^42; make_plan bounds K).
// Always writes INT64 partials; the reduce kernel rounds.
// ---------------------------------------------------------------------------
__global__ void gemm_reference(const int32_t* __restrict__ X, const int32_t* __restrict__ Y,
                               int64_t* __restrict__ partial, int M, int N, int K,
                               int Mp, int Np, int Kp, int splitK) {
    int c = blockIdx.x * 16 + threadIdx.x;
    int r = blockIdx.y * 16 + threadIdx.y;
    int part = blockIdx.z;
    if (r >= M || c >= N) return;
    int k0 = (int)(((int64_t)K * part) / splitK);
    int k1 = (int)(((int64_t)K * (part + 1)) / splitK);
    const int32_t* x = X + (size_t)r * Kp;
    const int32_t* y = Y + (size_t)c * Kp;
    int64_t acc = 0;
    for (int k = k0; k < k1; ++k) acc += (int64_t)x[k] * y[k];
    partial[((size_t)part * Mp + r) * Np + c] = acc;
}

// Sums split partials and rounds. Used whenever split_k > 1, and by Reference.
__global__ void reduce_round(const int64_t* __restrict__ partial, const int* __restrict__ eA,
                             const int* __restrict__ eB, float* __restrict__ C,
                             int M, int N, int Mp, int Np, int splitK, int e0) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * N) return;
    int r = (int)(idx / N), c = (int)(idx % N);
    int64_t t = 0;
    for (int p = 0; p < splitK; ++p) t += partial[((size_t)p * Mp + r) * Np + c];
    C[idx] = round_to_float(t, eA[r] + eB[c] + e0);
}

// Sum of four int8*int8 products packed in 32-bit words, plus c.
__device__ __forceinline__ int dot4(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    for (int i = 0; i < 4; ++i) c += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
    return c;
#endif
}

// k-tiles [kt0, kt1) of split `part`; make_plan ensures every split gets >= 1 tile.
__device__ __forceinline__ void split_range(int Kp, int part, int splitK, int& kt0, int& kt1) {
    int nkt = Kp / TBK;
    kt0 = (int)(((int64_t)nkt * part) / splitK);
    kt1 = (int)(((int64_t)nkt * (part + 1)) / splitK);
}

// ---------------------------------------------------------------------------
// Tiled (dp4a). 64x64 output tile per CTA, 256 threads, each thread owns a 4x4
// sub-tile at stride 16. Each 32-byte k-slice of a row is 8 int32 words, padded to
// 9 so per-column reads are bank-conflict free. Next k-tile prefetched in registers.
// ---------------------------------------------------------------------------
template<int S>
__global__ void __launch_bounds__(256)
gemm_tiled_dp4a(const int8_t* __restrict__ QA, const int8_t* __restrict__ QB,
                const int* __restrict__ eA, const int* __restrict__ eB,
                int64_t* __restrict__ partial, float* __restrict__ C,
                int M, int N, int Mp, int Np, int Kp, int splitK) {
    constexpr int W = TBK / 4, L = 2 * S - 1;
    __shared__ int As[S][TBM][W + 1];
    __shared__ int Bs[S][TBN][W + 1];
    int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    int row0 = blockIdx.y * TBM, col0 = blockIdx.x * TBN, part = blockIdx.z;
    int kt0, kt1;
    split_range(Kp, part, splitK, kt0, kt1);

    // Threads 0-127 load A, 128-255 load B: one 16-byte vector per digit plane per k-tile.
    int lrow = (tid & 127) >> 1, lhalf = tid & 1;
    bool isA = tid < 128;
    size_t plane = isA ? (size_t)Mp * Kp : (size_t)Np * Kp;
    const int8_t* src = (isA ? QA + (size_t)(row0 + lrow) * Kp : QB + (size_t)(col0 + lrow) * Kp) + lhalf * 16;
    int (*dst)[TBM][W + 1] = isA ? As : Bs;

    int acc[L][4][4];
#pragma unroll
    for (int l = 0; l < L; ++l)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[l][i][j] = 0;

    int4 reg[S];
#pragma unroll
    for (int p = 0; p < S; ++p)
        reg[p] = kt0 < kt1 ? *reinterpret_cast<const int4*>(src + p * plane + (size_t)kt0 * TBK) : make_int4(0, 0, 0, 0);

    for (int kt = kt0; kt < kt1; ++kt) {
#pragma unroll
        for (int p = 0; p < S; ++p) {
            dst[p][lrow][lhalf * 4 + 0] = reg[p].x;
            dst[p][lrow][lhalf * 4 + 1] = reg[p].y;
            dst[p][lrow][lhalf * 4 + 2] = reg[p].z;
            dst[p][lrow][lhalf * 4 + 3] = reg[p].w;
        }
        __syncthreads();
        if (kt + 1 < kt1) {
#pragma unroll
            for (int p = 0; p < S; ++p)
                reg[p] = *reinterpret_cast<const int4*>(src + p * plane + (size_t)(kt + 1) * TBK);
        }
#pragma unroll
        for (int w = 0; w < W; ++w) {
            int a[S][4], b[S][4];
#pragma unroll
            for (int p = 0; p < S; ++p) {
#pragma unroll
                for (int i = 0; i < 4; ++i) a[p][i] = As[p][ty + 16 * i][w];
#pragma unroll
                for (int j = 0; j < 4; ++j) b[p][j] = Bs[p][tx + 16 * j][w];
            }
#pragma unroll
            for (int pi = 0; pi < S; ++pi)
#pragma unroll
                for (int pj = 0; pj < S; ++pj)
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            acc[pi + pj][i][j] = dot4(a[pi][i], b[pj][j], acc[pi + pj][i][j]);
        }
        __syncthreads();
    }

    int e0 = -14 * (S - 1);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t t = 0;
#pragma unroll
            for (int l = 0; l < L; ++l) t += (int64_t)acc[l][i][j] * ((int64_t)1 << (7 * (L - 1 - l)));
            write_output(t, row0 + ty + 16 * i, col0 + tx + 16 * j, part, M, N, Mp, Np, splitK, e0, eA, eB, partial, C);
        }
}

// ---------------------------------------------------------------------------
// TensorCore (WMMA INT8). 64x64 output tile per CTA, 8 warps in a 4x2 grid, each
// warp computes 16x32 as 1x2 fragments of 16x16 per level. Shared tiles are laid
// out as [plane][k-half][row][16 bytes] so every fragment pointer is 256-bit aligned.
// B tiles come from the [Np][Kp] buffer, which is WMMA col_major for B.
// Epilogue: each level's fragments go through a per-warp shared staging tile so
// they can be combined in INT64 at known (row, col) positions.
// ---------------------------------------------------------------------------
template<int S>
__global__ void __launch_bounds__(256)
gemm_wmma_i8(const int8_t* __restrict__ QA, const int8_t* __restrict__ QB,
             const int* __restrict__ eA, const int* __restrict__ eB,
             int64_t* __restrict__ partial, float* __restrict__ C,
             int M, int N, int Mp, int Np, int Kp, int splitK) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
    using namespace nvcuda;
    constexpr int L = 2 * S - 1;
    __shared__ __align__(32) int8_t As[S][2][TBM][16];
    __shared__ __align__(32) int8_t Bs[S][2][TBN][16];
    __shared__ __align__(32) int stage[8][16 * 32];
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wm = warp / 2, wn = warp % 2;
    int row0 = blockIdx.y * TBM, col0 = blockIdx.x * TBN, part = blockIdx.z;
    int kt0, kt1;
    split_range(Kp, part, splitK, kt0, kt1);

    int lrow = (tid & 127) >> 1, lhalf = tid & 1;
    bool isA = tid < 128;
    size_t plane = isA ? (size_t)Mp * Kp : (size_t)Np * Kp;
    const int8_t* src = (isA ? QA + (size_t)(row0 + lrow) * Kp : QB + (size_t)(col0 + lrow) * Kp) + lhalf * 16;
    int8_t (*dst)[2][TBM][16] = isA ? As : Bs;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[L][2];
#pragma unroll
    for (int l = 0; l < L; ++l)
#pragma unroll
        for (int f = 0; f < 2; ++f) wmma::fill_fragment(acc[l][f], 0);

    int4 reg[S];
#pragma unroll
    for (int p = 0; p < S; ++p)
        reg[p] = kt0 < kt1 ? *reinterpret_cast<const int4*>(src + p * plane + (size_t)kt0 * TBK) : make_int4(0, 0, 0, 0);

    for (int kt = kt0; kt < kt1; ++kt) {
#pragma unroll
        for (int p = 0; p < S; ++p) *reinterpret_cast<int4*>(&dst[p][lhalf][lrow][0]) = reg[p];
        __syncthreads();
        if (kt + 1 < kt1) {
#pragma unroll
            for (int p = 0; p < S; ++p)
                reg[p] = *reinterpret_cast<const int4*>(src + p * plane + (size_t)(kt + 1) * TBK);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa[S];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb[S][2];
#pragma unroll
            for (int p = 0; p < S; ++p) {
                wmma::load_matrix_sync(fa[p], &As[p][h][wm * 16][0], 16);
#pragma unroll
                for (int f = 0; f < 2; ++f) wmma::load_matrix_sync(fb[p][f], &Bs[p][h][wn * 32 + f * 16][0], 16);
            }
#pragma unroll
            for (int pi = 0; pi < S; ++pi)
#pragma unroll
                for (int pj = 0; pj < S; ++pj)
#pragma unroll
                    for (int f = 0; f < 2; ++f)
                        wmma::mma_sync(acc[pi + pj][f], fa[pi], fb[pj][f], acc[pi + pj][f]);
        }
        __syncthreads();
    }

    // Combine levels in INT64. Lane owns row q, column `lane` of the warp's 16x32 tile.
    int64_t t[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) t[q] = 0;
    int* st = stage[warp];
#pragma unroll
    for (int l = 0; l < L; ++l) {
#pragma unroll
        for (int f = 0; f < 2; ++f) wmma::store_matrix_sync(st + f * 16, acc[l][f], 32, wmma::mem_row_major);
        __syncwarp();
        int64_t w = (int64_t)1 << (7 * (L - 1 - l));
#pragma unroll
        for (int q = 0; q < 16; ++q) t[q] += (int64_t)st[q * 32 + lane] * w;
        __syncwarp();
    }
    int e0 = -14 * (S - 1);
#pragma unroll
    for (int q = 0; q < 16; ++q)
        write_output(t[q], row0 + wm * 16 + q, col0 + wn * 32 + lane, part, M, N, Mp, Np, splitK, e0, eA, eB, partial, C);
#endif
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

namespace {

constexpr size_t ALIGN = 256;
size_t align_up(size_t x) { return (x + ALIGN - 1) / ALIGN * ALIGN; }
int round_up(int x, int m) { return (x + m - 1) / m * m; }

struct Layout {
    Plan plan;
    int S, Mp, Np, Kp;
    size_t qa, qb, ea, eb, part;  // byte offsets into the workspace
    bool has_partial;
};

// Largest |X| after the digit split: 127*128^(S-1) + 64*(128^(S-2) + ... + 1).
double max_fixed_point(int S) {
    double top = 127.0 * std::ldexp(1.0, 7 * (S - 1)), low = 0.0;
    for (int t = 0; t < S - 1; ++t) low += 64.0 * std::ldexp(1.0, 7 * t);
    return top + low;
}

cudaError_t make_layout(int M, int N, int K, Config cfg, Layout& L) {
    if (M <= 0 || N <= 0 || K <= 0 || cfg.split_k <= 0) return cudaErrorInvalidValue;
    if (cfg.slices < 1 || cfg.slices > MAX_SLICES) return cudaErrorInvalidValue;
    int S = cfg.slices;
    // Every partial and final sum is bounded by K * max|X| * max|Y| <= 2^62.
    double d = max_fixed_point(S);
    if ((double)K * d * d > std::ldexp(1.0, 62)) return cudaErrorInvalidValue;

    Algo algo = cfg.algo;
    if (algo == Algo::Auto || algo == Algo::TensorCore) {
        int dev, major, minor;
        cudaError_t e;
        if ((e = cudaGetDevice(&dev)) != cudaSuccess) return e;
        if ((e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)) != cudaSuccess) return e;
        if ((e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev)) != cudaSuccess) return e;
        bool tc = major * 10 + minor >= 72;
        if (algo == Algo::TensorCore && !tc) return cudaErrorNotSupported;
        algo = tc ? Algo::TensorCore : Algo::Tiled;
    }

    L.S = S;
    L.Mp = round_up(M, TBM); L.Np = round_up(N, TBN); L.Kp = round_up(K, TBK);
    int split;
    size_t q_elem;
    if (algo == Algo::Reference) {
        split = cfg.split_k < K ? cfg.split_k : K;
        q_elem = sizeof(int32_t);  // one int32 fixed-point value per element
        L.has_partial = true;
    } else {
        int nkt = L.Kp / TBK, max_tiles = MAX_TILES_PER_SPLIT_S1 / S;
        int min_split = (nkt + max_tiles - 1) / max_tiles;
        split = cfg.split_k > min_split ? cfg.split_k : min_split;
        if (split > nkt) split = nkt;
        q_elem = (size_t)S;        // S int8 digit planes
        L.has_partial = split > 1;
    }
    size_t qa_bytes = (size_t)L.Mp * L.Kp * q_elem;
    size_t qb_bytes = (size_t)L.Np * L.Kp * q_elem;
    size_t part_bytes = L.has_partial ? (size_t)split * L.Mp * L.Np * sizeof(int64_t) : 0;

    L.qa = 0;
    L.qb = L.qa + align_up(qa_bytes);
    L.ea = L.qb + align_up(qb_bytes);
    L.eb = L.ea + align_up((size_t)L.Mp * sizeof(int));
    L.part = L.eb + align_up((size_t)L.Np * sizeof(int));
    L.plan.algo = algo;
    L.plan.split_k = split;
    L.plan.workspace_bytes = L.part + align_up(part_bytes);
    return cudaSuccess;
}

// Quantize op(A) -> [Mp][Kp] and op(B) -> [Np][Kp].
template<int S, bool DIGITS>
void launch_quantize(const float* A, const float* B, const Config& cfg, const Layout& L,
                     int M, int N, int K, void* QA, void* QB, int* eA, int* eB, cudaStream_t stream) {
    if (!cfg.trans_a) quantize_kmajor<S, DIGITS><<<L.Mp, 256, 0, stream>>>(A, QA, eA, M, K, L.Mp, L.Kp);
    else quantize_kstrided<S, DIGITS><<<L.Mp / 32, dim3(32, 8), 0, stream>>>(A, QA, eA, M, K, L.Mp, L.Kp);
    if (cfg.trans_b) quantize_kmajor<S, DIGITS><<<L.Np, 256, 0, stream>>>(B, QB, eB, N, K, L.Np, L.Kp);
    else quantize_kstrided<S, DIGITS><<<L.Np / 32, dim3(32, 8), 0, stream>>>(B, QB, eB, N, K, L.Np, L.Kp);
}

template<int S>
void launch_all(const float* A, const float* B, float* C, int M, int N, int K, const Config& cfg,
                const Layout& L, char* base, cudaStream_t stream, const cudaEvent_t* ev) {
    auto record = [&](int i) { if (ev) cudaEventRecord(ev[i], stream); };
    void* QA = base + L.qa;
    void* QB = base + L.qb;
    int* eA = reinterpret_cast<int*>(base + L.ea);
    int* eB = reinterpret_cast<int*>(base + L.eb);
    int64_t* partial = L.has_partial ? reinterpret_cast<int64_t*>(base + L.part) : nullptr;
    int split = L.plan.split_k;
    int e0 = -14 * (S - 1);
    unsigned reduce_blocks = (unsigned)(((size_t)M * N + 255) / 256);

    record(0);
    if (L.plan.algo == Algo::Reference) {
        launch_quantize<S, false>(A, B, cfg, L, M, N, K, QA, QB, eA, eB, stream);
        record(1);
        dim3 grid((N + 15) / 16, (M + 15) / 16, split);
        gemm_reference<<<grid, dim3(16, 16), 0, stream>>>(static_cast<int32_t*>(QA), static_cast<int32_t*>(QB),
                                                          partial, M, N, K, L.Mp, L.Np, L.Kp, split);
        record(2);
        reduce_round<<<reduce_blocks, 256, 0, stream>>>(partial, eA, eB, C, M, N, L.Mp, L.Np, split, e0);
    } else {
        launch_quantize<S, true>(A, B, cfg, L, M, N, K, QA, QB, eA, eB, stream);
        record(1);
        dim3 grid(L.Np / TBN, L.Mp / TBM, split);
        const int8_t* qa = static_cast<const int8_t*>(QA);
        const int8_t* qb = static_cast<const int8_t*>(QB);
        if (L.plan.algo == Algo::TensorCore)
            gemm_wmma_i8<S><<<grid, 256, 0, stream>>>(qa, qb, eA, eB, partial, C, M, N, L.Mp, L.Np, L.Kp, split);
        else
            gemm_tiled_dp4a<S><<<grid, 256, 0, stream>>>(qa, qb, eA, eB, partial, C, M, N, L.Mp, L.Np, L.Kp, split);
        record(2);
        // split == 1: the GEMM already wrote C (rounding fused into its epilogue).
        if (split > 1)
            reduce_round<<<reduce_blocks, 256, 0, stream>>>(partial, eA, eB, C, M, N, L.Mp, L.Np, split, e0);
    }
    record(3);
}

} // namespace

const char* algo_name(Algo a) {
    switch (a) {
        case Algo::Auto: return "auto";
        case Algo::Reference: return "ref";
        case Algo::Tiled: return "tiled";
        case Algo::TensorCore: return "tc";
    }
    return "?";
}

cudaError_t make_plan(int M, int N, int K, Config cfg, Plan* plan) {
    if (!plan) return cudaErrorInvalidValue;
    Layout L;
    cudaError_t e = make_layout(M, N, K, cfg, L);
    if (e == cudaSuccess) *plan = L.plan;
    return e;
}

cudaError_t repro_gemm_fp32_ws(const float* A, const float* B, float* C,
                               int M, int N, int K, Config cfg,
                               void* workspace, size_t workspace_bytes,
                               cudaStream_t stream, const cudaEvent_t* ev) {
    if (!A || !B || !C) return cudaErrorInvalidValue;
    Layout L;
    cudaError_t e = make_layout(M, N, K, cfg, L);
    if (e != cudaSuccess) return e;
    if (!workspace || workspace_bytes < L.plan.workspace_bytes) return cudaErrorInvalidValue;
    char* base = static_cast<char*>(workspace);
    switch (L.S) {
        case 1: launch_all<1>(A, B, C, M, N, K, cfg, L, base, stream, ev); break;
        case 2: launch_all<2>(A, B, C, M, N, K, cfg, L, base, stream, ev); break;
        case 3: launch_all<3>(A, B, C, M, N, K, cfg, L, base, stream, ev); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t repro_gemm_fp32(const float* A, const float* B, float* C,
                            int M, int N, int K, Config cfg, cudaStream_t stream) {
    Plan plan;
    cudaError_t e = make_plan(M, N, K, cfg, &plan);
    if (e != cudaSuccess) return e;
    void* ws = nullptr;
    if ((e = cudaMalloc(&ws, plan.workspace_bytes)) != cudaSuccess) return e;
    e = repro_gemm_fp32_ws(A, B, C, M, N, K, cfg, ws, plan.workspace_bytes, stream);
    // Synchronize to surface asynchronous kernel errors at this API boundary.
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    cudaFree(ws);
    return e;
}

} // namespace repro_gemm
