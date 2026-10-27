#pragma once
// Helpers shared by the test, hash, and benchmark programs.
#include "repro_gemm.cuh"
#include "repro_gemm_math.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

inline void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e)); std::exit(1); }
}

// 64-bit FNV-1a over raw bytes. Not cryptographic, but any bit flip changes it.
inline uint64_t fnv1a64(const void* data, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

// Portable input generation. std:: distributions and libm (log/cos) can differ
// across standard libraries, so everything here uses only SplitMix64, exact
// integer->float conversions, additions, and ldexp. The same seed then gives the
// same input bits on every machine.
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    double u01() { return (double)(next() >> 40) * (1.0 / 16777216.0); }  // exact, [0,1)
    float uniform_pm1() { return (float)(2.0 * u01() - 1.0); }
    // Irwin-Hall: sum of 12 uniforms minus 6 ~ N(0,1). Sum is exact in double.
    double normal() { double t = 0; for (int i = 0; i < 12; ++i) t += u01(); return t - 6.0; }
    int range(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }
    float wide(int elo, int ehi) {
        double mant = 1.0 + u01();
        double v = std::ldexp(mant, range(elo, ehi));
        return (float)((next() & 1) ? -v : v);
    }
};

inline std::vector<std::string> split_list(const char* s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(tok);
    return v;
}

inline repro_gemm::Algo parse_algo(const std::string& s) {
    using repro_gemm::Algo;
    if (s == "ref") return Algo::Reference;
    if (s == "tiled") return Algo::Tiled;
    if (s == "tc") return Algo::TensorCore;
    if (s == "auto") return Algo::Auto;
    std::fprintf(stderr, "unknown algo '%s' (use ref, tiled, tc, auto)\n", s.c_str());
    std::exit(1);
}

// Exact CPU implementation of the definition in repro_gemm.cuh, written directly
// from the math (no digits, no tiling), for C = A * B with row-major A[M,K], B[K,N].
// Uses the same scalar helpers as the GPU, so the GPU output must match bit for bit.
inline void cpu_reference(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<float>& C, int M, int N, int K, int S) {
    using namespace repro_gemm;
    std::vector<int> eA(M), eB(N);
    std::vector<int32_t> X((size_t)M * K), Y((size_t)N * K);  // Y is op(B) transposed: [N][K]
    for (int r = 0; r < M; ++r) {
        float m = 0.0f;
        for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(A[(size_t)r * K + k]));
        eA[r] = scale_exponent(m);
        for (int k = 0; k < K; ++k) X[(size_t)r * K + k] = fixed_point(A[(size_t)r * K + k], eA[r], S);
    }
    for (int c = 0; c < N; ++c) {
        float m = 0.0f;
        for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(B[(size_t)k * N + c]));
        eB[c] = scale_exponent(m);
        for (int k = 0; k < K; ++k) Y[(size_t)c * K + k] = fixed_point(B[(size_t)k * N + c], eB[c], S);
    }
    C.assign((size_t)M * N, 0.0f);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            const int32_t* x = &X[(size_t)r * K];
            const int32_t* y = &Y[(size_t)c * K];
            int64_t t = 0;
            for (int k = 0; k < K; ++k) t += (int64_t)x[k] * y[k];
            C[(size_t)r * N + c] = round_to_float(t, eA[r] + eB[c] - 14 * (S - 1));
        }
}

// Number of elements whose bits differ; index of the first one in *first.
inline size_t count_bit_diffs(const std::vector<float>& a, const std::vector<float>& b, size_t* first) {
    size_t n = 0;
    *first = a.size();
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y;
        std::memcpy(&x, &a[i], 4); std::memcpy(&y, &b[i], 4);
        if (x != y) { if (!n) *first = i; ++n; }
    }
    return n;
}
