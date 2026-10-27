// Correctness suite. For every S in 1..3 and every algorithm available on this GPU:
//   - output equals the exact CPU implementation of the definition, bit for bit;
//   - repeated runs and several split_k values give identical bits;
//   - all four transpose combinations (A or A^T stored, B or B^T stored) give
//     the same bits as the plain case, i.e. C = A*B, C = A*B via B^T, etc.
// Usage: test_repro_gemm [M] [N] [K]   (defaults are deliberately not multiples of 16/32)
#include "repro_test_util.h"

static int failures = 0;
static void expect(bool ok, const char* what, int S, const char* algo, int split) {
    if (!ok) {
        std::printf("FAIL %-28s S=%d algo=%s split_k=%d\n", what, S, algo, split);
        ++failures;
    }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 133;
    int N = argc > 2 ? std::atoi(argv[2]) : 77;
    int K = argc > 3 ? std::atoi(argv[3]) : 301;
    size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    std::vector<float> hA(nA), hB(nB), hAt(nA), hBt(nB), out(nC), cpu;
    Rng g{2024};
    for (auto& x : hA) x = (float)g.normal();
    for (auto& x : hB) x = (float)g.normal();
    hA[5] = 1e6f;  // an outlier, so one row's scale differs a lot from the rest
    for (int r = 0; r < M; ++r) for (int k = 0; k < K; ++k) hAt[(size_t)k * M + r] = hA[(size_t)r * K + k];
    for (int k = 0; k < K; ++k) for (int c = 0; c < N; ++c) hBt[(size_t)c * K + k] = hB[(size_t)k * N + c];

    float *dA, *dB, *dAt, *dBt, *dC;
    ck(cudaMalloc(&dA, nA * 4), "malloc"); ck(cudaMalloc(&dAt, nA * 4), "malloc");
    ck(cudaMalloc(&dB, nB * 4), "malloc"); ck(cudaMalloc(&dBt, nB * 4), "malloc");
    ck(cudaMalloc(&dC, nC * 4), "malloc");
    ck(cudaMemcpy(dA, hA.data(), nA * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(dAt, hAt.data(), nA * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(dB, hB.data(), nB * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(dBt, hBt.data(), nB * 4, cudaMemcpyHostToDevice), "copy");

    // Runs one GEMM and returns whether its output equals `want` bit for bit.
    auto run_equals = [&](const float* a, const float* b, repro_gemm::Config cfg, const std::vector<float>& want) {
        ck(cudaMemset(dC, 0xFF, nC * 4), "poison C");
        ck(repro_gemm::repro_gemm_fp32(a, b, dC, M, N, K, cfg), "GEMM");
        ck(cudaMemcpy(out.data(), dC, nC * 4, cudaMemcpyDeviceToHost), "copy C");
        size_t first;
        return count_bit_diffs(out, want, &first) == 0;
    };

    std::printf("M=%d N=%d K=%d\n", M, N, K);
    for (int S = 1; S <= repro_gemm::MAX_SLICES; ++S) {
        cpu_reference(hA, hB, cpu, M, N, K, S);
        for (const char* name : {"ref", "tiled", "tc"}) {
            repro_gemm::Config cfg;
            cfg.algo = parse_algo(name);
            cfg.slices = S;
            repro_gemm::Plan plan;
            cudaError_t pe = repro_gemm::make_plan(M, N, K, cfg, &plan);
            if (pe == cudaErrorNotSupported) { std::printf("S=%d %-5s skipped (not supported on this GPU)\n", S, name); continue; }
            ck(pe, "plan");
            int before = failures;
            for (int split : {1, 2, 3, 7, 64}) {
                cfg.split_k = split;
                cfg.trans_a = cfg.trans_b = false;
                expect(run_equals(dA, dB, cfg, cpu), "matches CPU definition", S, name, split);
                expect(run_equals(dA, dB, cfg, cpu), "repeat run", S, name, split);
                cfg.trans_a = true;
                expect(run_equals(dAt, dB, cfg, cpu), "trans_a", S, name, split);
                cfg.trans_b = true;
                expect(run_equals(dAt, dBt, cfg, cpu), "trans_a + trans_b", S, name, split);
                cfg.trans_a = false;
                expect(run_equals(dA, dBt, cfg, cpu), "trans_b", S, name, split);
            }
            std::printf("S=%d %-5s %s\n", S, name, failures == before ? "PASS" : "FAIL");
        }
    }

    cudaFree(dA); cudaFree(dAt); cudaFree(dB); cudaFree(dBt); cudaFree(dC);
    if (failures) { std::printf("FAILED: %d checks\n", failures); return 2; }
    std::puts("PASS: all S, algorithms, split_k values, and transposes match the CPU definition bit for bit");
    return 0;
}
