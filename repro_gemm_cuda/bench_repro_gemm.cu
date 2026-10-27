// Kernel-level benchmark for repro_gemm.
//
// Usage: bench_repro_gemm [M] [N] [K] [split] [iters] [algos] [slices]
//   algos:  comma-separated from ref,tiled,tc,auto (default tiled,tc)
//   slices: S, 1..3 (default 3)
//
// The workspace is allocated once outside the timed loop, every algorithm is
// warmed up, and each stage (quantize, GEMM, reduce) is timed with CUDA events.
// Times are reported as median / min / p90 over `iters` runs. Every algorithm's
// output hash is checked against the Reference kernel, so an optimization that
// breaks bit-identity is reported as a FAIL, not as a speedup. cuBLAS FP32 SGEMM
// on the same inputs is timed as the speed you would be replacing.
#include "repro_test_util.h"
#include <cublas_v2.h>
#include <algorithm>

struct Stats { float med, min, p90; };
static Stats stats(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return {v[n / 2], v[0], v[std::min(n - 1, n * 9 / 10)]};
}

static void ckb(cublasStatus_t s, const char* where) {
    if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: cuBLAS error %d\n", where, (int)s); std::exit(1); }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 4096;
    int N = argc > 2 ? std::atoi(argv[2]) : 4096;
    int K = argc > 3 ? std::atoi(argv[3]) : 4096;
    int split = argc > 4 ? std::atoi(argv[4]) : 1;
    int iters = argc > 5 ? std::atoi(argv[5]) : 50;
    std::vector<std::string> algos = split_list(argc > 6 ? argv[6] : "tiled,tc");
    int S = argc > 7 ? std::atoi(argv[7]) : 3;
    const int warmup = 3;

    cudaDeviceProp prop;
    int dev = 0;
    ck(cudaGetDevice(&dev), "get device");
    ck(cudaGetDeviceProperties(&prop, dev), "device props");
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("M=%d N=%d K=%d S=%d split_k=%d iters=%d (times in ms: median [min, p90])\n", M, N, K, S, split, iters);

    size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    std::vector<float> hA(nA), hB(nB), hC(nC);
    Rng g{12345};
    for (auto& x : hA) x = (float)g.normal();
    for (auto& x : hB) x = (float)g.normal();
    float *dA, *dB, *dC;
    ck(cudaMalloc(&dA, nA * sizeof(float)), "malloc A");
    ck(cudaMalloc(&dB, nB * sizeof(float)), "malloc B");
    ck(cudaMalloc(&dC, nC * sizeof(float)), "malloc C");
    ck(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    ck(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice), "copy B");

    auto output_hash = [&]() {
        ck(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost), "copy C");
        return fnv1a64(hC.data(), nC * sizeof(float));
    };

    // Expected bits: one untimed run of the Reference kernel.
    repro_gemm::Config ref_cfg;
    ref_cfg.algo = repro_gemm::Algo::Reference;
    ref_cfg.slices = S;
    ck(repro_gemm::repro_gemm_fp32(dA, dB, dC, M, N, K, ref_cfg), "reference");
    uint64_t ref_hash = output_hash();
    std::printf("reference hash: %016llx\n\n", (unsigned long long)ref_hash);

    cudaEvent_t ev[4];
    for (auto& e : ev) ck(cudaEventCreate(&e), "event");
    // Effective rate: 2*M*N*K over time, i.e. FP32-GEMM-equivalent work, so rows are
    // directly comparable with cuBLAS. S slices do S^2 times this many int8 operations.
    // "final" is the split-K reduction; with split_k == 1 it is fused into the GEMM (~0).
    double ops = 2.0 * M * N * K;
    bool all_match = true;

    std::printf("%-6s %5s %-24s %-24s %-24s %-24s %9s %9s  %s\n",
                "algo", "split", "quantize", "gemm", "final", "total", "gemmTOPS", "e2eTOPS", "bits");
    auto fmt = [](Stats s) {
        static char buf[4][64];
        static int k = 0;
        char* b = buf[k++ % 4];
        std::snprintf(b, 64, "%.3f [%.3f, %.3f]", s.med, s.min, s.p90);
        return b;
    };

    for (const std::string& name : algos) {
        repro_gemm::Config cfg;
        cfg.split_k = split;
        cfg.algo = parse_algo(name);
        cfg.slices = S;
        repro_gemm::Plan plan;
        cudaError_t pe = repro_gemm::make_plan(M, N, K, cfg, &plan);
        if (pe == cudaErrorNotSupported) { std::printf("%-6s skipped: not supported on this GPU\n", name.c_str()); continue; }
        ck(pe, "plan");
        void* ws;
        ck(cudaMalloc(&ws, plan.workspace_bytes), "workspace");
        ck(cudaMemset(dC, 0xFF, nC * sizeof(float)), "poison C");

        for (int i = 0; i < warmup; ++i)
            ck(repro_gemm::repro_gemm_fp32_ws(dA, dB, dC, M, N, K, cfg, ws, plan.workspace_bytes), "warmup");
        ck(cudaDeviceSynchronize(), "warmup sync");

        std::vector<float> tq, tg, tr, tt;
        for (int i = 0; i < iters; ++i) {
            ck(repro_gemm::repro_gemm_fp32_ws(dA, dB, dC, M, N, K, cfg, ws, plan.workspace_bytes, nullptr, ev), "run");
            ck(cudaEventSynchronize(ev[3]), "sync");
            float q, gm, r, t;
            ck(cudaEventElapsedTime(&q, ev[0], ev[1]), "elapsed");
            ck(cudaEventElapsedTime(&gm, ev[1], ev[2]), "elapsed");
            ck(cudaEventElapsedTime(&r, ev[2], ev[3]), "elapsed");
            ck(cudaEventElapsedTime(&t, ev[0], ev[3]), "elapsed");
            tq.push_back(q); tg.push_back(gm); tr.push_back(r); tt.push_back(t);
        }
        bool match = output_hash() == ref_hash;
        all_match &= match;
        Stats sg = stats(tg), st = stats(tt);
        std::printf("%-6s %5d %-24s %-24s %-24s %-24s %9.2f %9.2f  %s\n",
                    repro_gemm::algo_name(plan.algo), plan.split_k, fmt(stats(tq)), fmt(sg), fmt(stats(tr)), fmt(st),
                    ops / (sg.med * 1e9), ops / (st.med * 1e9), match ? "MATCH" : "MISMATCH");
        cudaFree(ws);
    }

    // cuBLAS FP32 SGEMM. Row-major C = A*B is column-major C^T = B^T * A^T.
    cublasHandle_t h;
    ckb(cublasCreate(&h), "cublasCreate");
    const float one = 1.0f, zero = 0.0f;
    auto sgemm = [&]() { ckb(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, dB, N, dA, K, &zero, dC, N), "sgemm"); };
    for (int i = 0; i < warmup; ++i) sgemm();
    ck(cudaDeviceSynchronize(), "cublas warmup");
    std::vector<float> tc;
    for (int i = 0; i < iters; ++i) {
        ck(cudaEventRecord(ev[0]), "record");
        sgemm();
        ck(cudaEventRecord(ev[1]), "record");
        ck(cudaEventSynchronize(ev[1]), "sync");
        float t;
        ck(cudaEventElapsedTime(&t, ev[0], ev[1]), "elapsed");
        tc.push_back(t);
    }
    Stats sc = stats(tc);
    std::printf("%-6s %5s %-24s %-24s %-24s %-24s %9.2f %9.2f  %s\n",
                "cublas", "-", "-", fmt(sc), "-", fmt(sc), ops / (sc.med * 1e9), ops / (sc.med * 1e9), "(FP32, not reproducible)");
    cublasDestroy(h);

    for (auto& e : ev) cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (!all_match) { std::puts("\nFAIL: an algorithm's output differs from the reference kernel"); return 2; }
    return 0;
}
