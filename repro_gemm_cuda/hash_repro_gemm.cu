// Hashes the raw output bits of repro_gemm_fp32 across repeated runs, split-K
// values, and algorithms. Every (algo, split, run) must produce the same hash. The printed
// output hash can also be compared across processes, GPUs, and builds.
// For small problems it also checks every output bit against an exact CPU
// implementation of the definition, and reports accuracy against FP64.
//
// Usage: hash_repro_gemm [M] [N] [K] [runs] [splits] [seed] [dist] [algos] [slices]
//   slices: S, 1..3 (default 3)
//   splits: comma-separated list, e.g. 1,2,3,7,16
//   algos:  comma-separated from ref,tiled,tc,auto (default ref,tiled,tc;
//           tc is skipped on GPUs without INT8 Tensor Cores)
//   dist:   uniform  U[-1,1)
//           normal   N(0,1)
//           weights  N(0,0.02) like transformer init, 0.1% outliers x100
//           wide     random sign * 2^U[-40,40] * [1,2): huge dynamic range
//           stress   normal plus zero rows/cols, subnormal rows, 1e30 rows, mixed-scale cols
//           file:A.bin,B.bin   raw little-endian float32, row-major A[M,K] and B[K,N]
#include "repro_test_util.h"

static void fill(const std::string& dist, Rng& g, std::vector<float>& A, std::vector<float>& B,
                 int M, int N, int K) {
    if (dist == "uniform") {
        for (auto& x : A) x = g.uniform_pm1();
        for (auto& x : B) x = g.uniform_pm1();
    } else if (dist == "normal") {
        for (auto& x : A) x = (float)g.normal();
        for (auto& x : B) x = (float)g.normal();
    } else if (dist == "weights") {
        auto w = [&]() { double v = 0.02 * g.normal(); return (float)((g.next() % 1000 == 0) ? v * 100.0 : v); };
        for (auto& x : A) x = w();
        for (auto& x : B) x = w();
    } else if (dist == "wide") {
        for (auto& x : A) x = g.wide(-40, 40);
        for (auto& x : B) x = g.wide(-40, 40);
    } else if (dist == "stress") {
        for (auto& x : A) x = (float)g.normal();
        for (auto& x : B) x = (float)g.normal();
        for (int r = 0; r < M; ++r) {
            float* row = &A[(size_t)r * K];
            switch (r % 8) {
                case 1: for (int k = 0; k < K; ++k) row[k] = 0.0f; break;                    // zero row
                case 2: for (int k = 0; k < K; ++k) row[k] = (float)std::ldexp(g.normal(), -140); break; // subnormal
                case 3: for (int k = 0; k < K; ++k) row[k] = (float)(1e30 * g.normal()); break;        // huge
                case 4: row[g.range(0, K - 1)] = 1e6f; break;                                // one outlier
                case 5: for (int k = 0; k < K; ++k) row[k] = (k & 1) ? -0.0f : 0.0f; break;  // signed zeros
                default: break;
            }
        }
        for (int c = 0; c < N; ++c) {
            switch (c % 6) {
                case 1: for (int k = 0; k < K; ++k) B[(size_t)k * N + c] = 0.0f; break;     // zero column
                case 2: for (int k = 0; k < K; ++k) B[(size_t)k * N + c] = g.wide(-60, 60); break;
                case 3: for (int k = 0; k < K; ++k) B[(size_t)k * N + c] = (float)std::ldexp(g.normal(), -120); break;
                default: break;
            }
        }
    } else if (dist.rfind("file:", 0) == 0) {
        std::string spec = dist.substr(5);
        size_t comma = spec.find(',');
        if (comma == std::string::npos) { std::fprintf(stderr, "file: needs A.bin,B.bin\n"); std::exit(1); }
        auto load = [](const std::string& path, std::vector<float>& v) {
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
            size_t got = std::fread(v.data(), sizeof(float), v.size(), f);
            int extra = std::fgetc(f);
            std::fclose(f);
            if (got != v.size() || extra != EOF) {
                std::fprintf(stderr, "%s: expected exactly %zu float32 values\n", path.c_str(), v.size());
                std::exit(1);
            }
        };
        load(spec.substr(0, comma), A);
        load(spec.substr(comma + 1), B);
    } else {
        std::fprintf(stderr, "unknown dist '%s'\n", dist.c_str()); std::exit(1);
    }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 256;
    int N = argc > 2 ? std::atoi(argv[2]) : 256;
    int K = argc > 3 ? std::atoi(argv[3]) : 1024;
    int runs = argc > 4 ? std::atoi(argv[4]) : 5;
    std::vector<int> splits;
    for (auto& t : split_list(argc > 5 ? argv[5] : "1,2,3,4,7,8,16,64")) splits.push_back(std::atoi(t.c_str()));
    uint64_t seed = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 12345;
    std::string dist = argc > 7 ? argv[7] : "normal";
    std::vector<std::string> algos = split_list(argc > 8 ? argv[8] : "ref,tiled,tc");
    int S = argc > 9 ? std::atoi(argv[9]) : 3;

    size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    std::vector<float> hA(nA), hB(nB), hC(nC), ref;
    Rng g{seed};
    fill(dist, g, hA, hB, M, N, K);

    // NaN/Inf are outside the API contract; refuse rather than report a meaningless hash.
    for (float x : hA) if (!std::isfinite(x)) { std::fprintf(stderr, "A contains NaN/Inf (unsupported)\n"); return 1; }
    for (float x : hB) if (!std::isfinite(x)) { std::fprintf(stderr, "B contains NaN/Inf (unsupported)\n"); return 1; }

    cudaDeviceProp prop;
    int dev = 0;
    ck(cudaGetDevice(&dev), "get device");
    ck(cudaGetDeviceProperties(&prop, dev), "device props");
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::printf("device: %s (sm_%d%d)  runtime %d  driver %d\n", prop.name, prop.major, prop.minor, rt, drv);
    std::printf("M=%d N=%d K=%d S=%d runs=%d seed=%llu dist=%s\n", M, N, K, S, runs, (unsigned long long)seed, dist.c_str());
    std::printf("input hash A=%016llx B=%016llx\n",
                (unsigned long long)fnv1a64(hA.data(), nA * sizeof(float)),
                (unsigned long long)fnv1a64(hB.data(), nB * sizeof(float)));

    float *dA, *dB, *dC;
    ck(cudaMalloc(&dA, nA * sizeof(float)), "malloc A");
    ck(cudaMalloc(&dB, nB * sizeof(float)), "malloc B");
    ck(cudaMalloc(&dC, nC * sizeof(float)), "malloc C");
    ck(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    ck(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice), "copy B");

    uint64_t ref_hash = 0;
    bool have_ref = false, all_match = true;
    for (const std::string& algo : algos) {
    for (int split : splits) {
        repro_gemm::Config cfg;
        cfg.split_k = split;
        cfg.algo = parse_algo(algo);
        cfg.slices = S;
        repro_gemm::Plan plan;
        cudaError_t pe = repro_gemm::make_plan(M, N, K, cfg, &plan);
        if (pe == cudaErrorNotSupported) { std::printf("%-5s skipped: not supported on this GPU\n", algo.c_str()); break; }
        ck(pe, "plan");
        std::printf("%-5s split_k=%-4d", algo.c_str(), split);
        if (plan.split_k != split) std::printf("(ran %-3d)", plan.split_k);
        for (int r = 0; r < runs; ++r) {
            // Poison C (0xFF bytes = NaN) so stale output from a previous run cannot mask
            // a kernel that failed to write some elements.
            ck(cudaMemset(dC, 0xFF, nC * sizeof(float)), "poison C");
            ck(repro_gemm::repro_gemm_fp32(dA, dB, dC, M, N, K, cfg), "GEMM");
            ck(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost), "copy C");
            uint64_t h = fnv1a64(hC.data(), nC * sizeof(float));
            std::printf(" %016llx", (unsigned long long)h);
            if (!have_ref) { ref_hash = h; ref = hC; have_ref = true; continue; }
            if (h != ref_hash) {
                all_match = false;
                size_t first, diffs = count_bit_diffs(hC, ref, &first);
                std::printf(" [MISMATCH: %zu elems differ, first idx=%zu (r=%zu,c=%zu) %.9g vs ref %.9g]",
                            diffs, first, first / N, first % N, hC[first], ref[first]);
            }
        }
        std::printf("\n");
    }
    }

    size_t nonfinite = 0;
    for (float x : ref) if (!std::isfinite(x)) ++nonfinite;
    if (nonfinite) std::printf("note: %zu output elements are non-finite (FP32 overflow of the true result)\n", nonfinite);

    // Exact check: every output bit must equal the CPU implementation of the definition.
    // Then accuracy vs FP64 (informational). Both are O(MNK) on the host, so skipped
    // for large problems.
    bool exact_ok = true;
    if ((double)M * N * K <= (double)(1u << 28)) {
        std::vector<float> cpu;
        cpu_reference(hA, hB, cpu, M, N, K, S);
        size_t first, diffs = count_bit_diffs(ref, cpu, &first);
        exact_ok = diffs == 0;
        if (exact_ok) std::printf("exact vs CPU definition: MATCH (all %zu elements)\n", nC);
        else std::printf("exact vs CPU definition: MISMATCH %zu elements, first (r=%zu,c=%zu) gpu %a cpu %a\n",
                         diffs, first / N, first % N, ref[first], cpu[first]);

        std::vector<double> Bt(nB);
        for (int k = 0; k < K; ++k) for (int c = 0; c < N; ++c) Bt[(size_t)c * K + k] = hB[(size_t)k * N + c];
        double err2 = 0, ref2 = 0, max_rel_row = 0;
        for (int r = 0; r < M; ++r) {
            double rowerr2 = 0, rowref2 = 0;
            for (int c = 0; c < N; ++c) {
                double s = 0;
                const float* a = &hA[(size_t)r * K];
                const double* b = &Bt[(size_t)c * K];
                for (int k = 0; k < K; ++k) s += (double)a[k] * b[k];
                double d = (double)ref[(size_t)r * N + c] - s;
                if (std::isfinite(d)) { rowerr2 += d * d; rowref2 += s * s; }
            }
            err2 += rowerr2; ref2 += rowref2;
            if (rowref2 > 0) max_rel_row = std::fmax(max_rel_row, std::sqrt(rowerr2 / rowref2));
        }
        std::printf("accuracy vs FP64: relative Frobenius error %.3e, worst row %.3e\n",
                    ref2 > 0 ? std::sqrt(err2 / ref2) : 0.0, max_rel_row);
    } else {
        std::printf("exact vs CPU definition: skipped (M*N*K > 2^28)\n");
        std::printf("accuracy vs FP64: skipped (M*N*K > 2^28)\n");
    }

    std::printf("output hash: %016llx\n", (unsigned long long)ref_hash);
    // One grep-able line for cross-process / cross-machine comparison.
    bool pass = all_match && exact_ok;
    std::printf("RESULT M=%d N=%d K=%d S=%d seed=%llu dist=%s hash=%016llx status=%s\n", M, N, K, S,
                (unsigned long long)seed, dist.c_str(), (unsigned long long)ref_hash, pass ? "PASS" : "FAIL");
    if (!all_match) { std::puts("FAIL: outputs are not bit-identical across runs/splits/algos"); return 2; }
    if (!exact_ok) { std::puts("FAIL: output differs from the exact CPU definition"); return 2; }
    std::puts("PASS: all runs, split_k values, and algos bit-identical");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
