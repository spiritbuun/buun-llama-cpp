// EXP-40: Weaver scorer parity gate.
// Loads the weaver-scorer GGUF + golden vectors (weaver_golden_gen.py) and checks the
// ggml implementation reproduces the numpy reference (fp32 math, same f16 weights):
// prefix pass + a 3-node root chain, comparing the 512 candidate logits per node.
//
// usage: llama-weaver-parity <weaver.gguf> <weaver_golden.bin> [tol]

#include "weaver.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <weaver.gguf> <weaver_golden.bin> [tol]\n", argv[0]);
        return 1;
    }
    const float tol = argc > 3 ? (float) atof(argv[3]) : 5e-3f;

    std::ifstream f(argv[2], std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }
    char magic[4];
    int32_t version, k_depth, pool, n_nodes;
    f.read(magic, 4);
    f.read((char *) &version, 4);
    f.read((char *) &k_depth, 4);
    f.read((char *) &pool, 4);
    f.read((char *) &n_nodes, 4);
    if (memcmp(magic, "WGLD", 4) != 0 || version != 1) {
        fprintf(stderr, "bad golden file\n");
        return 1;
    }

    weaver_scorer * ws = weaver_init(argv[1], /*prefer_gpu=*/true, /*max_nodes=*/64);
    if (!ws) {
        return 1;
    }
    const auto & p = weaver_get_params(ws);
    if (p.depth_cap != k_depth || p.pool_size != pool) {
        fprintf(stderr, "golden/gguf mismatch: depth %d vs %d, pool %d vs %d\n",
                k_depth, p.depth_cap, pool, p.pool_size);
        return 1;
    }

    auto readv = [&](size_t n) { std::vector<float> v(n); f.read((char *) v.data(), n * 4); return v; };
    const auto tfh        = readv(p.d_model);
    const auto dh         = readv((size_t) k_depth * p.d_model);
    const auto embed_rows = readv((size_t) n_nodes * p.d_model);
    const auto lm_rows    = readv((size_t) pool * p.d_model);
    const auto scores     = readv(pool);
    const auto golden     = readv((size_t) n_nodes * pool);
    if (!f) {
        fprintf(stderr, "golden file truncated\n");
        return 1;
    }

    weaver_begin_step(ws, tfh.data(), dh.data(), k_depth);
    for (int d = 0; d < n_nodes; ++d) {
        // golden uses one shared candidate pool at every depth
        weaver_set_candidates(ws, d, lm_rows.data(), scores.data(), pool);
    }

    // 3-node root chain: node at depth d has ancestors = slots [0..d-1]
    std::vector<int32_t> anc;
    float max_abs = 0.0f, max_rel = 0.0f;
    bool ok = true;
    std::vector<float> logits(pool);
    for (int d = 0; d < n_nodes; ++d) {
        if (!weaver_expand(ws, embed_rows.data() + (size_t) d * p.d_model, d,
                           anc.data(), (int) anc.size(), d, logits.data())) {
            fprintf(stderr, "expand failed at node %d\n", d);
            return 1;
        }
        float node_abs = 0.0f;
        int   worst_i  = 0;
        for (int i = 0; i < pool; ++i) {
            const float ref = golden[(size_t) d * pool + i];
            const float ad = fabsf(logits[i] - ref);
            if (ad > node_abs) { node_abs = ad; worst_i = i; }
            max_abs = fmaxf(max_abs, ad);
            max_rel = fmaxf(max_rel, ad / fmaxf(1.0f, fabsf(ref)));
        }
        // argmax must agree (what the tree builder consumes)
        int am = 0, gm = 0;
        for (int i = 1; i < pool; ++i) {
            if (logits[i] > logits[am]) am = i;
            if (golden[(size_t) d * pool + i] > golden[(size_t) d * pool + gm]) gm = i;
        }
        printf("node %d: max|d|=%.3e (idx %d), argmax %d vs golden %d %s\n",
               d, node_abs, worst_i, am, gm, am == gm ? "OK" : "MISMATCH");
        ok = ok && (am == gm);
        anc.push_back(d);
    }

    printf("max abs diff = %.3e, max rel diff = %.3e, tol = %.1e\n", max_abs, max_rel, tol);
    ok = ok && (max_abs < tol);
    printf(ok ? "PARITY PASS\n" : "PARITY FAIL\n");

    weaver_free(ws);
    return ok ? 0 : 2;
}
