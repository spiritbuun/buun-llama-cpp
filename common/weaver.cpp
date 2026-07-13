#include "weaver.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define WVR_LOG(...) fprintf(stderr, __VA_ARGS__)

struct weaver_weights {
    ggml_tensor * embed_norm_w = nullptr, * embed_norm_b = nullptr;
    ggml_tensor * output_norm_w = nullptr, * output_norm_b = nullptr;
    ggml_tensor * token_in_w = nullptr, * token_in_b = nullptr;
    ggml_tensor * proposal_in_w = nullptr, * proposal_in_b = nullptr;
    ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
    ggml_tensor * q_w = nullptr, * k_w = nullptr, * v_w = nullptr, * o_w = nullptr;
    ggml_tensor * ffn_norm_w = nullptr, * ffn_norm_b = nullptr;
    ggml_tensor * up_w = nullptr, * up_b = nullptr, * down_w = nullptr, * down_b = nullptr;
    ggml_tensor * out_norm_w = nullptr, * out_norm_b = nullptr;
    ggml_tensor * lmq_w = nullptr;
    ggml_tensor * pos_emb = nullptr;
};

struct weaver_scorer {
    weaver_params params;

    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w   = nullptr; // weight tensor descriptors
    ggml_backend_buffer_t buf_w = nullptr;

    weaver_weights w;

    // persistent state (device): prefix KV, node KV pool, candidate pools
    ggml_context * ctx_s = nullptr;
    ggml_backend_buffer_t buf_s = nullptr;
    ggml_tensor * ext_k = nullptr;      // [hd, n_head, prefix_max]
    ggml_tensor * ext_v = nullptr;
    ggml_tensor * node_k = nullptr;     // [hd*n_head, max_nodes] (2D for get_rows)
    ggml_tensor * node_v = nullptr;
    ggml_tensor * cand_rows = nullptr;  // [d_model, pool, depth_cap] raw lm_head rows
    std::vector<std::vector<float>> cand_scores; // host prior logits per depth
    std::vector<int> cand_n;            // valid candidates per depth

    int max_nodes  = 0;
    int prefix_len = 0; // tokens in the current prefix (n_steps + 1)

    ~weaver_scorer() {
        if (buf_s) ggml_backend_buffer_free(buf_s);
        if (ctx_s) ggml_free(ctx_s);
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (ctx_w) ggml_free(ctx_w);
        if (backend) ggml_backend_free(backend);
    }
};

static ggml_tensor * wvr_require(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        WVR_LOG("weaver: missing tensor %s\n", name);
    }
    return t;
}

static bool wvr_kv_i32(gguf_context * g, const char * key, int & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    out = (int) gguf_get_val_u32(g, id);
    return true;
}

weaver_scorer * weaver_init(const char * gguf_path, bool prefer_gpu, int max_nodes) {
    auto * ws = new weaver_scorer();
    ws->max_nodes = max_nodes > 0 ? max_nodes : 257;

    // --- read gguf metadata + tensor descriptors ---
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_meta };
    gguf_context * g = gguf_init_from_file(gguf_path, gp);
    if (!g) {
        WVR_LOG("weaver: failed to open %s\n", gguf_path);
        delete ws;
        return nullptr;
    }

    auto & p = ws->params;
    bool ok = true;
    ok &= wvr_kv_i32(g, "weaver.d_model",   p.d_model);
    ok &= wvr_kv_i32(g, "weaver.d_rank",    p.d_rank);
    ok &= wvr_kv_i32(g, "weaver.n_head",    p.n_head);
    ok &= wvr_kv_i32(g, "weaver.mlp_dim",   p.mlp_dim);
    ok &= wvr_kv_i32(g, "weaver.depth_cap", p.depth_cap);
    ok &= wvr_kv_i32(g, "weaver.pool_size", p.pool_size);
    {
        const int64_t id = gguf_find_key(g, "weaver.rms_eps");
        if (id >= 0) p.rms_eps = gguf_get_val_f32(g, id);
    }
    if (!ok) {
        WVR_LOG("weaver: %s is missing weaver-scorer metadata\n", gguf_path);
        gguf_free(g); ggml_free(ctx_meta); delete ws;
        return nullptr;
    }

    // --- backend ---
    if (prefer_gpu) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (dev)  ws->backend = ggml_backend_dev_init(dev, nullptr);
    }
    if (!ws->backend) {
        ws->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    // --- weight tensors: copy descriptors into our ctx, alloc on backend, load data ---
    {
        const int n_tensors = (int) gguf_get_n_tensors(g);
        ggml_init_params ip = { (size_t) (n_tensors + 2) * ggml_tensor_overhead(), nullptr, true };
        ws->ctx_w = ggml_init(ip);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
            ggml_tensor * c = ggml_dup_tensor(ws->ctx_w, t);
            ggml_set_name(c, t->name);
        }
        ws->buf_w = ggml_backend_alloc_ctx_tensors(ws->ctx_w, ws->backend);
        if (!ws->buf_w) {
            WVR_LOG("weaver: weight alloc failed\n");
            gguf_free(g); ggml_free(ctx_meta); delete ws;
            return nullptr;
        }

        std::ifstream fin(gguf_path, std::ios::binary);
        const size_t data_off = gguf_get_data_offset(g);
        std::vector<char> stage;
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * dst = ggml_get_tensor(ws->ctx_w, name);
            const size_t off = gguf_get_tensor_offset(g, i);
            const size_t nb  = ggml_nbytes(dst);
            stage.resize(nb);
            fin.seekg((std::streamoff) (data_off + off));
            fin.read(stage.data(), (std::streamsize) nb);
            ggml_backend_tensor_set(dst, stage.data(), 0, nb);
        }
    }
    gguf_free(g);
    ggml_free(ctx_meta);

    auto & w = ws->w;
    w.embed_norm_w  = wvr_require(ws->ctx_w, "weaver.embed_norm.weight");
    w.embed_norm_b  = wvr_require(ws->ctx_w, "weaver.embed_norm.bias");
    w.output_norm_w = wvr_require(ws->ctx_w, "weaver.output_norm.weight");
    w.output_norm_b = wvr_require(ws->ctx_w, "weaver.output_norm.bias");
    w.token_in_w    = wvr_require(ws->ctx_w, "weaver.token_in.weight");
    w.token_in_b    = wvr_require(ws->ctx_w, "weaver.token_in.bias");
    w.proposal_in_w = wvr_require(ws->ctx_w, "weaver.proposal_in.weight");
    w.proposal_in_b = wvr_require(ws->ctx_w, "weaver.proposal_in.bias");
    w.attn_norm_w   = wvr_require(ws->ctx_w, "weaver.blk.0.attn_norm.weight");
    w.attn_norm_b   = wvr_require(ws->ctx_w, "weaver.blk.0.attn_norm.bias");
    w.q_w           = wvr_require(ws->ctx_w, "weaver.blk.0.attn_q.weight");
    w.k_w           = wvr_require(ws->ctx_w, "weaver.blk.0.attn_k.weight");
    w.v_w           = wvr_require(ws->ctx_w, "weaver.blk.0.attn_v.weight");
    w.o_w           = wvr_require(ws->ctx_w, "weaver.blk.0.attn_output.weight");
    w.ffn_norm_w    = wvr_require(ws->ctx_w, "weaver.blk.0.ffn_norm.weight");
    w.ffn_norm_b    = wvr_require(ws->ctx_w, "weaver.blk.0.ffn_norm.bias");
    w.up_w          = wvr_require(ws->ctx_w, "weaver.blk.0.ffn_up.weight");
    w.up_b          = wvr_require(ws->ctx_w, "weaver.blk.0.ffn_up.bias");
    w.down_w        = wvr_require(ws->ctx_w, "weaver.blk.0.ffn_down.weight");
    w.down_b        = wvr_require(ws->ctx_w, "weaver.blk.0.ffn_down.bias");
    w.out_norm_w    = wvr_require(ws->ctx_w, "weaver.out_norm.weight");
    w.out_norm_b    = wvr_require(ws->ctx_w, "weaver.out_norm.bias");
    w.lmq_w         = wvr_require(ws->ctx_w, "weaver.lm_head_query.weight");
    w.pos_emb       = wvr_require(ws->ctx_w, "weaver.pos_emb");
    for (ggml_tensor * t : { w.embed_norm_w, w.token_in_w, w.q_w, w.lmq_w, w.pos_emb }) {
        if (!t) { delete ws; return nullptr; }
    }

    // --- persistent state ---
    {
        const int hd = p.d_rank / p.n_head;
        const int prefix_max = p.depth_cap + 1;
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        ws->ctx_s = ggml_init(ip);
        ws->ext_k     = ggml_new_tensor_3d(ws->ctx_s, GGML_TYPE_F32, hd, p.n_head, prefix_max);
        ws->ext_v     = ggml_new_tensor_3d(ws->ctx_s, GGML_TYPE_F32, hd, p.n_head, prefix_max);
        ws->node_k    = ggml_new_tensor_2d(ws->ctx_s, GGML_TYPE_F32, p.d_rank, ws->max_nodes);
        ws->node_v    = ggml_new_tensor_2d(ws->ctx_s, GGML_TYPE_F32, p.d_rank, ws->max_nodes);
        ws->cand_rows = ggml_new_tensor_3d(ws->ctx_s, GGML_TYPE_F32, p.d_model, p.pool_size, p.depth_cap);
        ggml_set_name(ws->ext_k, "wvr_ext_k");
        ggml_set_name(ws->ext_v, "wvr_ext_v");
        ggml_set_name(ws->node_k, "wvr_node_k");
        ggml_set_name(ws->node_v, "wvr_node_v");
        ggml_set_name(ws->cand_rows, "wvr_cand_rows");
        ws->buf_s = ggml_backend_alloc_ctx_tensors(ws->ctx_s, ws->backend);
        if (!ws->buf_s) {
            WVR_LOG("weaver: state alloc failed\n");
            delete ws;
            return nullptr;
        }
        ws->cand_scores.resize(p.depth_cap);
        ws->cand_n.assign(p.depth_cap, 0);
    }

    WVR_LOG("weaver: loaded %s (d_model=%d d_rank=%d heads=%d depth_cap=%d pool=%d) on %s\n",
            gguf_path, p.d_model, p.d_rank, p.n_head, p.depth_cap, p.pool_size,
            ggml_backend_name(ws->backend));
    return ws;
}

void weaver_free(weaver_scorer * ws) {
    delete ws;
}

const weaver_params & weaver_get_params(const weaver_scorer * ws) {
    return ws->params;
}

// rms(x)*w + b, fp32 (WeaverRMSNorm has a bias)
static ggml_tensor * wvr_rmsnorm(ggml_context * ctx, ggml_tensor * x,
                                 ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    return ggml_add(ctx, y, b);
}

static ggml_tensor * wvr_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}

// shared block tail: x += o_proj(attn_out); x += fc2(gelu_erf(fc1(norm(x))))
static ggml_tensor * wvr_block_tail(ggml_context * ctx, const weaver_scorer * ws,
                                    ggml_tensor * x, ggml_tensor * attn_out) {
    const auto & w = ws->w;
    x = ggml_add(ctx, x, wvr_linear(ctx, w.o_w, attn_out, nullptr));
    ggml_tensor * h = wvr_rmsnorm(ctx, x, w.ffn_norm_w, w.ffn_norm_b, ws->params.rms_eps);
    h = wvr_linear(ctx, w.up_w, h, w.up_b);
    h = ggml_gelu_erf(ctx, h);
    h = wvr_linear(ctx, w.down_w, h, w.down_b);
    return ggml_add(ctx, x, h);
}

void weaver_begin_step(weaver_scorer * ws,
                       const float * target_final_hidden,
                       const float * drafter_hiddens, int n_steps) {
    const auto & p = ws->params;
    const auto & w = ws->w;
    const int hd = p.d_rank / p.n_head;
    const int T  = n_steps + 1;
    ws->prefix_len = T;
    for (auto & n : ws->cand_n) n = 0;

    ggml_init_params ip = { ggml_tensor_overhead() * 128 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * tfh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.d_model);
    ggml_tensor * dh  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.d_model, n_steps);
    ggml_set_input(tfh);
    ggml_set_input(dh);

    // prompt tokens: [proposal_in(output_norm(tfh)) ; proposal_in(output_norm(dh)) + pos_emb[:n]]
    ggml_tensor * t0 = wvr_linear(ctx, w.proposal_in_w,
            wvr_rmsnorm(ctx, tfh, w.output_norm_w, w.output_norm_b, p.rms_eps), w.proposal_in_b);
    ggml_tensor * tp = wvr_linear(ctx, w.proposal_in_w,
            wvr_rmsnorm(ctx, dh, w.output_norm_w, w.output_norm_b, p.rms_eps), w.proposal_in_b);
    ggml_tensor * pe = ggml_view_2d(ctx, w.pos_emb, p.d_rank, n_steps, w.pos_emb->nb[1], 0);
    tp = ggml_add(ctx, tp, pe);
    ggml_tensor * x = ggml_concat(ctx, ggml_reshape_2d(ctx, t0, p.d_rank, 1), tp, 1); // [d_rank, T]

    // single block, causal self-attention over T tokens
    ggml_tensor * h = wvr_rmsnorm(ctx, x, w.attn_norm_w, w.attn_norm_b, p.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, wvr_linear(ctx, w.q_w, h, nullptr), hd, p.n_head, T);
    ggml_tensor * k = ggml_reshape_3d(ctx, wvr_linear(ctx, w.k_w, h, nullptr), hd, p.n_head, T);
    ggml_tensor * v = ggml_reshape_3d(ctx, wvr_linear(ctx, w.v_w, h, nullptr), hd, p.n_head, T);

    // scores[t, s, head] = q[:,head,s]·k[:,head,t] / sqrt(hd)
    ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [hd, T, n_head]
    ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, T, n_head]
    ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)); // [hd, T, n_head]
    ggml_tensor * scores = ggml_mul_mat(ctx, kh, qh);                    // [T(k), T(q), n_head]
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float) hd));
    // causal mask: build host-side [T, T] with -inf above diagonal
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_input(mask);
    scores = ggml_add(ctx, scores, mask);
    ggml_tensor * probs = ggml_soft_max(ctx, scores);                    // over dim0 = keys
    // y[hd, q, head] = sum_t probs[t, q, head] * v[hd, t, head]
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [T, hd, n_head]
    ggml_tensor * y  = ggml_mul_mat(ctx, vt, probs);                      // [hd, T(q), n_head]
    y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));                 // [hd, n_head, T]
    y = ggml_reshape_2d(ctx, y, p.d_rank, T);

    x = wvr_block_tail(ctx, ws, x, y);
    GGML_UNUSED(x); // prompt pass output is unused; only this block's K/V persist

    // persist K/V into state
    ggml_tensor * ext_k_dst = ggml_view_3d(ctx, ws->ext_k, hd, p.n_head, T,
            ws->ext_k->nb[1], ws->ext_k->nb[2], 0);
    ggml_tensor * ext_v_dst = ggml_view_3d(ctx, ws->ext_v, hd, p.n_head, T,
            ws->ext_v->nb[1], ws->ext_v->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, ext_k_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, ext_v_dst));

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, ws->backend);
    GGML_ASSERT(buf && "weaver: prompt alloc failed");
    ggml_backend_tensor_set(tfh, target_final_hidden, 0, (size_t) p.d_model * sizeof(float));
    ggml_backend_tensor_set(dh, drafter_hiddens, 0, (size_t) p.d_model * n_steps * sizeof(float));
    {
        std::vector<float> m((size_t) T * T, 0.0f);
        for (int qq = 0; qq < T; ++qq) {
            for (int kk = qq + 1; kk < T; ++kk) {
                m[(size_t) qq * T + kk] = -INFINITY; // mask[k, q] layout: dim0 = keys
            }
        }
        // note: tensor dim0 = keys ⇒ element (k, q) at [q*T + k]; we masked k > q above
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    }
    ggml_backend_graph_compute(ws->backend, gf);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

void weaver_set_candidates(weaver_scorer * ws, int depth,
                           const float * lm_rows, const float * scores, int n_cand) {
    const auto & p = ws->params;
    GGML_ASSERT(depth >= 0 && depth < p.depth_cap);
    GGML_ASSERT(n_cand > 0 && n_cand <= p.pool_size);
    const size_t row = (size_t) p.d_model * sizeof(float);
    ggml_backend_tensor_set(ws->cand_rows, lm_rows,
            (size_t) depth * p.pool_size * row, (size_t) n_cand * row);
    ws->cand_scores[depth].assign(scores, scores + n_cand);
    ws->cand_n[depth] = n_cand;
}

bool weaver_expand(weaver_scorer * ws,
                   const float * embed_row, int depth,
                   const int32_t * ancestor_slots, int n_ancestors, int self_slot,
                   float * logits_out) {
    const auto & p = ws->params;
    const auto & w = ws->w;
    const int hd = p.d_rank / p.n_head;

    if (depth < 0 || depth >= p.depth_cap || self_slot < 0 || self_slot >= ws->max_nodes) {
        return false;
    }
    if (ws->cand_n[depth] <= 0 || ws->prefix_len <= 0) {
        return false;
    }
    const int n_cand = ws->cand_n[depth];
    const int P = ws->prefix_len;
    const int pos_row = depth < p.depth_cap ? depth : p.depth_cap - 1;

    ggml_init_params ip = { ggml_tensor_overhead() * 160 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.d_model);
    ggml_set_input(emb);

    // x = token_in(embed_norm(emb)) + pos_emb[depth]
    ggml_tensor * x = wvr_linear(ctx, w.token_in_w,
            wvr_rmsnorm(ctx, emb, w.embed_norm_w, w.embed_norm_b, p.rms_eps), w.token_in_b);
    ggml_tensor * pe = ggml_view_1d(ctx, w.pos_emb, p.d_rank, (size_t) pos_row * w.pos_emb->nb[1]);
    x = ggml_add(ctx, x, pe);

    ggml_tensor * h = wvr_rmsnorm(ctx, x, w.attn_norm_w, w.attn_norm_b, p.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, wvr_linear(ctx, w.q_w, h, nullptr), hd, p.n_head, 1);
    ggml_tensor * k_self = wvr_linear(ctx, w.k_w, h, nullptr); // [d_rank]
    ggml_tensor * v_self = wvr_linear(ctx, w.v_w, h, nullptr);

    // key set: prefix (P) + ancestors (n_anc) + self, as [hd, n_head, T]
    ggml_tensor * ext_k = ggml_view_3d(ctx, ws->ext_k, hd, p.n_head, P,
            ws->ext_k->nb[1], ws->ext_k->nb[2], 0);
    ggml_tensor * ext_v = ggml_view_3d(ctx, ws->ext_v, hd, p.n_head, P,
            ws->ext_v->nb[1], ws->ext_v->nb[2], 0);
    ggml_tensor * k_all;
    ggml_tensor * v_all;
    if (n_ancestors > 0) {
        ggml_tensor * anc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ancestors);
        ggml_set_input(anc);
        ggml_set_name(anc, "wvr_anc");
        ggml_tensor * anc_k = ggml_reshape_3d(ctx,
                ggml_get_rows(ctx, ws->node_k, anc), hd, p.n_head, n_ancestors);
        ggml_tensor * anc_v = ggml_reshape_3d(ctx,
                ggml_get_rows(ctx, ws->node_v, anc), hd, p.n_head, n_ancestors);
        k_all = ggml_concat(ctx, ext_k, anc_k, 2);
        v_all = ggml_concat(ctx, ext_v, anc_v, 2);
    } else {
        k_all = ext_k;
        v_all = ext_v;
    }
    k_all = ggml_concat(ctx, k_all, ggml_reshape_3d(ctx, k_self, hd, p.n_head, 1), 2);
    v_all = ggml_concat(ctx, v_all, ggml_reshape_3d(ctx, v_self, hd, p.n_head, 1), 2);
    const int T = P + n_ancestors + 1;

    // per-head attention: scores [T, 1, n_head]
    ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k_all, 0, 2, 1, 3)); // [hd, T, n_head]
    ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, v_all, 0, 2, 1, 3));
    ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));     // [hd, 1, n_head]
    ggml_tensor * scores = ggml_mul_mat(ctx, kh, qh);                        // [T, 1, n_head]
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float) hd));
    ggml_tensor * probs = ggml_soft_max(ctx, scores);
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3));    // [T, hd, n_head]
    ggml_tensor * y  = ggml_mul_mat(ctx, vt, probs);                         // [hd, 1, n_head]
    y = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3)), p.d_rank, 1);

    x = wvr_block_tail(ctx, ws, ggml_reshape_2d(ctx, x, p.d_rank, 1), y);

    // query = out_norm(x); residual = cand_rows[depth] @ (lmq_w @ query)
    ggml_tensor * query = wvr_rmsnorm(ctx, x, w.out_norm_w, w.out_norm_b, p.rms_eps);
    ggml_tensor * q5    = ggml_mul_mat(ctx, w.lmq_w, query);                 // [d_model, 1]
    ggml_tensor * rows  = ggml_view_2d(ctx, ws->cand_rows, p.d_model, n_cand,
            ws->cand_rows->nb[1], (size_t) depth * ws->cand_rows->nb[2]);
    ggml_tensor * resid = ggml_mul_mat(ctx, rows, q5);                       // [n_cand, 1]
    ggml_set_output(resid);
    ggml_build_forward_expand(gf, resid);

    // persist this node's K/V
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k_self,
            ggml_view_1d(ctx, ws->node_k, p.d_rank, (size_t) self_slot * ws->node_k->nb[1])));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v_self,
            ggml_view_1d(ctx, ws->node_v, p.d_rank, (size_t) self_slot * ws->node_v->nb[1])));

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, ws->backend);
    GGML_ASSERT(buf && "weaver: expand alloc failed");
    ggml_backend_tensor_set(emb, embed_row, 0, (size_t) p.d_model * sizeof(float));
    if (n_ancestors > 0) {
        ggml_tensor * anc = ggml_get_tensor(ctx, "wvr_anc");
        ggml_backend_tensor_set(anc, ancestor_slots, 0, (size_t) n_ancestors * sizeof(int32_t));
    }
    ggml_backend_graph_compute(ws->backend, gf);

    std::vector<float> resid_host(n_cand);
    ggml_backend_tensor_get(resid, resid_host.data(), 0, (size_t) n_cand * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    const auto & prior = ws->cand_scores[depth];
    for (int i = 0; i < n_cand; ++i) {
        logits_out[i] = prior[i] + resid_host[i];
    }
    return true;
}
