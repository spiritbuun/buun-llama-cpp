#include "weaver.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cinttypes>
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
    ggml_gallocr_t galloc  = nullptr; // reused compute buffer for all step graphs
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

    // serving path: borrowed target tensors (weaver_attach_target)
    ggml_tensor * tgt_tok_embd = nullptr;
    ggml_tensor * tgt_output   = nullptr;
    bool tok_embd_host = false; // host-side dequant gather vs device get_rows
    bool output_host   = false;

    int max_nodes  = 0;
    int prefix_len = 0; // tokens in the current prefix (n_steps + 1)

    ~weaver_scorer() {
        if (buf_s) ggml_backend_buffer_free(buf_s);
        if (ctx_s) ggml_free(ctx_s);
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (ctx_w) ggml_free(ctx_w);
        if (galloc) ggml_gallocr_free(galloc);
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
    ws->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ws->backend));

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
        // node slots may be read (masked to -inf) before ever being written when a
        // round's expansion is skipped — they must hold finite values, not NaN garbage
        ggml_backend_buffer_clear(ws->buf_s, 0);
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

    // the prompt pass output is unused — only this block's prefix K/V persist,
    // and those are plain projections of the normed input (no attention needed)
    ggml_tensor * h = wvr_rmsnorm(ctx, x, w.attn_norm_w, w.attn_norm_b, p.rms_eps);
    ggml_tensor * k = ggml_reshape_3d(ctx, wvr_linear(ctx, w.k_w, h, nullptr), hd, p.n_head, T);
    ggml_tensor * v = ggml_reshape_3d(ctx, wvr_linear(ctx, w.v_w, h, nullptr), hd, p.n_head, T);

    // persist K/V into state
    ggml_tensor * ext_k_dst = ggml_view_3d(ctx, ws->ext_k, hd, p.n_head, T,
            ws->ext_k->nb[1], ws->ext_k->nb[2], 0);
    ggml_tensor * ext_v_dst = ggml_view_3d(ctx, ws->ext_v, hd, p.n_head, T,
            ws->ext_v->nb[1], ws->ext_v->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, ext_k_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, ext_v_dst));

    const bool alloc_ok = ggml_gallocr_alloc_graph(ws->galloc, gf);
    GGML_ASSERT(alloc_ok && "weaver: prompt alloc failed");
    ggml_backend_tensor_set(tfh, target_final_hidden, 0, (size_t) p.d_model * sizeof(float));
    ggml_backend_tensor_set(dh, drafter_hiddens, 0, (size_t) p.d_model * n_steps * sizeof(float));
    ggml_backend_graph_compute(ws->backend, gf);
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

// batched expansion: one graph for R frontier nodes at consecutive slots
// [slot_base, slot_base + R). Node r's key set is the prefix (P) + every persisted
// node slot [0, slot_base) with non-ancestors masked to -inf + its own K/V (the mask
// keeps the math identical to a per-node ancestor gather while all R nodes share one
// mul_mat). embed_rows != nullptr → host f32 rows [R * d_model]; else tokens[R] are
// gathered on-device from the attached tok_embd.
static bool wvr_expand_batch_impl(weaver_scorer * ws,
                                  const float * embed_rows, const int32_t * tokens,
                                  const int32_t * depths, int R,
                                  const int32_t * anc_slots, const int32_t * anc_offs,
                                  int slot_base, float * logits_out) {
    const auto & p = ws->params;
    const auto & w = ws->w;
    const int hd = p.d_rank / p.n_head;

    if (R <= 0 || slot_base < 0 || slot_base + R > ws->max_nodes || ws->prefix_len <= 0) {
        return false;
    }
    const int n_cand = ws->cand_n[depths[0]];
    if (n_cand <= 0) {
        return false;
    }
    const int P = ws->prefix_len;
    const int S = slot_base; // persisted node slots visible to this round
    const int T = P + S + R;
    for (int r = 0; r < R; ++r) {
        if (depths[r] < 0 || depths[r] >= p.depth_cap || ws->cand_n[depths[r]] != n_cand) {
            return false;
        }
        // ancestors may be persisted slots OR earlier members of this batch (a chain
        // can be expanded in one graph: K/V depend only on each node's own input)
        for (int a = anc_offs[r]; a < anc_offs[r + 1]; ++a) {
            if (anc_slots[a] < 0 || anc_slots[a] >= slot_base + r) {
                return false;
            }
        }
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 192 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * emb;
    ggml_tensor * emb_in  = nullptr;
    ggml_tensor * emb_ids = nullptr;
    if (embed_rows) {
        emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.d_model, R);
        ggml_set_input(emb_in);
        emb = emb_in;
    } else {
        emb_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, R);
        ggml_set_input(emb_ids);
        emb = ggml_get_rows(ctx, ws->tgt_tok_embd, emb_ids); // [d_model, R] f32
    }

    // x = token_in(embed_norm(emb)) + pos_emb[depth_r]
    ggml_tensor * depth_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, R);
    ggml_set_input(depth_ids);
    ggml_tensor * x = wvr_linear(ctx, w.token_in_w,
            wvr_rmsnorm(ctx, emb, w.embed_norm_w, w.embed_norm_b, p.rms_eps), w.token_in_b);
    x = ggml_add(ctx, x, ggml_get_rows(ctx, w.pos_emb, depth_ids));

    ggml_tensor * h = wvr_rmsnorm(ctx, x, w.attn_norm_w, w.attn_norm_b, p.rms_eps);
    ggml_tensor * q      = ggml_reshape_3d(ctx, wvr_linear(ctx, w.q_w, h, nullptr), hd, p.n_head, R);
    ggml_tensor * k_self = wvr_linear(ctx, w.k_w, h, nullptr); // [d_rank, R]
    ggml_tensor * v_self = wvr_linear(ctx, w.v_w, h, nullptr);

    // key set: prefix (P) + node pool [0, S) + this round's R selves, as [hd, n_head, T]
    ggml_tensor * k_all = ggml_view_3d(ctx, ws->ext_k, hd, p.n_head, P,
            ws->ext_k->nb[1], ws->ext_k->nb[2], 0);
    ggml_tensor * v_all = ggml_view_3d(ctx, ws->ext_v, hd, p.n_head, P,
            ws->ext_v->nb[1], ws->ext_v->nb[2], 0);
    if (S > 0) {
        ggml_tensor * nk = ggml_view_3d(ctx, ws->node_k, hd, p.n_head, S,
                hd * sizeof(float), ws->node_k->nb[1], 0);
        ggml_tensor * nv = ggml_view_3d(ctx, ws->node_v, hd, p.n_head, S,
                hd * sizeof(float), ws->node_v->nb[1], 0);
        k_all = ggml_concat(ctx, k_all, nk, 2);
        v_all = ggml_concat(ctx, v_all, nv, 2);
    }
    k_all = ggml_concat(ctx, k_all, ggml_reshape_3d(ctx, k_self, hd, p.n_head, R), 2);
    v_all = ggml_concat(ctx, v_all, ggml_reshape_3d(ctx, v_self, hd, p.n_head, R), 2);

    // per-head attention: scores [T, R, n_head] + visibility mask [T, R] (bcast heads)
    ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k_all, 0, 2, 1, 3)); // [hd, T, n_head]
    ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, v_all, 0, 2, 1, 3));
    ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));     // [hd, R, n_head]
    ggml_tensor * scores = ggml_mul_mat(ctx, kh, qh);                        // [T, R, n_head]
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float) hd));
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, R);
    ggml_set_input(mask);
    scores = ggml_add(ctx, scores, mask);
    ggml_tensor * probs = ggml_soft_max(ctx, scores);
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3));    // [T, hd, n_head]
    ggml_tensor * y  = ggml_mul_mat(ctx, vt, probs);                         // [hd, R, n_head]
    y = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3)), p.d_rank, R);

    x = wvr_block_tail(ctx, ws, x, y);

    // query = out_norm(x); resid[r] = cand_rows[depth_r] @ (lmq_w @ query_r)
    ggml_tensor * query = wvr_rmsnorm(ctx, x, w.out_norm_w, w.out_norm_b, p.rms_eps);
    ggml_tensor * q5    = ggml_mul_mat(ctx, w.lmq_w, query);                 // [d_model, R]
    ggml_tensor * cand_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n_cand * R);
    ggml_set_input(cand_ids);
    ggml_tensor * flat = ggml_reshape_2d(ctx, ws->cand_rows,
            p.d_model, (int64_t) p.pool_size * p.depth_cap);
    ggml_tensor * rows = ggml_reshape_3d(ctx, ggml_get_rows(ctx, flat, cand_ids),
            p.d_model, n_cand, R);
    ggml_tensor * resid = ggml_mul_mat(ctx, rows,
            ggml_reshape_3d(ctx, q5, p.d_model, 1, R));                      // [n_cand, 1, R]
    ggml_set_output(resid);
    ggml_build_forward_expand(gf, resid);

    // persist this round's K/V at slots [slot_base, slot_base + R)
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k_self, ggml_view_2d(ctx, ws->node_k,
            p.d_rank, R, ws->node_k->nb[1], (size_t) slot_base * ws->node_k->nb[1])));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v_self, ggml_view_2d(ctx, ws->node_v,
            p.d_rank, R, ws->node_v->nb[1], (size_t) slot_base * ws->node_v->nb[1])));

    if (!ggml_gallocr_alloc_graph(ws->galloc, gf)) {
        WVR_LOG("weaver: expand graph alloc failed\n");
        ggml_free(ctx);
        return false;
    }

    if (emb_in) {
        ggml_backend_tensor_set(emb_in, embed_rows, 0, (size_t) p.d_model * R * sizeof(float));
    } else {
        ggml_backend_tensor_set(emb_ids, tokens, 0, (size_t) R * sizeof(int32_t));
    }
    ggml_backend_tensor_set(depth_ids, depths, 0, (size_t) R * sizeof(int32_t));
    {
        std::vector<float> m((size_t) R * T, -INFINITY);
        std::vector<int32_t> cids((size_t) R * n_cand);
        for (int r = 0; r < R; ++r) {
            float * mr = m.data() + (size_t) r * T; // dim0 = keys ⇒ element (k, r) at [r*T + k]
            for (int k = 0; k < P; ++k) {
                mr[k] = 0.0f;
            }
            for (int a = anc_offs[r]; a < anc_offs[r + 1]; ++a) {
                mr[P + anc_slots[a]] = 0.0f;
            }
            mr[P + S + r] = 0.0f;
            for (int i = 0; i < n_cand; ++i) {
                cids[(size_t) r * n_cand + i] = depths[r] * p.pool_size + i;
            }
        }
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
        ggml_backend_tensor_set(cand_ids, cids.data(), 0, cids.size() * sizeof(int32_t));
    }
    ggml_backend_graph_compute(ws->backend, gf);

    std::vector<float> resid_host((size_t) n_cand * R);
    ggml_backend_tensor_get(resid, resid_host.data(), 0, resid_host.size() * sizeof(float));
    ggml_free(ctx);

    for (int r = 0; r < R; ++r) {
        const auto & prior = ws->cand_scores[depths[r]];
        for (int i = 0; i < n_cand; ++i) {
            logits_out[(size_t) r * n_cand + i] = prior[i] + resid_host[(size_t) r * n_cand + i];
        }
    }
    return true;
}

bool weaver_expand(weaver_scorer * ws,
                   const float * embed_row, int depth,
                   const int32_t * ancestor_slots, int n_ancestors, int self_slot,
                   float * logits_out) {
    const int32_t d = depth;
    const int32_t offs[2] = { 0, n_ancestors };
    return wvr_expand_batch_impl(ws, embed_row, nullptr, &d, 1,
                                 ancestor_slots, offs, self_slot, logits_out);
}

// dequantize one row of a host-resident (possibly quantized) 2D tensor to f32
static bool wvr_dequant_row(const ggml_tensor * t, int64_t row, float * dst, int64_t n) {
    const char * src = (const char *) t->data + row * t->nb[1];
    switch (t->type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            return true;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
            return true;
        default: {
            const auto * traits = ggml_get_type_traits(t->type);
            if (!traits || !traits->to_float) {
                return false;
            }
            traits->to_float(src, dst, n);
            return true;
        }
    }
}

// host gather usable / device gather usable for a borrowed target tensor
static bool wvr_gather_mode(const weaver_scorer * ws, const ggml_tensor * t, bool & host) {
    if (!t || !t->buffer) {
        return false;
    }
    host = ggml_backend_buffer_is_host(t->buffer);
    if (host) {
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
            const auto * traits = ggml_get_type_traits(t->type);
            if (!traits || !traits->to_float) {
                WVR_LOG("weaver: no host dequant for %s (%s)\n", t->name, ggml_type_name(t->type));
                return false;
            }
        }
        return true;
    }
    if (!ggml_backend_supports_buft(ws->backend, ggml_backend_buffer_get_type(t->buffer))) {
        WVR_LOG("weaver: %s buffer (%s) not accessible from %s\n",
                t->name, ggml_backend_buffer_name(t->buffer), ggml_backend_name(ws->backend));
        return false;
    }
    return true;
}

bool weaver_attach_target(weaver_scorer * ws, ggml_tensor * tok_embd, ggml_tensor * output) {
    if (!tok_embd || !output) {
        WVR_LOG("weaver: target tok_embd/output tensor missing\n");
        return false;
    }
    if (tok_embd->ne[0] != ws->params.d_model || output->ne[0] != ws->params.d_model) {
        WVR_LOG("weaver: target width %" PRId64 "/%" PRId64 " != d_model %d\n",
                tok_embd->ne[0], output->ne[0], ws->params.d_model);
        return false;
    }
    if (!wvr_gather_mode(ws, tok_embd, ws->tok_embd_host) ||
        !wvr_gather_mode(ws, output,   ws->output_host)) {
        return false;
    }
    ws->tgt_tok_embd = tok_embd;
    ws->tgt_output   = output;
    WVR_LOG("weaver: target attached — tok_embd %s (%s gather), output %s (%s gather)\n",
            ggml_type_name(tok_embd->type), ws->tok_embd_host ? "host" : "device",
            ggml_type_name(output->type),   ws->output_host   ? "host" : "device");
    return true;
}

bool weaver_set_candidates_ids(weaver_scorer * ws,
                               const int32_t * ids, const float * scores,
                               int n_depths, int n_cand) {
    const auto & p = ws->params;
    if (!ws->tgt_output || n_depths <= 0 || n_depths > p.depth_cap ||
        n_cand <= 0 || n_cand > p.pool_size) {
        return false;
    }
    for (int d = 0; d < n_depths; ++d) {
        ws->cand_scores[d].assign(scores + (size_t) d * n_cand, scores + (size_t) (d + 1) * n_cand);
        ws->cand_n[d] = n_cand;
    }
    for (int d = n_depths; d < p.depth_cap; ++d) {
        ws->cand_n[d] = 0;
    }

    // padding ids (< 0) gather row 0 — their scores are masked at expansion
    const int N = n_depths * n_cand;
    std::vector<int32_t> ids_cl(ids, ids + N);
    for (auto & id : ids_cl) {
        if (id < 0) id = 0;
    }

    const size_t row = (size_t) p.d_model * sizeof(float);
    if (ws->output_host) {
        std::vector<float> stage((size_t) n_cand * p.d_model);
        for (int d = 0; d < n_depths; ++d) {
            for (int i = 0; i < n_cand; ++i) {
                if (!wvr_dequant_row(ws->tgt_output, ids_cl[(size_t) d * n_cand + i],
                                     stage.data() + (size_t) i * p.d_model, p.d_model)) {
                    return false;
                }
            }
            ggml_backend_tensor_set(ws->cand_rows, stage.data(),
                    (size_t) d * p.pool_size * row, (size_t) n_cand * row);
        }
        return true;
    }

    // device: one get_rows over all depths, copied into the strided pool views
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_input(ids_t);
    ggml_tensor * rows = ggml_get_rows(ctx, ws->tgt_output, ids_t);        // [d_model, N] f32
    rows = ggml_reshape_3d(ctx, rows, p.d_model, n_cand, n_depths);
    ggml_tensor * dst = ggml_view_3d(ctx, ws->cand_rows, p.d_model, n_cand, n_depths,
            ws->cand_rows->nb[1], ws->cand_rows->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, rows, dst));

    if (!ggml_gallocr_alloc_graph(ws->galloc, gf)) {
        WVR_LOG("weaver: candidate gather alloc failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(ids_t, ids_cl.data(), 0, (size_t) N * sizeof(int32_t));
    ggml_backend_graph_compute(ws->backend, gf);
    ggml_free(ctx);
    return true;
}

bool weaver_expand_token(weaver_scorer * ws,
                         int32_t token, int depth,
                         const int32_t * ancestor_slots, int n_ancestors, int self_slot,
                         float * logits_out) {
    const int32_t offs[2] = { 0, n_ancestors };
    return weaver_expand_batch(ws, &token, &depth, 1, ancestor_slots, offs, self_slot, logits_out);
}

bool weaver_expand_batch(weaver_scorer * ws,
                         const int32_t * tokens, const int32_t * depths, int n_nodes,
                         const int32_t * anc_slots, const int32_t * anc_offs,
                         int slot_base, float * logits_out) {
    if (!ws->tgt_tok_embd || n_nodes <= 0) {
        return false;
    }
    for (int r = 0; r < n_nodes; ++r) {
        if (tokens[r] < 0 || tokens[r] >= ws->tgt_tok_embd->ne[1]) {
            return false;
        }
    }
    if (ws->tok_embd_host) {
        std::vector<float> rows((size_t) n_nodes * ws->params.d_model);
        for (int r = 0; r < n_nodes; ++r) {
            if (!wvr_dequant_row(ws->tgt_tok_embd, tokens[r],
                                 rows.data() + (size_t) r * ws->params.d_model, ws->params.d_model)) {
                return false;
            }
        }
        return wvr_expand_batch_impl(ws, rows.data(), nullptr, depths, n_nodes,
                                     anc_slots, anc_offs, slot_base, logits_out);
    }
    return wvr_expand_batch_impl(ws, nullptr, tokens, depths, n_nodes,
                                 anc_slots, anc_offs, slot_base, logits_out);
}
