#pragma once

// Weaver (DFlash-TfM) tree-draft scorer — arXiv 2607.06763, EXP-40.
//
// A single-block transformer (d_rank 2048) that rescores DFlash's per-depth top-K
// candidate pools with CONDITIONAL logits: logits = raw target-lm_head logits of the
// drafter hidden (prior) + W_lm[cands] · (lm_head_query · out_norm(x)) (residual).
// Weights come from a `weaver-scorer` GGUF (convert_weaver_to_gguf.py); token
// embeddings and lm_head rows are the TARGET model's, supplied by the caller.
//
// Call pattern per drafting step:
//   weaver_begin_step(ws, target_final_hidden, drafter_hiddens, n_steps)
//   weaver_set_candidates(ws, depth, lm_rows, scores, n_cand)   // per depth used
//   weaver_expand(ws, ...)                                      // per tree node
//
// All inputs are host f32 in this version (device-tensor paths come with the
// server integration).

#include <cstdint>

struct weaver_scorer;

struct weaver_params {
    int   d_model   = 0; // verifier hidden size (5120)
    int   d_rank    = 0; // weaver width (2048)
    int   n_head    = 0; // 16
    int   mlp_dim   = 0; // 2048
    int   depth_cap = 0; // 15 (pos_emb rows; max tree depth)
    int   pool_size = 0; // 512 candidates per depth
    float rms_eps   = 1e-6f;
};

// loads the weaver GGUF onto a GPU backend when available (CPU otherwise)
weaver_scorer * weaver_init(const char * gguf_path, bool prefer_gpu, int max_nodes);
void            weaver_free(weaver_scorer * ws);

const weaver_params & weaver_get_params(const weaver_scorer * ws);

// Build the 16-token prefix KV for this drafting step.
// target_final_hidden: [d_model] — verifier final-norm hidden at the committed token.
// drafter_hiddens:     [n_steps * d_model] — DFlash block hiddens for depths 0..n_steps-1.
void weaver_begin_step(weaver_scorer * ws,
                       const float * target_final_hidden,
                       const float * drafter_hiddens, int n_steps);

// Candidate pool for one depth: lm_rows = raw target lm_head rows [n_cand * d_model],
// scores = raw target-lm_head logits of the drafter hidden at this depth [n_cand].
void weaver_set_candidates(weaver_scorer * ws, int depth,
                           const float * lm_rows, const float * scores, int n_cand);

// Expand one node: embed_row = raw target token-embedding row [d_model] of the node's
// token; ancestor_slots = node-pool slots of the root->parent path (ascending depth);
// this node's K/V are stored at self_slot. Writes logits_out[n_cand of that depth]
// (prior + residual). Returns false on invalid args.
bool weaver_expand(weaver_scorer * ws,
                   const float * embed_row, int depth,
                   const int32_t * ancestor_slots, int n_ancestors, int self_slot,
                   float * logits_out);
