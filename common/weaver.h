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

struct ggml_tensor;
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

// loads the weaver GGUF onto a GPU backend when available (CPU otherwise).
// max_depth > depth_cap extends the scoring horizon past the trained pos_emb
// rows (deeper depths reuse the last row — untrained extrapolation, used for
// chained multi-block drafting); pool_cap > 0 shrinks the candidate-pool
// state to that many entries per depth (the topk actually served).
weaver_scorer * weaver_init(const char * gguf_path, bool prefer_gpu, int max_nodes,
                            int max_depth = 0, int pool_cap = 0);
void            weaver_free(weaver_scorer * ws);

const weaver_params & weaver_get_params(const weaver_scorer * ws);
int                   weaver_max_nodes(const weaver_scorer * ws);
int                   weaver_max_depth(const weaver_scorer * ws);

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
// token; ancestor_slots = node-pool slots of the root->parent path (ascending depth,
// all < self_slot); this node's K/V are stored at self_slot. Writes logits_out[n_cand
// of that depth] (prior + residual). Returns false on invalid args.
bool weaver_expand(weaver_scorer * ws,
                   const float * embed_row, int depth,
                   const int32_t * ancestor_slots, int n_ancestors, int self_slot,
                   float * logits_out);

// --- serving path: candidate lm_head rows and token-embedding rows are gathered
// from the target model's (possibly quantized) tensors — on-device via get_rows
// when the tensor lives on the scorer's backend, host-side dequant otherwise.

// Borrow the target model's tok_embd/output tensors (llama_model_*_tensor()).
bool weaver_attach_target(weaver_scorer * ws,
                          struct ggml_tensor * tok_embd, struct ggml_tensor * output);

// Candidate pools for all depths in one call. ids/scores: [n_depths * n_cand],
// depth-major; scores = drafter logits (any per-depth constant shift cancels in
// the pool softmax, so top-K log-probs work as-is).
bool weaver_set_candidates_ids(weaver_scorer * ws,
                               const int32_t * ids, const float * scores,
                               int n_depths, int n_cand);

// weaver_expand with the embed row gathered from the attached tok_embd by token id.
bool weaver_expand_token(weaver_scorer * ws,
                         int32_t token, int depth,
                         const int32_t * ancestor_slots, int n_ancestors, int self_slot,
                         float * logits_out);

// Batched expansion: one graph for n_nodes nodes stored at the CONSECUTIVE slots
// [slot_base, slot_base + n_nodes). tokens/depths: [n_nodes] (depths pre-clamped to
// the pool horizon). Node r's ancestors are anc_slots[anc_offs[r] .. anc_offs[r+1])
// (anc_offs has n_nodes+1 entries); an ancestor may be any persisted slot OR an
// earlier member of THIS batch (slot < slot_base + r), so a whole speculative chain
// can be expanded in one graph. Writes logits_out[n_nodes * n_cand] node-major,
// n_cand = the (uniform) pool size per depth.
bool weaver_expand_batch(weaver_scorer * ws,
                         const int32_t * tokens, const int32_t * depths, int n_nodes,
                         const int32_t * anc_slots, const int32_t * anc_offs,
                         int slot_base, float * logits_out);
