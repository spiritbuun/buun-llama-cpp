#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx,
    const llama_memory_vbr_params & vbr) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr, vbr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        // QSA index keys come from a separate learned projection and are consumed by ordinary
        // selection matmuls. They are fixed F16 state, never attention Turbo/VBR storage.
        return new llama_kv_cache(
            model, hparams_idx, GGML_TYPE_F16, GGML_TYPE_F16, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const bool removed_idx = mem_idx->seq_rm(seq_id, p0, p1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

bool llama_memory_hybrid_idx::seq_rm_attn(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // The indexer is auxiliary attention state. Every attention-only edit must retain the same
    // cell membership in both children, while deliberately leaving recurrent state untouched.
    if (mem_idx) {
        const bool removed_idx = mem_idx->seq_rm(seq_id, p0, p1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
    }
    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

bool llama_memory_hybrid_idx::seq_rm_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (mem_idx) {
        const bool removed_idx = mem_idx->seq_rm_transient(seq_id, p0, p1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
    }
    return get_mem_attn()->seq_rm_transient(seq_id, p0, p1);
}

bool llama_memory_hybrid_idx::seq_rm_attn_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (mem_idx) {
        const bool removed_idx = mem_idx->seq_rm_attn_transient(seq_id, p0, p1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
    }
    return get_mem_attn()->seq_rm_attn_transient(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    (void) try_seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

bool llama_memory_hybrid_idx::try_seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    if (!llama_memory_hybrid::try_seq_cp(seq_id_src, seq_id_dst, p0, p1)) {
        // The base has already invalidated attention and recurrent state.
        const bool removed_idx = !mem_idx || mem_idx->seq_rm(seq_id_dst, -1, -1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
        return false;
    }

    if (!mem_idx || mem_idx->try_seq_cp(seq_id_src, seq_id_dst, p0, p1)) {
        return true;
    }

    // A failed index copy after the base succeeded invalidates the destination rather than
    // exposing children from different timelines.
    state_drop(seq_id_dst);
    return false;
}

bool llama_memory_hybrid_idx::try_seq_cp_transient(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    if (!llama_memory_hybrid::try_seq_cp_transient(seq_id_src, seq_id_dst, p0, p1)) {
        const bool removed_idx = !mem_idx || mem_idx->seq_rm_transient(seq_id_dst, -1, -1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
        return false;
    }

    if (!mem_idx || mem_idx->try_seq_cp_transient(seq_id_src, seq_id_dst, p0, p1)) {
        return true;
    }

    const bool removed_recr = get_mem_recr()->seq_rm(seq_id_dst, -1, -1);
    const bool removed_attn = get_mem_attn()->seq_rm_transient(seq_id_dst, -1, -1);
    const bool removed_idx = !mem_idx || mem_idx->seq_rm_transient(seq_id_dst, -1, -1);
    GGML_ASSERT(removed_recr && removed_attn && removed_idx);
    GGML_UNUSED(removed_recr);
    GGML_UNUSED(removed_attn);
    GGML_UNUSED(removed_idx);
    return false;
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    // Prove every affected active position is [t,t,t,0] before any child
    // mutates.  The indexer mirrors the attention cells, but checking both
    // also turns an already-drifted cache into a fail-closed error.
    GGML_ASSERT(get_mem_attn()->can_shift_qwen4_text_range(seq_id, p0, p1));
    GGML_ASSERT(!mem_idx || mem_idx->can_shift_qwen4_text_range(seq_id, p0, p1));

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // QSA caches the unnormalised, pre-RoPE index projection.  Only its
        // block-position metadata follows the edit; rotating these raw keys
        // would make the next QSA selection disagree with an unshifted decode.
        mem_idx->seq_add_raw_mrope(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(d);

    // A division gives the active M-RoPE sections different, position-dependent
    // deltas and cannot be represented by the single K-shift input.  Refuse
    // before mutating any of the three children.
    GGML_ABORT("Qwen4 indexed memory does not support seq_div / Self-Extend");
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}


//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

bool llama_memory_hybrid_idx_context::qsa_selection_safe(const llama_ubatch * ubatch) const {
    GGML_ASSERT(ubatch != nullptr);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    if (get_n_stream() != 1) {
        return true;
    }

    // One unified cache tensor has one cell->block map. It is safe when this ubatch addresses one
    // logical sequence: set_input_qsa filters the unified cache to that sequence, so other dormant
    // slots do not matter. Mixed logical queries need separate layouts and use dense attention.
    const llama_seq_id seq = ubatch->seq_id[0][0];
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        if (ubatch->n_seq_id[i] != 1 || ubatch->seq_id[i][0] != seq) {
            return false;
        }
    }
    return true;
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem->get_mem_idx() != nullptr);
    GGML_ASSERT(qsa_selection_safe(ubatch));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    GGML_ASSERT(r <= 64);
    GGML_ASSERT(r*n_blocks >= n_kv);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // one pass per stream: cell j is a different token in each, so no mapping is shared
    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        const auto pos_at = [&](int64_t sec, int64_t b) -> int32_t & {
            return dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b];
        };

        for (int64_t sec = 0; sec < 4; ++sec) {
            std::fill(dst_blk_pos + sec*(n_blocks*n_ns) + s*n_blocks,
                      dst_blk_pos + sec*(n_blocks*n_ns) + (s + 1)*n_blocks, 0);
        }

        // First try the overwhelmingly common one-sequence text layout in one linear pass. Keep
        // natural position-block ids here: unlike one shared "dead" block, that preserves the
        // distinction between an old incomplete block and the current causal tail.
        std::fill(cur_cell_blk,  cur_cell_blk  + n_kv,       -1);
        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, -1);

        bool direct = true;
        bool oor = false;
        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j) || !cells.seq_has((uint32_t) j, seq_of_stream)) {
                continue;
            }

            const llama_pos p = cells.pos_get(j);
            const int64_t pb = p/r;
            if (p < 0 || pb >= n_blocks) {
                oor = true;
                direct = false;
                break;
            }

            int32_t & slot = cur_blk_cells[pb*r + p%r];
            if (slot >= 0) {
                // Repeated temporal positions are M-RoPE and need y/x ranking.
                direct = false;
                break;
            }
            slot = (int32_t) j;
            cur_cell_blk[j] = (int32_t) pb;
        }

        int64_t n_active = 0;
        int32_t n_bid = 0;
        bool ranked = false;

        if (direct) {
            // Empty gather slots are masked before attention. Pointing them at cell zero keeps the
            // gather in range; an incomplete block's pooled score is used only for the forced tail.
            for (int64_t pb = 0; pb < n_blocks; ++pb) {
                int32_t rep = -1;
                bool full = true;
                for (int64_t slot = 0; slot < r; ++slot) {
                    rep = rep < 0 ? cur_blk_cells[pb*r + slot] : rep;
                    full &= cur_blk_cells[pb*r + slot] >= 0;
                }
                if (!full && !blk_bias) {
                    for (int64_t slot = 0; slot < r; ++slot) {
                        const int32_t cell = cur_blk_cells[pb*r + slot];
                        if (cell >= 0) {
                            cur_cell_blk[cell] = -1;
                        }
                    }
                }
                for (int64_t slot = 0; slot < r; ++slot) {
                    if (cur_blk_cells[pb*r + slot] < 0) {
                        cur_blk_cells[pb*r + slot] = 0;
                    }
                }
                pos_at(0, pb) = (int32_t) (pb*r);
                pos_at(1, pb) = rep;
                pos_at(2, pb) = full ? 1 : 0;
            }
            n_bid = (int32_t) n_blocks;
        } else {
            // The fallback is cold (M-RoPE or repeated positions). Each stream owns a separate
            // layout, so group all cells visible to that stream together. This lets a shared
            // prefix and a private suffix complete the same causal block after a sequence fork.
            std::fill(cur_cell_blk, cur_cell_blk + n_kv, -2);
            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j) && cells.seq_has((uint32_t) j, seq_of_stream)) {
                    cur_blk_cells[n_active++] = (int32_t) j;
                }
            }
            std::sort(cur_blk_cells, cur_blk_cells + n_active, [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);
                if (pa != pb) {
                    return pa < pb;
                }
                const auto & ea = cells.ext_get(a);
                const auto & eb = cells.ext_get(b);
                if (ea.y != eb.y) {
                    return ea.y < eb.y;
                }
                if (ea.x != eb.x) {
                    return ea.x < eb.x;
                }
                return a < b;
            });
            for (int64_t k = 1; k < n_active; ++k) {
                ranked |= cells.pos_get(cur_blk_cells[k - 1]) == cells.pos_get(cur_blk_cells[k]);
            }

            int32_t slots[64];
            if (ranked) {
                for (int64_t begin = 0; begin < n_active; begin += r) {
                    const int64_t count = std::min<int64_t>(r, n_active - begin);
                    for (int64_t slot = 0; slot < count; ++slot) {
                        cur_cell_blk[cur_blk_cells[begin + slot]] = -1;
                    }
                    if (count != r) {
                        continue;
                    }

                    GGML_ASSERT(n_bid < n_blocks);
                    const int32_t bid = n_bid++;
                    const int32_t first = cur_blk_cells[begin];
                    const auto & ext = cells.ext_get(first);
                    pos_at(0, bid) = cells.pos_get(first);
                    pos_at(1, bid) = ext.y;
                    pos_at(2, bid) = ext.x;
                    pos_at(3, bid) = cells.pos_get(first);
                    for (int64_t slot = 0; slot < r; ++slot) {
                        cur_cell_blk[cur_blk_cells[begin + slot]] = bid*r + slot;
                    }
                }
            } else {
                int64_t begin = 0;
                while (begin < n_active) {
                    const int64_t pb = cells.pos_get(cur_blk_cells[begin])/r;
                    std::fill(slots, slots + r, -1);
                    uint64_t used_slots = 0;
                    int64_t end = begin;
                    while (end < n_active && cells.pos_get(cur_blk_cells[end])/r == pb) {
                        const int32_t cell = cur_blk_cells[end++];
                        const int64_t slot = cells.pos_get(cell)%r;
                        slots[slot] = cell;
                        used_slots |= uint64_t(1) << slot;
                        cur_cell_blk[cell] = -1;
                    }

                    if (pb < 0 || pb >= n_blocks) {
                        oor = true;
                    } else if (used_slots == slots_full) {
                        GGML_ASSERT(n_bid < n_blocks);
                        const int32_t bid = n_bid++;
                        const int32_t p = (int32_t) (pb*r);
                        pos_at(0, bid) = p;
                        pos_at(1, bid) = p;
                        pos_at(2, bid) = p;
                        pos_at(3, bid) = p;
                        for (int64_t slot = 0; slot < r; ++slot) {
                            cur_cell_blk[slots[slot]] = bid*r + slot;
                        }
                    }
                    begin = end;
                }
            }

        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");
        GGML_ASSERT(n_bid <= n_blocks);

        // Only fallback layouts use a shared in-range block for unpooled cells. Multi-sequence
        // graphs carry per-cell bias; a one-sequence ranked layout has at most one incomplete tail.
        const bool have_dead = !direct && n_bid < n_blocks;
        const int32_t dead_bid = have_dead ? n_bid : n_blocks - 1;

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];
            bool query_ranked = false;
            if (ranked && ubatch->is_pos_2d()) {
                llama_pos last_pos = -1;
                bool have_last = false;
                for (int64_t k = 0; k < n_active; ++k) {
                    const int32_t cell = cur_blk_cells[k];
                    if (!cells.seq_has((uint32_t) cell, seq_id)) {
                        continue;
                    }
                    const llama_pos p = cells.pos_get(cell);
                    query_ranked |= have_last && p == last_pos;
                    last_pos = p;
                    have_last = true;
                }
            }

            if (query_ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                q = -1;
                int64_t rank = 0;
                for (int64_t k = 0; k < n_active; ++k) {
                    const int32_t cell = cur_blk_cells[k];
                    if (!cells.seq_has((uint32_t) cell, seq_id)) {
                        continue;
                    }
                    const llama_pos pc = cells.pos_get(cell);
                    const bool visible = pc < qt || (pc == qt && !cells.ext_get(cell).is_2d_gt(qx, qy));
                    if (visible) {
                        q = rank;
                    }
                    ++rank;
                }
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    const int32_t rep = direct ? pos_at(1, b) : 0;
                    if (b >= n_bid || rep < 0 ||
                        (direct && !cells.seq_has((uint32_t) rep, seq_id))) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    const int64_t block_start = query_ranked ? b*r : pos_at(0, b);
                    cur_blk_bias[b] = block_start >= tail_start ? 1e9f :
                        (direct && pos_at(2, b) == 0 ? -INFINITY : 0.0f);
                }

                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            if (query_ranked) {
                std::fill(cur_bias, cur_bias + n_kv, -INFINITY);
                int64_t rank = 0;
                for (int64_t k = 0; k < n_active; ++k) {
                    const int32_t cell = cur_blk_cells[k];
                    if (!cells.seq_has((uint32_t) cell, seq_id)) {
                        continue;
                    }
                    if (rank <= q) {
                        cur_bias[cell] = rank >= tail_start ? 1e9f :
                            (cur_cell_blk[cell] < 0 ? -INFINITY : 0.0f);
                    }
                    ++rank;
                }
            } else {
                for (int64_t j = 0; j < n_kv; ++j) {
                    float v = -INFINITY;

                    if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                        v = cells.pos_get(j) >= tail_start ? 1e9f :
                            (cur_cell_blk[j] < 0 ? -INFINITY : 0.0f);
                    }

                    cur_bias[j] = v;
                }
            }
        }

        if (!direct) {
            // The temporary non-negative cell value is its exact gather slot. Build the gather
            // only after every bias has consumed the sorted active-cell list, then expose block ids.
            std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);
            for (int64_t j = 0; j < n_kv; ++j) {
                if (cur_cell_blk[j] >= 0) {
                    const int32_t gather_slot = cur_cell_blk[j];
                    cur_blk_cells[gather_slot] = (int32_t) j;
                    cur_cell_blk[j] = gather_slot/r;
                }
            }
        } else {
            // The direct path kept a representative in section 1 for bias filtering.
            for (int64_t b = 0; b < n_blocks; ++b) {
                const int32_t p = pos_at(0, b);
                pos_at(1, b) = p;
                pos_at(2, b) = p;
                pos_at(3, b) = p;
            }
        }

        // The gather requires an in-range block index. Bias construction has already consumed
        // the negative sentinel that distinguishes an incomplete group.
        for (int64_t j = 0; j < n_kv; ++j) {
            if (cur_cell_blk[j] < 0) {
                cur_cell_blk[j] = dead_bid;
            }
        }
    }
}
