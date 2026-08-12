#include "llama-kv-cache-dsv4.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

static constexpr uint32_t DSV4_CSA_RATIO = 4;
static constexpr uint32_t DSV4_HCA_RATIO = 128;

static ggml_type dsv4_vbr_tier_type(uint8_t tier) {
    switch (tier) {
        case VBR_TIER_T8:     return GGML_TYPE_TURBO8_0;
        case VBR_TIER_T4:     return GGML_TYPE_TURBO4_0;
        case VBR_TIER_T3_TCQ: return GGML_TYPE_TURBO3_TCQ;
        case VBR_TIER_T2_TCQ: return GGML_TYPE_TURBO2_TCQ;
        case VBR_TIER_T1_TCQ: return GGML_TYPE_TURBO1_TCQ;
        default:              GGML_ABORT("invalid DSV4 VBR tier %d", (int) tier);
    }
}

static double dsv4_vbr_floor_bpv(double requested) {
    if (const char * env = getenv("VBR_MIN_BITS")) {
        requested = atof(env);
    }
    return requested > 0.0 ? requested :
        8.0 * ggml_type_size(GGML_TYPE_TURBO1_TCQ) /
            ggml_blck_size(GGML_TYPE_TURBO1_TCQ);
}

static constexpr uint32_t DSV4_STATE_MAGIC         = 0x34565344; // DSV4
static constexpr uint32_t DSV4_STATE_VERSION       = 1;
static constexpr uint32_t DSV4_STATE_MODE_FULL     = 0;
static constexpr uint32_t DSV4_STATE_MODE_PARTIAL  = 1;
static constexpr uint32_t DSV4_K_CACHE_STATE_VER   = 2;
static constexpr uint32_t DSV4_COMP_STATE_VER      = 1;

static uint32_t dsv4_comp_size(uint32_t kv_size, uint32_t ratio) {
    return std::max<uint32_t>(1, (kv_size + ratio - 1)/ratio);
}

static void dsv4_clear_tensor_stream(ggml_tensor * tensor, uint32_t stream) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->ne[3] == 1);
    GGML_ASSERT(stream < (uint32_t) tensor->ne[2]);

    const size_t stream_size = tensor->nb[2];
    ggml_backend_tensor_memset(tensor, 0, stream*stream_size, stream_size);
}

static uint32_t dsv4_state_n_used_k_rows(llama_pos pos_max, uint32_t ratio, uint32_t kv_size) {
    if (pos_max < 0) {
        return 0;
    }

    const uint64_t n_rows = ((uint64_t) pos_max + 1)/ratio;

    return (uint32_t) std::min<uint64_t>(kv_size, n_rows);
}

static int64_t dsv4_stream_offset(uint32_t n_stream, llama_seq_id seq_id, uint32_t size) {
    if (n_stream <= 1) {
        return 0;
    }
    if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
        throw std::runtime_error("DSV4 sequence id out of stream range");
    }

    return (int64_t) seq_id*size;
}

static bool dsv4_ubatch_has_coupled(const llama_ubatch & ubatch) {
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static bool dsv4_token_has_seq(const llama_ubatch & ubatch, uint32_t i, llama_seq_id seq_id) {
    for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
        if (ubatch.seq_id[i][s] == seq_id) {
            return true;
        }
    }

    return false;
}

static llama_ubatch dsv4_build_raw_write_ubatch(const llama_ubatch & ubatch) {
    if (!dsv4_ubatch_has_coupled(ubatch)) {
        return ubatch;
    }
    if (ubatch.embd) {
        throw std::runtime_error("DSV4 coupled embedding ubatches are not supported");
    }

    std::vector<uint32_t> counts(ubatch.n_seqs_unq, 0);
    uint32_t n_tokens = 0;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (dsv4_token_has_seq(ubatch, i, seq_id)) {
                ++counts[s];
                ++n_tokens;
            }
        }
    }

    if (n_tokens == 0) {
        return ubatch;
    }

    const uint32_t n_seq_tokens = counts[0];
    for (uint32_t s = 1; s < counts.size(); ++s) {
        if (counts[s] != n_seq_tokens) {
            throw std::runtime_error("DSV4 coupled raw writes require equal sequence lengths");
        }
    }

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->pos.resize((size_t) n_tokens*ubatch.n_pos);
    data->n_seq_id.reserve(n_tokens);
    data->seq_id.reserve(n_tokens);
    data->seq_id_data.reserve(n_tokens);
    data->seq_id_unq.assign(ubatch.seq_id_unq, ubatch.seq_id_unq + ubatch.n_seqs_unq);
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->output.assign(n_tokens, 0);
    if (ubatch.token) {
        data->token.reserve(n_tokens);
    }

    for (uint32_t s = 0; s < data->seq_id_unq.size(); ++s) {
        data->seq_idx[data->seq_id_unq[s]] = s;
    }

    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (!dsv4_token_has_seq(ubatch, i, seq_id)) {
                continue;
            }

            const uint32_t dst = data->n_seq_id.size();
            if (ubatch.token) {
                data->token.push_back(ubatch.token[i]);
            }
            for (uint32_t p = 0; p < ubatch.n_pos; ++p) {
                data->pos[(size_t) p*n_tokens + dst] = ubatch.pos[(size_t) p*ubatch.n_tokens + i];
            }
            data->n_seq_id.push_back(1);
            data->seq_id_data.push_back(seq_id);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->seq_id.push_back(&data->seq_id_data[i]);
    }

    llama_ubatch res {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ ubatch.n_seqs_unq,
        /*.n_seqs_unq   =*/ ubatch.n_seqs_unq,
        /*.n_pos        =*/ ubatch.n_pos,
        /*.token        =*/ data->token.empty() ? nullptr : data->token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ data->pos.data(),
        /*.n_seq_id     =*/ data->n_seq_id.data(),
        /*.seq_id       =*/ data->seq_id.data(),
        /*.seq_id_unq   =*/ data->seq_id_unq.data(),
        /*.seq_idx      =*/ data->seq_idx.data(),
        /*.output       =*/ data->output.data(),
        /*.data         =*/ data,
    };

    return res;
}

static std::vector<llama_ubatch> dsv4_build_raw_write_ubatches(const std::vector<llama_ubatch> & ubatches) {
    std::vector<llama_ubatch> res;
    res.reserve(ubatches.size());
    for (const llama_ubatch & ubatch : ubatches) {
        res.push_back(dsv4_build_raw_write_ubatch(ubatch));
    }
    return res;
}

static bool dsv4_batch_has_coupled(const llama_batch & batch) {
    if (!batch.n_seq_id) {
        return false;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static int64_t dsv4_comp_graph_n_stream(const llama_ubatch & ubatch, uint32_t n_stream) {
    // Coupled sequence sets must stay in one graph stream because their
    // compressed state is shared. Independent per-seq state can fan out.
    if (n_stream <= 1 || ubatch.n_seqs_unq <= 1 || dsv4_ubatch_has_coupled(ubatch)) {
        return 1;
    }

    return ubatch.n_seqs_unq;
}

static void dsv4_state_src_stream_range(
        uint32_t       n_stream,
        llama_seq_id   seq_id,
        uint32_t     & s0,
        uint32_t     & ns) {
    if (seq_id >= 0 && n_stream > 1) {
        if ((uint32_t) seq_id >= n_stream) {
            throw std::runtime_error("DSV4 state sequence id out of stream range");
        }

        s0 = (uint32_t) seq_id;
        ns = 1;
        return;
    }

    s0 = 0;
    ns = seq_id >= 0 ? 1 : n_stream;
}

static void dsv4_state_dst_stream_range(
        uint32_t       n_stream,
        llama_seq_id   seq_id,
        uint32_t       ns,
        uint32_t     & s0) {
    if (seq_id >= 0) {
        if (ns != 1) {
            throw std::runtime_error("DSV4 sequence state stream count mismatch");
        }
        if (n_stream > 1 && (uint32_t) seq_id >= n_stream) {
            throw std::runtime_error("DSV4 state sequence id out of stream range");
        }

        s0 = n_stream > 1 ? (uint32_t) seq_id : 0;
        return;
    }

    if (ns != n_stream) {
        throw std::runtime_error("DSV4 full state stream count mismatch");
    }

    s0 = 0;
}

static void dsv4_state_write_tensor_streams(
        llama_io_write_i & io,
        ggml_tensor      * tensor,
        uint32_t           tensor_rows,
        uint32_t           n_rows,
        uint32_t           s0,
        uint32_t           ns,
        const std::vector<uint32_t> * stream_ids = nullptr) {
    const int32_t  type_i   = (int32_t) tensor->type;
    const uint64_t ne0      = tensor->ne[0];
    const uint64_t rows     = n_rows;
    const uint64_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    if (n_rows > tensor_rows) {
        throw std::runtime_error("DSV4 state tensor row count exceeds storage");
    }

    io.write(&type_i,   sizeof(type_i));
    io.write(&ne0,      sizeof(ne0));
    io.write(&rows,     sizeof(rows));
    io.write(&row_size, sizeof(row_size));

    const size_t stream_stride = (size_t) tensor_rows*row_size;
    const size_t size          = (size_t) n_rows*row_size;
    if (size == 0) {
        return;
    }

    if (stream_ids && stream_ids->size() != ns) {
        throw std::runtime_error("DSV4 state tensor stream map size mismatch");
    }

    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t stream = stream_ids ? (*stream_ids)[s] : s0 + s;
        if ((int64_t) stream >= tensor->ne[2]) {
            throw std::runtime_error("DSV4 state tensor stream out of range");
        }
        const size_t offset = (size_t) stream*stream_stride;
        io.write_tensor(tensor, offset, size);
    }
}

static void dsv4_state_read_tensor_streams(
        llama_io_read_i & io,
        ggml_tensor     * tensor,
        uint32_t          tensor_rows,
        uint32_t          n_rows,
        uint32_t          s0,
        uint32_t          ns) {
    int32_t  type_i_ref;
    uint64_t ne0_ref;
    uint64_t rows_ref;
    uint64_t row_size_ref;

    io.read(&type_i_ref,   sizeof(type_i_ref));
    io.read(&ne0_ref,      sizeof(ne0_ref));
    io.read(&rows_ref,     sizeof(rows_ref));
    io.read(&row_size_ref, sizeof(row_size_ref));

    const int32_t  type_i   = (int32_t) tensor->type;
    const uint64_t ne0      = tensor->ne[0];
    const uint64_t rows     = n_rows;
    const uint64_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    if (type_i != type_i_ref || ne0 != ne0_ref || rows != rows_ref || row_size != row_size_ref) {
        throw std::runtime_error("DSV4 state tensor metadata mismatch");
    }
    if (n_rows > tensor_rows) {
        throw std::runtime_error("DSV4 state tensor row count exceeds storage");
    }

    const size_t stream_stride = (size_t) tensor_rows*row_size;
    const size_t size          = (size_t) n_rows*row_size;
    if (size == 0) {
        return;
    }

    for (uint32_t s = 0; s < ns; ++s) {
        const size_t offset = (size_t) (s0 + s)*stream_stride;
        io.read_tensor(tensor, offset, size);
    }
}

static void dsv4_state_write_k_cache(
        llama_io_write_i    & io,
        const llama_kv_cache * kv,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags,
        uint32_t              n_rows) {
    GGML_UNUSED(flags);

    uint32_t s0;
    uint32_t ns;
    dsv4_state_src_stream_range(kv->get_n_stream(), seq_id, s0, ns);

    const uint32_t version = DSV4_K_CACHE_STATE_VER;
    const uint32_t kv_size = kv->get_size();
    const auto layer_ids = kv->get_layer_ids();
    const uint32_t n_layer = layer_ids.size();

    if (n_rows > kv_size) {
        throw std::runtime_error("DSV4 K-cache state row count exceeds cache size");
    }

    io.write(&version, sizeof(version));
    io.write(&n_rows,  sizeof(n_rows));
    io.write(&ns,      sizeof(ns));
    io.write(&n_layer, sizeof(n_layer));

    for (uint32_t il : layer_ids) {
        io.write(&il, sizeof(il));
        dsv4_state_write_tensor_streams(io, kv->get_k_storage(il), kv_size, n_rows, s0, ns);
    }
}

static void dsv4_state_read_k_cache(
        llama_io_read_i  & io,
        llama_kv_cache   * kv,
        llama_seq_id       seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t n_rows_ref;
    uint32_t ns;
    uint32_t n_layer_ref;

    io.read(&version,     sizeof(version));
    io.read(&n_rows_ref,  sizeof(n_rows_ref));
    io.read(&ns,          sizeof(ns));
    io.read(&n_layer_ref, sizeof(n_layer_ref));

    if (version != 1 && version != DSV4_K_CACHE_STATE_VER) {
        throw std::runtime_error("DSV4 K-cache state version mismatch");
    }

    const uint32_t kv_size = kv->get_size();
    if (version == 1 && n_rows_ref != kv_size) {
        LLAMA_LOG_INFO("kv size ref %d kv %d\n", n_rows_ref, kv_size);
        throw std::runtime_error("DSV4 K-cache state size mismatch");
    }
    if (n_rows_ref > kv_size) {
        LLAMA_LOG_INFO("kv rows ref %d kv %d\n", n_rows_ref, kv_size);
        throw std::runtime_error("DSV4 K-cache state size mismatch");
    }

    uint32_t s0;
    dsv4_state_dst_stream_range(kv->get_n_stream(), seq_id, ns, s0);

    const auto layer_ids = kv->get_layer_ids();
    if (n_layer_ref != layer_ids.size()) {
        throw std::runtime_error("DSV4 K-cache layer count mismatch");
    }

    for (uint32_t il : layer_ids) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != il) {
            throw std::runtime_error("DSV4 K-cache layer id mismatch");
        }

        dsv4_state_read_tensor_streams(io, kv->get_k_storage(il), kv_size, n_rows_ref, s0, ns);
    }
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq,
        const std::vector<uint32_t> & rs_idx) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(ubatch, n_stream);

    // n_stream is the persistent cache/state layout; plan.n_stream is the
    // graph view for this ubatch and can be a subset of those streams.
    if (n_stream <= 1 && ubatch.n_seqs_unq > 1) {
        throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
    }

    const int64_t state_rows = (int64_t) state_size*n_stream;

    struct persist_row {
        int32_t dst;
        int32_t src;
        llama_pos pos;
    };

    std::vector<persist_row> persist_rows;

    // For the overlap compressor, build_overlap_compressed_kv_from_state() consumes
    // state_read_idxs as two contiguous halves: the first ratio*n_blocks entries are
    // the "previous-window" gather indices for every block, followed by the
    // "current-window" indices for every block. Collect them separately here and
    // append cur after prev once the loop has visited all completed blocks
    std::vector<int32_t> overlap_prev_reads;
    std::vector<int32_t> overlap_cur_reads;

    std::map<std::pair<llama_seq_id, llama_pos>, int64_t> curr_token_idx_map;
    std::map<llama_seq_id, uint32_t> state_write_counts;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            curr_token_idx_map[std::make_pair(ubatch.seq_id[i][s], ubatch.pos[i])] = i;
        }
    }

    const auto state_source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        if (pos < 0) {
            // The overlap compressor needs a zero/-inf source for the first
            // block's previous half. The graph appends that row after the
            // current-ubatch scratch rows.
            return (int32_t) (state_rows + ubatch.n_tokens);
        }

        const auto key = std::make_pair(seq_id, pos);
        if (curr_token_idx_map.find(key) != curr_token_idx_map.end()) {
            return (int32_t) (state_rows + curr_token_idx_map.at(key));
        }

        const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
        return (int32_t) (stream_off + pos%state_size);
    };

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos[i];

        if (pos < 0) {
            continue;
        }

        plan.state_pos.push_back((int32_t) (pos%ratio));

        const int64_t n_visible = (int64_t) (pos + 1)/ratio;
        plan.n_visible[i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
            const int32_t state_idx = (int32_t) (stream_off + pos%state_size);

            const auto it = std::find_if(persist_rows.begin(), persist_rows.end(),
                    [state_idx](const persist_row & row) {
                        return row.dst == state_idx;
                    });
            if (it == persist_rows.end()) {
                persist_rows.push_back({ state_idx, (int32_t) i, pos });
            } else if (pos > it->pos) {
                it->src = (int32_t) i;
                it->pos = pos;
            }

            if ((pos + 1) % ratio != 0) {
                continue;
            }

            const llama_pos source_start = pos + 1 - ratio;
            const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);

            plan.state_write_idxs.push_back(cache_off + pos/ratio);
            plan.state_write_pos.push_back((int32_t) source_start);
            ++state_write_counts[seq_id];

            if (overlap) {
                const llama_pos prev_start = source_start - ratio;

                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(state_source_idx(seq_id, prev_start + j));
                }
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_cur_reads.push_back(state_source_idx(seq_id, source_start + j));
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(state_source_idx(seq_id, source_start + j));
                }
            }
        }
    }

    if (ratio == DSV4_CSA_RATIO && !plan.state_pos.empty()) {
        assert(kv_size > 0);

        // Pad each stream to the reserve plan's block count.
        const auto append_dummy_block = [&](llama_seq_id seq_id, uint32_t i) {
            const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);
            const int32_t source_idx = state_source_idx(seq_id, ubatch.pos[i]);

            plan.state_write_idxs.push_back(cache_off + kv_size - 1);
            plan.state_write_pos .push_back(0);

            if (overlap) {
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(source_idx);
                    overlap_cur_reads .push_back(source_idx);
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(source_idx);
                }
            }
        };

        if (dsv4_ubatch_has_coupled(ubatch)) {
            if (plan.state_write_idxs.empty()) {
                uint32_t i = 0;
                while (i < ubatch.n_tokens && ubatch.pos[i] < 0) {
                    ++i;
                }
                assert(i < ubatch.n_tokens);
                append_dummy_block(ubatch.seq_id[i][0], i);
            }
        } else {
            const uint32_t n_blocks = (std::max<uint32_t>(1, ubatch.n_seq_tokens) + ratio - 1)/ratio;

            for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                const llama_seq_id seq_id = ubatch.seq_id_unq[s];
                const uint32_t n_writes = state_write_counts[seq_id];
                if (n_writes >= n_blocks) {
                    continue;
                }
                if (n_writes + 1 != n_blocks) {
                    throw std::runtime_error("DSV4 CSA sequence positions are not contiguous");
                }

                uint32_t i = 0;
                while (i < ubatch.n_tokens && (ubatch.pos[i] < 0 || !dsv4_token_has_seq(ubatch, i, seq_id))) {
                    ++i;
                }
                assert(i < ubatch.n_tokens);
                append_dummy_block(seq_id, i);
            }
        }
    }

    if (overlap) {
        // [ all blocks' prev-window indices | all blocks' cur-window indices ]
        plan.state_read_idxs.reserve(overlap_prev_reads.size() + overlap_cur_reads.size());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_prev_reads.begin(), overlap_prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_cur_reads.begin(), overlap_cur_reads.end());
    }

    plan.n_kv = GGML_PAD(plan.n_kv, 256u);

    std::sort(persist_rows.begin(), persist_rows.end(),
            [](const persist_row & a, const persist_row & b) {
                return a.dst < b.dst;
            });

    for (const persist_row & row : persist_rows) {
        plan.state_persist_src_idxs.push_back(row.src);
        plan.state_persist_dst_idxs.push_back(row.dst);
    }


    if (n_rs_seq > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];
            if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
                continue;
            }

            const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
            const uint32_t rollback = (uint32_t) seq_id < rs_idx.size() ? rs_idx[seq_id] : 0;
            // Keep the restore graph fixed-width when no rollback is pending.
            const int64_t src_plane = rollback > 0 && rollback <= n_rs_seq ? (int64_t) rollback*state_rows : 0;
            for (uint32_t r = 0; r < state_size; ++r) {
                plan.state_restore_src_idxs.push_back((int32_t) (src_plane + stream_off + r));
                plan.state_restore_dst_idxs.push_back((int32_t) (stream_off + r));
            }

            std::vector<uint32_t> token_idxs;
            token_idxs.reserve(ubatch.n_tokens);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (dsv4_token_has_seq(ubatch, i, seq_id)) {
                    token_idxs.push_back(i);
                }
            }
            if (token_idxs.empty()) {
                continue;
            }

            const uint32_t n_seq_tokens = (uint32_t) token_idxs.size();
            const int64_t scratch_off = (int64_t) state_rows*(1 + n_rs_seq);
            for (uint32_t d = 1; d <= n_rs_seq; ++d) {
                const int64_t dst_plane = (int64_t) d*state_rows;

                for (uint32_t r = 0; r < state_size; ++r) {
                    int32_t src;
                    if (d <= n_seq_tokens) {
                        const uint32_t prefix = n_seq_tokens - d;
                        src = (int32_t) (stream_off + r);

                        for (uint32_t j = 0; j < prefix; ++j) {
                            const uint32_t i_tok = token_idxs[j];
                            if (ubatch.pos[i_tok] >= 0 && (uint32_t) (ubatch.pos[i_tok]%state_size) == r) {
                                src = (int32_t) (scratch_off + i_tok);
                            }
                        }
                    } else {
                        const int64_t src_plane = (int64_t) (d - n_seq_tokens)*state_rows;
                        src = (int32_t) (src_plane + stream_off + r);
                    }

                    plan.state_snapshot_src_idxs.push_back(src);
                    plan.state_snapshot_dst_idxs.push_back((int32_t) (dst_plane + stream_off + r));
                }
            }
        }
    }

    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    if (debug) {
        LLAMA_LOG_INFO("%s: ratio=%u, n_tokens=%u, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, ratio, ubatch.n_tokens,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }

    return plan;
}

static std::vector<llama_kv_cache_dsv4_context::comp_plan> dsv4_build_comp_plans(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq,
        const std::vector<uint32_t> & rs_idx) {
    std::vector<llama_kv_cache_dsv4_context::comp_plan> plans;
    plans.reserve(ubatches.size());

    for (const llama_ubatch & ubatch : ubatches) {
        plans.push_back(dsv4_build_comp_plan(ubatch, ratio, overlap, state_size, kv_size, n_stream, n_rs_seq, rs_idx));
    }

    return plans;
}

static llama_kv_cache::slot_info_vec_t dsv4_build_comp_sinfos(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t n_stream) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (const llama_ubatch & ubatch : ubatches) {
        if (n_stream <= 1 && ubatch.n_seqs_unq > 1) {
            throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
        }

        const uint32_t ns = (uint32_t) dsv4_comp_graph_n_stream(ubatch, n_stream);
        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = n_stream > 1 ? LLAMA_MAX_SEQ : 0;
        sinfo.s1 = 0;
        sinfo.resize(ns);

        for (uint32_t s = 0; s < ns; ++s) {
            const llama_seq_id seq_id = n_stream > 1 ? ubatch.seq_id_unq[s] : 0;
            const uint32_t strm = (uint32_t) dsv4_stream_offset(n_stream, seq_id, 1);

            sinfo.s0 = std::min(sinfo.s0, strm);
            sinfo.s1 = std::max(sinfo.s1, strm);
            sinfo.strm[s] = strm;
            sinfo.idxs[s].resize(1, 0);
        }

        if (n_stream > 1 && sinfo.s1 - sinfo.s0 + 1 != ns) {
            throw std::runtime_error("DSV4 compressed streams are not contiguous in ubatch");
        }

        sinfos.push_back(std::move(sinfo));
    }

    return sinfos;
}

static llama_kv_cache::slot_info_vec_t dsv4_build_raw_read_sinfos(
        const llama_kv_cache::slot_info_vec_t & sinfos_write,
        const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (size_t i = 0; i < ubatches.size(); ++i) {
        const llama_ubatch & ubatch = ubatches[i];
        const auto & sinfo_write = sinfos_write[i];

        if (!dsv4_ubatch_has_coupled(ubatch)) {
            sinfos.push_back(sinfo_write);
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[0][0];
        uint32_t i_stream = 0;
        for (; i_stream < sinfo_write.n_stream(); ++i_stream) {
            if (sinfo_write.strm[i_stream] == seq_id) {
                break;
            }
        }
        if (i_stream == sinfo_write.n_stream()) {
            throw std::runtime_error("DSV4 raw write stream not found for coupled read");
        }

        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = sinfo_write.strm[i_stream];
        sinfo.s1 = sinfo_write.strm[i_stream];
        sinfo.resize(1);
        sinfo.strm[0] = sinfo_write.strm[i_stream];
        sinfo.idxs[0] = sinfo_write.idxs[i_stream];
        sinfos.push_back(std::move(sinfo));
    }

    return sinfos;
}

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_reserve_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(ubatch, n_stream);
    plan.n_kv = kv_size;

    if (ubatch.n_tokens == 0) {
        return plan;
    }

    const uint32_t n_seqs       = std::max<uint32_t>(1, ubatch.n_seqs);
    const uint32_t n_seq_tokens = std::max<uint32_t>(1, ubatch.n_seq_tokens);
    const uint64_t n_blocks_u64 = (uint64_t) n_seqs*((n_seq_tokens + ratio - 1)/ratio);
    const size_t n_blocks = (size_t) std::max<uint64_t>(1, n_blocks_u64);
    GGML_ASSERT((uint64_t) n_blocks == std::max<uint64_t>(1, n_blocks_u64));

    const uint64_t state_rows = (uint64_t) state_size*n_stream;
    const size_t n_persist = (size_t) std::min<uint64_t>(ubatch.n_tokens, state_rows);
    const size_t n_restore = n_rs_seq > 0 ? (size_t) state_size*std::max<uint32_t>(1, ubatch.n_seqs_unq) : 0;
    const size_t n_snapshot = (size_t) n_rs_seq*state_size*std::max<uint32_t>(1, ubatch.n_seqs_unq);

    plan.state_pos .resize(ubatch.n_tokens);
    plan.state_persist_src_idxs.resize(n_persist);
    plan.state_persist_dst_idxs.resize(n_persist);
    plan.state_restore_src_idxs.resize(n_restore);
    plan.state_restore_dst_idxs.resize(n_restore);
    plan.state_snapshot_src_idxs.resize(n_snapshot);
    plan.state_snapshot_dst_idxs.resize(n_snapshot);
    plan.state_read_idxs .resize((overlap ? 2u : 1u)*ratio*n_blocks);
    plan.state_write_idxs.resize(n_blocks);
    plan.state_write_pos .resize(n_blocks);

    return plan;
}

static void dsv4_make_k_only(llama_hparams & hparams) {
    // llama_kv_cache uses hparams.is_mla() to allocate K-only storage.
    hparams.n_embd_head_k_mla_impl = hparams.n_embd_head_k();
    hparams.n_embd_head_v_mla_impl = hparams.n_embd_head_k();
}

//
// llama_dsv4_comp_state
//

llama_dsv4_comp_state::llama_dsv4_comp_state(
        const llama_model & model,
                bool        offload,
                bool        unified,
            uint32_t        n_seq_max,
            uint32_t        ratio,
            uint32_t        state_size,
            uint32_t        n_embd_state,
            uint32_t        n_rs_seq,
        const char    * name,
        const llama_memory_i::layer_filter_cb & filter) :
    ratio(ratio),
    state_size(state_size),
    n_embd_state(n_embd_state),
    n_stream(unified ? 1 : n_seq_max),
    n_rs_seq(n_rs_seq) {
    const llama_hparams & hparams = model.hparams;

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*hparams.n_layer()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for DSV4 compressor state");
        }

        const uint32_t n_planes = n_stream*(1 + n_rs_seq);
        ggml_tensor * kv    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_state, state_size, n_planes);
        ggml_tensor * score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_state, state_size, n_planes);

        ggml_format_name(kv,    "dsv4_%s_state_kv_l%d",    name, il);
        ggml_format_name(score, "dsv4_%s_state_score_l%d", name, il);

        std::vector<ggml_tensor *> kv_stream;
        std::vector<ggml_tensor *> score_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            kv_stream.push_back(ggml_view_2d(ctx, kv, n_embd_state, state_size, kv->nb[1], s*kv->nb[2]));
            score_stream.push_back(ggml_view_2d(ctx, score, n_embd_state, state_size, score->nb[1], s*score->nb[2]));
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, kv, score, std::move(kv_stream), std::move(score_stream) });
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for DSV4 compressor state");
        }

        ggml_backend_buffer_clear(buf, 0);

        LLAMA_LOG_INFO("%s: %10s DSV4 %s state buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), name, ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    LLAMA_LOG_INFO("%s: %s ratio = %u, state = %u x %u, streams = %u, rs_seq = %u, layers = %zu, size = %7.2f MiB\n",
            __func__, name, ratio, state_size, n_embd_state, n_stream, n_rs_seq, layers.size(), total_size()/1024.0/1024.0);
}

void llama_dsv4_comp_state::clear(llama_seq_id seq_id, bool data) {
    if (!data) {
        return;
    }

    if (seq_id >= 0) {
        GGML_ASSERT((uint32_t) seq_id < n_stream);

        for (const auto & layer : layers) {
            for (uint32_t d = 0; d <= n_rs_seq; ++d) {
                const uint32_t stream = d*n_stream + (uint32_t) seq_id;
                dsv4_clear_tensor_stream(layer.kv,    stream);
                dsv4_clear_tensor_stream(layer.score, stream);
            }
        }
        return;
    }

    for (auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

void llama_dsv4_comp_state::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    GGML_ASSERT(seq_id_src >= 0 && (uint32_t) seq_id_src < n_stream);
    GGML_ASSERT(seq_id_dst >= 0 && (uint32_t) seq_id_dst < n_stream);

    if (seq_id_src == seq_id_dst) {
        return;
    }

    clear(seq_id_dst, true);

    sc_info.ssrc.push_back((uint32_t) seq_id_src);
    sc_info.sdst.push_back((uint32_t) seq_id_dst);
}

void llama_dsv4_comp_state::apply_copies(const stream_copy_info & sc_info) const {
    for (size_t i = 0; i < sc_info.ssrc.size(); ++i) {
        const uint32_t ssrc = sc_info.ssrc[i];
        const uint32_t sdst = sc_info.sdst[i];

        for (const auto & layer : layers) {
            ggml_backend_tensor_copy(layer.kv_stream[ssrc], layer.kv_stream[sdst]);
            ggml_backend_tensor_copy(layer.score_stream[ssrc], layer.score_stream[sdst]);
        }
    }
}

uint32_t llama_dsv4_comp_state::get_ratio() const {
    return ratio;
}

uint32_t llama_dsv4_comp_state::get_state_size() const {
    return state_size;
}

uint32_t llama_dsv4_comp_state::get_n_stream() const {
    return n_stream;
}

uint32_t llama_dsv4_comp_state::get_n_rs_seq() const {
    return n_rs_seq;
}

uint32_t llama_dsv4_comp_state::get_n_rows() const {
    return state_size*n_stream;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_dsv4_comp_state::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        ret[buft] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

void llama_dsv4_comp_state::state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        const std::vector<uint32_t> & rs_idx) const {
    GGML_UNUSED(flags);

    uint32_t s0;
    uint32_t ns;
    dsv4_state_src_stream_range(n_stream, seq_id, s0, ns);

    std::vector<uint32_t> stream_ids(ns);
    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t seq = seq_id >= 0 ? (uint32_t) seq_id : s0 + s;
        if (seq >= rs_idx.size() || rs_idx[seq] > n_rs_seq) {
            throw std::runtime_error("DSV4 recurrent state rollback index out of range");
        }
        stream_ids[s] = rs_idx[seq]*n_stream + s0 + s;
    }

    const uint32_t version      = DSV4_COMP_STATE_VER;
    const uint32_t n_layer      = layers.size();

    io.write(&version,      sizeof(version));
    io.write(&ratio,        sizeof(ratio));
    io.write(&state_size,   sizeof(state_size));
    io.write(&n_embd_state, sizeof(n_embd_state));
    io.write(&ns,           sizeof(ns));
    io.write(&n_layer,      sizeof(n_layer));

    for (const auto & layer : layers) {
        io.write(&layer.il, sizeof(layer.il));

        dsv4_state_write_tensor_streams(io, layer.kv,    state_size, state_size, s0, ns, &stream_ids);
        dsv4_state_write_tensor_streams(io, layer.score, state_size, state_size, s0, ns, &stream_ids);
    }
}

void llama_dsv4_comp_state::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t ratio_ref;
    uint32_t state_size_ref;
    uint32_t n_embd_state_ref;
    uint32_t ns;
    uint32_t n_layer_ref;

    io.read(&version,          sizeof(version));
    io.read(&ratio_ref,        sizeof(ratio_ref));
    io.read(&state_size_ref,   sizeof(state_size_ref));
    io.read(&n_embd_state_ref, sizeof(n_embd_state_ref));
    io.read(&ns,               sizeof(ns));
    io.read(&n_layer_ref,      sizeof(n_layer_ref));

    if (version != DSV4_COMP_STATE_VER) {
        throw std::runtime_error("DSV4 compressor state version mismatch");
    }
    if (ratio_ref != ratio || state_size_ref != state_size || n_embd_state_ref != n_embd_state) {
        throw std::runtime_error("DSV4 compressor state metadata mismatch");
    }
    if (n_layer_ref != layers.size()) {
        throw std::runtime_error("DSV4 compressor state layer count mismatch");
    }

    uint32_t s0;
    dsv4_state_dst_stream_range(n_stream, seq_id, ns, s0);

    for (const auto & layer : layers) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != layer.il) {
            throw std::runtime_error("DSV4 compressor state layer id mismatch");
        }

        dsv4_state_read_tensor_streams(io, layer.kv,    state_size, state_size, s0, ns);
        dsv4_state_read_tensor_streams(io, layer.score, state_size, state_size, s0, ns);
    }
}

ggml_tensor * llama_dsv4_comp_state::get_kv_all(ggml_context * ctx, int32_t il) const {
    const int32_t ids = map_layer_ids.at(il);
    ggml_tensor * state = layers[ids].kv;

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows()*(1 + n_rs_seq), state->nb[1], 0);
}

ggml_tensor * llama_dsv4_comp_state::get_score_all(ggml_context * ctx, int32_t il) const {
    const int32_t ids = map_layer_ids.at(il);
    ggml_tensor * state = layers[ids].score;

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows()*(1 + n_rs_seq), state->nb[1], 0);
}

ggml_tensor * llama_dsv4_comp_state::get_kv(ggml_context * ctx, int32_t il) const {
    ggml_tensor * state = get_kv_all(ctx, il);
    const size_t row_size = ggml_row_size(state->type, state->ne[0]);

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows(), state->nb[1], 0*row_size);
}

ggml_tensor * llama_dsv4_comp_state::get_score(ggml_context * ctx, int32_t il) const {
    ggml_tensor * state = get_score_all(ctx, il);
    const size_t row_size = ggml_row_size(state->type, state->ne[0]);

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows(), state->nb[1], 0*row_size);
}

ggml_tensor * llama_dsv4_comp_state::cpy_kv(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_kv_all(ctx, il), cur, idxs);
}

ggml_tensor * llama_dsv4_comp_state::cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_score_all(ctx, il), cur, idxs);
}

size_t llama_dsv4_comp_state::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

//
// llama_kv_cache_dsv4
//

llama_kv_cache_dsv4::llama_kv_cache_dsv4(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const llama_memory_vbr_params & vbr) :
    hparams_raw(model.hparams),
    hparams_csa(model.hparams),
    hparams_hca(model.hparams),
    hparams_lid(model.hparams),
    n_seq_max(n_seq_max),
    n_rs_seq(n_rs_seq),
    kv_size(kv_size),
    rs_idx(n_seq_max, 0) {

    const layer_filter_cb filter_raw = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return true;
    };

    // Speculative server contexts reserve one backup sequence per user slot;
    // DSV4 performs their bounded rollback through n_rs_seq snapshot planes,
    // so those backups do not require independent KV streams. Distinct user
    // slots do: compressed rows are addressed by (position / ratio), and two
    // unrelated sequences at the same position cannot share that row. Dynamic
    // VBR currently requires one unified VMM plane, so reject the unsupported
    // multi-slot geometry before slot 1 can overwrite or clear slot 0.
    const uint32_t n_user_seq = n_rs_seq > 0 && n_seq_max % 2 == 0
            ? n_seq_max / 2
            : n_seq_max;
    if (vbr.dynamic && !hparams_raw.no_alloc && n_user_seq > 1) {
        throw std::runtime_error(format(
                "DSV4 dynamic VBR currently supports one user sequence (-np 1); "
                "this context requests %u user sequences", n_user_seq));
    }

    // Dynamic VBR's VMM representation is one unified tensor plane. Recurrent
    // speculation does not need a second KV plane: its short rollback history
    // lives in llama_dsv4_comp_state's n_rs_seq snapshot planes. Keep the
    // established per-sequence layout for static DSV4 caches, but honor the
    // public unified contract when arming dynamic VBR.
    const bool unified_raw = unified && vbr.dynamic;

    hparams_raw.n_layer_nextn = 0;
    hparams_csa.n_layer_nextn = 0;
    hparams_hca.n_layer_nextn = 0;
    hparams_lid.n_layer_nextn = 0;

    // Turbo/VBR policy. Reader map: raw/csa/hca are read ONLY through fused
    // attention (deepseek4.cpp concats them into k_all for build_attn_mha; ggml_concat handles
    // same-type quantized tensors block-aligned), and the compressor reads only the untyped
    // comp-state streams — so STATIC turbo tiers are legal on raw/csa/hca. The lightning-indexer
    // cache (lid) is read by mul_mat (or the fused_lid op, which has no turbo decode yet) and its
    // rows are indexer-head-sized — pin it at F16. Dynamic VBR gives raw/csa/hca separate VMM
    // pools but one parent-owned policy boundary: a layer's raw and compressed tensors always
    // transcode together before the concat graph is built.
    llama_memory_vbr_params vbr_raw = vbr;
    vbr_raw.trace_label = "raw";
    const bool turbo_req = ggml_is_turbo_kv_type(type_k) || ggml_is_turbo_kv_type(type_v);
    ggml_type type_comp = type_k; // K==V enforced for this arch at init
    ggml_type type_lid  = type_k;
    if (turbo_req) {
        type_lid = GGML_TYPE_F16;
        vbr_raw = {};
        vbr_raw.compute_backend_for_buft = vbr.compute_backend_for_buft;
        LLAMA_LOG_INFO("%s: DSV4 static turbo KV: raw/csa/hca hold %s (fused-attention read), "
                "lid pinned at f16 (indexer reads via mul_mat/fused_lid)\n",
                __func__, ggml_type_name(type_k));
    } else if (vbr.dynamic) {
        type_lid = GGML_TYPE_F16;
        LLAMA_LOG_INFO("%s: DSV4 dynamic VBR: raw/csa/hca use one ganged per-layer ladder; "
                "lid pinned at f16\n", __func__);
    }

    // children need the backend-binding callback even without dynamic VBR: static turbo
    // K/V types resolve their decode backend through it (refused at init otherwise)
    llama_memory_vbr_params vbr_child = {};
    vbr_child.compute_backend_for_buft = vbr.compute_backend_for_buft;

    llama_memory_vbr_params vbr_csa = vbr.dynamic ? vbr : vbr_child;
    llama_memory_vbr_params vbr_hca = vbr.dynamic ? vbr : vbr_child;
    vbr_csa.trace_label = "csa";
    vbr_hca.trace_label = "hca";
    if (vbr.dynamic || vbr.budget_bytes > 0) {
        uint64_t n_raw_l = 0;
        uint64_t n_csa_l = 0;
        uint64_t n_hca_l = 0;
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            if (filter && !filter(il)) {
                continue;
            }
            ++n_raw_l;
            if (model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO) {
                ++n_csa_l;
            } else if (model.hparams.dsv4_compress_ratios[il] == DSV4_HCA_RATIO) {
                ++n_hca_l;
            }
        }
        const uint64_t raw_cells = swa_full ? kv_size :
            GGML_PAD(std::min(kv_size, model.hparams.n_swa + n_ubatch), 256u);
        const uint64_t csa_cells = GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u);
        const uint64_t hca_cells = GGML_PAD(dsv4_comp_size(kv_size, DSV4_HCA_RATIO), 256u);
        const double w_raw = (double) n_raw_l * raw_cells;
        const double w_csa = (double) n_csa_l * csa_cells;
        const double w_hca = (double) n_hca_l * hca_cells;
        const double w_all = w_raw + w_csa + w_hca;
        if (w_all > 0.0) {
            const double raw_frac = w_raw / w_all;
            const double csa_frac = w_csa / w_all;
            vbr_raw.device_share = vbr.device_share * raw_frac;
            vbr_csa.device_share = vbr.device_share * csa_frac;
            vbr_hca.device_share = vbr.device_share - vbr_raw.device_share - vbr_csa.device_share;
            if (vbr.budget_bytes > 0) {
                vbr_raw.budget_bytes = (uint64_t) ((double) vbr.budget_bytes * raw_frac);
                vbr_csa.budget_bytes = (uint64_t) ((double) vbr.budget_bytes * csa_frac);
                vbr_hca.budget_bytes = vbr.budget_bytes - vbr_raw.budget_bytes - vbr_csa.budget_bytes;
                LLAMA_LOG_INFO("%s: DSV4 VBR budget split: %.2f MiB raw / %.2f MiB CSA / "
                        "%.2f MiB HCA (by entry-tier allocated rows)\n", __func__,
                        vbr_raw.budget_bytes/1048576.0, vbr_csa.budget_bytes/1048576.0,
                        vbr_hca.budget_bytes/1048576.0);
            }
        }
    }

    LLAMA_LOG_INFO("%s: creating DSV4 raw KV cache\n", __func__);

    dsv4_make_k_only(hparams_raw);

    kv_raw = std::make_unique<llama_kv_cache_iswa>(
            model, hparams_raw, type_k, type_v,
            v_trans, offload, swa_full, unified_raw, kv_size, n_seq_max, n_ubatch, n_pad,
            nullptr, filter_raw, reuse, nullptr, vbr_raw);

    dsv4_make_k_only(hparams_csa);
    dsv4_make_k_only(hparams_hca);

    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_k_swa  = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_swa  = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;
    dsv4_make_k_only(hparams_lid);

    const layer_filter_cb filter_csa = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO;
    };

    const layer_filter_cb filter_hca = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_HCA_RATIO;
    };

    const bool unified_compressed = unified && vbr.dynamic;

    LLAMA_LOG_INFO("%s: creating DSV4 CSA compressed KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_CSA_RATIO));

    kv_csa = std::make_unique<llama_kv_cache>(
            model, hparams_csa, type_comp, type_comp,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, nullptr, nullptr, vbr_csa);

    LLAMA_LOG_INFO("%s: creating DSV4 HCA compressed KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_HCA_RATIO));

    kv_hca = std::make_unique<llama_kv_cache>(
            model, hparams_hca, type_comp, type_comp,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, DSV4_HCA_RATIO), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_hca, nullptr, nullptr, vbr_hca);

    LLAMA_LOG_INFO("%s: creating DSV4 lightning-indexer KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_CSA_RATIO));

    kv_lid = std::make_unique<llama_kv_cache>(
            model, hparams_lid, type_lid, type_lid,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, nullptr, nullptr, vbr_child);

    if (vbr.dynamic && !model.hparams.no_alloc) {
        llama_kv_cache * leader = kv_raw->get_swa();
        if (!leader->vbr_controller_active()) {
            throw std::runtime_error("DSV4 dynamic VBR could not arm the raw-cache controller");
        }

        // The generic order is normalized over the KV-bearing layers of one cache. CSA and HCA
        // each contain only a subset, so independently synthesizing their orders would rank the
        // same model layer differently. Install the raw cache's full-model order in both; their
        // absent-layer entries become no-ops and the next applicable step names exactly the
        // compressed partner of the raw step.
        const auto install_order = [&](llama_kv_cache * child) {
            if (!child->vbr_controller_active()) {
                return;
            }
            child->vbr_degrade_order_ = leader->vbr_degrade_order_;
            child->vbr_degrade_cursor_ = 0;
            child->t8_band_end_ = leader->t8_band_end_;
            child->vbr_floor_typed_ = leader->vbr_floor_typed_;
        };
        install_order(kv_csa.get());
        install_order(kv_hca.get());

        // CSA pads its compressor graph to the reserve-time block count by writing one
        // deliberately invisible row at the end of each cache tensor. Prefix-only VMM mapping
        // does not otherwise back that address when no complete 4-token block exists yet.
        // Record the semantic write requirement; the generic mapper follows its current-tier
        // byte address and protects only that one page from tail reclamation.
        for (auto & pool : kv_csa->vbr_pools_) {
            for (auto & extent : pool.k) {
                if (extent.t != nullptr) {
                    GGML_ASSERT(extent.t->ne[2] == 1); // dynamic VMM requires unified KV
                    extent.vmm_required_write_row = (uint32_t) extent.t->ne[1] - 1;
                }
            }
        }
        vbr_floor_group(GGML_TYPE_COUNT, vbr.min_bits, /*pooled_only=*/true,
                /*install_runtime_limit=*/true);
        const std::array<llama_kv_cache *, 3> group = { leader, kv_csa.get(), kv_hca.get() };
        if (vbr.budget_bytes > 0 && vbr.budget_explicit) {
            size_t floor_total = 0;
            size_t extra_weight_total = 0;
            for (const llama_kv_cache * child : group) {
                floor_total += child->vbr_floor_cost_bytes_;
                for (const auto & pool : child->vbr_pools_) {
                    if (pool.vmm != nullptr) {
                        extra_weight_total += pool.size > pool.budget_base
                            ? pool.size - pool.budget_base : 0;
                    }
                }
            }
            if (vbr.budget_bytes < floor_total) {
                throw std::runtime_error(format(
                        "DSV4 VBR budget %.2f MiB is below the page-exact %.2f MiB floor layout",
                        vbr.budget_bytes/1048576.0, floor_total/1048576.0));
            }
            size_t extra_left = (size_t) vbr.budget_bytes - floor_total;
            size_t weight_left = extra_weight_total;
            for (llama_kv_cache * child : group) {
                size_t child_budget = 0;
                for (auto & pool : child->vbr_pools_) {
                    if (pool.vmm == nullptr) {
                        continue;
                    }
                    const size_t weight = pool.size > pool.budget_base
                        ? pool.size - pool.budget_base : 0;
                    const size_t add = weight_left == 0 ? extra_left :
                        (size_t) ((long double) extra_left * weight / weight_left);
                    pool.budget = pool.budget_base + add;
                    pool.budget_eff_stamp = ~0ull;
                    child_budget += pool.budget;
                    extra_left -= add;
                    weight_left -= weight;
                }
                child->vbr_budget_bytes_ = child_budget;
            }
        } else {
            for (llama_kv_cache * child : group) {
                size_t child_budget = 0;
                for (auto & pool : child->vbr_pools_) {
                    if (pool.vmm != nullptr) {
                        pool.budget = std::max(pool.budget, pool.budget_base);
                        pool.budget_eff_stamp = ~0ull;
                        child_budget += pool.budget;
                    }
                }
                child->vbr_budget_bytes_ = std::max(child_budget, child->vbr_floor_cost_bytes_);
            }
        }

        // The current ledger tree represents at most two ordinary caches. Publishing three
        // independent markers from one DSV4 memory would double-count the same logical donor,
        // so grouped DSV4 starts with process-external co-tenancy disabled. The local aggregate
        // budget and all recoverable map/reserve behavior remain active.
        for (llama_kv_cache * child : { kv_raw->get_base(), leader, kv_csa.get(), kv_hca.get() }) {
            child->vbr_ledger_owner_ = false;
            child->vbr_ledger_root_ = nullptr;
            child->vbr_ledger_sibling_ = nullptr;
        }
        LLAMA_LOG_WARN("%s: DSV4 grouped VBR does not yet publish co-tenancy offers; "
                "runtime tiering and explicit/auto local budgets remain active\n", __func__);

        leader->vbr_group_prepare_set([this](uint32_t raw_wm) {
            return vbr_prepare_group(raw_wm);
        });
        vbr_group_enabled_ = true;
    }

    LLAMA_LOG_INFO("%s: creating DSV4 CSA compressor state\n", __func__);

    csa_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO,
            2*model.hparams.n_embd_head_k(), n_rs_seq, "csa", filter_csa);

    LLAMA_LOG_INFO("%s: creating DSV4 HCA compressor state\n", __func__);

    hca_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, DSV4_HCA_RATIO, DSV4_HCA_RATIO,
            model.hparams.n_embd_head_k(), n_rs_seq, "hca", filter_hca);

    LLAMA_LOG_INFO("%s: creating DSV4 lightning-indexer compressor state\n", __func__);

    lid_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO,
            2*model.hparams.indexer_head_size, n_rs_seq, "lid", filter_csa);

    // DSV4 attention reads compressed-K / compressor-state rows that the current
    // graph does not necessarily overwrite; uninitialized buffer contents would
    // otherwise leak in (instance-specific garbage) and corrupt recall. Zero all
    // compressed buffers up front so reads of un-written rows are deterministic.
    clear_compressed(-1, true);
}

double llama_kv_cache_dsv4::vbr_floor_group(
        ggml_type entry_k, double floor_bpv, bool pooled_only,
        bool install_runtime_limit) {
    llama_kv_cache * raw = kv_raw->get_swa();

    if (raw->vbr_degrade_order_.empty()) {
        raw->vbr_load_degrade_order();
    }
    // Every child must walk the full-model ordinal space. Missing layers are intentional
    // no-ops; this is what gives one ordinal the same model-layer meaning in all three caches.
    for (llama_kv_cache * child : { kv_csa.get(), kv_hca.get() }) {
        child->vbr_degrade_order_ = raw->vbr_degrade_order_;
        child->t8_band_end_ = raw->t8_band_end_;
    }

    struct sim_child {
        llama_kv_cache * cache;
        double rate;
        std::vector<ggml_type> types;
        double bits = 0.0;
        int64_t vals = 0;
    };
    sim_child sr { raw,          1.0,                 {} };
    sim_child sc { kv_csa.get(), 1.0/DSV4_CSA_RATIO, {} };
    sim_child sh { kv_hca.get(), 1.0/DSV4_HCA_RATIO, {} };
    std::array<sim_child *, 3> sims = { &sr, &sc, &sh };

    double sum_bits = 0.0;
    double sum_vals = 0.0;
    for (sim_child * s : sims) {
        s->cache->vbr_sim_seed(s->types, pooled_only, entry_k, entry_k,
                &s->bits, &s->vals, nullptr);
        sum_bits += s->rate * s->bits;
        sum_vals += s->rate * s->vals;
    }

    const double floor_eff = dsv4_vbr_floor_bpv(floor_bpv);
    size_t limit = raw->vbr_degrade_order_.size();
    for (size_t i = 0; i < raw->vbr_degrade_order_.size(); ++i) {
        const auto & order = raw->vbr_degrade_order_[i];
        if (order.is_v) {
            continue; // DSV4 stores one K==V tensor per component.
        }

        const uint32_t ratio = hparams_raw.dsv4_compress_ratios[order.il];
        sim_child * follower = ratio == DSV4_CSA_RATIO ? &sc :
                               ratio == DSV4_HCA_RATIO ? &sh : nullptr;
        size_t raw_slot = 0, follower_slot = 0;
        const ggml_tensor * raw_tensor = nullptr;
        const ggml_tensor * follower_tensor = nullptr;
        ggml_type raw_next = GGML_TYPE_COUNT;
        ggml_type follower_next = GGML_TYPE_COUNT;
        const bool raw_moves = raw->vbr_sim_step(
                sr.types, i, raw_slot, raw_tensor, raw_next);
        const bool follower_moves = follower != nullptr && follower->cache->vbr_sim_step(
                follower->types, i, follower_slot, follower_tensor, follower_next);
        if (!raw_moves && !follower_moves) {
            continue;
        }
        // Compressed layers have exactly one partner; ratio-0 layers attend raw KV directly.
        GGML_ASSERT(raw_moves && (follower == nullptr || follower_moves));

        const double raw_delta =
            - 8.0 * ggml_row_size(sr.types[raw_slot], raw_tensor->ne[0])
            + 8.0 * ggml_row_size(raw_next, raw_tensor->ne[0]);
        const double follower_delta = follower != nullptr ?
            - 8.0 * ggml_row_size(follower->types[follower_slot], follower_tensor->ne[0])
            + 8.0 * ggml_row_size(follower_next, follower_tensor->ne[0]) : 0.0;
        const double next_bits = sum_bits + raw_delta +
            (follower != nullptr ? follower->rate * follower_delta : 0.0);
        if (sum_vals > 0.0 && next_bits / sum_vals < floor_eff - 1e-9) {
            limit = i;
            break;
        }
        sr.types[raw_slot] = raw_next;
        if (follower != nullptr) {
            follower->types[follower_slot] = follower_next;
            follower->bits += follower_delta;
        }
        sr.bits += raw_delta;
        sum_bits = next_bits;
    }

    if (install_runtime_limit) {
        const auto install = [&](sim_child & s) {
            s.cache->vbr_degrade_limit_ = limit;
            s.cache->vbr_floor_cost_bytes_ = 0;
            for (auto & pool : s.cache->vbr_pools_) {
                size_t pool_floor = pool.mapped_base;
                for (size_t ikv = 0; ikv < s.cache->layers.size(); ++ikv) {
                    auto & extent = pool.k[ikv];
                    if (extent.t == nullptr) {
                        continue;
                    }
                    const ggml_type type = s.types[ikv*2];
                    if (type != GGML_TYPE_COUNT) {
                        const size_t need = ggml_row_size(type, extent.t->ne[0]) *
                            (size_t) extent.t->ne[1];
                        pool_floor += GGML_PAD(need, pool.gran);
                    }
                }
                pool.budget_base = pool_floor;
                pool.budget_eff_stamp = ~0ull;
                s.cache->vbr_floor_cost_bytes_ += pool_floor;
            }
        };
        install(sr);
        install(sc);
        install(sh);
        LLAMA_LOG_INFO("%s: DSV4 aggregate VBR floor %.4g bits/value clamps the ganged order at %zu/%zu steps\n",
                __func__, floor_eff, limit, raw->vbr_degrade_order_.size());
    }
    if (kv_size == 0) {
        return 0.0;
    }
    // Fit scales the bytes of the actually allocated, padded dry tensors. Preserve that exact
    // geometry here; the 1/ratio weights above are only the quality-floor definition.
    return sr.bits * (double) raw->get_size() / kv_size +
           sc.bits * (double) kv_csa->get_size() / kv_size +
           sh.bits * (double) kv_hca->get_size() / kv_size;
}

llama_kv_cache_dsv4::vbr_group_budget_state llama_kv_cache_dsv4::vbr_group_budget(
        uint32_t raw_wm, uint32_t csa_wm, uint32_t hca_wm, bool live) const {
    struct device_total {
        const ggml_vbr_backend_iface * be = nullptr;
        size_t projected = 0;
        size_t configured = 0;
        size_t mapped = 0;
        size_t growth_headroom = 0;
        bool explicit_budget = false;
        bool freeze = false;
    };
    struct child_state {
        const llama_kv_cache * cache;
        uint32_t wm;
    };
    const std::array<child_state, 3> children = {{
        { kv_raw->get_swa(), raw_wm },
        { kv_csa.get(), csa_wm },
        { kv_hca.get(), hca_wm },
    }};

    std::map<int, device_total> totals;
    for (const auto & child : children) {
        if (!child.cache->vbr_controller_active()) {
            continue;
        }
        for (const auto & pool : child.cache->vbr_pools_) {
            if (pool.vmm == nullptr) {
                continue;
            }
            auto & total = totals[pool.device];
            GGML_ASSERT(total.be == nullptr || total.be == pool.be);
            total.be = pool.be;
            total.projected += child.cache->vbr_vmm_projected_bytes(pool, child.wm);
            total.configured += pool.budget;
            total.mapped += pool.be->vmm_pool_mapped(pool.vmm);
            total.growth_headroom = std::max(
                    total.growth_headroom, child.cache->vbr_growth_headroom_);
            total.explicit_budget |= child.cache->vbr_budget_explicit_;
            total.freeze |= child.cache->vbr_freeze_;
        }
    }
    vbr_group_budget_state result;
    result.max_deficit = INT64_MIN;
    for (const auto & [device, total] : totals) {
        size_t effective = total.configured;
        if (live && !total.freeze) {
            size_t free_b = 0;
            size_t total_b = 0;
            total.be->get_device_memory(device, &free_b, &total_b);
            GGML_UNUSED(total_b);
            const size_t ledger_headroom = llama_vram_headroom_bytes();
            const size_t grow_room = effective > total.mapped
                ? effective - total.mapped : 0;
            const size_t headroom = total.explicit_budget
                ? std::max(std::min(total.growth_headroom, grow_room), ledger_headroom)
                : ledger_headroom;
            const size_t cap = total.mapped + (free_b > headroom ? free_b - headroom : 0);
            effective = std::min(effective, cap);
        }
        effective = std::max(effective, total.mapped);
        result.bytes_needed += total.projected;
        result.bytes_available += effective;
        const int64_t deficit = total.projected > effective
            ? (int64_t) std::min(total.projected - effective,
                    (size_t) std::numeric_limits<int64_t>::max())
            : - (int64_t) std::min(effective - total.projected,
                    (size_t) std::numeric_limits<int64_t>::max());
        result.max_deficit = std::max(result.max_deficit, deficit);
    }
    if (result.max_deficit == INT64_MIN) {
        result.max_deficit = 0;
    }
    return result;
}

double llama_kv_cache_dsv4::vbr_group_projected_bpv(
        uint32_t raw_wm, uint32_t csa_wm, uint32_t hca_wm) const {
    struct sim_child {
        const llama_kv_cache * cache;
        uint32_t wm;
        std::vector<ggml_type> types;
        double bits = 0.0;
        double vals = 0.0;
    };
    sim_child sr { kv_raw->get_swa(), raw_wm, {} };
    sim_child sc { kv_csa.get(), csa_wm, {} };
    sim_child sh { kv_hca.get(), hca_wm, {} };
    std::array<sim_child *, 3> children = { &sr, &sc, &sh };

    std::map<int, int64_t> projected;
    std::map<int, int64_t> configured;
    const auto extent_bytes = [](const llama_kv_cache::vbr_pool & pool,
                                 const llama_kv_cache::vbr_extent & extent,
                                 ggml_type type, uint32_t wm) {
        const size_t slot = GGML_PAD(
                ggml_row_size(GGML_TYPE_F16, extent.t->ne[0]) *
                    (size_t) extent.t->ne[1] * extent.t->ne[2],
                pool.gran);
        const size_t need = ggml_row_size(type, extent.t->ne[0]) * (size_t) wm;
        size_t bytes = std::min(slot, (size_t) GGML_PAD(need, pool.gran));
        if (extent.vmm_required_write_row != UINT32_MAX) {
            const size_t rel = ggml_row_size(type, extent.t->ne[0]) *
                extent.vmm_required_write_row;
            const size_t page = (rel / pool.gran) * pool.gran;
            if (page >= bytes) {
                bytes += pool.gran;
            }
        }
        return bytes;
    };

    for (sim_child * child : children) {
        double bits = 0.0;
        int64_t vals = 0;
        child->cache->vbr_sim_seed(child->types, /*pooled_only=*/true,
                GGML_TYPE_COUNT, GGML_TYPE_COUNT, &bits, &vals, nullptr);
        child->bits = bits * child->cache->get_size();
        child->vals = (double) vals * child->cache->get_size();
        for (const auto & pool : child->cache->vbr_pools_) {
            if (pool.vmm == nullptr) {
                continue;
            }
            configured[pool.device] += (int64_t) pool.budget;
            projected[pool.device] += (int64_t) pool.mapped_base;
            for (size_t ikv = 0; ikv < child->cache->layers.size(); ++ikv) {
                for (int side = 0; side < 2; ++side) {
                    const auto & extent = side ? pool.v[ikv] : pool.k[ikv];
                    if (extent.t != nullptr) {
                        projected[pool.device] += (int64_t) extent_bytes(
                                pool, extent, child->types[ikv*2 + side], child->wm);
                    }
                }
            }
        }
    }

    const auto fits = [&]() {
        for (const auto & [device, bytes] : projected) {
            if (bytes > configured[device]) {
                return false;
            }
        }
        return true;
    };
    const auto apply_step = [&](sim_child & child, size_t i) {
        size_t slot = 0;
        const ggml_tensor * tensor = nullptr;
        ggml_type next = GGML_TYPE_COUNT;
        if (!child.cache->vbr_sim_step(child.types, i, slot, tensor, next)) {
            return false;
        }
        const size_t ikv = slot/2;
        const bool is_v = (slot & 1) != 0;
        const ggml_type prev = child.types[slot];
        for (const auto & pool : child.cache->vbr_pools_) {
            if (pool.vmm == nullptr) {
                continue;
            }
            const auto & extent = is_v ? pool.v[ikv] : pool.k[ikv];
            if (extent.t == nullptr) {
                continue;
            }
            projected[pool.device] +=
                (int64_t) extent_bytes(pool, extent, next, child.wm) -
                (int64_t) extent_bytes(pool, extent, prev, child.wm);
        }
        child.bits += 8.0 * ((double) ggml_row_size(next, tensor->ne[0]) -
                (double) ggml_row_size(prev, tensor->ne[0])) * child.cache->get_size();
        child.types[slot] = next;
        return true;
    };

    const size_t end = std::min(sr.cache->vbr_degrade_order_.size(),
            sr.cache->vbr_degrade_limit_);
    for (size_t i = sr.cache->vbr_degrade_cursor_; i < end && !fits(); ++i) {
        const auto & order = sr.cache->vbr_degrade_order_[i];
        if (order.is_v) {
            continue;
        }
        const uint32_t ratio = hparams_raw.dsv4_compress_ratios[order.il];
        sim_child * follower = ratio == DSV4_CSA_RATIO ? &sc :
                               ratio == DSV4_HCA_RATIO ? &sh : nullptr;
        const bool raw_moves = apply_step(sr, i);
        const bool follower_moves = follower != nullptr && apply_step(*follower, i);
        GGML_ASSERT(!raw_moves || follower == nullptr || follower_moves);
    }

    double bits = 0.0;
    double vals = 0.0;
    for (const sim_child * child : children) {
        bits += child->bits;
        vals += child->vals;
    }
    // Keep the public BPV metric scoped to the raw/compressed attention cache.
    // LID is immutable auxiliary indexer state: fit prices its bytes separately,
    // but static and live codec telemetry should still report the selected tier.
    return vals > 0.0 ? bits/vals : -1.0;
}

bool llama_kv_cache_dsv4::vbr_prepare_group(uint32_t raw_wm) {
    llama_kv_cache * raw = kv_raw->get_swa();
    const uint32_t csa_wm = std::min(vbr_pending_csa_wm_, kv_csa->get_size());
    const uint32_t hca_wm = std::min(vbr_pending_hca_wm_, kv_hca->get_size());

    struct child_boundary {
        llama_kv_cache * cache;
        uint32_t wm;
    };
    const std::array<child_boundary, 3> children = {{
        { raw, raw_wm }, { kv_csa.get(), csa_wm }, { kv_hca.get(), hca_wm },
    }};

    for (const auto & child : children) {
        if (!child.cache->vbr_controller_active()) {
            continue;
        }
        child.cache->vbr_runtime_was_over_ = false;
        child.cache->vbr_reserve_failed_ = false;
        child.cache->vbr_hard_seal_blocked_ = false;
        child.cache->vbr_hard_seal_evidence_.clear();
        child.cache->vbr_flush_deferred_unmaps();
        child.cache->vbr_invalidate_dirty_stash();
        child.cache->vbr_last_prepare_ns_ = llama_vram_ledger_now_ns();
        if (!child.cache->vbr_budget_explicit_ && child.cache->vbr_boundary_count_ % 8 == 0) {
            child.cache->vbr_rederive_budget();
        }
    }

    uint32_t used_raw = 0;
    for (const auto & cells : raw->v_cells) {
        used_raw += cells.get_used();
    }
    if (raw->vbr_degrade_cursor_ > 0 && used_raw == 0 &&
            !raw->vbr_f5_preserve_empty_tiers_) {
        for (const auto & child : children) {
            if (child.cache->vbr_controller_active() && child.cache->vbr_degrade_cursor_ > 0) {
                child.cache->vbr_full_reset();
            }
        }
    }
    for (const auto & child : children) {
        if (child.cache->vbr_controller_active()) {
            child.cache->vbr_shrink_watermark_to(child.wm);
        }
    }

    const auto over_budget = [&]() {
        return vbr_group_budget(raw_wm, csa_wm, hca_wm, /*live=*/true).max_deficit > 0;
    };

    const auto reserve_unit_stashes = [&](llama_kv_cache * cache, int32_t il) {
        if (cache->vbr_stash_rows_ == 0) {
            return true;
        }
        const auto found = cache->map_layer_ids.find(il);
        GGML_ASSERT(found != cache->map_layer_ids.end());
        for (const auto & [pool, extent] : cache->vbr_units_of(found->second, false)) {
            if (pool->wm_cells == 0) {
                continue;
            }
            const bool capture = extent->stash_valid == 0 &&
                ggml_is_turbo_kv_type(extent->t->type) &&
                extent->t->type != GGML_TYPE_TURBO8_0;
            const uint32_t rows = capture
                ? std::min(cache->vbr_stash_rows_, pool->wm_cells)
                : extent->stash_valid;
            if (rows > 0) {
                const std::vector<llama_kv_cache::vbr_stash_request> request = {{ extent, rows }};
                if (!cache->vbr_stash_reserve(*pool, request)) {
                    return false;
                }
            }
        }
        return true;
    };

    const auto next_raw_step = [&]() -> std::pair<size_t, int32_t> {
        const size_t end = std::min(raw->vbr_degrade_order_.size(), raw->vbr_degrade_limit_);
        for (size_t i = raw->vbr_degrade_cursor_; i < end; ++i) {
            const auto & step = raw->vbr_degrade_order_[i];
            const auto found = raw->map_layer_ids.find(step.il);
            if (found == raw->map_layer_ids.end()) {
                continue;
            }
            const size_t ikv = found->second;
            ggml_tensor * tensor = step.is_v ? raw->layers[ikv].v : raw->layers[ikv].k;
            if (tensor == nullptr || raw->vbr_units_of(ikv, step.is_v != 0).empty() ||
                    !raw->vbr_unit_movable(tensor->type, step.is_v != 0)) {
                continue;
            }
            const ggml_type next = dsv4_vbr_tier_type(step.tier);
            if (tensor->type != next &&
                    ggml_row_size(next, tensor->ne[0]) < ggml_row_size(tensor->type, tensor->ne[0])) {
                return { i, step.il };
            }
        }
        return { end, -1 };
    };

    const bool frozen = raw->vbr_retier_freeze_depth_ > 0 ||
        kv_csa->vbr_retier_freeze_depth_ > 0 || kv_hca->vbr_retier_freeze_depth_ > 0;
    if (!frozen) {
        vbr_hard_seal_consult_session seal_session;
        while (over_budget()) {
            const auto [order_ordinal, il] = next_raw_step();
            if (il < 0) {
                break;
            }
            const uint32_t ratio = hparams_raw.dsv4_compress_ratios[il];
            llama_kv_cache * follower = ratio == DSV4_CSA_RATIO ? kv_csa.get() :
                                        ratio == DSV4_HCA_RATIO ? kv_hca.get() : nullptr;
            const uint32_t follower_wm = follower == kv_csa.get() ? csa_wm :
                                         follower == kv_hca.get() ? hca_wm : 0;

            // One raw logical range owns both physical halves of the pair. Consult that range
            // once against the canonical full-model ordinal: f16->t8 remains restorable and is
            // allowed under a lease, while the first t4-or-lower step reports typed evidence to
            // the server so it can retire the conflicting checkpoint and retry.
            if (raw->vbr_hard_seal_step_blocked(order_ordinal, seal_session)) {
                raw->vbr_hard_seal_blocked_ = true;
                raw->vbr_hard_seal_evidence_record(order_ordinal);
                LLAMA_LOG_WARN("%s: DSV4 ganged retier reached a hard-lease-protected step\n",
                        __func__);
                return false;
            }
            if (!reserve_unit_stashes(raw, il) ||
                    (follower != nullptr && !reserve_unit_stashes(follower, il))) {
                LLAMA_LOG_ERROR("%s: DSV4 ganged sink-stash reserve failed before tier mutation\n",
                        __func__);
                return false;
            }

            const auto follower_result = follower != nullptr ? follower->vbr_degrade_next(follower_wm, true) :
                llama_kv_cache::vbr_degrade_result::applied;
            const auto raw_result = raw->vbr_degrade_next(raw_wm, true);
            if (follower_result != llama_kv_cache::vbr_degrade_result::applied ||
                    raw_result != llama_kv_cache::vbr_degrade_result::applied) {
                GGML_ABORT("DSV4 ganged VBR invariant failed for layer %d", il);
            }
        }
    }

    for (const auto & child : children) {
        if (!child.cache->vbr_controller_active()) {
            continue;
        }
        child.cache->vbr_arm_wave_fences();
        if (!child.cache->vbr_vmm_try_map(child.wm)) {
            LLAMA_LOG_ERROR("%s: DSV4 VBR physical map to %u cells failed recoverably\n",
                    __func__, child.wm);
            return false;
        }
    }

    // Attention sees concat(raw, compressed) as one coupled K==V tensor. Reserve both backend
    // scratch sides for that combined attended width; reserving each child independently would
    // cover only max(raw, compressed) and leave the second half to grow inside the graph.
    const size_t attended_cells = (size_t) raw_wm + std::max(csa_wm, hca_wm);
    for (auto & pool : raw->vbr_pools_) {
        if (pool.be == nullptr || pool.device < 0) {
            continue;
        }
        size_t row = 0;
        for (const auto & extent : pool.k) {
            if (extent.t != nullptr && ggml_is_turbo_kv_type(extent.t->type)) {
                row = std::max(row, ggml_row_size(GGML_TYPE_F16, extent.t->ne[0]));
            }
        }
        const size_t bytes = row * attended_cells;
        if (bytes != 0 && !pool.be->kv_dequant_scratch_reserve(
                pool.compute_backend, bytes, bytes)) {
            for (const auto & child : children) {
                child.cache->vbr_flush_deferred_unmaps();
            }
            if (!pool.be->kv_dequant_scratch_reserve(pool.compute_backend, bytes, bytes)) {
                LLAMA_LOG_ERROR("%s: DSV4 coupled f16 dequant scratch reserve failed recoverably\n",
                        __func__);
                return false;
            }
        }
        pool.scratch_k_reserved = std::max(pool.scratch_k_reserved, bytes);
        pool.scratch_v_reserved = std::max(pool.scratch_v_reserved, bytes);
    }

    for (const auto & child : children) {
        if (!child.cache->vbr_controller_active()) {
            continue;
        }
        child.cache->vbr_boundary_count_++;
        child.cache->vbr_last_wm_ = child.wm;
        child.cache->vbr_trace_emit("prepare_group", child.wm, used_raw);
    }
    return true;
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    const bool raw_per_seq  = kv_raw->get_base()->get_n_stream() != 1;
    const bool comp_per_seq = csa_state->get_n_stream() > 1;
    const bool has_coupled = dsv4_batch_has_coupled(balloc.get_batch());

    const auto make_context = [&](std::vector<llama_ubatch> ubatches) -> llama_memory_context_ptr {
        auto ubatches_raw = dsv4_build_raw_write_ubatches(ubatches);

        if (vbr_group_enabled_) {
            const auto csa_plans = dsv4_build_comp_plans(
                    ubatches, DSV4_CSA_RATIO, true,
                    csa_state->get_state_size(), kv_csa->get_size(), csa_state->get_n_stream(),
                    n_rs_seq, rs_idx);
            const auto hca_plans = dsv4_build_comp_plans(
                    ubatches, DSV4_HCA_RATIO, false,
                    hca_state->get_state_size(), kv_hca->get_size(), hca_state->get_n_stream(),
                    n_rs_seq, rs_idx);
            vbr_pending_csa_wm_ = 0;
            vbr_pending_hca_wm_ = 0;
            for (const auto & plan : csa_plans) {
                vbr_pending_csa_wm_ = std::max<uint32_t>(vbr_pending_csa_wm_, (uint32_t) plan.n_kv);
            }
            for (const auto & plan : hca_plans) {
                vbr_pending_hca_wm_ = std::max<uint32_t>(vbr_pending_hca_wm_, (uint32_t) plan.n_kv);
            }
        }

        auto sinfos_raw_base_write = kv_raw->get_base()->prepare(ubatches_raw);
        if (sinfos_raw_base_write.empty()) {
            return nullptr;
        }

        auto sinfos_raw_swa_write = kv_raw->get_swa()->prepare(ubatches_raw);
        if (sinfos_raw_swa_write.empty()) {
            return nullptr;
        }

        auto sinfos_raw_swa_read = dsv4_build_raw_read_sinfos(sinfos_raw_swa_write, ubatches);

        return std::make_unique<llama_kv_cache_dsv4_context>(
                this,
                std::move(sinfos_raw_base_write),
                std::move(sinfos_raw_swa_write),
                std::move(sinfos_raw_swa_read),
                std::move(ubatches),
                std::move(ubatches_raw));
    };

    // Match llama_kv_cache_iswa splitting when DSV4 compressed state does not
    // require per-sequence graph layout.
    do {
        if (raw_per_seq || comp_per_seq) {
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (auto ctx = make_context(std::move(ubatches))) {
            return ctx;
        }
    } while (false);

    // When raw or compressed state is per-sequence, independent sequences can
    // share an equal-length ubatch. Coupled sequence sets still serialize until
    // DSV4 has explicit shared-state handling for compressed streams.
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (has_coupled) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                ubatch = balloc.split_equal(n_ubatch, raw_per_seq || comp_per_seq, 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (auto ctx = make_context(std::move(ubatches))) {
            return ctx;
        }
    } while (false);

    return std::make_unique<llama_kv_cache_dsv4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_full() {
    return std::make_unique<llama_kv_cache_dsv4_context>(this);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_dsv4_context>(
            this,
            lctx,
            optimize,
            std::move(csa_state->sc_info),
            std::move(hca_state->sc_info),
            std::move(lid_state->sc_info));
}

bool llama_kv_cache_dsv4::get_can_shift() const {
    // Compressed row metadata uses block-derived positions. Keep shifting
    // disabled until DSV4 compressed-cache shift semantics are wired.
    return false;
}

void llama_kv_cache_dsv4::breathe() {
    if (!vbr_group_enabled_) {
        kv_raw->breathe();
        kv_csa->breathe();
        kv_hca->breathe();
        kv_lid->breathe();
        return;
    }
    // Ordinary child ticks may retier independently. Reuse the parent boundary so an idle-time
    // auto-budget refresh cannot leave either half of a concat at a different row stride.
    (void) vbr_prepare_group(kv_raw->get_swa()->vbr_watermark_cells(0));
}

double llama_kv_cache_dsv4::kv_bpv() const {
    double bits = 0.0;
    double vals = 0.0;
    kv_raw->get_base()->kv_bpv_accum(bits, vals);
    kv_raw->get_swa ()->kv_bpv_accum(bits, vals);
    kv_csa->kv_bpv_accum(bits, vals);
    kv_hca->kv_bpv_accum(bits, vals);
    return vals > 0.0 ? bits / vals : -1.0;
}

llama_memory_vbr_state_data llama_kv_cache_dsv4::memory_vbr_state(
        llama_seq_id seq_id, uint32_t n_tokens_extra) const {
    const auto r = kv_raw->memory_vbr_state(seq_id, n_tokens_extra);
    const auto c = kv_csa->memory_vbr_state(seq_id,
            (n_tokens_extra + DSV4_CSA_RATIO - 1)/DSV4_CSA_RATIO);
    const auto h = kv_hca->memory_vbr_state(seq_id,
            (n_tokens_extra + DSV4_HCA_RATIO - 1)/DSV4_HCA_RATIO);
    llama_memory_vbr_state_data out = r;
    const uint32_t raw_wm = kv_raw->get_swa()->vbr_watermark_cells(n_tokens_extra);
    const uint32_t csa_wm = std::max(vbr_pending_csa_wm_,
            kv_csa->vbr_watermark_cells((n_tokens_extra + DSV4_CSA_RATIO - 1)/DSV4_CSA_RATIO));
    const uint32_t hca_wm = std::max(vbr_pending_hca_wm_,
            kv_hca->vbr_watermark_cells((n_tokens_extra + DSV4_HCA_RATIO - 1)/DSV4_HCA_RATIO));
    const auto budget_raw = vbr_group_budget(raw_wm, csa_wm, hca_wm, /*live=*/false);
    const auto budget_live = vbr_group_budget(raw_wm, csa_wm, hca_wm, /*live=*/true);
    out.deficit_raw = budget_raw.max_deficit;
    out.deficit_clamped = budget_live.max_deficit;
    out.bpv_if_degraded = budget_raw.max_deficit > 0
        ? vbr_group_projected_bpv(raw_wm, csa_wm, hca_wm)
        : kv_bpv();
    out.cursor = r.cursor;
    out.representation_epoch_swa ^= c.representation_epoch +
        0x9e3779b97f4a7c15ULL + (h.representation_epoch << 6) + (h.representation_epoch >> 2);
    out.retier_freeze_depth = std::max({ r.retier_freeze_depth, c.retier_freeze_depth, h.retier_freeze_depth });
    out.retier_env_freeze = std::max({ r.retier_env_freeze, c.retier_env_freeze, h.retier_env_freeze });
    out.retier_freeze_enters = std::max({ r.retier_freeze_enters, c.retier_freeze_enters, h.retier_freeze_enters });
    out.retier_freeze_exits = std::max({ r.retier_freeze_exits, c.retier_freeze_exits, h.retier_freeze_exits });
    out.retier_deferred_decisions = r.retier_deferred_decisions +
        c.retier_deferred_decisions + h.retier_deferred_decisions;
    out.retier_reconciles = std::max({ r.retier_reconciles, c.retier_reconciles, h.retier_reconciles });
    return out;
}

bool llama_kv_cache_dsv4::vbr_operation_armed() const {
    return vbr_group_enabled_ && kv_raw->vbr_operation_armed();
}

bool llama_kv_cache_dsv4::vbr_retier_freeze_begin(
        const char * owner, vbr_operation_id operation_id) {
    if (!vbr_operation_armed() || vbr_freeze_depth_ >= VBR_RETIER_FREEZE_MAX_DEPTH) {
        return false;
    }
    vbr_freeze_children frame = { operation_id };
    frame.raw = kv_raw->vbr_retier_freeze_begin(owner, operation_id);
    if (!frame.raw) {
        return false;
    }
    frame.csa = !kv_csa->vbr_operation_armed() || kv_csa->vbr_retier_freeze_begin(owner, operation_id);
    if (!frame.csa) {
        kv_raw->vbr_retier_freeze_end(owner, operation_id);
        return false;
    }
    frame.hca = !kv_hca->vbr_operation_armed() || kv_hca->vbr_retier_freeze_begin(owner, operation_id);
    if (!frame.hca) {
        if (kv_csa->vbr_operation_armed()) kv_csa->vbr_retier_freeze_end(owner, operation_id);
        kv_raw->vbr_retier_freeze_end(owner, operation_id);
        return false;
    }
    vbr_freeze_stack_[vbr_freeze_depth_++] = frame;
    return true;
}

void llama_kv_cache_dsv4::vbr_retier_freeze_end(
        const char * owner, vbr_operation_id operation_id) {
    GGML_ASSERT(vbr_freeze_depth_ > 0);
    const auto frame = vbr_freeze_stack_[vbr_freeze_depth_ - 1];
    GGML_ASSERT(frame.operation_id == operation_id);
    if (frame.raw) kv_raw->vbr_retier_freeze_end(owner, operation_id);
    if (frame.csa && kv_csa->vbr_operation_armed()) kv_csa->vbr_retier_freeze_end(owner, operation_id);
    if (frame.hca && kv_hca->vbr_operation_armed()) kv_hca->vbr_retier_freeze_end(owner, operation_id);
    vbr_freeze_stack_[--vbr_freeze_depth_] = {};
}

void llama_kv_cache_dsv4::vbr_commit_submitted() {
    kv_raw->vbr_commit_submitted(); kv_csa->vbr_commit_submitted(); kv_hca->vbr_commit_submitted();
}

void llama_kv_cache_dsv4::vbr_decode_ops_finish(bool ok) {
    kv_raw->vbr_decode_ops_finish(ok); kv_csa->vbr_decode_ops_finish(ok); kv_hca->vbr_decode_ops_finish(ok);
}

void llama_kv_cache_dsv4::vbr_adopt_operation(vbr_operation_id operation_id) {
    kv_raw->vbr_adopt_operation(operation_id); kv_csa->vbr_adopt_operation(operation_id); kv_hca->vbr_adopt_operation(operation_id);
}

void llama_kv_cache_dsv4::vbr_release_adopted() {
    kv_raw->vbr_release_adopted(); kv_csa->vbr_release_adopted(); kv_hca->vbr_release_adopted();
}

llama_memory_vbr_preflight_data llama_kv_cache_dsv4::vbr_retier_preflight(
        uint32_t n_tokens_extra) const {
    llama_memory_vbr_preflight_data out = {};
    out.fits = true;
    if (!vbr_group_enabled_) {
        return out;
    }

    llama_kv_cache * raw = kv_raw->get_swa();
    const uint32_t raw_wm = raw->vbr_watermark_cells(n_tokens_extra);
    const uint32_t csa_wm = std::max(vbr_pending_csa_wm_,
            kv_csa->vbr_watermark_cells((n_tokens_extra + DSV4_CSA_RATIO - 1)/DSV4_CSA_RATIO));
    const uint32_t hca_wm = std::max(vbr_pending_hca_wm_,
            kv_hca->vbr_watermark_cells((n_tokens_extra + DSV4_HCA_RATIO - 1)/DSV4_HCA_RATIO));
    const auto budget = vbr_group_budget(raw_wm, csa_wm, hca_wm, /*live=*/true);
    out.active = true;
    out.watermark_cells = std::max({ raw_wm, csa_wm, hca_wm });
    out.bytes_needed = budget.bytes_needed;
    out.bytes_available = budget.bytes_available;
    out.max_deficit = budget.max_deficit;
    out.fits = budget.max_deficit <= 0;

    struct device_growth {
        const ggml_vbr_backend_iface * be = nullptr;
        size_t kv = 0;
        size_t scratch_need = 0;
        size_t scratch_have = 0;
    };
    std::map<int, device_growth> growth;
    struct child_state { const llama_kv_cache * cache; uint32_t wm; };
    const std::array<child_state, 3> children = {{
        { raw, raw_wm }, { kv_csa.get(), csa_wm }, { kv_hca.get(), hca_wm },
    }};
    for (const auto & child : children) {
        for (const auto & pool : child.cache->vbr_pools_) {
            if (pool.vmm == nullptr) {
                continue;
            }
            out.pools++;
            auto & g = growth[pool.device];
            GGML_ASSERT(g.be == nullptr || g.be == pool.be);
            g.be = pool.be;
            const size_t needed = child.cache->vbr_vmm_projected_bytes(pool, child.wm);
            const size_t mapped = pool.be->vmm_pool_mapped(pool.vmm);
            g.kv += needed > mapped ? needed - mapped : 0;
        }
    }

    const size_t attended_cells = (size_t) raw_wm + std::max(csa_wm, hca_wm);
    for (const auto & pool : raw->vbr_pools_) {
        if (pool.be == nullptr || pool.device < 0) {
            continue;
        }
        size_t row = 0;
        for (const auto & extent : pool.k) {
            if (extent.t != nullptr && ggml_is_turbo_kv_type(extent.t->type)) {
                row = std::max(row, ggml_row_size(GGML_TYPE_F16, extent.t->ne[0]));
            }
        }
        auto & g = growth[pool.device];
        g.scratch_need = row * attended_cells;
        g.scratch_have = std::max(pool.scratch_k_reserved, pool.scratch_v_reserved);
    }
    for (const auto & [device, g] : growth) {
        size_t free_b = 0;
        size_t total_b = 0;
        g.be->get_device_memory(device, &free_b, &total_b);
        GGML_UNUSED(total_b);
        const size_t scratch_delta = g.scratch_need > g.scratch_have
            ? g.scratch_need - g.scratch_have : 0;
        const size_t physical_need = g.kv + 2*scratch_delta;
        const int64_t deficit = physical_need > free_b
            ? (int64_t) std::min(physical_need - free_b,
                    (size_t) std::numeric_limits<int64_t>::max()) : 0;
        out.physical_growth_needed += physical_need;
        out.physical_growth_available += free_b;
        out.max_deficit = std::max(out.max_deficit, deficit);
        out.fits = out.fits && deficit == 0;
    }
    return out;
}

double llama_kv_cache_dsv4::memory_vbr_floor_bits_per_token(
        ggml_type entry_k, ggml_type entry_v, double floor_bpv) {
    (void) entry_v; // DSV4 stores one K==V tensor.
    if (kv_size == 0) return 0.0;
    double bits = vbr_floor_group(entry_k, floor_bpv,
            /*pooled_only=*/false, /*install_runtime_limit=*/false);
    // LID is fixed F16 but is part of the measured KV bytes which fit scales by the returned
    // floor/price ratio. Keep its immutable cost in both numerator and denominator.
    for (const auto & layer : kv_lid->layers) {
        if (layer.k) bits += 8.0 * ggml_row_size(layer.k->type, layer.k->ne[0]) *
            (double) kv_lid->get_size() / kv_size;
    }
    return bits;
}

double llama_kv_cache_dsv4::memory_vbr_scratch_bytes_per_token(
        ggml_type entry_k, ggml_type entry_v, double floor_bpv) {
    const double raw_row = kv_raw->get_swa()->memory_vbr_scratch_bytes_per_token(
            entry_k, entry_v, floor_bpv);
    if (raw_row == 0.0 || kv_size == 0) return 0.0;
    const double width = (double) (kv_raw->get_swa()->get_size() +
            std::max(kv_csa->get_size(), kv_hca->get_size())) / kv_size;
    return 2.0 * raw_row * width; // coupled K==V materializes both backend scratch sides
}

void llama_kv_cache_dsv4::vbr_cotenancy_accum(
        uint64_t &, uint32_t &, uint64_t &, uint64_t &) const {
    // Grouped DSV4 is deliberately not published until the ledger supports >2 children.
}

bool llama_kv_cache_dsv4::vbr_ledger_tree_active() const { return false; }

void llama_kv_cache_dsv4::vbr_hard_seal_guard_set(vbr_hard_seal_guard guard) {
    kv_raw->vbr_hard_seal_guard_set(guard);
    kv_csa->vbr_hard_seal_guard_set(guard);
    kv_hca->vbr_hard_seal_guard_set(std::move(guard));
}

bool llama_kv_cache_dsv4::vbr_hard_seal_blocked_take(bool decode_failed) {
    return kv_raw->vbr_hard_seal_blocked_take(decode_failed) |
           kv_csa->vbr_hard_seal_blocked_take(decode_failed) |
           kv_hca->vbr_hard_seal_blocked_take(decode_failed);
}

void llama_kv_cache_dsv4::vbr_hard_seal_evidence_take(
        std::vector<vbr_hard_seal_subject> & out) {
    kv_raw->vbr_hard_seal_evidence_take(out);
    kv_csa->vbr_hard_seal_evidence_take(out);
    kv_hca->vbr_hard_seal_evidence_take(out);
}

void llama_kv_cache_dsv4::vbr_shared_scratch_detach() {
    kv_raw->vbr_shared_scratch_detach();
    kv_csa->vbr_shared_scratch_detach();
    kv_hca->vbr_shared_scratch_detach();
    kv_lid->vbr_shared_scratch_detach();
}

uint64_t llama_kv_cache_dsv4::get_vbr_epoch() const {
    return kv_raw->get_base()->vbr_representation_epoch() +
           kv_raw->get_swa()->vbr_representation_epoch() +
           kv_csa->vbr_representation_epoch() +
           kv_hca->vbr_representation_epoch();
}

void llama_kv_cache_dsv4::clear(bool data) {
    kv_raw->clear(data);
    clear_compressed(-1, true); // DSV4 compressed buffers must never expose stale/uninit rows
}

bool llama_kv_cache_dsv4::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p1 >= 0) {
        return false;
    }

    if (p0 > 0) {
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
            return false;
        }

        const llama_pos pos_max = kv_raw->seq_pos_max(seq_id);
        if (p0 > pos_max) {
            bool res = true;

            res = res & kv_raw->seq_rm(seq_id, p0, -1);
            res = res & kv_csa->seq_rm(seq_id, p0/DSV4_CSA_RATIO, -1);
            res = res & kv_hca->seq_rm(seq_id, p0/DSV4_HCA_RATIO, -1);
            res = res & kv_lid->seq_rm(seq_id, p0/DSV4_CSA_RATIO, -1);

            return res;
        }

        if (n_rs_seq == 0) {
            return false;
        }

        const llama_pos rollback = pos_max - (p0 - 1);
        if (rollback < 1 || rollback > (llama_pos) n_rs_seq) {
            return false;
        }

        const bool res = kv_raw->seq_rm(seq_id, p0, p1);
        if (res) {
            rs_idx[seq_id] = (uint32_t) rollback;
        }

        return res;
    }

    const bool res = kv_raw->seq_rm(seq_id, p0, p1);

    if (res) {
        clear_compressed(seq_id, true);
    }

    return res;
}

void llama_kv_cache_dsv4::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "DSV4 only supports full sequence copies");

    kv_raw->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_csa->seq_cp(seq_id_src, seq_id_dst, -1, -1);
    kv_hca->seq_cp(seq_id_src, seq_id_dst, -1, -1);
    kv_lid->seq_cp(seq_id_src, seq_id_dst, -1, -1);

    csa_state->seq_cp(seq_id_src, seq_id_dst);
    hca_state->seq_cp(seq_id_src, seq_id_dst);
    lid_state->seq_cp(seq_id_src, seq_id_dst);

    if (seq_id_src != seq_id_dst) {
        rs_idx[seq_id_dst] = 0;
    }
}

void llama_kv_cache_dsv4::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    kv_raw->seq_keep(seq_id);

    for (llama_seq_id id = 0; id < (llama_seq_id) n_seq_max; ++id) {
        if (id == seq_id) {
            continue;
        }

        kv_raw->seq_rm(id, -1, -1);
        clear_compressed(id, true);
    }
}

void llama_kv_cache_dsv4::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_raw->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_dsv4::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_raw->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_dsv4::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    // The raw SWA cache may contain a wider window, but the compressed DSV4
    // state cannot be rolled back within that window. Report only the current
    // boundary so server-context uses checkpoints for rollback.
    return kv_raw->seq_pos_max(seq_id);
}

llama_pos llama_kv_cache_dsv4::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    return kv_raw->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsv4::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_raw->memory_breakdown();
    for (const auto & buft_size : kv_csa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_hca->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_lid->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : csa_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : hca_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : lid_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_kv_cache_dsv4::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const bool partial_only = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

    const uint32_t magic   = DSV4_STATE_MAGIC;
    const uint32_t version = DSV4_STATE_VERSION;
    const uint32_t mode    = partial_only ? DSV4_STATE_MODE_PARTIAL : DSV4_STATE_MODE_FULL;

    io.write(&magic,   sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&mode,    sizeof(mode));

    kv_raw->state_write(io, seq_id, flags);

    if (!partial_only) {
        const llama_pos pos_max = seq_id >= 0 ? kv_raw->seq_pos_max(seq_id) : -1;

        //FIXME : note that we conflate token positions with rows, which is not true for multi-modal case.
        const uint32_t n_rows_csa = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_CSA_RATIO, kv_csa->get_size()) : kv_csa->get_size();
        const uint32_t n_rows_hca = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_HCA_RATIO, kv_hca->get_size()) : kv_hca->get_size();
        const uint32_t n_rows_lid = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_CSA_RATIO, kv_lid->get_size()) : kv_lid->get_size();

        dsv4_state_write_k_cache(io, kv_csa.get(), seq_id, flags, n_rows_csa);
        dsv4_state_write_k_cache(io, kv_hca.get(), seq_id, flags, n_rows_hca);
        dsv4_state_write_k_cache(io, kv_lid.get(), seq_id, flags, n_rows_lid);
    }

    csa_state->state_write(io, seq_id, flags, rs_idx);
    hca_state->state_write(io, seq_id, flags, rs_idx);
    lid_state->state_write(io, seq_id, flags, rs_idx);
}

void llama_kv_cache_dsv4::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    uint32_t magic;
    uint32_t version;
    uint32_t mode = DSV4_STATE_MODE_FULL;

    io.read(&magic,   sizeof(magic));
    io.read(&version, sizeof(version));

    if (magic != DSV4_STATE_MAGIC) {
        throw std::runtime_error("DSV4 state magic mismatch");
    }
    if (version != DSV4_STATE_VERSION) {
        throw std::runtime_error("DSV4 state version mismatch");
    }

    io.read(&mode, sizeof(mode));
    if (mode != DSV4_STATE_MODE_FULL && mode != DSV4_STATE_MODE_PARTIAL) {
        throw std::runtime_error("DSV4 state mode mismatch");
    }

    const bool partial_only = mode == DSV4_STATE_MODE_PARTIAL;
    if (partial_only != !!(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        throw std::runtime_error("DSV4 state flags mismatch");
    }

    kv_raw->state_read(io, seq_id, flags);

    if (!partial_only) {
        kv_csa->clear(true);
        kv_hca->clear(true);
        kv_lid->clear(true);

        dsv4_state_read_k_cache(io, kv_csa.get(), seq_id, flags);
        dsv4_state_read_k_cache(io, kv_hca.get(), seq_id, flags);
        dsv4_state_read_k_cache(io, kv_lid.get(), seq_id, flags);
    }

    csa_state->state_read(io, seq_id, flags);
    hca_state->state_read(io, seq_id, flags);
    lid_state->state_read(io, seq_id, flags);

    if (seq_id >= 0) {
        GGML_ASSERT((uint32_t) seq_id < n_seq_max);
        rs_idx[seq_id] = 0;
    } else {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
    }
}

llama_kv_cache_iswa * llama_kv_cache_dsv4::get_raw() const {
    return kv_raw.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_csa() const {
    return kv_csa.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_hca() const {
    return kv_hca.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_lid() const {
    return kv_lid.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_csa_state() const {
    return csa_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_hca_state() const {
    return hca_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_lid_state() const {
    return lid_state.get();
}

uint32_t llama_kv_cache_dsv4::get_n_rs_seq() const {
    return n_rs_seq;
}

const std::vector<uint32_t> & llama_kv_cache_dsv4::get_rs_idx() const {
    return rs_idx;
}

void llama_kv_cache_dsv4::reset_rs_idx_for_ubatches(const std::vector<llama_ubatch> & ubatches) {
    if (n_rs_seq == 0) {
        return;
    }

    for (const llama_ubatch & ubatch : ubatches) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = ubatch.seq_id[i][s];
                if (seq_id >= 0 && (uint32_t) seq_id < n_seq_max) {
                    rs_idx[seq_id] = 0;
                }
            }
        }
    }
}

void llama_kv_cache_dsv4::clear_compressed(llama_seq_id seq_id, bool data) {
    // fork: under the fit-probe dry context (hparams.no_alloc) buffers are sized but never
    // allocated — writing the zero-fill would trip GGML_ASSERT(tensor->data != NULL). The
    // metadata side (seq_rm, rs_idx) still runs so probe-time pricing stays consistent.
    data = data && !hparams_raw.no_alloc;

    if (seq_id < 0) {
        kv_csa->clear(data);
        kv_hca->clear(data);
        kv_lid->clear(data);
    } else {
        GGML_ASSERT((uint32_t) seq_id < n_seq_max);

        const auto clear_seq = [seq_id, data](llama_kv_cache * kv) {
            // Dynamic VMM is admitted only for one unified stream. Its tensor VA contains
            // intentionally unmapped pages, so the ordinary per-stream tensor memset below is
            // invalid; llama_kv_cache::clear instead zeroes only resident VMM pages.
            if (data && kv->vbr_vmm_active()) {
                GGML_ASSERT(kv->get_n_stream() == 1 && seq_id == 0);
                kv->clear(true);
                return;
            }
            kv->seq_rm(seq_id, -1, -1);

            if (data) {
                for (uint32_t il : kv->get_layer_ids()) {
                    dsv4_clear_tensor_stream(kv->get_k_storage(il), (uint32_t) seq_id);
                }
            }
        };

        clear_seq(kv_csa.get());
        clear_seq(kv_hca.get());
        clear_seq(kv_lid.get());
    }

    csa_state->clear(seq_id, data);
    hca_state->clear(seq_id, data);
    lid_state->clear(seq_id, data);

    if (seq_id >= 0) {
        rs_idx[seq_id] = 0;
    } else {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
    }
}

//
// llama_kv_cache_dsv4_raw_context
//

static llama_kv_cache::slot_info dsv4_build_full_sinfo(const llama_kv_cache * kv) {
    const uint32_t n_stream = kv->get_n_stream();

    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = n_stream - 1;
    sinfo.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfo.strm[s] = s;
        sinfo.idxs[s].resize(1, 0);
    }

    return sinfo;
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv) :
    kv_swa(kv->get_swa()),
    ctx_base_mem(nullptr),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
    sinfos_read.push_back(dsv4_build_full_sinfo(kv_swa));
    sinfos_write = sinfos_read;
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    kv_swa(kv->get_swa()),
    ctx_base_mem(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa_mem(kv->get_swa()->init_update(lctx, optimize)),
    n_kv(kv_swa->get_size()),
    status(llama_memory_status_combine(ctx_base_mem->get_status(), ctx_swa_mem->get_status())) {
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base_write,
        slot_info_vec_t sinfos_swa_write,
        slot_info_vec_t sinfos_swa_read,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_ubatch> ubatches_write) :
    kv_swa(kv->get_swa()),
    sinfos_write(std::move(sinfos_swa_write)),
    sinfos_read(std::move(sinfos_swa_read)),
    ubatches(std::move(ubatches)),
    ubatches_write(std::move(ubatches_write)),
    ctx_base_mem(std::make_unique<llama_kv_cache_context>(
                kv->get_base(), std::move(sinfos_base_write), this->ubatches_write)),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
}

bool llama_kv_cache_dsv4_raw_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (ctx_base_mem) {
        ctx_base_mem->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_raw_context::apply() {
    bool res = true;

    if (ctx_base_mem) {
        res = res & ctx_base_mem->apply();
    }
    if (ctx_swa_mem) {
        res = res & ctx_swa_mem->apply();
    }
    if (!ubatches_write.empty()) {
        kv_swa->apply_ubatch(sinfos_write[i_next], ubatches_write[i_next]);
        n_kv = kv_swa->get_n_kv(sinfos_read[i_next]);
    }

    return res;
}

uint64_t llama_kv_cache_dsv4_raw_context::get_vbr_epoch() const {
    return kv_swa != nullptr ? kv_swa->vbr_representation_epoch() : 0;
}

llama_memory_status llama_kv_cache_dsv4_raw_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv4_raw_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_write() const {
    if (ubatches_write.empty()) {
        return 0;
    }

    return ubatches_write[i_next].n_tokens;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv_swa->get_k(ctx, il, n_kv, sinfos_read[i_next]);
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    const auto & sinfo = sinfos_write[i_next];

    if (k_cur->ne[2] == k_idxs->ne[0]) {
        return kv_swa->cpy_k(ctx, k_cur, k_idxs, il, sinfo);
    }

    // k_idxs may be expanded to one block per stream while k_cur is only
    // the token block. Keep zero deps on all copies so each write executes.
    const int64_t n_fanout = (int64_t) sinfo.size()*sinfo.n_stream();

    GGML_ASSERT(sinfo.n_stream() > 1);
    GGML_ASSERT(k_cur->ne[2] == (int64_t) sinfo.size());
    GGML_ASSERT(k_idxs->ne[0] == n_fanout);

    ggml_tensor * res = nullptr;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        ggml_tensor * k_idxs_s = ggml_view_1d(ctx, k_idxs, sinfo.size(), s*sinfo.size()*ggml_element_size(k_idxs));
        ggml_tensor * cur = kv_swa->cpy_k(ctx, k_cur, k_idxs_s, il, sinfo);
        if (res == nullptr) {
            res = cur;
        } else {
            res = ggml_add(ctx, res, ggml_sub(ctx, cur, cur));
        }
    }

    return res;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatches_write.empty() ? ubatch.n_tokens : ubatches_write[i_next].n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_rot(ggml_context * ctx) const {
    return kv_swa->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_idxs(ggml_tensor * dst) const {
    kv_swa->set_input_k_idxs(dst, &ubatches_write[i_next], sinfos_write[i_next]);
}

void llama_kv_cache_dsv4_raw_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv_swa->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_rot(ggml_tensor * dst) const {
    kv_swa->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_comp_context
//

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(llama_kv_cache * kv) : kv(kv), n_kv(kv->get_size()) {
    const uint32_t n_stream = kv->get_n_stream();

    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(
        llama_kv_cache * kv,
        slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) :
    kv(kv),
    sinfos(std::move(sinfos)),
    ubatches(std::move(ubatches)),
    n_kv(kv->get_size()) {
}

bool llama_kv_cache_dsv4_comp_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

uint32_t llama_kv_cache_dsv4_comp_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_comp_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_context
//

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv) :
    kv(kv),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw())),
    ctx_csa_mem(kv->get_csa()->init_full()),
    ctx_hca_mem(kv->get_hca()->init_full()),
    ctx_lid_mem(kv->get_lid()->init_full()),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_csa())),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_hca())),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_lid())),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    reserve_plans(true),
    status(llama_memory_status_combine(
                llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status()))) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        llama_context * lctx,
        bool optimize,
        stream_copy_info sc_info_csa,
        stream_copy_info sc_info_hca,
        stream_copy_info sc_info_lid) :
    kv(kv),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw(), lctx, optimize)),
    ctx_csa_mem(kv->get_csa()->init_update(lctx, optimize)),
    ctx_hca_mem(kv->get_hca()->init_update(lctx, optimize)),
    ctx_lid_mem(kv->get_lid()->init_update(lctx, optimize)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    sc_info_csa(std::move(sc_info_csa)),
    sc_info_hca(std::move(sc_info_hca)),
    sc_info_lid(std::move(sc_info_lid)),
    status(llama_memory_status_combine(
                llama_memory_status_combine(
                    llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                    llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status())),
                this->sc_info_csa.empty() && this->sc_info_hca.empty() && this->sc_info_lid.empty() ?
                    LLAMA_MEMORY_STATUS_NO_UPDATE : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        slot_info_vec_t sinfos_raw_base_write,
        slot_info_vec_t sinfos_raw_swa_write,
        slot_info_vec_t sinfos_raw_swa_read,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_ubatch> ubatches_raw) :
    kv(kv),
    ubatches(std::move(ubatches)),
    plans_csa(dsv4_build_comp_plans(this->ubatches, DSV4_CSA_RATIO, true,
                kv->get_csa_state()->get_state_size(), kv->get_csa()->get_size(), kv->get_csa_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_rs_idx())),
    plans_hca(dsv4_build_comp_plans(this->ubatches, DSV4_HCA_RATIO, false,
                kv->get_hca_state()->get_state_size(), kv->get_hca()->get_size(), kv->get_hca_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_rs_idx())),
    plans_lid(dsv4_build_comp_plans(this->ubatches, DSV4_CSA_RATIO, true,
                kv->get_lid_state()->get_state_size(), kv->get_lid()->get_size(), kv->get_lid_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_rs_idx())),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(
                kv->get_raw(),
                std::move(sinfos_raw_base_write),
                std::move(sinfos_raw_swa_write),
                std::move(sinfos_raw_swa_read),
                this->ubatches,
                std::move(ubatches_raw))),
    ctx_csa_mem(nullptr),
    ctx_hca_mem(nullptr),
    ctx_lid_mem(nullptr),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_csa(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_csa()->get_n_stream()),
                this->ubatches)),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_hca(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_hca()->get_n_stream()),
                this->ubatches)),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_lid(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_lid()->get_n_stream()),
                this->ubatches)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    status(ctx_raw->get_status()) {
    kv->reset_rs_idx_for_ubatches(this->ubatches);
}

llama_kv_cache_dsv4_context::~llama_kv_cache_dsv4_context() = default;

bool llama_kv_cache_dsv4_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_raw->next();
    ctx_csa->next();
    ctx_hca->next();
    ctx_lid->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_raw->apply();

    if (ctx_csa_mem) {
        res = res & ctx_csa_mem->apply();
        res = res & ctx_hca_mem->apply();
        res = res & ctx_lid_mem->apply();
    }

    if (ubatches.empty()) {
        csa_state->apply_copies(sc_info_csa);
        hca_state->apply_copies(sc_info_hca);
        lid_state->apply_copies(sc_info_lid);
    }

    return res;
}

uint64_t llama_kv_cache_dsv4_context::get_vbr_epoch() const {
    return kv != nullptr ? kv->get_vbr_epoch() : 0;
}

llama_memory_status llama_kv_cache_dsv4_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv4_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_dsv4_raw_context * llama_kv_cache_dsv4_context::get_raw() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_raw.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_csa() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_csa.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_hca() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_hca.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_lid() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_lid.get();
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_csa_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return csa_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_hca_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return hca_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_lid_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return lid_state;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_csa.empty()) {
        return empty;
    }

    return plans_csa[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_hca.empty()) {
        return empty;
    }

    return plans_hca[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_lid.empty()) {
        return empty;
    }

    return plans_lid[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_csa_plan();
    }

    reserve_plan_csa = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_CSA_RATIO, true,
            csa_state->get_state_size(), get_csa()->get_n_kv(), csa_state->get_n_stream(), csa_state->get_n_rs_seq());

    return reserve_plan_csa;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_hca_plan();
    }

    reserve_plan_hca = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_HCA_RATIO, false,
            hca_state->get_state_size(), get_hca()->get_n_kv(), hca_state->get_n_stream(), hca_state->get_n_rs_seq());

    return reserve_plan_hca;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_lid_plan();
    }

    reserve_plan_lid = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_CSA_RATIO, true,
            lid_state->get_state_size(), get_lid()->get_n_kv(), lid_state->get_n_stream(), lid_state->get_n_rs_seq());

    return reserve_plan_lid;
}
