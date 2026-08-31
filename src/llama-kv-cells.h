#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <bitset>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <set>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

    // when tok = LLAMA_TOKEN_NULL when the cell is produced by embedding input (i.e. multimodal)
    // use case: n-gram embeddings hash
    llama_token tok = LLAMA_TOKEN_NULL;

    // return true if the current 2D spatial position is greater than other
    bool is_2d_gt(llama_pos ox, llama_pos oy) const {
        return (y > oy) || (y == oy && x > ox);
    }

    void reset() {
        static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>);

        *this = llama_kv_cell_ext{};
    }
};

// Index of the lowest and highest set bit. Callers guarantee x != 0.
#if defined(_MSC_VER)
static inline uint32_t llama_kv_ctz64(uint64_t x) {
    unsigned long r;
    _BitScanForward64(&r, x);
    return (uint32_t) r;
}

static inline uint32_t llama_kv_clz64(uint64_t x) {
    unsigned long r;
    _BitScanReverse64(&r, x);
    return (uint32_t) r;
}
#else
static inline uint32_t llama_kv_ctz64(uint64_t x) {
    return (uint32_t) __builtin_ctzll(x);
}

static inline uint32_t llama_kv_clz64(uint64_t x) {
    return (uint32_t) (63 - __builtin_clzll(x));
}
#endif

// Used cache cells are dense enough that a bitmap is both smaller and substantially cheaper to
// traverse than one tree node per index.
class llama_kv_idx_set {
public:
    void resize(uint32_t n) {
        bits.assign((n + 63)/64, 0);
        n_set = 0;
    }

    void clear() {
        std::fill(bits.begin(), bits.end(), 0);
        n_set = 0;
    }

    void insert(uint32_t i) {
        uint64_t & w = bits[i/64];
        const uint64_t b = 1ull << (i%64);

        n_set += (w & b) == 0;
        w |= b;
    }

    void erase(uint32_t i) {
        uint64_t & w = bits[i/64];
        const uint64_t b = 1ull << (i%64);

        n_set -= (w & b) != 0;
        w &= ~b;
    }

    uint32_t size() const { return n_set; }
    bool empty() const { return n_set == 0; }

    uint32_t first() const {
        for (size_t w = 0; w < bits.size(); ++w) {
            if (bits[w]) {
                return 64*w + llama_kv_ctz64(bits[w]);
            }
        }
        return 0;
    }

    uint32_t last() const {
        for (size_t w = bits.size(); w-- > 0;) {
            if (bits[w]) {
                return 64*w + llama_kv_clz64(bits[w]);
            }
        }
        return 0;
    }

    template <class F>
    void for_each(F && f) const {
        for (size_t w = 0; w < bits.size(); ++w) {
            uint64_t m = bits[w];
            while (m) {
                f((uint32_t) (64*w + llama_kv_ctz64(m)));
                m &= m - 1;
            }
        }
    }

private:
    std::vector<uint64_t> bits;
    uint32_t n_set = 0;
};

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class llama_kv_cells {
public:
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    void reset() {
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
        }

        has_shift = false;

        used.clear();

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
        }
    }

    void reset_shift() {
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            shift[i] = 0;
        }
    }

    uint32_t size() const {
        return pos.size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        ext.resize(n);
        shift.resize(n);
        seq.resize(n);
        used.resize(n);

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used.size();
    }

    // the index of the first cell that is used
    // return 0 if no cells are used
    uint32_t used_min() const {
        return used.empty() ? 0 : used.first();
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        return used.empty() ? 0 : used.last() + 1;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // move cell isrc to idst (used during defrag)
    //void mv(uint32_t isrc, uint32_t idst) {
    //    assert(isrc < pos.size());
    //    assert(idst < pos.size());

    //    assert(pos[idst] == -1);
    //    assert(pos[isrc] != -1);

    //    pos  [idst] = pos  [isrc];
    //    shift[idst] = shift[isrc];
    //    seq  [idst] = seq  [isrc];

    //    pos  [isrc] = -1;
    //    shift[isrc] =  0;
    //    seq  [isrc].reset();

    //    used.erase (isrc);
    //    used.insert(idst);
    //}

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llama_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = i + j;

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(i + j);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(i + j);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(i + j);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }

            assert(shift[idx] == 0);
        }
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = idxs[j];

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(idx);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(idx);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(idx);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }

            assert(shift[idx] == 0);
        }
    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);
        seq[i].reset();

        pos[i] = -1;
        ext[i].reset();
        shift[i] = 0;

        used.erase(i);
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, i);

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            seq_pos_rm(i);
            seq[i].reset();

            seq[i].set(seq_id);
            seq_pos_inc(seq_id, i);

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();

            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i].count();
    }

    // The complete visibility set is part of the QSA block identity in a unified cache.
    const seq_set_t & seq_get_all(uint32_t i) const {
        assert(i < pos.size());

        return seq[i];
    }

    // check if the cell contains seq_id
    // Iterate the sequences ACTUALLY occupying cell i — O(occupants), not O(LLAMA_MAX_SEQ).
    // Ownership-index maintenance walks this per touched cell on whole-cache edits.
    template <typename F>
    void seq_for_each(uint32_t i, F && f) const {
        assert(i < seq.size());
        const auto & set = seq[i];
#if defined(__GNUC__)
        for (size_t s = set._Find_first(); s < set.size(); s = set._Find_next(s)) {
            f((llama_seq_id) s);
        }
#else
        for (size_t s = 0; s < set.size(); ++s) {
            if (set.test(s)) {
                f((llama_seq_id) s);
            }
        }
#endif
    }

    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // Token in the sequence cell at the greatest position <= p. When several cells
    // share a temporal position (M-RoPE), the highest physical index wins, matching
    // the previous ascending-cell scan. Returns LLAMA_TOKEN_NULL when no predecessor
    // exists. Used by the PLE n-gram input path.
    llama_token seq_pos_tok_le(llama_seq_id seq_id, llama_pos p) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        const auto & positions = seq_pos[seq_id];
        auto it = positions.upper_bound({ p, std::numeric_limits<uint32_t>::max() });
        if (it == positions.begin()) {
            return LLAMA_TOKEN_NULL;
        }

        return ext[(--it)->second].tok;
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        seq[i].set(seq_id);
        seq_pos_inc(seq_id, i);
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llama_seq_id seq_get(uint32_t i) const {
        assert(seq[i].count() == 1);

        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                return s;
            }
        }

        return -1;
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        return seq_pos[seq_id].begin()->first;
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        return seq_pos[seq_id].rbegin()->first;
    }

    // Exact cardinality from the canonical ownership index rather than trusting
    // the covered mask's subset count. It allocates nothing and never scans empty
    // physical cells; the child supplies the visibility/serializer-manifest filter.
    uint32_t seq_pos_count_before(llama_seq_id seq_id, llama_pos frontier) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return static_cast<uint32_t>(std::distance(
                seq_pos[seq_id].begin(), seq_pos[seq_id].lower_bound({ frontier, 0 })));
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    const llama_kv_cell_ext & ext_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return ext[i];
    }

    // note: call only if the cell is not empty
    llama_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llama_pos p0, llama_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        pos[i] = p;

        used.insert(i);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        return pos_add_impl(i, d, true, false);
    }

    // Text-only M-RoPE broadcasts the temporal position into the x/y planes.
    // Keep all three pieces of metadata coherent while queuing the matching
    // rotary-key update.
    bool pos_add_mrope_text(uint32_t i, llama_pos d) {
        return pos_add_impl(i, d, true, true);
    }

    // Some auxiliary caches (Qwen4 QSA) retain raw, pre-RoPE keys.  Their
    // block metadata follows the shifted text position, but their key bytes
    // must never be submitted to the K-shift graph.
    bool pos_add_mrope_text_raw(uint32_t i, llama_pos d) {
        return pos_add_impl(i, d, false, true);
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
    }

private:
    bool pos_add_impl(uint32_t i, llama_pos d, bool rotate_key, bool broadcast_text) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        pos[i] += d;
        if (broadcast_text) {
            ext[i].x += d;
            ext[i].y += d;
        }
        if (rotate_key) {
            shift[i] += d;
        }

        has_shift = has_shift || rotate_key;

        if (pos[i] < 0) {
            seq[i].reset();
            pos[i] = -1;
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    llama_kv_idx_set used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // One (position, physical cell) entry per cell carrying sequence s. Including the
    // cell index preserves repeated temporal positions and permits logarithmic predecessor
    // lookup while begin/rbegin remain the min/max position authorities.
    // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
    std::set<std::pair<llama_pos, uint32_t>> seq_pos[LLAMA_MAX_SEQ];

    // helper functions for updating `seq_pos`, once cell at a time:

    void seq_pos_dec(llama_seq_id s, uint32_t i) {
        const size_t n = seq_pos[s].erase({ pos[i], i });
        assert(n == 1);
        GGML_UNUSED(n);
    }

    void seq_pos_inc(llama_seq_id s, uint32_t i) {
        const bool inserted = seq_pos[s].insert({ pos[i], i }).second;
        assert(inserted);
        GGML_UNUSED(inserted);
    }

    // remove cell i
    void seq_pos_rm(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_dec(s, i);
            }
        }
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_inc(s, i);
            }
        }
    }
};

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
