#pragma once

#include "llama.h"
#include "llama-graph.h"
#include "llama-vbr-operation.h"
#include "llama-vbr-hard-seal.h"

#include <map>
#include <memory>
#include <functional>
#include <limits>
#include <vector>

struct llama_ubatch;
// Internal physical-growth evidence retained across composite memory trees.
// The public preflight remains scalar, but composites must not sum per-device
// availability before checking the combined children that share that device.
struct llama_memory_vbr_physical_growth {
    // Process-local identity of the backend/device namespace. This is opaque on
    // purpose: composites only compare it and must never dereference it.
    const void * backend = nullptr;
    int device = -1;
    // KV mappings are distinct per child and add. Fattn scratch is shared by all
    // children on a device: merge the absolute target/current endpoints by max,
    // then charge their difference once per side.
    uint64_t kv_needed = 0;
    uint64_t scratch_k_needed = 0;
    uint64_t scratch_v_needed = 0;
    uint64_t scratch_k_current = 0;
    uint64_t scratch_v_current = 0;
    uint64_t available = 0;

    bool same_domain(const llama_memory_vbr_physical_growth & other) const noexcept {
        return backend == other.backend && device == other.device;
    }
    bool operator<(const llama_memory_vbr_physical_growth & other) const noexcept {
        if (backend != other.backend) {
            return std::less<const void *>{}(backend, other.backend);
        }
        return device < other.device;
    }
};

// Generic reduction for independently priced memory children. Logical totals
// add; physical rows sharing one backend/device are combined before fit is
// decided so tensor-split skew cannot disappear into scalar totals.
llama_memory_vbr_preflight_data llama_memory_vbr_merge_preflight_children(
    const llama_memory_vbr_preflight_data & first,
    const std::vector<llama_memory_vbr_physical_growth> & first_physical,
    const llama_memory_vbr_preflight_data & second,
    const std::vector<llama_memory_vbr_physical_growth> & second_physical,
    std::vector<llama_memory_vbr_physical_growth> * physical = nullptr,
    uint64_t * domain_comparisons = nullptr);

// Incremental reduction for a stable set of memory children. The physical
// domain shape of each leaf is immutable after build; replacing one leaf then
// recomputes only its ancestors.
class llama_memory_vbr_preflight_tree {
public:
    bool reset(size_t leaf_count) noexcept;
    bool set_leaf(
        size_t index,
        const llama_memory_vbr_preflight_data & preflight,
        const std::vector<llama_memory_vbr_physical_growth> & physical) noexcept;
    bool build() noexcept;
    bool replace_leaf(
        size_t index,
        const llama_memory_vbr_preflight_data & preflight,
        const std::vector<llama_memory_vbr_physical_growth> & physical) noexcept;

    bool ready() const noexcept;
    const llama_memory_vbr_preflight_data & preflight() const noexcept;
    const std::vector<llama_memory_vbr_physical_growth> & physical() const noexcept;
    uint64_t merge_count() const noexcept;
    uint64_t domain_comparison_count() const noexcept;

private:
    struct node {
        bool present = false;
        llama_memory_vbr_preflight_data preflight = {};
        std::vector<llama_memory_vbr_physical_growth> physical;
    };
    void merge_node(size_t index);

    size_t leaf_count_ = 0;
    size_t leaf_base_ = 0;
    bool ready_ = false;
    uint64_t merge_count_ = 0;
    uint64_t domain_comparison_count_ = 0;
    std::vector<node> tree_;
};

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;

    // Resolve a simple KV buffer type to this llama_context's main compute backend instance.
    // VBR uses the exact instance to reserve backend-context-owned flash-attention scratch.
    std::function<ggml_backend_t(ggml_backend_buffer_type_t)> compute_backend_for_buft;
};

// TurboQuant dynamic-VBR runtime parameters, threaded from llama_context_params through
// create_memory into the attention KV caches (backend side: ggml-vbr.h). The VBR_VMM /
// VBR_MODE / VBR_BUDGET_MIB / VBR_MIN_BITS environment variables remain available as
// developer overrides on top of these.
struct llama_memory_vbr_params {
    bool     dynamic      = false; // arm the VMM pool + decode-time degrade controller
    uint64_t budget_bytes = 0;     // mapped-physical KV budget; 0 = floor-layout-cost fallback
    double   min_bits     = 0.0;   // aggregate bits/value floor (0 = bottom-tier floor)
    // the floor was TYPED (flag or env): doubles as peer-yield consent down to it
    bool     min_bits_explicit = false;
    // explicit budgets are HARD CAPS: the runtime never re-derives them from live free VRAM
    // (an auto budget floats within [floor-layout cost, live reach] at decode boundaries)
    bool     budget_explicit = false;
    // free-VRAM headroom kept while growing a VBR pool (0 = 1 GiB default; the fit passes its
    // --fit-target so startup and runtime encode the same worst case)
    uint64_t growth_headroom_bytes = 0;
    // this cache's fraction of its device's spare VRAM (iSWA children share a device; the
    // parent splits by entry-tier footprint so the children never double-claim the same free)
    double   device_share = 1.0;
    // mixed-config side pins, see llama.h vbr_pin_k
    bool     pin_k = false;
    bool     pin_v = false;
    // VBR_TRACE path suffix so iSWA base/SWA children write to distinct files
    // instead of both truncating the same path. nullptr for a standalone cache.
    const char * trace_label = nullptr;

    std::function<ggml_backend_t(ggml_backend_buffer_type_t)> compute_backend_for_buft;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;

    // Maximum number of sequences a graph built against this context can represent
    // at the current physical allocation. Most memory types do not impose a tighter
    // bound than the owning llama_context's configured sequence maximum.
    virtual uint32_t get_max_graph_seqs() const {
        return std::numeric_limits<uint32_t>::max();
    }

    // TurboQuant: get rotation tensors for pre-rotate-queries optimization
    // Returns null for non-turbo memory types. Override in KV cache contexts.
    virtual ggml_tensor * get_turbo_rot_forward() const { return nullptr; }
    virtual ggml_tensor * get_turbo_rot_inverse() const { return nullptr; }

    // VBR tier-flip epoch: bumped by in-place tier flips that rewrite the cache tensors'
    // type/strides with no shape change any graph input can see. llm_graph_result::can_reuse
    // fences graph reuse on it once, for every input class. Composite memory types sum their
    // children so a flip in any child forces a rebuild. 0 for memory without VBR.
    virtual uint64_t get_vbr_epoch() const { return 0; }
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// Allocation-free ordered tier identity for scheduler-private automatic
// capture retry. Ordinary attention-content mutation does not move it.
struct llama_memory_vbr_representation_identity {
    uint64_t tier_epoch = 0;
    uint64_t tier_epoch_swa = 0;
};

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    // Whether this memory views KV cells owned by another context. Context linkage alone
    // cannot answer this: every MTP context points at its target, while only some model
    // families actually share the target's KV cells.
    virtual bool get_has_shared_cells() const { return false; }

    // whether arbitrary token ranges can be removed without discarding the whole sequence
    // default to the conservative answer for out-of-tree memory implementations
    virtual bool can_seq_rm_partial() const { return false; }

    // effective bits/value of the attention KV storage, aggregated over all KV tensors at
    // their CURRENT types (dynamic VBR tier flips move this at runtime; f16 = 16, q8_0 = 8.5,
    // turbo tiers struct-true). -1 when the memory holds no attention KV (recurrent-only).
    virtual double kv_bpv() const { return -1.0; }

    virtual llama_memory_vbr_representation_identity
    vbr_representation_identity() const {
        return {};
    }

    // dynamic-VBR pressure/quality state (llama.h: llama_memory_vbr_state_data_v2). Default =
    // all zeros: "no controller, no pressure, nothing resident" — safe for policy consumers.
    virtual llama_memory_vbr_state_data_v2 memory_vbr_state_v2(
            llama_seq_id /*seq_id*/, uint32_t /*n_tokens_extra*/) const {
        return {};
    }

    // Add the physical cells owned by exactly one sequence to caller-owned counters indexed by
    // sequence ID. This is the allocation-free pressure-policy view; unlike memory_vbr_state_v2,
    // it does not simulate the representation controller. A composite with independently
    // pressured children must preserve those domains or return false; summing them would rank
    // cells that cannot satisfy the active child's deficit as useful reclaim.
    virtual bool vbr_accumulate_exclusive_cells(
            uint32_t * /*counts*/, size_t /*size*/) const {
        return false;
    }

    // Physical-cell readiness for background capture. `logical_growth` is
    // the scheduler's already-admitted token growth; implementations project
    // it through their actual storage topology (dense, bounded SWA, hybrid,
    // or recurrent) rather than making the scheduler infer cells from prompt
    // length.
    virtual bool vbr_capture_readiness_cells(
            uint64_t /*logical_growth*/,
            uint64_t & committed,
            uint64_t & projected,
            uint64_t & capacity) const {
        committed = projected = capacity = 0;
        return false;
    }

    // Scoped dynamic-VBR representation freeze. The public top-level wrapper mints the
    // process-global operation ID once; composites only forward it. Non-VBR memories stay inert.
    virtual bool vbr_operation_armed() const { return false; }
    virtual bool vbr_retier_freeze_begin(
            const char * /*owner*/, vbr_operation_id /*operation_id*/) { return false; }
    virtual void vbr_retier_freeze_end(
            const char * /*owner*/, vbr_operation_id /*operation_id*/) {}
    // Promote deferred (submitted) extent entries at the context's existing synchronize
    // boundary. Inert for non-VBR memories; composites forward to their attention child.
    virtual void vbr_commit_submitted() {}
    // Resolve in-flight decode operations once the decode outcome is known.
    virtual void vbr_decode_ops_finish(bool /*ok*/) {}
    // Composite wrappers mint one operation per logical mutation and adopt
    // it into every armed child so children never mint divergent ids.
    virtual void vbr_adopt_operation(vbr_operation_id /*operation_id*/) {}
    virtual void vbr_release_adopted() {}
    virtual llama_memory_vbr_preflight_data vbr_retier_preflight(
            uint32_t /*n_tokens_extra*/,
            std::vector<llama_memory_vbr_physical_growth> * physical = nullptr) const {
        if (physical) {
            physical->clear();
        }
        llama_memory_vbr_preflight_data r = {};
        r.fits = true;
        return r;
    }

    // per-token KV bits of the layout the --vbr-floor clamp lands on when walking the degrade
    // order from the given entry types (GGML_TYPE_COUNT = each tensor's current type;
    // floor_bpv <= 0 = bottom-tier default). 0 = no VBR-capable cache. The fit pass calls this
    // on its dry-load context for floor-true capacity math (llama_vbr_floor_bits_per_token).
    virtual double memory_vbr_floor_bits_per_token(ggml_type /*entry_k*/, ggml_type /*entry_v*/, double /*floor_bpv*/) {
        return 0.0;
    }

    // Per-token bytes of the flash-attention f16 dequant scratch at the settled deep-fill tier
    // state, summed over KV-hosting devices — a context-linear consumer that lives OUTSIDE the
    // KV budget (it draws from the fit margin). The fit charges it in the total-VRAM wall
    // constraint only. 0 = no turbo/VBR-capable cache.
    virtual double memory_vbr_scratch_bytes_per_token(ggml_type /*entry_k*/, ggml_type /*entry_v*/, double /*floor_bpv*/) {
        return 0.0;
    }

    // deferred maintenance at a quiet moment (llama.h: llama_memory_breathe)
    virtual void breathe() {}

    // co-tenancy telemetry accumulator (llama-ext.h: llama_vram_cotenancy): unamortized
    // grant decrement bytes, live grant rows, current donation offer, unflushed pending
    virtual void vbr_cotenancy_accum(uint64_t & /*decrement*/, uint32_t & /*grants*/,
                                     uint64_t & /*offer*/, uint64_t & /*pending*/) const {}

    // True for the single root that owns a live dynamic-VBR ledger controller.
    // Composite memories forward this once for the whole tree.
    virtual bool vbr_ledger_tree_active() const { return false; }

    // Hard-seal enforcement. Non-VBR memories remain inert. The callback
    // is scheduler-owned and read-only; implementations invoke it before any
    // transcode, mapping, cursor publication, or backend mutation.
    virtual void vbr_hard_seal_guard_set(vbr_hard_seal_guard /*guard*/) {}
    virtual bool vbr_hard_seal_blocked_take(bool /*decode_failed*/) { return false; }
    virtual void vbr_hard_seal_evidence_take(
            std::vector<vbr_hard_seal_subject> & /*out*/) {}

    // Drop any non-owning registrations that refer to compute backends owned by the
    // enclosing llama_context. llama_context calls this before those backends are
    // destroyed; memory implementations without shared-KV consumers have nothing to do.
    virtual void vbr_shared_scratch_detach() {}

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual bool seq_rm_attn(llama_seq_id seq_id,                            llama_pos p0, llama_pos p1) {
        return seq_rm(seq_id, p0, p1);
    }
    // Internal speculative bookkeeping. These mutate real memory and retain normal
    // generation/ownership accounting, but attention implementations may suppress global
    // checkpoint-lineage publication for a proven disposable backup or rejected suffix.
    virtual bool seq_rm_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        return seq_rm(seq_id, p0, p1);
    }
    virtual bool seq_rm_attn_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        return seq_rm_attn(seq_id, p0, p1);
    }
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    // Internal detectable copy path. Memory implementations whose seq_cp cannot fail use
    // this fallback; composite/recurrent memories override it to report transactional failure.
    virtual bool try_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
        seq_cp(seq_id_src, seq_id_dst, p0, p1);
        return true;
    }
    virtual bool try_seq_cp_transient(
            llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
        return try_seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    // The subset of memory_breakdown() whose representation and physical footprint are selected
    // by the attention Turbo/VBR policy. This is deliberately narrower than all context-linear
    // memory: auxiliary index caches can grow with n_ctx while remaining fixed-layout and outside
    // the controller's budget. Empty for memories with no Turbo/VBR-managed attention storage.
    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown_vbr_managed() const { return {}; }

    // the subset of memory_breakdown() that does NOT scale with the context length (e.g. the
    // recurrent-state cache, sized by n_seq_max). Reported so budget math can charge it even
    // where the context-linear part cancels out of a projection. Empty for pure KV caches.
    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown_fixed() const { return {}; }

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;

    // DFlash: force per-seq ubatch splits so each ubatch carries exactly one slot's tokens.
    // Default no-op; hybrid memories override.
    virtual void set_force_split_seq(bool /*v*/) {}
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
