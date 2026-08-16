#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-cache-authority.h"
#include "server-cache-destruction-quote.h"
#include "server-cache-plan-authority.h"
#include "server-cache-plan-preflight-internal.h"
#include "server-cache-yield.h"
#include "server-vbr-artifact-store.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-recurrent-expansion.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "common-cache-plan.h"
#include "common-cache-plan-estimate.h"
#include "mtp-vocab-trim.h"
#include "fit.h"
#include "llama.h"
#include "../../src/llama-ext.h" // llama_vram_mark_serviced (fork ext API; fit.cpp precedent)
#include "../../src/llama-context.h"
#include "../../src/llama-model.h" // hparams.turbo_meansub_id for the capture identity digest
#include "../../src/llama-sha256.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "fit.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <filesystem>
#include <string>
#include <type_traits>
#include <thread>
#include <unordered_map>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

// The dynamic VBR runtime controller flips KV tensor types in place as the context fills; state
// save/restore, context checkpoints and cache reuse all assume a fixed cache layout and would
// restore/reuse bytes under the wrong tier. Gate them off whenever the controller can arm.
// (This replaces the old VBR_STAGE2A env gate — Stage2A itself is gone.)
static bool server_vbr_dynamic_active(const common_params & params) {
    return params.vbr_dynamic();
}

static std::string server_cache_capture_tenant_key(
        const server_http_req & req) {
    std::string credential =
        req.get_header_value("Authorization");
    if (credential.empty()) {
        credential = req.get_header_value("X-Api-Key");
    }
    if (credential.empty()) {
        credential = "authenticated-local";
    }
    llama_sha256_writer writer;
    static constexpr char DOMAIN[] =
        "buun.vbr.server-tenant/v1";
    writer.string(DOMAIN, sizeof(DOMAIN) - 1);
    writer.string(credential.data(), credential.size());
    const auto digest = writer.finish();
    static constexpr char HEX[] = "0123456789abcdef";
    std::string output;
    output.reserve(32);
    for (size_t i = 0; i < 16; ++i) {
        output.push_back(HEX[digest[i] >> 4]);
        output.push_back(HEX[digest[i] & 0x0f]);
    }
    return output;
}

// ONE consistency derivation for both the import JSON result and the terminal
// log line — a failed import is 'unavailable' on both surfaces, never a
// fabricated live_rebased.
static server_cache_import_consistency server_cache_import_route_consistency(
        server_vbr_artifact_import_status status,
        vbr_artifact_consistency_kind kind) {
    if (status != server_vbr_artifact_import_status::ok) {
        return server_cache_import_consistency::unavailable;
    }
    return kind == vbr_artifact_consistency_kind::capture_exact
        ? server_cache_import_consistency::capture_exact
        : server_cache_import_consistency::live_rebased;
}

static std::array<uint8_t, 32>
server_cache_capture_build_digest(const char * domain) {
    llama_sha256_writer writer;
    writer.string(domain, strlen(domain));
    const char * commit = llama_commit();
    writer.string(commit, strlen(commit));
    return writer.finish();
}

// A flip is deliberately per-boot: persisting migration confidence across a
// binary/model/config change would turn yesterday's evidence into authority for
// different code. Zero keeps the frontier path in permanent shadow mode.
static uint64_t server_frontier_ratchet_min_agreements() {
    static constexpr uint64_t DEFAULT_MIN_AGREEMENTS = 1024;

    const char * env = std::getenv("LLAMA_FRONTIER_RATCHET_MIN_AGREEMENTS");
    if (env == nullptr || env[0] == '\0') {
        return DEFAULT_MIN_AGREEMENTS;
    }

    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0') {
        SRV_WRN("invalid LLAMA_FRONTIER_RATCHET_MIN_AGREEMENTS='%s'; using %" PRIu64 "\n",
                env, DEFAULT_MIN_AGREEMENTS);
        return DEFAULT_MIN_AGREEMENTS;
    }

    return (uint64_t) parsed;
}

// WS-6: keep the C-style memory API balanced across every server early return. A scope only
// becomes active for a dynamic-VBR memory; non-VBR checkpoints preserve their old path exactly.
class server_vbr_retier_freeze_scope {
public:
    server_vbr_retier_freeze_scope(llama_memory_t mem, const char * owner) :
        mem(mem), owner(owner),
        operation_id(llama_memory_vbr_retier_freeze_begin(mem, owner)) {
    }

    ~server_vbr_retier_freeze_scope() {
        if (operation_id != 0) {
            llama_memory_vbr_retier_freeze_end(mem, owner, operation_id);
        }
    }

    server_vbr_retier_freeze_scope(const server_vbr_retier_freeze_scope &) = delete;
    server_vbr_retier_freeze_scope & operator=(const server_vbr_retier_freeze_scope &) = delete;

    bool active() const {
        return operation_id != 0;
    }

private:
    llama_memory_t mem;
    const char * owner;
    uint64_t operation_id;
};

// defined near get_model_info(); shared by the /props and /models responses
static json server_vbr_meta_json(const server_context_meta * meta);

static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch  = params.n_batch;

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return n_batch;
    }

    // the fork DFlash impl reports need_embd() on generating slots, so every prompt
    // chunk on a DFlash slot decodes with output-all embeddings. The cap must cover a
    // full batch or output_reserve aborts on the first prompt chunk longer than the
    // per-seq floor (~64 tokens) — e.g. any follow-up turn that re-processes context.
    // DFlash may only be detected from the draft GGUF after the target context is
    // created (same ordering problem as the verify floor below), so gate on has_dft()
    // too. output_reserve grows its buffers lazily, so the higher cap costs nothing
    // until a batch actually requests that many outputs (the pre-cap reservation).
    if (params.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) ||
        params.speculative.has_dft()) {
        return n_batch;
    }

    const uint32_t n_outputs_per_seq = 1 + common_speculative_n_max(&params.speculative);

    const uint64_t n_outputs = (uint64_t) params.n_parallel * n_outputs_per_seq;

    return std::max<uint32_t>(1, std::min<uint64_t>(n_batch, n_outputs));
}

struct server_shared_draft_device_config {
    bool prepared = false;
    size_t n_weight_devices = 0;
    std::vector<ggml_backend_dev_t> devices;
    std::vector<float> tensor_split;
};

struct server_resolved_draft_params {
    common_params params;
    bool cpu_dspark_backbone = false;
};

static bool server_has_cpu_dspark_backbone(const common_params & params) {
    return params.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) &&
           params.speculative.draft.n_gpu_layers == 0;
}

static void server_append_tensor_override(
        common_params & params, llama_model_tensor_buft_override tensor_override) {
    if (!params.tensor_buft_overrides.empty() &&
        params.tensor_buft_overrides.back().pattern == nullptr) {
        params.tensor_buft_overrides.pop_back();
    }
    params.tensor_buft_overrides.push_back(tensor_override);
    params.tensor_buft_overrides.push_back({ nullptr, nullptr });
}

static std::vector<ggml_backend_dev_t> server_configured_devices(const common_params & params) {
    std::vector<ggml_backend_dev_t> result;
    if (!params.devices.empty()) {
        for (ggml_backend_dev_t device : params.devices) {
            if (device == nullptr) {
                break;
            }
            result.push_back(device);
        }
        return result;
    }

    std::vector<ggml_backend_dev_t> rpc_devices;
    std::vector<ggml_backend_dev_t> gpu_devices;
    ggml_backend_dev_t igpu_device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            if (igpu_device == nullptr) {
                igpu_device = device;
            }
            continue;
        }
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (reg && std::string(ggml_backend_reg_name(reg)) == "RPC") {
            rpc_devices.push_back(device);
            continue;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(device, &props);
        const bool duplicate = std::any_of(gpu_devices.begin(), gpu_devices.end(), [&](ggml_backend_dev_t existing) {
            ggml_backend_dev_props existing_props;
            ggml_backend_dev_get_props(existing, &existing_props);
            return props.device_id && existing_props.device_id &&
                   std::string(props.device_id) == existing_props.device_id;
        });
        if (!duplicate) {
            gpu_devices.push_back(device);
        }
    }

    result.insert(result.end(), rpc_devices.begin(), rpc_devices.end());
    result.insert(result.end(), gpu_devices.begin(), gpu_devices.end());
    if (gpu_devices.empty() && igpu_device != nullptr) {
        result.push_back(igpu_device);
    }
    return result;
}

static std::vector<ggml_backend_dev_t> server_target_fit_devices(const common_params & params) {
    std::vector<ggml_backend_dev_t> devices = server_configured_devices(params);
    if (params.split_mode != LLAMA_SPLIT_MODE_NONE) {
        return devices;
    }
    if (params.main_gpu < 0 || (size_t) params.main_gpu >= devices.size()) {
        return {};
    }
    return { devices[params.main_gpu] };
}

static server_shared_draft_device_config server_prepare_shared_draft_devices(const common_params & params) {
    server_shared_draft_device_config result;
    const auto & types = params.speculative.types;
    const bool has_shared_draft =
        std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != types.end() ||
        std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != types.end();
    if (!has_shared_draft) {
        return result;
    }

    const std::vector<ggml_backend_dev_t> target_devices = server_configured_devices(params);

    const auto & draft_devices = params.speculative.draft.devices;
    const bool automatic = draft_devices.empty();
    const bool cpu_only = !automatic && draft_devices.front() == nullptr;

    std::vector<ggml_backend_dev_t> weight_devices;
    if (!automatic && !cpu_only) {
        for (ggml_backend_dev_t device : draft_devices) {
            if (device == nullptr) {
                break;
            }
            if (std::find(weight_devices.begin(), weight_devices.end(), device) == weight_devices.end()) {
                weight_devices.push_back(device);
            }
        }
    }

    if (automatic) {
        if (target_devices.empty()) {
            return result;
        }

        ggml_backend_dev_t target_primary = nullptr;
        if (params.main_gpu >= 0) {
            if (!params.devices.empty() && (size_t) params.main_gpu < params.devices.size()) {
                target_primary = params.devices[params.main_gpu];
            } else if (params.devices.empty() && (size_t) params.main_gpu < target_devices.size()) {
                target_primary = target_devices[params.main_gpu];
            }
        }

        ggml_backend_dev_t draft_primary = nullptr;
        size_t draft_free = 0;
        for (ggml_backend_dev_t device : target_devices) {
            if (target_devices.size() > 1 && device == target_primary) {
                continue;
            }
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(device, &free, &total);
            if (draft_primary == nullptr || free > draft_free) {
                draft_primary = device;
                draft_free = free;
            }
        }

        GGML_ASSERT(draft_primary != nullptr);
        weight_devices.push_back(draft_primary);
        SRV_INF("[spec] auto-selected %s as the primary draft device\n", ggml_backend_dev_name(draft_primary));
    }

    result.devices = weight_devices;
    size_t n_added = 0;
    for (ggml_backend_dev_t device : target_devices) {
        if (std::find(result.devices.begin(), result.devices.end(), device) == result.devices.end()) {
            result.devices.push_back(device);
            n_added++;
        }
    }
    result.devices.push_back(nullptr);

    result.prepared = true;
    result.n_weight_devices = weight_devices.size();
    result.tensor_split.resize(result.n_weight_devices, 0.0f);
    if (result.n_weight_devices == 1) {
        result.tensor_split[0] = 1.0f;
    } else if (result.n_weight_devices > 1) {
        bool has_user_split = false;
        for (size_t i = 0; i < result.n_weight_devices; i++) {
            result.tensor_split[i] = params.tensor_split[i];
            has_user_split = has_user_split || result.tensor_split[i] != 0.0f;
        }
        if (!has_user_split) {
            for (size_t i = 0; i < result.n_weight_devices; i++) {
                size_t free = 0;
                size_t total = 0;
                ggml_backend_dev_memory(weight_devices[i], &free, &total);
                result.tensor_split[i] = std::max(1.0f, (float) (free / (1024 * 1024)));
            }
        }
    }

    if (cpu_only && n_added > 0) {
        SRV_INF("[spec] added %zu target device(s) to the CPU draft scheduler for shared tensors\n", n_added);
    } else if (!automatic && n_added > 0) {
        SRV_INF("[spec] added %zu target device(s) to the draft scheduler for shared tensors\n", n_added);
    }
    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

// outcome of an attempt to save a slot's state to the host prompt cache [I7]. A live slot may be
// cleared without losing state only when its state is now durable in the cache (published OR already
// cached under the same identity); a `failed` save must NOT be followed by a clear.
enum class prompt_save_result {
    published,       // a new entry was saved to the cache
    already_durable, // the state was already fully cached under this adapter identity
    failed,          // could not be cached (empty prompt, zero/short state, over-limit, OOM, short write)
};

static inline bool prompt_save_durable(prompt_save_result r) {
    return r == prompt_save_result::published || r == prompt_save_result::already_durable;
}

// B0 shadow cache-plan observer state [P2 §7.7]. Debug-only record/serialization layer over the
// authority substrate: records are allocated, populated, and serialized only under
// params_base.cache_debug. Observer faults are caught outside the shipped decision path and become
// shadow_unavailable — never a changed live choice. Both counters are surfaced on every record.
struct server_cache_plan_observer {
    uint64_t records_finalized  = 0;
    uint64_t shadow_unavailable = 0;
    // planner-attempt accounting (verify-r1 finding 8): refusals counted exactly once,
    // distinct from observer faults; the per-record closed status carries the WHY
    uint64_t planner_ok      = 0;
    uint64_t planner_refused = 0;

    // B-2 stable calibration-profile id ("{model class}/{hardware class}/b{batch}"),
    // composed once at observer init; copied onto every record. Empty = unprofiled.
    std::string calibration_profile;
};

// D-S4 keeps retention policy at the server's logical-operation boundary. These functions
// are the closed physical seams allowed to invoke the low-level destructive primitives.
// Callers must first admit their logical manifest, or be an explicitly transient operation.
static bool server_cache_live_range_drop_impl(
        llama_memory_t mem,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        bool attention_only = false) {
    return attention_only
        ? llama_memory_seq_rm_attn(mem, seq_id, p0, p1)
        : llama_memory_seq_rm(mem, seq_id, p0, p1);
}

static void server_cache_live_range_drop_impl(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    common_context_seq_rm(ctx, seq_id, p0, p1);
}

static bool server_cache_mandatory_recovery_reset_impl(
        llama_memory_t mem,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    return llama_memory_seq_rm(mem, seq_id, p0, p1);
}

static void server_cache_mandatory_recovery_reset_impl(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    common_context_seq_rm(ctx, seq_id, p0, p1);
}

static bool server_cache_transient_seq_rm_impl(
        llama_memory_t mem,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    return llama_memory_seq_rm_transient(mem, seq_id, p0, p1);
}

static server_cache_control_status server_cache_family_resolve_for_launch(
        server_cache_control_authority * authority,
        server_cache_control_token token,
        common_cache_family_binding & out) noexcept {
    out = {};
    if (!token) {
        return server_cache_control_status::ok;
    }
    return authority
        ? authority->resolve_family_binding(token, out)
        : server_cache_control_status::not_found;
}

struct server_slot {
    int id;

    // Optional E1 declared-family state, resolved by the scheduler at launch.
    common_cache_family_binding cache_family;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;
    server_cache_destruction_observer * destruction_obs = nullptr;
    server_retention_sidecar_store * retention_obs = nullptr;
    server_cache_lease_table * lease_obs = nullptr;
    // One scheduler-owned context scope follows this slot across immutable
    // append replacements. Reusing it lets grant_soft renew the migrated
    // implicit lease instead of accumulating one lease per completion turn.
    server_cache_context_scope_id implicit_soft_lease_scope;
    const std::string * lease_execution_identity = nullptr;
    server_cache_authority * lifecycle_authority = nullptr;
    bool cache_debug_observability = false;
    common_retention_pool retention_pool = common_retention_pool::attention;
    server_cache_checkpoint_attempt_latch checkpoint_attempts;
    const common_prompt_checkpoint * checkpoint_seam_heuristic = nullptr;
    common_cache_plan_destruction_reason checkpoint_floor_refusal =
        common_cache_plan_destruction_reason::mandatory_anchor;
    common_cache_plan_destruction_reason checkpoint_thinning_refusal =
        common_cache_plan_destruction_reason::none;

    // B0 shadow cache-plan record [P2]: in-flight (allocated per request only when the
    // observer is enabled — absence IS the disabled state) + this slot's last finalized
    // record, serialized once at finalize and reused verbatim by /slots
    std::unique_ptr<common_cache_plan_record> cache_plan;
    json cache_plan_json;
    server_cache_plan_execution cache_plan_execution;
    // D-A5 recovery source remains pinned through the dependent B execution.
    // reset() closes it after the request, allowing later priced retention.
    server_cache_recovery_pin cache_plan_destruction_recovery_pin;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // diffusion self-speculation
    bool                 diff_self_spec     = false;
    int32_t              diff_draft_length  = 4;
    llama_token          diff_mask_token_id = LLAMA_TOKEN_NULL;
    llama_token          diff_think_open_id = LLAMA_TOKEN_NULL;
    llama_token          diff_think_close_id = LLAMA_TOKEN_NULL;
    std::vector<float>   diff_prev_logits;
    std::vector<llama_token> diff_prev_assistant_tokens;

    // speculative decoding
    common_speculative_ptr spec;
    common_speculative * spec_shared = nullptr; // non-owning

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    // Read-only E1 range qualification shared by the VBR controller and the
    // three legacy reclaim skip guards. A missing sidecar/identity cannot
    // correspond to a granted hard lease, so the legacy behavior remains the
    // neutral result. The lease table remains the one evaluator.
    bool hard_lease_blocks_live_range(
            uint64_t first_token, uint64_t token_count) const {
        if (!lifecycle_authority || !lease_obs || !retention_obs ||
            !lease_execution_identity ||
            prompt.n_tokens() <= 0 || token_count == 0) {
            return false;
        }
        const auto artifact = retention_obs->artifact_id(
            server_retention_instance_key::for_slot(id));
        server_cache_lease_identity identity;
        if (artifact.v == 0 || !server_cache_lease_build_identity(
                *lease_execution_identity, lora_config_identity(lora),
                prompt.tokens, prompt.n_tokens(), identity)) {
            return false;
        }
        return server_cache_hard_lease_blocks_range(
            lease_obs, artifact, identity, prompt.sequence_epoch,
            first_token, token_count);
    }

    bool hard_lease_blocks_live_prefix() const {
        return hard_lease_blocks_live_range(
            0, static_cast<uint64_t>(std::max(0, prompt.n_tokens())));
    }

    prompt_save_result prompt_save(
            server_prompt_cache & prompt_cache,
            bool refresh_exact = false,
            server_prompt_cache::iterator * published = nullptr) const {
        if (published) {
            *published = prompt_cache.states.end();
        }
        if (prompt.tokens.size() == 0) {
            return prompt_save_result::failed;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        // A zero-size declared state is never a valid snapshot [I10]: an empty target would publish
        // an entry that "restores" as a 0 == 0 false success, and an empty draft alongside a valid
        // target would restore target-only, leaving the two sides on different states. Require a
        // non-empty target, and a non-empty draft whenever a draft context exists.
        if (cur_size_tgt == 0 || (ctx_dft && cur_size_dft == 0)) {
            SLT_WRN(*this, "prompt cache save skipped: zero-size state (target %zu, draft %zu)\n", cur_size_tgt, cur_size_dft);
            return prompt_save_result::failed;
        }

        // Everything below allocates (identity key, staged node, token/checkpoint clones). Any OOM
        // must degrade the save to `failed` (the caller keeps the live slot) rather than escape and
        // abort the server. stage() is already allocation-neutral internally; this catch also covers
        // the identity-string construction and the by-value key hand-off.
        try {
            const std::string adapter_key = lora_config_identity(lora);

            // Ordinary durability accepts an existing identical token key.
            // D-A5 requests a fresh exact three-payload image so recovery of
            // the displaced live state cannot cite an older checkpoint or
            // accelerator payload that happens to share those tokens.
            if (!refresh_exact &&
                prompt_cache.contains(prompt.tokens, adapter_key)) {
                return prompt_save_result::already_durable;
            }

            const size_t cur_size = cur_size_tgt + cur_size_dft;
            SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                    (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

            // stage -> fill -> validate -> publish [I7]. Fill the staged node and verify the writer
            // returned the exact declared length BEFORE publishing; a short write (e.g. a dynamic VBR
            // state_write refusal after a degrade) aborts the save without touching the cache,
            // instead of publishing a truncated entry [I10]. On any failure the state is NOT durable.
            auto staged = prompt_cache.stage(prompt, cur_size_tgt, cur_size_dft, adapter_key);
            if (staged.empty()) {
                return prompt_save_result::failed;
            }
            auto & entry = staged.front();
            // An idle save has no active task and therefore keeps the
            // historical automatic-main default. Checkpoint pricing below is
            // intentionally the opposite polarity: no request means no
            // provisional automatic-main claim. A declaration overrides both.
            server_prompt_cache_apply_family(
                entry, cache_family, !task || !task->is_child());

            size_t n_tgt = llama_state_seq_get_data_ext(ctx_tgt, entry.data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (server_fault("save_short")) { n_tgt = cur_size_tgt > 0 ? cur_size_tgt - 1 : 0; } // [P0 gate]
            if (n_tgt != cur_size_tgt) {
                SLT_WRN(*this, "prompt cache save aborted: target state write %zu != %zu bytes\n", n_tgt, cur_size_tgt);
                return prompt_save_result::failed;
            }

            if (ctx_dft) {
                const size_t n_dft = llama_state_seq_get_data_ext(ctx_dft, entry.data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
                if (n_dft != cur_size_dft) {
                    SLT_WRN(*this, "prompt cache save aborted: draft state write %zu != %zu bytes\n", n_dft, cur_size_dft);
                    return prompt_save_result::failed;
                }
            }

            if (!prompt_cache.publish(
                    std::move(staged), &prompt, id, published)) {
                SLT_WRN(*this, "%s",
                        "prompt cache save refused at lifecycle admission boundary\n");
                return prompt_save_result::failed;
            }

            return prompt_save_result::published;
        } catch (const std::bad_alloc &) {
            SLT_WRN(*this, "%s", "prompt cache save aborted: out of memory\n");
            return prompt_save_result::failed;
        }
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens, const std::string & adapter_config_key, common_cache_plan_record * obs = nullptr, int32_t required_source_id = -1) {
        // No-restore is a successful identity operation. Seed the out-value
        // with the live lineage; a committed host restore overwrites it with
        // delivery.cache_family inside load_impl.
        common_cache_family_binding restored_family = cache_family;
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id,
                                     adapter_config_key, obs, required_source_id,
                                     &restored_family);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        } else {
            // A host image replaces the live frontier wholesale, including
            // its immutable family provenance. A declared current request may
            // deliberately override this later at the launch boundary.
            cache_family = restored_family;
            if (lifecycle_authority) {
                checkpoint_ring_changed();
            }
        }

        return res;
    }

    using checkpoint_iterator = std::list<common_prompt_checkpoint>::iterator;

    static checkpoint_iterator checkpoint_drop_authority_adapter(
            void * owner,
            checkpoint_iterator first,
            checkpoint_iterator last) {
        return static_cast<server_slot *>(owner)->
            server_cache_checkpoint_drop_impl(first, last);
    }

    server_cache_checkpoint_authority_context checkpoint_authority_context() noexcept {
        return {
            id,
            prompt.checkpoints,
            lifecycle_authority,
            retention_obs,
            destruction_obs,
            lease_obs,
            checkpoint_attempts,
            checkpoint_seam_heuristic,
            checkpoint_thinning_refusal,
            checkpoint_floor_refusal,
            // Unlike idle prompt_save(), an absent request does not
            // provisionally classify a checkpoint as automatic-main.
            common_cache_family_main_family(
                cache_family, task && !task->is_child()),
            cache_family,
            cache_debug_observability,
            this,
            checkpoint_drop_authority_adapter,
        };
    }

    void checkpoint_ring_changed() noexcept {
        auto context = checkpoint_authority_context();
        server_cache_checkpoint_ring_changed(context);
    }

    bool checkpoint_thinning_attempt_begin(bool capacity_mode) noexcept {
        auto context = checkpoint_authority_context();
        return server_cache_checkpoint_thinning_attempt_begin(
            context, capacity_mode);
    }

    bool checkpoint_refusal_state_changed(
            common_cache_plan_destruction_reason reason,
            bool publication_skip = false) noexcept {
        auto context = checkpoint_authority_context();
        return server_cache_checkpoint_refusal_state_changed(
            context, reason, publication_skip);
    }

    void server_cache_slot_drop_impl(bool retire_retention = true) {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());
        if (retire_retention && retention_obs) {
            retention_obs->retire_slot(id);
        }

        mem.seq_rm(id, -1, -1);

        if (lifecycle_authority && !prompt.checkpoints.empty()) {
            checkpoint_ring_changed();
        }
        prompt.clear();
        cache_family = {};
    }

    void prompt_clear_certified(
            server_cache_prepared_release_capability & capability,
            common_cache_plan_destruction_quote & quote,
            common_cache_plan_record & rec) {
        GGML_ASSERT(lifecycle_authority && retention_obs);
        GGML_ASSERT(capability.ready());
        const auto admission = observe_full_slot(
            server_cache_destruction_class::slot_drop,
            server_cache_destruction_reason::slot_rebind);

        // Scheduler ownership is exclusive from prepare through this raw
        // sequence mutation and the conditional release terminal. The gap
        // contains no callback and no ledger writer: seq removal, prompt
        // clearing, and the ring-generation latch are physical/local only.
        const auto scheduler_owner = std::this_thread::get_id();
        server_cache_slot_drop_impl(false);
        GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
        const auto released = capability.commit(
            cache_plan_destruction_recovery_pin);
        GGML_ASSERT(released ==
                    common_cache_plan_destruction_reason::none);
        GGML_ASSERT(retention_obs->retire_slot_after_committed_release(
            id,
            quote.receipt.selected_attention,
            quote.receipt.selected_recurrent));

        quote.receipt.state =
            common_cache_plan_destruction_state::executed;
        quote.receipt.actual_accounting_serial =
            lifecycle_authority->ledger.snapshot().serial;
        rec.destruction = quote.receipt;
        std::vector<common_cache_plan_yield_domain> observed_after;
        if (lifecycle_authority->observe_release_domains(
                quote.projected_domains, observed_after)) {
            common_cache_plan_fill_actual_yield(
                rec.yield, quote.projected_domains, observed_after);
        } else {
            rec.yield.actual_domains.clear();
            rec.yield.actual_state =
                common_cache_plan_yield_actual_state::unavailable;
        }
        lifecycle_authority->destruction_counters.observe(
            rec.selection, rec.destruction, false);
        if (destruction_obs) {
            destruction_obs->note_live_displacement_executed(
                admission.sequence);
        }
        if (cache_debug_observability) {
            try {
                uint64_t projected = 0;
                (void) common_cache_plan_projected_release_bytes(
                    quote.projected_domains, projected);
                auto payload = server_cache_destruction_receipt_json(
                    rec.destruction, projected, "live_slot_displacement");
                if (rec.destruction.plan_candidate >= 0 &&
                    uint32_t(rec.destruction.plan_candidate) <
                        rec.n_inventory) {
                    const auto & eviction = rec.inventory[size_t(
                        rec.destruction.plan_candidate)].cost_terms[size_t(
                            llama_cache_acct_cost_kind::eviction)];
                    payload["eviction_bytes"] =
                        common_cache_plan_value_json(eviction.raw);
                    payload["eviction_us"] =
                        common_cache_plan_value_json(eviction.estimated_us);
                }
                SRV_INF("CACHE_HOST_DESTRUCTION %s\n",
                        payload.dump().c_str());
            } catch (...) {
            }
        }
    }

    server_cache_destruction_admission observe_full_slot(
            server_cache_destruction_class cls,
            server_cache_destruction_reason reason) const {
        GGML_ASSERT(
            cls == server_cache_destruction_class::slot_drop ||
            cls == server_cache_destruction_class::mandatory_recovery_reset);
        server_cache_destruction_admission admission;
        admission.cls    = cls;
        admission.reason = reason;
        admission.issued = true;
        if (!destruction_obs) {
            return admission;
        }
        server_cache_destruction_request request;
        request.cls    = cls;
        request.reason = reason;
        for (const auto kind : {
                server_cache_destruction_target_kind::live_target,
                server_cache_destruction_target_kind::token_ledger,
                server_cache_destruction_target_kind::checkpoint_ring,
                server_cache_destruction_target_kind::rolling_window,
                server_cache_destruction_target_kind::typed_accelerator }) {
            if (kind == server_cache_destruction_target_kind::live_target &&
                retention_obs) {
                request.add_target(
                    kind,
                    id,
                    retention_obs->artifact_id(
                        server_retention_instance_key::for_slot(id)));
            } else {
                request.add_target(kind, id);
            }
        }
        if (ctx_dft) {
            request.add_target(
                server_cache_destruction_target_kind::live_draft, id);
        }
        for (const auto category : {
                llama_cache_acct_category::live_attention_state,
                llama_cache_acct_category::live_recurrent_state,
                llama_cache_acct_category::checkpoint_state_payload,
                llama_cache_acct_category::typed_accelerator_payload,
                llama_cache_acct_category::rolling_window_tape }) {
            request.add_yield(category);
        }
        return server_cache_retention_admit(destruction_obs, request);
    }

    void prompt_clear(
            server_cache_destruction_reason reason =
                server_cache_destruction_reason::slot_rebind) {
        (void) observe_full_slot(
            server_cache_destruction_class::slot_drop, reason);
        server_cache_slot_drop_impl();
    }

    server_cache_destruction_admission observe_live_range_drop(
            server_cache_destruction_reason reason,
            bool include_checkpoints = false) const {
        server_cache_destruction_admission admission;
        admission.cls    = server_cache_destruction_class::live_range_drop;
        admission.reason = reason;
        admission.issued = true;
        if (!destruction_obs) {
            return admission;
        }
        server_cache_destruction_request request;
        request.cls    = server_cache_destruction_class::live_range_drop;
        request.reason = reason;
        request.add_target(
            server_cache_destruction_target_kind::live_target,
            id,
            retention_obs ? retention_obs->artifact_id(
                server_retention_instance_key::for_slot(id))
                : llama_cache_acct_artifact_id{});
        if (ctx_dft) {
            request.add_target(
                server_cache_destruction_target_kind::live_draft, id);
        }
        request.add_target(
            server_cache_destruction_target_kind::token_ledger, id);
        if (include_checkpoints) {
            request.add_target(
                server_cache_destruction_target_kind::checkpoint_ring, id);
        }
        for (const auto category : {
                llama_cache_acct_category::live_attention_state,
                llama_cache_acct_category::live_recurrent_state }) {
            request.add_yield(category);
        }
        return server_cache_retention_admit(destruction_obs, request);
    }

    server_cache_destruction_admission observe_mandatory_recovery_reset(
            server_cache_destruction_reason reason) const {
        return observe_full_slot(
            server_cache_destruction_class::mandatory_recovery_reset,
            reason);
    }

    // Slot-file restore has already installed target bytes when this cleanup runs. It clears
    // only the displaced prompt metadata and draft state; the target must remain intact.
    void server_cache_mandatory_recovery_reset_impl(bool clear_draft) {
        if (retention_obs) {
            retention_obs->retire_slot(id);
        }
        if (lifecycle_authority && !prompt.checkpoints.empty()) {
            checkpoint_ring_changed();
        }
        prompt.clear();
        cache_family = {};
        if (clear_draft && ctx_dft) {
            ::server_cache_mandatory_recovery_reset_impl(ctx_dft, id, -1, -1);
        }
    }

    void mandatory_recovery_reset(server_cache_destruction_reason reason) {
        observe_mandatory_recovery_reset(reason);
        server_cache_slot_drop_impl();
    }

    void server_cache_live_range_drop_impl(llama_tokens && retained_tokens) {
        if (retention_obs) {
            retention_obs->retire_slot(id);
        }
        if (lifecycle_authority && !prompt.checkpoints.empty()) {
            checkpoint_ring_changed();
        }
        prompt.clear();
        prompt.tokens.insert(retained_tokens);
        if (prompt.tokens.empty()) {
            cache_family = {};
        }
    }

    checkpoint_iterator server_cache_checkpoint_drop_impl(
            checkpoint_iterator first,
            checkpoint_iterator last) {
        if (lifecycle_authority && first != last) {
            checkpoint_ring_changed();
        }
        return prompt.checkpoints.erase(first, last);
    }

    checkpoint_iterator checkpoint_drop_joined_impl(
            checkpoint_iterator first,
            checkpoint_iterator last) {
        if (retention_obs) {
            for (auto it = first; it != last; ++it) {
                retention_obs->retire(
                    server_retention_instance_key::for_checkpoint(
                        id, &*it));
            }
        }
        return server_cache_checkpoint_drop_impl(first, last);
    }

    server_cache_destruction_admission observe_checkpoint_drop(
            server_cache_destruction_reason reason,
            llama_cache_acct_artifact_id artifact = {}) {
        auto context = checkpoint_authority_context();
        return server_cache_checkpoint_observe_drop(context, reason, artifact);
    }

    checkpoint_iterator checkpoint_drop(
            checkpoint_iterator first,
            checkpoint_iterator last,
            server_cache_destruction_reason reason) {
        if (destruction_obs && first != last) {
            (void) observe_checkpoint_drop(
                reason,
                retention_obs ? retention_obs->artifact_id(
                    server_retention_instance_key::for_checkpoint(
                        id, &*first))
                    : llama_cache_acct_artifact_id{});
        }
        // Legacy/pass-through release owner. D-A4's certified path calls the
        // raw list eraser first, commits the prepared exact release, then
        // retires only the descriptor through retire_after_committed_release.
        return checkpoint_drop_joined_impl(first, last);
    }

    bool checkpoint_thin_priced(
            int checkpoint_task_id,
            uint64_t max_replay_tokens,
            const common_prompt_checkpoint * seam_heuristic,
            bool capacity_mode,
            bool attempt_claimed = false) noexcept {
        auto context = checkpoint_authority_context();
        return server_cache_checkpoint_thin_priced(
            context, checkpoint_task_id, max_replay_tokens,
            seam_heuristic, capacity_mode, attempt_claimed);
    }

    bool checkpoint_capacity_floor(
            int checkpoint_task_id,
            const common_prompt_checkpoint * seam_heuristic,
            checkpoint_iterator & victim,
            common_cache_plan_destruction_reason & refusal) noexcept {
        auto context = checkpoint_authority_context();
        return server_cache_checkpoint_capacity_floor(
            context, checkpoint_task_id, seam_heuristic, victim, refusal);
    }

    void checkpoint_publication_skipped(
            common_cache_plan_destruction_reason reason) noexcept {
        auto context = checkpoint_authority_context();
        server_cache_checkpoint_publication_skipped(context, reason);
    }

    void server_cache_token_ledger_truncate_impl(size_t n_tokens) {
        prompt.tokens.keep_first(n_tokens);
    }

    void server_cache_transient_token_truncate_impl(size_t n_tokens) {
        prompt.tokens.keep_first(n_tokens);
    }

    // No standalone caller exists yet: current physical token truncations are children of
    // live_range_drop or mandatory_recovery_reset. Keep this closed admission seam ready for
    // the first independent retained-ledger truncation rather than weakening the 1:1 inventory.
    void token_ledger_truncate(
            size_t n_tokens,
            server_cache_destruction_reason reason) {
        if (destruction_obs && n_tokens < prompt.tokens.size()) {
            server_cache_destruction_request request;
            request.cls    =
                server_cache_destruction_class::token_ledger_truncate;
            request.reason = reason;
            request.add_target(
                server_cache_destruction_target_kind::token_ledger, id);
            request.add_yield(
                llama_cache_acct_category::artifact_descriptor_metadata);
            (void) server_cache_retention_admit(destruction_obs, request);
        }
        server_cache_token_ledger_truncate_impl(n_tokens);
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    // stats
    size_t n_sent_text = 0; // number of sent text character

    // TODO @ngxson : move all metrics to a sub-struct for clarity
    int64_t t_start_process_prompt;
    int64_t t_start_generation;
    int64_t t_print_last = 0;
    int32_t n_decoded_last = 0;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted
    int32_t n_draft_verif_steps = 0; // Total draft token verification steps by the target model
    std::vector<int32_t> n_accepted_per_pos; // Accepted tokens per draft position
    // Gate-5 observation: exact joint distribution needed to weight plane-vs-tape
    // capture and rollback costs. Outer index = generated draft length; inner
    // index = rejected target rows (rollback depth).
    std::vector<std::vector<int32_t>> n_verify_rollback;

    // Hybrid model: recurrent state backup for speculative decoding
    bool has_draft_backup = false;
    llama_seq_id seq_id_backup = -1;
    int  n_tokens_before_draft = 0; // prompt token count before draft tokens were added

    // [obs] human-readable reason for how the last prompt's prefix was (or wasn't) reused, surfaced
    // in GET /slots so a user can see WHY a request cold-processed instead of guessing. Set at the
    // prompt-reuse decision points; greppable by tests (e.g. the I9 VBR-reject reason).
    std::string cache_status;

    // Phase-1 computation-frontier read ratchet [WS-4]. Only checkpoint-selection
    // decisions qualify the agreement streak; other consistency checks may
    // reset/fallback it but cannot make a vacuous workload flip authority.
    uint64_t frontier_ratchet_min_agreements = 1024;
    uint64_t frontier_ratchet_agreement_streak = 0;
    uint64_t frontier_ratchet_agreements_total = 0;
    uint64_t frontier_ratchet_disagreements_total = 0;
    uint64_t frontier_ratchet_flips_total = 0;
    uint64_t frontier_ratchet_fallbacks_total = 0;
    bool frontier_ratchet_flipped = false;

    void frontier_ratchet_disagreement(
            const char * site,
            int64_t legacy_choice,
            int64_t frontier_choice) {
        frontier_ratchet_disagreements_total++;
        frontier_ratchet_agreement_streak = 0;

        if (frontier_ratchet_flipped) {
            frontier_ratchet_flipped = false;
            frontier_ratchet_fallbacks_total++;
            SLT_ERR(*this,
                    "FRONTIER_RATCHET event=post_flip_disagreement action=fallback_legacy "
                    "site=%s legacy=%" PRId64 " frontier=%" PRId64
                    " disagreements=%" PRIu64 " fallbacks=%" PRIu64 "\n",
                    site, legacy_choice, frontier_choice,
                    frontier_ratchet_disagreements_total,
                    frontier_ratchet_fallbacks_total);
        } else {
            SLT_WRN(*this,
                    "FRONTIER_RATCHET event=shadow_disagreement action=keep_legacy "
                    "site=%s legacy=%" PRId64 " frontier=%" PRId64
                    " disagreements=%" PRIu64 "\n",
                    site, legacy_choice, frontier_choice,
                    frontier_ratchet_disagreements_total);
        }
    }

    // Returns whether the frontier path owns this selection. `eligible` means
    // at least one dual-written record was actually evaluated; empty/legacy-only
    // scans do not count toward "sustained".
    bool frontier_ratchet_selection(
            bool eligible,
            bool agreement,
            int64_t legacy_choice,
            int64_t frontier_choice) {
        if (!eligible) {
            if (frontier_ratchet_flipped) {
                frontier_ratchet_disagreement(
                    "checkpoint_selection_no_frontier_record",
                    legacy_choice, frontier_choice);
            }
            return false;
        }

        if (!agreement) {
            frontier_ratchet_disagreement(
                "checkpoint_selection", legacy_choice, frontier_choice);
            return false;
        }

        frontier_ratchet_agreements_total++;
        if (!frontier_ratchet_flipped &&
            frontier_ratchet_min_agreements > 0) {
            frontier_ratchet_agreement_streak++;
            if (frontier_ratchet_agreement_streak >=
                    frontier_ratchet_min_agreements) {
                frontier_ratchet_flipped = true;
                frontier_ratchet_flips_total++;
                SLT_INF(*this,
                        "FRONTIER_RATCHET event=flip action=frontier_reads "
                        "streak=%" PRIu64 " threshold=%" PRIu64
                        " flips=%" PRIu64 "\n",
                        frontier_ratchet_agreement_streak,
                        frontier_ratchet_min_agreements,
                        frontier_ratchet_flips_total);
            }
        }

        return frontier_ratchet_flipped;
    }

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;
        cache_plan_execution.clear();
        cache_plan_destruction_recovery_pin = {};

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
        n_draft_verif_steps = 0;
        n_accepted_per_pos.clear();
        n_verify_rollback.clear();
        has_draft_backup = false;
        seq_id_backup = -1;
        n_tokens_before_draft = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();

    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (get_spec() && common_speculative_need_embd(get_spec()));
    }

    bool need_embd_nextn() const {
        GGML_ASSERT(task);
        return get_spec() && common_speculative_need_embd_nextn(get_spec());
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return spec || spec_shared;
    }

    common_speculative * get_spec() const {
        return spec ? spec.get() : spec_shared;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        if (common_sampler_grammar_is_active(smpl.get())) {
            return 0;
        }

        const int n_draft_min = common_speculative_n_min(get_spec(), task->params.speculative);


        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            // B0: a request that ends before its record finalized (error, cancel) resolves
            // the in-flight record here, exactly once, by deliberate drop — an incomplete
            // request has no truthful outcome to emit
            cache_plan.reset();

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // clean up speculative backup sequence to avoid orphaned KV cells
            if (has_draft_backup && seq_id_backup >= 0) {
                server_cache_transient_seq_rm_impl(
                    llama_get_memory(ctx_tgt), seq_id_backup, -1, -1);
            }

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            // [P0b/C] A checkpoint beyond the request prompt is generation-phase, but generated
            // tokens normally become the next turn's prefix. Keep those checkpoints while their
            // logical frontier is still represented by prompt.tokens. Only a checkpoint beyond
            // BOTH the request prompt and the accepted/retained token ledger is unreachable by
            // any future LCP and therefore genuinely dead (for example, an abandoned speculative
            // branch). This is position-layout independent: recurrent pos_min is the full sequence
            // frontier, so the upstream `pos_min > prompt_end` test would erase every recurrent
            // generation checkpoint. Erasing metadata here cannot alter the live model state, and
            // retaining continuation checkpoints is the output-neutral choice when lineage is not
            // provably dead. Child slots were cleared above and deliberately skip this lifecycle.
            if (!task->is_child()) {
                const int64_t prompt_end   = task->n_tokens();
                const int64_t retained_end = prompt.n_tokens();
                int n_erased = 0;

                for (auto it = prompt.checkpoints.begin(); it != prompt.checkpoints.end();) {
                    const bool is_generation_checkpoint = it->n_tokens > prompt_end;
                    const bool is_outside_retained       = it->n_tokens > retained_end;

                    if (is_generation_checkpoint && is_outside_retained) {
                        SLT_DBG(*this,
                                "release: erasing dead generation checkpoint "
                                "(n_tokens=%" PRId64 ", prompt_end=%" PRId64 ", retained_end=%" PRId64 ")\n",
                                it->n_tokens, prompt_end, retained_end);
                        it = checkpoint_drop(
                            it, std::next(it),
                            server_cache_destruction_reason::checkpoint_invalidated);
                        n_erased++;
                    } else {
                        ++it;
                    }
                }

                if (n_erased > 0) {
                    SLT_INF(*this,
                            "release: erased %d dead generation checkpoint(s) "
                            "(prompt_end=%" PRId64 ", retained_end=%" PRId64 ")\n",
                            n_erased, prompt_end, retained_end);
                }
            }

            if (retention_obs) {
                if (!task->is_child() && prompt.n_tokens() > 0) {
                    const auto live_key =
                        server_retention_instance_key::for_slot(id);
                    const auto prior_artifact =
                        retention_obs->artifact_id(live_key);
                    server_cache_lease_identity identity;
                    const bool identity_known = lease_obs &&
                        lease_execution_identity &&
                        server_cache_lease_build_identity(
                            *lease_execution_identity,
                            lora_config_identity(lora), prompt.tokens,
                            prompt.n_tokens(), identity);
                    const server_cache_lease_frontier frontier {
                        prompt.sequence_epoch,
                        uint64_t(prompt.n_tokens()),
                        prompt.n_tokens(),
                    };
                    // launch_slot_with_task leaves the prior association in
                    // place only for exact append continuity. Publication
                    // still refreshes the immutable retention record, then
                    // atomically migrates matching leases to its fresh
                    // artifact after checking identity and proven-frontier
                    // containment. A trim/rebind has no prior association and
                    // therefore reaches the normal subject_lost terminal.
                    const bool published = retention_obs->publish(
                            live_key,
                            retention_pool,
                            task->params.message_spans,
                            !task->params.message_spans.spans.empty(),
                            uint64_t(prompt.n_tokens()),
                            uint64_t(prompt.n_tokens()),
                            prompt.n_tokens() >= 0,
                            identity_known ? &identity : nullptr,
                            identity_known && prior_artifact.v != 0
                                ? &frontier : nullptr);
                    if (published && lease_obs) {
                        const auto artifact = retention_obs->artifact_id(
                            live_key);
                        if (!implicit_soft_lease_scope.v) {
                            implicit_soft_lease_scope =
                                lease_obs->new_context_scope();
                        }
                        const server_cache_lease_subject subject {
                            artifact,
                            common_retention_artifact_kind::live_slot,
                            id,
                        };
                        if (identity_known &&
                            subject.valid() &&
                            implicit_soft_lease_scope.v != 0) {
                            (void) lease_obs->grant_soft(
                                subject,
                                server_cache_lease_scope::from(implicit_soft_lease_scope),
                                identity,
                                server_cache_lease_table::IMPLICIT_SOFT_TTL_NS);
                        } else if (subject.valid()) {
                            lease_obs->artifact_identity_unavailable(subject);
                        }
                    }
                } else {
                    retention_obs->retire(
                        server_retention_instance_key::for_slot(id));
                }
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        // live effective KV bits/value (moves under dynamic VBR as tiers degrade/reset)
            if (ctx_tgt != nullptr) {
            timings.kv_bpv = llama_memory_kv_bpv(llama_get_memory(ctx_tgt));
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = 1e3 / (t_token_generation)   * (n_decoded);
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (n_decoded - n_decoded_last);

        t_print_last = t_now;
        n_decoded_last = n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", n_decoded, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());

            std::string verify_rollback_hist;
            for (size_t n_draft = 0; n_draft < n_verify_rollback.size(); ++n_draft) {
                for (size_t rollback = 0;
                     rollback < n_verify_rollback[n_draft].size();
                     ++rollback) {
                    const int32_t count = n_verify_rollback[n_draft][rollback];
                    if (count == 0) {
                        continue;
                    }
                    if (!verify_rollback_hist.empty()) {
                        verify_rollback_hist += ", ";
                    }
                    verify_rollback_hist += string_format(
                        "%zu/%zu:%d", n_draft, rollback, count);
                }
            }
            SLT_INF(*this,
                    "verify/rollback histogram (draft/rejected:cycles) = (%s)\n",
                    verify_rollback_hist.c_str());
        }

        common_speculative_print_stats(spec.get());
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
            {"computation_frontier_ratchet", {
                {"read_path", frontier_ratchet_flipped ? "frontier" : "legacy"},
                {"threshold", frontier_ratchet_min_agreements},
                {"agreement_streak", frontier_ratchet_agreement_streak},
                {"agreements_total", frontier_ratchet_agreements_total},
                {"disagreements_total", frontier_ratchet_disagreements_total},
                {"flips_total", frontier_ratchet_flips_total},
                {"fallbacks_total", frontier_ratchet_fallbacks_total},
            }},
        };

        // live effective KV bits/value (moves under dynamic VBR); pollable via GET /slots
        if (ctx_tgt != nullptr) {
            const double kv_bpv = llama_memory_kv_bpv(llama_get_memory(ctx_tgt));
            if (kv_bpv >= 0.0) {
                res["kv_bpv"] = kv_bpv;
            }
            // co-tenancy state (all-zero = inert): what this server is offering peers,
            // and what it has yielded that a co-tenant is still claiming
            const auto ct = llama_vram_cotenancy(ctx_tgt);
            if (ct.shed_offer > 0 || ct.grants_active > 0 || ct.grant_decrement > 0) {
                res["cotenancy"] = json {
                    { "shed_offer",      ct.shed_offer },
                    { "grants_active",   ct.grants_active },
                    { "grant_decrement", ct.grant_decrement },
                    { "grant_pending",   ct.grant_pending },
                };
            }
            const auto vbr = llama_memory_vbr_state(llama_get_memory(ctx_tgt), id, 0);
            if (vbr.retier_freeze_enters > 0 || vbr.retier_freeze_depth > 0) {
                res["vbr_retier_freeze"] = json {
                    { "depth",              vbr.retier_freeze_depth },
                    { "env_freeze",         vbr.retier_env_freeze != 0 },
                    { "enters_total",       vbr.retier_freeze_enters },
                    { "exits_total",        vbr.retier_freeze_exits },
                    { "deferred_total",     vbr.retier_deferred_decisions },
                    { "reconciles_total",   vbr.retier_reconciles },
                };
            }
        }

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            if (!cache_status.empty()) {
                res["cache_status"] = cache_status; // [obs] why the last prompt (partially) reprocessed
            }
            // B0 shadow decision record [P2 §7.7]: last finalized record for this slot,
            // built once at finalize. Only ever non-null under --cache-debug.
            if (!cache_plan_json.is_null()) {
                res["cache_plan"] = cache_plan_json;
            }
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    // returns false if the state copy could not be performed [I13] (e.g. the recurrent pool has no
    // free cell for the child) -- the caller must not run the child on empty state
    bool copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        other.observe_live_range_drop(
            server_cache_destruction_reason::child_release, true);
        ::server_cache_live_range_drop_impl(ctx_tgt, other.id, -1, -1);
        if (!llama_memory_try_seq_cp(llama_get_memory(ctx_tgt), id, other.id, -1, -1)) {
            return false;
        }

        if (ctx_dft) {
            ::server_cache_live_range_drop_impl(ctx_dft, other.id, -1, -1);
            if (!llama_memory_try_seq_cp(llama_get_memory(ctx_dft), id, other.id, -1, -1)) {
                return false;
            }
        }

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.cache_family = cache_family;
        other.init_sampler();
        return true;
    }

    // returns 0 on success
    // caller need to update prompt.tokens after a successful call to keep track of the processing progress
    int process_mtmd_chunk(size_t idx, size_t & n_tokens_out) {
        GGML_ASSERT(mctx);
        const auto & input_tokens = task->tokens;
        const auto & chunk = input_tokens.find_chunk(idx);
        int32_t res = 0;

        auto try_decode = [&]() -> int32_t {
            if (mbatch) {
                float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
                if (embd) {
                    void * cb_data = spec.get();
                    static auto cb = [](llama_batch batch, void * user_data) {
                        common_speculative * spec = static_cast<common_speculative *>(user_data);
                        if (!common_speculative_process(spec, batch)) {
                            return 1;
                        }
                        return 0;
                    };

                    llama_pos new_n_past; // unused for now
                    res = mtmd_helper_decode_image_chunk(
                        mctx,
                        ctx_tgt,
                        chunk.get(),
                        embd,
                        prompt.tokens.pos_next(),
                        id,
                        llama_n_batch(ctx_tgt),
                        &new_n_past,
                        cb,
                        cb_data
                    );
                    if (res != 0) {
                        SLT_ERR(*this, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                        return -1;
                    }
                    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                    return 0; // success
                }
            }
            return 1; // (non-error) need to create & encode batch
        };

        // if the batch is already exist, try searching & encode
        res = try_decode();
        if (res == 0) {
            return 0;
        }
        if (res < 0) {
            // fatal error
            return res;
        }

        // otherwise, the batch is either uninitialized or is used up
        // we need to create & encode a new batch
        mbatch.reset(mtmd_batch_init(mctx));
        res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
        GGML_ASSERT(res == 0); // we should never have an empty batch

        // try batching as much as possible
        int n_added = 1;
        size_t idx_cur = idx;
        while (res == 0) {
            auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
            if (next_chunk == nullptr) {
                break;
            }
            res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
            n_added += (res == 0 ? 1 : 0);
            idx_cur = next_idx;
            SLT_DBG(*this, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
            // if res != 0, batch is full or chunk is not compatible -> this loop breaks
        }

        // TODO @ngxson : move this log line to debug when it become more stable
        SLT_TRC(*this, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

        res = mtmd_batch_encode(mbatch.get());
        if (res != 0) {
            SLT_ERR(*this, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
            return -1;
        }

        return try_decode();
    }
};

server_cache_family_slot_round_trip_result
server_cache_family_slot_round_trip_for_test(
        server_cache_control_authority & authority,
        server_cache_control_token binding_token,
        server_cache_control_token second_binding_token) {
    server_cache_family_slot_round_trip_result result;
    common_cache_family_binding incoming;
    result.resolved = server_cache_family_resolve_for_launch(
        &authority, binding_token, incoming) ==
            server_cache_control_status::ok;
    if (!result.resolved) {
        return result;
    }

    server_slot slot {};
    slot.id = 0;
    slot.cache_family = common_cache_family_follow_lineage(
        {}, incoming, 0, 0);
    slot.prompt.tokens = server_tokens(llama_tokens { 1, 2, 3 }, false);
    slot.prompt.sequence_epoch = 1;

    // The actual slot wrapper seeds the no-restore out parameter with its
    // current lineage. Drive the real cache selection path with no host rows;
    // this is the D1-1 identity terminal that previously returned a default.
    server_prompt_cache empty_cache(0, 0);
    common_cache_family_binding restored = slot.cache_family;
    const server_tokens resumed(
        llama_tokens { 1, 2, 3, 4 }, false);
    result.no_restore_resume = empty_cache.load(
        slot.prompt, resumed, nullptr, nullptr, slot.id, "", nullptr, -1,
        &restored);
    if (result.no_restore_resume) {
        slot.cache_family = restored;
    }
    result.binding_intact = slot.cache_family == incoming;

    auto staged = empty_cache.stage(slot.prompt, 8, 0, "");
    if (!staged.empty()) {
        server_prompt_cache_apply_family(
            staged.front(), slot.cache_family, true);
        result.host_save_carries =
            staged.front().cache_family == incoming &&
            staged.front().main_family ==
                common_cache_family_main_family(incoming, true);
    }
    common_prompt_checkpoint checkpoint;
    checkpoint.cache_family = slot.cache_family;
    result.checkpoint_carries = checkpoint.cache_family == incoming;

    if (second_binding_token) {
        common_cache_family_binding second;
        result.second_resolved = server_cache_family_resolve_for_launch(
            &authority, second_binding_token, second) ==
                server_cache_control_status::ok;
        result.roles_distinct = result.second_resolved &&
            incoming.family == second.family && incoming.role != second.role;
        if (result.roles_distinct) {
            server_slot second_slot {};
            second_slot.id = 1;
            second_slot.cache_family = common_cache_family_follow_lineage(
                {}, second, 0, 0);
            second_slot.prompt.tokens = server_tokens(
                llama_tokens { 5, 6, 7 }, false);
            second_slot.prompt.sequence_epoch = 2;

            server_prompt_cache two_slot_cache(0, 0);
            auto first_staged = two_slot_cache.stage(
                slot.prompt, 8, 0, "family-pair-first");
            auto second_staged = two_slot_cache.stage(
                second_slot.prompt, 8, 0, "family-pair-second");
            if (!first_staged.empty() && !second_staged.empty()) {
                server_prompt_cache_apply_family(
                    first_staged.front(), slot.cache_family, true);
                server_prompt_cache_apply_family(
                    second_staged.front(), second_slot.cache_family, true);
                result.host_roles_distinct =
                    first_staged.front().cache_family == incoming &&
                    second_staged.front().cache_family == second;
            }
        }
    }
    return result;
}



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    uint64_t n_draft_tokens_total      = 0;
    uint64_t n_draft_accepted_total    = 0;
    uint64_t n_draft_verif_steps_total = 0;
    std::vector<uint64_t> n_accepted_per_pos_total;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;

        n_draft_tokens_total      += slot.n_draft_total;
        n_draft_accepted_total    += slot.n_draft_accepted;
        n_draft_verif_steps_total += slot.n_draft_verif_steps;

        if (n_accepted_per_pos_total.size() < slot.n_accepted_per_pos.size()) {
            n_accepted_per_pos_total.resize(slot.n_accepted_per_pos.size(), 0);
        }
        for (size_t i = 0; i < slot.n_accepted_per_pos.size(); i++) {
            n_accepted_per_pos_total[i] += slot.n_accepted_per_pos[i];
        }
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;
    common_params params_load;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    // DFlash: one drafter context shared across all slots'
    // common_speculative states (non-owning refs). Must outlive all specs — the
    // destroy() order below (specs first, then this) enforces that; when destroy()
    // isn't called explicitly, member-destructor order (reverse-declaration) frees
    // specs via `slots` before this unique_ptr runs.
    llama_context_ptr ctx_dft_shared;

    server_batch batch;

    // fork: owning smart-pointers for the DFlash/draft model + drafter context.
    // (Upstream's raw model_dft/ctx_dft + the pimpl init-result owner are NOT used here —
    // the fork creation path in load_model()/create_mtp_context() manages these.)
    llama_model_ptr   model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;
    bool is_diffusion  = false;

    // hybrid/recurrent models need re-evaluation of accepted tokens after
    // rejecting draft tokens, because the recurrent state cannot be rolled back
    bool needs_reeval = false;
    int  n_parallel_user = 0;
    int  n_seq_max_full = 0; // target n_seq_max after expansion (2*n_parallel_user)

    server_recurrent_expansion_lifecycle recurrent_expansion;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    // In-process execution/lineage namespace for computation-frontier records.
    // Checkpoints are never portable across this random model-instance key;
    // slot-file restore explicitly invalidates them below.
    std::string frontier_execution_identity;
    uint64_t frontier_next_sequence_epoch = 1;
    uint64_t frontier_ratchet_threshold = 1024;

    // P2 F0b authority substrate (C0 ledger + coordinator + leases + retention + destruction).
    // Constructed under (cache_debug || cache_lifecycle). Declared before cache_plan_obs and
    // prompt_cache so it outlives both the observer that references it and the cache's
    // accounting-release destructor.
    std::unique_ptr<server_cache_authority> cache_authority;
    // F3 artifact machinery is a lifecycle-authority sibling: it references
    // the frozen ledger but is destroyed before it. It exists only for an
    // armed dynamic-VBR memory after the one-shot manifest and ring admission.
    std::unique_ptr<server_vbr_artifact_store> vbr_artifact_store;
    // E1.1a is lazily constructed by its scheduler-only task. destroy()
    // explicitly closes it before prompt_cache because host proofs point into
    // cache list nodes; this declaration after the F store is the secondary
    // reverse-destruction guard for retained package proofs. With no E1
    // route/flag, production allocates and executes nothing.
    std::unique_ptr<server_cache_control_authority> cache_control_authority;
    std::thread::id vbr_capture_scheduler_thread;
    std::thread::id cache_plan_preflight_scheduler_thread;

    // B-A authority substrate. Independent of debug serialization: non-off
    // graduated levels dual-run even when CACHE_PLAN JSON is disabled.
    std::unique_ptr<server_cache_plan_authority> cache_plan_authority;
    // Feature-neutral once-per-model profile source. Observability, B/D
    // authority, and E0 all copy from this one composition result without
    // forcing any sibling feature object to exist.
    std::string cache_plan_calibration_profile;

    // B0 shadow cache-plan observer [P2]. Constructed ONLY under params_base.cache_debug (B-a
    // literal: no observer object, no record init, no hook work of any kind on the disabled path
    // — absence IS the disabled state). References cache_authority for the substrate it records.
    std::unique_ptr<server_cache_plan_observer> cache_plan_obs;

    void cache_authority_config_failed(bool mirror_to_shadow) noexcept {
        cache_authority->configured = false;
        if (mirror_to_shadow && cache_plan_obs) {
            cache_plan_obs->shadow_unavailable++;
        }
    }

    // D-S1/F0b live producer. During initialization it configures authority under
    // debug||lifecycle; later debug finalization refreshes the same gauges for records. It
    // measures every distinct live target/draft context's resident cache
    // allocations from the library's canonical breakdown, then updates the manifested device
    // cells. A multi-device meta row cannot be assigned without per-shard bytes and therefore
    // fails closed rather than applying topology weights to a physical measurement.
    bool cache_plan_observe_live_memory(bool certify) noexcept {
        if (!cache_authority || cache_authority->live_device_domains.empty()) {
            return true;
        }

        try {
            using live_row = llama_live_memory_breakdown_data;
            std::vector<live_row> totals(cache_authority->live_device_domains.size());
            bool complete = true;

            const auto add_checked = [&complete](size_t & dst, size_t value) {
                if (value > std::numeric_limits<size_t>::max() - dst) {
                    complete = false;
                    return;
                }
                dst += value;
            };

            const llama_context * contexts[] = {
                ctx_tgt,
                ctx_dft.get(),
                ctx_dft_shared.get(),
            };
            for (size_t i_ctx = 0; i_ctx < std::size(contexts); ++i_ctx) {
                const llama_context * ctx = contexts[i_ctx];
                if (!ctx || std::find(contexts, contexts + i_ctx, ctx) != contexts + i_ctx) {
                    continue;
                }
                const auto breakdown = llama_get_live_memory_breakdown(ctx);
                for (const auto & [raw_buft, row] : breakdown) {
                    if (!raw_buft || ggml_backend_buft_is_host(raw_buft)) {
                        continue;
                    }

                    ggml_backend_buffer_type_t buft = raw_buft;
                    if (ggml_backend_buft_is_meta(buft)) {
                        if (ggml_backend_meta_buft_n_bufts(buft) != 1) {
                            complete = false;
                            continue;
                        }
                        buft = ggml_backend_meta_buft_simple_buft(buft, 0);
                    }

                    ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
                    const auto it = std::find_if(
                        cache_authority->live_device_domains.begin(),
                        cache_authority->live_device_domains.end(),
                        [device](const auto & binding) {
                            return binding.device == device;
                        });
                    if (it == cache_authority->live_device_domains.end()) {
                        complete = false;
                        continue;
                    }

                    auto & total = totals[size_t(
                        it - cache_authority->live_device_domains.begin())];
                    add_checked(total.attention,           row.attention);
                    add_checked(total.recurrent,           row.recurrent);
                    add_checked(total.recurrent_rollback,  row.recurrent_rollback);
                    add_checked(total.rolling_window_tape, row.rolling_window_tape);
                }
            }

            // complete is fixed after accumulation; when unavailable, every domain is marked
            // the same way, so the fail path is hoisted out of the per-domain gauge loop.
            if (!complete) {
                for (const auto & binding : cache_authority->live_device_domains) {
                    for (const auto category : {
                            llama_cache_acct_category::live_attention_state,
                            llama_cache_acct_category::live_recurrent_state,
                            llama_cache_acct_category::recurrent_rollback_planes,
                            llama_cache_acct_category::rolling_window_tape }) {
                        cache_authority->ledger.mark_unavailable(
                            category, binding.domain,
                            llama_cache_acct_measure::resident_allocated);
                    }
                    cache_authority->ledger.mark_producer_unavailable(
                        binding.domain, llama_cache_acct_producer::live_memory);
                }
                return false;
            }

            for (size_t i = 0; i < totals.size(); ++i) {
                const auto & domain = cache_authority->live_device_domains[i].domain;
                const auto & total  = totals[i];
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::live_attention_state, domain,
                    llama_cache_acct_measure::resident_allocated, total.attention);
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::live_recurrent_state, domain,
                    llama_cache_acct_measure::resident_allocated, total.recurrent);
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::recurrent_rollback_planes, domain,
                    llama_cache_acct_measure::resident_allocated,
                    total.recurrent_rollback);
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::rolling_window_tape, domain,
                    llama_cache_acct_measure::resident_allocated,
                    total.rolling_window_tape);
                if (certify) {
                    (void) cache_authority->ledger.certify_complete(
                        domain, llama_cache_acct_producer::live_memory);
                }
            }

            return complete;
        } catch (...) {
            for (const auto & binding : cache_authority->live_device_domains) {
                cache_authority->ledger.mark_producer_unavailable(
                    binding.domain, llama_cache_acct_producer::live_memory);
            }
            return false;
        }
    }

    // D-S2/F0b reserve-time capture. This runs once under debug||lifecycle.
    // The scheduler's reserved compute allocation is CONFIGURED
    // headroom; it is not an empirical peak and is never relabeled measured here.
    bool cache_plan_configure_budget() noexcept {
        if (!cache_authority) {
            return true;
        }
        try {
            cache_authority->budget_devices.clear();
            cache_authority->budget_devices.reserve(
                cache_authority->live_device_domains.size());
            for (const auto & binding : cache_authority->live_device_domains) {
                llama_cache_budget_device_input input;
                input.backend_device = binding.device;
                input.domain = binding.domain;
                input.phys_state = llama_cache_budget_capacity_state::unavailable;
                input.compute_state = llama_cache_budget_capacity_state::known;
                input.cache_cap_state = llama_cache_budget_capacity_state::unbounded;
                input.reserve_provenance =
                    llama_cache_budget_reserve_provenance::configured;
                cache_authority->budget_devices.push_back(input);
            }

            bool complete = true;
            const auto add_checked = [&complete](uint64_t & dst, size_t value) {
                if (uint64_t(value) >
                    std::numeric_limits<uint64_t>::max() - dst) {
                    complete = false;
                    return;
                }
                dst += uint64_t(value);
            };

            const llama_context * contexts[] = {
                ctx_tgt,
                ctx_dft.get(),
                ctx_dft_shared.get(),
            };
            for (size_t i_ctx = 0; i_ctx < std::size(contexts); ++i_ctx) {
                const llama_context * ctx = contexts[i_ctx];
                if (!ctx ||
                    std::find(contexts, contexts + i_ctx, ctx) !=
                        contexts + i_ctx) {
                    continue;
                }
                for (const auto & [raw_buft, row] :
                        llama_get_memory_breakdown(ctx)) {
                    if (row.compute == 0 || !raw_buft ||
                        ggml_backend_buft_is_host(raw_buft)) {
                        continue;
                    }
                    ggml_backend_buffer_type_t buft = raw_buft;
                    if (ggml_backend_buft_is_meta(buft)) {
                        if (ggml_backend_meta_buft_n_bufts(buft) != 1) {
                            complete = false;
                            continue;
                        }
                        buft = ggml_backend_meta_buft_simple_buft(buft, 0);
                    }
                    const ggml_backend_dev_t device =
                        ggml_backend_buft_get_device(buft);
                    auto it = std::find_if(
                        cache_authority->budget_devices.begin(),
                        cache_authority->budget_devices.end(),
                        [device](const auto & input) {
                            return input.backend_device == device;
                        });
                    if (it == cache_authority->budget_devices.end()) {
                        complete = false;
                        continue;
                    }
                    add_checked(it->current_compute_allocated, row.compute);
                }
            }

            for (auto & input : cache_authority->budget_devices) {
                input.configured_compute_reserve =
                    input.current_compute_allocated;
                if (!complete) {
                    input.compute_state =
                        llama_cache_budget_capacity_state::unavailable;
                }
            }
            return complete;
        } catch (...) {
            cache_authority->budget_devices.clear();
            return false;
        }
    }

    // D-S2 point-in-time shadow surface. Capacity sampling and all coordinator
    // arithmetic occur only under --cache-debug, immediately before emission.
    void cache_plan_emit_budget(
            const llama_cache_acct_snapshot & snapshot) noexcept {
        if (!cache_plan_obs) {
            return;
        }
        try {
            llama_cache_budget_config config;
            if (!cache_authority->sample_budget(config)) {
                cache_plan_obs->shadow_unavailable++;
                return;
            }

            llama_cache_budget_plan baseline;
            baseline.accounting_serial = snapshot.serial;
            if (!cache_authority->budget.reset(snapshot, config)) {
                cache_plan_obs->shadow_unavailable++;
                return;
            }
            const auto result = cache_authority->budget.fits(baseline);

            const auto compute_json = [](
                    const llama_cache_budget_device_input & input,
                    uint64_t value) {
                const llama_cache_acct_value typed =
                    input.compute_state ==
                        llama_cache_budget_capacity_state::known
                        ? llama_cache_acct_value::measured(value)
                        : llama_cache_acct_value {
                            0, llama_cache_acct_known::unavailable,
                        };
                return common_cache_plan_value_json(typed);
            };
            for (const auto & group : result.groups) {
                if (group.resource.kind !=
                    llama_cache_budget_resource_kind::physical_device) {
                    continue;
                }
                const auto input_it = std::find_if(
                    config.devices.begin(), config.devices.end(),
                    [&](const auto & input) {
                        return input.backend_device ==
                            group.resource.backend_device;
                    });
                if (input_it == config.devices.end()) {
                    cache_plan_obs->shadow_unavailable++;
                    continue;
                }

                std::string topology_digest;
                const auto topo_it = std::find_if(
                    snapshot.topologies.begin(), snapshot.topologies.end(),
                    [&](const auto & row) {
                        return row.id == input_it->domain.topology;
                    });
                if (topo_it != snapshot.topologies.end()) {
                    topology_digest = common_cache_plan_sha256_hex_digest(
                        topo_it->topology.digest.bytes());
                }

                const char * reason = "none";
                if (group.state ==
                    llama_cache_budget_fit_state::unavailable) {
                    reason = group.ceiling_state ==
                            llama_cache_budget_capacity_state::unavailable
                        ? "capacity_unavailable"
                        : "accounting_unavailable";
                } else if (group.state ==
                           llama_cache_budget_fit_state::exceeds) {
                    reason = "budget_exceeded";
                }

                const json line = {
                    { "accounting_serial", result.accounting_serial },
                    { "domain", {
                        { "residency", common_cache_acct_residency_name(
                            input_it->domain.residency) },
                        { "device_ordinal",
                            input_it->domain.device_ordinal.v },
                        { "topology_id", input_it->domain.topology.v },
                        { "topology_digest", topology_digest.empty()
                            ? json("unavailable")
                            : json(topology_digest) },
                    } },
                    { "device", {
                        { "name", ggml_backend_dev_name(
                            reinterpret_cast<ggml_backend_dev_t>(
                                const_cast<void *>(
                                    input_it->backend_device))) },
                        { "physical_total", input_it->physical_total },
                        { "physical_free", input_it->physical_free },
                        { "physical_state",
                            llama_cache_budget_capacity_state_name(
                                input_it->phys_state) },
                    } },
                    { "current_cache_resident",
                        common_cache_plan_value_json(
                            group.current_resident) },
                    { "current_compute_allocated",
                        compute_json(
                            *input_it, input_it->current_compute_allocated) },
                    { "configured_compute_reserve",
                        compute_json(
                            *input_it, input_it->configured_compute_reserve) },
                    { "reserve_provenance",
                        llama_cache_budget_reserve_provenance_name(
                            input_it->reserve_provenance) },
                    { "derived_cache_ceiling",
                        group.ceiling_state ==
                            llama_cache_budget_capacity_state::known
                            ? common_cache_plan_value_json(group.ceiling)
                            : json(llama_cache_budget_capacity_state_name(
                                group.ceiling_state)) },
                    { "fit", {
                        { "state", llama_cache_budget_fit_state_name(
                            group.state) },
                        { "reason", reason },
                        { "before",
                            common_cache_plan_value_json(group.before) },
                        { "after",
                            common_cache_plan_value_json(group.after) },
                        { "headroom_after",
                            group.headroom_state ==
                                llama_cache_budget_capacity_state::known
                                ? common_cache_plan_value_json(
                                    group.headroom_after)
                                : json(llama_cache_budget_capacity_state_name(
                                    group.headroom_state)) },
                    } },
                };
                SRV_INF("CACHE_BUDGET %s\n", line.dump().c_str());
            }
        } catch (...) {
            cache_plan_obs->shadow_unavailable++;
        }
    }

    bool cache_plan_assemble_yield_candidates(
            const std::vector<server_retention_candidate> & catalog,
            std::vector<server_cache_yield_candidate> & candidates) noexcept {
        try {
            const auto resolve = [&](const server_retention_candidate & source,
                                     server_cache_yield_candidate & candidate,
                                     server_cache_lease_identity & identity,
                                     bool & identity_known) {
                switch (source.instance_key.kind) {
                    case common_retention_artifact_kind::host_entry: {
                        if (prompt_cache &&
                            source.instance_key.owner_slot == -1 &&
                            source.instance_key.instance != 0) {
                            const auto backing = std::find_if(
                                prompt_cache->states.begin(),
                                prompt_cache->states.end(),
                                [&](const auto & entry) {
                                    return server_retention_instance_key::
                                               for_host_entry(&entry).instance ==
                                           source.instance_key.instance;
                                });
                            if (backing == prompt_cache->states.end()) {
                                candidate.availability =
                                    server_retention_candidate_availability::
                                        backing_missing_or_stale;
                                break;
                            }
                            const auto coverage =
                                source.record.stamp.coverage_tokens;
                            identity_known =
                                coverage <= uint64_t(INT64_MAX) &&
                                server_cache_lease_build_identity(
                                    frontier_execution_identity,
                                    backing->adapter_config_key,
                                    backing->prompt.tokens,
                                    int64_t(coverage),
                                    identity);
                            for (const auto op : {
                                    backing->acct_op_snapshot,
                                    backing->acct_op_ckpt,
                                    backing->acct_op_accel }) {
                                if (op) {
                                    candidate.release_ops.push_back(op);
                                }
                            }
                            if (candidate.release_ops.empty()) {
                                candidate.availability =
                                    server_retention_candidate_availability::
                                        backing_missing_or_stale;
                            }
                        } else {
                            candidate.availability =
                                server_retention_candidate_availability::
                                    backing_missing_or_stale;
                        }
                        break;
                    }
                    case common_retention_artifact_kind::live_slot: {
                        const auto slot_it = std::find_if(
                            slots.begin(), slots.end(),
                            [&](const auto & candidate_slot) {
                                return candidate_slot.id ==
                                    source.instance_key.owner_slot &&
                                    server_retention_instance_key::for_slot(
                                        candidate_slot.id).instance ==
                                        source.instance_key.instance;
                            });
                        if (slot_it != slots.end()) {
                            const auto coverage =
                                source.record.stamp.coverage_tokens;
                            identity_known =
                                coverage <= uint64_t(INT64_MAX) &&
                                server_cache_lease_build_identity(
                                    frontier_execution_identity,
                                    lora_config_identity(slot_it->lora),
                                    slot_it->prompt.tokens,
                                    int64_t(coverage),
                                    identity);
                            candidate.availability = slot_it->is_processing()
                                ? server_retention_candidate_availability::
                                      in_flight_mutation
                                : server_retention_candidate_availability::
                                      available;
                        } else {
                            candidate.availability =
                                server_retention_candidate_availability::
                                    backing_missing_or_stale;
                        }
                        // Sequence removal does not free the fixed pooled KV
                        // backing. The live descriptor therefore contributes
                        // a known-zero transactional release; independently
                        // owned checkpoint children in the same slot manifest
                        // contribute their real operations.
                        candidate.has_unsupported_host_spill = false;
                        break;
                    }
                    case common_retention_artifact_kind::checkpoint: {
                        // D-A4 live members publish exact independently releasable
                        // payload operations. Host-entry checkpoint clones remain
                        // aggregate-owned and therefore stay fail-closed here.
                        candidate.availability =
                            server_retention_candidate_availability::
                                backing_missing_or_stale;
                        candidate.has_unsupported_host_spill = true;
                        if (source.instance_key.instance == 0) {
                            break;
                        }

                        for (const auto & owner : slots) {
                            if (source.instance_key.owner_slot != owner.id) {
                                continue;
                            }
                            const auto checkpoint = std::find_if(
                                owner.prompt.checkpoints.begin(),
                                owner.prompt.checkpoints.end(),
                                [&](const auto & item) {
                                    return server_retention_instance_key::
                                               for_checkpoint(
                                                   owner.id, &item).instance ==
                                           source.instance_key.instance;
                                });
                            if (checkpoint !=
                                owner.prompt.checkpoints.end()) {
                                identity_known =
                                    server_cache_lease_build_identity(
                                        frontier_execution_identity,
                                        lora_config_identity(owner.lora),
                                        owner.prompt.tokens,
                                        checkpoint->n_tokens,
                                        identity);
                                if (owner.is_processing()) {
                                    candidate.availability =
                                        server_retention_candidate_availability::
                                            in_flight_mutation;
                                } else if (!source.release_ops.empty()) {
                                    candidate.availability =
                                        server_retention_candidate_availability::
                                            available;
                                    candidate.has_unsupported_host_spill = false;
                                }
                                break;
                            }
                        }
                        if (!identity_known &&
                            source.instance_key.owner_slot == -1 &&
                            prompt_cache) {
                            for (const auto & owner : prompt_cache->states) {
                                const auto checkpoint = std::find_if(
                                    owner.prompt.checkpoints.begin(),
                                    owner.prompt.checkpoints.end(),
                                    [&](const auto & item) {
                                        return server_retention_instance_key::
                                                   for_checkpoint(
                                                       -1, &item).instance ==
                                               source.instance_key.instance;
                                    });
                                if (checkpoint !=
                                    owner.prompt.checkpoints.end()) {
                                    identity_known =
                                        server_cache_lease_build_identity(
                                            frontier_execution_identity,
                                            owner.adapter_config_key,
                                            owner.prompt.tokens,
                                            checkpoint->n_tokens,
                                            identity);
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    case common_retention_artifact_kind::_count:
                        candidate.availability =
                            server_retention_candidate_availability::
                                backing_missing_or_stale;
                        break;
                }
            };
            if (!server_cache_yield_assemble(
                    catalog, cache_authority->leases, resolve, candidates)) {
                candidates.clear();
                return false;
            }

            return true;
        } catch (...) {
            candidates.clear();
            return false;
        }
    }

    struct cache_plan_host_source_registry {
        explicit cache_plan_host_source_registry(bool publish) noexcept
            : publish_(publish) {}

        void begin(server_prompt_cache * cache) noexcept {
            if (publish_ && cache) {
                cache->cache_plan_begin_inventory();
                published_.clear();
                published_.reserve(cache->states.size());
                for (auto & state : cache->states) {
                    published_.emplace(
                        server_retention_instance_key::for_host_entry(
                            &state).instance,
                        &state);
                }
            }
        }

        bool get(
                server_prompt_cache & cache,
                server_prompt_cache_state & state,
                int32_t & source_id) {
            if (publish_) {
                return cache.cache_plan_get_source_id(state, source_id);
            }
            return local_.get_or_assign(
                server_retention_instance_key::for_host_entry(
                    &state).instance,
                source_id);
        }

        bool find(uintptr_t instance, int32_t & source_id) const noexcept {
            if (!publish_) {
                return local_.find(instance, source_id);
            }
            const auto found = published_.find(instance);
            if (found == published_.end() ||
                found->second->cache_plan_source_id < 0) {
                source_id = -1;
                return false;
            }
            source_id = found->second->cache_plan_source_id;
            return true;
        }

    private:
        bool publish_ = false;
        server_cache_plan_local_source_registry local_;
        std::unordered_map<uintptr_t, const server_prompt_cache_state *>
            published_;
    };

    bool cache_plan_destruction_artifacts(
            std::vector<server_cache_destruction_artifact> & artifacts,
            const cache_plan_host_source_registry * source_registry =
                nullptr) noexcept {
        artifacts.clear();
        if (!cache_authority) {
            return false;
        }
        try {
            const auto catalog =
                cache_authority->retention.candidate_snapshot();
            std::vector<server_cache_yield_candidate> normalized;
            if (!cache_plan_assemble_yield_candidates(catalog, normalized) ||
                normalized.size() != catalog.size()) {
                return false;
            }
            std::unordered_map<uintptr_t, int32_t> existing_host_source_ids;
            if (!source_registry && prompt_cache) {
                existing_host_source_ids.reserve(prompt_cache->states.size());
                for (const auto & state : prompt_cache->states) {
                    existing_host_source_ids.emplace(
                        server_retention_instance_key::for_host_entry(
                            &state).instance,
                        state.cache_plan_source_id);
                }
            }
            const auto find_host_source = [&](
                    uintptr_t instance, int32_t & source_id) noexcept {
                if (source_registry) {
                    return source_registry->find(instance, source_id);
                }
                const auto found = existing_host_source_ids.find(instance);
                if (found == existing_host_source_ids.end() ||
                    found->second < 0) {
                    source_id = -1;
                    return false;
                }
                source_id = found->second;
                return true;
            };
            artifacts.reserve(catalog.size());
            for (size_t i = 0; i < catalog.size(); ++i) {
                server_cache_destruction_artifact artifact;
                artifact.candidate = std::move(normalized[i]);
                // Retiring a sidecar association also retires its capacity-
                // participating descriptor. D-A's exact union therefore
                // includes the provenance operation as well as payload ops;
                // the post-commit retirement terminal clears both owners.
                if (catalog[i].provenance_op) {
                    artifact.candidate.release_ops.push_back(
                        catalog[i].provenance_op);
                    std::sort(
                        artifact.candidate.release_ops.begin(),
                        artifact.candidate.release_ops.end());
                    artifact.candidate.release_ops.erase(
                        std::unique(
                            artifact.candidate.release_ops.begin(),
                            artifact.candidate.release_ops.end()),
                        artifact.candidate.release_ops.end());
                }
                artifact.kind = catalog[i].instance_key.kind;
                artifact.owner_slot = catalog[i].instance_key.owner_slot;
                artifact.pool = catalog[i].record.stamp.pool;
                artifact.mandatory_anchor =
                    catalog[i].record.stamp.mandatory_anchor;
                artifact.fixed_pool_logical_ownership =
                    artifact.kind ==
                    common_retention_artifact_kind::live_slot;
                if (artifact.kind ==
                        common_retention_artifact_kind::host_entry) {
                    (void) find_host_source(
                        catalog[i].instance_key.instance,
                        artifact.host_source_id);
                }
                artifacts.push_back(std::move(artifact));
            }
            return true;
        } catch (...) {
            artifacts.clear();
            return false;
        }
    }

    void cache_plan_quote_destruction(
            int32_t legacy_target_slot,
            bool host_lookup_enabled,
            common_cache_plan_record & rec,
            server_cache_destruction_quote_options options,
            uint64_t * admission_sequence,
            common_cache_plan_destruction_counters & counters,
            const cache_plan_host_source_registry * source_registry) noexcept {
        if (!cache_authority || !options.lifecycle_available) {
            rec.destruction.state = common_cache_plan_destruction_state::refused;
            rec.destruction.reason =
                common_cache_plan_destruction_reason::lifecycle_disabled;
            return;
        }
        try {
            const auto fail_whole_pass = [&](
                    common_cache_plan_destruction_state state,
                    common_cache_plan_destruction_reason reason) {
                rec.destruction.state = state;
                rec.destruction.reason = reason;
                counters.observe(rec.selection, rec.destruction);
            };
            const int32_t legacy_candidate =
                server_cache_plan_legacy_candidate(
                    rec, legacy_target_slot, host_lookup_enabled);
            if (!cache_authority->configured) {
                fail_whole_pass(
                    common_cache_plan_destruction_state::refused,
                    common_cache_plan_destruction_reason::lifecycle_disabled);
                return;
            }
            if (legacy_candidate < 0 ||
                uint32_t(legacy_candidate) >= rec.n_inventory ||
                rec.inventory_saturated()) {
                fail_whole_pass(
                    common_cache_plan_destruction_state::refused,
                    common_cache_plan_destruction_reason::manifest_incomplete);
                return;
            }
            if (!server_cache_destruction_has_effect(
                    rec, legacy_candidate,
                    server_cache_plan_nonconsuming_host_effects(
                        params_base.cache_lifecycle))) {
                return;
            }

            const auto snapshot = cache_authority->ledger.snapshot();
            std::vector<server_cache_destruction_artifact> artifacts;
            if (!cache_plan_destruction_artifacts(
                    artifacts, source_registry)) {
                fail_whole_pass(
                    common_cache_plan_destruction_state::refused,
                    common_cache_plan_destruction_reason::manifest_incomplete);
                return;
            }

            llama_cache_budget_config config;
            llama_cache_budget_coordinator quote_budget;
            const bool budget_ready =
                cache_authority->sample_budget(config) &&
                quote_budget.reset(snapshot, config);
            const auto preview = [&](const auto & ops,
                                     uint64_t serial,
                                     auto & out) {
                return cache_authority->ledger.preview_release_set(
                    ops, serial, out);
            };
            const auto project = [&](
                    const llama_cache_acct_release_set_preview & release,
                    std::vector<common_cache_plan_yield_domain> & out) {
                out.clear();
                if (!budget_ready ||
                    release.accounting_serial != snapshot.serial) {
                    return false;
                }
                llama_cache_budget_plan plan;
                if (!server_cache_yield_release_plan(
                        release, snapshot.serial, plan)) {
                    return false;
                }
                const auto fit = quote_budget.fits(plan);
                if (fit.accounting_serial != snapshot.serial ||
                    fit.state != llama_cache_budget_fit_state::fits) {
                    return false;
                }
                out.reserve(fit.domains.size());
                for (const auto & row : fit.domains) {
                    common_cache_plan_yield_domain lowered;
                    if (!server_cache_yield_lower_domain(row, lowered)) {
                        out.clear();
                        return false;
                    }
                    out.push_back(std::move(lowered));
                }
                return true;
            };
            // Production advances its sequence only at the original
            // post-assembly boundary. A read-only caller passes null and can
            // never consume production quote identity.
            if (admission_sequence) {
                if (*admission_sequence == UINT64_MAX) {
                    fail_whole_pass(
                        common_cache_plan_destruction_state::failed,
                        common_cache_plan_destruction_reason::internal_fault);
                    return;
                }
                options.admission_sequence = ++*admission_sequence;
            }
            if (!server_cache_destruction_quote_all(
                    rec, legacy_candidate, artifacts, snapshot.serial,
                    preview, project, options, counters)) {
                if (rec.destruction.reason ==
                    common_cache_plan_destruction_reason::none) {
                    rec.destruction.state =
                        common_cache_plan_destruction_state::failed;
                    rec.destruction.reason =
                        common_cache_plan_destruction_reason::internal_fault;
                }
                return;
            }

            // D-A5 supplies B's previously unavailable eviction term before
            // the single planner optimum runs. The cost is the fitted cost of
            // restoring the displaced complete state, multiplied by the
            // landed soft-lease/main-family retention weights. Hard and
            // unavailable evidence never receive a priced term and remain
            // behind the certification envelope. An unsaved live slot has no
            // host-entry size proxy, and dynamic recurrent/VBR state cannot be
            // inferred from token count, so ranking requires the exact metadata
            // size query here. It is memoized once per victim; the landed D-A0a
            // 27B trace measured the complete quote pass at p95 101 us, well
            // inside the normative 2 ms launch-path bound.
            const auto * calib = common_cache_plan_calib_find(
                rec.calibration_profile);
            if (params_base.cache_lifecycle && calib) {
                std::unordered_map<int32_t, llama_cache_acct_value>
                    victim_state_sizes;
                for (const auto & quote : rec.destruction_quotes) {
                    const auto & receipt = quote.receipt;
                    if (receipt.state !=
                            common_cache_plan_destruction_state::quoted ||
                        receipt.plan_candidate < 0 ||
                        uint32_t(receipt.plan_candidate) >= rec.n_inventory) {
                        continue;
                    }
                    auto & candidate =
                        rec.inventory[size_t(receipt.plan_candidate)];
                    if (common_cache_plan_provider_is_live(
                            candidate.provider)) {
                        // These plans consume the resident prefix and may
                        // trim only a suffix later. A whole-slot capability
                        // cannot honestly authorize that narrower seam.
                        continue;
                    }
                    auto * victim = cache_plan_slot_by_exact_id(
                        candidate.target_slot_id);
                    if (!victim || victim->prompt.tokens.empty() ||
                        victim->is_processing()) {
                        continue;
                    }
                    auto [size_it, inserted] = victim_state_sizes.emplace(
                        victim->id, llama_cache_acct_value{});
                    if (inserted) {
                        const size_t target_bytes =
                            llama_state_seq_get_size_ext(
                                ctx_tgt, victim->id,
                                LLAMA_STATE_SEQ_FLAGS_NONE);
                        const size_t draft_bytes = ctx_dft
                            ? llama_state_seq_get_size_ext(
                                  ctx_dft.get(), victim->id,
                                  LLAMA_STATE_SEQ_FLAGS_NONE)
                            : 0;
                        if (target_bytes <= SIZE_MAX - draft_bytes) {
                            size_it->second = llama_cache_acct_value::measured(
                                target_bytes + draft_bytes);
                        }
                    }
                    if (size_it->second.state !=
                            llama_cache_acct_known::known) {
                        continue;
                    }
                    const uint64_t bytes = size_it->second.value;
                    // Child slots clear their prompts on release, so every
                    // retained non-empty idle conversation is provisionally
                    // main-family until E1 supplies declared identity.
                    const bool main_family = common_cache_family_main_family(
                        victim->cache_family,
                        !victim->task || !victim->task->is_child());
                    uint32_t weight = 0;
                    uint64_t price_us = 0;
                    if (!server_cache_host_retention_price_us(
                            *calib, bytes,
                            receipt.lease_verdict ==
                                common_cache_plan_destruction_lease_verdict::
                                    soft_leased,
                            main_family, weight, price_us)) {
                        continue;
                    }
                    auto & term = candidate.cost_terms[size_t(
                        llama_cache_acct_cost_kind::eviction)];
                    term.raw = llama_cache_acct_value::measured(bytes);
                    term.estimated_us =
                        llama_cache_acct_value::measured(price_us);
                    term.estimator_version = calib->estimator_version;
                }
            }
        } catch (...) {
            rec.destruction_quotes.clear();
            rec.destruction.state = common_cache_plan_destruction_state::failed;
            rec.destruction.reason =
                common_cache_plan_destruction_reason::internal_fault;
            counters.observe(rec.selection, rec.destruction);
        }
    }

    struct live_displacement_certification {
        bool ready = false;
        common_cache_plan_destruction_effect_set effects = 0;
        server_cache_prepared_release_capability capability;
        common_cache_plan_destruction_quote quote;
    };

    live_displacement_certification cache_plan_certify_live_displacement(
            server_slot & victim,
            server_slot & legacy_target,
            common_cache_plan_record & rec) noexcept {
        live_displacement_certification out;
        if (!params_base.cache_lifecycle || !cache_authority ||
            !prompt_cache || !server_cache_plan_shadow_choice_valid(rec) ||
            rec.destruction_legacy_plan_candidate < 0 ||
            !cache_plan_authority ||
            !server_cache_plan_level_enabled(
                cache_plan_authority->configured_level,
                server_cache_plan_level_of(rec.selection)) ||
            !server_cache_plan_candidate_prequalified(rec) ||
            victim.prompt.tokens.empty()) {
            return out;
        }
        if (rec.inventory[size_t(rec.shadow_choice)].target_slot_id !=
                victim.id) {
            return out;
        }
        out.effects = server_cache_destruction_effects_for(
            rec, rec.shadow_choice,
            rec.destruction_legacy_plan_candidate,
            server_cache_plan_nonconsuming_host_effects(true));
        if (out.effects == 0) {
            return out;
        }
        if ((out.effects & ~SERVER_CACHE_LIVE_DISPLACEMENT_EFFECTS) != 0) {
            return out;
        }
        const auto & planned = rec.inventory[size_t(rec.shadow_choice)];
        if (common_cache_plan_provider_is_live(planned.provider)) {
            // Live-prefix/suffix mutation has its own range semantics. D-A5's
            // full-slot terminal must never clear the state the selected live
            // replay intends to consume; those candidates remain behind the
            // envelope until a range-exact capability is available.
            // This silent decline is not a refusal transition: the quote is
            // still evidence and must not be counted twice.
            return out;
        }

        const auto quote_it = std::find_if(
            rec.destruction_quotes.begin(), rec.destruction_quotes.end(),
            [&](const auto & candidate) {
                return candidate.receipt.plan_candidate == rec.shadow_choice;
            });
        if (quote_it == rec.destruction_quotes.end()) {
            return out;
        }
        out.quote = *quote_it;
        const auto emit_refusal = [&](bool observe_transition) {
            rec.destruction = out.quote.receipt;
            if (observe_transition) {
                cache_authority->destruction_counters.observe(
                    rec.selection, rec.destruction, false);
            }
            cache_authority->destruction.note_live_displacement_refused();
            if (cache_plan_obs) {
                try {
                    const auto payload = server_cache_destruction_receipt_json(
                        rec.destruction, 0, "live_slot_displacement");
                    SRV_INF("CACHE_HOST_DESTRUCTION %s\n",
                            payload.dump().c_str());
                } catch (...) {
                }
            }
        };
        const auto refuse = [&](common_cache_plan_destruction_reason reason) {
            out.quote.receipt.state =
                common_cache_plan_destruction_state::refused;
            out.quote.receipt.reason = reason;
            emit_refusal(true);
        };
        if (out.quote.receipt.state !=
                common_cache_plan_destruction_state::quoted) {
            emit_refusal(false);
            return out;
        }

        // Resolve the prospective citation to the node THIS save publishes,
        // not another token-identical peer. Pin the victim first: a later
        // legacy-prefix publication can then neither dedup the victim nor
        // leave the shorter legacy frontier without an exact durable copy.
        server_prompt_cache::iterator saved_victim;
        const auto saved = victim.prompt_save(
            *prompt_cache, true, &saved_victim);
        if (saved != prompt_save_result::published ||
            saved_victim == prompt_cache->states.end()) {
            refuse(common_cache_plan_destruction_reason::recovery_unavailable);
            return out;
        }

        llama_cache_acct_artifact_id recovery_artifact;
        std::vector<llama_cache_acct_op_id> recovery_ops;
        server_cache_recovery_pin recovery_pin;
        if (!prompt_cache->acquire_durable_recovery(
                saved_victim,
                recovery_artifact, recovery_ops, recovery_pin)) {
            refuse(common_cache_plan_destruction_reason::recovery_unavailable);
            return out;
        }
        prompt_cache->update();
        if (&legacy_target != &victim) {
            const auto legacy_saved = legacy_target.prompt_save(*prompt_cache);
            prompt_cache->update();
            if (!prompt_save_durable(legacy_saved) ||
                !recovery_pin.valid() ||
                !prompt_cache->contains(
                    legacy_target.prompt.tokens,
                    lora_config_identity(legacy_target.lora))) {
                refuse(
                    common_cache_plan_destruction_reason::recovery_unavailable);
                return out;
            }
        }

        std::vector<server_cache_destruction_artifact> current;
        if (!cache_plan_destruction_artifacts(current)) {
            refuse(common_cache_plan_destruction_reason::manifest_incomplete);
            return out;
        }
        const server_cache_destruction_projection_callback project =
            [&](const auto & released, auto & domains) {
                return cache_authority->project_release(released, domains);
            };
        const auto fresh = cache_authority->ledger.snapshot();
        auto prepared = server_cache_prepare_release_set(
            out.quote, current, cache_authority->ledger,
            fresh.serial, project, std::move(recovery_pin));
        if (prepared.status !=
                server_cache_prepare_release_status::prepared) {
            refuse(prepared.reason);
            return out;
        }

        server_cache_destruction_certify_receipt(
            out.quote.receipt,
            common_cache_plan_displaced_fate::retained_host,
            recovery_artifact, recovery_ops);
        rec.destruction = out.quote.receipt;
        cache_authority->destruction_counters.observe(
            rec.selection, rec.destruction, false);
        cache_authority->destruction.note_live_displacement_certified();
        out.capability = std::move(prepared.capability);
        out.ready = true;
        return out;
    }

    void cache_plan_run_yield(
            const llama_cache_acct_snapshot & snapshot,
            std::vector<server_retention_candidate> catalog) noexcept {
        if (!cache_plan_obs) {
            return;
        }
        try {
            std::vector<server_cache_yield_candidate> candidates;
            if (!cache_plan_assemble_yield_candidates(catalog, candidates)) {
                cache_authority->last_yield = {};
                cache_authority->last_yield.accounting_serial =
                    snapshot.serial;
                return;
            }

            const auto preview = [&](const auto & ops,
                                     uint64_t expected_serial,
                                     auto & out) {
                return cache_authority->ledger.preview_release_set(
                    ops, expected_serial, out);
            };
            const auto fit = [&](const llama_cache_budget_plan & plan) {
                // The empty baseline has no prefix preview, so probe the live
                // ledger here to preserve Lock A4's atomic-capture serial guard.
                llama_cache_acct_release_set_preview serial_probe;
                if (!cache_authority->ledger.preview_release_set(
                        {}, snapshot.serial, serial_probe)) {
                    llama_cache_budget_result unavailable;
                    unavailable.accounting_serial = snapshot.serial;
                    return unavailable;
                }
                return cache_authority->budget.fits(plan);
            };
            cache_authority->last_yield = server_cache_yield_plan(
                candidates, snapshot.serial, preview, fit);
        } catch (...) {
            cache_authority->last_yield = {};
            cache_authority->last_yield.accounting_serial =
                snapshot.serial;
            cache_plan_obs->shadow_unavailable++;
        }
    }

    // D-S6 observer-only surface. D-S7 owns the later schema-4 projection;
    // this line deliberately does not enter the cache-plan wire record.
    void cache_plan_emit_yield(
            const server_cache_yield_result & result) noexcept {
        if (!cache_plan_obs) {
            return;
        }
        try {
            const json line = {
                { "status", server_cache_yield_status_name(result.status) },
                { "yield_policy_version", result.yield_policy_version },
                { "accounting_serial", result.accounting_serial },
                { "n_selected_attention",
                    result.selected[size_t(
                        common_retention_pool::attention)].size() },
                { "n_selected_recurrent",
                    result.selected[size_t(
                        common_retention_pool::recurrent)].size() },
                { "n_plan_entries", result.plan.size() },
                { "n_unsupported", result.unsupported.size() },
            };
            SRV_INF("CACHE_YIELD %s\n", line.dump().c_str());
        } catch (...) {
            cache_plan_obs->shadow_unavailable++;
        }
    }

    // D-S7 schema-v4 lowering boundary. Budget types remain server/process-local;
    // only accounting domains and typed values cross into the common wire record.
    // Build off-record and move once so an allocation/mapping fault cannot expose a
    // partial projection or suppress the independently-finalized B0 record.
    void cache_plan_project_yield(
            common_cache_plan_record & rec,
            const server_cache_yield_result & result) noexcept {
        try {
            common_cache_plan_yield_record projected;
            const auto actual_state = rec.yield.actual_state;
            auto actual_domains = std::move(rec.yield.actual_domains);
            if (result.accounting_serial != rec.acct.serial) {
                throw std::runtime_error(
                    "cache-yield record/accounting serial mismatch");
            }
            switch (result.status) {
                case server_cache_yield_status::fits:
                    projected.status = common_cache_plan_yield_status::fits;
                    break;
                case server_cache_yield_status::insufficient_yield:
                    projected.status =
                        common_cache_plan_yield_status::insufficient_yield;
                    break;
                case server_cache_yield_status::unsupported_required:
                    projected.status =
                        common_cache_plan_yield_status::unsupported_required;
                    break;
                case server_cache_yield_status::unavailable:
                    projected.status = common_cache_plan_yield_status::unavailable;
                    break;
                case server_cache_yield_status::_count:
                    throw std::runtime_error("invalid cache-yield status");
            }

            projected.yield_policy_version = result.yield_policy_version;
            projected.accounting_serial = result.accounting_serial;
            projected.selected_attention =
                result.selected[size_t(common_retention_pool::attention)];
            projected.selected_recurrent =
                result.selected[size_t(common_retention_pool::recurrent)];
            projected.unsupported = result.unsupported;

            const bool selected_any =
                !projected.selected_attention.empty() ||
                !projected.selected_recurrent.empty();
            if (result.status == server_cache_yield_status::fits) {
                if (result.projected_fit.accounting_serial !=
                        result.accounting_serial ||
                    result.projected_fit.state !=
                        llama_cache_budget_fit_state::fits) {
                    throw std::runtime_error(
                        "cache-yield projection serial/state mismatch");
                }
                projected.plan_state = selected_any
                    ? common_cache_plan_yield_plan_state::planned
                    : common_cache_plan_yield_plan_state::not_required;
                // Baseline-fit rows remain in the server pre-image as serial-bound
                // evidence, but no-eviction wire records carry no projected action.
                if (selected_any) {
                    if (result.plan.empty() ||
                        result.projected_fit.domains.empty()) {
                        throw std::runtime_error(
                            "cache-yield planned projection is empty");
                    }
                    projected.projected_domains.reserve(
                        result.projected_fit.domains.size());
                    for (const auto & row : result.projected_fit.domains) {
                        common_cache_plan_yield_domain lowered;
                        if (!server_cache_yield_lower_domain(row, lowered)) {
                            throw std::runtime_error(
                                "non-domain cache-yield projection row");
                        }
                        projected.projected_domains.push_back(
                            std::move(lowered));
                    }
                }
            } else {
                projected.plan_state =
                    common_cache_plan_yield_plan_state::unavailable;
            }

            // D-S never executes an eviction. D-A may fill this already-versioned
            // slot only after it has authoritative post-mutation measurements.
            if (rec.destruction.state ==
                    common_cache_plan_destruction_state::executed) {
                projected.actual_state = actual_state;
                projected.actual_domains = std::move(actual_domains);
            } else {
                projected.actual_state =
                    common_cache_plan_yield_actual_state::not_observed;
            }
            rec.yield = std::move(projected);
        } catch (...) {
            rec.yield = {};
            if (cache_plan_obs) {
                cache_plan_obs->shadow_unavailable++;
            }
        }
    }

    // B0 finalize [P2 §7.7]: exactly one final record per request, emitted only after the
    // actual restore/cold path AND all fallible trims/fallbacks and the prompt replay have
    // settled — the call sites ride the SAME timing points that feed metrics.on_prompt_eval
    // (no second clock). ttft_known=false for embedding/rerank: their prompt completes but
    // there is no first generated token, so TTFT stays typed unavailable. Every observer
    // fault is caught here, outside the shipped decision path.
    void cache_plan_finalize(server_slot & slot, bool ttft_known = true) {
        if (!slot.cache_plan) {
            return;
        }
        try {
            auto & rec = *slot.cache_plan;

            // once-only: outcome != unknown IS the finalized state; a second finalization
            // attempt on the same record is an observer fault, never a second emission
            if (rec.outcome != common_cache_plan_outcome::unknown) {
                if (cache_plan_obs) {
                    cache_plan_obs->shadow_unavailable++;
                }
                slot.cache_plan.reset();
                return;
            }

            rec.n_prompt_tokens   = llama_cache_acct_value::measured((uint64_t) slot.task->n_tokens());
            rec.n_reused_tokens   = llama_cache_acct_value::measured((uint64_t) slot.n_prompt_tokens_cache);
            rec.n_replayed_tokens = llama_cache_acct_value::measured((uint64_t) slot.n_prompt_tokens_processed);
            if (ttft_known) {
                // measured actual (t_prompt_processing is ms) — never an estimate slot
                rec.ttft_us = llama_cache_acct_value::measured((uint64_t) (slot.t_prompt_processing * 1000.0));
            }

            // live-slot evidence comes from the FINAL post-trim state, not the selection
            // heuristics: realized prefix reuse IS live-slot delivery (with the default zero
            // similarity threshold an LRU pick can still reuse its own prefix), and a
            // stage-1 heuristic reject is superseded by that fact. PROVENANCE guard (F2): a
            // host restore replaces the slot state wholesale, so a positive final reuse
            // count after host delivery is the HOST's prefix — never attributed to the
            // original live slot.
            {
                const auto * host = rec.selected_row(common_cache_plan_provider::host_cache_entry);
                auto *       live = rec.selected_row(common_cache_plan_provider::live_slot);
                if (slot.n_prompt_tokens_cache > 0 && !(host && host->delivered) && live) {
                    live->disposition = common_cache_plan_disposition::accepted;
                    live->delivered   = true;
                    live->reason      = COMMON_CACHE_PLAN_REASON_NONE;
                    live->lcp_tokens  = llama_cache_acct_value::measured((uint64_t) slot.n_prompt_tokens_cache);
                }
            }

            // cold replay is the always-valid floor: it always has a row, and it delivered
            // exactly when nothing else did
            if (auto * cold = rec.find_or_add(
                    common_cache_plan_provider::cold_replay,
                    COMMON_CACHE_PLAN_SOURCE_AGGREGATE, uint8_t(0), rec.id_slot,
                    rec.selection)) {
                cold->disposition = common_cache_plan_disposition::accepted;
                rec.select(common_cache_plan_provider::cold_replay, cold);
                rec.note_inventory_complete(common_cache_plan_provider::cold_replay);
            }

            // chosen = the TERMINAL delivered provider (delivery is data recorded at each
            // site; the causal chain is emitted separately, so composition is not lost)
            const common_cache_plan_provider terminal_order[] = {
                common_cache_plan_provider::live_context_checkpoint,
                common_cache_plan_provider::host_cache_entry,
                common_cache_plan_provider::live_slot,
            };
            const common_cache_plan_candidate * delivered_row = nullptr;
            for (const auto prov : terminal_order) {
                const auto * c = rec.selected_row(prov);
                if (c && c->delivered) {
                    delivered_row = c;
                    rec.chosen    = prov;
                    break;
                }
            }

            if (delivered_row) {
                rec.outcome = common_cache_plan_outcome::restored;
            } else if (rec.restore_attempt_failed) {
                rec.outcome = common_cache_plan_outcome::restore_failed_fell_back_cold;
                rec.chosen  = common_cache_plan_provider::cold_replay;
            } else {
                rec.outcome = common_cache_plan_outcome::cold;
                rec.chosen  = common_cache_plan_provider::cold_replay;
            }
            if (rec.chosen == common_cache_plan_provider::cold_replay) {
                if (auto * cold = rec.selected_row(common_cache_plan_provider::cold_replay)) {
                    cold->delivered = true;
                }
            }

            // composed plan as a first-class candidate (verify-r1 finding 1): a
            // multi-provider delivery (host snapshot + checkpoint continuation) gets ONE
            // chain row — DELIVERED, since it is what actually ran — and the chain IS the
            // complete shipped plan. The bare checkpoint row becomes component-only: its
            // state was reachable only through the delivered host entry, so it can never
            // enter the root optimum as a standalone plan.
            common_cache_plan_compose_chains(rec);

            if (cache_plan_obs) {
                if (!cache_plan_observe_live_memory(false)) {
                    cache_plan_obs->shadow_unavailable++;
                }
                rec.acct = cache_authority->ledger.snapshot();
                auto retention_candidates =
                    cache_authority->retention.candidate_snapshot();
                cache_plan_emit_budget(rec.acct);
                cache_plan_run_yield(
                    rec.acct, std::move(retention_candidates));
                cache_plan_project_yield(rec, cache_authority->last_yield);
                server_cache_destruction_finalize_projection(
                    rec, cache_authority->last_yield);
                cache_authority->destruction_counters.last_receipt =
                    rec.destruction;
                cache_authority->destruction_counters.has_receipt = true;
                cache_plan_emit_yield(cache_authority->last_yield);
            }

            // ---- planner boundary [A2]: estimation, tie-set construction, and shadow
            // choice run HERE, inside their own boundary — failure clears planner outputs
            // only; B0 finalization and emission continue unconditionally. Every attempt
            // outcome is a closed status on the record, and refusals move the observer
            // counter exactly once (verify-r1 finding 8).
            if (!rec.planner_precomputed) {
                common_cache_plan_run_planner(rec);
            }
            if (cache_plan_obs) {
                if (rec.planner_status == common_cache_plan_planner_status::ok) {
                    cache_plan_obs->planner_ok++;
                } else {
                    cache_plan_obs->planner_refused++;
                }
            }
            if (cache_plan_authority) {
                cache_plan_authority->finalize_execution(rec);
            } else {
                common_cache_plan_finalize_shadow_authority(rec);
            }

            if (cache_plan_obs) {
                json out = common_cache_plan_record_json(rec);
                cache_plan_obs->records_finalized++;
                json authority_by_tier = json::object();
                // cache_plan_obs implies the shared debug-or-authority substrate,
                // so cache_plan_authority is always present here.
                GGML_ASSERT(cache_plan_authority);
                const auto & authority = cache_plan_authority->counters;
                for (size_t tier = 0;
                     tier < size_t(common_cache_plan_selection::_count); tier++) {
                    authority_by_tier[common_cache_plan_selection_name(
                        common_cache_plan_selection(tier))] = json {
                        { "observed", authority.observed[tier] },
                        { "eligible", authority.authority_eligible[tier] },
                        { "executed", authority.authority_executed[tier] },
                        { "agree", authority.agree[tier] },
                        { "disagree", authority.disagree[tier] },
                        { "fallback_legacy", authority.fallback_legacy[tier] },
                    };
                }
                out["observer"] = json {
                    { "records_finalized",  cache_plan_obs->records_finalized },
                    { "shadow_unavailable", cache_plan_obs->shadow_unavailable },
                    { "planner_ok",         cache_plan_obs->planner_ok },
                    { "planner_refused",    cache_plan_obs->planner_refused },
                    { "authority",          std::move(authority_by_tier) },
                };

                SRV_INF("CACHE_PLAN %s\n", out.dump().c_str());
                slot.cache_plan_json = std::move(out); // built once; /slots reuses it verbatim
            }
            slot.cache_plan.reset();
        } catch (...) {
            if (cache_plan_obs) {
                cache_plan_obs->shadow_unavailable++;
            }
            slot.cache_plan.reset();
        }
    }

    uint64_t ensure_frontier_sequence_epoch(server_prompt & prompt) {
        if (prompt.sequence_epoch == 0) {
            GGML_ASSERT(frontier_next_sequence_epoch != 0);
            prompt.sequence_epoch = frontier_next_sequence_epoch++;
            GGML_ASSERT(frontier_next_sequence_epoch != 0);
        }
        return prompt.sequence_epoch;
    }

    static bool computation_frontiers_equal(
            const common_computation_frontier & a,
            const common_computation_frontier & b) {
        return a.version == b.version &&
               a.sequence_epoch == b.sequence_epoch &&
               a.token_count == b.token_count &&
               a.next_position == b.next_position &&
               a.execution_identity == b.execution_identity &&
               a.adapter_config_identity == b.adapter_config_identity &&
               a.media_content_identity == b.media_content_identity;
    }

    bool checkpoint_frontier_is_current(
            const server_slot & slot,
            const common_prompt_checkpoint & checkpoint,
            const std::string & adapter_identity) const {
        const auto & frontier = checkpoint.computation_frontier;
        if (!frontier.valid() ||
            frontier.sequence_epoch != slot.prompt.sequence_epoch ||
            frontier.execution_identity != frontier_execution_identity ||
            frontier.adapter_config_identity != adapter_identity ||
            frontier.token_count != checkpoint.n_tokens ||
            checkpoint.pos_max < 0 ||
            frontier.next_position <= 0 ||
            frontier.next_position - 1 != checkpoint.pos_max) {
            return false;
        }

        std::string media_identity;
        return slot.prompt.tokens.media_content_identity(
                   frontier.token_count, media_identity) &&
               media_identity == frontier.media_content_identity;
    }

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    // MTP↔mmproj GPU swap state
    bool mmproj_gpu_swap = false;
    bool mmproj_is_on_gpu = false;
    bool mtp_was_active_before_swap = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        if (ctx_tgt) {
            llama_get_memory(ctx_tgt)->vbr_hard_seal_guard_set({});
        }
        // E1 host-fallback proofs call back into prompt-cache list nodes, and
        // F subject/fallback proofs retain the artifact store. Close every
        // holder/lease/proof explicitly while both owners are still alive;
        // reverse member destruction would otherwise free prompt_cache first.
        cache_control_authority.reset();
        // Slot-held D-A recovery pins point into cache_authority->retention.
        // `slots` is declared before that owner and would otherwise be
        // destroyed after it; close every dependency explicitly while the
        // storage callback target is still alive.
        for (auto & slot : slots) {
            slot.cache_plan_destruction_recovery_pin = {};
        }
        if (cache_authority && !cache_authority->summary_emitted) {
            SRV_INF(
                "CACHE_AUTHORITY_SUMMARY configured=%d commits=%" PRIu64
                " refusals=%" PRIu64 " retries=%" PRIu64
                " rollbacks=%" PRIu64 "\n",
                cache_authority->configured,
                cache_authority->admission_commits,
                cache_authority->admission_refusals,
                cache_authority->admission_retries,
                cache_authority->admission_rollbacks);
            cache_authority->summary_emitted = true;
        }

        spec.reset();

        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        for (server_slot & slot : slots) {
            if (slot.can_speculate()) {
                slot.spec.reset();
            }
        }

        ctx_dft_shared.reset();

        // note: batch is a server_batch — its destructor frees the llama_batch
    }

    mtmd_context_params make_mmproj_params(bool use_gpu, mtmd_progress_callback progress_cb = nullptr, void * progress_ud = nullptr) const {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = use_gpu;
        mparams.print_timings    = false;
        mparams.n_threads        = params_base.cpuparams.n_threads;
        mparams.flash_attn_type  = params_base.flash_attn_type;
        mparams.warmup           = use_gpu; // only warmup when on GPU
        mparams.image_min_tokens = params_base.image_min_tokens;
        mparams.image_max_tokens = params_base.image_max_tokens;
        mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
        mparams.media_marker     = get_media_marker();
        mparams.progress_callback           = progress_cb;
        mparams.progress_callback_user_data = progress_ud;
        return mparams;
    }

    void reload_mmproj(bool use_gpu) {
        for (server_slot & slot : slots) {
            slot.mbatch.reset();
        }
        mtmd_free(mctx);
        mctx = nullptr;
        auto mparams = make_mmproj_params(use_gpu);
        mctx = mtmd_init_from_file(params_base.mmproj.path.c_str(), model_tgt, mparams);
        for (server_slot & slot : slots) {
            slot.mctx = mctx;
        }
    }

    llama_context * create_mtp_context() {
        auto cparams = common_context_params_to_llama(params_base);
        // Auto-fit mutates the target's llama_context_params, not params_base.  Reuse the
        // realized target width here; otherwise n_ctx=0 expands the MTP cache to n_ctx_train even
        // when the fitted target is much smaller.
        cparams.n_ctx         = llama_n_ctx_seq(ctx_tgt);
        cparams.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
        cparams.type_k        = params_base.speculative.draft.cache_type_k;
        cparams.type_v        = params_base.speculative.draft.cache_type_v;
        cparams.n_rs_seq      = 0;
        cparams.n_outputs_max = params_base.n_parallel;
        cparams.ctx_other     = ctx_tgt;
        return llama_init_from_model(model_tgt, cparams);
    }

    void swap_mtp_to_mmproj_gpu() {
        SRV_INF("%s", "swapping MTP out, loading mmproj to GPU...\n");
        int64_t t0 = ggml_time_us();

        for (server_slot & slot : slots) {
            slot.spec.reset();
            slot.spec_shared = nullptr;
        }
        spec.reset();

        mtp_was_active_before_swap = ctx_dft != nullptr;
        if (ctx_dft) {
            ctx_dft.reset();
            params_base.speculative.draft.ctx_dft = nullptr;
        }

        reload_mmproj(true);
        if (!mctx) {
            SRV_ERR("%s", "failed to load mmproj to GPU, falling back to CPU\n");
            reload_mmproj(false);
        } else {
            mmproj_is_on_gpu = true;
        }

        SRV_INF("swap done in %" PRId64 " ms\n", (ggml_time_us() - t0) / 1000);
    }

    void swap_mmproj_to_mtp() {
        SRV_INF("%s", "unloading mmproj from GPU, restoring prior state...\n");
        int64_t t0 = ggml_time_us();

        mmproj_is_on_gpu = false;
        reload_mmproj(false);

        if (mtp_was_active_before_swap) {
            // Already had MTP before the swap — recreate it
            ctx_dft.reset(create_mtp_context());
            if (!ctx_dft) {
                SRV_ERR("%s", "failed to recreate MTP context after mmproj swap\n");
                return;
            }

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();

            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to reinit speculative context: %s\n", e.what());
            }

            for (server_slot & slot : slots) {
                slot.ctx_dft = ctx_dft.get();
                if (spec) {
                    slot.spec_shared = spec.get();
                    common_speculative_set_seq_id(slot.get_spec(), slot.id);
                }
            }
        } else {
            // No MTP was active — mmproj stays on CPU until next image arrives
            SRV_INF("%s", "(no MTP to restore, mmproj will reload on next image)\n");
        }

        SRV_INF("swap done in %" PRId64 " ms\n", (ggml_time_us() - t0) / 1000);
    }

    // dynamic VBR: clear-only reclaim of idle slots (the prompt cache is disabled under the VBR
    // gates, so unlike cache_idle_slots there is nothing to save into — the cost is a re-prefill
    // if that conversation returns). Never touches processing slots or a slot an explicitly
    // pinned deferred task is waiting on. Returns the number of slots cleared.
    int vbr_clear_idle_slots(int except_id, const char * reason) {
        int cleared = 0;
        for (auto & s : slots) {
            if (s.id == except_id || s.is_processing() || s.prompt.n_tokens() == 0) {
                continue;
            }
            // E1.1c approved guard, not D-A certification: the legacy eraser
            // simply does not consider a hard-leased live prefix.
            if (s.hard_lease_blocks_live_prefix()) {
                SLT_INF(s, "vbr reclaim (%s): kept — hard lease seals the live prefix\n", reason);
                continue;
            }
            if (queue_tasks.has_deferred_for_slot(s.id)) {
                SLT_INF(s, "vbr reclaim (%s): kept — a deferred id_slot task pins this slot\n", reason);
                continue;
            }
            SLT_WRN(s, "vbr reclaim (%s): clearing %d cached tokens\n", reason, (int) s.prompt.n_tokens());
            s.prompt_clear(server_cache_destruction_reason::idle_reclaim);
            cleared++;
        }
        return cleared;
    }

    // reclaim-before-degrade: when the incoming work's projected footprint would push the
    // degrade ladder BELOW the quality floor, pay with idle caches instead of tiers. Above the
    // floor the trade inverts — a within-band degrade is near-lossless and cheaper than
    // destroying another conversation's re-prefillable cache, so caches are kept. Policy reads
    // deficit_raw only: page-exact and free-VRAM independent, so WHETHER a cache is erased
    // never depends on driver jitter or co-tenants (the cache's own live clamp still handles
    // those with tier degrades).
    void vbr_reclaim_before_degrade(int except_id, uint32_t n_tokens_extra, const char * reason) {
        if (!server_vbr_dynamic_active(params_base) || params_base.vbr_reclaim_floor_bpv <= 0.0f) {
            return;
        }
        const auto st = llama_memory_vbr_state(llama_get_memory(ctx_tgt), -1, n_tokens_extra);
        if (st.deficit_raw <= 0 || st.bpv_if_degraded >= (double) params_base.vbr_reclaim_floor_bpv) {
            return;
        }
        const int cleared = vbr_clear_idle_slots(except_id, reason);
        if (cleared > 0) {
            SRV_WRN("vbr reclaim (%s): cleared %d idle slot(s) — deficit %.2f MiB, degrading instead would land %.3f bpv < floor %.3f\n",
                    reason, cleared, st.deficit_raw / 1024.0 / 1024.0, st.bpv_if_degraded,
                    (double) params_base.vbr_reclaim_floor_bpv);
        }
    }

    // reset-on-low-LCP (dynamic VBR): tiers are per-tensor and promotion cannot cross the tap
    // boundary, so the ONLY full-quality recovery is the lossless empty-cache reset. When a
    // DEGRADED conversation keeps a token-trivial prefix anyway (rolling-window agent
    // harnesses rewrite mid-context every turn), trade the prefix for the reset: clear the
    // idle slots (recovery reclaim — deliberately NOT deficit-gated: budget slack is exactly
    // the state that starves the pressure path, and worthless idle caches are the only
    // obstacle to recovery that benefits every stream), and if the pool then holds nothing
    // but this slot's cells, drop n_past to 0 — the full re-prefill re-enters at the entry
    // tier, so turn-N cache quality equals turn-1. Returns the (possibly zeroed) n_past.
    int vbr_reset_on_low_lcp(
            server_slot & slot,
            int n_past,
            server_cache_destruction_admission & admission) {
        if (!server_vbr_dynamic_active(params_base) || params_base.vbr_reset_keep_frac <= 0.0f) {
            return n_past;
        }
        const int n_prompt = slot.task->n_tokens();
        if (n_prompt <= 0 || (float) n_past >= params_base.vbr_reset_keep_frac * (float) n_prompt) {
            return n_past;
        }
        llama_memory_t mem = llama_get_memory(ctx_tgt);
        auto st = llama_memory_vbr_state(mem, slot.id, 0);
        if (st.cursor < 2) {
            return n_past; // pristine or one transient step — a full re-prefill buys ~nothing
        }
        // E1.1c approved guard, not a destructive capability. The range is
        // exactly the prefix this recovery path would discard.
        if (slot.hard_lease_blocks_live_range(
                0, static_cast<uint64_t>(std::max(0, n_past)))) {
            SLT_INF(slot, "%s", "vbr reset blocked: hard lease seals the reusable prefix\n");
            return n_past;
        }
        vbr_clear_idle_slots(slot.id, "reset-recovery");
        st = llama_memory_vbr_state(mem, slot.id, 0);
        if (st.used_cells_other > 0) {
            SLT_INF(slot, "vbr reset blocked: %u used cells belong to other sequences (pinned or processing slots)\n",
                    st.used_cells_other);
            return n_past;
        }
        SLT_WRN(slot, "vbr reset: cursor %d and only %d/%d prompt tokens reusable (< %.2f) — dropping the prefix; "
                "the full re-prefill re-enters at the entry tier\n",
                st.cursor, n_past, n_prompt, (double) params_base.vbr_reset_keep_frac);
        admission = slot.observe_live_range_drop(
            server_cache_destruction_reason::low_lcp_reset, true);
        if (admission.execution !=
                server_cache_destruction_execution::pass_through) {
            return n_past;
        }
        (void) slot.checkpoint_drop_joined_impl(
            slot.prompt.checkpoints.begin(), slot.prompt.checkpoints.end());
        return 0;
    }

    void recurrent_shrink_for_prefill(const char * reason) {
        if (!needs_reeval || n_seq_max_full <= n_parallel_user) {
            return;
        }

        // An allocation failure suppresses retries for the rest of the active request.
        // Rearm exactly once at the next quiescent request/prefill boundary.
        if (recurrent_expansion.retry_deferred) {
            for (const server_slot & slot : slots) {
                if (slot.is_processing() || slot.has_draft_backup) {
                    SRV_DBG("not rearming recurrent expansion (%s): slot %d processing=%d has_backup=%d\n",
                            reason, slot.id, slot.is_processing(), slot.has_draft_backup);
                    return;
                }
            }
            if (recurrent_expansion.rearm_if_idle(true)) {
                SRV_INF("rearmed recurrent expansion retry at idle prefill boundary (%s)\n", reason);
            }
        }

        if (recurrent_expansion.state != server_recurrent_expansion_state::expanded) {
            return;
        }

        // DFlash: backup cells are ~1.5 MB each, but the shrink/expand cycle costs
        // ~100 ms of buffer realloc + CUDA-graph re-capture at the first draft of EVERY
        // request (measured on a 3090: -np 4 lost ~12% end-to-end on a 101-token reply).
        // The memory savings never outweigh that — keep the backup cells resident.
        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            return;
        }

        for (const server_slot & slot : slots) {
            if (slot.is_processing() || slot.has_draft_backup) {
                SRV_DBG("not shrinking recurrent state for prefill (%s): slot %d processing=%d has_backup=%d\n",
                        reason, slot.id, slot.is_processing(), slot.has_draft_backup);
                return;
            }
        }

        auto * mem = llama_get_memory(ctx_tgt);
        for (const server_slot & slot : slots) {
            const llama_seq_id seq_backup = slot.id + n_parallel_user;
            server_cache_transient_seq_rm_impl(mem, seq_backup, -1, -1);
        }

        if (llama_memory_recurrent_shrink(mem, n_parallel_user)) {
            recurrent_expansion.state = server_recurrent_expansion_state::contracted;
            SRV_INF("shrunk recurrent state to %d cells for prefill (%s, removed %d backup cells)\n",
                    n_parallel_user, reason, n_seq_max_full - n_parallel_user);
        } else {
            SRV_ERR("failed to shrink recurrent state to %d cells for prefill (%s)\n",
                    n_parallel_user, reason);
        }
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        // co-tenancy: declare serviced BEFORE any context init — the first marker publish
        // rides context creation, and an explicit-budget resident's fields never change
        // again, so a late flag would stay unpublished forever
        llama_vram_mark_serviced();

        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;
        if (!is_resume) {
            params_load = params;
        }

        params_base = params_load;
        params_base.n_outputs_max = server_n_outputs_max(params_base);

        // Qwen-27B MTP can derive and reuse a frequency-prior compact head. A
        // standalone sidecar is repacked as a smaller child; a combined GGUF emits
        // only a compact-head artifact which is attached to the native model, so its
        // token embedding and NextN block remain shared. This runs before fit so the
        // first launch can reserve the attached head. The source is never modified.
        const bool draft_vocab_requested =
                params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP) &&
                !params_base.speculative.draft.draft_vocab_pack.empty();
        const bool native_vocab_tensor_split =
                draft_vocab_requested && !params_base.speculative.has_dft() &&
                params_base.split_mode == LLAMA_SPLIT_MODE_TENSOR;
        if (draft_vocab_requested && !native_vocab_tensor_split) {
            SRV_INF("[spec] checking Qwen MTP vocabulary-trim cache (pack=%s, vocab=32768)\n",
                    params_base.speculative.draft.draft_vocab_pack.c_str());
            const bool        external_mtp = params_base.speculative.has_dft();
            const std::string source_path  = external_mtp
                                           ? params_base.speculative.draft.mparams.path
                                           : params_base.model.path;
            auto trim = common_mtp_vocab_trim_prepare(
                    source_path,
                    params_base.speculative.draft.draft_vocab_pack);
            if (trim.status == common_mtp_vocab_trim_status::created) {
                SRV_INF("[spec] created Qwen MTP vocabulary-trim artifact '%s' (%s)\n",
                        trim.path.c_str(), trim.detail.c_str());
            } else if (trim.status == common_mtp_vocab_trim_status::cached) {
                SRV_INF("[spec] using cached Qwen MTP vocabulary-trim artifact '%s' (%s)\n",
                        trim.path.c_str(), trim.detail.c_str());
            } else if (trim.status == common_mtp_vocab_trim_status::failed) {
                SRV_WRN("[spec] Qwen MTP vocabulary trim unavailable: %s; using original %s\n",
                        trim.detail.c_str(), external_mtp ? "sidecar" : "embedded MTP layer");
            } else {
                SRV_WRN("[spec] Qwen MTP vocabulary pack '%s' is not applicable: %s; using original %s\n",
                        params_base.speculative.draft.draft_vocab_pack.c_str(), trim.detail.c_str(),
                        external_mtp ? "sidecar" : "embedded MTP layer");
            }
            if (trim.status == common_mtp_vocab_trim_status::created ||
                trim.status == common_mtp_vocab_trim_status::cached) {
                if (trim.native_head) {
                    params_base.speculative.draft.draft_vocab_artifact_path = std::move(trim.path);
                    params_base.speculative.draft.draft_vocab_resident_bytes = trim.resident_bytes;
                } else {
                    params_base.speculative.draft.mparams.path = std::move(trim.path);
                }
            }
        } else if (native_vocab_tensor_split) {
            SRV_WRN("%s", "[spec] --spec-draft-vocab does not support tensor-split native output heads; "
                          "using the full MTP vocabulary without changing fit margins\n");
        }

        const bool has_mmproj = !params_base.mmproj.path.empty();
        const bool has_draft = params_base.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;
        const server_shared_draft_device_config shared_draft_devices = server_prepare_shared_draft_devices(params_base);

        auto make_params_dft = [&]() -> server_resolved_draft_params {
            common_params params_dft = common_base_params_to_speculative(params_base);
            const bool cpu_dspark_backbone = server_has_cpu_dspark_backbone(params_base);
            // Tensor split is a target-only topology. Normalize the draft once
            // here so fit measurement and the eventual child load see identical
            // devices/buckets instead of measuring a meta device and loading a
            // layer-split model later.
            if (params_dft.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
                params_dft.split_mode = LLAMA_SPLIT_MODE_LAYER;
            }
            if (shared_draft_devices.prepared) {
                params_dft.devices = shared_draft_devices.devices;
                params_dft.main_gpu = 0;
                params_dft.split_mode = LLAMA_SPLIT_MODE_LAYER;
                std::fill(std::begin(params_dft.tensor_split), std::end(params_dft.tensor_split), 0.0f);
                std::copy(shared_draft_devices.tensor_split.begin(), shared_draft_devices.tensor_split.end(),
                          std::begin(params_dft.tensor_split));
                if (shared_draft_devices.n_weight_devices == 0) {
                    params_dft.n_gpu_layers = 0;
                }

                // The DSpark experts remain CPU-resident, but its output and
                // lightweight layers are substantially faster on the selected draft GPU.
                // Preserve an explicit device=none request and provide a dedicated opt-out
                // for users who prefer the previous, smaller GPU allocation.
                const bool dspark_gpu_assist =
                    params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) &&
                    params_base.speculative.draft.dspark_gpu_assist &&
                    params_base.speculative.draft.n_gpu_layers == 0 &&
                    params_base.moe_cache.mode != COMMON_MOE_CACHE_MODE_OFF &&
                    shared_draft_devices.n_weight_devices > 0;
                if (dspark_gpu_assist) {
                    // n_gpu_layers=4 includes every lightweight repeating layer, whose
                    // expert tensors are several GiB. Keep those experts on CPU so GPU
                    // assist moves only the inexpensive dense/tail tensors. User
                    // overrides remain first in the list and retain precedence.
                    server_append_tensor_override(params_dft, llm_ffn_exps_cpu_override());
                    params_dft.n_gpu_layers = 4;
                    SRV_INF("[spec] enabling DSpark GPU assist on %s (disable with --no-spec-dspark-gpu-assist)\n",
                            ggml_backend_dev_name(shared_draft_devices.devices[0]));
                }

                // A CPU-layer DSpark drafter is dominated by its vocabulary-sized
                // output/Markov tail. Keep the 3-layer backbone (including its large
                // expert tensors) on the CPU, but place the small Markov/confidence
                // weights with the shared target output on the selected draft GPU.
                // An explicit --spec-draft-device none remains fully CPU-resident.
                if (params_dft.n_gpu_layers == 0 &&
                    shared_draft_devices.n_weight_devices > 0) {
                    server_append_tensor_override(params_dft, {
                            "^(markov_w[12]|conf_proj)\\.",
                            ggml_backend_dev_buffer_type(shared_draft_devices.devices[0]) });
                    if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
                        SRV_INF("[spec] keeping CPU DSpark Markov/confidence tail on %s\n",
                                ggml_backend_dev_name(shared_draft_devices.devices[0]));
                    }
                }
            }
            return { std::move(params_dft), cpu_dspark_backbone };
        };

        // One server load includes the target, linked draft/MTP, multimodal,
        // slot, tape and compatibility allocations below. The target/common
        // loader nests inside this scope but cannot finish the process claim.
        llama_vram_load_begin(has_spec);
        bool load_succeeded = false;
        struct load_scope {
            bool & succeeded;
            ~load_scope() { llama_vram_load_end(succeeded); }
        } load_guard { load_succeeded };

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        const std::string & mmproj_path = params_base.mmproj.path;

        // measure mmproj memory for auto-fit (upstream #21489)
        // Only reserve mmproj space when auto-fit is actively selecting context size.
        // When -c is explicit, the fitter doesn't run so there's no need to reserve.
        if (has_mmproj && params_base.fit_params && params_base.n_ctx == 0 && !params_base.mmproj_gpu_swap) {
            auto mparams_measure = make_mmproj_params(params_base.mmproj_use_gpu);
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams_measure);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                const std::vector<ggml_backend_dev_t> target_fit_devices = server_target_fit_devices(params_base);
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < target_fit_devices.size(); i++) {
                        if (target_fit_devices[i] == dev) {
                            GGML_ASSERT(i < params_base.fit_params_target.size());
                            SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                            params_base.fit_params_target[i] += size;
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        n_parallel_user = params_base.n_parallel;
        recurrent_expansion.state = server_recurrent_expansion_state::expanded;
        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            // Native MTP is context-sized by the target fit, so its reservation is solved after
            // the speculative n_parallel expansion below.  Every entry here has a separate
            // draft path, including external MTP sidecars, and must be measured from that child
            // GGUF rather than charging a second copy of the target model.
            if (has_draft) {
                common_params params_dft = std::move(make_params_dft().params);

                auto mparams_dft = common_model_params_to_llama(
                        params_dft, common_model_role::speculative_child);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto mparams_tgt = common_model_params_to_llama(params_base);
                    auto cparams_tgt = common_context_params_to_llama(params_base);
                    const char * path_dft = params_dft.model.path.c_str();
                    auto measure_spec_memory = [&](const llama_model_params & mparams_measure,
                                                   const llama_model_params & mparams_parent,
                                                   std::vector<ggml_backend_dev_t> & measured_devs,
                                                   uint32_t & measured_ngl,
                                                   uint32_t & measured_nct,
                                                   uint32_t & measured_nex) {
                        auto data = common_get_device_memory_data_with_parent(
                            path_dft, &mparams_measure, &cparams_dft,
                            params_base.model.path.c_str(), &mparams_parent, &cparams_tgt,
                            measured_devs, measured_ngl, measured_nct, measured_nex, GGML_LOG_LEVEL_ERROR);
                        if (!spec_mtp) {
                            return data;
                        }

                        std::vector<ggml_backend_dev_t> target_devs;
                        uint32_t target_ngl = 0;
                        uint32_t target_nct = 0;
                        uint32_t target_nex = 0;
                        const auto target = common_get_device_memory_data(
                            params_base.model.path.c_str(), &mparams_parent, &cparams_tgt,
                            target_devs, target_ngl, target_nct, target_nex, GGML_LOG_LEVEL_ERROR);
                        if (target_devs != measured_devs || target.size() != data.size()) {
                            throw std::runtime_error("MTP and target memory devices differ");
                        }
                        for (size_t i = 0; i < data.size(); i++) {
                            if (target[i].compute > SIZE_MAX - data[i].compute) {
                                throw std::runtime_error("MTP memory estimate overflowed");
                            }
                            data[i].compute += target[i].compute;
                        }
                        return data;
                    };

                    const llama_model_params & mparams_measure = mparams_dft;
                    auto dmd = measure_spec_memory(
                        mparams_measure, mparams_tgt, devs, hp_ngl, hp_nct, hp_nex);

                    std::vector<std::pair<ggml_backend_dev_t, size_t>> reservations;
                    auto add_reservations = [&](const common_device_memory_data_vec & data,
                                                const std::vector<ggml_backend_dev_t> & devices) {
                        GGML_ASSERT(data.size() >= devices.size());
                        for (size_t i = 0; i < devices.size(); i++) {
                            const size_t bytes = data[i].model + data[i].context + data[i].compute;
                            auto found = std::find_if(reservations.begin(), reservations.end(), [&](const auto & entry) {
                                return entry.first == devices[i];
                            });
                            if (found == reservations.end()) {
                                reservations.emplace_back(devices[i], bytes);
                            } else {
                                found->second = std::max(found->second, bytes);
                            }
                        }
                    };
                    add_reservations(dmd, devs);

                    // Shared-weight DFlash/DSpark children can move target-owned tensors onto
                    // the target's main device, so also measure that placement. An external
                    // MTP sidecar owns its embedding and output tensors and has no such second
                    // placement; measuring it again would merely duplicate the layer-split
                    // result (and used to substitute the target GGUF by mistake).
                    if (shared_draft_devices.prepared && mparams_tgt.split_mode != LLAMA_SPLIT_MODE_NONE &&
                        mparams_tgt.main_gpu >= 0) {
                        try {
                            llama_model_params mparams_tgt_main = mparams_tgt;
                            mparams_tgt_main.split_mode = LLAMA_SPLIT_MODE_NONE;
                            llama_model_tensor_buft_override mtp_overrides[2] = {};
                            if (spec_mtp) {
                                mtp_overrides[0] = {
                                    common_moe_cache_tensor_override_pattern(), ggml_backend_cpu_buffer_type() };
                                mparams_tgt_main.tensor_buft_overrides = mtp_overrides;
                                mparams_tgt_main.use_extra_bufts = false;
                            }
                            std::vector<ggml_backend_dev_t> devs_main;
                            uint32_t hp_ngl_main = 0;
                            uint32_t hp_nct_main = 0;
                            uint32_t hp_nex_main = 0;
                            const auto dmd_main = measure_spec_memory(
                                mparams_dft, mparams_tgt_main,
                                devs_main, hp_ngl_main, hp_nct_main, hp_nex_main);
                            add_reservations(dmd_main, devs_main);
                        } catch (const std::exception & e) {
                            SRV_DBG("[spec] failed to measure main-device shared-tensor memory: %s\n", e.what());
                        }
                    }

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    const std::vector<ggml_backend_dev_t> tgt_devices = server_target_fit_devices(params_base);

                    for (const auto & [device, bytes] : reservations) {
                        total += bytes;
                        if (bytes == 0) {
                            continue;
                        }
                        // fork: publish the draft demand to the co-tenancy ledger and RAISE the
                        // per-device fit margin to at least the reservation (not +=): the aux
                        // plan hint already charges the drafter bytes to the shared ledger, so
                        // accumulating here would double-count against co-tenants
                        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                            ggml_backend_dev_props props;
                            ggml_backend_dev_get_props(device, &props);
                            if (props.device_id != nullptr) {
                                llama_vram_plan_aux(props.device_id, bytes);
                            }
                        }
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == device) {
                                if (bytes > params_base.fit_params_target[i]) {
                                    SRV_DBG("[spec] raising fit_params_target to %.2f MiB for device %s\n",
                                            bytes / (1024.0 * 1024.0), ggml_backend_dev_name(device));
                                    params_base.fit_params_target[i] = bytes;
                                }
                                break;
                            }
                        }
                    }
                    SRV_INF("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_WRN("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                    // benign for ctx_other-family drafters (dflash/eagle3): they borrow the
                    // target's tok_embd/output, so a standalone context cannot be created
                    // before the target exists (llama_init_from_model logs the reason at
                    // WARN, which the measurement's log filter demotes to DEBUG). The only
                    // effect is that the draft model VRAM is not charged to the fit reserve.
                }
            }
        }

        // When mmproj GPU swap is active, run the fitter before n_parallel doubling.  The MTP
        // context and GPU mmproj are never resident together: the image path destroys MTP before
        // loading mmproj, then unloads mmproj before recreating MTP.  Reserve the ordinary margin
        // plus the larger of those two phase-local consumers, not their sum.  Native MTP is sized
        // from the fitted context, so solve the resulting dependency to a fixed point.
        //
        // The doubled n_parallel used for speculative rollback would also make the dry target
        // context carry decode-only recurrent backup cells during this prefill-oriented solve.
        const bool has_mtp = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
        if (params_base.mmproj_gpu_swap && has_mtp && has_mmproj
                && params_base.fit_params && params_base.n_ctx == 0) {
            std::vector<size_t> margins_base = params_base.fit_params_target;
            GGML_ASSERT(!margins_base.empty());

            std::vector<ggml_backend_dev_t> tgt_devices = params_base.devices;
            if (tgt_devices.empty()) {
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    tgt_devices.push_back(ggml_backend_dev_get(i));
                }
            }

            // This special solve is the final context advert: unlike the ordinary VBR growth
            // ceiling, it must be fillable from the memory that is actually available at
            // startup.  Charge memory already unavailable on each target GPU (CUDA/driver
            // residency, plus any non-cooperating process) instead of letting the total-based
            // dry fit spend it a second time.
            std::vector<size_t> startup_unavailable(margins_base.size(), 0);
            for (size_t i = 0; i < tgt_devices.size() && i < margins_base.size(); ++i) {
                if (ggml_backend_dev_type(tgt_devices[i]) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                    continue;
                }
                size_t free = 0;
                size_t total = 0;
                ggml_backend_dev_memory(tgt_devices[i], &free, &total);
                startup_unavailable[i] = total > free ? total - free : 0;
                margins_base[i] += startup_unavailable[i];
            }

            std::vector<size_t> mmproj_by_device(margins_base.size(), 0);
            auto mparams_gpu = make_mmproj_params(true);
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams_gpu);
            for (auto & [dev, size] : mmproj_mem) {
                for (size_t i = 0; i < tgt_devices.size() && i < mmproj_by_device.size(); ++i) {
                    if (tgt_devices[i] == dev) {
                        mmproj_by_device[i] += size;
                        break;
                    }
                }
            }

            if (std::all_of(mmproj_by_device.begin(), mmproj_by_device.end(),
                            [](size_t bytes) { return bytes == 0; })) {
                std::error_code ec;
                const auto fsize = std::filesystem::file_size(mmproj_path, ec);
                if (!ec && fsize > 0) {
                    // The GPU backend chosen by mtmd is the first available GPU.  If its dry
                    // measurement failed, conservatively charge the fallback to the first target
                    // device rather than multiplying one projector across every split device.
                    mmproj_by_device[0] = fsize + 256ull * 1024 * 1024;
                }
            }

            auto fit_swap_step = [&](const std::vector<size_t> & margins,
                                     std::vector<size_t>       & margins_needed,
                                     uint32_t                  & n_ctx_fit,
                                     size_t                    & total_mtp) {
                std::vector<size_t> margins_work = margins;
                auto mparams_fit = common_model_params_to_llama(params_base);
                auto cparams_fit = common_context_params_to_llama(params_base);

                const auto fit_status = common_fit_params(
                        params_base.model.path.c_str(), &mparams_fit, &cparams_fit,
                        params_base.tensor_split,
                        params_base.tensor_buft_overrides.data(),
                        &params_base.moe_cache,
                        margins_work.data(),
                        params_base.fit_params_min_ctx,
                        params_base.verbosity >= 4 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR,
                        params_base.speculative.draft.draft_vocab_resident_bytes);
                if (fit_status != COMMON_PARAMS_FIT_STATUS_SUCCESS || cparams_fit.n_ctx == 0) {
                    return false;
                }

                common_params params_mtp = params_base;
                params_mtp.n_parallel = n_parallel_user;
                params_mtp.n_ctx = cparams_fit.n_ctx;
                params_mtp.cache_type_k = params_base.speculative.draft.cache_type_k;
                params_mtp.cache_type_v = params_base.speculative.draft.cache_type_v;

                auto cparams_mtp = common_context_params_to_llama(params_mtp);
                cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                cparams_mtp.n_rs_seq = 0;
                cparams_mtp.n_outputs_max = n_parallel_user;

                std::vector<ggml_backend_dev_t> devs_mtp;
                uint32_t hp_ngl_mtp = 0;
                uint32_t hp_nct_mtp = 0;
                uint32_t hp_nex_mtp = 0;
                const auto dmd_mtp = common_get_device_memory_data(
                        params_mtp.model.path.c_str(), &mparams_fit, &cparams_mtp,
                        devs_mtp, hp_ngl_mtp, hp_nct_mtp, hp_nex_mtp, GGML_LOG_LEVEL_ERROR);

                std::vector<size_t> mtp_by_device(margins_base.size(), 0);
                total_mtp = 0;
                for (size_t j = 0; j < devs_mtp.size(); ++j) {
                    const size_t bytes = dmd_mtp[j].context + dmd_mtp[j].compute;
                    total_mtp += bytes;
                    for (size_t i = 0; i < tgt_devices.size() && i < mtp_by_device.size(); ++i) {
                        if (tgt_devices[i] == devs_mtp[j]) {
                            mtp_by_device[i] += bytes;
                            break;
                        }
                    }
                }

                margins_needed = margins_base;
                for (size_t i = 0; i < margins_needed.size(); ++i) {
                    margins_needed[i] += std::max(mtp_by_device[i], mmproj_by_device[i]);
                }
                n_ctx_fit = cparams_fit.n_ctx;
                return true;
            };

            std::vector<size_t> margins_trial = margins_base;
            uint32_t n_ctx_prev = 0;
            uint32_t n_ctx_selected = 0;
            bool converged = false;

            constexpr int SWAP_FIT_MAX_ITERS = 8;
            for (int iter = 0; iter < SWAP_FIT_MAX_ITERS; ++iter) {
                std::vector<size_t> margins_next;
                uint32_t n_ctx_fit = 0;
                size_t total_mtp = 0;
                if (!fit_swap_step(margins_trial, margins_next, n_ctx_fit, total_mtp)) {
                    SRV_WRN("mmproj GPU swap fit stopped at iteration %d: target fit failed\n", iter + 1);
                    break;
                }

                SRV_DBG("mmproj GPU swap fit %d: n_ctx=%u, MTP context+compute=%.2f MiB, "
                        "device 0 margin=%.2f MiB\n",
                        iter + 1, n_ctx_fit, total_mtp / (1024.0 * 1024.0),
                        margins_next[0] / (1024.0 * 1024.0));

                n_ctx_selected = n_ctx_fit;
                if (margins_next == margins_trial) {
                    converged = true;
                    break;
                }
                if (n_ctx_fit == n_ctx_prev) {
                    for (size_t i = 0; i < margins_trial.size(); ++i) {
                        margins_trial[i] = std::max(margins_trial[i], margins_next[i]);
                    }
                    converged = true;
                    break;
                }

                n_ctx_prev = n_ctx_fit;
                margins_trial = std::move(margins_next);
            }

            bool validated = false;
            constexpr int SWAP_FIT_VALIDATE_ITERS = 4;
            for (int iter = 0; iter < SWAP_FIT_VALIDATE_ITERS; ++iter) {
                std::vector<size_t> margins_needed;
                uint32_t n_ctx_fit = 0;
                size_t total_mtp = 0;
                if (!fit_swap_step(margins_trial, margins_needed, n_ctx_fit, total_mtp)) {
                    break;
                }
                validated = true;
                for (size_t i = 0; i < margins_trial.size(); ++i) {
                    if (margins_trial[i] < margins_needed[i]) {
                        margins_trial[i] = margins_needed[i];
                        validated = false;
                    }
                }
                n_ctx_selected = n_ctx_fit;
                SRV_DBG("mmproj GPU swap validation %d: n_ctx=%u, MTP context+compute=%.2f MiB%s\n",
                        iter + 1, n_ctx_fit, total_mtp / (1024.0 * 1024.0),
                        validated ? " (covered)" : " (margin raised)");
                if (validated) {
                    break;
                }
            }

            if (!validated || n_ctx_selected == 0) {
                SRV_ERR("%s", "mmproj GPU swap: could not find a safe target/auxiliary fit\n");
                return false;
            }

            params_base.fit_params_target = std::move(margins_trial);
            params_base.n_ctx = n_ctx_selected;
            params_base.fit_params = false;
            SRV_INF("mmproj GPU swap fit %s and validated: n_ctx=%u, device 0 margin=%.2f MiB\n",
                    converged ? "converged" : "bounded", params_base.n_ctx,
                    params_base.fit_params_target[0] / (1024.0 * 1024.0));
            SRV_INF("mmproj GPU swap fit: device 0 startup-unavailable charge=%.2f MiB\n",
                    startup_unavailable[0] / (1024.0 * 1024.0));
        }

        // Always enable kv_unified for single-slot servers — simplifies CUDA graph topology,
        // giving ~28% faster prompt eval even without speculative decoding.
        if (n_parallel_user == 1 && !params_base.kv_unified) {
            params_base.kv_unified = true;
            SRV_INF("%s", "auto-enabled kv-unified: single-slot server doesn't need separate KV stream\n");
        }

        // Double n_parallel only when actual speculative decoding is active
        // (draft model, MTP, or model-free self-speculation), not for phantom
        // --spec-type draft without -md. Model-free types verify their drafts through
        // the TARGET context too, so hybrid/recurrent targets need the same backup
        // sequence for partial-accept rollback: without the doubling n_seq_max stays at
        // n_parallel_user and llama_memory_recurrent::seq_cp silently no-ops on the
        // out-of-range backup seq — rollback then WIPES the recurrent state instead of
        // restoring it (#74: output stays plausible but wrong, degrading over time).
        if (params_base.speculative.has_dft() ||
            params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP) ||
            params_base.speculative.has_model_free_type()) {
            params_base.n_parallel = n_parallel_user * 2;
            n_seq_max_full = params_base.n_parallel;
            recurrent_expansion.state = server_recurrent_expansion_state::contracted;

            // The backup sequences double n_seq_max. Without unified KV each sequence gets its
            // own n_ctx / n_seq_max stream, so the doubling silently HALVES every slot's usable
            // context on top of the user's n_parallel division (-np 2 + MTP -> n_ctx/4 per
            // slot). Unified KV shares one full-length cell pool across sequences instead —
            // same auto-enable precedent as the single-slot case above.
            if (!params_base.kv_unified) {
                params_base.kv_unified = true;
                SRV_INF("%s", "auto-enabled kv-unified: speculative decoding doubles n_seq_max, "
                        "which would halve the per-slot context with per-sequence KV streams\n");
            }

            // n_outputs_max was computed above (server_n_outputs_max) with the pre-doubling
            // n_parallel. The target context is created just below with the DOUBLED n_parallel,
            // so its n_seq_max grows accordingly and output_reserve(n_seq_max) needs the cap to
            // cover it. For spec types whose n_outputs_per_seq is 1 (notably the fork DFlash
            // type, which returns n_max==0), the original cap stayed at n_parallel_user and
            // tripped GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) in output_reserve.
            // Recompute with the doubled n_parallel so the cap tracks the real n_seq_max.
            params_base.n_outputs_max = server_n_outputs_max(params_base);

            // The fork DFlash drafter verifies a full block (up to block_size tokens, typ. 16,
            // plus ddtree branches) against the TARGET per step, so the target's output_reserve
            // peaks well above n_seq_max. But common_speculative_n_max is 0 for the fork DFlash
            // type and block_size isn't known until the draft model loads (which happens after
            // this target context is created), so server_n_outputs_max undercounts here.
            // Reserve a generous per-sequence output budget as a floor. The cap only bounds
            // output_reserve's lazily-grown buffers and is clamped to n_batch — the same
            // ceiling a non-speculative context gets by default — so it costs nothing until
            // a batch actually requests that many outputs.
            constexpr int32_t DFLASH_VERIFY_OUTPUTS_PER_SEQ = 32; // ~2x block_size headroom
            const int32_t dflash_verify_floor = std::min<int32_t>(
                (int32_t) params_base.n_batch, (int32_t) n_seq_max_full * DFLASH_VERIFY_OUTPUTS_PER_SEQ);
            params_base.n_outputs_max = std::max<int32_t>(params_base.n_outputs_max, dflash_verify_floor);
        }

        // Native MTP contributes a context-dependent KV cache plus compute buffers after the
        // target fit.  Solve that dependency instead of measuring MTP at n_ctx_train and treating
        // the result as a fixed margin: the latter grossly over-reserves small fitted contexts,
        // while max(margin, MTP) spends the caller's safety margin on MTP itself.
        if (spec_mtp && !has_draft && params_base.fit_params) {
            const std::vector<size_t> margins_base = params_base.fit_params_target;
            std::vector<size_t> margins_trial = margins_base;
            std::vector<ggml_backend_dev_t> tgt_devices = params_base.devices;
            if (tgt_devices.empty()) {
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    tgt_devices.push_back(ggml_backend_dev_get(i));
                }
            }

            auto fit_mtp_step = [&](const std::vector<size_t> & margins,
                                    std::vector<size_t>       & margins_needed,
                                    uint32_t                  & n_ctx_fit,
                                    size_t                    & total_mtp) {
                std::vector<size_t> margins_work = margins;
                common_params params_trial = params_base;
                auto mparams_trial = common_model_params_to_llama(params_trial);
                auto cparams_trial = common_context_params_to_llama(params_trial);

                const auto fit_status = common_fit_params(
                        params_trial.model.path.c_str(), &mparams_trial, &cparams_trial,
                        params_trial.tensor_split,
                        params_trial.tensor_buft_overrides.data(),
                        &params_trial.moe_cache,
                        margins_work.data(),
                        params_trial.fit_params_min_ctx,
                        params_trial.verbosity >= 4 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR,
                        params_trial.speculative.draft.draft_vocab_resident_bytes);
                if (fit_status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
                    return false;
                }

                common_params params_mtp = params_base;
                params_mtp.n_parallel = n_parallel_user;
                // A successful VBR fit deliberately leaves n_ctx == 0 when the full
                // trained context is reachable at the requested floor.  Preserve that
                // default for the MTP dry context; its metadata probe below reports the
                // concrete trained limit used by both contexts.
                params_mtp.n_ctx = cparams_trial.n_ctx;
                params_mtp.cache_type_k = params_base.speculative.draft.cache_type_k;
                params_mtp.cache_type_v = params_base.speculative.draft.cache_type_v;

                auto cparams_mtp = common_context_params_to_llama(params_mtp);
                cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                cparams_mtp.n_rs_seq = 0;
                cparams_mtp.n_outputs_max = n_parallel_user;

                std::vector<ggml_backend_dev_t> devs_mtp;
                uint32_t hp_ngl_mtp = 0;
                uint32_t hp_nct_mtp = 0;
                uint32_t hp_nex_mtp = 0;
                const auto dmd_mtp = common_get_device_memory_data(
                        params_mtp.model.path.c_str(), &mparams_trial, &cparams_mtp,
                        devs_mtp, hp_ngl_mtp, hp_nct_mtp, hp_nex_mtp, GGML_LOG_LEVEL_ERROR);
                if (hp_nct_mtp == 0) {
                    return false;
                }

                margins_needed = margins_base;
                total_mtp = 0;
                for (size_t j = 0; j < devs_mtp.size(); ++j) {
                    const size_t bytes = dmd_mtp[j].context + dmd_mtp[j].compute;
                    total_mtp += bytes;
                    for (size_t i = 0; i < tgt_devices.size() && i < margins_needed.size(); ++i) {
                        if (tgt_devices[i] == devs_mtp[j]) {
                            margins_needed[i] += bytes;
                            break;
                        }
                    }
                }
                n_ctx_fit = cparams_trial.n_ctx != 0 ? cparams_trial.n_ctx : hp_nct_mtp;
                return true;
            };

            uint32_t n_ctx_prev = 0;
            bool converged = false;

            constexpr int MTP_FIT_MAX_ITERS = 8;
            for (int iter = 0; iter < MTP_FIT_MAX_ITERS; ++iter) {
                std::vector<size_t> margins_next;
                uint32_t n_ctx_fit = 0;
                size_t total_mtp = 0;
                if (!fit_mtp_step(margins_trial, margins_next, n_ctx_fit, total_mtp)) {
                    SRV_WRN("[spec] MTP fit solve stopped at iteration %d: target fit failed\n", iter + 1);
                    break;
                }

                SRV_DBG("[spec] MTP fit solve %d: n_ctx=%u, MTP context+compute=%.2f MiB\n",
                        iter + 1, n_ctx_fit, total_mtp / (1024.0 * 1024.0));

                if (margins_next == margins_trial) {
                    converged = true;
                    break;
                }

                // A repeated context means the discrete fit result is stable even if a backend's
                // byte estimate jitters slightly.  Keep the larger component so the final real fit
                // cannot spend bytes that MTP needs.
                if (n_ctx_fit == n_ctx_prev) {
                    for (size_t i = 0; i < margins_trial.size(); ++i) {
                        margins_trial[i] = std::max(margins_trial[i], margins_next[i]);
                    }
                    converged = true;
                    break;
                }

                n_ctx_prev = n_ctx_fit;
                margins_trial = std::move(margins_next);
            }

            // Discrete layer/context choices can oscillate around the fixed point.  Validate the
            // selected side explicitly: the final target fit must leave at least base margin plus
            // the MTP bytes measured at that fit's own context.  Grow only deficient components.
            bool validated = false;
            constexpr int MTP_FIT_VALIDATE_ITERS = 4;
            for (int iter = 0; iter < MTP_FIT_VALIDATE_ITERS; ++iter) {
                std::vector<size_t> margins_needed;
                uint32_t n_ctx_fit = 0;
                size_t total_mtp = 0;
                if (!fit_mtp_step(margins_trial, margins_needed, n_ctx_fit, total_mtp)) {
                    break;
                }
                validated = true;
                for (size_t i = 0; i < margins_trial.size(); ++i) {
                    if (margins_trial[i] < margins_needed[i]) {
                        margins_trial[i] = margins_needed[i];
                        validated = false;
                    }
                }
                SRV_DBG("[spec] MTP fit validation %d: n_ctx=%u, MTP context+compute=%.2f MiB%s\n",
                        iter + 1, n_ctx_fit, total_mtp / (1024.0 * 1024.0),
                        validated ? " (covered)" : " (margin raised)");
                if (validated) {
                    break;
                }
            }

            if (!validated) {
                SRV_ERR("%s", "[spec] could not find a safe target/MTP fit\n");
                return false;
            }

            params_base.fit_params_target = std::move(margins_trial);
            SRV_INF("[spec] MTP fit solve %s and validated; device 0 target margin is %.2f MiB\n",
                    converged ? "converged" : "bounded",
                    params_base.fit_params_target[0] / (1024.0 * 1024.0));
        }

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);
        params_base.n_parallel = n_parallel_user;

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        // context creation can fail after the model loaded (e.g. KV/compute OOM) —
        // bail before anything derefs ctx_tgt
        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create context, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        // co-tenancy: the server runs more init-time decodes after common's warmup
        // (speculative compat probes, template detection) — keep the not-real-traffic
        // flag up for all of them; closed at the end of init
        llama_set_warmup(ctx_tgt, true);

        vocab = llama_model_get_vocab(model_tgt);

        needs_reeval = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
        is_diffusion = llama_model_is_diffusion(model_tgt);

        dflash_tape_ok = llama_dflash_tape_replay_available(ctx_tgt);
        if (needs_reeval && !dflash_tape_ok && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            SRV_INF("%s", "DFlash rollback: GPU tape replay unavailable (recurrent states on "
                    "host or multiple devices) — partial accepts re-decode the accepted tokens\n");
        }

        if (is_diffusion) {
            SRV_INF("%s", "diffusion model detected — enabling self-speculation\n");
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_draft) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;

            SRV_TRC("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            // common_base_params_to_speculative copies the draft-side model/devices/ngl/cache
            // types/threads/overrides AND strips the base params' default-on dynamic-VBR flags —
            // a raw params_base copy here used to arm a second dynamic-VBR context and trip the
            // one-marker-per-process co-tenancy guard, failing draft-context creation.
            auto resolved_dft = make_params_dft();
            auto & params_dft = resolved_dft.params;

            // the helper pins n_outputs_max to the base n_parallel (MTP-path semantics);
            // this path historically inherited the base value — keep that (0 = derive from
            // n_batch), a DFlash drafter emits logits for whole blocks
            params_dft.n_outputs_max = params_base.n_outputs_max;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx_tgt) : params_spec.n_ctx;
            params_dft.n_batch      = params_dft.n_ctx;

            auto mparams_dft = common_model_params_to_llama(
                    params_dft, common_model_role::speculative_child);

            // progress callback
            mparams_dft.progress_callback           = load_progress_callback;
            mparams_dft.progress_callback_user_data = &load_progress_spec;

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            // Auto-detect DFlash from drafter model architecture. This tree carries TWO
            // DFlash implementations: the fork DeltaNet cross-attention drafter (arches
            // dflash-draft / gemma4-dflash-draft) and upstream's block-diffusion drafter
            // (arch dflash, which is also the DSpark backbone). Both report
            // dflash_block_size > 0, so discriminate on the architecture — it selects the
            // model graph, and each graph only works under its own driving protocol
            // (routing an upstream-format drafter into the fork impl leaves it with 0
            // capture layers and it silently never drafts). For arch dflash, a sidecar
            // carrying the DSpark Markov head must run the anchor-first DSpark protocol
            // (mirrors the download-plan rule: "dspark outranks dflash"); markov-less
            // sidecars run plain block-diffusion.
            if (llama_model_dflash_block_size(model_dft.get()) > 0 &&
                params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH &&
                !params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) &&
                !params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
                char arch_buf[128] = {};
                llama_model_meta_val_str(model_dft.get(), "general.architecture", arch_buf, sizeof(arch_buf));
                const std::string arch_dft = arch_buf;

                if (arch_dft == "dflash-draft" || arch_dft == "gemma4-dflash-draft") {
                    params_base.speculative.set_type(COMMON_SPECULATIVE_TYPE_DFLASH);
                    SRV_INF("auto-detected DFlash drafter (block_size=%d)\n",
                            llama_model_dflash_block_size(model_dft.get()));
                } else if (llama_model_dspark_has_markov_head(model_dft.get())) {
                    params_base.speculative.set_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK);
                    SRV_INF("auto-detected upstream DSpark drafter (block_size=%d)\n",
                            llama_model_dflash_block_size(model_dft.get()));
                } else {
                    params_base.speculative.set_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
                    SRV_INF("auto-detected upstream block-diffusion DFlash drafter (block_size=%d)\n",
                            llama_model_dflash_block_size(model_dft.get()));
                }
            }

            // Architecture-based detection happens after draft parameter resolution.
            // Refresh the semantic placement flag now that DSpark is known.
            resolved_dft.cpu_dspark_backbone = server_has_cpu_dspark_backbone(params_base);

            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                const int block_size = llama_model_dflash_block_size(model_dft.get());
                params_dft.n_ubatch = LLAMA_DFLASH_MAX_SLOTS * block_size;
                params_dft.n_parallel = std::max(1,
                    std::min(params_base.speculative.dflash_max_slots, params_base.n_parallel));

                // --spec-dflash-default leaves draft-max at -1 = auto: the drafter emits
                // at most block_size - 1 tokens per step and the full depth strictly wins
                // (EXP-37i depth sweep) — resolve it here so slot task defaults see it
                if (params_base.speculative.n_max < 0) {
                    params_base.speculative.n_max = block_size > 1 ? block_size - 1 : 12;
                    SRV_INF("draft-max auto (DFlash): %d (drafter block_size %d)\n",
                            params_base.speculative.n_max, block_size);
                }
            }

            if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
                params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
                // the shared multi-seq speculative state addresses drafter sequences by
                // slot id, so the drafter context needs one sequence per server slot
                params_dft.n_parallel = params_base.n_parallel;

                // phase C single-graph fused cycle (GGML_DFLASH_ONEGRAPH, default on):
                // reserve one extra drafter seq for the fused decode's padding rows and
                // keep the drafter KV unified so the mixed-seq fused batch stays a
                // single ubatch (n_stream must be 1). The impl enables the fused path
                // only when it sees the spare seq, so ONEGRAPH=0 also skips this.
                // DSV4-backbone drafters carry their own kill switch on top
                // (GGML_DFLASH_ONEGRAPH_DSV4=0 restores their two-decode path — do not
                // reshape their cache then); stage-init failures (CPU/TP targets) are
                // not knowable here and leave a harmless unused seq.
                {
                    const char * env_og   = getenv("GGML_DFLASH_ONEGRAPH");
                    const char * env_og4  = getenv("GGML_DFLASH_ONEGRAPH_DSV4");
                    const bool   oneg     = !(env_og  && atoi(env_og)  == 0);
                    const bool   oneg4    = !(env_og4 && atoi(env_og4) == 0);
                    if (oneg && !resolved_dft.cpu_dspark_backbone &&
                        (oneg4 || !llama_model_dflash_dsv4_backbone(model_dft.get()))) {
                        params_dft.n_parallel += 1;
                        params_dft.kv_unified  = true;
                    }
                }

                // draft depth: keep upstream's draft.n_max default (3) / the user's
                // --draft-max — on Muse-Glimmer the shallow depth clearly beats the
                // full trained block (71.5 t/s @3 vs 66.8 @15 vs 62.2 @7; acceptance
                // falls 68% -> 22% with depth), unlike the fork DeltaNet impl where
                // full depth wins (EXP-37i). Only resolve a negative auto value.
                // DSpark's anchor-first layout yields a full block_size of drafts
                // (block-diffusion yields block_size - 1).
                if (params_base.speculative.draft.n_max < 0) {
                    const int block_size = llama_model_dflash_block_size(model_dft.get());
                    const int depth_max  = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
                                         ? block_size : block_size - 1;
                    params_base.speculative.draft.n_max = depth_max > 0 ? depth_max : 12;
                }
                if (params_base.speculative.n_max < 0) {
                    params_base.speculative.n_max = params_base.speculative.draft.n_max;
                }

                // Bound the draft context's compute buffers. The drafter's KV must span
                // the target prompt (features are injected at absolute positions, so
                // n_ctx stays inherited), but no single drafter decode ever carries more
                // than one speculative cycle's rows: draft() batches every drafting
                // sequence's [anchor + noise block] and process() chunks prefill
                // injection by n_ubatch. Inheriting n_batch = n_ctx blew the ubatch up
                // to the full target context (LLM_ARCH_DFLASH used to hit the encoder
                // n_ubatch clamp), sizing pp compute buffers at ~1 MiB/token — ~8.5 GiB
                // at -c 8192 and an aborted ggml_backend_sched_new at -c 131072.
                {
                    const int32_t block_size = llama_model_dflash_block_size(model_dft.get());
                    // floor: one full cycle for every sequence in a single non-causal
                    // decode (draft() emits at most block_size + 1 rows per sequence)
                    const int32_t n_rows_cycle = params_dft.n_parallel * (std::max(block_size, 1) + 1);

                    int32_t n_ubatch_cap = 512;
                    if (const char * env = getenv("GGML_DFLASH_DRAFT_UBATCH")) {
                        const int v = atoi(env);
                        if (v > 0) {
                            n_ubatch_cap = (int32_t) v;
                        }
                    }

                    params_dft.n_ubatch = std::max(n_rows_cycle, std::min(params_dft.n_ubatch, n_ubatch_cap));
                    params_dft.n_batch  = params_dft.n_ubatch;

                    SRV_INF("draft ctx buffers bounded: n_ubatch %d, n_batch %d (compute buffers scale with "
                            "ubatch; drafter prefill chunks by ubatch; KV spans target ctx %d; "
                            "override cap with GGML_DFLASH_DRAFT_UBATCH)\n",
                            params_dft.n_ubatch, params_dft.n_batch, params_dft.n_ctx);
                }
            }

            // The fork DeltaNet drafter doesn't need the target's full context (it has no
            // KV cache — hidden states ride the cross ring): its buffers cost ~1 MiB/token,
            // so inheriting a large -c wastes GiBs and OOMs outright at 16k (measured
            // 2026-08-10, crosskv A/B). Auto-detected fork drafters get the same small
            // default the --spec-dflash-default handler applies in arg.cpp; an explicit
            // -cd N still wins (it lands in params_spec.n_ctx).
            // Do NOT apply this to the upstream draft-dflash/draft-dspark impls: they
            // inject target features into the drafter KV at absolute prompt positions, so
            // the drafter context must span the target prompt — at -cd 256 any prompt
            // past ~256 tokens fails decode ("failed to find a memory slot") and the
            // request 500s. Their per-token drafter KV is small (DSpark: 3 MLA layers,
            // ~3.4 KiB/token), so inheriting -c is cheap there.
            if (params_spec.n_ctx == 0 &&
                params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                params_dft.n_ctx   = 256;
                params_dft.n_batch = std::max(params_dft.n_ctx, params_dft.n_ubatch);
                SRV_INF("draft ctx auto (DFlash): %d (drafter doesn't need the full main ctx; pass -cd N to override)\n",
                        params_dft.n_ctx);
            }

            params_base.speculative.model_dft = model_dft.get();
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);
            // share buffers with the target context (upstream #24922 family)
            params_base.speculative.cparams_dft.ctx_other = ctx_tgt;

            // The upstream draft-dflash/draft-dspark graphs also read the target's
            // tok_embd/output — through the ctx_other fallback, which references the
            // target tensors directly. A -sm layer target puts output.weight on the
            // last GPU, and a drafter pinned elsewhere (--spec-draft-device) then
            // aborts at sched split ("pre-allocated tensor in a buffer that cannot
            // run the operation"). Route them through the same share/gather helper
            // the fork DFlash type uses: it shares by pointer when the drafter can
            // schedule the tensor (identical graphs to the ctx_other fallback) and
            // gathers a drafter-device copy (one INFO log) when it cannot.
            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH ||
                params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
                params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
                llama_model_share_tensors(model_dft.get(), llama_get_model(ctx_tgt));
            }

            // Upstream block-diffusion DFlash / DSpark: create the drafter context here
            // (it shares tok_embd/output with the target through cparams_dft.ctx_other)
            // and wire the draft params so the shared speculative init below picks the
            // draft-dflash / draft-dspark implementation.
            if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
                params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
                params_base.speculative.cparams_dft.n_rs_seq = 0;
                ctx_dft.reset(llama_init_from_model(model_dft.get(), params_base.speculative.cparams_dft));
                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create DFlash draft context\n");
                    return false;
                }
                ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft.get();
            }

            // Upstream MTP: create draft context from target model's MTP heads
            if (spec_mtp && params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
                auto cparams = common_context_params_to_llama(params_dft);
                cparams.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
                cparams.n_rs_seq  = 0;
                cparams.ctx_other = ctx_tgt;
                ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create draft context\n");
                    return false;
                }
                ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft.get();
            }
        } else if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) {
            // no new model load, so we simply report 0.0 and 1.0 progress
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            SRV_INF("creating MTP draft context against the target model '%s'\n",
                    params_base.model.path.c_str());

            ctx_dft.reset(create_mtp_context());
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mmproj_gpu_swap = params_base.mmproj_gpu_swap && !model_dft;
            // Note: ctx_dft is NOT required here. Without MTP, the swap simply
            // keeps mmproj on CPU until an image arrives, loads it to GPU for
            // encoding, then moves it back. No MTP state to swap out.

            const bool use_gpu = mmproj_gpu_swap ? false : params_base.mmproj_use_gpu;
            auto mparams = make_mmproj_params(use_gpu, load_progress_callback, &load_progress_mmproj);
            if (!mmproj_gpu_swap) {
                mparams.warmup = params_base.warmup;
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }

            if (mmproj_gpu_swap) {
                SRV_INF("loaded multimodal model on CPU (GPU swap enabled), '%s'\n", mmproj_path.c_str());
            } else {
                SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());
            }

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }

        }

        // cache receipt (§7.7 security contract): keyed by default; an unkeyed
        // chain leaks prompt-content comparability and requires the explicit
        // debug flag. Fail closed at startup, never silently downgrade.
        if (params_base.cache_receipt &&
            params_base.cache_receipt_key.empty() &&
            !params_base.cache_receipt_unkeyed_debug) {
            SRV_ERR("%s\n", "--cache-receipt requires --cache-receipt-key "
                    "(or --cache-receipt-unkeyed-debug for trusted local use)");
            return false;
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        // cache_reuse splices the ATTENTION KV to close the gap left by a mid-prompt edit, but the
        // recurrent state is a running frontier the splice does not correct: afterwards it still
        // encodes the pre-splice token sequence, and the position-only checkpoint selector would
        // pair that stale recurrent state with the spliced attention timeline -> silent wrong state
        // [I12/N2]. llama_memory_can_shift() above reports only the ATTENTION child's capability for
        // hybrid models (recurrent get_can_shift() returns true unconditionally), so the gate does
        // not catch this. Disable cache_reuse for any recurrent/hybrid model.
        if (params_base.n_cache_reuse &&
            (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt))) {
            params_base.n_cache_reuse = 0;
            SRV_WRN("%s\n", "cache_reuse is not supported by recurrent/hybrid models (the recurrent state cannot be spliced), it will be disabled");
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();
        frontier_execution_identity =
            "server-execution-v1:" + random_string();
        frontier_next_sequence_epoch = 1;
        frontier_ratchet_threshold =
            server_frontier_ratchet_min_agreements();
        SRV_INF(
            "computation-frontier ratchet: legacy reads authoritative, "
            "flip after %" PRIu64 " qualifying agreements%s\n",
            frontier_ratchet_threshold,
            frontier_ratchet_threshold == 0 ? " (shadow-only)" : "");

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        const bool can_spec = (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO);

        // DFlash multi-slot: --dflash-max-slots caps how many server slots keep DFlash state;
        // slots above the cap fall back to non-speculative decode (slot.spec stays null). The
        // matching tape/hidden buffers are allocated after the per-slot init loop (set_dflash_capture
        // runs inside common_speculative_init for slot 0, so dflash_capture must exist first).
        int dflash_slots_cap = 0;
        if (can_spec && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            dflash_slots_cap = std::max(1, std::min(params_base.speculative.dflash_max_slots, params_base.n_parallel));
            if (dflash_slots_cap < params_base.n_parallel) {
                SRV_INF("DFlash enabled for slots 0..%d; slots %d+ will use non-speculative decode\n",
                        dflash_slots_cap - 1, dflash_slots_cap);
            } else {
                SRV_INF("DFlash enabled for all %d slots\n", dflash_slots_cap);
            }

            // Create the shared DFlash drafter context once;
            // every slot's common_speculative gets a non-owning reference to it.
            // dflash_slots_cap is passed at init so the initial graph reserve sizes
            // the compute buffer for the requested width — single-slot servers stay
            // narrow (cheap), multi-slot servers get a compute buffer big enough for
            // the batched cross-attention. Runtime widening past this cap requires a
            // larger compute buffer than is available.
            ctx_dft_shared.reset(common_speculative_create_ctx_dft(params_base.speculative, dflash_slots_cap));
            if (!ctx_dft_shared) {
                SRV_ERR("%s", "failed to create shared DFlash drafter context\n");
                return false;
            }
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot, params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding (upstream shared spec — not used by fork types which init per-slot)
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO && !dflash_slots_cap) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            // fork: only drop the (possibly-null) shared MTP/draft ctx here. Do NOT free
            // model_dft — in pure-DFlash mode the shared `spec` never inits so this else
            // always runs, yet params_base.speculative.model_dft and every slot's drafter
            // state still point at model_dft (share_tensors). Freeing it = UAF on first draft.
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.n_ctx   = n_ctx_slot;
            slot.frontier_ratchet_min_agreements =
                frontier_ratchet_threshold;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            const bool slot_can_spec = can_spec &&
                (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH || i < dflash_slots_cap);

            if (is_diffusion) {
                slot.diff_self_spec = true;
                slot.diff_mask_token_id = llama_vocab_mask(vocab);
                slot.diff_draft_length = 4;
                auto think_open  = common_tokenize(vocab, "<think>",  false, true);
                auto think_close = common_tokenize(vocab, "</think>", false, true);
                if (think_open.size()  == 1) slot.diff_think_open_id  = think_open[0];
                if (think_close.size() == 1) slot.diff_think_close_id = think_close[0];
                SLT_INF(slot, "diffusion self-spec enabled, mask_id=%d, draft_length=%d, think_suppress=%d/%d\n",
                        slot.diff_mask_token_id, slot.diff_draft_length, slot.diff_think_open_id, slot.diff_think_close_id);
            } else if (slot_can_spec) {
                slot.spec.reset(common_speculative_init(params_base.speculative, slot.ctx_tgt, ctx_dft_shared.get()));
                if (!slot.spec && spec) {
                    slot.spec_shared = spec.get();
                }
                if (slot.can_speculate()) {
                    common_speculative_set_seq_id(slot.get_spec(), slot.id);
                    SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                }
            }

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        // safety net (#74): hybrid/recurrent partial-accept rollback requires one backup
        // sequence per user slot. If a future speculative type slips past the n_parallel
        // doubling above, refuse to speculate instead of silently corrupting the
        // recurrent state on the first partially-accepted draft.
        if (needs_reeval && ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
            llama_n_seq_max(ctx_tgt) < (uint32_t) (2 * n_parallel_user)) {
            bool any_spec = spec != nullptr;
            for (auto & slot : slots) {
                any_spec = any_spec || slot.can_speculate();
                slot.spec.reset();
                slot.spec_shared = nullptr;
            }
            spec.reset();
            if (any_spec) {
                SRV_ERR("speculative decoding disabled: hybrid/recurrent rollback needs n_seq_max >= %d, context has %u\n",
                        2 * n_parallel_user, llama_n_seq_max(ctx_tgt));
            }
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        // Allocate DFlash per-slot tape + hidden buffers now that common_speculative_init
        // (run for slot 0 above) has created dflash_capture on the target context.
        // Runs even when !dflash_tape_ok: the call also sizes the per-slot hidden-capture
        // buffers, which every DFlash target needs (the unused tape itself is small).
        if (dflash_slots_cap > 0) {
            llama_dflash_allocate_slots(ctx_tgt, dflash_slots_cap);
        }


        // Under --split-mode tensor the load-time capability probe was predictive; now
        // that capture setup exists the tape allocation (with its shard-consistency
        // check) has run, so re-probe for the authoritative answer.
        if (dflash_tape_ok && !llama_dflash_tape_replay_available(ctx_tgt)) {
            dflash_tape_ok = false;
            SRV_INF("%s", "DFlash rollback: GPU tape rejected after allocation (shard "
                    "mismatch) — partial accepts re-decode the accepted tokens\n");
        }

        // DFlash + hybrid target: expand the recurrent state to its full backup-cell count
        // NOW instead of at the first draft — the expand costs ~100 ms of buffer realloc +
        // CUDA-graph re-capture, which otherwise lands on the first request's latency.
        // (recurrent_shrink_for_prefill never shrinks DFlash, so this is a one-time cost.)
        if (needs_reeval && can_spec &&
            params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
            ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
            n_seq_max_full > n_parallel_user &&
            recurrent_expansion.state == server_recurrent_expansion_state::contracted) {
            auto * mem = llama_get_memory(ctx_tgt);
            const bool expanded = llama_memory_recurrent_expand(mem, n_seq_max_full);
            recurrent_expansion.complete_expand(expanded);
            if (expanded) {
                SRV_INF("pre-expanded recurrent state to %d cells for speculative backup\n", n_seq_max_full);
            } else {
                SRV_ERR("failed to pre-expand recurrent state to %d cells; deferring speculation until the next request\n",
                        n_seq_max_full);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (server_vbr_dynamic_active(params_base)) {
            // Disabled: these mechanisms serialize/shift the ATTENTION KV, whose tensor tiers flip
            // in place at runtime under the dynamic VBR controller (a FLAGS_NONE state restore
            // would land bytes under the wrong tier or past the VMM watermark; cache_reuse needs
            // shifts, and get_can_shift() is false under VMM anyway).
            if (params_base.cache_ram_mib != 0) {
                params_base.cache_ram_mib = 0;
                SRV_WRN("%s\n", "prompt cache state storage is not supported by dynamic VBR (KV tiers change at runtime), it will be disabled");
            }
            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by dynamic VBR (KV tiers change at runtime), it will be disabled");
            }
            if (!params_base.slot_save_path.empty()) {
                // llama_state_seq_save_file carries full tier-typed attention KV: a save taken
                // after any degrade is refused at the lib level, and even entry-tier saves stop
                // restoring once the target cache has degraded — predictably off beats flaky
                params_base.slot_save_path.clear();
                SRV_WRN("%s\n", "slot save/restore (--slot-save-path) is not supported by dynamic VBR (KV tiers change at runtime), it will be disabled");
            }
            // Context checkpoints (PARTIAL_ONLY):
            //   ordinary hybrid (swa_type == NONE, n_swa == 0) — routed to the recurrent state only
            //     (attention KV skipped, see llama_memory_hybrid::state_write): tier-agnostic and
            //     load-bearing for prompt rewind, so they stay ENABLED (the n_swa > 0 gate is false);
            //   ANY SWA model (n_swa > 0), INCLUDING hybrid+iSWA — the SWA attention KV IS serialized
            //     under PARTIAL_ONLY (llama_kv_cache_iswa::state_write, reached for pure iSWA and, via
            //     llama_memory_hybrid_iswa, for hybrid+iSWA too), so once a tier flips every restore
            //     would fail: disable them. The previous `!is_hybrid` term wrongly exempted the
            //     hybrid+iSWA case, whose checkpoints carry tier-sensitive attention bytes. [I8]
            if (params_base.n_ctx_checkpoints > 0 &&
                llama_model_n_swa(model_tgt) > 0) {
                params_base.n_ctx_checkpoints = 0;
                SRV_WRN("%s\n", "context checkpoints are not supported by dynamic VBR on SWA models (the SWA attention KV is part of the checkpoint), they will be disabled");
            }
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        // P2 F0b authority substrate: constructed and configured under
        // (cache_debug || cache_lifecycle). Neither flag remains the strictly-zero-work legacy
        // path; debug alone observes the shared substrate, while lifecycle enables publication
        // admission without allocating or emitting a cache-plan observer.
        if (params_base.cache_debug) {
            cache_plan_obs = std::make_unique<server_cache_plan_observer>();
        }
        if (params_base.cache_debug || params_base.cache_lifecycle) {
            cache_authority = std::make_unique<server_cache_authority>();
        }

        // B0 shadow observer [P2]: constructed only under params_base.cache_debug — the observer
        // object and both surfaces exist iff the flag is set. Off = strictly zero observer work
        // (B-a). It records the shared authority substrate above (owned by server_context).
        const auto plan_authority_level = params_base.cache_plan_authority;
        if (params_base.cache_debug ||
            plan_authority_level != common_cache_plan_authority_level::off) {
            cache_plan_authority =
                std::make_unique<server_cache_plan_authority>(plan_authority_level);
        }

        if (params_base.cache_debug || params_base.cache_lifecycle) {
            // Policy substrate is live under either gate. CACHE_PLAN JSON/log
            // emission remains below behind cache_plan_obs/cache_debug, but
            // lifecycle-only operation must still inspect WS-D leases before
            // every censused destructive primitive.
            cache_authority->destruction.lease_context = &cache_authority->leases;
            cache_authority->destruction.lease_evaluator =
                server_cache_lease_evaluate_request;
            for (auto & slot : slots) {
                slot.destruction_obs = &cache_authority->destruction;
                slot.retention_obs = &cache_authority->retention;
                slot.lease_obs = &cache_authority->leases;
                slot.lease_execution_identity = &frontier_execution_identity;
                slot.lifecycle_authority = params_base.cache_lifecycle
                    ? cache_authority.get()
                    : nullptr;
                slot.cache_debug_observability = params_base.cache_debug;
                slot.retention_pool =
                    (llama_model_is_recurrent(model_tgt) ||
                     llama_model_is_hybrid(model_tgt))
                        ? common_retention_pool::recurrent
                        : common_retention_pool::attention;
            }
            if (prompt_cache) {
                prompt_cache->debug_observability = params_base.cache_debug;
                prompt_cache->destruction_obs = &cache_authority->destruction;
                prompt_cache->retention_obs = &cache_authority->retention;
                prompt_cache->lease_obs = &cache_authority->leases;
                prompt_cache->lease_execution_identity =
                    &frontier_execution_identity;
            }
            if (params_base.cache_lifecycle) {
                llama_get_memory(ctx_tgt)->vbr_hard_seal_guard_set(
                    vbr_hard_seal_guard {
                        [this]() {
                            return cache_authority != nullptr &&
                                server_cache_has_hard_lease(
                                    &cache_authority->leases);
                        },
                        [this](const vbr_hard_seal_subject &,
                               const std::vector<vbr_hard_seal_range> & ranges) {
                            for (const auto & range : ranges) {
                                const auto slot = std::find_if(
                                    slots.begin(), slots.end(),
                                    [&](const server_slot & value) {
                                        return value.id == range.sequence;
                                    });
                                if (slot != slots.end() &&
                                    slot->hard_lease_blocks_live_range(
                                        range.first_token, range.token_count)) {
                                    return vbr_hard_seal_guard_result::hard_lease_blocked;
                                }
                            }
                            return vbr_hard_seal_guard_result::allow;
                        },
                    });
            }
        }

        std::vector<std::string> gpu_descs;
        std::vector<vbr_artifact_portable_topology>
            capture_topologies;
        std::vector<vbr_explicit_capture_runtime_pool>
            capture_runtime_pools;
        std::vector<vbr_explicit_capture_pool_binding>
            capture_pool_bindings;
        std::vector<vbr_capture_lane> capture_lanes;
        uint32_t capture_attention_children = 0;
        bool capture_manifest_enabled = false;
        const bool capture_requested =
            params_base.cache_lifecycle &&
            server_vbr_dynamic_active(params_base);
        int ngl_eff = 0;
        if (cache_authority) {
            const auto observer_domain = llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::not_applicable);
            const auto host_domain = llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host);

            // D-S1 manifest ordering: resolve the same effective placement used by the
            // calibration profile, build/intern every usable device domain, and retain its
            // runtime device binding BEFORE the one-shot producer manifest is configured.
            // Multi-device auto placement has no resolved weights here; the canonical
            // topology builder refuses it rather than fabricating equal shares.
            std::vector<ggml_backend_dev_t> gpu_devices;
            std::vector<std::string> gpu_identities;
            if (!params_base.devices.empty()) {
                // the explicit --device list is nullptr-TERMINATED (parse_device_list);
                // "none" is a single nullptr -> empty -> "cpu"
                for (auto * dev : params_base.devices) {
                    if (dev) {
                        gpu_devices.push_back(dev);
                        gpu_descs.push_back(ggml_backend_dev_description(dev));
                        gpu_identities.push_back(
                            std::string(ggml_backend_dev_name(dev)) + "\n" +
                            ggml_backend_dev_description(dev));
                    }
                }
            } else {
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    auto * dev = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        gpu_devices.push_back(dev);
                        gpu_descs.push_back(ggml_backend_dev_description(dev));
                        gpu_identities.push_back(
                            std::string(ggml_backend_dev_name(dev)) + "\n" +
                            ggml_backend_dev_description(dev));
                    }
                }
            }
            // Auto-fit writes the resolved layer count back into params. A remaining
            // negative sentinel means the loader's negative-is-all default.
            ngl_eff = params_base.n_gpu_layers >= 0
                ? params_base.n_gpu_layers
                : llama_model_n_layer(model_tgt) + 1;
            if (ngl_eff > 0 && !gpu_identities.empty()) {
                llama_cache_acct_shard_topology topology;
                if (llama_cache_acct_build_shard_topology(
                        gpu_identities,
                        int16_t(params_base.split_mode),
                        params_base.main_gpu,
                        params_base.tensor_split,
                        topology)) {
                    if (capture_requested) {
                        capture_topologies.push_back(topology);
                    }
                    for (size_t i = 0; i < gpu_identities.size(); ++i) {
                        llama_cache_acct_resource_domain domain;
                        if (!cache_authority->ledger.make_device_domain(
                                topology,
                                llama_cache_acct_device_ordinal{ uint16_t(i) },
                                domain)) {
                            cache_authority_config_failed(true);
                            cache_authority->live_device_domains.clear();
                            break;
                        }
                        cache_authority->live_device_domains.push_back({
                            gpu_devices[i], domain,
                        });
                    }
                } else {
                    cache_authority_config_failed(true);
                }
            }

            // F3.3 construction discovery is read-only and occurs before the
            // one-shot C manifest. Every attention child and every pool must
            // be armed and exactly attributable to a topology device.
            if (capture_requested &&
                cache_authority->configured &&
                capture_topologies.size() == 1 &&
                vbr_explicit_capture_runtime_pools(
                    *llama_get_memory(ctx_tgt),
                    capture_runtime_pools,
                    capture_attention_children)) {
                for (const auto & pool : capture_runtime_pools) {
                    const auto domain = std::find_if(
                        cache_authority->live_device_domains.begin(),
                        cache_authority->live_device_domains.end(),
                        [&](const auto & binding) {
                            return binding.device ==
                                pool.backend_device;
                        });
                    if (domain ==
                        cache_authority->live_device_domains.end()) {
                        capture_pool_bindings.clear();
                        capture_lanes.clear();
                        break;
                    }
                    auto lane = std::find_if(
                        capture_lanes.begin(), capture_lanes.end(),
                        [&](const auto & candidate) {
                            return candidate.device ==
                                pool.backend_device;
                        });
                    uint32_t lane_index = 0;
                    if (lane == capture_lanes.end()) {
                        ggml_backend_t ring_backend = pool.backend;
                        if (ring_backend == nullptr) {
                            ring_backend =
                                ctx_tgt->backend_for_device(
                                    pool.backend_device);
                            if (ring_backend == nullptr) {
                                capture_pool_bindings.clear();
                                capture_lanes.clear();
                                break;
                            }
                        }
                        lane_index = uint32_t(capture_lanes.size());
                        capture_lanes.push_back({
                            pool.backend_device, ring_backend, false,
                        });
                    } else {
                        lane_index = uint32_t(
                            lane - capture_lanes.begin());
                    }
                    capture_pool_bindings.push_back({
                        pool.instance_id,
                        pool.device,
                        0,
                        domain->domain.device_ordinal.v,
                        lane_index,
                    });
                }
                capture_manifest_enabled =
                    !capture_pool_bindings.empty() &&
                    !capture_lanes.empty() &&
                    capture_pool_bindings.size() ==
                        capture_runtime_pools.size();
            }

            // C schema-v2 completeness manifest is owned by configuration, not by either
            // producer. Device/live-memory rows are part of this SAME one-shot call; no
            // producer is allowed to append its own domain after cells exist.
            // Artifact cells and the pinned domain therefore precede ring-budget
            // admission. If that later admission refuses the ring, these manifested
            // cells remain certified measured-zero. This is intentional, harmless,
            // limited to lifecycle+armed VBR, and must not become a second manifest.
            std::vector<llama_cache_acct_completeness_requirement> required;
            required.reserve(
                4 + cache_authority->live_device_domains.size());
            required.push_back({
                observer_domain, llama_cache_acct_producer::observer_init,
            });
            if (prompt_cache) {
                required.push_back({
                    host_domain, llama_cache_acct_producer::host_cache,
                });
            }
            required.push_back({
                host_domain, llama_cache_acct_producer::retention_sidecar,
            });
            const auto pinned_domain =
                llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::pinned_host);
            if (capture_manifest_enabled) {
                required.push_back({
                    pinned_domain,
                    llama_cache_acct_producer::retention_sidecar,
                });
            }
            for (const auto & binding : cache_authority->live_device_domains) {
                required.push_back({
                    binding.domain, llama_cache_acct_producer::live_memory,
                });
            }
            if (!cache_authority->ledger.configure_required_producers(
                    required.data(), required.size())) {
                cache_authority_config_failed(false);
            }
            if (capture_manifest_enabled) {
                // Capture reservations span pageable metadata/companions and
                // topology-qualified device payloads. This runs BEFORE the
                // ordinary host/live producers below, so it co-initializes every
                // host-scope participating cell in this domain to measured-zero —
                // including the F0b host-cache cells (full_snapshot/checkpoint_state/
                // typed_accelerator) — so the host domain is complete when the
                // store's first host reservation prices it. The later F0b block
                // re-gauges those same cells to the same 0 (gauge_set is idempotent),
                // so this co-init is load-bearing here yet behavior-neutral for F0b.
                if (!server_vbr_artifact_store_observe_empty_accounting(
                        cache_authority->ledger, host_domain)) {
                    cache_authority_config_failed(true);
                }
                for (const auto & binding :
                        cache_authority->live_device_domains) {
                    if (!server_vbr_artifact_store_observe_empty_accounting(
                            cache_authority->ledger,
                            binding.domain)) {
                        cache_authority_config_failed(true);
                    }
                }
                if (!server_vbr_artifact_store_configure_pinned_accounting(
                        cache_authority->ledger, pinned_domain)) {
                    cache_authority_config_failed(true);
                }
            }
            cache_authority->retention.configure(
                &cache_authority->ledger, host_domain, &cache_authority->leases);
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::
                        artifact_descriptor_metadata,
                    host_domain, measure, 0);
            }

            // Host-cache absence is itself an observed zero. Initialize all three
            // transactional leaves before an optional cache attaches; later C
            // transactions update these same resident/reserved cells.
            for (const auto cat : {
                    llama_cache_acct_category::full_snapshot_payload,
                    llama_cache_acct_category::checkpoint_state_payload,
                    llama_cache_acct_category::typed_accelerator_payload }) {
                for (const auto measure : {
                        llama_cache_acct_measure::logical_payload,
                        llama_cache_acct_measure::resident_allocated,
                        llama_cache_acct_measure::reserved }) {
                    cache_authority->ledger.gauge_set(
                        cat, host_domain, measure, 0);
                }
            }
            if (prompt_cache) {
                // Empty-cache gauges are real observations. Certification follows them,
                // rather than merely proving that wiring code ran.
                (void) cache_authority->ledger.certify_complete(
                    host_domain, llama_cache_acct_producer::host_cache);
            }
            if (capture_manifest_enabled) {
                for (const auto category : {
                        llama_cache_acct_category::
                            artifact_reference_metadata,
                        llama_cache_acct_category::transfer_staging,
                        llama_cache_acct_category::codec_workspace }) {
                    for (const auto measure : {
                            llama_cache_acct_measure::logical_payload,
                            llama_cache_acct_measure::resident_allocated,
                            llama_cache_acct_measure::reserved }) {
                        cache_authority->ledger.gauge_set(
                            category, host_domain, measure, 0);
                    }
                }
                for (const auto & binding :
                        cache_authority->live_device_domains) {
                    for (const auto category : {
                            llama_cache_acct_category::unit_version_payload,
                            llama_cache_acct_category::clean_stash_payload,
                            llama_cache_acct_category::transfer_staging,
                            llama_cache_acct_category::codec_workspace }) {
                        for (const auto measure : {
                                llama_cache_acct_measure::logical_payload,
                                llama_cache_acct_measure::resident_allocated,
                                llama_cache_acct_measure::reserved }) {
                            cache_authority->ledger.gauge_set(
                                category, binding.domain, measure, 0);
                        }
                    }
                }
            }
            (void) cache_authority->ledger.certify_complete(
                host_domain,
                llama_cache_acct_producer::retention_sidecar);
            // the parked rolling-window tape's zero is OBSERVED, not implied: an explicit
            // measured-zero gauge, distinct from every unknown cell
            cache_authority->ledger.gauge_set(
                    llama_cache_acct_category::rolling_window_tape,
                    observer_domain,
                    llama_cache_acct_measure::logical_payload, 0);
            (void) cache_authority->ledger.certify_complete(
                observer_domain, llama_cache_acct_producer::observer_init);

            // Observe real resident allocations before certifying live_memory. A failed
            // attribution leaves every affected domain unavailable; a measured zero is
            // valid (e.g. no recurrent layers on an attention-only target).
            if (!cache_plan_observe_live_memory(true)) {
                cache_authority_config_failed(true);
            }
            if (!cache_plan_configure_budget()) {
                cache_authority_config_failed(true);
            }
            if (capture_manifest_enabled) {
                std::vector<llama_cache_acct_resource_domain>
                    capture_accounting_domains;
                capture_accounting_domains.reserve(
                    2 + cache_authority->live_device_domains.size());
                capture_accounting_domains.push_back(host_domain);
                capture_accounting_domains.push_back(pinned_domain);
                for (const auto & binding :
                        cache_authority->live_device_domains) {
                    capture_accounting_domains.push_back(
                        binding.domain);
                }
                if (!server_vbr_artifact_store_verify_accounting(
                        cache_authority->ledger,
                        capture_accounting_domains)) {
                    cache_authority_config_failed(true);
                }
            }

            // Establish an initial coherent budget object so lifecycle-only is a fully configured
            // authority, not merely a set of producer pointers. Each admission re-samples physical
            // capacity before composing, because free memory is time-sensitive.
            if (!cache_authority->sample_budget(cache_authority->budget_config) ||
                !cache_authority->budget.reset(
                    cache_authority->ledger.snapshot(),
                    cache_authority->budget_config)) {
                cache_authority_config_failed(false);
            }

            if (capture_manifest_enabled &&
                cache_authority->configured) {
                server_vbr_artifact_store_config config;
                config.ledger = &cache_authority->ledger;
                config.pinned_domain = pinned_domain;
                config.topologies = capture_topologies;
                config.pool_bindings = capture_pool_bindings;
                config.lanes = capture_lanes;
                config.attention_children =
                    capture_attention_children;
                static constexpr uint64_t RING_FLOOR =
                    64ull*1024*1024;
                static constexpr size_t RING_CHUNK =
                    8ull*1024*1024;
                const uint64_t lane_floor =
                    uint64_t(config.lanes.size())*2*RING_CHUNK;
                config.ring_bytes = std::min<uint64_t>(
                    VBR_CAPTURE_PINNED_RING_MAX_BYTES,
                    std::max<uint64_t>(RING_FLOOR, lane_floor));
                config.chunk_bytes = RING_CHUNK;
                config.turbo_meansub_id =
                    ctx_tgt->get_model().hparams.turbo_meansub_id;
                config.budget_context = cache_authority.get();
                config.sample_budget = [](
                        void * context,
                        llama_cache_budget_config & output) noexcept {
                    auto * authority =
                        static_cast<server_cache_authority *>(context);
                    if (!authority->sample_budget(output)) {
                        return false;
                    }
                    // The pinned ring and pageable artifact bytes share the
                    // same physical host headroom. Preserve the sampled
                    // pageable ceiling as a shared host-total constraint and
                    // leave the pinned leaf policy-unbounded within it.
                    output.host.total_cap = output.host.pageable_cap;
                    output.host.total_state =
                        output.host.pageable_state;
                    output.host.pinned_state =
                        llama_cache_budget_capacity_state::unbounded;
                    return true;
                };
                server_vbr_artifact_capture_status status;
                server_vbr_artifact_store_create_diagnostics diagnostics;
                vbr_artifact_store =
                    server_vbr_artifact_store::create(
                        config, status, &diagnostics);
                if (!vbr_artifact_store) {
                    SRV_WRN(
                        "VBR_ARTIFACT_CAPTURE store unavailable status=%s "
                        "failure=%s ring_status=%s ring_failure=%s "
                        "attention_children=%u lanes=%zu "
                        "requested_ring=%" PRIu64
                        " attempted_ring=%" PRIu64
                        " chunk=%zu\n",
                        server_vbr_artifact_capture_status_name(status),
                        server_vbr_artifact_store_create_failure_name(
                            diagnostics.failure),
                        vbr_capture_stream_status_name(
                            diagnostics.ring_status),
                        vbr_capture_ring_create_failure_name(
                            diagnostics.ring_failure),
                        diagnostics.attention_children,
                        diagnostics.lane_count,
                        diagnostics.requested_ring_bytes,
                        diagnostics.attempted_ring_bytes,
                        diagnostics.chunk_bytes);
                } else {
                    SRV_INF(
                        "VBR_ARTIFACT_CAPTURE store ready "
                        "attention_children=%u lanes=%zu "
                        "requested_ring=%" PRIu64
                        " constructed_ring=%" PRIu64
                        " chunk=%zu\n",
                        diagnostics.attention_children,
                        diagnostics.lane_count,
                        diagnostics.requested_ring_bytes,
                        diagnostics.constructed_ring_bytes,
                        diagnostics.chunk_bytes);
                }
            }

            if (prompt_cache) {
                prompt_cache->acct = &cache_authority->ledger;
                if (params_base.cache_lifecycle) {
                    prompt_cache->publish_authority =
                        cache_authority.get();
                }
            }
        }

        if (cache_plan_obs || cache_plan_authority || cache_authority ||
            params_base.cache_plan_preflight) {
            // B-2: compose the stable calibration-profile id ONCE. The model class comes
            // from loaded-model CONTENT (llama_model_desc: arch + params + quant class),
            // never a filesystem label — renaming a different file to the same basename
            // must not select fitted coefficients (verify-r1 finding 5). The hardware
            // class covers the COMPLETE GPU topology (per-description counts + split
            // mode), not just the first device. Composition/sanitize rule lives beside
            // the calib struct (single tested spelling).
            {
                char desc[256] = {0};
                llama_model_desc(model_tgt, desc, sizeof(desc));

                cache_plan_calibration_profile = common_cache_plan_calib_profile(
                    desc,
                    common_cache_plan_calib_hw(gpu_descs, ngl_eff,
                                               int(params_base.split_mode),
                                               params_base.main_gpu,
                                               params_base.tensor_split),
                    params_base.n_batch,
                    // a vbr side is a RUNTIME REGIME, not a ggml type (cache_type_k/v still
                    // hold the f16 entry tier), and the regime's COST depends on the whole
                    // ladder configuration — budget mode, aggregate floor, VRAM budget,
                    // policy — not just which sides took the alias. VBR can also arm from
                    // those knobs with no `-ct vbr` at all (common_params::vbr_enabled),
                    // which would otherwise key as plain f16. Sides name the regime; the
                    // regime signature disambiguates configurations within it.
                    // [D pins r1 finding: post-CLEAN follow-up 1]
                    [&] {
                        const auto vbr = common_cache_plan_vbr_regime_from_params(
                            params_base, [](const char * name) { return std::getenv(name); });
                        return common_cache_plan_calib_kv(
                            vbr,
                            ggml_type_name(params_base.cache_type_k),
                            ggml_type_name(params_base.cache_type_v));
                    }());
                if (cache_plan_obs) {
                    cache_plan_obs->calibration_profile =
                        cache_plan_calibration_profile;
                }
                if (cache_plan_authority) {
                    cache_plan_authority->calibration_profile =
                        cache_plan_calibration_profile;
                }
                if (cache_authority) {
                    cache_authority->calibration_profile =
                        cache_plan_calibration_profile;
                }
            }
            if (cache_plan_obs) {
                SRV_INF("cache-debug enabled: shadow cache-plan records per request (JSON log line + /slots.cache_plan), calibration profile '%s'\n",
                        cache_plan_obs->calibration_profile.c_str());
            }
        }

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to the initial caller
        if (!is_resume) {
            params = params_base;
        }

        if (!is_resume) {
            load_succeeded = init();
        } else {
            if (callback_state) {
                callback_state(SERVER_STATE_READY, {});
            }
            load_succeeded = true;
        }

        return load_succeeded;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });
        queue_tasks.on_idle([this]() {
            // quiet moment on the decode thread: flush deferred VBR maintenance.
            // target-only is deliberate: draft/MTP KV types are parsed with
            // allow_vbr=false, so those contexts can never have anything to breathe
            SRV_TRC("%s", "idle tick: memory breathe\n");
            llama_memory_breathe(llama_get_memory(ctx_tgt));
        });


        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        // Shrink recurrent state to free backup cells during prefill.
        // Must happen after init-time decodes (common_speculative_is_compat, warmup)
        // so the scheduler's CUDA graph state isn't stale.
        if (recurrent_expansion.state == server_recurrent_expansion_state::contracted &&
            needs_reeval &&
            params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
            auto * mem = llama_get_memory(ctx_tgt);
            if (llama_memory_recurrent_shrink(mem, n_parallel_user)) {
                SRV_INF("shrunk recurrent state to %d cells for prefill (deferred %d backup cells)\n",
                        n_parallel_user, n_seq_max_full - n_parallel_user);
            }
        }

        // co-tenancy: init-time decodes end HERE — this closes the bracket opened right
        // after context creation (common's own warmup bracket already closed inside
        // common_init_from_params). Everything above (speculative compat probes, template
        // detection) is not real traffic and must not fire claim-complete: a held demand's
        // claim survives to the first real request. INVARIANT: no callee between the two
        // brackets may self-manage warmup on ctx_tgt (drafter code brackets ctx_dft only).
        llama_set_warmup(ctx_tgt, false);

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    uint64_t cache_plan_capability_snapshot(const server_task & task) const noexcept {
        uint64_t hash = 1469598103934665603ull;
        hash = server_cache_plan_capability_fold(hash, uint64_t(uint32_t(task.id_slot)));
        hash = server_cache_plan_capability_fold(hash, task.tokens.size());
        for (const auto & slot : slots) {
            hash = server_cache_plan_capability_fold(hash, uint64_t(uint32_t(slot.id)));
            hash = server_cache_plan_capability_fold(hash, slot.is_processing());
            hash = server_cache_plan_capability_fold(hash, slot.prompt.tokens.size());
            hash = server_cache_plan_capability_fold(hash, slot.prompt.checkpoints.size());
            hash = server_cache_plan_capability_fold(hash, uint64_t(slot.t_last_used));
            hash = server_cache_plan_capability_fold(hash, slot.can_speculate());
        }
        if (prompt_cache) {
            hash = server_cache_plan_capability_fold(hash, prompt_cache->states.size());
            for (const auto & state : prompt_cache->states) {
                hash = server_cache_plan_capability_fold(hash, state.prompt.tokens.size());
                hash = server_cache_plan_capability_fold(hash, state.prompt.checkpoints.size());
                hash = server_cache_plan_capability_fold(hash, state.data.size());
                hash = server_cache_plan_capability_fold(hash,
                    std::hash<std::string>{}(state.adapter_config_key));
            }
        }
        return hash;
    }

    common_cache_plan_selection cache_plan_origin_for(
            const server_task & task,
            uint64_t lcp) const noexcept {
        if (task.id_slot != -1) {
            return common_cache_plan_selection::by_id;
        }
        const float sim = task.tokens.empty()
            ? 0.0f : float(lcp) / float(task.tokens.size());
        // Keep in lockstep with the shipped tier boundaries below: strict
        // similarity threshold, then dynamic-VBR route-home, then LRU.
        if (common_cache_plan_strict_similarity(
                sim, slot_prompt_similarity)) {
            return common_cache_plan_selection::similarity;
        }
        if (server_vbr_dynamic_active(params_base) && lcp > 0) {
            return common_cache_plan_selection::route_home;
        }
        return common_cache_plan_selection::lru;
    }

    bool cache_plan_target_in_domain(
            const server_task & task,
            const server_slot & slot) const noexcept {
        return !slot.is_processing() &&
               (task.id_slot == -1 || slot.id == task.id_slot);
    }

    server_slot * cache_plan_slot_by_exact_id(int32_t id) noexcept {
        // get_slot_by_id deliberately modulo-wraps client routing ids; a
        // planner capability names one exact recorded physical slot instead.
        for (auto & slot : slots) {
            if (slot.id == id) {
                return &slot;
            }
        }
        return nullptr;
    }

    bool cache_plan_retarget_target_current(
            const common_cache_plan_record & rec,
            const server_slot & slot) const noexcept {
        // Token contents cannot change inside this synchronous scheduler block.
        // Reuse the LCP/sim result captured by the complete inventory instead
        // of rescanning a potentially 200k-token prompt; only busy/empty can
        // drift independently before the mutation seam.
        if (slot.is_processing() || slot.prompt.tokens.empty()) {
            return false;
        }
        for (uint32_t i = 0; i < rec.n_inventory; ++i) {
            const auto & row = rec.inventory[i];
            if (row.provider != common_cache_plan_provider::live_slot ||
                row.target_slot_id != slot.id || row.source_id != slot.id ||
                row.lcp_tokens.state != llama_cache_acct_known::known ||
                !row.sim_known || !row.viable() ||
                !common_cache_plan_origin_in_domain(
                    row.origin_tier, rec.selection)) {
                continue;
            }
            return rec.selection != common_cache_plan_selection::similarity ||
                   common_cache_plan_strict_similarity(
                       row.sim, slot_prompt_similarity);
        }
        return false;
    }

    bool cache_plan_planned_slot_current(
            const common_cache_plan_record & rec,
            const server_slot & legacy_target,
            const server_slot * planned,
            common_cache_plan_provider provider,
            bool adapter_matches) const noexcept {
        if (!planned || planned->is_processing() ||
            (provider != common_cache_plan_provider::cold_replay &&
             !adapter_matches)) {
            return false;
        }
        return planned == &legacy_target ||
               cache_plan_retarget_target_current(rec, *planned);
    }

    bool cache_plan_inventory_live_rows(
            const server_task & task,
            const std::vector<uint64_t> & slot_lcps,
            common_cache_plan_record & rec) {
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto & slot = slots[i];
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }
            const auto origin = cache_plan_origin_for(task, slot_lcps[i]);
            auto * row = rec.find_or_add(
                common_cache_plan_provider::live_slot, slot.id,
                COMMON_CACHE_PLAN_PHASE_LRU, slot.id, origin);
            if (!row) {
                return false;
            }
            server_cache_plan_apply_live(row, server_cache_plan_evaluate_live(
                slot.is_processing(), !slot.prompt.tokens.empty(), slot_lcps[i],
                task.tokens.size(), slot.prompt.tokens.size()));
            // The shipped similarity scan visits below-threshold rows too;
            // authority membership is the first tier that can select the target.
            row->origin_tier = origin;
            row->t_last_used_us = llama_cache_acct_value::measured(
                uint64_t(std::max<int64_t>(slot.t_last_used, 0)));
            row->spec_capable = slot.can_speculate();
            row->spec_capable_known = true;
        }
        rec.note_inventory_complete(common_cache_plan_provider::live_slot);
        return true;
    }

    bool cache_plan_inventory_host_rows(
            const server_task & task,
            const std::vector<uint64_t> & slot_lcps,
            const std::string & incoming_adapter,
            bool recurrent,
            common_cache_plan_record & rec,
            cache_plan_host_source_registry & source_registry) {
        if (!prompt_cache) {
            rec.note_inventory_complete(common_cache_plan_provider::host_cache_entry);
            return true;
        }
        for (auto & state : prompt_cache->states) {
            int32_t host_source_id = -1;
            if (!source_registry.get(
                    *prompt_cache, state, host_source_id)) {
                rec.inventory_states[size_t(
                    common_cache_plan_provider::host_cache_entry)] =
                    common_cache_plan_inventory_state::overflowed;
                return false;
            }
            const uint64_t lcp = state.prompt.tokens.get_common_prefix(task.tokens);
            const auto host_eval = server_cache_plan_evaluate_host(
                !state.data.main.empty(),
                state.adapter_config_key == incoming_adapter,
                lcp, task.tokens.size(), state.prompt.tokens.size(),
                state.data.size());
            std::vector<server_cache_plan_checkpoint_evaluation> checkpoint_evals;
            checkpoint_evals.reserve(state.prompt.checkpoints.size());
            for (const auto & checkpoint : state.prompt.checkpoints) {
                checkpoint_evals.push_back(server_cache_plan_evaluate_checkpoint(
                    !checkpoint.empty(), true, recurrent, true,
                    checkpoint.pos_min, checkpoint.pos_max,
                    int64_t(lcp), 0, checkpoint.size()));
            }

            for (size_t target_i = 0; target_i < slots.size(); ++target_i) {
                const auto & candidate_target = slots[target_i];
                if (!cache_plan_target_in_domain(task, candidate_target)) {
                    continue;
                }
                const auto origin = cache_plan_origin_for(task, slot_lcps[target_i]);
                auto * host = rec.find_or_add(
                    common_cache_plan_provider::host_cache_entry, host_source_id,
                    COMMON_CACHE_PLAN_PHASE_HOST_SCAN, candidate_target.id, origin);
                if (!host) {
                    return false;
                }
                server_cache_plan_apply_host(host, host_eval);

                int32_t checkpoint_ordinal = 0;
                for (const auto & eval : checkpoint_evals) {
                    const int32_t checkpoint_source =
                        server_cache_plan_host_checkpoint_source_id(
                            host_source_id, checkpoint_ordinal++);
                    if (checkpoint_source < 0) {
                        rec.inventory_states[size_t(
                            common_cache_plan_provider::live_context_checkpoint)] =
                            common_cache_plan_inventory_state::overflowed;
                        return false;
                    }
                    auto * checkpoint = rec.find_or_add(
                        common_cache_plan_provider::live_context_checkpoint,
                        checkpoint_source,
                        COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                        candidate_target.id, origin);
                    if (!checkpoint) {
                        return false;
                    }
                    server_cache_plan_apply_checkpoint(checkpoint, eval);
                    if (server_cache_plan_viable(host_eval.reason) &&
                        server_cache_plan_viable(eval.reason)) {
                        checkpoint->component_only = true;
                        checkpoint->dependent_host_source_id = host_source_id;
                        auto * chain = rec.add_chain(
                            common_cache_plan_provider::host_cache_entry,
                            int32_t(host - rec.inventory.data()),
                            int32_t(checkpoint - rec.inventory.data()));
                        if (!chain) {
                            return false;
                        }
                        chain->disposition =
                            common_cache_plan_disposition::valid_not_chosen_cost;
                    }
                }
            }
        }
        rec.note_inventory_complete(common_cache_plan_provider::host_cache_entry);
        return true;
    }

    bool cache_plan_inventory_checkpoint_rows(
            const server_task & task,
            const std::vector<uint64_t> & slot_lcps,
            bool recurrent,
            common_cache_plan_record & rec) {
        // Complete planner scan. The shipped restore keeps its reverse find_if
        // and its shipped-visited transport below.
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto & candidate_target = slots[i];
            if (!cache_plan_target_in_domain(task, candidate_target)) {
                continue;
            }
            const llama_pos pos_next = task.tokens.pos_next(slot_lcps[i]);
            const bool has_new_tokens = slot_lcps[i] < task.tokens.size();
            const llama_pos pos_min_threshold =
                std::max<llama_pos>(
                    0, pos_next - n_swa - (has_new_tokens ? 0 : 1));
            const auto vbr_now = llama_memory_vbr_state(
                llama_get_memory(ctx_tgt), candidate_target.id, 0);
            std::string adapter_identity;
            bool adapter_known = true;
            try {
                adapter_identity = lora_config_identity(candidate_target.lora);
            } catch (...) {
                adapter_known = false;
            }
            int32_t checkpoint_ordinal = 0;
            for (const auto & checkpoint : candidate_target.prompt.checkpoints) {
                const bool checkpoint_lineage_matches =
                    !recurrent ||
                    common_prompt_checkpoint_lineage_matches(checkpoint, vbr_now);
                const bool frontier_current =
                    !candidate_target.frontier_ratchet_flipped ||
                    (adapter_known && checkpoint_frontier_is_current(
                        candidate_target, checkpoint, adapter_identity));
                auto * row = rec.find_or_add(
                    common_cache_plan_provider::live_context_checkpoint,
                    checkpoint_ordinal++, COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                    candidate_target.id,
                    cache_plan_origin_for(task, slot_lcps[i]));
                if (!row) {
                    return false;
                }
                server_cache_plan_apply_checkpoint(row,
                    server_cache_plan_evaluate_checkpoint(
                        !checkpoint.empty(), frontier_current, recurrent,
                        checkpoint_lineage_matches, checkpoint.pos_min,
                        checkpoint.pos_max, pos_next, pos_min_threshold,
                        checkpoint.size()));
            }
        }
        rec.note_inventory_complete(
            common_cache_plan_provider::live_context_checkpoint);
        rec.authority_inventory_complete = true;
        return true;
    }

    bool cache_plan_inventory_cold_rows(
            const server_task & task,
            const std::vector<uint64_t> & slot_lcps,
            common_cache_plan_record & rec) {
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto & candidate_target = slots[i];
            if (!cache_plan_target_in_domain(task, candidate_target)) {
                continue;
            }
            auto * cold = rec.find_or_add(
                common_cache_plan_provider::cold_replay,
                COMMON_CACHE_PLAN_SOURCE_AGGREGATE, 0, candidate_target.id,
                cache_plan_origin_for(task, slot_lcps[i]));
            if (!cold) {
                return false;
            }
            cold->accept();
        }
        rec.note_inventory_complete(common_cache_plan_provider::cold_replay);
        return true;
    }

    void cache_plan_inventory_before_mutation(
            const server_task & task,
            server_slot & target,
            const std::string & incoming_adapter,
            common_cache_plan_record & rec,
            cache_plan_host_source_registry & source_registry) {
        rec.id_slot = target.id;
        rec.n_prompt_tokens = llama_cache_acct_value::measured(task.tokens.size());
        rec.identity.adapter_config_digest = llama_cache_acct_value::measured(
            std::hash<std::string>{}(incoming_adapter));

        source_registry.begin(prompt_cache.get());

        std::vector<uint64_t> slot_lcps;
        slot_lcps.reserve(slots.size());
        for (const auto & slot : slots) {
            slot_lcps.push_back(slot.prompt.tokens.get_common_prefix(task.tokens));
        }
        const bool recurrent = llama_model_is_recurrent(model_tgt) ||
                               llama_model_is_hybrid(model_tgt);

        // A fixed record is an honest bounded surface, not a promise that a
        // cross-product always fits. Any failed append latches provider
        // overflow (or derived_plans_incomplete); stop paying work immediately.
        if (!cache_plan_inventory_live_rows(task, slot_lcps, rec) ||
            !cache_plan_inventory_host_rows(
                task, slot_lcps, incoming_adapter, recurrent, rec,
                source_registry) ||
            !cache_plan_inventory_checkpoint_rows(
                task, slot_lcps, recurrent, rec) ||
            !cache_plan_inventory_cold_rows(task, slot_lcps, rec)) {
            GGML_ASSERT(rec.inventory_saturated());
            return;
        }
    }

    bool cache_plan_execution_revalidate(
            const server_task & task,
            const server_slot & target,
            bool adapter_matches,
            const std::string & incoming_adapter,
            const server_cache_plan_execution & execution,
            bool destruction_certified = false) {
        if (!execution.authoritative() || execution.target != target.id ||
            target.is_processing()) {
            return false;
        }
        if (execution.kind != server_cache_plan_execution_kind::cold_replay &&
            !adapter_matches) {
            // The existing launch seam clears on an adapter rebind. Until a
            // future ratchet can bind-before-restore, such a plan is not an
            // executable reuse plan and must fall back before mutation.
            return false;
        }

        const auto find_host = [&](int32_t source_id)
                -> const server_prompt_cache_state * {
            if (!prompt_cache || source_id < 0) {
                return nullptr;
            }
            for (auto & state : prompt_cache->states) {
                int32_t observed = -1;
                if (prompt_cache->cache_plan_get_source_id(state, observed) &&
                    observed == source_id && !state.data.main.empty() &&
                    state.adapter_config_key == incoming_adapter) {
                    return &state;
                }
            }
            return nullptr;
        };
        const auto checkpoint_alive = [](
                const auto & checkpoints,
                int32_t source_id,
                int32_t host_source_id) {
            const int32_t ordinal =
                server_cache_plan_checkpoint_ordinal_from_source_id(
                    source_id, host_source_id);
            return ordinal >= 0 && size_t(ordinal) < checkpoints.size() &&
                   !std::next(checkpoints.begin(), ordinal)->empty();
        };

        switch (execution.kind) {
            case server_cache_plan_execution_kind::live_replay:
                return !target.prompt.tokens.empty();
            case server_cache_plan_execution_kind::host_restore: {
                const auto * host = find_host(execution.host_source_id);
                return task.type == SERVER_TASK_TYPE_COMPLETION && host;
            }
            case server_cache_plan_execution_kind::checkpoint_restore:
                return checkpoint_alive(target.prompt.checkpoints,
                    execution.checkpoint_source_id, -1);
            case server_cache_plan_execution_kind::host_checkpoint_restore: {
                const auto * host = find_host(execution.host_source_id);
                return task.type == SERVER_TASK_TYPE_COMPLETION &&
                       host && checkpoint_alive(host->prompt.checkpoints,
                           execution.checkpoint_source_id,
                           execution.host_source_id);
            }
            case server_cache_plan_execution_kind::cold_replay:
                // Pre-D-A cold authority is limited to construction-empty
                // targets. Clearing retained live/checkpoint state is a
                // divergent destruction plan and remains D-A's authority.
                return destruction_certified ||
                    server_cache_plan_cold_target_current(
                        execution, target.prompt.tokens.empty(),
                        target.prompt.checkpoints.empty());
            case server_cache_plan_execution_kind::legacy:
            case server_cache_plan_execution_kind::_count:
                return false;
        }
        return false;
    }

    static void cache_plan_host_restore_failed(
            server_slot & slot,
            common_cache_plan_record * rec) {
        if (rec) {
            rec->restore_attempt_failed = true;
        }
        slot.mandatory_recovery_reset(
            server_cache_destruction_reason::restore_failure);
    }

    void cache_plan_fallback_legacy(
            server_slot & slot,
            common_cache_plan_authority_fallback reason) {
        if (!slot.cache_plan_execution.authoritative()) {
            return;
        }
        GGML_ASSERT(cache_plan_authority && slot.cache_plan);
        cache_plan_authority->fallback_legacy(*slot.cache_plan, reason);
        slot.cache_plan_execution.clear();
    }

    bool cache_plan_checkpoint_execution_revalidate(
            const server_slot & slot,
            llama_pos pos_next,
            llama_pos pos_min_threshold) const {
        const auto & execution = slot.cache_plan_execution;
        if (!execution.restores_checkpoint()) {
            return true;
        }
        int32_t ordinal = -1;
        if (!server_cache_plan_checkpoint_override_ordinal(
                execution, slot.prompt.checkpoints.size(), true, ordinal)) {
            return false;
        }
        const auto & checkpoint = *std::next(
            slot.prompt.checkpoints.begin(), ordinal);
        const bool recurrent = llama_model_is_recurrent(model_tgt) ||
                               llama_model_is_hybrid(model_tgt);
        const auto vbr_now = llama_memory_vbr_state(
            llama_get_memory(ctx_tgt), slot.id, 0);
        const bool checkpoint_lineage_matches =
            !recurrent ||
            common_prompt_checkpoint_lineage_matches(checkpoint, vbr_now);
        bool frontier_current = false;
        try {
            // Authority requires current durable frontier evidence even while
            // the legacy frontier ratchet remains in shadow. Host-composed
            // inventory was optimistic; this is its first exact live check.
            frontier_current = checkpoint_frontier_is_current(
                slot, checkpoint, lora_config_identity(slot.lora));
        } catch (...) {
            return false;
        }
        return server_cache_plan_viable(
            server_cache_plan_evaluate_checkpoint(
                !checkpoint.empty(), frontier_current, recurrent,
                checkpoint_lineage_matches, checkpoint.pos_min,
                checkpoint.pos_max, pos_next, pos_min_threshold,
                checkpoint.size()).reason);
    }

    std::list<common_prompt_checkpoint>::reverse_iterator
    cache_plan_override_checkpoint(
            server_slot & slot,
            std::list<common_prompt_checkpoint>::reverse_iterator fallback) {
        if (!slot.cache_plan_execution.restores_checkpoint()) {
            return fallback;
        }
        int32_t ordinal = -1;
        if (!server_cache_plan_checkpoint_override_ordinal(
                slot.cache_plan_execution, slot.prompt.checkpoints.size(),
                true, ordinal)) {
            // A late container drift demotes to the same iterator the shipped
            // scan selected. Missing authority evidence must never synthesize
            // a cold reset.
            cache_plan_fallback_legacy(
                slot, common_cache_plan_authority_fallback::stale_capability);
            return fallback;
        }

        auto selected_it = std::make_reverse_iterator(
            std::next(slot.prompt.checkpoints.begin(), ordinal + 1));
        const auto & execution = slot.cache_plan_execution;
        if (slot.cache_plan) {
            auto * selected = slot.cache_plan->find_or_add(
                common_cache_plan_provider::live_context_checkpoint,
                execution.checkpoint_source_id,
                COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                slot.id,
                slot.cache_plan->selection);
            if (selected) {
                selected->accept();
                slot.cache_plan->select(
                    common_cache_plan_provider::live_context_checkpoint,
                    selected);
            }
        }
        return selected_it;
    }

    struct cache_plan_stage1_selection {
        server_slot * target = nullptr;
        bool update_cache = false;
        bool selection_deferred_busy = false;
        std::unique_ptr<common_cache_plan_record> record;
    };

    cache_plan_stage1_selection cache_plan_select_before_mutation(
            const server_task & task,
            server_cache_plan_authority * plan_authority,
            bool is_preflight) {
        server_slot * ret = nullptr;

        bool update_cache = false;
        bool selection_deferred_busy = false;

        // best similarity seen even BELOW the threshold — feeds the vbr route-home tier and the
        // LRU log line (a hopping conversation was undiagnosable: "selected by LRU" never said
        // how close the rejected candidates came)
        float sim_best_any = 0;

        // B0: begin the multi-stage record BEFORE any selection-side mutation [B-a]. The
        // selection scan below is read-only; the record exists only under --cache-debug, and
        // every row write from here on is noexcept (fixed-array record), so the shipped path
        // cannot throw through the observer. An allocation fault is swallowed and counted.
        std::unique_ptr<common_cache_plan_record> plan_rec;
        if (plan_authority) {
            try {
                plan_rec = std::make_unique<common_cache_plan_record>();
                plan_rec->id_task = task.id;
                // B-2: the profile is composed once at init and copied here (inside the
                // creation try — never from a selector hook)
                plan_rec->calibration_profile =
                    plan_authority->calibration_profile;
                // opaque identity evidence from already-computed keys (never raw values);
                // identities the server has not computed stay typed unknown
                plan_rec->identity.model_digest = llama_cache_acct_value::measured(
                    std::hash<std::string>{}(model_name));
                plan_rec->identity.execution_digest = llama_cache_acct_value::measured(
                    std::hash<std::string>{}(frontier_execution_identity));
                // SAMPLED-PREFIX TELEMETRY (not an identity): FNV-1a over at most the
                // prompt's first 64 token ids + that sampled count. Useful for OFFLINE
                // clustering of workload variants; it is NOT an exact family key and must
                // never drive a shipped decision — prompts sharing 64 leading tokens merge
                // deliberately, the folded count saturates at the window, and FNV collides.
                // Opaque + content-minimizing like every identity digest.
                {
                    uint64_t h = 1469598103934665603ull;
                    const auto fold = [&h](uint64_t v) {
                        h = (h ^ v) * 1099511628211ull;
                    };
                    const size_t n = std::min<size_t>(task.tokens.size(), 64);
                    for (size_t i = 0; i < n; i++) {
                        fold((uint64_t)(uint32_t) task.tokens[i]);
                    }
                    fold((uint64_t) n);
                    plan_rec->identity.prefix_token_digest = llama_cache_acct_value::measured(h);
                }
            } catch (...) {
                if (!is_preflight && cache_plan_obs) {
                    cache_plan_obs->shadow_unavailable++;
                }
            }
        }

        // one observer row per slot a loop classifies (merge key = slot id across all
        // loops); nullptr when the observer is off or the inventory overflowed [A2]
        const auto slot_row = [&](const server_slot & slot, uint8_t phase) {
            return plan_rec
                ? plan_rec->find_or_add(
                      common_cache_plan_provider::live_slot, slot.id, phase, slot.id,
                      common_cache_plan_origin_for_phase(phase))
                : (common_cache_plan_candidate *) nullptr;
        };

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                if (!is_preflight) {
                    SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
                }
                if (plan_rec) {
                    plan_rec->selection = common_cache_plan_selection::by_id;
                    // forced selection carries no reuse verdict, but the slot WAS observed —
                    // its row exists with disposition unavailable; finalize upgrades the
                    // selected row from realized prefix reuse
                    slot_row(*ret, COMMON_CACHE_PLAN_PHASE_BY_ID);
                }
            }
        }

        // find the slot that has at least n% prompt similarity.
        // Observer transport [A2, noexcept]: one row per slot this loop classifies (merge key
        // = slot id across all three loops); every evaluated survivor starts as a cost loser
        // and the shipped winner is promoted after the scan. A nullptr row (inventory
        // overflow) is skipped — the shipped scan never changes.
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            // shipped logic written ONCE; the unobserved instantiation compiles every
            // observer block away — literal zero work with --cache-debug off [A2/F7]
            const auto similarity_scan = [&](auto obs_c) {
                constexpr bool Obs = decltype(obs_c)::value;
                for (server_slot & slot : slots) {
                    if (task.id_slot != -1 && slot.id != task.id_slot) {
                        continue; // id filter, not a classification: no row
                    }

                    [[maybe_unused]] common_cache_plan_candidate * row = nullptr;
                    if constexpr (Obs) {
                        row = slot_row(slot, COMMON_CACHE_PLAN_PHASE_SIMILARITY);
                    }

                    // skip the slot if it is not available
                    if (slot.is_processing()) {
                        if constexpr (Obs) {
                            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY); }
                        }
                        continue;
                    }

                    const auto & tokens = slot.prompt.tokens;

                    // skip the slot if it does not contains cached tokens
                    if (tokens.empty()) {
                        if constexpr (Obs) {
                            // truly stateless slot [B-8]: unavailable, never coverage
                            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE); }
                        }
                        continue;
                    }

                    // fraction of the Longest Common Prefix length with respect to the input prompt length
                    const size_t lcp_cur = tokens.get_common_prefix(task.tokens);
                    const float  sim_cur = float(lcp_cur) / task.tokens.size();

                    sim_best_any = std::max(sim_best_any, sim_cur);

                    if constexpr (Obs) {
                        server_cache_plan_apply_live(row,
                            server_cache_plan_evaluate_live(
                                false, true, lcp_cur, task.tokens.size()));
                    }

                    // select the current slot if the criteria match
                    if (sim_cur > f_sim_best &&
                        common_cache_plan_strict_similarity(
                            sim_cur, slot_prompt_similarity)) {
                        f_sim_best = sim_cur;

                        ret = &slot;
                    }
                }
            };
            plan_rec ? similarity_scan(std::true_type{}) : similarity_scan(std::false_type{});

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    if (!is_preflight) {
                        SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                                f_sim_best, slot_prompt_similarity, f_keep);
                    }
                    if (plan_rec) {
                        plan_rec->selection = common_cache_plan_selection::similarity;
                        if (auto * row = slot_row(*ret, COMMON_CACHE_PLAN_PHASE_SIMILARITY)) {
                            row->accept();
                            row->sim    = f_sim_best; row->sim_known    = true;
                            row->f_keep = f_keep;   row->f_keep_known = true;
                        }
                    }
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }

            if (plan_rec) {
                plan_rec->sim_best_any       = sim_best_any;
                plan_rec->sim_best_any_known = true;
            }
        }

        // dynamic VBR: route a returning conversation HOME before the LRU tier colonizes a fresh
        // slot. In the unified pool an idle slot's cells are live cost for everyone (mapped pages,
        // attention span, the one shared quality budget), and the LRU tier below ranks never-used
        // slots FIRST — a rewritten-context prompt that fails the similarity threshold would stack
        // slot after slot while its own history rots elsewhere. Any nonzero LCP (a shared system
        // prompt suffices) identifies the home slot; genuinely-new streams (zero LCP everywhere)
        // still take the LRU spread below.
        if (ret == nullptr && server_vbr_dynamic_active(params_base)) {
            size_t lcp_best = 0;

            const auto route_home_scan = [&](auto obs_c) {
                constexpr bool Obs = decltype(obs_c)::value;
                for (server_slot & slot : slots) {
                    [[maybe_unused]] common_cache_plan_candidate * row = nullptr;
                    if constexpr (Obs) {
                        row = slot_row(slot, COMMON_CACHE_PLAN_PHASE_ROUTE_HOME);
                    }

                    if (slot.is_processing()) {
                        if constexpr (Obs) {
                            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY); }
                        }
                        continue;
                    }

                    const auto & tokens = slot.prompt.tokens;

                    if (tokens.empty()) {
                        if constexpr (Obs) {
                            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE); }
                        }
                        continue;
                    }

                    const size_t lcp = tokens.get_common_prefix(task.tokens);
                    if constexpr (Obs) {
                        server_cache_plan_apply_live(row,
                            server_cache_plan_evaluate_live(
                                false, true, lcp, task.tokens.size()));
                    }
                    if (lcp > lcp_best) {
                        lcp_best = lcp;

                        ret = &slot;
                    }
                }
            };
            plan_rec ? route_home_scan(std::true_type{}) : route_home_scan(std::false_type{});

            if (ret != nullptr) {
                if (!is_preflight) {
                    SLT_INF(*ret, "selected slot by route-home (vbr), lcp = %zu tokens (sim %.3f <= %.3f thold)\n",
                            lcp_best, (double) lcp_best / task.tokens.size(), slot_prompt_similarity);
                }

                if (plan_rec) {
                    plan_rec->selection = common_cache_plan_selection::route_home;
                    if (auto * row = slot_row(*ret, COMMON_CACHE_PLAN_PHASE_ROUTE_HOME)) {
                        row->accept();
                        row->lcp_tokens = llama_cache_acct_value::measured(lcp_best);
                    }
                }

                update_cache = true;
            }
        }

        // find the slot that has been least recently used
        // prefer spec-capable (DFlash) slots so requests get speculative decoding
        if (ret == nullptr) {
            int64_t t_last = -1;

            const auto lru_scan = [&](auto obs_c) {
                constexpr bool Obs = decltype(obs_c)::value;
                for (server_slot & slot : slots) {
                    [[maybe_unused]] common_cache_plan_candidate * row = nullptr;
                    if constexpr (Obs) {
                        row = slot_row(slot, COMMON_CACHE_PLAN_PHASE_LRU);
                    }

                    // skip the slot if it is not available
                    if (slot.is_processing()) {
                        if constexpr (Obs) {
                            if (row) { row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY); }
                        }
                        continue;
                    }

                    if constexpr (Obs) {
                        if (row) {
                            // scalars THIS loop computes; reuse verdicts (if any) came from the
                            // reuse-evaluating loops via the merged row
                            row->t_last_used_us = llama_cache_acct_value::measured(
                                (uint64_t) std::max<int64_t>(slot.t_last_used, 0));
                            row->spec_capable       = slot.can_speculate();
                            row->spec_capable_known = true;
                        }
                    }

                    // strongly prefer spec-capable slots: pick a spec slot over a non-spec
                    // slot regardless of LRU, then use LRU within the same capability tier
                    const bool curr_spec = ret && ret->can_speculate();
                    const bool slot_spec = slot.can_speculate();
                    if (!ret ||
                        (slot_spec && !curr_spec) ||
                        (slot_spec == curr_spec && slot.t_last_used < t_last)) {
                        t_last = slot.t_last_used;
                        ret = &slot;
                    }
                }
            };
            plan_rec ? lru_scan(std::true_type{}) : lru_scan(std::false_type{});

            if (ret != nullptr) {
                if (!is_preflight) {
                    SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 " (best rejected sim = %.3f)\n",
                            t_last, sim_best_any);
                }

                if (plan_rec) {
                    plan_rec->selection = common_cache_plan_selection::lru;
                }

                update_cache = true;
            }
        }

        if (ret && ret->is_processing()) {
            // A by-id task targeting a busy slot is deferred by the caller.
            // Do not replace that slot's in-flight evidence record with this
            // retry's local record; shipped selection/mutation remains intact.
            // E0 preflight does retain the request-local record so PROVIDER_BUSY
            // remains visible as typed evidence without arming the slot.
            selection_deferred_busy = true;
            if (!is_preflight) {
                plan_rec.reset();
            }
        }

        return {
            ret,
            update_cache,
            selection_deferred_busy,
            std::move(plan_rec),
        };
    }

    struct cache_plan_stage1_inventory {
        std::vector<common_adapter_lora_info> incoming_loras;
        std::string incoming_adapter;
        bool incoming_adapter_ready = false;
        bool incoming_adapter_matches = false;
        bool host_lookup_enabled = false;
    };

    struct cache_plan_stage1_mode {
        server_cache_plan_authority * plan_authority = nullptr;
        bool preflight = false;
    };

    cache_plan_stage1_mode cache_plan_stage1_mode_for(
            server_cache_plan_authority * plan_authority,
            bool is_preflight) noexcept {
        GGML_ASSERT(plan_authority);
        return { plan_authority, is_preflight };
    }

    void cache_plan_derive_incoming_adapter(
            const server_task & task,
            const server_slot & target,
            cache_plan_stage1_inventory & out) {
        out.incoming_loras = task.params.lora.empty()
            ? params_base.lora_adapters
            : construct_lora_list(task.params.lora);
        out.incoming_adapter = lora_config_identity(out.incoming_loras);
        out.incoming_adapter_ready = true;
        out.incoming_adapter_matches =
            are_lora_equal(out.incoming_loras, target.lora);
    }

    void cache_plan_inventory_and_plan_before_mutation(
            const server_task & task,
            server_slot & target,
            bool update_cache,
            common_cache_plan_record & rec,
            cache_plan_stage1_inventory & out,
            const cache_plan_stage1_mode & mode) {
        GGML_ASSERT(mode.plan_authority);
        const uint64_t capability_before =
            cache_plan_capability_snapshot(task);
        cache_plan_host_source_registry source_registry(
            !mode.preflight);
        cache_plan_derive_incoming_adapter(task, target, out);
        cache_plan_inventory_before_mutation(
            task, target, out.incoming_adapter, rec, source_registry);
        const uint64_t capability_after =
            cache_plan_capability_snapshot(task);
        const auto semantics = server_cache_plan_stage1_semantics_for(
            mode.preflight,
            task.type == SERVER_TASK_TYPE_COMPLETION,
            update_cache,
            prompt_cache != nullptr,
            out.incoming_adapter_matches);
        out.host_lookup_enabled = semantics.host_lookup_enabled;
        common_cache_plan_destruction_counters throwaway_destruction_counters;
        auto * destruction_counters = mode.preflight
            ? &throwaway_destruction_counters
            : (cache_authority
                ? &cache_authority->destruction_counters
                : nullptr);
        uint64_t * production_quote_sequence =
            !mode.preflight && cache_authority
                ? &cache_authority->destruction_quote_sequence
                : nullptr;
        const bool preview_lifecycle_available =
            params_base.cache_lifecycle;
        const bool quote_lifecycle_available = mode.preflight
            ? preview_lifecycle_available
            : true;
        if (destruction_counters &&
            (cache_authority || !quote_lifecycle_available)) {
            const int64_t destruction_quote_started = ggml_time_us();
            server_cache_destruction_quote_options options {
                quote_lifecycle_available,
                semantics.recovery_citation,
                0,
                server_cache_plan_nonconsuming_host_effects(
                    params_base.cache_lifecycle),
                mode.preflight,
            };
            cache_plan_quote_destruction(
                target.id, out.host_lookup_enabled, rec,
                options, production_quote_sequence,
                *destruction_counters, &source_registry);
            const uint64_t destruction_quote_duration = uint64_t(
                std::max<int64_t>(0,
                    ggml_time_us() - destruction_quote_started));
            rec.destruction.quote_duration_us = destruction_quote_duration;
            for (auto & quote : rec.destruction_quotes) {
                quote.receipt.quote_duration_us = destruction_quote_duration;
            }
            destruction_counters->quote_samples++;
            destruction_counters->quote_duration_us_total +=
                destruction_quote_duration;
            destruction_counters->quote_duration_us_max = std::max(
                destruction_counters->quote_duration_us_max,
                destruction_quote_duration);
        }
        mode.plan_authority->plan_before_mutation(
            rec, capability_before, capability_after);
        if (destruction_counters &&
            (mode.preflight || cache_plan_obs != nullptr)) {
            const int32_t legacy_candidate =
                rec.destruction_legacy_plan_candidate >= 0
                    ? rec.destruction_legacy_plan_candidate
                    : server_cache_plan_legacy_candidate(
                        rec, target.id, out.host_lookup_enabled);
            server_cache_destruction_select_preview(
                rec, *destruction_counters, legacy_candidate,
                preview_lifecycle_available,
                server_cache_plan_nonconsuming_host_effects(
                    params_base.cache_lifecycle));
        }
    }

    server_slot * get_available_slot(const server_task & task) {
        auto stage1 = cache_plan_select_before_mutation(
            task, cache_plan_authority.get(), false);
        server_slot * ret = stage1.target;
        bool update_cache = stage1.update_cache;
        const bool selection_deferred_busy = stage1.selection_deferred_busy;
        auto plan_rec = std::move(stage1.record);

        cache_plan_stage1_inventory stage1_inventory;
        auto & incoming_loras = stage1_inventory.incoming_loras;
        auto & incoming_adapter = stage1_inventory.incoming_adapter;
        auto & incoming_adapter_ready = stage1_inventory.incoming_adapter_ready;
        auto & incoming_adapter_matches =
            stage1_inventory.incoming_adapter_matches;

        if (ret) {
            if (!selection_deferred_busy) {
                // E0.0 boundary: this real-request mutation deliberately stays
                // outside both reusable stage-1 functions. A preflight caller
                // must never inherit stale-arm cleanup.
                // An idle slot cannot own an in-flight directive. Clear any
                // abandoned pre-launch arm before this request installs its
                // own record; this also covers observer allocation failure.
                server_cache_plan_disarm_unlaunched(
                    ret->cache_plan_execution, ret->cache_plan,
                    ret->cache_plan_destruction_recovery_pin);
            }
            if (plan_rec) {
                // plan_rec is created only by the debug-or-authority substrate,
                // which always owns cache_plan_authority.
                GGML_ASSERT(cache_plan_authority);
                server_slot * const legacy_ret = ret;
                try {
                    cache_plan_inventory_and_plan_before_mutation(
                        task, *legacy_ret, update_cache, *plan_rec,
                        stage1_inventory,
                        cache_plan_stage1_mode_for(
                            cache_plan_authority.get(), false));
                    const bool host_lookup_enabled =
                        stage1_inventory.host_lookup_enabled;

                    // B-A2 may choose another strict-similarity target, but
                    // only the authority layer interprets that choice. Lower
                    // levels keep the B-A1/legacy target exactly as selected.
                    const bool selection_admits_retarget =
                        server_cache_plan_selection_admits_retarget(
                            cache_plan_authority->configured_level,
                            plan_rec->selection);
                    const int32_t planned_target =
                        server_cache_plan_planned_target(
                            *plan_rec,
                            cache_plan_authority->configured_level,
                            legacy_ret->id);
                    server_slot * planned_ret = planned_target == legacy_ret->id
                        ? legacy_ret
                        : cache_plan_slot_by_exact_id(planned_target);
                    const bool planned_adapter_matches =
                        planned_ret == legacy_ret
                            ? incoming_adapter_matches
                            : (planned_ret &&
                               are_lora_equal(
                                   incoming_loras, planned_ret->lora));
                    const auto planned_provider =
                        server_cache_plan_shadow_choice_valid(*plan_rec)
                            ? plan_rec->inventory[size_t(
                                  plan_rec->shadow_choice)].provider
                            : common_cache_plan_provider::cold_replay;
                    const bool planned_slot_current =
                        cache_plan_planned_slot_current(
                            *plan_rec, *legacy_ret, planned_ret,
                            planned_provider, planned_adapter_matches);
                    live_displacement_certification displacement;
                    if (planned_ret) {
                        displacement = cache_plan_certify_live_displacement(
                            *planned_ret, *legacy_ret, *plan_rec);
                    }
                    auto execution = cache_plan_authority->authorize(
                        *plan_rec, legacy_ret->id,
                        host_lookup_enabled,
                        planned_adapter_matches,
                        server_cache_plan_nonconsuming_host_effects(
                            params_base.cache_lifecycle) |
                        (displacement.ready ? displacement.effects : 0));
                    if (execution.authoritative() &&
                        server_cache_plan_retarget_currency_required(
                            selection_admits_retarget, planned_target,
                            legacy_ret->id) &&
                        !planned_slot_current) {
                        cache_plan_authority->fallback_legacy(
                            *plan_rec,
                            common_cache_plan_authority_fallback::stale_capability);
                        execution.clear();
                    }
                    if (execution.authoritative() &&
                        !cache_plan_execution_revalidate(
                            task, *planned_ret, planned_adapter_matches,
                            incoming_adapter, execution,
                            displacement.ready)) {
                        cache_plan_authority->fallback_legacy(
                            *plan_rec,
                            common_cache_plan_authority_fallback::stale_capability);
                        execution.clear();
                    }
                    if (execution.authoritative() &&
                        planned_ret != legacy_ret &&
                        !displacement.ready) {
                        // Retargeting leaves the legacy-selected slot idle and
                        // exposed to ordinary idle/VBR reclaim. Preserve the
                        // durability side effect legacy would have performed
                        // before switching targets; without a durable copy the
                        // retarget remains outside the pre-D-A envelope.
                        bool displaced_durable = false;
                        if (prompt_cache &&
                            task.type == SERVER_TASK_TYPE_COMPLETION) {
                            const auto saved =
                                legacy_ret->prompt_save(*prompt_cache);
                            if (prompt_save_durable(saved)) {
                                // Match the shipped save/load maintenance
                                // boundary even when no load follows.
                                prompt_cache->update();
                                displaced_durable = prompt_cache->contains(
                                    legacy_ret->prompt.tokens,
                                    lora_config_identity(legacy_ret->lora));
                            }
                        }
                        if (!displaced_durable) {
                            cache_plan_authority->fallback_legacy(
                                *plan_rec,
                                common_cache_plan_authority_fallback::
                                    destruction_authority_required);
                            execution.clear();
                        }
                    }
                    if (execution.authoritative() &&
                        planned_ret != legacy_ret) {
                        // Clear the target's prior arm before a certified
                        // displacement installs a fresh recovery pin. That
                        // new pin then survives through dependent execution
                        // and closes only at reset or an unlaunched disarm.
                        server_cache_plan_disarm_unlaunched(
                            planned_ret->cache_plan_execution,
                            planned_ret->cache_plan,
                            planned_ret->cache_plan_destruction_recovery_pin);
                    }
                    if (execution.authoritative() &&
                        displacement.ready) {
                        // Physical slot removal is the only irreversible step.
                        // The capability was freshly prepared after both
                        // durability publications, so no save/accounting write
                        // can interleave before this same-frame commit.
                        planned_ret->prompt_clear_certified(
                            displacement.capability,
                            displacement.quote,
                            *plan_rec);
                    }
                    if (execution.authoritative() &&
                        planned_ret != legacy_ret) {
                        // The exact target gets the same stale-arm hygiene as
                        // a legacy-selected idle slot. The displaced live row
                        // remains counterfactual evidence, but is no longer a
                        // second accepted shipped winner.
                        for (uint32_t i = 0; i < plan_rec->n_inventory; ++i) {
                            auto & row = plan_rec->inventory[i];
                            if (row.provider ==
                                    common_cache_plan_provider::live_slot &&
                                row.target_slot_id == legacy_ret->id &&
                                row.source_id == legacy_ret->id) {
                                row.note_reject(
                                    COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
                                break;
                            }
                        }
                        ret = planned_ret;
                        incoming_adapter_matches = planned_adapter_matches;
                        update_cache = false;
                        SLT_INF(*ret,
                            "selected slot by cache-plan %s authority (legacy slot %d)\n",
                            common_cache_plan_selection_name(
                                plan_rec->selection),
                            legacy_ret->id);
                    }
                    ret->cache_plan_execution = execution;
                } catch (...) {
                    cache_plan_authority->fail_closed(*plan_rec);
                    ret->cache_plan_execution.clear();
                    if (cache_plan_obs) {
                        cache_plan_obs->shadow_unavailable++;
                    }
                }
            }

            // B0: the record follows the chosen slot; stages 2-3 and finalize find it there
            if (plan_rec) {
                plan_rec->id_slot = ret->id;
                // the slot the request landed on is the shipped live-slot selection; the
                // whole fleet was visited without short-circuit, so the live-slot inventory
                // is complete over the declared domain
                plan_rec->select(common_cache_plan_provider::live_slot,
                                 plan_rec->find_or_add(common_cache_plan_provider::live_slot,
                                                       ret->id, uint8_t(0), ret->id,
                                                       plan_rec->selection));
                plan_rec->note_inventory_complete(common_cache_plan_provider::live_slot);
                ret->cache_plan = std::move(plan_rec);
            }

            recurrent_shrink_for_prefill("before prompt cache save/load");

            // note: prompt_save() itself is a no-op when the slot's context is empty

            // A deferred busy slot retains the directive of its in-flight
            // request. It is reachable here, but must not execute that plan a
            // second time for the deferred task.
            common_cache_plan_record * request_cache_plan =
                selection_deferred_busy ? nullptr : ret->cache_plan.get();
            bool authority_exec =
                !selection_deferred_busy &&
                ret->cache_plan_execution.authoritative();

            update_cache = update_cache && prompt_cache;
            update_cache = update_cache &&
                           task.type == SERVER_TASK_TYPE_COMPLETION;

            int64_t prompt_cache_t_start = 0;
            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");
                prompt_cache_t_start = ggml_time_us();

                // Saving the displaced state is an independent durability
                // side effect of the shipped path. Authority changes which
                // complete reuse plan runs; it does not gain D-A destruction
                // authority to skip this save.
                ret->prompt_save(*prompt_cache);

                if (!incoming_adapter_ready) {
                    cache_plan_derive_incoming_adapter(
                        task, *ret, stage1_inventory);
                }
                if (request_cache_plan) {
                    request_cache_plan->identity.adapter_config_digest =
                        llama_cache_acct_value::measured(
                            std::hash<std::string>{}(incoming_adapter));
                }

                // save/publish can deduplicate or evict host nodes. Revalidate
                // the exact source after that mutation and before the required
                // load. A soft miss is a typed legacy fallback, not a restore
                // failure and never a mandatory slot reset.
                if (authority_exec &&
                    (ret->cache_plan_execution.kind ==
                         server_cache_plan_execution_kind::host_restore ||
                     ret->cache_plan_execution.kind ==
                         server_cache_plan_execution_kind::host_checkpoint_restore) &&
                    !cache_plan_execution_revalidate(
                        task, *ret, incoming_adapter_matches,
                        incoming_adapter, ret->cache_plan_execution)) {
                    cache_plan_fallback_legacy(
                        *ret,
                        common_cache_plan_authority_fallback::stale_capability);
                    authority_exec = false;
                }
            }

            if (authority_exec) {
                GGML_ASSERT(request_cache_plan != nullptr);
                switch (ret->cache_plan_execution.kind) {
                    case server_cache_plan_execution_kind::host_restore:
                    case server_cache_plan_execution_kind::host_checkpoint_restore:
                        GGML_ASSERT(prompt_cache && incoming_adapter_ready);
                        if (!ret->prompt_load(
                                *prompt_cache, task.tokens, incoming_adapter,
                                request_cache_plan,
                                ret->cache_plan_execution.host_source_id)) {
                            cache_plan_host_restore_failed(
                                *ret, request_cache_plan);
                        }
                        break;
                    case server_cache_plan_execution_kind::cold_replay:
                        if (!request_cache_plan ||
                            request_cache_plan->destruction.state !=
                                common_cache_plan_destruction_state::executed) {
                            ret->prompt_clear(
                                server_cache_destruction_reason::slot_rebind);
                        }
                        break;
                    case server_cache_plan_execution_kind::live_replay:
                    case server_cache_plan_execution_kind::checkpoint_restore:
                        // The selected live state is already resident. The
                        // checkpoint route is applied later at the existing
                        // restore seam; neither plan mutates host-cache state.
                        break;
                    case server_cache_plan_execution_kind::legacy:
                    case server_cache_plan_execution_kind::_count:
                        GGML_ABORT("invalid B-A1 authority execution");
                }
            }

            if (update_cache && !authority_exec) {
                // Only restore from the host cache when the selected slot's current adapter identity
                // already matches the incoming request's [finding 1]. Otherwise launch_slot_with_task
                // clears this slot for the adapter change, so a restore here would consume a cache
                // entry only to have it immediately wiped. (Cross-identity reuse -- moving a request
                // onto a differently-bound LRU slot -- needs identity-aware selection, deferred.)
                if (incoming_adapter_matches) {
                    // B0 stage-2: per-entry host rows ride the shipped lookup [B-a/A2]
                    if (!ret->prompt_load(*prompt_cache, task.tokens, incoming_adapter,
                                          request_cache_plan)) {
                        cache_plan_host_restore_failed(
                            *ret, request_cache_plan);
                    }
                } else if (request_cache_plan) {
                    // shipped path skips the lookup entirely for the adapter change [finding 1]:
                    // no entries were scanned (empty declared domain) — the classified fact is
                    // provider-level, carried by one aggregate row (source -1)
                    auto * row = request_cache_plan->find_or_add(
                        common_cache_plan_provider::host_cache_entry,
                        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, uint8_t(0), ret->id,
                        request_cache_plan->selection);
                    if (row) {
                        row->note_reject(COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);
                    }
                    request_cache_plan->note_inventory_complete(common_cache_plan_provider::host_cache_entry);
                }
            }
            if (update_cache) {
                prompt_cache->update();
                SRV_TRC("prompt cache update took %.2f ms\n",
                        (ggml_time_us() - prompt_cache_t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                // E1.1c approved guard, not D-A certification.
                if (slot.hard_lease_blocks_live_prefix()) {
                    SLT_INF(slot, "%s", "idle purge skipped: hard lease seals the live prefix\n");
                    continue;
                }
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(
                    server_cache_destruction_reason::idle_reclaim);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        common_cache_family_binding incoming_family;
        if (server_cache_family_resolve_for_launch(
                cache_control_authority.get(),
                task.cache_family_binding_token,
                incoming_family) != server_cache_control_status::ok) {
            send_error(
                task, "cache family binding is unavailable",
                ERROR_TYPE_INVALID_REQUEST);
            return false;
        }


        // process per-request lora adapters (fall back to the server's base adapters when the
        // request carries none). Identity is checked for BOTH branches [I6]: a slot that computed
        // its prompt state under one adapter set must not silently continue it under a different
        // one. The previous code checked equality only for the per-request branch and rebound the
        // no-lora branch unconditionally, so a request with no `lora` field could keep the KV/
        // recurrent state built under a per-request adapter. Also use prompt_clear() (not the
        // struct-only prompt.clear()) so the recurrent state is reset too on hybrid models — a
        // token-only clear leaves stale recurrent state that the reprocess would continue from.
        const auto task_loras = task.params.lora.empty()
            ? params_base.lora_adapters
            : construct_lora_list(task.params.lora);


        if (!are_lora_equal(task_loras, slot.lora)) {
            // called only after establishing inequality, as lora_should_clear_cache requires
            if (lora_should_clear_cache(slot.lora, task_loras)) {
                SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task_loras.size());
                slot.prompt_clear();
            } else {
                SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
            }
            slot.lora = task_loras;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // diffusion self-spec needs raw logits for acceptance checking
            backend_sampling &= !slot.diff_self_spec;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        if (slot.diff_self_spec && task.tokens.size() > 0) {
            // Work at the token level to avoid detokenize→retokenize roundtrip issues.
            llama_tokens toks(task.tokens.get_tokens());
            const llama_token think_open  = slot.diff_think_open_id;
            const llama_token think_close = slot.diff_think_close_id;
            auto nl_vec = common_tokenize(vocab, "\n", false, false);
            const llama_token nl_tok = nl_vec.empty() ? LLAMA_TOKEN_NULL : nl_vec[0];

            // Strip ALL closed <think>...</think> blocks (compact or newlined).
            // The Jinja template injects them into every assistant message but they
            // cause verbatim repetition in multi-turn. We re-add the correct form at the end.
            if (think_open != LLAMA_TOKEN_NULL && think_close != LLAMA_TOKEN_NULL) {
                llama_tokens fixed;
                fixed.reserve(toks.size());
                for (size_t ti = 0; ti < toks.size(); ti++) {
                    if (toks[ti] == think_open) {
                        size_t end = ti + 1;
                        if (end < toks.size() && nl_tok != LLAMA_TOKEN_NULL && toks[end] == nl_tok) end++;
                        if (end < toks.size() && toks[end] == think_close) {
                            end++;
                            if (end < toks.size() && nl_tok != LLAMA_TOKEN_NULL && toks[end] == nl_tok) end++;
                            ti = end - 1; // skip the entire closed think block
                            continue;
                        }
                    }
                    fixed.push_back(toks[ti]);
                }
                toks = std::move(fixed);
            }

            // Truncate looped content from previous assistant responses in history.
            // Once a response loops, the contaminated text poisons all future turns.
            {
                const llama_token im_start = 10, im_end = 11;
                const llama_token tok_ass = 1503, tok_ist = 19464;

                // Find all completed assistant blocks (not the final generation block)
                std::vector<std::pair<int,int>> asst_ranges; // (content_start, im_end_pos)
                for (int i = 0; i + 3 < (int)toks.size(); i++) {
                    if (toks[i] == im_start && toks[i+1] == tok_ass && toks[i+2] == tok_ist) {
                        int content_start = i + 4; // skip: im_start, ass, istant, \n
                        if (content_start >= (int)toks.size()) break;
                        int end_pos = content_start;
                        while (end_pos < (int)toks.size() && toks[end_pos] != im_end) end_pos++;
                        if (end_pos < (int)toks.size()) { // only completed blocks (have im_end)
                            asst_ranges.push_back({content_start, end_pos});
                        }
                    }
                }

                int total_removed = 0;
                for (auto it = asst_ranges.rbegin(); it != asst_ranges.rend(); ++it) {
                    auto [cs, ep] = *it;
                    int len = ep - cs;
                    if (len < 16) continue;

                    int truncate_at = -1;
                    for (int pos = cs + 16; pos <= ep; pos++) {
                        int n = pos - cs;
                        for (int period = 4; period <= std::min(32, n/2); period++) {
                            bool match = true;
                            for (int j = 0; j < period; j++) {
                                if (toks[pos - 1 - j] != toks[pos - 1 - j - period]) {
                                    match = false;
                                    break;
                                }
                            }
                            if (match) {
                                truncate_at = pos - 2 * period;
                                goto found_loop;
                            }
                        }
                    }
                    found_loop:
                    if (truncate_at > cs && truncate_at < ep) {
                        int remove_count = ep - truncate_at;
                        toks.erase(toks.begin() + truncate_at, toks.begin() + ep);
                        total_removed += remove_count;
                    }
                }
                if (total_removed > 0) {
                    SLT_INF(slot, "diff: stripped %d looped tokens from %d assistant blocks in prompt\n",
                            total_removed, (int)asst_ranges.size());
                }
            }

            // Strip trailing open <think> not followed by </think>
            {
                int think_pos = -1;
                for (int ti = (int)toks.size() - 1; ti >= 0; ti--) {
                    if (toks[ti] == think_open) { think_pos = ti; break; }
                }
                if (think_pos >= 0) {
                    bool has_close = false;
                    for (int ti = think_pos + 1; ti < (int)toks.size(); ti++) {
                        if (toks[ti] == think_close) { has_close = true; break; }
                    }
                    if (!has_close) {
                        toks.resize(think_pos);
                    }
                }
            }

            // Ensure prompt ends with <think>\n</think>\n
            {
                int n = (int)toks.size();
                bool has_think_block = (n >= 4 && nl_tok != LLAMA_TOKEN_NULL &&
                    toks[n-4] == think_open && toks[n-3] == nl_tok &&
                    toks[n-2] == think_close && toks[n-1] == nl_tok);
                if (!has_think_block && nl_tok != LLAMA_TOKEN_NULL) {
                    toks.push_back(think_open);
                    toks.push_back(nl_tok);
                    toks.push_back(think_close);
                    toks.push_back(nl_tok);
                }
            }

            task.tokens = server_tokens(toks, task.tokens.has_mtmd);
        }

        // The binding follows retained immutable content. Only a full-prefix
        // append/resume keeps the existing lineage; a branch/replacement
        // adopts the incoming declaration even when BOS or a shared system
        // prefix overlaps. Children carry the opaque token and later copy the
        // parent's resolved slot binding.
        const size_t retained_prefix =
            slot.prompt.tokens.get_common_prefix(task.tokens);
        // A live-prefix lease binds the execution/adapter/media identity while
        // its proven token boundary lives in the lease frontier. Preserve the
        // sidecar artifact only when the entire decoded ledger is an exact
        // prefix of the incoming request. That is an append of the same
        // physical live copy: inspect must report partially_stale until renew.
        // A trim, branch, or zero-overlap replacement retires the artifact and
        // keeps subject_lost as the fail-closed identity-change terminal.
        const bool append_continuity =
            !slot.prompt.tokens.empty() &&
            retained_prefix == slot.prompt.tokens.size();
        if (slot.retention_obs && !append_continuity) {
            slot.retention_obs->retire(
                server_retention_instance_key::for_slot(slot.id));
        }
        slot.cache_family = common_cache_family_follow_lineage(
            slot.cache_family, incoming_family, retained_prefix,
            slot.prompt.tokens.size());
        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // Gate slot save/restore/erase on slot content (does it hold media),
    // not model capability: a multimodal model may hold a pure-text slot.
    bool check_slot_no_media(const server_slot & slot, const int id_task) {
        if (slot.prompt.tokens.has_media()) {
            send_error(id_task,
                "This operation is not supported while the slot holds image/audio tokens (a pure-text prefix is supported)",
                ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        // cache receipt (§7.7): keyed chained block-hash over the slot's cached
        // token prefix + frontier identity. Untrusted hint, never authorization.
        // Portable tokenizer/template digests are §7.2 scope; the chain is
        // versioned by hash_spec + model identity here.
        if (params_base.cache_receipt) {
            const llama_tokens receipt_tokens = slot.prompt.tokens.get_text_tokens();
            std::string identity = lora_config_identity(slot.lora);
            if (identity == "0") {
                identity = "base:no-lora";
            }
            const bool keyed = !params_base.cache_receipt_key.empty();
            res->cache_receipt = json {
                {"version",          1},
                {"hash_spec",        keyed ? "sha256-chain-trunc64/v1"
                                           : "sha256-chain-trunc64-unkeyed-debug/v1"},
                {"block_tokens",     1024},
                {"model",            slot.task->params.oaicompat_model.empty()
                                         ? params_base.model.path
                                         : slot.task->params.oaicompat_model},
                {"sequence_epoch",   slot.prompt.sequence_epoch},
                {"token_count",      (int64_t) receipt_tokens.size()},
                {"adapter_identity", identity},
                {"chain",            cache_receipt_chain(
                                         receipt_tokens, 1024,
                                         params_base.cache_receipt_key)},
            };
        }

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    // note: the former create_checkpoint() helper was dead (zero callers) and diverged from the live
    // capture path below (it alone populated data_dft/spec state via update_dft/common_speculative_get_state,
    // which the live path never fills). Removed [R6/N5]; the single capture path is the inline one in
    // update_slots. Draft/speculative checkpoint aux state now lives in the typed
    // common_prompt_checkpoint::accel record (Phase-1 typed accelerators): ring is
    // mandatory-on-presence (fail-closed at the restore site), spec is optional.
    // The live capture path fills accel.ring only; accel.spec remains a stash slot
    // for spec impls that need it (e.g. eagle3), applied on restore when present.

    void assert_scheduler_thread(
            std::thread::id & pinned,
            const char * what) {
        const auto current = std::this_thread::get_id();
        if (pinned == std::thread::id{}) {
            pinned = current;
        }
        if (pinned != current) {
            GGML_ABORT("%s\n", what);
        }
    }

    bool cache_control_refresh_prompt(
            const server_prompt & prompt,
            const std::string & adapter_identity,
            server_cache_lease_identity & identity,
            server_cache_lease_frontier & frontier) noexcept {
        const int64_t tokens = prompt.n_tokens();
        if (!server_cache_lease_build_identity(
                frontier_execution_identity, adapter_identity,
                prompt.tokens, tokens, identity)) {
            return false;
        }
        frontier = {
            prompt.sequence_epoch, uint64_t(tokens), tokens,
        };
        return frontier.valid();
    }

    bool cache_control_refresh_subject(
            const server_cache_control_selector & selector,
            server_cache_lease_identity & identity,
            server_cache_lease_frontier & frontier) noexcept {
        if (selector.kind == server_cache_control_subject_kind::live_prefix) {
            const auto slot = std::find_if(
                slots.begin(), slots.end(), [&](const server_slot & value) {
                    return value.id == selector.retention_key.owner_slot;
                });
            if (slot == slots.end()) {
                return false;
            }
            return cache_control_refresh_prompt(
                slot->prompt, lora_config_identity(slot->lora),
                identity, frontier);
        }
        if (selector.kind == server_cache_control_subject_kind::host_snapshot) {
            if (!prompt_cache) {
                return false;
            }
            const auto * wanted =
                reinterpret_cast<const server_prompt_cache_state *>(
                    selector.retention_key.instance);
            const auto state = std::find_if(
                prompt_cache->states.begin(), prompt_cache->states.end(),
                [&](const server_prompt_cache_state & value) {
                    return &value == wanted;
                });
            if (state == prompt_cache->states.end()) {
                return false;
            }
            return cache_control_refresh_prompt(
                state->prompt, state->adapter_config_key,
                identity, frontier);
        }
        return false;
    }

    server_cache_control_status cache_control_resolve_semantic_selector(
            server_cache_control_selector & selector,
            const server_task::cache_control_semantic_selector & semantic) noexcept {
        if (selector.kind == server_cache_control_subject_kind::vbr_reference ||
            selector.kind == server_cache_control_subject_kind::live_checkpoint) {
            return server_cache_control_status::ok;
        }
        if (selector.kind == server_cache_control_subject_kind::live_prefix) {
            const auto slot = std::find_if(
                slots.begin(), slots.end(), [&](const server_slot & value) {
                    return value.id == semantic.slot_id;
                });
            if (slot == slots.end() || slot->prompt.n_tokens() == 0) {
                return server_cache_control_status::not_found;
            }
            if (slot->is_processing()) {
                return server_cache_control_status::subject_busy;
            }
            selector.retention_key =
                server_retention_instance_key::for_slot(slot->id);
            return cache_control_refresh_prompt(
                slot->prompt, lora_config_identity(slot->lora),
                selector.identity, selector.frontier)
                    ? server_cache_control_status::ok
                    : server_cache_control_status::identity_unavailable;
        }
        if (selector.kind == server_cache_control_subject_kind::host_snapshot) {
            if (!prompt_cache || !semantic.tokens) {
                return server_cache_control_status::not_found;
            }
            const auto requested_lora = semantic.lora.empty()
                ? params_base.lora_adapters
                : construct_lora_list(semantic.lora);
            const std::string adapter_identity =
                lora_config_identity(requested_lora);
            const auto state = std::find_if(
                prompt_cache->states.begin(), prompt_cache->states.end(),
                [&](const server_prompt_cache_state & value) {
                    return value.adapter_config_key == adapter_identity &&
                        value.prompt.tokens.size() == semantic.tokens->size() &&
                        value.prompt.tokens.get_common_prefix(*semantic.tokens) ==
                            semantic.tokens->size();
                });
            if (state == prompt_cache->states.end()) {
                return server_cache_control_status::not_found;
            }
            selector.retention_key =
                server_retention_instance_key::for_host_entry(&*state);
            return cache_control_refresh_prompt(
                state->prompt, state->adapter_config_key,
                selector.identity, selector.frontier)
                    ? server_cache_control_status::ok
                    : server_cache_control_status::identity_unavailable;
        }
        return server_cache_control_status::not_supported;
    }

    bool cache_plan_preflight_inputs_current(
            const common_cache_plan_record & rec,
            const server_slot & legacy_target,
            const cache_plan_stage1_inventory & inventory) noexcept {
        if (rec.planner_status != common_cache_plan_planner_status::ok ||
            !server_cache_plan_shadow_choice_valid(rec)) {
            return false;
        }
        const auto & selected = rec.inventory[size_t(rec.shadow_choice)];
        server_slot * planned = cache_plan_slot_by_exact_id(
            selected.target_slot_id);
        const bool adapter_matches = planned &&
            are_lora_equal(inventory.incoming_loras, planned->lora);
        return cache_plan_planned_slot_current(
            rec, legacy_target, planned, selected.provider,
            adapter_matches);
    }

    server_cache_plan_preflight_view cache_plan_preflight(
            const server_task & task) {
        assert_scheduler_thread(
            cache_plan_preflight_scheduler_thread,
            "cache-plan preflight must run on scheduler thread");
        server_cache_plan_preflight_view view;
        std::optional<server_cache_plan_authority> local_authority;
        server_cache_plan_authority * plan_authority =
            cache_plan_authority.get();
        if (!plan_authority) {
            // E0 must not allocate the production per-request observer merely
            // because a preflight exists. Reuse an already-composed lifecycle
            // profile when present; otherwise the local planner reports the
            // typed no_profile refusal.
            local_authority.emplace(params_base.cache_plan_authority);
            local_authority->calibration_profile =
                cache_plan_calibration_profile;
            plan_authority = &*local_authority;
        }

        auto stage1 = cache_plan_select_before_mutation(
            task, plan_authority, true);
        if (!stage1.target) {
            view.status = server_cache_plan_preflight_status::no_target;
            return view;
        }
        if (!stage1.record) {
            view.status = server_cache_plan_preflight_status::internal_fault;
            return view;
        }

        cache_plan_stage1_inventory inventory;
        cache_plan_inventory_and_plan_before_mutation(
            task, *stage1.target, stage1.update_cache, *stage1.record,
            inventory,
            cache_plan_stage1_mode_for(plan_authority, true));
        const bool inputs_current = cache_plan_preflight_inputs_current(
            *stage1.record, *stage1.target, inventory);
        if (!server_cache_plan_preflight_build_view(
                *stage1.record, stage1.target->id, inputs_current, view)) {
            view.status = server_cache_plan_preflight_status::internal_fault;
        }
        return view;
    }

    bool build_capture_request(
            server_slot & slot,
            vbr_explicit_capture_request & request,
            server_vbr_artifact_capture_status & status) {
        const int64_t token_count = slot.prompt.n_tokens();
        const llama_pos next_position =
            slot.prompt.tokens.pos_next();
        std::string media_identity;
        if (next_position < 0 ||
            !slot.prompt.tokens.media_content_identity(
                token_count, media_identity)) {
            status =
                server_vbr_artifact_capture_status::identity_unavailable;
            return false;
        }
        const uint64_t sequence_epoch =
            ensure_frontier_sequence_epoch(slot.prompt);
        request.identity = {
            frontier_execution_identity,
            lora_config_identity(slot.lora),
            std::move(media_identity),
            sequence_epoch,
            token_count,
            next_position,
        };
        request.token_block.reserve(size_t(token_count));
        for (int64_t i = 0; i < token_count; ++i) {
            request.token_block.push_back(slot.prompt.tokens[size_t(i)]);
        }
        request.sequence = slot.id;
        request.frontier.execution_identity =
            request.identity.execution_identity.data();
        request.frontier.execution_identity_len =
            request.identity.execution_identity.size();
        request.frontier.adapter_config_identity =
            request.identity.adapter_config_identity.data();
        request.frontier.adapter_config_identity_len =
            request.identity.adapter_config_identity.size();
        request.frontier.media_content_identity =
            request.identity.media_content_identity.data();
        request.frontier.media_content_identity_len =
            request.identity.media_content_identity.size();
        request.frontier.sequence_epoch = sequence_epoch;
        request.frontier.token_count = token_count;
        request.frontier.next_position = next_position;
        request.idle_decode_thread = true;

        const auto pageable_domain =
            vbr_artifact_portable_domain {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX, UINT16_MAX,
            };
        if (ctx_dft) {
            vbr_explicit_companion_provider draft;
            draft.kind =
                vbr_artifact_companion_kind::required_spec_payload;
            draft.build_identity_digest =
                server_cache_capture_build_digest(
                    "buun.vbr.draft-state-codec/v1");
            draft.domain = pageable_domain;
            draft.context = ctx_dft.get();
            draft.size = [](
                    const void * context,
                    llama_seq_id sequence,
                    uint64_t & output) noexcept {
                const auto size =
                    llama_state_seq_get_size_ext(
                        static_cast<llama_context *>(
                            const_cast<void *>(context)),
                        sequence,
                        LLAMA_STATE_SEQ_FLAGS_NONE);
                output = uint64_t(size);
                return size != 0;
            };
            draft.capture = [](
                    const void * context,
                    llama_seq_id sequence,
                    std::vector<uint8_t> & output) noexcept {
                try {
                    auto * ctx =
                        static_cast<llama_context *>(
                            const_cast<void *>(context));
                    const size_t size =
                        llama_state_seq_get_size_ext(
                            ctx, sequence,
                            LLAMA_STATE_SEQ_FLAGS_NONE);
                    if (size == 0) {
                        return false;
                    }
                    output.resize(size);
                    return llama_state_seq_get_data_ext(
                               ctx, output.data(), size, sequence,
                               LLAMA_STATE_SEQ_FLAGS_NONE) == size;
                } catch (...) {
                    output.clear();
                    return false;
                }
            };
            request.companions.push_back(draft);
        }
        if (slot.can_speculate()) {
            const size_t ring_size =
                common_speculative_ring_state_size(slot.get_spec());
            if (ring_size != 0) {
                vbr_explicit_companion_provider accelerator;
                accelerator.kind =
                    vbr_artifact_companion_kind::typed_accelerator;
                accelerator.build_identity_digest =
                    server_cache_capture_build_digest(
                        "buun.vbr.accelerator-ring-codec/v1");
                accelerator.domain = pageable_domain;
                accelerator.context = &slot;
                accelerator.size = [](
                        const void * context,
                        llama_seq_id,
                        uint64_t & output) noexcept {
                    const auto * owner =
                        static_cast<const server_slot *>(context);
                    const size_t size =
                        common_speculative_ring_state_size(
                            owner->get_spec());
                    output = uint64_t(size);
                    return size != 0;
                };
                accelerator.capture = [](
                        const void * context,
                        llama_seq_id,
                        std::vector<uint8_t> & output) noexcept {
                    try {
                        const auto * owner =
                            static_cast<const server_slot *>(context);
                        const size_t size =
                            common_speculative_ring_state_size(
                                owner->get_spec());
                        if (size == 0) {
                            return false;
                        }
                        output.resize(size);
                        common_speculative_ring_state_save(
                            owner->get_spec(),
                            output.data(), output.size());
                        return true;
                    } catch (...) {
                        output.clear();
                        return false;
                    }
                };
                request.companions.push_back(accelerator);
            }
        }
        status = server_vbr_artifact_capture_status::ok;
        return true;
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            server_cache_plan_disarm_unlaunched(
                                slot->cache_plan_execution, slot->cache_plan,
                                slot->cache_plan_destruction_recovery_pin);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            server_cache_plan_disarm_unlaunched(
                                slot->cache_plan_execution, slot->cache_plan,
                                slot->cache_plan_destruction_recovery_pin);
                            break; // drop the task
                        }
                    } else {
                        // Unified KV: check if launching this task would overflow the shared cell pool.
                        // Use max(current, planned) since a just-launched slot hasn't filled yet.
                        if (params_base.kv_unified && task.n_tokens() > 0) {
                            int64_t cells_committed = 0;
                            for (const auto & s : slots) {
                                if (s.is_processing() && s.task) {
                                    cells_committed += std::max((int64_t) s.prompt.n_tokens(), (int64_t) s.task->n_tokens());
                                }
                            }
                            const int64_t cells_available = (int64_t) slot->n_ctx - cells_committed;
                            if (cells_available < (int64_t) task.n_tokens()) {
                                // never-fits: reject instead of deferring forever (silent hang)
                                if (task.n_tokens() > slot->n_ctx) {
                                    send_error(task.id,
                                               string_format(
                                                   "request (%d tokens) exceeds the total context size (%d tokens), try increasing it",
                                                   task.n_tokens(), slot->n_ctx),
                                                   ERROR_TYPE_EXCEED_CONTEXT_SIZE, task.n_tokens(), slot->n_ctx);
                                    server_cache_plan_disarm_unlaunched(
                                        slot->cache_plan_execution,
                                        slot->cache_plan,
                                        slot->cache_plan_destruction_recovery_pin);
                                    break; // drop the task
                                }
                                SRV_DBG("defer task %d: needs %d tokens but only %" PRId64 " cells available (%" PRId64 " committed by active slots)\n",
                                        id_task, task.n_tokens(), cells_available, cells_committed);
                                server_cache_plan_disarm_unlaunched(
                                    slot->cache_plan_execution,
                                    slot->cache_plan,
                                    slot->cache_plan_destruction_recovery_pin);
                                queue_tasks.defer(std::move(task));
                                break;
                            }
                        }

                        // dynamic VBR: if this launch's projected footprint would degrade the
                        // pool below the quality floor, clear idle caches first. The probe is
                        // sized by the post-LCP SUFFIX — the common prefix refills in place with
                        // zero cell growth, and probing the full prompt evicted other clients'
                        // caches for growth that was never going to happen.
                        if (server_vbr_dynamic_active(params_base) && task.n_tokens() > 0) {
                            const size_t lcp = slot->prompt.tokens.get_common_prefix(task.tokens);
                            vbr_reclaim_before_degrade(slot->id, (uint32_t) (task.n_tokens() - lcp), "launch");
                        }

                        if (!launch_slot_with_task(*slot, std::move(task))) {
                            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                            server_cache_plan_disarm_unlaunched(
                                slot->cache_plan_execution, slot->cache_plan,
                                slot->cache_plan_destruction_recovery_pin);
                            break; // drop the task
                        }
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                const prompt_save_result saved = slot.prompt_save(*prompt_cache);
                                if (saved == prompt_save_result::published) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                }

                                // clear the live slot only if its state is now durable in the cache
                                // [I7]; a failed save must not destroy the only copy of the state
                                if (params_base.kv_unified && prompt_save_durable(saved)) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear(
                                        server_cache_destruction_reason::idle_reclaim);
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    res->n_draft_tokens_total      = metrics.n_draft_tokens_total;
                    res->n_draft_accepted_total    = metrics.n_draft_accepted_total;
                    res->n_draft_verif_steps_total = metrics.n_draft_verif_steps_total;
                    res->n_accepted_per_pos_total  = metrics.n_accepted_per_pos_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!check_slot_no_media(*slot, task.id)) {
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens tokens = slot->prompt.tokens.get_text_tokens();
                    const size_t token_count = tokens.size();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    // Server semantic envelope [I10]: the lib slot file carries the checksummed state
                    // but no adapter identity, and a restore into a slot bound to a different adapter
                    // is NOT caught by the launch-time check (that compares against slot.lora, not the
                    // file). Record the adapter identity this state was computed under in a sidecar so
                    // the restore can reject a cross-adapter mismatch.
                    if (nwrite > 0) {
                        std::ofstream f(filepath + ".lora", std::ios::binary | std::ios::trunc);
                        if (f) {
                            f << lora_config_identity(slot->lora);
                        }
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_CACHE_PLAN_PREFLIGHT:
                {
                    auto res = std::make_unique<
                        server_task_result_cache_plan_preflight>();
                    res->id = task.id;
                    try {
                        res->view = cache_plan_preflight(task);
                    } catch (...) {
                        res->view.status =
                            server_cache_plan_preflight_status::internal_fault;
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_CACHE_HOLDER_CREATE:
            case SERVER_TASK_TYPE_CACHE_HOLDER_CLOSE:
            case SERVER_TASK_TYPE_CACHE_HOLDER_REATTACH:
            case SERVER_TASK_TYPE_CACHE_FAMILY_REGISTER:
            case SERVER_TASK_TYPE_CACHE_FAMILY_BIND:
            case SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE:
            case SERVER_TASK_TYPE_CACHE_LEASE_INSPECT:
            case SERVER_TASK_TYPE_CACHE_LEASE_RENEW:
            case SERVER_TASK_TYPE_CACHE_LEASE_RELEASE:
            case SERVER_TASK_TYPE_CACHE_CONTROL_EVENTS:
                {
                    auto res = std::make_unique<server_task_result_cache_control>();
                    res->id = task.id;
                    static_assert(
                        SERVER_TASK_TYPE_CACHE_CONTROL_EVENTS -
                                SERVER_TASK_TYPE_CACHE_HOLDER_CREATE + 1 ==
                            int(server_cache_control_operation::_count));
                    const auto operation = static_cast<
                        server_cache_control_operation>(
                            task.type - SERVER_TASK_TYPE_CACHE_HOLDER_CREATE);
                    res->operation = operation;
                    const auto precheck = server_cache_control_task_precheck(
                        task.cache_control != nullptr,
                        params_base.cache_lifecycle,
                        cache_authority != nullptr);
                    if (precheck != server_cache_control_status::ok) {
                        res->result.status = precheck;
                    } else {
                        try {
                            if (!cache_control_authority) {
                                server_cache_control_config config;
                                config.leases = &cache_authority->leases;
                                config.retention = &cache_authority->retention;
                                config.artifacts = vbr_artifact_store.get();
                                config.refresh_context = this;
                                config.refresh_subject = [](void * context,
                                    const server_cache_control_selector & selector,
                                    server_cache_lease_identity & identity,
                                    server_cache_lease_frontier & frontier) noexcept {
                                    return static_cast<server_context_impl *>(context)
                                        ->cache_control_refresh_subject(
                                            selector, identity, frontier);
                                };
                                config.host_proof_context = prompt_cache.get();
                                config.acquire_host_proof = [](void * context,
                                    const server_cache_control_selector & selector) noexcept {
                                    auto * cache = static_cast<server_prompt_cache *>(context);
                                    return cache
                                        ? server_prompt_cache_host_fallback_proof(
                                            *cache, selector)
                                        : server_cache_durable_fallback_proof{};
                                };
                                config.selector_evidence_context = prompt_cache.get();
                                config.selector_evidence = [](void *,
                                    const server_cache_control_selector & selector,
                                    uint64_t & bytes, bool & shared) noexcept {
                                    bytes = 0;
                                    shared = false;
                                    if (selector.kind !=
                                            server_cache_control_subject_kind::host_snapshot ||
                                        selector.retention_key.kind !=
                                            common_retention_artifact_kind::host_entry ||
                                        selector.retention_key.instance == 0) {
                                        return false;
                                    }
                                    const auto * state = reinterpret_cast<
                                        const server_prompt_cache_state *>(
                                            selector.retention_key.instance);
                                    bytes = state->size();
                                    shared = state->recovery_pins > 1;
                                    return true;
                                };
                                cache_control_authority = std::make_unique<
                                    server_cache_control_authority>(config);
                            }
                            server_cache_control_request request =
                                *task.cache_control;
                            server_cache_control_status selector_status =
                                server_cache_control_status::ok;
                            if (operation ==
                                    server_cache_control_operation::lease_acquire) {
                                if (selector_status ==
                                        server_cache_control_status::ok) {
                                    selector_status =
                                        cache_control_resolve_semantic_selector(
                                            request.subject,
                                            task.cache_control_subject);
                                }
                                if (selector_status ==
                                        server_cache_control_status::ok &&
                                    request.requested_class ==
                                        server_cache_lease_class::hard) {
                                    selector_status =
                                        cache_control_resolve_semantic_selector(
                                            request.fallback,
                                            task.cache_control_fallback);
                                    if (selector_status !=
                                            server_cache_control_status::ok &&
                                        request.allow_soft_fallback) {
                                        // Keep the unresolved exact selector:
                                        // the authority remains the sole policy
                                        // point for an explicitly authorized
                                        // hard->soft downgrade and reports the
                                        // effective class.
                                        selector_status =
                                            server_cache_control_status::ok;
                                    } else if (selector_status ==
                                            server_cache_control_status::not_found) {
                                        selector_status =
                                            server_cache_control_status::fallback_unavailable;
                                    }
                                }
                            } else if (operation ==
                                    server_cache_control_operation::lease_renew &&
                                request.fallback.kind <
                                    server_cache_control_subject_kind::_count) {
                                selector_status =
                                    cache_control_resolve_semantic_selector(
                                        request.fallback,
                                        task.cache_control_fallback);
                                if (selector_status ==
                                        server_cache_control_status::not_found) {
                                    selector_status =
                                        server_cache_control_status::fallback_unavailable;
                                }
                            }
                            if (selector_status !=
                                    server_cache_control_status::ok) {
                                res->result.status = selector_status;
                            } else {
                                res->result = cache_control_authority->execute(
                                    operation, request);
                            }
                        } catch (...) {
                            res->result.status =
                                server_cache_control_status::internal_fault;
                        }
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_CACHE_CAPTURE:
                {
                    auto send_capture = [&](server_vbr_artifact_capture_status status) {
                        auto res =
                            std::make_unique<server_task_result_cache_capture>();
                        res->id = task.id;
                        res->id_slot = task.cache_capture.id_slot;
                        res->status = status;
                        queue_results.send(std::move(res));
                    };

                    // The route remains present so gated configurations return
                    // a typed result, but no slot/model inspection occurs when
                    // the full lifecycle+armed construction gate did not pass.
                    if (!vbr_artifact_store) {
                        send_capture(
                            server_vbr_artifact_capture_status::unsupported);
                        break;
                    }
                    try {
                    assert_scheduler_thread(
                        vbr_capture_scheduler_thread,
                        "VBR artifact work must run on scheduler thread");

                    server_slot * slot =
                        get_slot_by_id(task.cache_capture.id_slot);
                    if (slot == nullptr) {
                        send_capture(
                            server_vbr_artifact_capture_status::invalid_slot);
                        break;
                    }
                    if (slot->is_processing()) {
                        SRV_DBG(
                            "requested capture slot is processing; defer task, id_task = %d\n",
                            task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    if (slot->state != SLOT_STATE_IDLE ||
                        slot->prompt.n_tokens() <= 0) {
                        send_capture(
                            server_vbr_artifact_capture_status::stale_frontier);
                        break;
                    }

                    vbr_explicit_capture_request request;
                    server_vbr_artifact_capture_status request_status;
                    if (!build_capture_request(
                            *slot, request, request_status)) {
                        send_capture(request_status);
                        break;
                    }

                    if (!cache_plan_observe_live_memory(false)) {
                        send_capture(
                            server_vbr_artifact_capture_status::unavailable);
                        break;
                    }
                    SRV_INF(
                        "VBR_ARTIFACT_CAPTURE begin task=%d slot=%d controllers=%u\n",
                        task.id, slot->id,
                        vbr_artifact_store->attention_children());
                    const int64_t started = ggml_time_us();
                    const auto captured = vbr_artifact_store->capture(
                        *llama_get_memory(ctx_tgt), std::move(request),
                        task.cache_capture.tenant_key);
                    const auto & capture_totals =
                        vbr_artifact_store->counters();
                    const double duration_ms =
                        (ggml_time_us() - started)/1000.0;
                    SRV_INF(
                        "VBR_ARTIFACT_CAPTURE end task=%d slot=%d status=%s "
                        "library_status=%s phase=%s inner_status=%s "
                        "generation_failure=%s size_failure=%s "
                        "reservation_group=%s prepare_status=%s "
                        "admission_status=%s failed_leaf=%zu "
                        "consistency=%s units=%u payload=%" PRIu64
                        " stash=%" PRIu64 " companion=%" PRIu64
                        " chunks=%" PRIu64 " backpressure=%" PRIu64
                        " events=%" PRIu64 " sync_fallbacks=%" PRIu64
                        " dedup=%d duration_ms=%.3f"
                        " total_requested=%" PRIu64
                        " total_published=%" PRIu64
                        " total_refused=%" PRIu64
                        " total_unavailable=%" PRIu64
                        " total_internal=%" PRIu64
                        " total_payload=%" PRIu64
                        " total_stash=%" PRIu64
                        " total_companion=%" PRIu64
                        " pinned_bytes=%" PRIu64
                        " total_chunks=%" PRIu64
                        " total_events=%" PRIu64
                        " total_sync_fallbacks=%" PRIu64
                        " total_backpressure=%" PRIu64
                        " total_dedup_hits=%" PRIu64
                        " total_dedup_misses=%" PRIu64
                        " staging_overlap_refusals=%" PRIu64 "\n",
                        task.id, slot->id,
                        server_vbr_artifact_capture_status_name(
                            captured.status),
                        vbr_explicit_capture_status_name(
                            captured.library_status),
                        vbr_explicit_capture_phase_name(
                            captured.phase),
                        vbr_capture_stream_status_name(
                            captured.inner_stream_status),
                        vbr_explicit_generation_failure_name(
                            captured.generation_failure),
                        vbr_explicit_size_failure_name(
                            captured.size_failure),
                        vbr_capture_reservation_group_name(
                            captured.begin_diagnostics.
                                reservation_group),
                        llama_cache_prepare_status_name(
                            captured.begin_diagnostics.
                                prepare_status),
                        llama_cache_admission_status_name(
                            captured.begin_diagnostics.
                                admission_status),
                        captured.begin_diagnostics.failed_leaf,
                        captured.status ==
                                server_vbr_artifact_capture_status::ok
                            ? "capture_exact" : "unavailable",
                        captured.units, captured.payload_bytes,
                        captured.stash_bytes,
                        captured.companion_bytes, captured.chunks,
                        captured.backpressure_waits,
                        captured.event_completions,
                        captured.synchronous_fallbacks,
                        captured.dedup, duration_ms,
                        capture_totals.requested,
                        capture_totals.exact_published,
                        capture_totals.refused,
                        capture_totals.unavailable,
                        capture_totals.internal_error,
                        capture_totals.payload_bytes,
                        capture_totals.stash_bytes,
                        capture_totals.companion_bytes,
                        capture_totals.pinned_bytes,
                        capture_totals.chunks,
                        capture_totals.event_completions,
                        capture_totals.synchronous_fallbacks,
                        capture_totals.backpressure_waits,
                        capture_totals.dedup_hits,
                        capture_totals.dedup_misses,
                        capture_totals.staging_overlap_refusals);

                    auto res =
                        std::make_unique<server_task_result_cache_capture>();
                    res->id = task.id;
                    res->id_slot = slot->id;
                    res->status = captured.status;
                    if (captured.status ==
                            server_vbr_artifact_capture_status::ok) {
                        res->consistency =
                            server_cache_capture_consistency::capture_exact;
                    }
                    res->reference = captured.reference;
                    res->controllers = captured.controllers;
                    res->units = captured.units;
                    res->companions = captured.companions;
                    res->payload_bytes = captured.payload_bytes;
                    res->stash_bytes = captured.stash_bytes;
                    res->companion_bytes = captured.companion_bytes;
                    res->chunks = captured.chunks;
                    res->backpressure_waits =
                        captured.backpressure_waits;
                    res->event_completions =
                        captured.event_completions;
                    res->synchronous_fallbacks =
                        captured.synchronous_fallbacks;
                    res->dedup = captured.dedup;
                    queue_results.send(std::move(res));
                    } catch (...) {
                        send_capture(
                            server_vbr_artifact_capture_status::
                                internal_error);
                    }
                } break;
            case SERVER_TASK_TYPE_CACHE_IMPORT:
                {
                    const auto send_import = [&task, this](
                            const server_vbr_artifact_import_output & imported) {
                        auto res =
                            std::make_unique<server_task_result_cache_import>();
                        res->id = task.id;
                        res->id_slot = task.cache_import.id_slot;
                        res->status = imported.status;
                        res->validation_status = imported.validation_status;
                        res->stage_status = imported.stage_status;
                        res->downward_reserve_status =
                            imported.downward_reserve_status;
                        res->adopt_status = imported.adopt_status;
                        res->adopt_attempted = imported.adopt_attempted;
                        res->phase = imported.phase;
                        res->downward_subphase = imported.downward_subphase;
                        res->downward_edge = imported.downward_edge;
                        res->decision = imported.decision;
                        res->consistency = server_cache_import_route_consistency(
                            imported.status, imported.consistency);
                        res->units = imported.units;
                        res->companions = imported.companions;
                        res->payload_bytes = imported.payload_bytes;
                        res->companion_bytes = imported.companion_bytes;
                        queue_results.send(std::move(res));
                    };
                    const auto terminal = [&](server_vbr_artifact_import_status status) {
                        server_vbr_artifact_import_output output;
                        output.status = status;
                        send_import(output);
                    };

                    if (!vbr_artifact_store) {
                        terminal(server_vbr_artifact_import_status::unsupported);
                        break;
                    }
                    try {
                        assert_scheduler_thread(
                            vbr_capture_scheduler_thread,
                            "VBR artifact work must run on scheduler thread");

                        server_slot * slot = get_slot_by_id(
                            task.cache_import.id_slot);
                        llama_memory_i * memory = llama_get_memory(ctx_tgt);
                        const bool slot_empty = slot && memory &&
                            slot->state == SLOT_STATE_IDLE &&
                            slot->prompt.n_tokens() == 0 &&
                            slot->prompt.checkpoints.empty() &&
                            slot->prompt.sequence_epoch == 0 &&
                            memory->seq_pos_min(slot->id) < 0 &&
                            memory->seq_pos_max(slot->id) < 0;
                        const auto precheck =
                            server_vbr_artifact_import_route_precheck(
                                vbr_artifact_store != nullptr, slot != nullptr,
                                slot && slot->is_processing(), memory != nullptr,
                                slot_empty);
                        if (precheck ==
                                server_vbr_artifact_import_status::slot_processing) {
                            SRV_DBG(
                                "requested import slot is processing; defer task, id_task = %d\n",
                                task.id);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (precheck != server_vbr_artifact_import_status::ok) {
                            terminal(precheck);
                            break;
                        }
                        if (!cache_plan_observe_live_memory(false)) {
                            terminal(server_vbr_artifact_import_status::unavailable);
                            break;
                        }

                        struct import_publish_state {
                            server_slot * slot = nullptr;
                            server_prompt prompt;
                            bool ready = false;
                        } publish_state { slot, {}, false };
                        server_vbr_artifact_import_request request;
                        request.memory = memory;
                        request.destination = slot->id;
                        request.reference = task.cache_import.reference;
                        request.tenant_key = task.cache_import.tenant_key;
                        request.execution_identity = frontier_execution_identity;
                        request.adapter_config_identity =
                            lora_config_identity(slot->lora);
                        // An erased construction-empty slot has no prior import
                        // state to preserve. Normal decode history was removed by
                        // the explicit erase and does not force a rebase.
                        // F4 imports erase into a fresh logical target. F5's
                        // persistent receipt/re-import seam will supply this
                        // bit once prior-import evidence has durable authority.
                        request.previously_observed = false;
                        request.publish_context = &publish_state;
                        request.prepare_publish = [](
                                void * opaque,
                                const std::vector<llama_token> & tokens,
                                uint64_t sequence_epoch) noexcept {
                            try {
                                auto * state =
                                    static_cast<import_publish_state *>(opaque);
                                if (!state || !state->slot || state->ready ||
                                    tokens.empty() || sequence_epoch == 0 ||
                                    state->slot->prompt.n_tokens() != 0 ||
                                    !state->slot->prompt.checkpoints.empty() ||
                                    state->slot->prompt.sequence_epoch != 0) {
                                    return false;
                                }
                                // Match SLOT_RESTORE's decode-side establishment:
                                // populate the existing slot prompt rather than
                                // replacing it with a default-constructed one.
                                // In particular, has_mtmd is slot configuration,
                                // not artifact payload. Prepare its value off-side
                                // so the phase-12 publish remains allocation-free.
                                state->prompt.tokens.has_mtmd =
                                    state->slot->prompt.tokens.has_mtmd;
                                state->prompt.tokens.insert(tokens);
                                state->prompt.sequence_epoch = sequence_epoch;
                                state->ready =
                                    state->prompt.n_tokens() ==
                                        int(tokens.size());
                                return state->ready;
                            } catch (...) {
                                return false;
                            }
                        };
                        request.publish = [](void * opaque) noexcept {
                            auto * state =
                                static_cast<import_publish_state *>(opaque);
                            GGML_ASSERT(state && state->slot && state->ready);
                            using std::swap;
                            // SLOT_RESTORE installs its decoded token ledger into
                            // the existing prompt object. Do the same here: a
                            // whole-prompt swap would overwrite slot-owned decode
                            // configuration with server_prompt defaults. The
                            // target was proved empty before prepare_publish, so
                            // checkpoints remain empty and these two no-throw
                            // assignments are the complete publication surface.
                            swap(state->slot->prompt.tokens,
                                 state->prompt.tokens);
                            state->slot->prompt.sequence_epoch =
                                state->prompt.sequence_epoch;
                            // F-reference family provenance is not part of the
                            // v1 artifact metadata. A foreign import therefore
                            // starts undeclared rather than inheriting a stale
                            // binding from the destination slot.
                            state->slot->cache_family = {};
                            state->ready = false;
                        };

                        SRV_INF(
                            "VBR_ARTIFACT_IMPORT begin task=%d slot=%d\n",
                            task.id, slot->id);
                        const int64_t started = ggml_time_us();
                        const auto imported =
                            vbr_artifact_store->import(std::move(request));
                        const auto & totals = vbr_artifact_store->counters();
                        const double duration_ms =
                            (ggml_time_us() - started)/1000.0;
                        const std::string edge_name =
                            imported.downward_edge == UINT32_MAX
                                ? "none"
                                : std::to_string(imported.downward_edge);
                        const char * phase_name = imported.adopt_attempted
                            ? vbr_adopt_phase_name(imported.phase) : "none";
                        const char * subphase_name = imported.adopt_attempted
                            ? vbr_downward_adopt_subphase_name(
                                  imported.downward_subphase)
                            : "none";
                        SRV_INF(
                            "VBR_ARTIFACT_IMPORT end task=%d slot=%d status=%s "
                            "validation=%s stage=%s downward_reserve=%s "
                            "adopt=%s phase=%s subphase=%s edge=%s "
                            "decision=%s consistency=%s units=%u companions=%u "
                            "payload=%" PRIu64 " companion_bytes=%" PRIu64
                            " duration_ms=%.3f total_requested=%" PRIu64
                            " total_succeeded=%" PRIu64
                            " total_report_only=%" PRIu64
                            " total_not_found=%" PRIu64
                            " total_refused=%" PRIu64
                            " total_unavailable=%" PRIu64 "\n",
                            task.id, slot->id,
                            server_vbr_artifact_import_status_name(imported.status),
                            vbr_manifest_validation_status_name(
                                imported.validation_status),
                            vbr_adopt_stage_status_name(imported.stage_status),
                            vbr_downward_reserve_status_name(
                                imported.downward_reserve_status),
                            vbr_adopt_status_name(imported.adopt_status),
                            phase_name, subphase_name,
                            edge_name.c_str(),
                            vbr_import_decision_name(imported.decision),
                            server_cache_import_consistency_name(
                                server_cache_import_route_consistency(
                                    imported.status, imported.consistency)),
                            imported.units, imported.companions,
                            imported.payload_bytes,
                            imported.companion_bytes, duration_ms,
                            totals.imports_requested,
                            totals.imports_succeeded,
                            totals.imports_report_only,
                            totals.imports_not_found,
                            totals.imports_refused,
                            totals.imports_unavailable);
                        send_import(imported);
                    } catch (...) {
                        terminal(
                            server_vbr_artifact_import_status::internal_error);
                    }
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    // Server semantic envelope [I10]: reject a cross-adapter restore BEFORE mutating
                    // the target -- restoring state computed under one adapter into a slot bound to a
                    // different one would serve the wrong adapter's KV. Sidecar-absent (legacy/base)
                    // files are allowed for back-compat.
                    {
                        std::ifstream f(filepath + ".lora", std::ios::binary);
                        if (f) {
                            const std::string saved_identity((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                            if (saved_identity != lora_config_identity(slot->lora)) {
                                send_error(task, "Slot file was saved under a different adapter configuration than the target slot", ERROR_TYPE_INVALID_REQUEST);
                                break;
                            }
                        }
                    }

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        // a partial/failed load may have written some target cells; prompt_clear()
                        // (seq_rm) resets the target AND draft sequences, not just the bookkeeping
                        // that the struct-level prompt.clear() would [R7/N7]
                        slot->mandatory_recovery_reset(
                            server_cache_destruction_reason::restore_failure);
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    if (slot->prompt.sequence_epoch != 0 ||
                        !slot->prompt.checkpoints.empty()) {
                        SLT_INF(*slot,
                                "FRONTIER_RECORD event=invalidate "
                                "reason=slot_file_restore checkpoints=%zu "
                                "sequence_epoch=%" PRIu64 "\n",
                                slot->prompt.checkpoints.size(),
                                slot->prompt.sequence_epoch);
                    }
                    slot->observe_mandatory_recovery_reset(
                        server_cache_destruction_reason::slot_rebind);
                    slot->server_cache_mandatory_recovery_reset_impl(ctx_dft != nullptr);
                    slot->prompt.tokens.insert(tokens);
                    if (slot->retention_obs) {
                        const common_chat_msg_spans unavailable_spans;
                        (void) slot->retention_obs->publish(
                            server_retention_instance_key::for_slot(slot->id),
                            slot->retention_pool,
                            unavailable_spans,
                            false,
                            uint64_t(slot->prompt.n_tokens()),
                            uint64_t(slot->prompt.n_tokens()),
                            true);
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    // Gate on slot content, consistent with save/restore.
                    if (!check_slot_no_media(*slot, task.id)) {
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;

    // fork: spec-cycle instrumentation + cross-phase batch state.
    // Set in pre_decode(), consumed by decode()/post_decode()/post_cycle() —
    // the update_slots() split turned the old monolithic-loop locals into members.
    int64_t t_cycle_start   = 0;
    int64_t t_draft_total   = 0;
    int64_t t_verify_total  = 0;
    int64_t t_accept_total  = 0;
    int     n_slots_drafted = 0;
    int     n_slots_draft_decode_succeeded = 0;
    bool    cycle_has_output = false;
    bool    cycle_failed = false;

    // TG tokens in the current batch — pure-verify batches allow multi-seq batching
    int32_t n_tg_tokens = 0;

    // DFlash tape recording armed for this cycle (turned off in post_cycle())
    bool dflash_tape_active = false;
    // Target-side argmax for one pure-greedy DFlash verify batch.
    bool dflash_target_argmax_active = false;
    llama_seq_id dflash_target_argmax_slot = -1;
    // target can replay the tape losslessly on GPU after a partial accept; when false,
    // no tape is recorded and rollback re-decodes the accepted tokens instead
    bool dflash_tape_ok = false;
    // pure-TG batch → multi-seq ubatch allowed (force_split_seq restored in post_cycle())
    bool can_batch_multiseq = false;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // E1 holder/lease expiry is scheduler-owned even when no further E1
        // task arrives. This is the existing update-slots lifecycle point;
        // the authority itself pins and asserts the owning thread.
        if (cache_control_authority) {
            cache_control_authority->lifecycle_point();
        }

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");
                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            cycle_failed = true;
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));
        }

        // fork: the chunk loop below never runs for an empty batch, so the empty-batch
        // warn/abort lives here — with the diffusion exemption (diffusion slots contribute
        // no main-batch tokens; they decode their own batches in post_cycle())
        if (batch.size() == 0) {
            const bool has_diff_gen = std::any_of(slots.begin(), slots.end(),
                [](const server_slot & s) { return s.diff_self_spec && s.state == SLOT_STATE_GENERATING; });
            if (!has_diff_gen) {
                SRV_WRN("%s", "no tokens to decode\n");
                if (++n_empty_consecutive > 3) {
                    GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
                }
            }
        } else {
            n_empty_consecutive = 0;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                cycle_failed = true;
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                cycle_failed = true;
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }

        // fork: once-per-cycle tail — diffusion self-speculation, spec-cycle report,
        // DFlash tape-off, force_split_seq restore
        try {
            post_cycle();
        } catch (const std::exception & e) {
            cycle_failed = true;
            SRV_ERR("post_cycle() failed: %s\n", e.what());
            abort_all_slots("post_cycle() failed: " + std::string(e.what()));
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.observe_live_range_drop(
                    server_cache_destruction_reason::context_shift, true);
                ::server_cache_live_range_drop_impl(
                    ctx_tgt, slot.id, n_keep, n_keep + n_discard);
                common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                if (ctx_dft) {
                    ::server_cache_live_range_drop_impl(
                        ctx_dft.get(), slot.id, n_keep, n_keep + n_discard);
                    common_context_seq_add(ctx_dft.get(), slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.server_cache_live_range_drop_impl(
                        llama_tokens(std::move(new_tokens)));
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        // fork: reset the spec-cycle instrumentation (reported in post_cycle())
        t_cycle_start   = ggml_time_us();
        t_draft_total   = 0;
        t_verify_total  = 0;
        t_accept_total  = 0;
        n_slots_drafted = 0;
        n_slots_draft_decode_succeeded = 0;
        cycle_has_output = false;
        cycle_failed = false;

        std::vector<llama_tokens> batched_drafts(slots.size());
        std::vector<bool> batched_draft_decode_succeeded(slots.size(), false);
        const auto recurrent_speculation_deferred = [&]() {
            return needs_reeval &&
                ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                recurrent_expansion.retry_deferred;
        };
        if (ctx_dft_shared) {
            int n_drafting = 0;
            for (const auto & slot : slots) {
                if (!recurrent_speculation_deferred() &&
                    slot.state == SLOT_STATE_GENERATING && slot.can_speculate() && slot.get_n_draft_max() > 0) {
                    n_drafting++;
                }
            }
            llama_set_dflash_n_slots(ctx_dft_shared.get(), std::max(1, n_drafting));

            if (n_drafting >= 2 && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                std::vector<common_speculative *> batch_specs;
                std::vector<llama_token>          batch_id_lasts;
                std::vector<int>                  batch_slot_ids;

                for (auto & slot : slots) {
                    if (!recurrent_speculation_deferred() &&
                        slot.state == SLOT_STATE_GENERATING && slot.can_speculate() && slot.get_n_draft_max() > 0) {
                        batch_specs.push_back(slot.get_spec());
                        batch_id_lasts.push_back(slot.sampled);
                        batch_slot_ids.push_back(slot.id);
                    }
                }

                std::vector<llama_tokens> batch_results;
                const int64_t t_batch_start = ggml_time_us();
                common_speculative_draft_batch(
                        batch_specs, ctx_dft_shared.get(),
                        params_base.speculative, batch_id_lasts, batch_results);
                t_draft_total = ggml_time_us() - t_batch_start;

                for (size_t i = 0; i < batch_slot_ids.size(); i++) {
                    batched_draft_decode_succeeded[batch_slot_ids[i]] =
                            common_speculative_last_draft_model_decode_succeeded(batch_specs[i]);
                    batched_drafts[batch_slot_ids[i]] = std::move(batch_results[i]);
                }
            }
        }

        // first, add sampled tokens from any ongoing sequences (and draft per slot)
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.diff_self_spec) {
                return; // diffusion slots handled separately after main decode
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            const int n_draft_max = recurrent_speculation_deferred() ? 0 : slot.get_n_draft_max();
            if (n_draft_max > 0) {
                const int64_t t_draft_slot_start = ggml_time_us();

                llama_tokens draft;
                if (!batched_drafts[slot.id].empty()) {
                    draft = std::move(batched_drafts[slot.id]);
                } else if (!slot.spec && spec &&
                           (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
                            params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK))) {
                    // upstream shared multi-seq state (block-diffusion DFlash / DSpark):
                    // arm this slot's per-seq draft params — the fork single-seq wrapper
                    // below always drives drafter seq 0, which only matches slot 0
                    const llama_tokens & cached_text_tokens = slot.prompt.tokens.get_text_tokens();
                    auto & dp   = common_speculative_get_draft_params(spec.get(), slot.id);
                    dp.drafting = true;
                    dp.n_max    = n_draft_max;
                    dp.n_past   = slot.prompt.tokens.pos_next();
                    dp.id_last  = slot.sampled;
                    dp.prompt   = &cached_text_tokens;
                    dp.result   = &draft;
                    common_speculative_draft(spec.get());

                    // the draft decode wrote its noise block into the drafter KV at the
                    // frontier positions; trim back to the committed target frontier so
                    // the verify-batch injection in common_speculative_process stays the
                    // only source of drafter cells (upstream's server does this same
                    // post-draft seq_rm; it was dropped in the fork-owned merge)
                    if (ctx_dft) {
                        llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), slot.id,
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) + 1, -1);
                    }
                } else {
                    const llama_tokens & cached_text_tokens = slot.prompt.tokens.get_text_tokens();
                    const auto & params_spec = slot.task->params.speculative;
                    const llama_pos n_past = slot.prompt.tokens.pos_next();
                    draft = common_speculative_draft(slot.get_spec(), params_spec, cached_text_tokens, slot.sampled, nullptr, n_past);
                }
                if (batched_draft_decode_succeeded[slot.id] ||
                    common_speculative_last_draft_model_decode_succeeded(slot.get_spec())) {
                    n_slots_draft_decode_succeeded++;
                }

                if (draft.size() > (size_t) n_draft_max) {
                    SLT_WRN(slot, "draft size %d exceeds max %d, truncating\n", (int) draft.size(), n_draft_max);
                    draft.resize(n_draft_max);
                }

                if (needs_reeval &&
                    params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                    !llama_tape_replay_sync(ctx_tgt)) {
                    // A deferred exact replay could not publish its conv
                    // frontier. Abort before adding this slot to the decode
                    // batch; the rollback backup was consumed after launch.
                    SLT_ERR(slot, "%s\n", "exact tape replay synchronization failed; resetting slot");
                    send_error(slot, "Compute error completing exact tape replay");
                    slot.release();
                    slot.mandatory_recovery_reset(
                        server_cache_destruction_reason::restore_failure);
                    slot.has_draft_backup = false;
                    slot.seq_id_backup = -1;
                    return;
                }

                slot.n_tokens_before_draft = slot.prompt.n_tokens();

                slot.spec_i_batch.push_back(batch.size());
                batch.add(slot.id, slot.sampled, slot.prompt.tokens.pos_next(), true);
                slot.prompt.tokens.push_back(slot.sampled);

                // an empty draft (e.g. ngram-mod with no index match) must take the
                // no-speculation path: the else-branch would leave spec_draft empty while
                // spec_i_batch holds one index, so the accept loop (gated on
                // !spec_draft.empty()) never consumes it and never samples a token this
                // cycle — the stale index leaks into the next cycle, re-decoding
                // slot.sampled every step until a real draft breaks the
                // idxs.size() == draft.size() + 1 invariant (#74)
                if (draft.empty() || slot.task->params.speculative.n_min > (int) draft.size()) {
                    SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int) draft.size(), slot.task->params.speculative.n_min);
                    slot.i_batch = slot.spec_i_batch[0];
                    slot.spec_draft.clear();
                    slot.spec_i_batch.clear();
                } else {
                    enum class draft_batch_path {
                        speculative,
                        non_spec_fallback,
                    };
                    draft_batch_path batch_path = draft_batch_path::speculative;

                    // keep the spec checkpoint's position bookkeeping current (upstream #24536 family);
                    // the fork accept loop rolls back via backup seqs / seq_rm, so only the cheap
                    // position update is taken here (no target/draft state snapshot)
                    slot.spec_ckpt.update_pos(
                            slot.n_tokens_before_draft,
                            llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                            llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                    if (needs_reeval) {
                        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                            const int n_batch_tokens = 1 + (int) draft.size();
                            std::vector<int32_t> linear_parents(n_batch_tokens);
                            linear_parents[0] = -1;
                            for (int i = 1; i < n_batch_tokens; i++) {
                                linear_parents[i] = i - 1;
                            }
                            llama_set_tree_parent_ids(ctx_tgt, linear_parents.data(), n_batch_tokens);
                        }

                        // RS contexts handle rollback internally via seq_rm snapshots. Keep them
                        // contracted: only non-RS contexts may allocate backup sequence cells.
                        auto expansion_action = recurrent_expansion.action(
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS);
                        if (expansion_action != server_recurrent_speculation_action::rs_plane) {
                            slot.has_draft_backup = false;
                            slot.seq_id_backup = -1;

                            if (expansion_action == server_recurrent_speculation_action::try_expand) {
                                auto * mem = llama_get_memory(ctx_tgt);
                                const bool expanded = llama_memory_recurrent_expand(mem, n_seq_max_full);
                                expansion_action = recurrent_expansion.complete_expand(expanded);
                                if (expanded) {
                                    SRV_INF("expanded recurrent state to %d cells for speculative backup\n", n_seq_max_full);
                                } else {
                                    SRV_ERR("failed to expand recurrent state to %d cells; using non-speculative decode until the next request\n",
                                            n_seq_max_full);
                                }
                            }
                            if (expansion_action == server_recurrent_speculation_action::non_speculative) {
                                batch_path = draft_batch_path::non_spec_fallback;
                            }

                            if (expansion_action == server_recurrent_speculation_action::backup_ready) {
                                const llama_seq_id seq_backup = slot.id + n_parallel_user;
                                auto * mem = llama_get_memory(ctx_tgt);
                                const bool backup_cleared = server_cache_transient_seq_rm_impl(
                                    mem, seq_backup, -1, -1);
                                // Arm rollback only after both cleanup and copy succeed. Non-RS
                                // recurrent targets have no plane-based rollback substitute.
                                if (backup_cleared &&
                                    llama_memory_try_seq_cp_transient(mem, slot.id, seq_backup, -1, -1)) {
                                    slot.has_draft_backup = true;
                                    slot.seq_id_backup = seq_backup;
                                } else {
                                    recurrent_expansion.defer();
                                    batch_path = draft_batch_path::non_spec_fallback;
                                    SLT_WRN(slot, "%s\n", "speculative backup preparation failed; using non-speculative decode");
                                }
                            }
                        }
                    }

                    if (batch_path == draft_batch_path::non_spec_fallback) {
                        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                            llama_clear_tree_parent_ids(ctx_tgt);
                        }
                        slot.i_batch = slot.spec_i_batch[0];
                        slot.spec_draft.clear();
                        slot.spec_i_batch.clear();
                    } else {
                        slot.n_draft_total += draft.size();
                        for (size_t i = 0; i < draft.size(); i++) {
                            slot.spec_i_batch.push_back(batch.size());
                            batch.add(slot.id, draft[i], slot.prompt.tokens.pos_next(), true);
                            slot.prompt.tokens.push_back(draft[i]);
                        }
                        slot.spec_draft = std::move(draft);
                    }
                }
                t_draft_total += ggml_time_us() - t_draft_slot_start;
                n_slots_drafted++;
            } else {
                slot.i_batch = batch.size();

                batch.add(slot.id, slot.sampled, slot.prompt.tokens.pos_next(), true);

                slot.prompt.tokens.push_back(slot.sampled);

                SLT_DBG(slot, "slot decode token, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                        slot.n_ctx, slot.prompt.n_tokens(), slot.truncated);
            }
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        // track how many TG tokens are in the batch vs total, to detect
        // pure-verify batches where multi-seq batching is safe.
        n_tg_tokens = batch.size();

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;
                    std::unique_ptr<server_vbr_retier_freeze_scope> vbr_restore_freeze;
                    bool return_after_vbr_restore_trim = false;
                    bool checkpoint_tgt_recurrent_installed = false;
                    bool checkpoint_dft_recurrent_installed = false;
                    llama_pos checkpoint_installed_pos = -1;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int    n_past        = 0;
                        size_t n_past_common = 0;
                        size_t n_past_keep   = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // Gate-5 edit-regime observation. Capture the logical
                            // branch point before checkpoint restore or VBR policy
                            // can rewind the physical frontier. The raw token LCP
                            // and actually reusable LCP are intentionally separate:
                            // adapter/alora constraints can add replay work without
                            // changing where the user's content diverged.
                            const size_t n_cached_before = slot.prompt.tokens.size();
                            const size_t n_content_lcp =
                                slot.prompt.tokens.get_common_prefix(input_tokens);

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = (int) n_content_lcp;

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                // [P0b/A] Preserve the true token LCP before checkpoint restore
                                // rewinds n_past to a smaller physical frontier. n_past_keep is the
                                // separately bounded amount that prompt.tokens may continue to
                                // claim after restore; see the restore-site recurrent guard below.
                                // Keeping token bookkeeping never mutates KV/state by itself.
                                n_past_common = n_past;
                                n_past_keep   = n_past_common;

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            // The splice rewrites live cell positions (and later
                                            // re-RoPEs their K bytes) beneath any retained context
                                            // checkpoint. On non-hybrid (e.g. pure-SWA) models the
                                            // checkpoint selector has no epoch guard, so a stale
                                            // checkpoint over spliced cells would restore silently
                                            // wrong state. Invalidate before the first mutation.
                                            // [WS-7 review F7]
                                            slot.observe_live_range_drop(
                                                server_cache_destruction_reason::live_prefix_replace,
                                                true);
                                            if (!slot.prompt.checkpoints.empty()) {
                                                SLT_WRN(slot, "cache-reuse splice invalidates %zu context checkpoint(s)\n",
                                                        slot.prompt.checkpoints.size());
                                                (void) slot.checkpoint_drop_joined_impl(
                                                    slot.prompt.checkpoints.begin(),
                                                    slot.prompt.checkpoints.end());
                                            }

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            ::server_cache_live_range_drop_impl(
                                                ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                ::server_cache_live_range_drop_impl(
                                                    ctx_dft.get(), slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft.get(), slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            const size_t n_reusable_lcp =
                                slot.task->params.cache_prompt ? n_past_common : 0;
                            const size_t n_rewind =
                                n_cached_before > n_content_lcp
                                    ? n_cached_before - n_content_lcp
                                    : 0;
                            const size_t n_append =
                                input_tokens.size() > n_content_lcp
                                    ? input_tokens.size() - n_content_lcp
                                    : 0;
                            SLT_INF(
                                slot,
                                "edit/divergence sample "
                                "(cached/incoming/lcp/reusable/rewind/append/cache_prompt) = "
                                "(%zu/%zu/%zu/%zu/%zu/%zu/%d)\n",
                                n_cached_before, input_tokens.size(),
                                n_content_lcp, n_reusable_lcp,
                                n_rewind, n_append,
                                slot.task->params.cache_prompt ? 1 : 0);

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    // [WS-1] fail-closed coordinated restore. The token ledger claims a
                                    // prefix (n_past > 0) but this sequence's KV is empty (pos_min == -1) --
                                    // a token-ledger/KV desync. Upstream GGML_ABORTs here, taking down the
                                    // WHOLE server (every client) on a "should not happen" invariant. P0's
                                    // I7/I10 closed the known host-cache route (empty-main select + non-
                                    // consuming both-or-nothing load); this converts any remaining/latent
                                    // desync into a per-slot recovery, matching the fork's recoverable-failure
                                    // rule (llama-kv-cache.cpp VBR try_map: "fail this batch recoverably
                                    // instead of a process abort killing every client"). Coordinate the two
                                    // structures -- clear KV (tgt+dft) + the stale ledger via prompt_clear --
                                    // and full-reprocess. Loud ERR so a real desync is never silently masked.
                                    SLT_ERR(slot, "token-ledger/KV desync (n_past = %d, pos_min = -1, tokens = %d, seq_id = %d) -- failing closed: clearing slot and reprocessing from scratch\n",
                                            n_past, (int) slot.prompt.tokens.size(), slot.id);
                                    slot.mandatory_recovery_reset(
                                        server_cache_destruction_reason::restore_failure);
                                    n_past            = 0;
                                    n_past_common     = 0;
                                    n_past_keep       = 0;
                                    pos_next          = 0;
                                    slot.cache_status = "full reprocess: token-ledger/KV desync recovered [WS-1]";
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                // The planner does not yet carry SWA/recurrent
                                // frontier coverage. Revalidate the checkpoint
                                // predicate and the legacy coverage trigger at
                                // the last seam before recovery mutation.
                                if (slot.cache_plan_execution.restores_checkpoint()) {
                                    GGML_ASSERT(cache_plan_authority &&
                                                slot.cache_plan);
                                    int32_t checked_ordinal = -1;
                                    server_cache_plan_revalidate_checkpoint_execution(
                                        *cache_plan_authority, *slot.cache_plan,
                                        slot.cache_plan_execution,
                                        slot.prompt.checkpoints.size(),
                                        cache_plan_checkpoint_execution_revalidate(
                                            slot, pos_next, pos_min_thold),
                                        checked_ordinal);
                                }
                                if (slot.cache_plan_execution.authoritative()) {
                                    GGML_ASSERT(cache_plan_authority &&
                                                slot.cache_plan);
                                    server_cache_plan_demote_for_coverage_recovery(
                                        *cache_plan_authority, *slot.cache_plan,
                                        slot.cache_plan_execution,
                                        pos_min, pos_min_thold);
                                }

                                if (pos_min >= pos_min_thold ||
                                    slot.cache_plan_execution.restores_checkpoint()) {
                                    // [I9] current attention-content lineage epoch(s). A recurrent-only
                                    // checkpoint remains valid across a lossless in-place retier, but not
                                    // across occupied-cell reuse, clear/reset, or state adoption. All-zero
                                    // when VBR is inactive, so this rejects nothing then.
                                    const auto vbr_now = llama_memory_vbr_state(llama_get_memory(ctx_tgt), slot.id, 0);
                                    const bool recurrent =
                                        llama_model_is_recurrent(model_tgt) ||
                                        llama_model_is_hybrid(model_tgt);

                                    // [obs] count checkpoints rejected specifically for changed attention
                                    // lineage, to explain a resulting cold reprocess in /slots
                                    int n_ckpt_rejected_vbr = 0;

                                    // B/A2 candidate transport: which scan is authoritative
                                    // (legacy vs frontier) is decided AFTER both run, so each
                                    // lambda notes its visits into a fixed stack buffer
                                    // (classification + scalars the scan itself touched —
                                    // noexcept, no allocation); once the ratchet picks the
                                    // authoritative scan, ONLY that buffer becomes candidate
                                    // rows. The declared domain stays the shipped-visited set.
                                    struct ckpt_obs_visit {
                                        common_cache_plan_reason cls;
                                        int32_t  pos_max;
                                        uint64_t bytes;
                                    };
                                    struct ckpt_obs_buf {
                                        // deliberately NO default member initializers: the
                                        // disabled path pays a stack reservation and nothing
                                        // else (verify-r2 finding 3); fields are initialized
                                        // only under the observer flag below
                                        std::array<ckpt_obs_visit, COMMON_CACHE_PLAN_MAX_CANDIDATES> v;
                                        uint32_t n;
                                        bool     overflow;
                                        // cls NONE = eligible (the short-circuiting selection)
                                        void note(common_cache_plan_reason cls, int32_t pos_max, uint64_t bytes) noexcept {
                                            if (n >= v.size()) { overflow = true; return; }
                                            v[n++] = { cls, pos_max, bytes };
                                        }
                                    };
                                    ckpt_obs_buf obs_legacy_buf, obs_frontier_buf;
                                    const bool ckpt_obs_on = slot.cache_plan != nullptr;
                                    if (ckpt_obs_on) {
                                        obs_legacy_buf.n   = 0; obs_legacy_buf.overflow   = false;
                                        obs_frontier_buf.n = 0; obs_frontier_buf.overflow = false;
                                    }

                                    // Legacy selection remains authoritative while the
                                    // computation-frontier path runs in shadow [WS-4].
                                    // shipped predicate written ONCE; observer notes vanish
                                    // from the unobserved instantiation [A2/F7]
                                    const auto make_legacy_pred = [&](auto obs_c) {
                                        return [&, obs_c](const auto & cur) {
                                            constexpr bool Obs = decltype(obs_c)::value;
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // A capture is durable only after its target payload is
                                            // complete. Old/foreign half-filled entries must never
                                            // reach the mutating restore path.
                                            [[maybe_unused]] uint64_t obs_bytes = 0;
                                            if constexpr (Obs) { obs_bytes = (uint64_t) cur.size(); }
                                            const bool checkpoint_lineage_matches =
                                                !recurrent ||
                                                common_prompt_checkpoint_lineage_matches(cur, vbr_now);
                                            const auto evaluation =
                                                server_cache_plan_evaluate_checkpoint(
                                                    !cur.empty(), true, recurrent,
                                                    checkpoint_lineage_matches,
                                                    cur.pos_min, cur.pos_max, pos_next,
                                                    pos_min_thold, obs_bytes);
                                            const bool ok = server_cache_plan_viable(
                                                evaluation.reason);
                                            if (!cur.empty() && !checkpoint_lineage_matches) {
                                                n_ckpt_rejected_vbr++;
                                            }
                                            if constexpr (Obs) {
                                                obs_legacy_buf.note(
                                                    ok ? COMMON_CACHE_PLAN_REASON_NONE
                                                       : evaluation.reason,
                                                    cur.pos_max, obs_bytes);
                                            }
                                            return ok;
                                        };
                                    };
                                    const auto it_legacy = ckpt_obs_on
                                        ? std::find_if(slot.prompt.checkpoints.rbegin(),
                                                       slot.prompt.checkpoints.rend(),
                                                       make_legacy_pred(std::true_type{}))
                                        : std::find_if(slot.prompt.checkpoints.rbegin(),
                                                       slot.prompt.checkpoints.rend(),
                                                       make_legacy_pred(std::false_type{}));

                                    // Shadow the same choice from the dual-written logical
                                    // frontier. pos_min remains the physical attention-coverage
                                    // guard until plan_resume lands; logical eligibility comes
                                    // from next_position rather than pos_max.
                                    std::string frontier_adapter_identity;
                                    bool frontier_eligible = false;
                                    // per-selector [I9] lineage-rejection count: after a WS-4 flip
                                    // the authoritative scan is this one, and its epoch
                                    // rejections are NOT the legacy scan's
                                    int n_ckpt_rejected_vbr_frontier = 0;
                                    auto it_frontier =
                                        slot.prompt.checkpoints.rend();
                                    bool frontier_shadow_available = true;
                                    try {
                                        frontier_adapter_identity =
                                            lora_config_identity(slot.lora);
                                        const auto make_frontier_pred = [&](auto obs_c) {
                                            return [&, obs_c](const auto & cur) {
                                                constexpr bool Obs = decltype(obs_c)::value;
                                                [[maybe_unused]] uint64_t obs_bytes = 0;
                                                if constexpr (Obs) { obs_bytes = (uint64_t) cur.size(); }
                                                if (slot.frontier_ratchet_flipped &&
                                                    server_fault(
                                                        "frontier_disagree_after_flip")) {
                                                    if constexpr (Obs) { obs_frontier_buf.note(COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID, cur.pos_max, obs_bytes); }
                                                    return false;
                                                }
                                                if (cur.empty() ||
                                                    !checkpoint_frontier_is_current(
                                                        slot, cur,
                                                        frontier_adapter_identity)) {
                                                    if constexpr (Obs) {
                                                        obs_frontier_buf.note(cur.empty()
                                                            ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY
                                                            : COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID, cur.pos_max, obs_bytes);
                                                    }
                                                    return false;
                                                }
                                                frontier_eligible = true;
                                                if (recurrent) {
                                                    if (!common_prompt_checkpoint_lineage_matches(
                                                            cur, vbr_now)) {
                                                        n_ckpt_rejected_vbr_frontier++;
                                                        if constexpr (Obs) { obs_frontier_buf.note(COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED, cur.pos_max, obs_bytes); }
                                                        return false;
                                                    }
                                                    const bool ok = cur.computation_frontier.next_position <=
                                                           pos_next;
                                                    if constexpr (Obs) { obs_frontier_buf.note(ok ? COMMON_CACHE_PLAN_REASON_NONE : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT, cur.pos_max, obs_bytes); }
                                                    return ok;
                                                }
                                                if (cur.computation_frontier.next_position - 1 >
                                                    pos_next) {
                                                    if constexpr (Obs) { obs_frontier_buf.note(COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT, cur.pos_max, obs_bytes); }
                                                    return false;
                                                }
                                                const bool ok = cur.pos_min < pos_min_thold ||
                                                       cur.pos_min == 0;
                                                if constexpr (Obs) { obs_frontier_buf.note(ok ? COMMON_CACHE_PLAN_REASON_NONE : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT, cur.pos_max, obs_bytes); }
                                                return ok;
                                            };
                                        };
                                        it_frontier = ckpt_obs_on
                                            ? std::find_if(slot.prompt.checkpoints.rbegin(),
                                                           slot.prompt.checkpoints.rend(),
                                                           make_frontier_pred(std::true_type{}))
                                            : std::find_if(slot.prompt.checkpoints.rbegin(),
                                                           slot.prompt.checkpoints.rend(),
                                                           make_frontier_pred(std::false_type{}));
                                    } catch (const std::exception & e) {
                                        // Shadow bookkeeping must never make the
                                        // authoritative legacy selector less available.
                                        frontier_shadow_available = false;
                                        SLT_WRN(slot,
                                                "FRONTIER_RATCHET "
                                                "event=shadow_unavailable "
                                                "action=keep_legacy error=%s\n",
                                                e.what());
                                    }

                                    const auto choice_index = [&](const auto & choice) {
                                        return choice ==
                                                slot.prompt.checkpoints.rend()
                                            ? int64_t(-1)
                                            : (int64_t) std::distance(
                                                slot.prompt.checkpoints.rbegin(),
                                                choice);
                                    };
                                    const int64_t legacy_choice =
                                        choice_index(it_legacy);
                                    const int64_t frontier_choice =
                                        choice_index(it_frontier);
                                    bool use_frontier = false;
                                    if (frontier_shadow_available) {
                                        use_frontier =
                                            slot.frontier_ratchet_selection(
                                                frontier_eligible,
                                                it_legacy == it_frontier,
                                                legacy_choice,
                                                frontier_choice);
                                    } else {
                                        slot.frontier_ratchet_disagreement(
                                            "checkpoint_selection_shadow_unavailable",
                                            legacy_choice, -2);
                                    }
                                    auto it = use_frontier ? it_frontier : it_legacy;

                                    // B-A1 executes the planner-selected checkpoint,
                                    // not the first match from the legacy/frontier scan.
                                    // Keep this upstream collision-prone container
                                    // translation behind one named helper.
                                    it = cache_plan_override_checkpoint(slot, it);

                                    bool do_reset = it == slot.prompt.checkpoints.rend();
                                    bool vbr_preflight_rejected = false;

                                    // B candidate transport [A2, noexcept]: the AUTHORITATIVE
                                    // scan's visit buffer becomes checkpoint candidate rows.
                                    // Reverse visit positions are translated to the stable
                                    // forward container ordinal before row identity joins.
                                    // The short-circuit
                                    // leaves later siblings outside the declared domain —
                                    // recorded as truncation, never extrapolated (r3 A1).
                                    if (slot.cache_plan) {
                                        auto & rec = *slot.cache_plan;
                                        const auto & buf = use_frontier ? obs_frontier_buf : obs_legacy_buf;
                                        int32_t checkpoint_host_source = -1;
                                        if (const auto * host = rec.selected_row(
                                                common_cache_plan_provider::host_cache_entry);
                                            host && host->delivered && host->source_id >= 0) {
                                            checkpoint_host_source = host->source_id;
                                        }
                                        for (uint32_t i = 0; i < buf.n; i++) {
                                            const int32_t checkpoint_source =
                                                server_cache_plan_checkpoint_source_id_from_reverse(
                                                    slot.prompt.checkpoints.size(), i,
                                                    checkpoint_host_source);
                                            if (checkpoint_source < 0) {
                                                rec.inventory_states[size_t(
                                                    common_cache_plan_provider::live_context_checkpoint)] =
                                                    common_cache_plan_inventory_state::overflowed;
                                                break;
                                            }
                                            auto * row = rec.find_or_add(
                                                common_cache_plan_provider::live_context_checkpoint,
                                                checkpoint_source,
                                                COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                                                slot.id, rec.selection);
                                            if (!row) {
                                                break; // record inventory overflowed (state latched)
                                            }
                                            row->lcp_tokens    = llama_cache_acct_value::measured(
                                                (uint64_t) std::max<int32_t>(buf.v[i].pos_max, 0));
                                            row->payload_bytes = llama_cache_acct_value::measured(buf.v[i].bytes);
                                            row->component_only = checkpoint_host_source >= 0;
                                            row->dependent_host_source_id = checkpoint_host_source;
                                            if (buf.v[i].cls == COMMON_CACHE_PLAN_REASON_NONE) {
                                                // the scan's first eligible entry = the shipped selection;
                                                // restore failures below demote it via note_reject
                                                row->accept();
                                                if (!slot.cache_plan_execution.restores_checkpoint() ||
                                                    row->source_id == slot.cache_plan_execution.
                                                        checkpoint_source_id) {
                                                    rec.select(
                                                        common_cache_plan_provider::
                                                            live_context_checkpoint,
                                                        row);
                                                }
                                            } else {
                                                row->note_reject(buf.v[i].cls);
                                            }
                                        }
                                        if (buf.overflow) {
                                            rec.inventory_states[size_t(common_cache_plan_provider::live_context_checkpoint)] =
                                                common_cache_plan_inventory_state::overflowed;
                                        } else if (!do_reset && buf.n < slot.prompt.checkpoints.size()) {
                                            if (!rec.authority_inventory_complete) {
                                                rec.note_inventory_truncated(common_cache_plan_provider::live_context_checkpoint);
                                            }
                                        } else {
                                            rec.note_inventory_complete(common_cache_plan_provider::live_context_checkpoint);
                                        }
                                    }

                                    if (!do_reset) {
                                        // [WS-6] A live_rebased PARTIAL_ONLY restore keeps the
                                        // current attention representation. Prove the current-tier
                                        // footprint fits before either the restore or its later
                                        // suffix trim mutates state, then hold a scoped retier
                                        // freeze across both. This consumer performs no bounded
                                        // replay inside the scope, so n_tokens_extra is exactly 0.
                                        const auto preflight =
                                            llama_memory_vbr_retier_preflight(
                                                llama_get_memory(ctx_tgt), 0);
                                        if (preflight.active) {
                                            const bool preflight_fault =
                                                server_fault("vbr_retier_preflight_fail");
                                            const bool preflight_fits =
                                                preflight.fits && !preflight_fault;
                                            SLT_INF(slot,
                                                    "VBR_RETIER_PREFLIGHT owner=server_checkpoint_restore "
                                                    "result=%s fault=%u pools=%u watermark=%u needed=%" PRIu64
                                                    " available=%" PRIu64 " physical_growth=%" PRIu64
                                                    " physical_free=%" PRIu64 " max_deficit=%" PRId64 "\n",
                                                    preflight_fits ? "fits" : "reject",
                                                    preflight_fault ? 1u : 0u,
                                                    preflight.pools,
                                                    preflight.watermark_cells,
                                                    preflight.bytes_needed,
                                                    preflight.bytes_available,
                                                    preflight.physical_growth_needed,
                                                    preflight.physical_growth_available,
                                                    preflight.max_deficit);
                                            if (!preflight_fits) {
                                                SLT_WRN(slot, "%s",
                                                        "checkpoint restore rejected before mutation: "
                                                        "current VBR tiers cannot cover the frozen footprint\n");
                                                slot.cache_status =
                                                    "full reprocess: VBR frozen-footprint preflight rejected";
                                                vbr_preflight_rejected = true;
                                                do_reset = true;
                                            } else {
                                                vbr_restore_freeze =
                                                    std::make_unique<server_vbr_retier_freeze_scope>(
                                                        llama_get_memory(ctx_tgt),
                                                        "server_checkpoint_restore");
                                                if (!vbr_restore_freeze->active()) {
                                                    SLT_ERR(slot, "%s",
                                                            "VBR preflight was active but scoped freeze acquisition failed\n");
                                                    vbr_restore_freeze.reset();
                                                    do_reset = true;
                                                    // B0: an attempted restore failed on
                                                    // infrastructure, not eligibility
                                                    if (slot.cache_plan) {
                                                        if (auto * sel_row = slot.cache_plan->selected_row(
                                                                common_cache_plan_provider::live_context_checkpoint)) {
                                                            sel_row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY);
                                                        }
                                                        slot.cache_plan->restore_attempt_failed = true;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        // note: keep the raw size-checked restore (instead of it->load_tgt which aborts on
                                        //       mismatch) so a failed restore falls back to a full re-process (do_reset)
                                        const size_t checkpoint_size = it->data_tgt.size();
                                        const size_t n = do_reset ? 0 :
                                            llama_state_seq_set_data_ext(
                                                ctx_tgt, it->data_tgt.data(),
                                                checkpoint_size, slot.id,
                                                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (!do_reset && n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                            if (slot.cache_plan) {
                                                if (auto * sel_row = slot.cache_plan->selected_row(
                                                        common_cache_plan_provider::live_context_checkpoint)) {
                                                    sel_row->note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
                                                }
                                                slot.cache_plan->restore_attempt_failed = true;
                                            }
                                        } else if (!do_reset) {
                                            // restore the draft-side state, if any (draft-model checkpoints)
                                            it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                            // restore the drafter's speculative state (per-slot spec or shared)
                                            common_speculative_set_state(slot.get_spec(), slot.id, it->accel.spec);

                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);

                                            // [P0b/A] Attention/SWA follows upstream #24891: the
                                            // checkpoint's logical token count bounds the true LCP
                                            // retained in bookkeeping. Recurrent/hybrid PARTIAL_ONLY
                                            // restore is different: it installs recurrent state only
                                            // at the checkpoint frontier, so even a matching later
                                            // token prefix is not backed by that state. Cap to n_past,
                                            // which was derived from BOTH restored pos_next and
                                            // checkpoint.n_tokens. Thus keep_first below can never
                                            // claim a token beyond the restored recurrent frontier;
                                            // the conservative replay changes work only, never output.
                                            if (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt)) {
                                                n_past_keep = std::min(n_past_keep, (size_t) n_past);
                                            } else {
                                                n_past_keep = std::min(n_past_keep, (size_t) it->n_tokens);
                                            }

                                            bool checkpoint_aux_ok = true;
                                            if (slot.can_speculate() && !it->accel.ring.empty() &&
                                                !common_speculative_ring_state_load(
                                                    slot.get_spec(), it->accel.ring.data(), it->accel.ring.size())) {
                                                SLT_ERR(slot,
                                                        "failed to restore checkpoint DFlash ring state "
                                                        "(n_tokens = %" PRId64 ", ring size = %zu) -- failing closed\n",
                                                        it->n_tokens, it->accel.ring.size());
                                                checkpoint_aux_ok = false;
                                            }

                                            if (!checkpoint_aux_ok) {
                                                do_reset = true;
                                                if (slot.cache_plan) {
                                                    if (auto * sel_row = slot.cache_plan->selected_row(
                                                            common_cache_plan_provider::live_context_checkpoint)) {
                                                        sel_row->note_reject(COMMON_CACHE_PLAN_REASON_ACCELERATOR_UNRESTORABLE);
                                                    }
                                                    slot.cache_plan->restore_attempt_failed = true;
                                                }
                                            } else {
                                                checkpoint_tgt_recurrent_installed =
                                                    llama_model_is_hybrid(model_tgt);
                                                checkpoint_dft_recurrent_installed =
                                                    ctx_dft && model_dft &&
                                                    llama_model_is_hybrid(model_dft.get()) &&
                                                    !it->data_dft.empty();
                                                checkpoint_installed_pos = it->pos_max;

                                                // [WS-1 / #25592] The checkpoint that successfully
                                                // restored this task is now part of this task's active
                                                // lineage. Adopt it here, unconditionally, rather than
                                                // relying on a later create-time dedup: ordinary
                                                // mid-prompt batches suppress checkpoint creation, and
                                                // the next min-step thinning pass could otherwise erase
                                                // the exact anchor we just restored.
                                                it->id_task = slot.task->id;

                                                SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) checkpoint_size / 1024 / 1024);
                                                slot.cache_status = "restored context checkpoint";
                                                // B0 stage-3 [B-a]: transport the shipped
                                                // restore's own values into the observer row.
                                                // This site IS the delivery point — recorded
                                                // as data, never inferred.
                                                if (slot.cache_plan) {
                                                    if (auto * sel_row = slot.cache_plan->selected_row(
                                                            common_cache_plan_provider::live_context_checkpoint)) {
                                                        sel_row->accept();
                                                        sel_row->delivered = true;
                                                        // logical restored token count, not the physical
                                                        // pos_max position; bytes = every applied component
                                                        // (target+draft+accelerators) [verify-r1 finding 3]
                                                        sel_row->lcp_tokens =
                                                            llama_cache_acct_value::measured((uint64_t) std::max(n_past, 0));
                                                        sel_row->payload_bytes =
                                                            llama_cache_acct_value::measured((uint64_t) it->size());
                                                        // rows the AUTHORITATIVE shipped scan
                                                        // actually visited (frontier after a
                                                        // WS-4 flip, else legacy) + that same
                                                        // selector's epoch rejections
                                                        sel_row->siblings_scanned =
                                                            (use_frontier
                                                                ? obs_frontier_buf.n
                                                                : obs_legacy_buf.n);
                                                        sel_row->siblings_rejected_epoch = (uint32_t)
                                                            (use_frontier ? n_ckpt_rejected_vbr_frontier
                                                                          : n_ckpt_rejected_vbr);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        // [obs] explain the cold reprocess: a checkpoint rejected for a
                                        // changed VBR representation [I9] is the common surprising cause
                                        slot.cache_status = vbr_preflight_rejected
                                            ? "full reprocess: VBR frozen-footprint preflight rejected"
                                            : n_ckpt_rejected_vbr > 0
                                                ? "full reprocess: checkpoint(s) rejected -- attention lineage changed [I9]"
                                                : "full reprocess: no reusable context checkpoint";
                                        // B0 stage-3: the same shipped classification, as a
                                        // closed reason on the checkpoint candidate row. A
                                        // concrete restore-failure reason recorded at its
                                        // failure site takes precedence — the generic
                                        // eligibility classification applies only to a row
                                        // with no reason yet.
                                        if (slot.cache_plan) {
                                            const int n_rejected_sel = use_frontier
                                                ? n_ckpt_rejected_vbr_frontier
                                                : n_ckpt_rejected_vbr;
                                            // the selected row carries a concrete restore-failure
                                            // reason from its failure site; when the scan selected
                                            // NOTHING, the provider-level classification rides one
                                            // aggregate row (source -1). Per-sibling verdicts are
                                            // already in the inventory from the scan transport.
                                            auto * row = slot.cache_plan->selected_row(
                                                common_cache_plan_provider::live_context_checkpoint);
                                            if (!row) {
                                                row = slot.cache_plan->find_or_add(
                                                    common_cache_plan_provider::live_context_checkpoint,
                                                    COMMON_CACHE_PLAN_SOURCE_AGGREGATE,
                                                    COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,
                                                    slot.id, slot.cache_plan->selection);
                                            }
                                            if (row) {
                                                row->siblings_scanned =
                                                    (use_frontier
                                                        ? obs_frontier_buf.n
                                                        : obs_legacy_buf.n);
                                                row->siblings_rejected_epoch = (uint32_t) n_rejected_sel;
                                                if (row->reason == COMMON_CACHE_PLAN_REASON_NONE) {
                                                    if (vbr_preflight_rejected) {
                                                        // restoring would demand a retier under the frozen
                                                        // footprint: a representation-tier fit failure
                                                        row->note_reject(COMMON_CACHE_PLAN_REASON_REPRESENTATION_TIER_UNSUPPORTED);
                                                    } else if (n_rejected_sel > 0) {
                                                        row->note_reject(COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED);
                                                    } else if (slot.prompt.checkpoints.empty()) {
                                                        row->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE);
                                                    } else {
                                                        row->note_reject(COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT);
                                                    }
                                                }
                                            }
                                            // F1: this do_reset discards EVERYTHING already
                                            // installed for the request (pos_next/n_past go
                                            // to zero) — revoke historical deliveries; cold
                                            // fallback dominates them
                                            slot.cache_plan->revoke_deliveries();
                                        }
                                        pos_next = 0;
                                        n_past = 0;
                                        n_past_keep = 0;
                                    }
                                }
                            }

                            {
                                // [P0b/B] pos_next may have been rewound by restore and is not an
                                // invalidation frontier. Use the real physical end of the incoming
                                // prompt, as upstream #24891 does, plus the logical true LCP as a
                                // lineage guard. A checkpoint after the true LCP may describe the old
                                // branch even when its position fits inside the new prompt; recurrent
                                // state makes that especially dangerous because it summarizes every
                                // preceding token. Checkpoint.n_tokens is the architecture-neutral
                                // capture frontier (unlike recurrent pos_min), so checkpoints through
                                // the true LCP survive even when restore rewound n_past below them.
                                // This changes only checkpoint metadata retention; N1-A selection
                                // (strict pos_max < pos_next) and I9 epoch rejection above are
                                // untouched. Every retained checkpoint is on the verified common
                                // lineage and within the new physical prompt; erasure can only cause
                                // replay, never install a different state.
                                const llama_pos prompt_pos_end = input_tokens.pos_next();
                                const llama_pos common_pos_end = slot.prompt.tokens.pos_next(n_past_common);

                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    const bool beyond_prompt = cur.pos_max > prompt_pos_end;
                                    const bool beyond_lcp    = cur.n_tokens > (int64_t) n_past_common;
                                    const bool invalidated   = beyond_prompt || beyond_lcp;

                                    if (invalidated) {
                                        SLT_TRC(slot,
                                                "erased invalidated context checkpoint "
                                                "(pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                                                ", n_swa = %d, prompt_pos_end = %d, common_pos_end = %d, size = %.3f MiB)\n",
                                                cur.pos_min, cur.pos_max, cur.n_tokens, n_swa,
                                                prompt_pos_end, common_pos_end, (float) cur.size() / 1024 / 1024);
                                        it = slot.checkpoint_drop(
                                            it, std::next(it),
                                            server_cache_destruction_reason::checkpoint_invalidated);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // dynamic VBR: trade a token-trivial reusable prefix for the lossless
                        // full reset (n_past -> 0 makes the seq_rm below a full clear; the
                        // cache empties and the next prepare restores every entry tier)
                        server_cache_destruction_admission
                            vbr_low_lcp_admission;
                        {
                            const int n_past_before_vbr_reset = n_past;
                            n_past = vbr_reset_on_low_lcp(
                                slot, n_past, vbr_low_lcp_admission);
                            if (n_past < n_past_before_vbr_reset) {
                                // The reset decision depends on the ownership
                                // re-sample after idle reclaim, so it cannot be
                                // predicted safely before the checkpoint seam.
                                // Once applied, however, it supersedes the
                                // authorized plan just like coverage recovery:
                                // record typed capability drift, not an
                                // apparent post-authorization internal fault.
                                if (slot.cache_plan_execution.authoritative()) {
                                    GGML_ASSERT(cache_plan_authority &&
                                                slot.cache_plan);
                                    server_cache_plan_demote_for_vbr_low_lcp_reset(
                                        *cache_plan_authority, *slot.cache_plan,
                                        slot.cache_plan_execution, true);
                                }
                                // A VBR reset physically cleared state; do not let true-LCP token
                                // retention claim the cleared prefix.
                                n_past_keep = std::min(n_past_keep, (size_t) n_past);
                                if (n_past == 0) {
                                    // B0/F1: the full reset physically discarded whatever any
                                    // provider installed — a policy reset, not a failed
                                    // attempt, but delivery is revoked either way
                                    if (slot.cache_plan) {
                                        slot.cache_plan->revoke_deliveries();
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            // The last token must really be re-evaluated for logits; preserving it
                            // only in token bookkeeping would skip that decode.
                            n_past_keep = std::min(n_past_keep, (size_t) n_past);
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                            if (server_cache_plan_live_replay_lost_to_logits(
                                    slot.cache_plan_execution, n_past)) {
                                // A one-token exact hit becomes cold so logits
                                // can be produced. The live plan was superseded,
                                // not internally failed.
                                cache_plan_fallback_legacy(
                                    slot,
                                    common_cache_plan_authority_fallback::
                                        stale_capability);
                            }
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        const size_t n_tokens_keep =
                            std::max(n_past_keep, (size_t) n_past);
                        if (vbr_low_lcp_admission.covers(
                                server_cache_destruction_class::live_range_drop,
                                server_cache_destruction_reason::low_lcp_reset) &&
                            vbr_low_lcp_admission.execution ==
                                server_cache_destruction_execution::pass_through) {
                            // The joined low-LCP manifest was admitted before its checkpoint
                            // invalidation; carry that same authority to this later phase.
                            slot.server_cache_token_ledger_truncate_impl(n_tokens_keep);
                        } else {
                            slot.observe_live_range_drop(
                                server_cache_destruction_reason::live_prefix_replace);
                            slot.server_cache_token_ledger_truncate_impl(n_tokens_keep);
                        }

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            // A checkpoint restore's scoped VBR freeze must cover the paired
                            // attention trim below. Delay only this lambda return until the trim;
                            // unscoped/non-VBR behavior remains byte-for-byte on the old branch.
                            if (!vbr_restore_freeze) {
                                return;
                            }
                            return_after_vbr_restore_trim = true;
                        }
                    }

                    const int64_t t_now = ggml_time_us();
                    slot.t_prompt_processing = (t_now - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();
                    // (B0 finalize deliberately does NOT happen here: replay tokens and the
                    // fallible post-restore trim are still ahead — the record finalizes at
                    // the metrics.on_prompt_eval timing points)

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    // Checkpoint restores reset recurrent rollback depth. In particular,
                    // [TAG_PROMPT_LOGITS] can move p0 one token behind the restored
                    // frontier, making this bounded trim legitimately fallible.
                    // A successful PARTIAL_ONLY hybrid restore has already installed the
                    // recurrent row at checkpoint_installed_pos. When the suffix begins at
                    // exactly the next position, trimming that row only to reinstall the same
                    // checkpoint bytes is redundant. Keep the old whole-memory path for every
                    // other case, especially [TAG_PROMPT_LOGITS]'s one-behind trim: recurrent
                    // rollback depth is then a real validity check and must still fail closed.
                    const bool trim_tgt_attn_only =
                        checkpoint_tgt_recurrent_installed &&
                        (int64_t) p0 == (int64_t) checkpoint_installed_pos + 1;
                    const bool trim_dft_attn_only =
                        checkpoint_dft_recurrent_installed &&
                        (int64_t) p0 == (int64_t) checkpoint_installed_pos + 1;
                    if (trim_tgt_attn_only || trim_dft_attn_only) {
                        SLT_INF(slot,
                                "CHECKPOINT_ATTN_ONLY_TRIM p0=%d target=%u draft=%u\n",
                                p0,
                                trim_tgt_attn_only ? 1u : 0u,
                                trim_dft_attn_only ? 1u : 0u);
                    }
                    const auto checkpoint_lineage_before_trim =
                        trim_tgt_attn_only
                            ? llama_memory_vbr_state(
                                llama_get_memory(ctx_tgt), slot.id, 0)
                            : llama_memory_vbr_state_data{};
                    bool trim_ok = ::server_cache_live_range_drop_impl(
                        llama_get_memory(ctx_tgt), slot.id, p0, -1,
                        trim_tgt_attn_only);
                    if (trim_ok && ctx_dft) {
                        trim_ok = ::server_cache_live_range_drop_impl(
                            llama_get_memory(ctx_dft.get()), slot.id, p0, -1,
                            trim_dft_attn_only);
                    }

                    if (trim_ok && trim_tgt_attn_only) {
                        const auto checkpoint_lineage_after_trim =
                            llama_memory_vbr_state(
                                llama_get_memory(ctx_tgt), slot.id, 0);
                        if (server_cache_checkpoint_rebase_preserved_suffix(
                                slot.prompt.checkpoints,
                                checkpoint_lineage_before_trim,
                                checkpoint_lineage_after_trim,
                                p0) > 0) {
                            slot.checkpoint_ring_changed();
                        }
                    }

                    if (!trim_ok) {
                        SLT_WRN(slot, "memory_seq_rm [%d, end) rejected; falling back to full prompt re-processing\n", p0);

                        // Full removal is infallible for memory implementations and retains
                        // common_context_seq_rm's asserting contract. Clear both target and
                        // draft so a failure on either side cannot leave them out of sync.
                        slot.observe_mandatory_recovery_reset(
                            server_cache_destruction_reason::trim_rejection);
                        ::server_cache_mandatory_recovery_reset_impl(
                            ctx_tgt, slot.id, -1, -1);
                        if (ctx_dft) {
                            ::server_cache_mandatory_recovery_reset_impl(
                                ctx_dft.get(), slot.id, -1, -1);
                        }

                        slot.server_cache_token_ledger_truncate_impl(0);
                        slot.n_prompt_tokens_cache = 0;
                        slot.n_prompt_tokens_processed = 0;

                        // B0/F1: the suffix trim of the restore transaction was rejected and
                        // everything installed was cleared — the delivered rows failed to
                        // realize their frontier. Record the precise reason on each, revoke
                        // delivery, and mark the attempt failed: cold fallback dominates.
                        if (slot.cache_plan) {
                            for (const auto prov : { common_cache_plan_provider::host_cache_entry,
                                                     common_cache_plan_provider::live_context_checkpoint }) {
                                if (auto * c = slot.cache_plan->selected_row(prov)) {
                                    if (c->delivered) {
                                        c->note_reject(COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID);
                                    }
                                }
                            }
                            slot.cache_plan->revoke_deliveries();
                            slot.cache_plan->restore_attempt_failed = true;
                        }
                    }

                    // Retiering resumes only after the coordinated restore+trim transaction.
                    // The outer exit arms a fresh controller pass at the next safe boundary.
                    vbr_restore_freeze.reset();
                    if (return_after_vbr_restore_trim) {
                        return;
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0 ||
                            llama_model_is_hybrid(model_tgt));

                    bool has_mtmd = false;

                    // swap MTP out of VRAM so mmproj can use GPU for image encoding
                    const bool needs_mmproj_swap = mmproj_gpu_swap && !mmproj_is_on_gpu
                        && slot.prompt.n_tokens() < slot.task->n_tokens()
                        && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL;

                    if (needs_mmproj_swap) {
                        swap_mtp_to_mmproj_gpu();
                    }

                    // check if we should process the image
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = slot.process_mtmd_chunk(cur_token_idx, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    if (needs_mmproj_swap && mmproj_is_on_gpu) {
                        swap_mmproj_to_mtp();
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() &&
                           batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // no need for empty or small checkpoints
                    // for hybrid/recurrent models, lower the checkpoint threshold so short prompts also get checkpointed
                    const int checkpoint_min_tokens = (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt)) ? 4 : 64;
                    do_checkpoint = do_checkpoint && (pos_min >= 0 && slot.prompt.n_tokens() >= checkpoint_min_tokens);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);

                    const bool checkpoint_exact_frontier =
                        llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);

                    // [WS-1 / #25592 frontier validity] A hybrid/recurrent checkpoint captures a
                    // point-in-time state valid ONLY at its exact frontier -- never a [pos_min, pos_max]
                    // range. If seq_pos_min and seq_pos_max disagree (a hybrid whose attention and
                    // recurrent frontiers diverged), that range is a LIE for this state: a later restore
                    // reads pos_min (see :4578) and would install the frontier state at the wrong
                    // position -> wrong continuation. Fail CLOSED here -- skip the capture and keep
                    // existing good checkpoints (do NOT evict for one we won't create). In a coherent
                    // recurrent/hybrid cache equality holds (one recurrent frontier row) so this should
                    // never fire; if it does, it is a real memory-frontier bug worth the loud warning.
                    if (do_checkpoint && checkpoint_exact_frontier && pos_min != pos_max) {
                        SLT_WRN(slot, "[frontier-validity] skipping context checkpoint: recurrent/hybrid seq_pos_min (%d) != seq_pos_max (%d) -- range invalid for a point-in-time state\n",
                                pos_min, pos_max);
                        do_checkpoint = false;
                    }
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    const int ckpt_id_task = slot.task->id;
                    const int64_t ckpt_n_tokens = slot.prompt.n_tokens() - n_tokens_cur;
                    const llama_pos ckpt_pos_min = checkpoint_exact_frontier ? pos_max : pos_min;
                    llama_memory_vbr_state_data vbr_now = {};
                    if (do_checkpoint) {
                        vbr_now = llama_memory_vbr_state(llama_get_memory(ctx_tgt), slot.id, 0);
                    }

                    common_computation_frontier ckpt_frontier;
                    if (do_checkpoint) {
                        try {
                            ckpt_frontier.version =
                                common_computation_frontier::VERSION;
                            ckpt_frontier.sequence_epoch =
                                ensure_frontier_sequence_epoch(slot.prompt);
                            ckpt_frontier.token_count = ckpt_n_tokens;
                            ckpt_frontier.next_position =
                                slot.prompt.tokens.pos_next(ckpt_n_tokens);
                            ckpt_frontier.execution_identity =
                                frontier_execution_identity;
                            ckpt_frontier.adapter_config_identity =
                                lora_config_identity(slot.lora);

                            if (!slot.prompt.tokens.media_content_identity(
                                    ckpt_n_tokens,
                                    ckpt_frontier.media_content_identity)) {
                                SLT_WRN(slot,
                                        "FRONTIER_RECORD event=capture_reject "
                                        "reason=unverifiable_media "
                                        "n_tokens=%" PRId64 "\n",
                                        ckpt_n_tokens);
                                do_checkpoint = false;
                            } else if (pos_max < 0 ||
                                       ckpt_frontier.next_position <= 0 ||
                                       ckpt_frontier.next_position - 1 != pos_max) {
                                // Dual-write only records a frontier when the live
                                // logical ledger and legacy physical end agree.
                                SLT_WRN(slot,
                                        "FRONTIER_RECORD event=capture_reject "
                                        "reason=legacy_frontier_disagreement "
                                        "n_tokens=%" PRId64
                                        " next_position=%d pos_max=%d\n",
                                        ckpt_n_tokens,
                                        ckpt_frontier.next_position,
                                        pos_max);
                                do_checkpoint = false;
                            }
                        } catch (const std::exception & e) {
                            SLT_WRN(slot,
                                    "FRONTIER_RECORD event=capture_reject "
                                    "reason=identity_failure error=%s\n",
                                    e.what());
                            do_checkpoint = false;
                        }
                    }

                    // [WS-1 / #25592] dedup: an equivalent checkpoint already exists (e.g. one just
                    // restored, or an unchanged frontier). Length + pos_max alone are NOT an
                    // identity: SWA may have a different pos_min, and VBR can change the paired
                    // attention representation without moving the logical frontier. Only adopt a
                    // complete snapshot whose full validity and representation key match.
                    if (do_checkpoint && !slot.prompt.checkpoints.empty()) {
                        auto & last = slot.prompt.checkpoints.back();
                        if (!last.empty() &&
                            last.n_tokens                == ckpt_n_tokens &&
                            last.pos_min                 == ckpt_pos_min &&
                            last.pos_max                 == pos_max &&
                            last.checkpoint_epoch     == vbr_now.checkpoint_epoch &&
                            last.checkpoint_epoch_swa == vbr_now.checkpoint_epoch_swa &&
                            computation_frontiers_equal(
                                last.computation_frontier,
                                ckpt_frontier)) {
                            last.id_task = ckpt_id_task;
                            SLT_DBG(slot, "context checkpoint dedup: last already covers n_tokens = %" PRId64 ", pos_min = %d, pos_max = %d, checkpoint epochs = (%" PRIu64 ", %" PRIu64 ") -- adopting\n",
                                    ckpt_n_tokens, ckpt_pos_min, pos_max,
                                    vbr_now.checkpoint_epoch, vbr_now.checkpoint_epoch_swa);
                            do_checkpoint = false;

                        }
                    }

                    std::list<common_prompt_checkpoint> staged;
                    if (do_checkpoint) {
                        // Stage the complete checkpoint in a detached list node before evicting
                        // anything. Allocation failure or a short state write must retain every good
                        // published checkpoint and must never leave a half-filled entry selectable.
                        try {
                            staged.emplace_back(); // allocate the eventual list node before mutation
                            auto & next = staged.back();

                            next.id_task  = ckpt_id_task;
                            next.pos_min  = ckpt_pos_min;
                            next.pos_max  = pos_max;
                            next.n_tokens = ckpt_n_tokens;
                            next.checkpoint_epoch     = vbr_now.checkpoint_epoch;
                            next.checkpoint_epoch_swa = vbr_now.checkpoint_epoch_swa;
                            next.computation_frontier = ckpt_frontier;
                            next.cache_family = slot.cache_family;

                            const size_t checkpoint_size =
                                llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                            if (checkpoint_size == 0) {
                                SLT_ERR(slot, "%s", "skipping context checkpoint: target state size is zero\n");
                                staged.clear();
                            } else {
                                next.data_tgt.resize(checkpoint_size);
                                const size_t n = llama_state_seq_get_data_ext(
                                    ctx_tgt, next.data_tgt.data(), checkpoint_size, slot.id,
                                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                if (n != checkpoint_size) {
                                    SLT_ERR(slot,
                                            "skipping context checkpoint: target state short write (expected %zu, got %zu)\n",
                                            checkpoint_size, n);
                                    staged.clear();
                                }
                            }

                            // Allocate the optional DFlash ring while still detached, then fill it.
                            // Its save routine is exact-size and non-consuming.
                            if (!staged.empty() && slot.can_speculate()) {
                                const size_t ring_size = common_speculative_ring_state_size(slot.get_spec());
                                if (ring_size > 0) {
                                    next.accel.ring.resize(ring_size);
                                    common_speculative_ring_state_save(
                                        slot.get_spec(), next.accel.ring.data(), ring_size);
                                }
                            }
                        } catch (const std::exception & e) {
                            SLT_ERR(slot, "skipping context checkpoint: staging failed: %s\n", e.what());
                            staged.clear();
                        }

                        if (staged.empty()) {
                            do_checkpoint = false;
                        }
                    }

                    if (do_checkpoint) {
                        // [WS-1 / #25592] eviction: first THIN checkpoints that sit within
                        // checkpoint_min_step of the previous KEPT one (redundantly close), never
                        // evicting the current task's own -- this keeps early anchors alive across
                        // edited/compacted agentic histories instead of FIFO-dropping the oldest.
                        // Best-effort noise reduction only: D-A4 correctness
                        // comes from the bounded same-lineage replay proof in
                        // checkpoint_thin_priced, not from reproducing B's
                        // planner-selected checkpoint here.
                        const common_prompt_checkpoint *
                            seam_heuristic_checkpoint =
                                slot.checkpoint_seam_heuristic;
                        bool checkpoint_publication_allowed = true;
                        const bool optional_thinning_attempt =
                            slot.lifecycle_authority &&
                            slot.checkpoint_thinning_attempt_begin(false);
                        if (optional_thinning_attempt) {
                            seam_heuristic_checkpoint = nullptr;
                            const llama_pos seam_next =
                                slot.prompt.tokens.pos_next();
                            const bool has_new_tokens =
                                slot.prompt.n_tokens() < slot.task->n_tokens();
                            const llama_pos seam_min = std::max(
                                0, seam_next - n_swa -
                                    (has_new_tokens ? 0 : 1));
                            const bool recurrent =
                                llama_model_is_recurrent(model_tgt) ||
                                llama_model_is_hybrid(model_tgt);
                            const auto adapter = lora_config_identity(slot.lora);
                            for (auto it = slot.prompt.checkpoints.rbegin();
                                 it != slot.prompt.checkpoints.rend(); ++it) {
                                const bool frontier_current =
                                    checkpoint_frontier_is_current(
                                        slot, *it, adapter);
                                const bool checkpoint_lineage_matches =
                                    common_prompt_checkpoint_lineage_matches(
                                        *it, vbr_now);
                                const auto evaluation =
                                    server_cache_plan_evaluate_checkpoint(
                                        !it->empty(), frontier_current,
                                        recurrent, checkpoint_lineage_matches,
                                        it->pos_min, it->pos_max,
                                        seam_next, seam_min, it->size());
                                if (evaluation.reason ==
                                        COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL) {
                                    seam_heuristic_checkpoint = &*it;
                                    break;
                                }
                            }
                            // The pointer remains valid while the ring generation
                            // is unchanged. A committed erase/publication clears it
                            // in checkpoint_ring_changed(). The heuristic is only
                            // noise reduction; bounded replay is the proof.
                            slot.checkpoint_seam_heuristic =
                                seam_heuristic_checkpoint;
                        }
                        if (optional_thinning_attempt) {
                            bool attempt_claimed = true;
                            while (slot.checkpoint_thin_priced(
                                    ckpt_id_task,
                                    uint64_t(std::max(
                                        0, params_base.checkpoint_min_step)),
                                    seam_heuristic_checkpoint,
                                    false,
                                    attempt_claimed)) {
                                // Rebuild after every committed member removal;
                                // recovery-source pins never leak into the next union.
                                attempt_claimed = false;
                            }
                            // A failed D-A proof may not delete the incumbent,
                            // but appending a same-lineage checkpoint within the
                            // bounded replay window would retain another ~full
                            // state image on every turn. Keep the immutable older
                            // recovery point and refuse the redundant incoming
                            // publication instead. The ring is unchanged, the
                            // refusal remains typed/auditable, and the same
                            // bounded-replay relation used by the thinning proof
                            // limits the marginal replay cost.
                            if (slot.checkpoint_thinning_refusal !=
                                    common_cache_plan_destruction_reason::none &&
                                !slot.prompt.checkpoints.empty() &&
                                server_cache_checkpoint_bounded_replay(
                                    slot.prompt.checkpoints.back(), staged.back(),
                                    uint64_t(std::max(
                                        0, params_base.checkpoint_min_step)))) {
                                slot.checkpoint_publication_skipped(
                                    slot.checkpoint_thinning_refusal);
                                checkpoint_publication_allowed = false;
                            }
                        } else if (!slot.lifecycle_authority) {
                            int64_t last = -1;
                            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                                if (it->id_task != ckpt_id_task && last >= 0 &&
                                    it->n_tokens <= last + params_base.checkpoint_min_step) {
                                    SLT_TRC(slot,
                                            "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ")\n",
                                            it->pos_min, it->pos_max, it->n_tokens);
                                    it = slot.checkpoint_drop(
                                        it, std::next(it),
                                        server_cache_destruction_reason::checkpoint_thin);
                                    continue;
                                }
                                last = it->n_tokens;
                                ++it;
                            }
                        }

                        while (checkpoint_publication_allowed &&
                               slot.prompt.checkpoints.size() >=
                                   (size_t) params_base.n_ctx_checkpoints) {
                            if (slot.lifecycle_authority &&
                                slot.checkpoint_thin_priced(
                                    ckpt_id_task,
                                    uint64_t(std::max(
                                        0, params_base.checkpoint_min_step)),
                                    seam_heuristic_checkpoint,
                                    true)) {
                                continue;
                            }
                            auto victim = slot.prompt.checkpoints.begin();
                            if (slot.lifecycle_authority) {
                                common_cache_plan_destruction_reason refusal;
                                if (!slot.checkpoint_capacity_floor(
                                        ckpt_id_task,
                                        seam_heuristic_checkpoint,
                                        victim, refusal)) {
                                    slot.checkpoint_publication_skipped(refusal);
                                    checkpoint_publication_allowed = false;
                                    break;
                                }
                            }
                            // Make room for the new checkpoint in legacy order
                            // among members not protected by the D-A floor.
                            const auto & cur = *victim;

                            SLT_WRN(slot,
                                    "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                                    ", size = %.3f MiB)\n",
                                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.data_tgt.size() / 1024 / 1024);

                            if (seam_heuristic_checkpoint == &cur) {
                                seam_heuristic_checkpoint = nullptr;
                            }

                            (void) slot.checkpoint_drop(
                                victim, std::next(victim),
                                server_cache_destruction_reason::checkpoint_capacity);
                        }

                        if (!checkpoint_publication_allowed) {
                            staged.clear();
                        } else {
                            slot.prompt.checkpoints.splice(
                                slot.prompt.checkpoints.end(), staged);
                            if (slot.lifecycle_authority) {
                                slot.checkpoint_ring_changed();
                            }
                            const auto & cur = slot.prompt.checkpoints.back();
                            if (slot.retention_obs) {
                                const bool frontier_valid =
                                    cur.computation_frontier.valid() &&
                                    cur.computation_frontier.token_count ==
                                        cur.n_tokens &&
                                    cur.n_tokens >= 0 &&
                                    cur.n_tokens <= slot.task->n_tokens();
                                server_cache_lease_identity checkpoint_identity;
                                if (frontier_valid) {
                                    checkpoint_identity.execution_identity =
                                        cur.computation_frontier.execution_identity;
                                    checkpoint_identity.adapter_config_identity =
                                        cur.computation_frontier.adapter_config_identity;
                                    checkpoint_identity.media_content_identity =
                                        cur.computation_frontier.media_content_identity;
                                }
                                const bool published =
                                    slot.retention_obs->publish(
                                        server_retention_instance_key::
                                            for_checkpoint(slot.id, &cur),
                                        slot.retention_pool,
                                        slot.task->params.message_spans,
                                        !slot.task->params.message_spans.spans.empty(),
                                        uint64_t(slot.task->n_tokens()),
                                        cur.n_tokens >= 0
                                            ? uint64_t(cur.n_tokens)
                                            : 0,
                                        frontier_valid,
                                        checkpoint_identity.valid()
                                            ? &checkpoint_identity : nullptr);
                                if (published && slot.lifecycle_authority) {
                                    const auto key =
                                        server_retention_instance_key::
                                            for_checkpoint(slot.id, &cur);
                                    const auto artifact =
                                        slot.retention_obs->artifact_id(key);
                                    std::vector<llama_cache_acct_op_id> ops;
                                    const uint64_t accel_bytes =
                                        cur.accel.size();
                                    const uint64_t checkpoint_bytes =
                                        cur.size() >= accel_bytes
                                            ? cur.size() - accel_bytes
                                            : 0;
                                    if (slot.lifecycle_authority->
                                            admit_live_checkpoint(
                                                artifact,
                                                checkpoint_bytes,
                                                accel_bytes,
                                                ops) &&
                                        !slot.retention_obs->attach_release_ops(
                                            key, std::move(ops))) {
                                        SLT_WRN(slot, "%s\n",
                                            "checkpoint payload ownership attach failed; member remains fail-closed");
                                    }
                                }
                            }

                            SLT_WRN(slot,
                                    "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                                    ", size = %.3f MiB)\n",
                                    (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                                    cur.pos_max, cur.n_tokens, (float) cur.size() / 1024.0 / 1024.0);
                        }
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }

        // DFlash: enable tape recording if any slot has draft backup (needs tape replay for rollback).
        // The state stays enabled across consecutive speculative cycles and is disabled by the
        // next pre-decode batch that does not need rollback.
        // Only when GPU tape replay is available (see llama_dflash_tape_replay_available) —
        // otherwise rollback re-decodes the accepted tokens (partial-accept branch below).
        dflash_tape_active = needs_reeval
            && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH
            && dflash_tape_ok
            && std::any_of(slots.begin(), slots.end(), [](const server_slot & s) { return s.has_draft_backup; });
        // Keep the capture-enabled verification graph installed across
        // consecutive speculative cycles. This setter is idempotent; it only
        // changes the graph when the assembled batch enters or leaves a
        // rollback-capable DFlash phase.
        llama_set_tape_recording(ctx_tgt, dflash_tape_active);

        // Keep target verification logits on the device for the
        // single-slot, raw-argmax case. The batch-shape checks prove that every
        // requested output belongs to this one speculative verification; all
        // other requests retain host sampling and full-logits extraction.
        //
        // The whole DFlash family qualifies: the fork DeltaNet drafter (DFLASH)
        // and the upstream block-diffusion drafters (DRAFT_DFLASH, and
        // DRAFT_DSPARK — its anchor-first block layout only shapes the drafter
        // decode; the target verify batch is built by the shared code above as
        // [sampled, draft...], so the coverage proof is unchanged). All of them
        // verify through the shared spec_i_batch accept loop below.
        //
        // A recycle impl alongside (types is a list) reads raw target logits in
        // its update_logits(); keep those configurations on host sampling.
        dflash_target_argmax_active = false;
        dflash_target_argmax_slot = -1;
        const common_speculative_type spec_type = params_base.speculative.type();
        const bool spec_type_target_argmax =
            spec_type == COMMON_SPECULATIVE_TYPE_DFLASH ||
            spec_type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
            spec_type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        if (params_base.n_parallel == 1 && spec_type_target_argmax &&
            !params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_RECYCLE)) {
            server_slot * verify_slot = nullptr;
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING ||
                    !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }
                if (verify_slot != nullptr) {
                    verify_slot = nullptr;
                    break;
                }
                verify_slot = &slot;
            }
            if (verify_slot && verify_slot->task &&
                verify_slot->spec_i_batch.size() == (size_t) batch.size() &&
                common_sampler_raw_argmax_exact(verify_slot->smpl.get())) {
                bool covers_batch = true;
                for (size_t i = 0; i < verify_slot->spec_i_batch.size(); ++i) {
                    covers_batch &= verify_slot->spec_i_batch[i] == (int32_t) i;
                }
                if (covers_batch) {
                    dflash_target_argmax_active = true;
                    dflash_target_argmax_slot = verify_slot->id;
                }
            }
        }
        ctx_tgt->set_dflash_target_argmax(dflash_target_argmax_active);

        // allow multi-seq batching when the batch is pure TG (no prompt tokens).
        // This lets concurrent slots' verify tokens be processed in a single
        // multi-seq ubatch instead of N sequential per-seq ubatches.
        // (force_split_seq is restored in post_cycle())
        can_batch_multiseq = (n_tg_tokens == batch.size() && n_tg_tokens > 0
            && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH);
        if (can_batch_multiseq) {
            llama_set_force_split_seq(ctx_tgt, false);
        }

        // dynamic VBR: mid-decode pressure — generation can cross a page/budget boundary long
        // after launch. Same floor-gated reclaim before the cache pays with tiers (idle slots
        // only; every generating slot is processing and untouchable).
        if (batch.size() > 0) {
            vbr_reclaim_before_degrade(-1, (uint32_t) batch.size(), "decode");
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        if (batch.size() == 0) {
            // note: normally unreachable — update_slots() skips decode() for an empty batch and
            //       handles the empty-batch warn/abort (with the diffusion exemption) itself
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft.get()) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        // Ensure causal attention is on for prompt eval (diffusion draft toggles it off)
        llama_set_causal_attn(ctx_tgt, true);

        const int64_t t_verify_start = ggml_time_us();
        const int ret = llama_decode(ctx_tgt, batch_view);
        const int64_t t_verify_elapsed = ggml_time_us() - t_verify_start;
        t_verify_total += t_verify_elapsed;
        SRV_DBG("  verify ubatch: %d tok, %.1fms (%.2fms/tok)\n",
                batch_view.n_tokens, t_verify_elapsed / 1e3, t_verify_elapsed / 1e3 / std::max(1, batch_view.n_tokens));

        metrics.on_decoded(slots);

        auto * memory = llama_get_memory(ctx_tgt);
        std::vector<vbr_hard_seal_subject> hard_seal_evidence;
        memory->vbr_hard_seal_evidence_take(hard_seal_evidence);
        if (params_base.cache_debug) {
            for (const auto & step : hard_seal_evidence) {
                const json payload = {
                    {"evidence_event", "sealed_step"},
                    {"order_ordinal", step.order_ordinal},
                    {"layer", step.il},
                    {"side", step.is_v ? "v" : "k"},
                };
                SRV_INF("CACHE_VBR_HARD_SEAL %s\n", payload.dump().c_str());
            }
        }
        const bool hard_seal_terminal =
            memory->vbr_hard_seal_blocked_take(ret != 0);
        if (ret != 0) {
            if (ret == 1 && hard_seal_terminal) {
                constexpr const char * refusal = "hard_lease_blocked";
                SRV_WRN("VBR pressure refused at the hard-lease seal; completing the in-flight request with %s\n",
                        refusal);
                for (auto & slot : slots) {
                    if (slot.is_processing()) {
                        send_error(slot, refusal, ERROR_TYPE_HARD_LEASE_BLOCKED);
                        // prepare() failed before committing this batch. Keep
                        // the proven live prefix and its lease intact; unlike
                        // an ordinary cache-placement failure this must never
                        // clear an idle leased slot or retry forever.
                        slot.release();
                    }
                }
                throw std::runtime_error(refusal);
            }
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.mandatory_recovery_reset(
                                server_cache_destruction_reason::restore_failure);
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        }


        if (batch_view.logits != nullptr) {
            for (int32_t i = 0; i < batch_view.n_tokens; ++i) {
                cycle_has_output |= batch_view.logits[i] != 0;
            }
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (!common_speculative_process(spec.get(), batch_view)) {
            SRV_ERR("%s", "failed to process speculative batch\n");

            // TODO: handle error
            throw std::runtime_error("failed to process speculative batch");
        }

        // DFlash: flush captured hidden states into the ring buffer before
        // the next llama_decode resets the capture buffer. This lets
        // checkpoint-split prefill preserve all hidden states incrementally.
        for (auto & slot : slots) {
            if ((slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) && slot.can_speculate()) {
                common_speculative_flush_prefill(slot.get_spec());
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    if (!slot.copy_state_to(*child)) {
                        // the recurrent pool had no free cell to clone into: fail the child rather
                        // than run it on empty state [I13/N4]
                        SLT_ERR(*child, "%s\n", "failed to copy parent state to child (recurrent pool full)");
                        send_error(*child, "Failed to clone parent state to child slot");
                        child->release();
                        child->mandatory_recovery_reset(
                            server_cache_destruction_reason::restore_failure);
                        continue;
                    }
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {

                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    // B0: no first generated token exists — TTFT stays typed unavailable
                    cache_plan_finalize(slot, /*ttft_known=*/false);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    cache_plan_finalize(slot, /*ttft_known=*/false);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.diff_self_spec) {
                    float * logits = llama_get_logits(ctx_tgt);
                    int32_t nv = llama_vocab_n_tokens(vocab);
                    slot.diff_prev_logits.resize(nv);
                    std::memcpy(slot.diff_prev_logits.data(), logits, nv * sizeof(float));

                    // Extract previous assistant response tokens for repetition penalty.
                    // Find all <|im_start|>assistant blocks, take the second-to-last
                    // (the last is the current generation position).
                    slot.diff_prev_assistant_tokens.clear();
                    {
                        const auto & toks = slot.prompt.tokens.get_text_tokens();
                        const llama_token im_start = 10;  // <|im_start|>
                        const llama_token im_end   = 11;  // <|im_end|>
                        const llama_token tok_ass  = 1503; // "ass"
                        const llama_token tok_ist  = 19464; // "istant"
                        int n = (int)toks.size();

                        // Find all assistant block start positions
                        std::vector<int> asst_starts;
                        for (int j = 0; j + 2 < n; j++) {
                            if (toks[j] == im_start && toks[j+1] == tok_ass && toks[j+2] == tok_ist) {
                                asst_starts.push_back(j);
                            }
                        }

                        // If there are 2+ assistant blocks, the second-to-last has
                        // the previous response
                        if (asst_starts.size() >= 2) {
                            int prev_start = asst_starts[asst_starts.size() - 2];
                            int content_start = prev_start + 4; // <|im_start|> ass istant \n
                            // Find <|im_end|> after content
                            int end = content_start;
                            while (end < n && toks[end] != im_end) end++;
                            for (int j = content_start; j < end; j++) {
                                slot.diff_prev_assistant_tokens.push_back(toks[j]);
                            }
                        }
                    }

                    SLT_DBG(slot, "diff prefill: n_prompt=%d, prev_asst_tokens=%d\n",
                            slot.prompt.n_tokens(), (int)slot.diff_prev_assistant_tokens.size());
                    slot.t_start_generation = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                    cache_plan_finalize(slot); // B0: same timing point as the metrics clock

                    slot.i_batch = -1;
                    return;
                }

                if (slot.can_speculate()) {
                    if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                        llama_dflash_set_active_slot(ctx_tgt, slot.id);
                    }
                    if (slot.spec) {
                        // fork per-slot speculative state
                        common_speculative_begin(slot.get_spec(), slot.prompt.tokens.get_text_tokens());
                    } else if (spec) {
                        // upstream shared multi-seq speculative state
                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                    }
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // update DFlash hidden state ring buffer with the decoded token's hidden states.
            // Skip on the first sample after prompt: common_speculative_begin() above already
            // populated the ring with all prefill hiddens. The capture buffer at this point
            // still holds prefill hiddens (no new decode happened), so ring_write(1) here would
            // append a stale duplicate at the position that should later hold `id`'s hidden —
            // silently corrupting the drafter's cross-attention context on every subsequent
            // verify. Fires correctly on the fallback non-spec path during generation
            // (draft too small → single-token decode), where slot.sampled was just decoded.
            if (slot.can_speculate() && slot.n_decoded > 0) {
                if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                }
                llama_tokens batch_tokens = { id };
                common_speculative_update_logits(slot.get_spec(), ctx_tgt, batch_tokens, 1);
            }

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.n_decoded += 1;

            if (slot.n_decoded == 1) {
                slot.t_start_generation = t_now;
                slot.t_print_last = t_now;
                slot.n_decoded_last = 0;
                slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                metrics.on_prompt_eval(slot);
                cache_plan_finalize(slot); // B0: replay + every trim/fallback settled, TTFT real
            }

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                metrics.on_prediction(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        const int64_t t_accept_start = ggml_time_us();
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                continue;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            // defensive (#74): never feed a desynced spec state into
            // common_sampler_sample_and_accept_n (it hard-asserts on
            // idxs.size() == draft.size() + 1). The batch build site always appends
            // the current cycle's (n_draft + 1) contiguous indices LAST, so any
            // excess can only be a stale prefix leaked from an earlier cycle.
            // Unreachable with the empty-draft gate in place — this converts any
            // future recurrence from a server abort into a loud warning.
            if (slot.spec_i_batch.size() != n_draft + 1) {
                SLT_WRN(slot, "spec state desync (spec_i_batch = %zu, spec_draft = %zu) - recovering\n",
                        slot.spec_i_batch.size(), n_draft);

                if (slot.spec_i_batch.size() > n_draft + 1) {
                    // drop the stale prefix, keep the current cycle's indices
                    slot.spec_i_batch.erase(slot.spec_i_batch.begin(), slot.spec_i_batch.end() - (n_draft + 1));
                } else {
                    // should be unreachable: verify only as many draft tokens as
                    // there are logit indices (n_draft keeps the original size so
                    // the prompt rollback below stays consistent)
                    slot.spec_draft.resize(slot.spec_i_batch.empty() ? 0 : slot.spec_i_batch.size() - 1);
                    if (slot.spec_i_batch.empty()) {
                        slot.spec_i_batch.push_back(batch.size() > 0 ? batch.size() - 1 : 0);
                    }
                }
            }

            // The accepted tokens from the speculation. For the narrowly
            // proven raw-argmax case, verification copied only token ids from
            // the target graph; advance the ordinary sampler history with the
            // exact same accepted sequence. Unsupported configurations never
            // arm this path and continue sampling the full host logits.
            std::vector<llama_token> ids;
            bool accepted_from_target_argmax = false;
            if (dflash_target_argmax_active &&
                dflash_target_argmax_slot == slot.id) {
                const int32_t * argmax = llama_get_logits_argmax(ctx_tgt);
                if (argmax != nullptr) {
                    if (llama_get_logits_argmax_k(ctx_tgt) != 1 ||
                        llama_get_logits_argmax_n(ctx_tgt) !=
                            (int32_t) slot.spec_i_batch.size()) {
                        throw std::runtime_error(
                            "DFlash target argmax output shape mismatch");
                    }

                    llama_tokens sampled;
                    sampled.reserve(slot.spec_i_batch.size());
                    for (size_t i = 0; i < slot.spec_i_batch.size(); ++i) {
                        const llama_token id = ctx_tgt->get_logits_argmax_ith(
                            slot.spec_i_batch[i]);
                        if (id == LLAMA_TOKEN_NULL) {
                            throw std::runtime_error(
                                "DFlash target argmax row lookup failed");
                        }
                        sampled.push_back(id);
                    }
                    ids = common_sampler_accept_draft(
                        slot.smpl.get(), sampled, slot.spec_draft);
                    accepted_from_target_argmax = true;
                }
            }
            if (!accepted_from_target_argmax) {
                ids = common_sampler_sample_and_accept_n(
                    slot.smpl.get(), ctx_tgt,
                    slot.spec_i_batch, slot.spec_draft);
            }


            // update DFlash hidden state ring + CopySpec prompt window with accepted tokens.
            // Must run BEFORE rollback (matches speculative-simple ordering) and BEFORE clearing
            // slot.spec_draft so batch_tokens reflects the full verification batch [id_last, drafts].
            {
                if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                }
                llama_tokens batch_tokens;
                batch_tokens.push_back(slot.sampled);
                batch_tokens.insert(batch_tokens.end(), slot.spec_draft.begin(), slot.spec_draft.end());
                common_speculative_update_logits(slot.get_spec(), ctx_tgt, batch_tokens, (int) ids.size());
            }

            slot.spec_i_batch.clear();
            slot.spec_draft.clear();

            const int64_t t_current = ggml_time_us();

            slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

            // update how many tokens out of those tested were accepted
            slot.n_draft_accepted += ids.size() - 1;
            slot.n_draft_verif_steps += 1;

            // Verification evaluates [sampled, draft...]. The sampled row is
            // always retained; rollback depth is exactly the rejected draft
            // suffix. Preserve the joint distribution because marginal
            // acceptance rates cannot recover capture cost by verify length.
            const size_t n_accepted_draft = ids.size() - 1;
            GGML_ASSERT(n_accepted_draft <= n_draft);
            const size_t rollback_depth = n_draft - n_accepted_draft;
            if (slot.n_verify_rollback.size() <= n_draft) {
                slot.n_verify_rollback.resize(n_draft + 1);
            }
            auto & rollback_counts = slot.n_verify_rollback[n_draft];
            if (rollback_counts.size() <= rollback_depth) {
                rollback_counts.resize(rollback_depth + 1, 0);
            }
            rollback_counts[rollback_depth]++;

            // per-position acceptance histogram (/metrics)
            if (slot.n_accepted_per_pos.empty()) {
                slot.n_accepted_per_pos.resize(common_speculative_n_max(&params_base.speculative), 0);
            }
            for (size_t i = 0; i < ids.size() - 1 && i < slot.n_accepted_per_pos.size(); ++i) {
                slot.n_accepted_per_pos[i]++;
            }

            // notify the shared (upstream) speculative state, if this slot uses it
            if (!slot.spec && spec) {
                common_speculative_accept(spec.get(), slot.id, ids.size() - 1);
            }

            // add accepted tokens to the prompt
            slot.server_cache_transient_token_truncate_impl(
                slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert(llama_tokens(ids.begin(), ids.end() - 1));

            if (slot.has_draft_backup) {
                const llama_seq_id seq_backup = slot.seq_id_backup;
                const bool all_accepted = (ids.size() == n_draft + 1);

                const bool is_dflash = params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH;
                if (is_dflash) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                    llama_clear_tree_parent_ids(ctx_tgt);
                }

                bool restored_by_tape = false;
                if (is_dflash && !all_accepted && dflash_tape_ok) {
                    // lossless GPU tape replay of the accepted tokens' GDN state updates
                    restored_by_tape = llama_dflash_rollback(
                        ctx_tgt, slot.id, seq_backup,
                        slot.n_tokens_before_draft, (int) ids.size());
                    if (!restored_by_tape) {
                        SLT_WRN(
                            slot, "%s\n",
                            "exact tape replay launch failed; restoring backup and re-decoding accepted tokens");
                    }
                }
                if (!restored_by_tape) {
                    // no tape recorded (see dflash_tape_active) or non-DFlash drafter:
                    // restore from backup and re-decode the accepted tokens (exact,
                    // replay-free). dflash_rollback deliberately retains the backup
                    // when its exact replay launch fails.
                    auto * mem = llama_get_memory(ctx_tgt);

                    if (all_accepted) {
                        server_cache_transient_seq_rm_impl(
                            mem, seq_backup, -1, -1);
                        server_cache_transient_seq_rm_impl(
                            mem, slot.id, slot.prompt.tokens.pos_next(), -1);
                    } else {
                        const int n_past_before = slot.n_tokens_before_draft;

                        // Non-RS recurrent state cannot trim a rejected suffix in place.
                        // Free the known destination first so restoring a known-good
                        // backup does not require a third cell in a full live+backup pool.
                        const bool live_removed = server_cache_transient_seq_rm_impl(
                            mem, slot.id, -1, -1);
                        if (!live_removed ||
                            !llama_memory_try_seq_cp_transient(mem, seq_backup, slot.id, -1, -1)) {
                            // could not restore the backup into the slot (recurrent pool exhausted):
                            // the slot is left at n_past_before with the accepted tokens unresolved.
                            // Reset it coherently rather than continue on inconsistent state [I13].
                            SLT_ERR(slot, "%s\n", "failed to restore speculative backup; resetting slot");
                            send_error(slot, "Compute error restoring speculative backup");
                            slot.release();
                            slot.mandatory_recovery_reset(
                                server_cache_destruction_reason::restore_failure);
                            slot.has_draft_backup = false;
                            slot.seq_id_backup = -1;
                            continue;
                        } else {
                            server_cache_transient_seq_rm_impl(
                                mem, seq_backup, -1, -1);

                            const int n_reeval = slot.prompt.n_tokens() - n_past_before;
                            if (n_reeval > 0) {
                                llama_batch batch_reeval = llama_batch_init(n_reeval, 0, 1);
                                const auto & toks = slot.prompt.tokens.get_text_tokens();
                                for (int j = n_past_before; j < slot.prompt.n_tokens(); ++j) {
                                    common_batch_add(batch_reeval, toks[j], j, { slot.id }, false);
                                }
                                const int ret_reeval = llama_decode(ctx_tgt, batch_reeval);
                                llama_batch_free(batch_reeval);
                                if (ret_reeval != 0) {
                                    // the backup was restored to n_past_before but slot.prompt.tokens
                                    // already holds the accepted tokens, so a failed re-decode leaves
                                    // memory and bookkeeping desynced. Abort this slot and reset it
                                    // coherently rather than generating on inconsistent state [R8/N8].
                                    SLT_ERR(slot, "failed to re-decode accepted tokens after partial accept (ret = %d)\n", ret_reeval);
                                    send_error(slot, "Compute error re-decoding accepted tokens");
                                    slot.release();
                                    slot.mandatory_recovery_reset(
                                        server_cache_destruction_reason::restore_failure);
                                    slot.has_draft_backup = false;
                                    slot.seq_id_backup = -1;
                                    continue;
                                }
                            }
                        }
                    }
                }

                slot.has_draft_backup = false;
                slot.seq_id_backup = -1;
            } else if (!server_cache_transient_seq_rm_impl(
                           llama_get_memory(ctx_tgt), slot.id,
                           slot.prompt.tokens.pos_next(), -1)) {
                // RS contexts own their rollback snapshot internally, so this is the
                // only target-state rollback operation on the no-backup path. Refusal
                // cannot be ignored: accepted-token bookkeeping would otherwise advance
                // over an untrimmed rejected suffix.
                SLT_ERR(slot, "%s\n", "failed to roll back rejected speculative suffix; resetting slot");
                send_error(slot, "Compute error rolling back speculative tokens");
                slot.release();
                slot.mandatory_recovery_reset(
                    server_cache_destruction_reason::restore_failure);
                continue;
            }

            common_speculative_rollback_dft(slot.get_spec(), slot.id, slot.prompt.n_tokens(), (uint16_t)(ids.size() - 1));

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.n_decoded += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    break;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
        }
        t_accept_total += ggml_time_us() - t_accept_start;
    }

    // fork: once-per-update_slots() work that must run after ALL sub-batches:
    // diffusion self-speculation, the spec-cycle report, DFlash tape-off and
    // force_split_seq restore (kept out of post_decode(), which runs per sub-batch)
    void post_cycle() {
        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        // diffusion self-speculation — runs independently of the main batch
        for (auto & slot : slots) {
            if (!slot.diff_self_spec || slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            llama_memory_t mem = llama_get_memory(ctx_tgt);
            const int32_t k = slot.diff_draft_length;

            const float temperature = slot.task->params.sampling.temp;
            std::mt19937 rng(slot.task->params.sampling.seed != LLAMA_DEFAULT_SEED
                ? slot.task->params.sampling.seed : std::random_device{}());
            std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

            auto argmax = [&](const float * l) -> llama_token {
                return (llama_token)(std::max_element(l, l + n_vocab) - l);
            };

            auto suppress_think = [&](float * l) {
                if (slot.diff_think_open_id  != LLAMA_TOKEN_NULL) l[slot.diff_think_open_id]  = -INFINITY;
                if (slot.diff_think_close_id != LLAMA_TOKEN_NULL) l[slot.diff_think_close_id] = -INFINITY;
            };

            // Temperature sampling: softmax(logits/T) → categorical sample
            auto sample_temp = [&](const float * logits) -> llama_token {
                float max_val = *std::max_element(logits, logits + n_vocab);
                float sum = 0;
                std::vector<float> probs(n_vocab);
                for (int32_t v = 0; v < n_vocab; v++) {
                    probs[v] = std::exp((logits[v] - max_val) / temperature);
                    sum += probs[v];
                }
                float r = uniform01(rng) * sum;
                float cumsum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    cumsum += probs[v];
                    if (r <= cumsum) return (llama_token)v;
                }
                return (llama_token)(n_vocab - 1);
            };

            auto sample_adjusted = [&](const float * p_logits, const float * q_logits) -> llama_token {
                float p_max = *std::max_element(p_logits, p_logits + n_vocab);
                float q_max = *std::max_element(q_logits, q_logits + n_vocab);
                float p_sum = 0, q_sum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    p_sum += std::exp((p_logits[v] - p_max) / temperature);
                    q_sum += std::exp((q_logits[v] - q_max) / temperature);
                }
                std::vector<float> adj(n_vocab);
                float sum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    float pv = std::exp((p_logits[v] - p_max) / temperature) / p_sum;
                    float qv = std::exp((q_logits[v] - q_max) / temperature) / q_sum;
                    adj[v] = std::max(0.0f, pv - qv);
                    sum += adj[v];
                }
                if (sum <= 0) return sample_temp(p_logits);
                float r = uniform01(rng) * sum;
                float cumsum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    cumsum += adj[v];
                    if (r <= cumsum) return (llama_token)v;
                }
                return (llama_token)(n_vocab - 1);
            };

            // Acceptance probability: min(1, p(x)/q(x)) where p=target, q=draft
            auto accept_prob = [&](const float * p_logits, const float * q_logits, llama_token tok) -> float {
                float p_max = *std::max_element(p_logits, p_logits + n_vocab);
                float q_max = *std::max_element(q_logits, q_logits + n_vocab);
                float p_sum = 0, q_sum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    p_sum += std::exp((p_logits[v] - p_max) / temperature);
                    q_sum += std::exp((q_logits[v] - q_max) / temperature);
                }
                float p_tok = std::exp((p_logits[tok] - p_max) / temperature) / p_sum;
                float q_tok = std::exp((q_logits[tok] - q_max) / temperature) / q_sum;
                return std::min(1.0f, p_tok / std::max(1e-10f, q_tok));
            };

            // Cross-turn: penalize tokens from previous assistant response
            std::unordered_map<llama_token, int> prev_asst_freq;
            for (auto tok : slot.diff_prev_assistant_tokens) {
                prev_asst_freq[tok]++;
            }
            std::vector<llama_token> gen_history;

            const float rep_penalty = 1.3f;
            auto apply_rep_penalty = [&](float * l) {
                for (const auto & [tok, freq] : prev_asst_freq) {
                    if (l[tok] > 0) l[tok] /= rep_penalty;
                    else             l[tok] *= rep_penalty;
                }
            };
            auto detect_loop = [&]() -> bool {
                int n = (int)gen_history.size();

                // Short-period exact match (periods 4-32)
                if (n >= 16) {
                    for (int period = 4; period <= std::min(32, n/2); period++) {
                        bool match = true;
                        for (int i = 0; i < period; i++) {
                            if (gen_history[n - 1 - i] != gen_history[n - 1 - i - period]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            SLT_INF(slot, "diff loop detected (period=%d), stopping generation\n", period);
                            return true;
                        }
                    }
                }

                // Long-period structural loop: any 6-gram appearing 6+ times in last 300 tokens
                const int struct_window = std::min(n, 300);
                if (struct_window >= 60) {
                    int start = n - struct_window;
                    std::unordered_map<uint64_t, int> ng6_freq;
                    for (int i = start; i <= n - 6; i++) {
                        uint64_t h = 0;
                        for (int j = 0; j < 6; j++) {
                            h = h * 100003ULL + (uint64_t)gen_history[i + j];
                        }
                        ng6_freq[h]++;
                    }
                    for (const auto & [h, freq] : ng6_freq) {
                        if (freq >= 6) {
                            SLT_INF(slot, "diff structural loop detected (6-gram freq=%d), stopping generation\n", freq);
                            return true;
                        }
                    }
                }
                return false;
            };

            std::vector<llama_token> draft(k);
            const bool use_temp = temperature > 0;
            std::vector<float> saved_draft_logits;
            if (use_temp) {
                saved_draft_logits.resize(k * n_vocab);
            }
            bool stopped = false;
            int64_t t_draft_us = 0, t_verify_us = 0, t_bonus_us = 0, t_loop_us = 0, n_cycles = 0;

            while (!stopped) {
                const int64_t t_loop_start = ggml_time_us();
                const int32_t committed = slot.prompt.n_tokens();
                const int32_t draft_len = std::min(k, (int32_t)(slot.n_ctx - committed - 1));
                if (draft_len <= 0) {
                    slot.truncated      = true;
                    slot.stop           = STOP_TYPE_LIMIT;
                    slot.has_next_token = false;
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();
                    break;
                }

                // DRAFT: bidirectional — single pass, all masks denoised in parallel
                llama_set_causal_attn(ctx_tgt, false);
                {
                    llama_batch bd = llama_batch_init(draft_len, 0, 1);
                    bd.n_tokens = draft_len;
                    for (int32_t j = 0; j < draft_len; j++) {
                        bd.token[j]     = slot.diff_mask_token_id;
                        bd.pos[j]       = committed + j;
                        bd.n_seq_id[j]  = 1;
                        bd.seq_id[j][0] = slot.id;
                        bd.logits[j]    = 1;
                    }
                    int64_t t0 = ggml_time_us();
                    int ret = llama_decode(ctx_tgt, bd);
                    t_draft_us += ggml_time_us() - t0;
                    llama_batch_free(bd);
                    if (ret != 0) {
                        SRV_ERR("diffusion draft decode failed, ret = %d\n", ret);
                        stopped = true;
                        break;
                    }
                }

                {
                    float * logits = llama_get_logits(ctx_tgt);
                    for (int32_t j = 0; j < draft_len; j++) {
                        suppress_think(logits + j * n_vocab);
                        apply_rep_penalty(logits + j * n_vocab);
                    }
                    if (use_temp) {
                        std::memcpy(saved_draft_logits.data(), logits, draft_len * n_vocab * sizeof(float));
                        for (int32_t j = 0; j < draft_len; j++) {
                            draft[j] = sample_temp(logits + j * n_vocab);
                        }
                    } else {
                        for (int32_t j = 0; j < draft_len; j++) {
                            draft[j] = argmax(logits + j * n_vocab);
                        }
                    }
                }

                server_cache_transient_seq_rm_impl(
                    mem, slot.id, committed, committed + draft_len);

                // VERIFY: causal — single pass with draft tokens
                llama_set_causal_attn(ctx_tgt, true);
                {
                    llama_batch bv = llama_batch_init(draft_len, 0, 1);
                    bv.n_tokens = draft_len;
                    for (int32_t j = 0; j < draft_len; j++) {
                        bv.token[j]     = draft[j];
                        bv.pos[j]       = committed + j;
                        bv.n_seq_id[j]  = 1;
                        bv.seq_id[j][0] = slot.id;
                        bv.logits[j]    = 1;
                    }
                    int64_t t1 = ggml_time_us();
                    int ret = llama_decode(ctx_tgt, bv);
                    t_verify_us += ggml_time_us() - t1;
                    llama_batch_free(bv);
                    if (ret != 0) {
                        SRV_ERR("diffusion verify decode failed, ret = %d\n", ret);
                        stopped = true;
                        break;
                    }
                }

                float * verify_logits = llama_get_logits(ctx_tgt);
                suppress_think(slot.diff_prev_logits.data());
                apply_rep_penalty(slot.diff_prev_logits.data());
                for (int32_t j = 0; j < draft_len; j++) {
                    suppress_think(verify_logits + j * n_vocab);
                    apply_rep_penalty(verify_logits + j * n_vocab);
                }

                int32_t n_accept = 0;
                llama_token bonus;

                if (use_temp) {
                    // Rejection sampling: accept draft[j] with prob min(1, p(d)/q(d))
                    float ap = accept_prob(slot.diff_prev_logits.data(), saved_draft_logits.data(), draft[0]);
                    if (uniform01(rng) < ap) {
                        n_accept = 1;
                        for (int32_t j = 0; j < draft_len - 1; j++) {
                            ap = accept_prob(verify_logits + j * n_vocab,
                                           saved_draft_logits.data() + (j + 1) * n_vocab, draft[j + 1]);
                            if (uniform01(rng) < ap) {
                                n_accept++;
                            } else {
                                break;
                            }
                        }
                    }
                    if (n_accept == draft_len) {
                        bonus = sample_temp(verify_logits + (draft_len - 1) * n_vocab);
                    } else if (n_accept == 0) {
                        bonus = sample_adjusted(slot.diff_prev_logits.data(), saved_draft_logits.data());
                    } else {
                        bonus = sample_adjusted(verify_logits + (n_accept - 1) * n_vocab,
                                              saved_draft_logits.data() + n_accept * n_vocab);
                    }
                } else {
                    if (argmax(slot.diff_prev_logits.data()) == draft[0]) {
                        n_accept = 1;
                        for (int32_t j = 0; j < draft_len - 1; j++) {
                            if (argmax(verify_logits + j * n_vocab) == draft[j + 1]) {
                                n_accept++;
                            } else {
                                break;
                            }
                        }
                    }
                    bonus = (n_accept == 0)
                        ? argmax(slot.diff_prev_logits.data())
                        : argmax(verify_logits + (n_accept - 1) * n_vocab);
                }

                if (n_accept < draft_len) {
                    server_cache_transient_seq_rm_impl(
                        mem, slot.id, committed + n_accept,
                        committed + draft_len);
                }

                // BONUS: decode one token — produces new prev_logits for next cycle
                {
                    llama_batch bb = llama_batch_init(1, 0, 1);
                    bb.n_tokens     = 1;
                    bb.token[0]     = bonus;
                    bb.pos[0]       = committed + n_accept;
                    bb.n_seq_id[0]  = 1;
                    bb.seq_id[0][0] = slot.id;
                    bb.logits[0]    = 1;
                    int64_t t2 = ggml_time_us();
                    int ret = llama_decode(ctx_tgt, bb);
                    t_bonus_us += ggml_time_us() - t2;
                    llama_batch_free(bb);
                    if (ret != 0) {
                        SRV_ERR("diffusion bonus decode failed, ret = %d\n", ret);
                        stopped = true;
                        break;
                    }
                    std::memcpy(slot.diff_prev_logits.data(), llama_get_logits(ctx_tgt), n_vocab * sizeof(float));
                    suppress_think(slot.diff_prev_logits.data());
                }

                n_cycles++;

                // output accepted draft tokens + bonus
                for (int32_t j = 0; j < n_accept && !stopped; j++) {
                    completion_token_output result;
                    result.tok          = draft[j];
                    result.text_to_send = common_token_to_piece(ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f;
                    slot.n_decoded += 1;
                    slot.prompt.tokens.push_back(draft[j]);
                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();
                        stopped = true;
                    }
                }
                if (!stopped) {
                    completion_token_output result;
                    result.tok          = bonus;
                    result.text_to_send = common_token_to_piece(ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f;
                    slot.n_decoded += 1;
                    slot.prompt.tokens.push_back(bonus);
                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();
                        stopped = true;
                    }
                }

                if (!stopped) {
                    for (int32_t j = 0; j < n_accept; j++) {
                        gen_history.push_back(draft[j]);
                    }
                    gen_history.push_back(bonus);
                    if (detect_loop()) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();
                        stopped = true;
                    }

                    slot.n_draft_accepted += n_accept;
                    slot.t_token_generation = std::max<int64_t>(1, ggml_time_us() - slot.t_start_generation) / 1e3;
                    t_loop_us += ggml_time_us() - t_loop_start;
                    SLT_DBG(slot, "diff cycle %lld: %d/%d accepted, bonus=%d, pos=%d\n",
                            (long long)n_cycles, n_accept, draft_len, bonus, slot.prompt.n_tokens());
                }
            }

            if (n_cycles > 0) {
                int64_t t_decode_us = t_draft_us + t_verify_us + t_bonus_us;
                int64_t t_overhead_us = t_loop_us - t_decode_us;
                SLT_INF(slot, "diff stats: %lld cycles, draft=%.1fms verify=%.1fms bonus=%.1fms overhead=%.1fms (%.1f/%.1f/%.1f/%.1f ms/cycle)\n",
                        (long long)n_cycles,
                        t_draft_us/1e3, t_verify_us/1e3, t_bonus_us/1e3, t_overhead_us/1e3,
                        t_draft_us/1e3/n_cycles, t_verify_us/1e3/n_cycles, t_bonus_us/1e3/n_cycles, t_overhead_us/1e3/n_cycles);
            }

            llama_set_causal_attn(ctx_tgt, true);
        }

        // --- profiling: log per-cycle breakdown ---
        if (n_slots_drafted > 0) {
            const int64_t t_cycle_total = ggml_time_us() - t_cycle_start;
            const int64_t t_other = t_cycle_total - t_draft_total - t_verify_total - t_accept_total;
            SRV_DBG("spec cycle (%d slots): draft=%.1fms verify=%.1fms accept=%.1fms other=%.1fms total=%.1fms\n",
                    n_slots_drafted,
                    t_draft_total / 1e3, t_verify_total / 1e3, t_accept_total / 1e3,
                    t_other / 1e3, t_cycle_total / 1e3);
        }

        // restore force_split_seq for the next cycle (prompt batches need it)
        if (can_batch_multiseq) {
            llama_set_force_split_seq(ctx_tgt, true);
        }

        // The generic target-output hook is intentionally disabled for a
        // composite speculative load: the drafter can allocate after that
        // target output. Complete only after a successful cycle crossed the
        // draft path. If speculative initialization fell back to target-only,
        // preserve the direct first-output behavior.
        const bool any_spec_context = std::any_of(slots.begin(), slots.end(),
                [](const server_slot & slot) { return slot.can_speculate(); });
        if (!cycle_failed && (n_slots_draft_decode_succeeded > 0 || (cycle_has_output && !any_spec_context))) {
            llama_vram_load_complete();
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* vbr_enabled            */ impl->params_base.vbr_enabled(),
        /* vbr_dynamic            */ impl->params_base.vbr_dynamic(),
        /* vbr_type_k             */ impl->params_base.vbr_cache_type_k,
        /* vbr_type_v             */ impl->params_base.vbr_cache_type_v,
        /* vbr_min_bits           */ impl->params_base.vbr_min_bits_value,
        /* vbr_capacity_bits      */ impl->params_base.vbr_capacity_bits,
        /* vbr_selected_bpv       */ impl->params_base.vbr_selected_bpv,
        /* vbr_selected_kld       */ impl->params_base.vbr_selected_kld,
        /* vbr_vram_budget_bytes  */ impl->params_base.vbr_vram_budget_bytes,
        /* vbr_selected_family    */ impl->params_base.vbr_selected_family,
        /* vbr_selected_policy    */ impl->params_base.vbr_selected_policy,
        /* vbr_selected_schedule  */ impl->params_base.vbr_selected_schedule,
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
    impl->queue_tasks.on_sleeping_state([this](bool sleeping) {
        if (sleeping) {
            impl->callback_state(SERVER_STATE_SLEEPING, {});
        }
        // for sleeping == false, event is emitted by load_model()
    });
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        auto delimiters = common_chat_msg_delimiters_parse(json_value(data, "message_delimiters", json::array()));
        delimiters.tokenize(ctx_server.vocab);

        server_cache_control_token family_binding_token;
        if (data.contains("family_binding")) {
            if (!params.cache_control_api ||
                !data.at("family_binding").is_string() ||
                !server_cache_control_decode_handle(
                    server_cache_control_handle_kind::family_binding,
                    data.at("family_binding").get<std::string>(),
                    family_binding_token)) {
                throw std::runtime_error(
                    "family_binding requires --cache-control-api and a valid opaque binding");
            }
        }

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            task.cache_family_binding_token = family_binding_token;
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "spec_decode_num_draft_tokens_total"},
                    {"help",  "Total draft tokens generated"},
                    {"value",  res_task->n_draft_tokens_total}
            }, {
                    {"name",  "spec_decode_num_accepted_tokens_total"},
                    {"help",  "Total draft tokens accepted by the target model"},
                    {"value",  res_task->n_draft_accepted_total}
            }, {
                    {"name",  "spec_decode_num_drafts_total"},
                    {"help",  "Total speculative decoding verification steps"},
                    {"value",  res_task->n_draft_verif_steps_total}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        // labeled counter: one time series per draft position
        if (!res_task->n_accepted_per_pos_total.empty()) {
            prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                          " Accepted tokens per draft position\n"
                       << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
            for (size_t i = 0; i < res_task->n_accepted_per_pos_total.size(); i++) {
                prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                           << i << "\"} " << res_task->n_accepted_per_pos_total[i] << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "capture") {
            return handle_slots_capture(req, id_slot);
        }
        if (action == "import") {
            return handle_slots_import(req, id_slot);
        }
        // Erase does no state-file IO (prompt clear + seq_rm only), so it is
        // NOT gated on --slot-save-path. Dynamic VBR clears slot_save_path at
        // startup (legacy state files cannot carry tier-typed KV), and import
        // REQUIRES an explicit erase to produce its empty target — keeping
        // erase behind the path gate made import's precondition unreachable
        // on exactly the servers that support import.
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->post_cache_plan = [this](const server_http_req & req) {
        auto res = create_response();
        // E0 is point-in-time advice. Every terminal, including parse and
        // feature-gate errors, is route-locally non-cacheable.
        res->headers["Cache-Control"] = "no-store";

        // Disabled means zero tokenization/planning work and no cache oracle.
        if (!params.cache_plan_preflight) {
            res->error(format_error_response(
                "This server does not support cache-plan preflight. Start it with `--cache-plan-preflight`",
                ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (!server_cache_plan_preflight_exposure_allowed(
                params.hostname, params.api_keys.size())) {
            res->error(format_error_response(
                "Cache-plan preflight requires a trusted-local single-principal server",
                ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        try {
            const json data = json::parse(req.body);
            if (!data.is_object()) {
                throw std::runtime_error("cache-plan preflight body must be an object");
            }
            for (const auto & field : data.items()) {
                if (!server_cache_plan_preflight_request_field_allowed(
                        field.key())) {
                    throw std::runtime_error(
                        "unsupported cache-plan preflight field: " +
                        field.key());
                }
            }
            if (!data.contains("prompt")) {
                throw std::runtime_error(
                    "cache-plan preflight requires prompt");
            }
            if (!req.files.empty()) {
                throw std::runtime_error(
                    "cache-plan preflight does not accept multipart files");
            }

            const auto & prompt = data.at("prompt");
            if (prompt.is_array() &&
                !json_is_array_and_contains_numbers(prompt) &&
                prompt.size() != 1) {
                throw std::runtime_error(
                    "cache-plan preflight accepts exactly one prompt");
            }

            auto inputs = tokenize_input_prompts(
                ctx_server.vocab, ctx_server.mctx, prompt,
                true, true);
            if (inputs.size() != 1) {
                throw std::runtime_error(
                    "cache-plan preflight accepts exactly one prompt");
            }

            server_task task(SERVER_TASK_TYPE_CACHE_PLAN_PREFLIGHT);
            task.id = res->rd.get_new_id();
            task.tokens = std::move(inputs.front());
            task.id_slot = data.contains("id_slot")
                ? data.at("id_slot").get<int>() : -1;
            task.params.cache_prompt = data.contains("cache_prompt")
                ? data.at("cache_prompt").get<bool>()
                : params.cache_prompt;
            if (data.contains("lora")) {
                if (!data.at("lora").is_array()) {
                    throw std::runtime_error(
                        "cache-plan preflight lora must use the native array form");
                }
                task.params.lora = parse_lora_request(data.at("lora"));
            }
            if (data.contains("message_delimiters") &&
                !data.at("message_delimiters").is_array()) {
                throw std::runtime_error(
                    "cache-plan preflight message_delimiters must be an array");
            }
            auto delimiters = common_chat_msg_delimiters_parse(
                data.contains("message_delimiters")
                    ? data.at("message_delimiters") : json::array());
            delimiters.tokenize(ctx_server.vocab);
            task.params.message_spans =
                task.tokens.find_message_spans(delimiters);

            // The scheduler-thread assertion lives inside cache_plan_preflight;
            // the HTTP worker has no direct call door to the planning kernel.
            res->rd.post_task(std::move(task));
        } catch (const std::exception & e) {
            res->error(format_error_response(
                e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * preview = dynamic_cast<
            server_task_result_cache_plan_preflight *>(result.get());
        GGML_ASSERT(preview != nullptr);
        res->ok(preview->to_json());
        return res;
    };

    this->post_cache_control = [this](const server_http_req & req) {
        auto res = create_response();
        res->headers["Cache-Control"] = "no-store";

        server_cache_control_operation operation =
            server_cache_control_operation::_count;
        GGML_ASSERT(params.cache_control_api &&
                    server_cache_control_operation_for_path(req.path, operation));

        auto request = std::make_shared<server_cache_control_request>();
        server_task task = server_task::cache_control_task(operation);
        task.id = res->rd.get_new_id();

        const auto typed_status = [&](server_cache_control_status status,
                                      int http_status = 200) {
            server_cache_control_result result;
            result.status = status;
            const auto body = server_cache_control_json(operation, result);
            if (http_status == 400) {
                res->status = 400;
                res->data = safe_json_to_str(body);
            } else {
                res->ok(body);
            }
        };
        const auto parse_selector = [&](const json & value,
                                        server_cache_control_selector & selector,
                                        server_task::cache_control_semantic_selector & semantic) {
            if (!value.is_object() || !value.contains("kind") ||
                !value.at("kind").is_string()) {
                return false;
            }
            const std::string kind = value.at("kind").get<std::string>();
            if (!server_cache_control_parse_subject_kind(kind, selector.kind)) {
                return false;
            }
            // Authorize the complete shape before prompt tokenization. No
            // rejected field may consume tokenizer/model work on the worker.
            for (const auto & field : value.items()) {
                if (!server_cache_control_selector_field_allowed(
                        selector.kind, field.key())) {
                    return false;
                }
            }
            if (selector.kind == server_cache_control_subject_kind::live_prefix) {
                if (!value.contains("slot_id") ||
                    !value.at("slot_id").is_number_integer()) {
                    return false;
                }
                const int64_t slot_id = value.at("slot_id").get<int64_t>();
                if (slot_id < std::numeric_limits<int32_t>::min() ||
                    slot_id > std::numeric_limits<int32_t>::max()) {
                    return false;
                }
                semantic.slot_id = int32_t(slot_id);
            } else if (selector.kind == server_cache_control_subject_kind::host_snapshot) {
                if (!value.contains("prompt")) {
                    return false;
                }
                auto inputs = tokenize_input_prompts(
                    ctx_server.vocab, ctx_server.mctx,
                    value.at("prompt"), true, true);
                if (inputs.size() != 1) {
                    return false;
                }
                if (value.contains("lora")) {
                    if (!value.at("lora").is_array()) {
                        return false;
                    }
                    semantic.lora = parse_lora_request(value.at("lora"));
                }
                if (value.contains("message_delimiters") &&
                    !value.at("message_delimiters").is_array()) {
                    return false;
                }
                semantic.tokens = std::make_shared<const server_tokens>(
                    std::move(inputs.front()));
            } else if (selector.kind == server_cache_control_subject_kind::vbr_reference) {
                if (!value.contains("reference") ||
                    !value.at("reference").is_string()) {
                    return false;
                }
                selector.reference = value.at("reference").get<std::string>();
                selector.tenant_key = server_cache_capture_tenant_key(req);
            }
            return true;
        };

        try {
            if (!req.files.empty()) {
                throw std::runtime_error("cache-control routes do not accept multipart files");
            }
            const json body = json::parse(req.body);
            const auto prepared = server_cache_control_prepare_request(
                operation, body, *request);
            if (prepared != server_cache_control_status::ok) {
                typed_status(prepared, 400);
                return res;
            }
            bool valid = false;
            switch (operation) {
                case server_cache_control_operation::holder_create:
                case server_cache_control_operation::holder_reattach:
                case server_cache_control_operation::holder_close:
                case server_cache_control_operation::family_register:
                case server_cache_control_operation::family_bind:
                    valid = true;
                    break;
                case server_cache_control_operation::lease_acquire: {
                    valid = true;
                    if (valid) {
                        valid = parse_selector(
                            body.at("subject"), request->subject,
                            task.cache_control_subject);
                    }
                    if (valid && body.contains("fallback")) {
                        valid = parse_selector(
                            body.at("fallback"), request->fallback,
                            task.cache_control_fallback);
                    } else if (valid && request->requested_class ==
                            server_cache_lease_class::hard) {
                        valid = false;
                    } else if (valid) {
                        request->fallback.kind =
                            server_cache_control_subject_kind::_count;
                    }
                    if (valid) {
                        valid = !request->allow_soft_fallback ||
                            request->requested_class ==
                                server_cache_lease_class::hard;
                    }
                } break;
                case server_cache_control_operation::lease_inspect:
                case server_cache_control_operation::lease_release:
                    valid = true;
                    break;
                case server_cache_control_operation::lease_renew:
                    valid = true;
                    request->fallback.kind =
                        server_cache_control_subject_kind::_count;
                    if (valid && body.contains("fallback")) {
                        valid = parse_selector(
                            body.at("fallback"), request->fallback,
                            task.cache_control_fallback);
                    }
                    break;
                case server_cache_control_operation::events:
                    valid = true;
                    break;
                case server_cache_control_operation::_count:
                    valid = false;
                    break;
            }
            if (!valid) {
                typed_status(server_cache_control_status::invalid_request, 400);
                return res;
            }
            task.cache_control = std::move(request);
            res->rd.post_task(std::move(task));
        } catch (const std::exception &) {
            typed_status(server_cache_control_status::invalid_request, 400);
            return res;
        }

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * controlled = dynamic_cast<
            server_task_result_cache_control *>(result.get());
        GGML_ASSERT(controlled != nullptr);
        res->ok(controlled->to_json());
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_ftype",                 meta->model_ftype },
            { "model_path",                  meta->model_path },
            { "vbr",                         server_vbr_meta_json(meta.get()) },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"video",  meta->has_inp_video},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "ui",                          params.ui },
            { "ui_settings",                 meta->json_ui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

// the /props and /models "vbr" object — built in ONE place. selected_kld/schedule are
// static-ladder measurements: null under the dynamic controller (like realized_bpv), not 0.
static json server_vbr_meta_json(const server_context_meta * meta) {
    return json {
        {"enabled",           meta->vbr_enabled},
        {"dynamic",           meta->vbr_dynamic},
        {"type_k",            meta->vbr_type_k},
        {"type_v",            meta->vbr_type_v},
        {"floor_bpv",         meta->vbr_min_bits},
        {"capacity_floor_bpv", meta->vbr_capacity_bits},
        // realized bits/value is a fixed number only for static schedules; under the
        // dynamic runtime controller it varies with occupancy — null, not a fiction
        {"realized_bpv",      meta->vbr_dynamic ? json() : json(meta->vbr_capacity_bits)},
        {"selected_family",   meta->vbr_selected_family},
        {"selected_policy",   meta->vbr_selected_policy},
        {"selected_bpv",      meta->vbr_dynamic ? json() : json(meta->vbr_selected_bpv)},
        {"selected_kld",      meta->vbr_dynamic ? json() : json(meta->vbr_selected_kld)},
        {"selected_schedule", meta->vbr_dynamic ? json() : json(meta->vbr_selected_schedule)},
        {"vram_budget_bytes", meta->vbr_vram_budget_bytes},
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"vbr",         server_vbr_meta_json(meta.get())},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
            {"ftype",       meta->model_ftype},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_capture(
        const server_http_req & req,
        int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_CACHE_CAPTURE);
        task.id = rd.get_new_id();
        task.cache_capture.id_slot = id_slot;
        task.cache_capture.tenant_key =
            server_cache_capture_tenant_key(req);
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }
    auto * captured =
        dynamic_cast<server_task_result_cache_capture *>(result.get());
    GGML_ASSERT(captured != nullptr);
    res->ok(captured->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_import(
        const server_http_req & req,
        int id_slot) {
    auto res = create_response();
    std::string reference;
    try {
        const json request_data = json::parse(req.body);
        reference = request_data.at("reference").get<std::string>();
    } catch (const std::exception &) {
        res->error(format_error_response(
            "Import requires a string reference",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_CACHE_IMPORT);
        task.id = rd.get_new_id();
        task.cache_import.id_slot = id_slot;
        task.cache_import.tenant_key =
            server_cache_capture_tenant_key(req);
        task.cache_import.reference = std::move(reference);
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }
    auto * imported =
        dynamic_cast<server_task_result_cache_import *>(result.get());
    GGML_ASSERT(imported != nullptr);
    res->ok(imported->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}
