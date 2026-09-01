#include "fit.h"

#include "common.h"
#include "log.h"

#include "../ggml/src/ggml-backend-moe-cache.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cinttypes>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static ggml_type common_vbr_floor_price_tier(double floor_bpv); // defined near the bottom

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

uint32_t common_fit_extra_context_size(
        uint32_t target_n_ctx,
        uint32_t target_n_streams,
        bool follows_target_per_sequence,
        uint32_t fixed_n_ctx) {
    if (fixed_n_ctx > 0) {
        return fixed_n_ctx;
    }
    if (!follows_target_per_sequence) {
        return target_n_ctx;
    }

    const uint32_t n_streams = std::max<uint32_t>(1, target_n_streams);
    const uint32_t n_ctx     = GGML_PAD(target_n_ctx, 256);
    return GGML_PAD(n_ctx / n_streams, 256);
}

using common_fit_extra_mapped_memory = std::vector<llama_memory_breakdown_data>;

struct common_fit_extra_cache_state {
    common_fit_extra_mapped_memory mapped;
    uint32_t n_ctx = 0;
    bool unavailable = false;
};

struct common_fit_extra_cache_request {
    uint32_t n_ctx;
    bool refresh_always;
    bool optional;
};

struct common_fit_extra_measurement {
    bool available;
    common_fit_extra_mapped_memory mapped;
};

static std::vector<const common_fit_extra_model *> common_fit_extra_model_chain(
        const common_fit_extra_model * extra) {
    std::vector<const common_fit_extra_model *> result;
    for (const common_fit_extra_model * current = extra; current != nullptr; current = current->next) {
        result.push_back(current);
    }
    return result;
}

static void common_fit_add_breakdown(
        llama_memory_breakdown_data & dst,
        const llama_memory_breakdown_data & src) {
    dst.model               += src.model;
    dst.context             += src.context;
    dst.compute             += src.compute;
    dst.context_fixed       += src.context_fixed;
    dst.context_vbr_managed += src.context_vbr_managed;
}

template <typename Measure>
static bool common_fit_extra_cache_update(
        std::vector<common_fit_extra_cache_state> & states,
        const std::vector<common_fit_extra_cache_request> & requests,
        size_t n_destinations,
        Measure && measure,
        common_fit_extra_mapped_memory & aggregate) {
    GGML_ASSERT(states.size() == requests.size());

    for (size_t i = 0; i < requests.size(); ++i) {
        auto & state = states[i];
        const auto & request = requests[i];
        const bool refresh = !state.unavailable &&
            (state.mapped.empty() || state.n_ctx != request.n_ctx || request.refresh_always);
        if (!refresh) {
            continue;
        }

        common_fit_extra_measurement measured = measure(i, request.n_ctx);
        if (!measured.available) {
            if (!request.optional) {
                return false;
            }
            state.mapped.clear();
            state.n_ctx = request.n_ctx;
            state.unavailable = true;
            continue;
        }
        if (measured.mapped.size() != n_destinations) {
            return false;
        }

        state.mapped = std::move(measured.mapped);
        state.n_ctx = request.n_ctx;
    }

    aggregate.assign(n_destinations, {});
    for (const auto & state : states) {
        if (state.unavailable) {
            continue;
        }
        if (state.mapped.size() != n_destinations) {
            return false;
        }
        for (size_t id = 0; id < n_destinations; ++id) {
            common_fit_add_breakdown(aggregate[id], state.mapped[id]);
        }
    }
    return true;
}

common_fit_extra_cache_probe_result common_fit_extra_cache_probe(
        const common_fit_extra_model * extra,
        const std::vector<uint32_t> & target_n_ctx_steps,
        uint32_t target_n_streams,
        const std::vector<common_fit_extra_cache_probe_source> & sources) {
    common_fit_extra_cache_probe_result result = {
        /*.success =*/ true,
        /*.aggregate_bytes_by_device =*/ {},
        /*.measurement_counts =*/ std::vector<size_t>(sources.size(), 0),
    };
    const auto models = common_fit_extra_model_chain(extra);
    if (models.size() != sources.size()) {
        result.success = false;
        return result;
    }
    if (models.empty()) {
        return result;
    }

    const size_t n_destinations = sources.front().base_bytes_by_device.size();
    std::vector<common_fit_extra_cache_state> states(sources.size());
    common_fit_extra_mapped_memory aggregate;

    for (uint32_t target_n_ctx : target_n_ctx_steps) {
        std::vector<common_fit_extra_cache_request> requests;
        requests.reserve(sources.size());
        for (size_t i = 0; i < sources.size(); ++i) {
            const auto & source = sources[i];
            const auto * model = models[i];
            if (source.base_bytes_by_device.size() != n_destinations) {
                result.success = false;
                return result;
            }
            requests.push_back({
                common_fit_extra_context_size(
                    target_n_ctx, target_n_streams,
                    model->follows_target_per_sequence, model->fixed_n_ctx),
                model->shares_model,
                model->optional_if_no_mtp,
            });
        }

        result.success = common_fit_extra_cache_update(
            states, requests, n_destinations,
            [&](size_t i, uint32_t n_ctx) {
                result.measurement_counts[i]++;
                if (!sources[i].available) {
                    return common_fit_extra_measurement { false, {} };
                }
                common_fit_extra_mapped_memory mapped(n_destinations);
                for (size_t id = 0; id < n_destinations; ++id) {
                    mapped[id].context = sources[i].base_bytes_by_device[id] + n_ctx;
                }
                return common_fit_extra_measurement { true, std::move(mapped) };
            }, aggregate);
        if (!result.success) {
            return result;
        }
    }

    result.aggregate_bytes_by_device.resize(aggregate.size());
    for (size_t id = 0; id < aggregate.size(); ++id) {
        result.aggregate_bytes_by_device[id] = aggregate[id].total();
    }
    return result;
}

// floor-true KV pricing inputs/outputs for dynamic VBR (see common_params_fit_impl): the dry
// context is created with PRICE-tier types (the movable swap in common_fit_params), while the
// runtime floor clamp lands on a discrete tier MIX along the degrade order — capacity math must
// use the mix cost, queried from the dry context via llama_vbr_floor_bits_per_token.
struct common_vbr_fit_costs {
    ggml_type entry_k = GGML_TYPE_COUNT; // in: true entry types (cparams' are price-swapped)
    ggml_type entry_v = GGML_TYPE_COUNT;
    double    bits_pt_floor = 0.0;       // out: per-token KV bits at the achievable clamped mix
    double    bits_pt_price = 0.0;       // out: per-token KV bits at cparams' (price) types
    // out: model requires matching K/V cache types (MLA-family). Such caches run a static cap
    // under dynamic VBR (no per-tier degrade), so the dry-load KV bytes are already the truth
    // and the floor/price capacity scaling must not be applied.
    bool      types_coupled = false;
    // Per-token bytes of the flash-attention f16 dequant scratch at settled deep fill: a
    // context-linear consumer OUTSIDE the KV budget (it draws from the fit margin). Charged in
    // the total-VRAM wall constraint only, never in the budget-capacity solves.
    double    scratch_bytes_pt = 0.0;    // out
};

double common_vbr_fit_kv_scale(double floor_bits_per_token, double price_bits_per_token, bool types_coupled) {
    if (types_coupled || floor_bits_per_token <= 0.0 || price_bits_per_token <= 0.0) {
        return 1.0;
    }
    return std::max(1.0, floor_bits_per_token / price_bits_per_token);
}

const char * common_moe_cache_tensor_override_pattern() {
    return "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
}

struct common_moe_cache_fit_pool {
    ggml_type type = GGML_TYPE_COUNT;
    size_t expert_size = 0;
    size_t pool_bytes = 0;
    size_t tensor_bytes = 0;
    size_t scratch_bytes = 0;
};

common_moe_cache_fit_result common_moe_cache_plan_fit(
        const std::vector<common_moe_cache_fit_device_input> & device_inputs,
        const std::vector<common_moe_cache_fit_shape_input> & shapes,
        size_t reserve_bytes,
        size_t budget_bytes,
        int min_devices,
        size_t minimum_slab_bytes) {
    common_moe_cache_fit_result result;

    for (const common_moe_cache_fit_device_input & input : device_inputs) {
        if (input.physical_device < 0 || input.free_bytes < 0 || input.used_bytes > INT64_MAX) {
            result.reason = "device memory accounting overflowed";
            return result;
        }

        size_t device_index = result.devices.size();
        for (size_t candidate = 0; candidate < result.devices.size(); candidate++) {
            if (result.devices[candidate].physical_device == input.physical_device) {
                device_index = candidate;
                break;
            }
        }
        if (device_index == result.devices.size()) {
            common_moe_cache_fit_device device;
            device.physical_device = input.physical_device;
            device.compute_capability = input.compute_capability;
            device.free_bytes = input.free_bytes;
            device.reserve_bytes = input.reserve_bytes;
            result.devices.push_back(device);
        }

        common_moe_cache_fit_device & device = result.devices[device_index];
        device.free_bytes = std::min(device.free_bytes, input.free_bytes);
        device.compute_capability = std::min(device.compute_capability, input.compute_capability);
        device.reserve_bytes = std::max(device.reserve_bytes, input.reserve_bytes);
        if ((int64_t)input.used_bytes > INT64_MAX - device.used_bytes) {
            result.reason = "device memory accounting overflowed";
            return result;
        }
        device.used_bytes += (int64_t)input.used_bytes;
    }
    if (result.devices.empty()) {
        result.reason = "no selected device satisfies the cache hardware policy";
        return result;
    }

    for (common_moe_cache_fit_device & device : result.devices) {
        const int64_t projected_free = device.free_bytes - device.used_bytes;
        const size_t effective_reserve = device.reserve_bytes > 0
            ? device.reserve_bytes : reserve_bytes;
        if (projected_free <= 0 || (uint64_t)projected_free <= effective_reserve) {
            continue;
        }
        device.cache_bytes = (size_t)projected_free - effective_reserve;
        if (budget_bytes > 0) {
            device.cache_bytes = std::min(device.cache_bytes, budget_bytes);
        }
    }

    std::vector<common_moe_cache_fit_pool> pools;
    for (const common_moe_cache_fit_shape_input & shape : shapes) {
        if (shape.tensor_bytes == 0 || shape.tensor_bytes > SIZE_MAX - result.expert_bytes) {
            result.reason = "the routed expert tensor inventory overflowed";
            return result;
        }
        result.expert_bytes += shape.tensor_bytes;
        if (!shape.cacheable) {
            continue;
        }
        bool found = false;
        for (common_moe_cache_fit_pool & pool : pools) {
            if (pool.type == shape.type && pool.expert_size == shape.expert_size) {
                pool.pool_bytes = std::max(pool.pool_bytes, shape.pool_bytes);
                pool.tensor_bytes += shape.tensor_bytes;
                pool.scratch_bytes = std::max(pool.scratch_bytes, shape.scratch_bytes);
                found = true;
                break;
            }
        }
        if (!found) {
            pools.push_back({shape.type, shape.expert_size, shape.pool_bytes,
                    shape.tensor_bytes, shape.scratch_bytes});
        }
    }

    size_t scratch_bytes = 0;
    size_t supported_bytes = 0;
    for (const common_moe_cache_fit_pool & pool : pools) {
        if (pool.tensor_bytes < pool.pool_bytes) {
            continue;
        }
        supported_bytes += pool.tensor_bytes;
        scratch_bytes = std::max(scratch_bytes, pool.scratch_bytes);
    }
    if (supported_bytes == 0) {
        result.reason = "no routed expert shape is cacheable";
        return result;
    }
    if (supported_bytes != result.expert_bytes) {
        result.reason = "some routed expert weights would remain permanently uncached";
        return result;
    }

    size_t minimum_pool_bytes = 0;
    for (const common_moe_cache_fit_pool & pool : pools) {
        if (pool.tensor_bytes < pool.pool_bytes) {
            continue;
        }
        if (pool.pool_bytes > SIZE_MAX - minimum_pool_bytes) {
            result.reason = "the minimum cache pool inventory overflowed";
            return result;
        }
        minimum_pool_bytes += pool.pool_bytes;
    }
    minimum_pool_bytes = std::max(minimum_pool_bytes, minimum_slab_bytes);
    if (minimum_pool_bytes > SIZE_MAX - scratch_bytes) {
        result.reason = "the minimum cache pool inventory overflowed";
        return result;
    }
    result.minimum_device_bytes = scratch_bytes + minimum_pool_bytes;

    int useful_devices = 0;
    for (const common_moe_cache_fit_device & device : result.devices) {
        if (device.cache_bytes < result.minimum_device_bytes) {
            continue;
        }
        useful_devices++;
        if (device.cache_bytes > SIZE_MAX - result.cache_bytes) {
            result.cache_bytes = SIZE_MAX;
        } else {
            result.cache_bytes += device.cache_bytes;
        }
    }
    if (useful_devices < min_devices) {
        result.reason = "too few devices can hold the minimum expert pools";
        return result;
    }

    result.feasible = true;
    result.reason = "cache pools are feasible";
    return result;
}

static common_moe_cache_fit_result common_moe_cache_evaluate_fit(
        const common_moe_cache_params * params,
        const std::vector<llama_moe_tensor_info> & tensors,
        const std::vector<ggml_backend_dev_t> & devices,
        const std::vector<llama_device_memory_data> & memory,
        const std::vector<int64_t> & margins) {
    common_moe_cache_fit_result result;
    if (!params || params->mode == COMMON_MOE_CACHE_MODE_OFF) {
        result.reason = "disabled";
        return result;
    }
    if (!ggml_moe_cache.query_config || !ggml_moe_cache.query_device ||
        !ggml_moe_cache.query_shape) {
        result.reason = "no cache provider is loaded";
        return result;
    }

    int automatic = -1;
    if (params->mode_explicit) {
        automatic = params->mode == COMMON_MOE_CACHE_MODE_AUTO ? 1 : 0;
    }
    ggml_moe_cache_config config = {};
    if (!ggml_moe_cache.query_config(automatic, params->budget_mib, &config)) {
        result.reason = "the cache provider is disabled";
        return result;
    }
    if (tensors.empty()) {
        result.reason = "the model has no routed expert weight tensors";
        return result;
    }
    if (memory.size() != devices.size() + 1 || margins.size() != devices.size()) {
        result.reason = "the fitted device inventory changed";
        return result;
    }

    std::vector<common_moe_cache_fit_device_input> device_inputs;
    size_t min_expert_bytes = 0;
    for (size_t index = 0; index < devices.size(); index++) {
        ggml_moe_cache_device_caps caps = {};
        if (!ggml_moe_cache.query_device(devices[index], &config, &caps)) {
            continue;
        }
        if (margins[index] < 0 || memory[index].free < margins[index]) {
            result.reason = "the fitted device margin exceeds free memory";
            return result;
        }
        device_inputs.push_back({caps.physical_device, caps.compute_capability,
                memory[index].free - margins[index], memory[index].mb.total(),
                caps.recommended_reserve_bytes});
        min_expert_bytes = std::max(min_expert_bytes, caps.min_expert_bytes);
    }

    std::vector<common_moe_cache_fit_shape_input> shape_inputs;
    shape_inputs.reserve(tensors.size());
    for (const llama_moe_tensor_info & tensor : tensors) {
        if (tensor.n_expert <= 0 || tensor.expert_size == 0 ||
            (uint64_t)tensor.n_expert > SIZE_MAX / tensor.expert_size) {
            result.reason = "the model has an invalid routed expert tensor size";
            return result;
        }
        const size_t tensor_bytes = (size_t)tensor.n_expert * tensor.expert_size;
        ggml_moe_cache_shape_caps caps = {};
        const bool cacheable = tensor.expert_size >= min_expert_bytes &&
            ggml_moe_cache.query_shape(tensor.type, tensor.n_input, tensor.n_output,
                    tensor.n_expert, tensor.expert_size, &caps);
        shape_inputs.push_back({tensor.type, tensor.expert_size, tensor_bytes,
                caps.scratch_bytes, caps.pool_bytes, cacheable});
    }

    return common_moe_cache_plan_fit(
            device_inputs, shape_inputs, config.reserve_bytes, config.budget_bytes,
            config.min_devices, config.minimum_slab_bytes);
}

struct common_fit_logger_guard {
    ggml_log_callback original_callback;
    void * original_user_data;
    ggml_log_level min_level;

    explicit common_fit_logger_guard(ggml_log_level min_level) : min_level(min_level) {
        llama_log_get(&original_callback, &original_user_data);
        llama_log_set(callback, this);
    }

    ~common_fit_logger_guard() {
        llama_log_set(original_callback, original_user_data);
    }

    static void callback(ggml_log_level level, const char * text, void * user_data) {
        const common_fit_logger_guard * guard = (const common_fit_logger_guard *) user_data;
        const ggml_log_level level_eff = level >= guard->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        guard->original_callback(level_eff, text, guard->original_user_data);
    }
};

static std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        common_vbr_fit_costs * vbr_costs = nullptr,
        bool plan_hint = false,
        std::vector<llama_moe_tensor_info> * moe_tensors = nullptr,
        llama_context * ctx_parent = nullptr,
        bool share_parent_tensors = false,
        bool optional_if_no_mtp = false) {
    common_fit_logger_guard logger_guard(log_level);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model_ptr model(llama_model_load_from_file(path_model, mparams_copy));
    if (model == nullptr) {
        throw std::runtime_error("failed to load model");
    }
    if (optional_if_no_mtp && !llama_model_has_mtp(model.get())) {
        return {};
    }

    llama_context_params cparams_copy = *cparams;
    if (ctx_parent != nullptr) {
        cparams_copy.ctx_other = ctx_parent;
        if (share_parent_tensors) {
            // DFlash/DSpark sidecars omit the target embedding and output tensors.
            // Mirror the real-load share-or-gather step before reserving their graph;
            // a tensor-sharded parent's Meta tensors cannot execute in the ordinary
            // (layer-split or pinned) draft scheduler.
            llama_model_share_tensors(model.get(), llama_get_model(ctx_parent));
        }
    }

    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams_copy));
    if (ctx == nullptr) {
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model.get());
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx.get());

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model         += mb.model;
            ret.back().mb.context       += mb.context;
            ret.back().mb.compute       += mb.compute;
            ret.back().mb.context_fixed += mb.context_fixed;
            ret.back().mb.context_vbr_managed += mb.context_vbr_managed;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model.get(), i)) {
                ret[i].mb.model         += mb.model;
                ret[i].mb.context       += mb.context;
                ret[i].mb.compute       += mb.compute;
                ret[i].mb.context_fixed += mb.context_fixed;
                ret[i].mb.context_vbr_managed += mb.context_vbr_managed;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model.get(), i);

        size_t free  = 0;
        size_t total = 0;
        if (ggml_backend_dev_is_meta(dev)) {
            // SPLIT_MODE_TENSOR: the meta device would report the SUM of its simple devices'
            // memory, but the per-layer shard rotation (llama_meta_device_get_split_state)
            // spreads bytes evenly across them, so the usable aggregate is n_devs x the
            // tightest device (a co-tenant on one GPU binds all of them)
            const size_t n_simple = ggml_backend_meta_dev_n_devs(dev);
            size_t min_free  = std::numeric_limits<size_t>::max();
            size_t min_total = std::numeric_limits<size_t>::max();
            for (size_t j = 0; j < n_simple; j++) {
                size_t free_j, total_j;
                ggml_backend_dev_memory(ggml_backend_meta_dev_simple_dev(dev, j), &free_j, &total_j);
                min_free  = std::min(min_free,  free_j);
                min_total = std::min(min_total, total_j);
            }
            free  = n_simple * min_free;
            total = n_simple * min_total;
        } else {
            ggml_backend_dev_memory(dev, &free, &total);
        }

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model.get()); i++) {
        devs.push_back(llama_model_get_device(model.get(), i));
    }

    // co-tenancy plan hint: this process intends to allocate model+context+compute on each
    // device, published as a held demand's joint cross-device estimate. plan_hint defaults
    // FALSE so a forgotten tag fails toward the designed no-hint/est_partial fallback —
    // only the requested-full-configuration measurement is tagged (a hint that lies is
    // worse than none: it carries a one-revision fuse). When the fit later shrinks the
    // config the standing hint overstates; the satisfied phase caps the harm. Meta devices
    // (SPLIT_MODE_TENSOR) split evenly, matching the shard rotation's balanced layout.
    if (plan_hint) {
        auto hint = [](ggml_backend_dev_t d, uint64_t bytes) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(d, &props);
            if (props.device_id != nullptr) {
                llama_vram_plan_hint(props.device_id, bytes);
            }
        };
        for (size_t i = 0; i < nd; i++) {
            ggml_backend_dev_t dev = llama_model_get_device(model.get(), i);
            const uint64_t planned = ret[i].mb.model + ret[i].mb.context + ret[i].mb.compute;
            if (ggml_backend_dev_is_meta(dev)) {
                const size_t n_simple = ggml_backend_meta_dev_n_devs(dev);
                for (size_t j = 0; j < n_simple; j++) {
                    hint(ggml_backend_meta_dev_simple_dev(dev, j), planned / n_simple);
                }
            } else {
                hint(dev, planned);
            }
        }
    }

    hp_ngl         = llama_model_n_layer(model.get());
    if (mparams->load_mtp) {
        hp_ngl    += llama_model_n_layer_nextn(model.get());
    }
    hp_n_ctx_train = llama_model_n_ctx_train(model.get());
    hp_n_expert    = llama_model_n_expert(model.get());

    if (moe_tensors) {
        const size_t count = llama_model_get_moe_tensor_info(model.get(), nullptr, 0);
        moe_tensors->resize(count);
        const size_t written = llama_model_get_moe_tensor_info(model.get(), moe_tensors->data(), moe_tensors->size());
        GGML_ASSERT(written == count);
    }

    common_memory_breakdown_print(ctx.get());

    if (vbr_costs != nullptr) {
        vbr_costs->bits_pt_floor = llama_vbr_floor_bits_per_token(ctx.get(), vbr_costs->entry_k, vbr_costs->entry_v, cparams->vbr_min_bits);
        if (vbr_costs->bits_pt_floor < 0.0) {
            throw std::runtime_error("VBR aggregate floor exceeds the selected mixed K/V entry layout");
        }
        vbr_costs->bits_pt_price = llama_vbr_entry_bits_per_token(ctx.get(), cparams->type_k, cparams->type_v);
        vbr_costs->scratch_bytes_pt = llama_vbr_scratch_bytes_per_token(ctx.get(), vbr_costs->entry_k, vbr_costs->entry_v, cparams->vbr_min_bits);
        vbr_costs->types_coupled = llama_model_kv_cache_types_coupled(model.get());
    }

    return ret;
}

common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

common_device_memory_data_vec common_get_device_memory_data_with_parent(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        const char * path_parent,
        const llama_model_params * mparams_parent,
        const llama_context_params * cparams_parent,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        bool share_parent_tensors) {
    common_fit_logger_guard logger_guard(log_level);

    llama_model_params mparams_parent_copy = *mparams_parent;
    mparams_parent_copy.no_alloc  = true;
    mparams_parent_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model_ptr model_parent(llama_model_load_from_file(path_parent, mparams_parent_copy));
    if (model_parent == nullptr) {
        throw std::runtime_error("failed to load parent model");
    }

    llama_context_ptr ctx_parent(llama_init_from_model(model_parent.get(), *cparams_parent));
    if (ctx_parent == nullptr) {
        throw std::runtime_error("failed to create parent llama_context");
    }

    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert,
            log_level, nullptr, false, nullptr, ctx_parent.get(), share_parent_tensors);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

bool common_model_uses_recurrent_memory(
        const char * path_model,
        const llama_model_params * mparams,
        ggml_log_level log_level) {
    common_fit_logger_guard logger_guard(log_level);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model_ptr model(llama_model_load_from_file(path_model, mparams_copy));
    if (model == nullptr) {
        throw std::runtime_error("failed to load model");
    }
    return llama_model_is_recurrent(model.get()) || llama_model_is_hybrid(model.get());
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        common_moe_cache_params * moe_cache, size_t * margins_s, uint32_t n_ctx_min,
        const common_fit_extra_model * extra, enum ggml_log_level log_level,
        ggml_type type_k_entry, ggml_type type_v_entry) {
    // SPLIT_MODE_TENSOR runs through the single-device paths below: the model exposes exactly one
    // meta device whose memory report is the balanced-equivalent aggregate of the real GPUs (see
    // common_get_device_memory_data_impl) and whose margin is the sum of the per-device targets.
    // Only the step-3+ layer redistribution is unavailable (guarded before step 3).
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert
    std::vector<llama_moe_tensor_info> moe_tensors;

    if (moe_cache) {
        moe_cache->fit_selected = false;
    }

    // with non-unified kv, we need to take into account n_streams
    // for example, if memory can hold more than model's trained context size, we must extend the n_ctx to hold enough n_streams
    const uint32_t n_streams  = cparams->kv_unified ? 1 : std::max<uint32_t>(1, cparams->n_seq_max);
    const bool     n_ctx_auto = cparams->n_ctx == 0;

    const auto extra_models = common_fit_extra_model_chain(extra);
    std::vector<common_fit_extra_cache_state> extra_caches(extra_models.size());
    dmds_t dmds_extra; // latest aggregate, laid out on the target devices

    // the extra model competes for the same memory as the main model, add it to every measurement
    // its memory is measured again whenever the context it follows changes. A shared-model MTP
    // context also follows the target placement, so candidate placements must not reuse the
    // initial extra-model measurement.
    auto add_extra_memory = [&](dmds_t & dmds, const llama_model_params * target_mparams) {
        if (extra == nullptr) {
            return;
        }

        std::vector<common_fit_extra_cache_request> requests;
        requests.reserve(extra_models.size());
        for (const common_fit_extra_model * current : extra_models) {
            const uint32_t n_ctx_current = common_fit_extra_context_size(
                cparams->n_ctx, n_streams,
                current->follows_target_per_sequence, current->fixed_n_ctx);
            current->cparams->n_ctx = n_ctx_current;
            requests.push_back({
                n_ctx_current,
                current->shares_model,
                current->optional_if_no_mtp,
            });
        }

        common_fit_extra_mapped_memory aggregate;
        const bool extra_ok = common_fit_extra_cache_update(
            extra_caches, requests, devs.size() + 1,
            [&](size_t iextra, uint32_t) {
                const common_fit_extra_model * current = extra_models[iextra];
                std::vector<ggml_backend_dev_t> devs_extra;
                uint32_t ngl_extra = 0;
                uint32_t nct_extra = 0;
                uint32_t nex_extra = 0;

                llama_model_params mparams_extra = *current->mparams;
                if (current->shares_model) {
                    GGML_ASSERT(target_mparams != nullptr);
                    const bool load_mtp = mparams_extra.load_mtp;
                    mparams_extra = *target_mparams;
                    // The target candidate owns placement, but the extra probe still has to load
                    // the native MTP tensors selected by the speculative configuration.
                    mparams_extra.load_mtp = load_mtp;
                }

                LOG_TRC("%s: getting device memory data for an extra model at a context size of %" PRIu32 ":\n",
                    __func__, current->cparams->n_ctx);

                dmds_t measured;
                try {
                    measured = common_get_device_memory_data_impl(
                        current->path_model, &mparams_extra, current->cparams,
                        devs_extra, ngl_extra, nct_extra, nex_extra, log_level,
                        /* vbr_costs = */ nullptr,
                        /* plan_hint = */ false,
                        /* moe_tensors = */ nullptr,
                        /* ctx_parent = */ nullptr,
                        /* share_parent_tensors = */ false,
                        current->optional_if_no_mtp);
                } catch (const std::runtime_error & e) {
                    throw common_params_fit_exception(string_format(
                        "failed to measure a required speculative model/context: %s", e.what()));
                }

                if (measured.empty() && current->optional_if_no_mtp) {
                    return common_fit_extra_measurement { false, {} };
                }

                if (measured.size() != devs_extra.size() + 1) {
                    throw common_params_fit_exception(
                        "required speculative model/context returned an invalid device inventory");
                }

                common_fit_extra_mapped_memory mapped(devs.size() + 1);
                llama_memory_breakdown_data mb_host = measured.back().mb;
                if (current->shares_model) {
                    mb_host.model = 0;
                }
                common_fit_add_breakdown(mapped.back(), mb_host);

                for (size_t je = 0; je < devs_extra.size(); je++) {
                    llama_memory_breakdown_data mb_extra = measured[je].mb;
                    if (current->shares_model) {
                        mb_extra.model = 0;
                    }

                    bool mapped_device = false;
                    for (size_t id = 0; id < devs.size(); id++) {
                        if (devs_extra[je] == devs[id]) {
                            common_fit_add_breakdown(mapped[id], mb_extra);
                            mapped_device = true;
                            break;
                        }
                    }

                    if (!mapped_device && mb_extra.total() != 0) {
                        throw common_params_fit_exception(string_format(
                            "required speculative model/context uses device %s outside the target fit authority",
                            ggml_backend_dev_name(devs_extra[je])));
                    }
                }
                return common_fit_extra_measurement { true, std::move(mapped) };
            }, aggregate);

        if (!extra_ok) {
            throw common_params_fit_exception(
                "required speculative model/context returned no usable memory inventory");
        }

        dmds_extra.assign(aggregate.size(), {});
        for (size_t id = 0; id < dmds.size(); id++) {
            dmds_extra[id].mb = aggregate[id];
            common_fit_add_breakdown(dmds[id].mb, aggregate[id]);
        }
    };

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    common_vbr_fit_costs vbr_costs;
    vbr_costs.entry_k = type_k_entry;
    vbr_costs.entry_v = type_v_entry;
    dmds_t dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level,
            cparams->vbr_dynamic ? &vbr_costs : nullptr, /*plan_hint=*/true, &moe_tensors);

    // saturate instead of overflowing, this also preserves the UINT32_MAX sentinel of n_ctx_min:
    const uint32_t n_ctx_max       = (uint32_t) std::min<uint64_t>(uint64_t(hp_nct)    * n_streams, UINT32_MAX);
    const uint32_t n_ctx_min_total = (uint32_t) std::min<uint64_t>(uint64_t(n_ctx_min) * n_streams, UINT32_MAX);

    // llama_context would use only hp_nct in total for n_ctx == 0, resolve the context before measuring anything else:
    if (n_ctx_auto) {
        cparams->n_ctx = n_ctx_max;
        if (n_streams > 1) {
            LOG_TRC("%s: context size unset and KV cache not unified -> using %" PRIu32 " for %" PRIu32 " sequences:\n",
                __func__, n_ctx_max, n_streams);
            dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level,
                    cparams->vbr_dynamic ? &vbr_costs : nullptr, /*plan_hint=*/true, &moe_tensors);
        }
    }
    const dmds_t dmds_target_full = dmds_full;
    add_extra_memory(dmds_full, mparams);
    const dmds_t dmds_extra_full = dmds_extra;

    const size_t nd = devs.size(); // number of devices

    // dynamic VBR: measured dry-load KV bytes are PRICE-tier-priced (largest tier <= the floor)
    // but the runtime clamp holds the aggregate at a discrete tier MIX >= the literal floor —
    // capacity estimates must scale measured KV cost up by mix/price or they over-advertise
    // (e.g. floor 6: price t4 = 4.125 bpv vs an achievable mix of ~6.04)
    const double vbr_kv_scale = common_vbr_fit_kv_scale(
        vbr_costs.bits_pt_floor, vbr_costs.bits_pt_price, vbr_costs.types_coupled);
    if (vbr_costs.types_coupled) {
        // coupled-KV model (MLA family): dynamic VBR runs a static cap there, the dry-load
        // bytes are already the achievable cost — floor/price scaling would over-advertise
        LOG_INF("%s: VBR dynamic: coupled-KV model runs a static cap — pricing KV at dry-load bytes\n",
                __func__);
    } else {
        if (vbr_kv_scale > 1.0 + 1e-6) {
            LOG_INF("%s: VBR dynamic: floor mix costs %.4g bits/token vs %.4g at the pricing tier "
                    "— scaling KV capacity math by %.3f\n",
                    __func__, vbr_costs.bits_pt_floor, vbr_costs.bits_pt_price, vbr_kv_scale);
        }
    }

    const bool sm_tensor = mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR; // nd == 1, a meta device wrapping the real GPUs

    auto log_stock_fit = [&] {
        if (moe_cache && moe_cache->mode != COMMON_MOE_CACHE_MODE_OFF &&
            !moe_tensors.empty() &&
            (!mparams->tensor_buft_overrides || !mparams->tensor_buft_overrides[0].pattern)) {
            LOG_INF("%s: MoE cache fit kept stock placement because the complete model already meets the fit targets\n", __func__);
        }
    };

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    int64_t margin_per_dev = margins_s[0];
    if (sm_tensor) {
        // the single meta device carries the SUM of the per-real-device margins, while the
        // headroom handed to the VBR runtime stays a single-device figure — the controller
        // applies it to each device's own free memory
        int64_t sum    = 0;
        margin_per_dev = 0;
        for (size_t j = 0; j < ggml_backend_meta_dev_n_devs(devs[0]); j++) {
            sum           += margins_s[j];
            margin_per_dev = std::max<int64_t>(margin_per_dev, margins_s[j]);
        }
        margins.push_back(sum);
    } else if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    auto sum_context_bytes = [&](const dmds_t & dmds) {
        int64_t result = 0;
        if (nd == 0) {
            result += dmds.back().mb.context;
        } else {
            for (size_t id = 0; id < nd; id++) {
                result += dmds[id].mb.context;
            }
        }
        return result;
    };

    // Context bytes whose representation is selected by Turbo/VBR. This is intentionally not
    // `context - context_fixed`: an auxiliary index cache can scale with n_ctx while remaining
    // fixed-layout and outside the controller budget.
    auto sum_context_vbr_managed_bytes = [&](const dmds_t & dmds) {
        int64_t result = 0;
        if (nd == 0) {
            result += dmds.back().mb.context_vbr_managed;
        } else {
            for (size_t id = 0; id < nd; id++) {
                result += dmds[id].mb.context_vbr_managed;
            }
        }
        return result;
    };

    auto round_ctx_down = [](uint64_t n_ctx) {
        n_ctx -= n_ctx % 256;
        n_ctx = std::max<uint64_t>(n_ctx, 256);
        n_ctx = std::min<uint64_t>(n_ctx, std::numeric_limits<uint32_t>::max());
        return (uint32_t) n_ctx;
    };

    // The dynamic-VBR runtime controller's fit pass owns the "auto" KV VRAM budget. In
    // dynamic mode the KV is priced at the FLOOR tier throughout this function (swap in
    // common_fit_params) — the runtime cache starts at its configured entry and degrades toward the floor as
    // it fills, with mapped-physical bytes capped at the budget. The BUDGET handed to the
    // controller (env VBR_BUDGET_MIB) = explicit --vbr-vram, or the bytes the KV can take on
    // this box: its (floor-priced) projected footprint plus whatever would remain free above the
    // margin. That formula preserves the margin exactly: mapped - floor_cost <= free - margin.
    auto vbr_type_bits = [](enum ggml_type t) {
        return 8.0 * ggml_type_size(t) / ggml_blck_size(t);
    };

    auto vbr_scale_est = [&](uint64_t est) {
        // beyond the trained context rope is invalid and compute growth unaccounted — cap
        // every VBR-derived advert (explicit -c bypasses the estimators for power users)
        est = std::min<uint64_t>(est, hp_nct);
        return round_ctx_down(est);
    };
    // Advert-honesty cap for dynamic auto mode: the floor capacity of the GROWTH-REACHABLE
    // budget (device total - model - compute - fixed context - margin). Deliberately NOT the
    // armed snapshot: kv_size bakes at construction, so a co-tenant present at startup must not
    // permanently shrink the context ceiling the runtime budget can later grow back into.
    // Returns 0 when no cap is needed (the full trained context is servable at the floor).
    //
    // The flash-attention f16 dequant scratch grows linearly with attended width at depth
    // (turbo/degraded tiers materialize K/V to f16). It lives OUTSIDE the KV budget — it is
    // paid from the fit MARGIN — so it is charged against the margin, not on top of it:
    // vbr_scratch_excess is the scratch demand beyond the summed margins, and only that excess
    // tightens the wall. Charging margin + scratch would double-count (on a typical 24GB
    // single-model box S(n_ctx_train) ~= the 1 GiB margin and the advert is unchanged —
    // matching measured full-context fills); the budget solves must not carry it at all.
    // Single home for the scratch-vs-margin comparison — the cap subtracts it, the warn in
    // vbr_dynamic_arm_budget fires when it is > 0 for the requested context.
    auto vbr_scratch_excess = [&](uint64_t n_tokens) -> int64_t {
        double margin_total = 0.0;
        for (size_t id = 0; id < nd; id++) {
            margin_total += (double) margins[id];
        }
        return (int64_t) std::max(0.0, vbr_costs.scratch_bytes_pt * (double) n_tokens - margin_total);
    };
    auto vbr_growth_reachable_ctx_cap = [&]() -> uint32_t {
        if (nd == 0 || hp_nct == 0) {
            return 0;
        }
        int64_t budget_gr = 0;
        int64_t ctx_kv    = 0; // price-tier KV cost of the full context (RS excluded)
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & target = dmds_target_full[id];
            const llama_memory_breakdown_data extra_mb = dmds_extra_full.empty()
                ? llama_memory_breakdown_data{} : dmds_extra_full[id].mb;

            // The runtime VBR budget controls only the target KV. Reserve the complete extra
            // model/context/compute allocation first; its fixed-F16 KV cannot be spent by the
            // target controller or scaled by the target's floor-mix ratio.
            const int64_t context_fixed_layout =
                (int64_t) target.mb.context - (int64_t) target.mb.context_vbr_managed;
            budget_gr += std::max<int64_t>(0,
                target.total - (int64_t) target.mb.model - (int64_t) target.mb.compute
                    - context_fixed_layout - (int64_t) extra_mb.total() - margins[id]);
            ctx_kv += (int64_t) target.mb.context_vbr_managed;
        }
        ctx_kv = (int64_t) ((double) ctx_kv * vbr_kv_scale); // floor-mix cost, not price-tier
        budget_gr -= vbr_scratch_excess(hp_nct);
        if (ctx_kv <= 0 || budget_gr <= 0) {
            return 0;
        }
        const uint64_t cap_tokens = (uint64_t) budget_gr * hp_nct / (uint64_t) ctx_kv;
        if (cap_tokens >= hp_nct) {
            return 0;
        }
        return vbr_scale_est(cap_tokens);
    };
    // min-cap an auto-derived advert (never an explicit -c) + the honesty log
    auto vbr_cap_advert = [&]() {
        if (!cparams->vbr_dynamic || cparams->n_ctx == 0) {
            return;
        }
        const uint32_t cap = vbr_growth_reachable_ctx_cap();
        if (cap != 0 && cap < cparams->n_ctx) {
            LOG_INF("%s: VBR dynamic: advertised n_ctx capped %u -> %u = growth-reachable budget capacity at the quality floor\n",
                    __func__, cparams->n_ctx, cap);
            cparams->n_ctx = cap;
        }
    };
    // captured BEFORE any mutation: vbr_dynamic_arm_budget writes the resolved auto budget back
    // into cparams->vbr_vram_budget_bytes, so explicit-vs-auto checks must use this snapshot
    const uint64_t vbr_budget_explicit = cparams->vbr_vram_budget_bytes;
    // "the VBR/turbo cache family is in play": turbo-typed KV or any dynamic-VBR input. A plain
    // -ctk turbo3_tcq gets the same capacity estimation as --vbr-bits t3 — same cache, same math.
    const bool vbr_selected = ggml_is_turbo_kv_type(cparams->type_k) || ggml_is_turbo_kv_type(cparams->type_v) ||
                              cparams->vbr_dynamic || vbr_budget_explicit != 0 || cparams->vbr_min_bits > 0.0;
    auto vbr_dynamic_arm_budget = [&](const std::vector<int64_t> & projected_free_per_device,
                                      int64_t projected_free_host) {
        if (!cparams->vbr_dynamic) {
            return;
        }
        uint64_t budget = vbr_budget_explicit; // explicit --vbr-vram
        if (budget == 0) {
            // b algebraically reduces to free_measured - model - compute - margin: the context
            // term cancels against its copy inside projected_used. The n_ctx-INVARIANT part of
            // the context (recurrent state, sized by n_seq_max) must NOT ride that cancellation —
            // it is a real allocation the KV budget can never reuse, so charge it explicitly.
            // (It used to be charged by accident: the probe physically allocated RS, depressing
            // measured free. Once the probe honors no_alloc, only this subtraction charges it.)
            int64_t b = 0;
            if (nd == 0) {
                b = (int64_t) dmds_target_full.back().mb.context_vbr_managed
                    + projected_free_host - margins[0];
            } else {
                for (size_t id = 0; id < nd; id++) {
                    b += std::max<int64_t>(0,
                            (int64_t) dmds_target_full[id].mb.context_vbr_managed
                            + projected_free_per_device[id] - margins[id]);
                }
            }
            if (b <= 0) {
                LOG_WRN("%s: VBR dynamic: no VRAM headroom for an auto KV budget — the runtime falls "
                        "back to the floor-layout cost of the full context\n", __func__);
                return;
            }
            budget = (uint64_t) b;
        }
        // hand the resolved budget to the runtime through cparams (context not created yet),
        // plus the fit target as the runtime's growth headroom: startup arming and runtime
        // re-derivation then encode the SAME worst case, and raising -fitt hardens both
        cparams->vbr_vram_budget_bytes = budget;
        cparams->vbr_growth_headroom_bytes = (uint64_t) margin_per_dev;
        LOG_INF("%s: VBR dynamic: KV VRAM budget %" PRIu64 " MiB (%s) — decode-time degrade controller armed\n",
                __func__, std::max<uint64_t>(1, budget / MiB),
                vbr_budget_explicit != 0 ? "explicit" : "auto, from remaining memory");
        // the fit prices KV at the largest tier <= the floor; the runtime clamp lands on the
        // achievable tier MIX (vbr_kv_scale x the priced cost). Warn up front when the budget
        // cannot deliver the full context there (the runtime will also warn, at fill).
        const double floor_bpv = cparams->vbr_min_bits > 0.0 ? cparams->vbr_min_bits
                                                             : vbr_type_bits(GGML_TYPE_TURBO1_TCQ);
        const int64_t ctx_priced = sum_context_vbr_managed_bytes(dmds_target_full);
        // fire for the tier-exact default floor too (scale 1): a budget below the floor-cost
        // full context is the only startup signal for the runtime's warn-once-exceed state
        if (ctx_priced > 0 && (double) budget < (double) ctx_priced * vbr_kv_scale) {
            LOG_WRN("%s: VBR dynamic: the KV budget (%" PRIu64 " MiB) is below the full-context cost "
                    "at the %.4g bits/value floor (~%.0f MiB) — the deepest fills will hit the floor "
                    "clamp early\n", __func__, budget / MiB,
                    floor_bpv, (double) ctx_priced * vbr_kv_scale / (double) MiB);
        }
        // An explicit -c bypasses every advert estimator, so the only startup honesty signal
        // for an overcommitted context is this warn: the f16 dequant scratch is paid from the
        // fit margin, and when the full-context scratch outgrows it (shared vbr_scratch_excess,
        // same margin basis as the growth-reachable cap) the deepest fills can stop short of -c
        // — recoverably (per-request context-exceeded), not with an abort.
        if (cparams->n_ctx != 0) {
            const int64_t excess = vbr_scratch_excess(cparams->n_ctx);
            if (excess > 0) {
                LOG_WRN("%s: VBR dynamic: the f16 dequant scratch outgrows the fit margin by "
                        "~%" PRId64 " MiB at -c %u — the deepest fills may stop short of the full "
                        "context (recoverably)\n", __func__, excess / MiB, cparams->n_ctx);
            }
        }
    };

    auto vbr_estimate_ctx_from_total_budget = [&](uint64_t budget_bytes, const dmds_t & dmds_full) {
        if (!vbr_selected || cparams->n_ctx != 0 || hp_nct == 0 || budget_bytes == 0) {
            return uint32_t(0);
        }
        // in dynamic mode the KV is priced at the PRICE tier for the whole fit (see
        // common_fit_params); scale to the achievable floor-mix cost so these byte->token
        // estimates are true floor capacities, capped at the trained context (beyond it rope
        // is invalid and compute growth unaccounted)
        const int64_t ctx_full = (int64_t) ((double) sum_context_vbr_managed_bytes(dmds_full) * vbr_kv_scale);
        if (ctx_full <= 0) {
            return uint32_t(0);
        }

        const uint32_t n_ctx_probe = std::min<uint32_t>(hp_nct, std::max<uint32_t>(256, std::min<uint32_t>(n_ctx_min, hp_nct)));
        if (n_ctx_probe >= hp_nct) {
            return vbr_scale_est((uint64_t) budget_bytes * hp_nct / (uint64_t) ctx_full);
        }

        cparams->n_ctx = n_ctx_probe;
        const dmds_t dmds_probe = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        cparams->n_ctx = 0;

        const int64_t ctx_probe = (int64_t) ((double) sum_context_vbr_managed_bytes(dmds_probe) * vbr_kv_scale);
        if (ctx_probe <= 0 || ctx_full <= ctx_probe) {
            return vbr_scale_est((uint64_t) budget_bytes * hp_nct / (uint64_t) ctx_full);
        }

        uint64_t n_ctx_est = n_ctx_probe;
        if (budget_bytes > (uint64_t) ctx_probe) {
            n_ctx_est += (budget_bytes - (uint64_t) ctx_probe) * (hp_nct - n_ctx_probe) / (uint64_t) (ctx_full - ctx_probe);
        } else {
            n_ctx_est = (uint64_t) budget_bytes * n_ctx_probe / (uint64_t) ctx_probe;
        }

        return vbr_scale_est(n_ctx_est);
    };

    auto vbr_estimate_ctx_from_remaining = [&](const dmds_t & dmds_target, const std::vector<int64_t> & projected_free_per_device, int64_t projected_free_host) {
        if (!vbr_selected || cparams->n_ctx != 0 || vbr_budget_explicit != 0 || hp_nct == 0) {
            return uint32_t(0);
        }
        // dynamic mode with an auto budget: keep the model default UNLESS even the
        // growth-reachable budget cannot serve it at the quality floor — advertising cells the
        // box can never hold at the floor tier ends in the warn-once-then-exceed state at depth
        if (cparams->vbr_dynamic) {
            return vbr_growth_reachable_ctx_cap();
        }

        const uint32_t n_ctx_probe = std::min<uint32_t>(hp_nct, std::max<uint32_t>(256, std::min<uint32_t>(n_ctx_min, hp_nct)));
        if (n_ctx_probe >= hp_nct) {
            return uint32_t(0);
        }

        cparams->n_ctx = n_ctx_probe;
        const dmds_t dmds_probe = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        cparams->n_ctx = 0;

        uint64_t n_ctx_est = std::numeric_limits<uint64_t>::max();
        auto update_estimate = [&](int64_t ctx_full, int64_t ctx_probe, int64_t projected_free, int64_t margin) {
            if (ctx_full <= 0) {
                return;
            }
            const int64_t ctx_budget = ctx_full + projected_free - margin;
            if (ctx_budget <= ctx_full || ctx_budget <= 0) {
                return;
            }

            uint64_t local_est = hp_nct;
            if (ctx_probe > 0 && ctx_full > ctx_probe) {
                local_est = n_ctx_probe + (uint64_t) (ctx_budget - ctx_probe) * (hp_nct - n_ctx_probe) / (uint64_t) (ctx_full - ctx_probe);
            } else {
                local_est = (uint64_t) ctx_budget * hp_nct / (uint64_t) ctx_full;
            }
            n_ctx_est = std::min(n_ctx_est, local_est);
        };

        if (nd == 0) {
            update_estimate(dmds_target.back().mb.context, dmds_probe.back().mb.context, projected_free_host, margins[0]);
        } else {
            for (size_t id = 0; id < nd; id++) {
                update_estimate(dmds_target[id].mb.context, dmds_probe[id].mb.context, projected_free_per_device[id], margins[id]);
            }
        }

        if (n_ctx_est == std::numeric_limits<uint64_t>::max() || n_ctx_est <= hp_nct) {
            return uint32_t(0);
        }

        // vbr_scale_est caps at n_ctx_train: an uncapped estimate here advertised megatoken
        // contexts on small models (RoPE-invalid + compute-buffer OOM at warmup)
        return vbr_scale_est(n_ctx_est);
    };

    auto vbr_select_ctx = [&](const std::vector<int64_t> & projected_free_per_device, int64_t projected_free_host) {
        // the controller budget is independent of context selection — arm it even when -c is
        // explicit (the estimators below no-op on n_ctx != 0) or the estimate comes up empty
        vbr_dynamic_arm_budget(projected_free_per_device, projected_free_host);

        uint32_t vbr_n_ctx = 0;
        if (vbr_budget_explicit != 0) {
            vbr_n_ctx = vbr_estimate_ctx_from_total_budget(vbr_budget_explicit, dmds_target_full);
        } else {
            vbr_n_ctx = vbr_estimate_ctx_from_remaining(dmds_target_full, projected_free_per_device, projected_free_host);
        }
        if (vbr_n_ctx == 0) {
            return false;
        }

        cparams->n_ctx = vbr_n_ctx;
        if (cparams->vbr_dynamic) {
            LOG_INF("%s: VBR dynamic: advertised n_ctx = %" PRIu32 " = KV budget capacity at the %.4g bits/value floor (%s budget)\n",
                __func__, cparams->n_ctx,
                cparams->vbr_min_bits > 0.0 ? cparams->vbr_min_bits : vbr_type_bits(GGML_TYPE_TURBO1_TCQ),
                vbr_budget_explicit != 0 ? "explicit" : "auto");
        } else {
            LOG_TRC("%s: VBR selected n_ctx = %" PRIu32 " from %s\n",
                __func__, cparams->n_ctx, vbr_budget_explicit != 0 ? "explicit KV VRAM budget" : "remaining memory budget");
        }
        return true;
    };

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            if (vbr_select_ctx(projected_free_per_device, sum_projected_free)) {
                return;
            }
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                if (vbr_select_ctx(projected_free_per_device, 0)) {
                    return;
                }
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                log_stock_fit();
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                if (vbr_select_ctx(projected_free_per_device, 0)) {
                    return;
                }
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                log_stock_fit();
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    // the controller budget is context-independent (b reduces to free - model - compute -
    // margin), but it was only armed on the margins-met paths above — every step-2/3/4 exit
    // (context shrink, explicit -c margin miss, layer redistribution) left the runtime on the
    // conservative floor-cost fallback. Arm it here, once, for all of them. Host-only (nd == 0)
    // is skipped: dynamic VBR needs a VMM-capable device and the controller is inert without one.
    if (nd >= 1) {
        vbr_dynamic_arm_budget(projected_free_per_device, 0);
    }

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (n_ctx_auto) {
                if (n_ctx_max > n_ctx_min_total) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min_total;
                    dmds_t dmds_min_ctx = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    add_extra_memory(dmds_min_ctx, mparams);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (n_ctx_max - n_ctx_min_total) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        // round down context for CUDA backend, keep it divisible by the number of streams:
                        const uint32_t align = 256 * n_streams;
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % align, n_ctx_min_total);

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (n_ctx_max - n_ctx_min_total);
                        const int64_t memory_reduction = (n_ctx_max - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                        vbr_cap_advert(); // shrunk advert must still be floor-servable
                        if (nd <= 1) {
                            LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                        vbr_cap_advert();
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, n_ctx_max);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, n_ctx_max, n_ctx_min_total);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (sm_tensor) {
        // every layer is sharded across every device — there is no layer redistribution or CPU
        // overflow to fall back on, and tensor_split already belongs to the user
        throw common_params_fit_exception("model does not fit at the minimum context size and layer "
            "redistribution is not available under SPLIT_MODE_TENSOR, abort");
    }
    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        dmds_t dmd_nl = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmd_nl, &mparams_copy);

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    auto set_cache_layer_split = [&](const std::vector<uint32_t> & layers,
            llama_model_params & candidate, float * split,
            llama_model_tensor_buft_override * overrides) {
        GGML_ASSERT(layers.size() == nd);
        std::fill(split, split + llama_max_devices(), 0.0f);
        candidate.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            if ((uint64_t)candidate.n_gpu_layers + layers[id] > INT32_MAX) {
                throw std::runtime_error("cache candidate layer count overflowed");
            }
            candidate.n_gpu_layers += layers[id];
            if (nd > 1) {
                split[id] = layers[id];
            }
        }
        candidate.tensor_split = split;
        overrides[0] = {common_moe_cache_tensor_override_pattern(), ggml_backend_cpu_buffer_type()};
        overrides[1] = {nullptr, nullptr};
        candidate.tensor_buft_overrides = overrides;
        candidate.use_extra_bufts = false;
    };

    auto get_cache_candidate_memory = [&](const std::vector<uint32_t> & layers,
            dmds_t & candidate_memory) {
        std::vector<float> candidate_split(llama_max_devices(), 0.0f);
        std::vector<llama_model_tensor_buft_override> candidate_overrides(ntbo, {nullptr, nullptr});
        llama_model_params candidate = *mparams;
        set_cache_layer_split(layers, candidate, candidate_split.data(), candidate_overrides.data());

        std::vector<ggml_backend_dev_t> candidate_devs;
        uint32_t candidate_ngl = 0;
        uint32_t candidate_nct = 0;
        uint32_t candidate_nex = 0;
        candidate_memory = common_get_device_memory_data_impl(
                path_model, &candidate, cparams, candidate_devs,
                candidate_ngl, candidate_nct, candidate_nex, log_level);
        if (candidate_devs != devs || candidate_memory.size() != nd + 1) {
            return false;
        }
        add_extra_memory(candidate_memory, &candidate);
        for (size_t id = 0; id < nd; id++) {
            if (candidate_memory[id].mb.total() > INT64_MAX ||
                candidate_memory[id].free - (int64_t)candidate_memory[id].mb.total() < margins[id]) {
                return false;
            }
        }
        return true;
    };

    std::vector<uint32_t> cache_layers;
    dmds_t cache_memory;
    bool cache_candidate_valid = false;
    bool cache_candidate_main = false;

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {common_moe_cache_tensor_override_pattern(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        dmds_t dmds_cpu_moe = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmds_cpu_moe, mparams);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }

    if (hp_nex > 0 && global_surplus_cpu_moe > 0 && moe_cache &&
        moe_cache->mode != COMMON_MOE_CACHE_MODE_OFF && !moe_tensors.empty()) {
        std::vector<uint32_t> dense_layers(nd, 0);
        uint64_t assigned_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            dense_layers[id] = ngl_per_device[id].n_layer;
            assigned_layers += dense_layers[id];
        }

        const uint64_t required_layers = (uint64_t)hp_ngl + 1;
        if (assigned_layers == required_layers) {
            const int main_gpu = mparams->main_gpu;
            if (main_gpu >= 0 && main_gpu < (int)nd && required_layers <= UINT32_MAX) {
                std::vector<uint32_t> main_layers(nd, 0);
                main_layers[main_gpu] = (uint32_t)required_layers;
                if (get_cache_candidate_memory(main_layers, cache_memory)) {
                    cache_layers = std::move(main_layers);
                    cache_candidate_valid = true;
                    cache_candidate_main = true;
                }
            }
            if (!cache_candidate_valid && get_cache_candidate_memory(dense_layers, cache_memory)) {
                cache_layers = std::move(dense_layers);
                cache_candidate_valid = true;
            }
        }
    }

    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    bool stock_spills_experts = false;
    for (const ngl_t & layers : ngl_per_device) {
        if (layers.n_part > 0) {
            stock_spills_experts = true;
            break;
        }
    }

    if (moe_cache && moe_cache->mode != COMMON_MOE_CACHE_MODE_OFF) {
        if (!stock_spills_experts) {
            LOG_INF("%s: MoE cache fit kept stock placement because all routed expert weights fit in VRAM\n", __func__);
        } else if (moe_cache->mode == COMMON_MOE_CACHE_MODE_SOFT) {
            // Step 1: try spare-VRAM (stock placement, no expert eviction)
            common_moe_cache_fit_result soft_fit = common_moe_cache_evaluate_fit(
                    moe_cache, moe_tensors, devs, dmds_full, margins);
            if (soft_fit.feasible) {
                moe_cache->fit_selected = true;

                const double coverage = soft_fit.expert_bytes > 0
                    ? 100.0 * (double)std::min(soft_fit.cache_bytes, soft_fit.expert_bytes) /
                        (double)soft_fit.expert_bytes
                    : 0.0;
                LOG_INF("%s: MoE cache soft mode selected stock placement with %zu MiB projected cache capacity for %zu MiB of routed expert weights (up to %.1f%% coverage, no expert eviction)\n",
                        __func__,
                        soft_fit.cache_bytes / MiB, soft_fit.expert_bytes / MiB, coverage);
                for (const common_moe_cache_fit_device & device : soft_fit.devices) {
                    LOG_INF("%s: MoE cache fit CUDA%d leaves %zu MiB after reserve; minimum complete pool set is %zu MiB\n",
                            __func__, device.physical_device, device.cache_bytes / MiB,
                            soft_fit.minimum_device_bytes / MiB);
                }
            } else if (cache_candidate_valid) {
                // Step 2: spare-VRAM insufficient, try partial expert eviction
                // Binary search for minimum evicted layers where cache pools fit
                LOG_INF("%s: MoE cache soft mode: spare-VRAM insufficient (%s), searching for minimum expert eviction\n",
                        __func__, soft_fit.reason.c_str());

                const int total_layers = (int)(hp_ngl + 1);
                static std::vector<std::string> pattern_strings;
                pattern_strings.clear();
                pattern_strings.reserve(total_layers);

                // Per-layer routed-expert bytes from tensor metadata. Evicting
                // layers in ascending footprint order makes the search minimize
                // the bytes evicted, not just the number of prefix layers.
                std::vector<int64_t> layer_bytes(total_layers, 0);
                for (const llama_moe_tensor_info & tensor : moe_tensors) {
                    if (tensor.layer >= 0 && tensor.layer < total_layers &&
                        tensor.n_expert > 0 && tensor.expert_size > 0 &&
                        (uint64_t)tensor.n_expert <= INT64_MAX / (int64_t)tensor.expert_size) {
                        layer_bytes[tensor.layer] +=
                            (int64_t)tensor.n_expert * (int64_t)tensor.expert_size;
                    }
                }
                std::vector<int> order(total_layers);
                for (int i = 0; i < total_layers; i++) {
                    order[i] = i;
                }
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    return layer_bytes[a] != layer_bytes[b]
                        ? layer_bytes[a] < layer_bytes[b] : a < b;
                });

                int lo = 0, hi = total_layers;
                int best_n_evict = -1;
                int64_t best_evicted_bytes = 0;
                dmds_t best_memory;
                common_moe_cache_fit_result best_cache_fit;

                while (lo <= hi) {
                    const int mid = lo + (hi - lo) / 2;

                    // Build candidate: all layers on main GPU, the 'mid'
                    // smallest-footprint layers' experts moved to CPU
                    std::vector<float> candidate_split(llama_max_devices(), 0.0f);
                    std::vector<llama_model_tensor_buft_override> candidate_overrides(ntbo, {nullptr, nullptr});
                    llama_model_params candidate = *mparams;

                    candidate.n_gpu_layers = 0;
                    for (size_t id = 0; id < nd; id++) {
                        if ((uint64_t)candidate.n_gpu_layers + cache_layers[id] > INT32_MAX) break;
                        candidate.n_gpu_layers += cache_layers[id];
                        if (nd > 1) candidate_split[id] = (float)cache_layers[id];
                    }
                    candidate.tensor_split = candidate_split.data();

                    // Per-layer overrides: evict the chosen layers' experts to CPU
                    pattern_strings.clear();
                    int n_overrides = 0;
                    for (int i = 0; i < mid && n_overrides < (int)ntbo - 1; i++) {
                        pattern_strings.push_back(
                            llm_ffn_block_regex(order[i], LLM_FFN_EXPS_REGEX));
                        candidate_overrides[n_overrides++] = {pattern_strings.back().c_str(), ggml_backend_cpu_buffer_type()};
                    }
                    candidate_overrides[n_overrides] = {nullptr, nullptr};
                    candidate.tensor_buft_overrides = candidate_overrides.data();
                    candidate.use_extra_bufts = false;

                    // Evaluate memory and cache for this candidate
                    std::vector<ggml_backend_dev_t> candidate_devs;
                    uint32_t c_ngl = 0, c_nct = 0, c_nex = 0;
                    dmds_t candidate_memory = common_get_device_memory_data_impl(
                            path_model, &candidate, cparams, candidate_devs,
                            c_ngl, c_nct, c_nex, log_level);

                    bool candidate_valid = (candidate_devs == devs && candidate_memory.size() == nd + 1);
                    if (candidate_valid) {
                        add_extra_memory(candidate_memory, &candidate);
                        for (size_t id = 0; id < nd; id++) {
                            if (candidate_memory[id].mb.total() > INT64_MAX ||
                                candidate_memory[id].free - (int64_t)candidate_memory[id].mb.total() < margins[id]) {
                                candidate_valid = false;
                                break;
                            }
                        }
                    }

                    if (candidate_valid) {
                        common_moe_cache_fit_result cf = common_moe_cache_evaluate_fit(
                                moe_cache, moe_tensors, devs, candidate_memory, margins);
                        if (cf.feasible) {
                            best_n_evict = mid;
                            best_evicted_bytes = 0;
                            for (int i = 0; i < mid; i++) {
                                best_evicted_bytes += layer_bytes[order[i]];
                            }
                            best_memory = std::move(candidate_memory);
                            best_cache_fit = cf;
                            hi = mid - 1;  // Try fewer evictions
                        } else {
                            lo = mid + 1;  // Need more evictions
                        }
                    } else {
                        lo = mid + 1;
                    }
                }

                if (best_n_evict >= 0) {
                    // Apply the best placement with minimal expert eviction
                    std::fill(tensor_split, tensor_split + llama_max_devices(), 0.0f);
                    mparams->n_gpu_layers = 0;
                    for (size_t id = 0; id < nd; id++) {
                        mparams->n_gpu_layers += cache_layers[id];
                        if (nd > 1) tensor_split[id] = (float)cache_layers[id];
                    }
                    mparams->tensor_split = tensor_split;

                    // Generate final per-layer overrides
                    pattern_strings.clear();
                    int n_overrides = 0;
                    for (int i = 0; i < best_n_evict && n_overrides < (int)ntbo - 1; i++) {
                        pattern_strings.push_back(
                            llm_ffn_block_regex(order[i], LLM_FFN_EXPS_REGEX));
                        tensor_buft_overrides[n_overrides++] = {pattern_strings.back().c_str(), ggml_backend_cpu_buffer_type()};
                    }
                    tensor_buft_overrides[n_overrides] = {nullptr, nullptr};
                    mparams->tensor_buft_overrides = tensor_buft_overrides;
                    mparams->use_extra_bufts = false;
                    moe_cache->fit_selected = true;

                    const int n_kept = total_layers - best_n_evict;
                    const double coverage = best_cache_fit.expert_bytes > 0
                        ? 100.0 * (double)std::min(best_cache_fit.cache_bytes, best_cache_fit.expert_bytes) /
                            (double)best_cache_fit.expert_bytes
                        : 0.0;
                    const int64_t kept_bytes = best_cache_fit.expert_bytes > (size_t)best_evicted_bytes
                        ? (int64_t)best_cache_fit.expert_bytes - best_evicted_bytes : 0;
                    LOG_INF("%s: MoE cache soft mode selected partial-eviction placement: %d/%d layers keep experts GPU-resident, "
                            "%zu MiB of %zu MiB routed expert bytes evicted (%zu MiB kept), "
                            "%zu MiB projected cache capacity (up to %.1f%% coverage)\n",
                            __func__, n_kept, total_layers,
                            best_evicted_bytes / MiB, best_cache_fit.expert_bytes / MiB,
                            kept_bytes / MiB,
                            best_cache_fit.cache_bytes / MiB, coverage);
                    for (const common_moe_cache_fit_device & device : best_cache_fit.devices) {
                        LOG_INF("%s: MoE cache fit CUDA%d leaves %zu MiB after reserve; minimum complete pool set is %zu MiB\n",
                                __func__, device.physical_device, device.cache_bytes / MiB,
                                best_cache_fit.minimum_device_bytes / MiB);
                    }
                    return;
                }
                LOG_INF("%s: MoE cache soft mode kept stock placement (partial eviction could not fit cache pools)\n",
                        __func__);
            } else {
                LOG_INF("%s: MoE cache soft mode kept stock placement (spare-VRAM insufficient): %s\n",
                        __func__, soft_fit.reason.c_str());
            }
        } else if (!cache_candidate_valid) {
            LOG_INF("%s: MoE cache fit kept stock placement because canonical dense weights do not meet the fit targets\n", __func__);
        } else {
            common_moe_cache_fit_result cache_fit = common_moe_cache_evaluate_fit(
                    moe_cache, moe_tensors, devs, cache_memory, margins);
            if (cache_fit.feasible) {
                set_cache_layer_split(cache_layers, *mparams, tensor_split, tensor_buft_overrides);
                moe_cache->fit_selected = true;

                const double coverage = cache_fit.expert_bytes > 0
                    ? 100.0 * (double)std::min(cache_fit.cache_bytes, cache_fit.expert_bytes) /
                        (double)cache_fit.expert_bytes
                    : 0.0;
                LOG_INF("%s: MoE cache fit selected %s dense placement with %zu MiB projected cache capacity for %zu MiB of routed expert weights (up to %.1f%% coverage)\n",
                        __func__, cache_candidate_main ? "main-device" : "packed",
                        cache_fit.cache_bytes / MiB, cache_fit.expert_bytes / MiB, coverage);
                for (const common_moe_cache_fit_device & device : cache_fit.devices) {
                    LOG_INF("%s: MoE cache fit CUDA%d leaves %zu MiB after reserve; minimum complete pool set is %zu MiB\n",
                            __func__, device.physical_device, device.cache_bytes / MiB,
                            cache_fit.minimum_device_bytes / MiB);
                }
                return;
            }
            LOG_INF("%s: MoE cache fit kept stock placement: %s\n", __func__, cache_fit.reason.c_str());
        }
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

// largest turbo tier whose bits/value does not exceed the requested floor (t1 when 0/auto);
// used to PRICE the KV during fitting in dynamic VBR mode
static ggml_type common_vbr_floor_price_tier(double floor_bpv) {
    const ggml_type tiers[] = {
        GGML_TYPE_F16, GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ,
    };
    if (floor_bpv > 0.0) {
        for (ggml_type t : tiers) {
            if (8.0 * ggml_type_size(t) / ggml_blck_size(t) <= floor_bpv + 1e-9) {
                return t;
            }
        }
    }
    return GGML_TYPE_TURBO1_TCQ;
}

ggml_type common_vbr_fit_price_type(ggml_type entry, double floor_bpv, bool pinned) {
    if (pinned || (entry != GGML_TYPE_F16 && !ggml_is_turbo_kv_type(entry))) {
        return entry;
    }
    const ggml_type price_t = common_vbr_floor_price_tier(floor_bpv);
    const double price_bits = 8.0 * ggml_type_size(price_t) / ggml_blck_size(price_t);
    const double entry_bits = 8.0 * ggml_type_size(entry) / ggml_blck_size(entry);
    return price_bits < entry_bits ? price_t : entry;
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        common_moe_cache_params * moe_cache,
        size_t * margins,
        uint32_t n_ctx_min,
        const common_fit_extra_model * extra,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;

    // SPLIT_MODE_TENSOR with everything explicit (-c AND --vbr-vram): when there is no extra
    // model/context and no MoE fit authority, nothing remains to size or arm (the estimators
    // no-op on explicit values and there is no layer redistribution). Required extras and MoE
    // policy must still pass through the aggregate-device measurement.
    const bool moe_fit_active = moe_cache != nullptr && moe_cache->mode != COMMON_MOE_CACHE_MODE_OFF;
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR && cparams->vbr_dynamic &&
            cparams->n_ctx != 0 && cparams->vbr_vram_budget_bytes != 0 &&
            extra == nullptr && !moe_fit_active) {
        LOG_INF("%s: SPLIT_MODE_TENSOR: -c and --vbr-vram both explicit — nothing to fit, skipping\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_SUCCESS;
    }

    // Dynamic VBR: price the KV at the degrade FLOOR for the whole fit, not at the configured entry
    // tier. The runtime controller caps mapped-physical KV at the budget and the layout sits at
    // the floor when the context is full — so floor cost is what capacity/fitting must assume.
    // With this swap every fit branch does the right thing: a memory-constrained box shrinks
    // n_ctx to the FLOOR capacity of its VRAM (not the ~6.5x smaller entry-tier capacity), and
    // the auto KV budget prices its headroom correctly. Entry types are restored for the real
    // load — the cache still starts at its configured entry and degrades toward the floor as it fills.
    const ggml_type type_k_entry = cparams->type_k;
    const ggml_type type_v_entry = cparams->type_v;
    if (cparams->vbr_dynamic) {
        const ggml_type price_t = common_vbr_floor_price_tier(cparams->vbr_min_bits);
        // only the degradable sides price at the floor: a swappable type (F16 dynamic entry or
        // a turbo tier) on a side that is not pin-flagged. A PINNED side (vbr_pin_k/v, or an
        // explicit q8_0/bf16 the runtime cannot transcode) keeps its real cost — pricing it at
        // the floor tier would over-advertise capacity for that half. Mirrors the runtime's
        // vbr_unit_movable contract: type swappable AND side not pinned.
        auto movable = [](ggml_type t, bool pinned) {
            return !pinned && (t == GGML_TYPE_F16 || ggml_is_turbo_kv_type(t));
        };
        if (movable(cparams->type_k, cparams->vbr_pin_k)) {
            cparams->type_k = common_vbr_fit_price_type(type_k_entry, cparams->vbr_min_bits, false);
        }
        if (movable(cparams->type_v, cparams->vbr_pin_v)) {
            cparams->type_v = common_vbr_fit_price_type(type_v_entry, cparams->vbr_min_bits, false);
        }
        LOG_INF("%s: VBR dynamic: fitting with KV priced at the %s floor tier%s\n",
                __func__, ggml_type_name(price_t),
                (movable(type_k_entry, cparams->vbr_pin_k) && movable(type_v_entry, cparams->vbr_pin_v))
                    ? "" : " (pinned side at its own cost)");
    }

    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides,
                moe_cache, margins, n_ctx_min, extra, log_level,
                type_k_entry, type_v_entry);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }

    cparams->type_k = type_k_entry;
    cparams->type_v = type_v_entry;

    // fit-derived adverts are already floor-mix-priced (the estimators inside
    // common_params_fit_impl scale measured price-tier KV bytes by the degrade-order walk's
    // achievable mix cost) — no post-adjustment needed here

    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
