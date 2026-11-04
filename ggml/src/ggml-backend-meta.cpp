#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-meta-step-cache.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <thread>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <atomic>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_DISJOINT:
            return "DISJOINT";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < this->simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (this->simple_devs[i]);
            description += ggml_backend_dev_description(this->simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

// Host-weight expert matmuls (CPU-resident experts during prefill) are offloaded when every simple
// device would offload them; calculate_split_state gives the scheduler's weight copy the weight's
// expert split, found by name. Dense weights, sidecar operands (src[3]+, e.g. EXL3 scales) and copy
// names that GGML_MAX_NAME would truncate have no split rule for their copy, so they stay on the host.
static bool ggml_backend_meta_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    if (op->op != GGML_OP_MUL_MAT_ID || op->src[3] != nullptr) {
        return false;
    }
    // The copy is named "Meta(<dev>,<dev>...)#<weight>#<n>", as the meta backend names itself. Size the
    // weight as the layer's longest expert name, so gate/up/down of one layer all offload or none do.
    const char * weight = op->src[0]->name;
    const char * ffn    = strstr(weight, "ffn_");
    size_t name_len = strlen("Meta()") + meta_dev_ctx->simple_devs.size() - 1 + strlen("##0") +
        (ffn ? size_t(ffn - weight) + strlen("ffn_gate_up_exps.weight") : strlen(weight));
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        name_len += strlen(ggml_backend_dev_name(simple_dev));
    }
    if (name_len >= GGML_MAX_NAME) {
        return false;
    }
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_offload_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ ggml_backend_meta_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < this->simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(this->simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

// Shape of simple buffer j's shard of a tensor split along ss.axis (0 <= axis < GGML_MAX_DIMS).
static void ggml_backend_meta_shard_shape(const ggml_tensor * tensor, const ggml_backend_meta_split_state & ss,
        size_t n_bufs, size_t j, int64_t ne[GGML_MAX_DIMS], size_t nb[GGML_MAX_DIMS]) {
    ne[ss.axis] = 0;
    for (size_t s = 0; s < ss.n_segments; s++) {
        ne[ss.axis] += ss.ne[s*n_bufs + j] * ss.nr[s];
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor->nb[i] > tensor->nb[ss.axis]) {
            nb[i] = tensor->nb[i] * ne[ss.axis]/tensor->ne[ss.axis];
        }
    }
}

// Total bytes across all simple buffers, as ggml_backend_meta_alloc_tensors_ext would place them:
// a mirrored tensor costs a full copy per device, a split tensor one shard per device.
// Sized from the registered split rules (statically allocated tensors are leaves); a tensor without
// a rule keeps the max-over-devices size.
static size_t ggml_backend_meta_buffer_type_get_alloc_size_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) ggml_backend_buft_get_device(buft)->context;

    // an empty shard still gets the child's padding, unless every shard on that device is empty
    std::vector<size_t> sizes(n_simple_bufts, 0);
    std::vector<bool> any_nonzero(n_simple_bufts, false);
    for (int i = 0; i < n_tensors; i++) {
        const ggml_tensor * t = tensors[i];
        if (t->data != nullptr || t->view_src != nullptr || ggml_nelements(t) == 0) {
            continue;
        }
        const ggml_backend_meta_split_state ss = dev_ctx->get_split_state(t, dev_ctx->get_split_state_ud);
        for (size_t j = 0; j < n_simple_bufts; j++) {
            ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, j);
            size_t size;
            if (ss.axis >= 0 && ss.axis < GGML_MAX_DIMS) {
                ggml_tensor shard = *t;
                ggml_backend_meta_shard_shape(t, ss, n_simple_bufts, j, shard.ne, shard.nb);
                any_nonzero[j] = any_nonzero[j] || ggml_nelements(&shard) != 0;
                size = ggml_backend_buft_get_alloc_size(simple_buft, &shard);
            } else if (ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || ss.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                any_nonzero[j] = true;
                size = ggml_backend_buft_get_alloc_size(simple_buft, t);
            } else {
                any_nonzero[j] = any_nonzero[j] || j == 0;
                size = j == 0 ? ggml_backend_meta_buffer_type_get_alloc_size(buft, t) : 0;
            }
            sizes[j] += GGML_PAD(size, ggml_backend_buft_get_alignment(simple_buft));
        }
    }
    size_t total = 0;
    for (size_t j = 0; j < n_simple_bufts; j++) {
        total += any_nonzero[j] ? sizes[j] : 0;
    }
    return total;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name            = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_meta_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_meta_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_context_ptr> full; // filled chunks, kept until the next clear
    size_t mem_size = 0;
    uint64_t gen = 0; // allocation of the owning graph these shards belong to
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) : mem_size(params.mem_size) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}

    ggml_context * ctx(size_t j) {
        ggml_context * c = ctxs[j].get();
        if (ggml_get_mem_size(c) - ggml_used_mem(c) < ggml_tensor_overhead()) {
            full.push_back(std::move(ctxs[j]));
            ctxs[j].reset(ggml_init({ mem_size, nullptr, true }));
        }
        return ctxs[j].get();
    }

    void clear() {
        for (ggml_context_ptr & c : ctxs) {
            ggml_reset(c.get());
        }
        full.clear();
        simple_tensors.clear();
    }
};

// Externally created views (and compute tensors) need shards too. They belong to the graph being allocated,
// so they are kept per owner = (scheduler, graph slot) and dropped when that owner allocates again: graphs
// parked in other slots, or allocated by another scheduler on the same weights, keep theirs.
static thread_local uint64_t meta_cur_owner = 0;
static std::mutex meta_owner_mutex;
static std::unordered_map<uint64_t, uint64_t> meta_owner_gens;
static uint64_t meta_owner_next_gen = 1;

static_assert(GGML_SCHED_MAX_GRAPH_SLOTS <= 16, "owner keys pack the slot into 4 bits");

static uint64_t ggml_backend_meta_owner_key(const void * sched, int slot) {
    return ((uint64_t) (uintptr_t) sched << 4) | (uint64_t) slot;
}

static uint64_t ggml_backend_meta_owner_gen(uint64_t owner) {
    std::lock_guard<std::mutex> lock(meta_owner_mutex);
    auto it = meta_owner_gens.find(owner);
    return it == meta_owner_gens.end() ? 0 : it->second;
}

void ggml_backend_meta_set_graph_owner(const void * sched, int slot, bool realloc) {
    GGML_ASSERT(slot >= 0 && slot < GGML_SCHED_MAX_GRAPH_SLOTS);
    meta_cur_owner = ggml_backend_meta_owner_key(sched, slot);
    std::lock_guard<std::mutex> lock(meta_owner_mutex);
    // every current owner has a nonzero generation, so a container never matches a forgotten one
    auto it = meta_owner_gens.find(meta_cur_owner);
    if (realloc || it == meta_owner_gens.end()) {
        meta_owner_gens[meta_cur_owner] = meta_owner_next_gen++;
    }
}

void ggml_backend_meta_forget_graph_owner(const void * sched) {
    std::lock_guard<std::mutex> lock(meta_owner_mutex);
    for (int slot = 0; slot < GGML_SCHED_MAX_GRAPH_SLOTS; slot++) {
        // a missing owner has generation 0, which no container holds, so its shards are never found again
        meta_owner_gens.erase(ggml_backend_meta_owner_key(sched, slot));
    }
}

struct ggml_backend_meta_buffer_context {
    ggml_backend_meta_simple_tensor_container stc_static;
    std::unordered_map<uint64_t, ggml_backend_meta_simple_tensor_container> stc_owned;
    ggml_init_params params_owned;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            const ggml_init_params & params_owned,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), params_owned(params_owned) {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    // static shards first, then the current owner's, then those of any other owner whose graph is still allocated
    const std::vector<ggml_tensor *> * find_simple_tensors(const ggml_tensor * tensor) {
        auto it = stc_static.simple_tensors.find(tensor);
        if (it != stc_static.simple_tensors.end()) {
            return &it->second;
        }
        auto find_owned = [&](uint64_t owner, ggml_backend_meta_simple_tensor_container & stc) -> const std::vector<ggml_tensor *> * {
            if (stc.gen != ggml_backend_meta_owner_gen(owner)) {
                return nullptr;
            }
            auto it_owned = stc.simple_tensors.find(tensor);
            return it_owned == stc.simple_tensors.end() ? nullptr : &it_owned->second;
        };
        auto cur = stc_owned.find(meta_cur_owner);
        if (cur != stc_owned.end()) {
            if (const auto * found = find_owned(cur->first, cur->second)) {
                return found;
            }
        }
        for (auto & [owner, stc] : stc_owned) {
            if (owner == meta_cur_owner) {
                continue;
            }
            if (const auto * found = find_owned(owner, stc)) {
                return found;
            }
        }
        return nullptr;
    }

    // the current owner's container, emptied first if that owner allocated a new graph since it was filled
    ggml_backend_meta_simple_tensor_container & owned_container() {
        auto it = stc_owned.find(meta_cur_owner);
        if (it == stc_owned.end()) {
            // recycle the container of an owner that reallocated or was freed, so freed schedulers do not accumulate
            for (auto stale = stc_owned.begin(); stale != stc_owned.end(); ++stale) {
                if (stale->second.gen != ggml_backend_meta_owner_gen(stale->first)) {
                    auto node = stc_owned.extract(stale);
                    node.key() = meta_cur_owner;
                    node.mapped().clear();
                    node.mapped().gen = ggml_backend_meta_owner_gen(meta_cur_owner);
                    it = stc_owned.insert(std::move(node)).position;
                    break;
                }
            }
        }
        if (it == stc_owned.end()) {
            it = stc_owned.emplace(meta_cur_owner, ggml_backend_meta_simple_tensor_container(params_owned, (int) bufs.size())).first;
        }
        const uint64_t gen = ggml_backend_meta_owner_gen(meta_cur_owner);
        if (it->second.gen != gen) {
            it->second.clear();
            it->second.gen = gen;
        }
        return it->second;
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    if (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        GGML_ABORT("tensor '%s' (%s) lives in buffer '%s', not a meta buffer", tensor->name, ggml_op_desc(tensor),
            tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : "(none)");
    }
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    const std::vector<ggml_tensor *> * simple_tensors = buf_ctx->find_simple_tensors(tensor);
    return simple_tensors ? (*simple_tensors)[index] : nullptr;
}

static bool ggml_backend_meta_split_state_has_sources(const ggml_tensor * tensor) {
    return ggml_nelements(tensor) != 0 &&
        (ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE || tensor->view_src != nullptr);
}

static const ggml_tensor * ggml_backend_meta_split_state_source(const ggml_tensor * tensor, size_t i) {
    // ggml_view_tensor is an OP_NONE full view, not an operation with src[0].
    // Resolve it through the iterative dependency stack too.
    if (tensor->op == GGML_OP_NONE && tensor->view_src != nullptr) {
        return i == 0 ? tensor->view_src : nullptr;
    }
    return tensor->src[i];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_calculate_split_state(
        const struct ggml_tensor * tensor, bool assume_sync, const std::vector<ggml_backend_meta_split_state> & src_ss) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            // name the node and its sources before aborting: "axis unknown" alone is undebuggable
            fprintf(stderr, "ggml_backend_meta: no split rule for %s '%s' [%lld,%lld,%lld,%lld] (scalar_only=%d):\n",
                    ggml_op_name(tensor->op), tensor->name,
                    (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3], scalar_only);
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) continue;
                fprintf(stderr, "    src%zu %s '%s' [%lld,%lld,%lld,%lld] axis=%d n_segments=%zu\n", i,
                        ggml_op_name(tensor->src[i]->op), tensor->src[i]->name,
                        (long long) tensor->src[i]->ne[0], (long long) tensor->src[i]->ne[1], (long long) tensor->src[i]->ne[2], (long long) tensor->src[i]->ne[3],
                        (int) src_ss[i].axis, (size_t) src_ss[i].n_segments);
            }
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        // Expert parallelism. A disjoint expert output scaled by the replicated router weights starts the
        // partial branch that the expert-slot sum and the delayed all-reduce complete (the planner sees a
        // reduced value when asked to assume the sync). Two disjoint outputs of the same routing (gate and
        // up) combine element-wise into a disjoint one; their sum is a partial sum.
        if ((tensor->op == GGML_OP_MUL || tensor->op == GGML_OP_DIV) &&
                src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_DISJOINT && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_DISJOINT && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_DISJOINT) {
            if (tensor->op == GGML_OP_MUL) {
                return src_ss[0];
            }
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        // A sharded operand combined with a full-width MIRRORED one (e.g. the turbo V-mean tap
        // adding a per-channel mu row to a channel-sharded attention output). The fast path above
        // only covers a mirrored operand that BROADCASTS along the split axis (ne[axis] == 1);
        // here mu is full width, so it is resident in its entirety on every device and each shard
        // uses its own slice. Result keeps the src[0] split.
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        // expert parallelism: the weights are split on the expert axis and every device runs mul_mat_id over
        // all tokens with its own expert window (foreign experts give zero rows), so the output is a partial
        // sum. The down projection consumes the (disjoint) partial up/gate activation of the same layer.
        if (tensor->op == GGML_OP_MUL_MAT_ID && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                // up/gate: every (token, slot) row is produced on exactly one device, the others hold zeros
                return {GGML_BACKEND_SPLIT_AXIS_DISJOINT, {0}, {1}, 1};
            }
            if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_DISJOINT || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                // down: the rows are still device-disjoint; the expert-slot sum that follows makes it a partial sum
                return {GGML_BACKEND_SPLIT_AXIS_DISJOINT, {0}, {1}, 1};
            }
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool allow_permuted_src = false) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                if (ggml_is_permuted(tensor) || (!allow_permuted_src && ggml_is_permuted(tensor->src[0]))) {
                    const ggml_tensor * s = tensor->src[0];
                    fprintf(stderr, "ggml_backend_meta: %s '%s' [%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] of %s '%s' [%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] (src split axis %d): permuted reshape not supported\n",
                            ggml_op_name(tensor->op), tensor->name,
                            (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3],
                            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3],
                            ggml_op_name(s->op), s->name, (long long) s->ne[0], (long long) s->ne[1], (long long) s->ne[2], (long long) s->ne[3],
                            s->nb[0], s->nb[1], s->nb[2], s->nb[3], (int) src_ss[0].axis);
                }
                GGML_ASSERT(!ggml_is_permuted(tensor));
                if (!allow_permuted_src) {
                    GGML_ASSERT(!ggml_is_permuted(tensor->src[0]));
                }
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            case GGML_BACKEND_SPLIT_AXIS_DISJOINT: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // a same-shape copy (materialising a permuted view, e.g. the selected keys of an attention layer)
        // is element-wise: every device copies its own shard, the split state carries over unchanged
        if (ggml_are_same_shape(tensor, tensor->src[0]) && tensor->src[1] == nullptr) {
            return src_ss[0];
        }
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss, /*allow_permuted_src =*/ !ggml_is_permuted(tensor));
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis == GGML_BACKEND_SPLIT_AXIS_3) {
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] > 1 && tensor->nb[dim] == tensor->src[0]->nb[3]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL ||
                src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_DISJOINT) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            case GGML_BACKEND_SPLIT_AXIS_DISJOINT: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            case GGML_BACKEND_SPLIT_AXIS_DISJOINT: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    // The model's name-based split rules, as registered on the meta device.
    auto user_split_state = [&](const ggml_tensor * t) -> ggml_backend_meta_split_state {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
        const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
        return dev_ctx->get_split_state(t, dev_ctx->get_split_state_ud);
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (!ggml_backend_meta_split_state_has_sources(tensor)) {
            ggml_backend_meta_split_state ret = user_split_state(tensor);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        // Owner placement: a tensor held whole by one device (empty on the others) combined only with
        // mirrored tensors is computed on that device alone, and its result is held whole there too.
        // A projection back to the residual stream (an axis 0 weight) ends the owner chain as a partial
        // sum that the non-owners contribute zeros to.
        int owner = -1;
        bool owner_chain = n_bufs > 1;
        for (size_t i = 0; owner_chain && i < GGML_MAX_SRC; i++) {
            const ggml_tensor * src = ggml_backend_meta_split_state_source(tensor, i);
            if (src == nullptr || src == tensor || src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                continue;
            }
            const ggml_backend_meta_split_state & ss = src_ss[i];
            owner_chain = ss.axis >= 0 && ss.axis < GGML_MAX_DIMS && ss.n_segments == 1 && ss.nr[0] == 1;
            for (size_t j = 0; owner_chain && j < n_bufs; j++) {
                if (ss.ne[j] == 0) {
                    continue;
                }
                owner_chain = ss.ne[j] == src->ne[ss.axis] && (owner < 0 || owner == int(j));
                owner = j;
            }
        }
        if (owner_chain && owner >= 0) {
            // (handle_mul_mat would reject an axis 0 weight against a mirrored input, which is valid here)
            if (tensor->op == GGML_OP_MUL_MAT && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->src[0]->ne[1] > 1 &&
                    ggml_backend_buffer_get_usage(tensor->src[0]->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
            }
            ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            ret.ne[owner] = tensor->ne[0];
            return ret;
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                if (tensor->view_src != nullptr) {
                    return src_ss[0];
                }
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                // A scheduler copy of a host-resident expert weight ("<backend>#<weight>#<n>") keeps the
                // weight's expert split, so each device uploads and runs only its own experts.
                const char * first = strchr(tensor->name, '#');
                const char * last  = strrchr(tensor->name, '#');
                if (first != nullptr && last > first + 1) {
                    ggml_tensor weight = *tensor;
                    snprintf(weight.name, sizeof(weight.name), "%.*s", int(last - first - 1), first + 1);
                    const ggml_backend_meta_split_state ss = user_split_state(&weight);
                    if (ss.axis == GGML_BACKEND_SPLIT_AXIS_2) {
                        return ss; // skip the ratio take-over below: a copy has no split source to take it from
                    }
                }
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT: {
                split_state = handle_reshape(src_ss, /*allow_permuted_src=*/ true);
            } break;
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARANGE: {
                // ARANGE is source-less, so every device constructs the same values.
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_PARAMS:
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU:
            case GGML_OP_TURBO_WHT: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    ggml_backend_meta_split_state ret = calculate_split_state();
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

// Keep whole-cache invalidation: cached dependents may also be stale. A miss
// must not recursively walk the residual stream with large split-state frames.
static const ggml_backend_meta_split_state * ggml_backend_meta_find_split_state(const ggml_tensor * tensor, bool assume_sync) {
    auto * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    auto it = buf_ctx->split_state_cache.find({tensor, assume_sync});
    if (it == buf_ctx->split_state_cache.end()) {
        return nullptr;
    }
    if (memcmp(it->second.second, tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        return nullptr;
    }
    return &it->second.first;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    // The common hit path does no allocation or dependency traversal.
    if (const auto * cached = ggml_backend_meta_find_split_state(tensor, assume_sync)) {
        return *cached;
    }

    struct frame {
        const ggml_tensor * tensor;
        bool assume_sync;
        size_t next_src = 0;
        std::vector<ggml_backend_meta_split_state> src_ss;

        frame(const ggml_tensor * t, bool sync) : tensor(t), assume_sync(sync) {
            if (ggml_backend_meta_split_state_has_sources(t)) {
                src_ss.resize(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1});
            }
        }
    };

    frame current(tensor, assume_sync);
    // Allocate the parent stack only if a dependency is also missing. Copies
    // of resolved source states survive cache invalidation in a later sibling,
    // just as they did in the recursive resolver; do not retain map references.
    std::vector<frame> parents;
    for (;;) {
        if (current.next_src < current.src_ss.size()) {
            const auto * src = ggml_backend_meta_split_state_source(current.tensor, current.next_src);
            if (src == nullptr || src == current.tensor) {
                ++current.next_src;
            } else if (const auto * cached = ggml_backend_meta_find_split_state(src, true)) {
                GGML_ASSERT(cached->axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
                current.src_ss[current.next_src++] = *cached;
            } else {
                parents.push_back(std::move(current));
                current = frame(src, true);
            }
            continue;
        }

        const auto * tensor = current.tensor;
        auto * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
        const auto ret = ggml_backend_meta_calculate_split_state(tensor, current.assume_sync, current.src_ss);
        auto & entry = buf_ctx->split_state_cache[{tensor, current.assume_sync}];
        entry.first = ret;
        memcpy(entry.second, tensor, sizeof(entry.second));
        if (buf_ctx->debug > 0) {
            const size_t n_bufs = buf_ctx->bufs.size();
            std::string srcs_info;
            for (size_t i = 0; i < current.src_ss.size(); i++) {
                const auto * src = ggml_backend_meta_split_state_source(tensor, i);
                if (src == nullptr || src == tensor) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state & split_state = current.src_ss[i];
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(src->name) + "[" + ggml_op_name(src->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ne_info += std::to_string(ret.ne[j]) + "x" + std::to_string(ret.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(ret.axis), ne_info.c_str());
        }
        if (parents.empty()) {
            return ret;
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        current = std::move(parents.back());
        parents.pop_back();
        current.src_ss[current.next_src++] = ret;
    }
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}


static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctx(j);
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ggml_backend_meta_shard_shape(tensor, split_state, n_simple_bufs, j, ne, nb);
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        if (tensor->op == GGML_OP_MUL_MAT_ID && tensor->src[0] != nullptr && ggml_backend_buffer_is_meta(tensor->src[0]->buffer)) {
            // expert parallelism: this device computes only its own experts of the routed expert space
            const ggml_backend_meta_split_state ss0 = ggml_backend_meta_get_split_state(tensor->src[0], /*assume_sync =*/ true);
            if (ss0.axis == GGML_BACKEND_SPLIT_AXIS_2 && ss0.n_segments == 1) {
                int32_t lo = 0;
                for (size_t jj = 0; jj < j; jj++) {
                    lo += (int32_t) ss0.ne[jj];
                }
                // (the per-device src pointers are wired below; set the params directly)
                ggml_set_op_params_i32(t_ij, GGML_MMID_WINDOW_LO, lo);
                ggml_set_op_params_i32(t_ij, GGML_MMID_WINDOW_N_LOCAL, (int32_t) ss0.ne[j]);
            }
        }
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            if (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } else if (simple_buf != nullptr) {
            if (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        if (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } else {
            t_ij->extra = tensor->extra;
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const bool is_static = buf_ctx->stc_static.simple_tensors.count(tensor) != 0;
    return ggml_backend_meta_buffer_init_tensor_impl(is_static ? buf_ctx->stc_static : buf_ctx->owned_container(), tensor);
}

// per-owner shard tensors are created in chunks of this many
static const ggml_init_params ggml_backend_meta_params_owned = {
    /*.mem_size   =*/ 4096*ggml_tensor_overhead(),
    /*.mem_buffer =*/ nullptr,
    /*.no_alloc   =*/ true,
};

// Byte ranges of a tensor split along axis 0-2. Every stripe (one step of the next axis) holds each
// device's pieces of every segment/repeat in turn, and a shard stripe holds that device's pieces in
// the same order, so any byte range maps onto one contiguous range of each shard: all get/set/memset
// of split tensors, whole-tensor or not, sync or async, go through these spans.
struct ggml_backend_meta_span {
    size_t full;  // offset in the meta tensor
    size_t local; // offset in the shard
    size_t size;
};

// The pieces of [offset, offset + size) that device j holds, in order. Along axis 0 a piece is made of
// quantization blocks, or for EXL3 of 16x16 tiles (32*bits bytes each) taken from each 16-row stripe.
static std::vector<ggml_backend_meta_span> ggml_backend_meta_spans(const ggml_tensor * tensor,
        const ggml_backend_meta_split_state & split, size_t n_bufs, size_t j, size_t offset, size_t size) {
    const int axis = split.axis;
    const bool exl3 = axis == GGML_BACKEND_SPLIT_AXIS_0 && ggml_type_is_exl3(tensor->type);
    const int64_t unit_elems = exl3 ? 16 : axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
    const size_t  unit_bytes = exl3 ? 32 * ggml_exl3_bits(tensor->type) : tensor->nb[axis];
    // single-segment split states need not carry ne: each piece is then a whole shard stripe
    const bool whole = split.n_segments == 1 && split.nr[0] == 1;
    std::vector<size_t> piece(split.n_segments * n_bufs);
    size_t local_stripe = 0;
    for (size_t s = 0; s < split.n_segments; ++s) {
        for (size_t rank = 0; rank < n_bufs; ++rank) {
            const int64_t ne = whole ? ggml_backend_meta_buffer_simple_tensor(tensor, rank)->ne[axis] : split.ne[s*n_bufs + rank];
            GGML_ASSERT(ne % unit_elems == 0);
            piece[s*n_bufs + rank] = ne / unit_elems * unit_bytes;
        }
        local_stripe += split.nr[s] * piece[s*n_bufs + j];
    }
    const size_t stripe_bytes = tensor->ne[axis] / unit_elems * unit_bytes;
    const size_t n_stripes    = ggml_nbytes(tensor) / stripe_bytes;

    std::vector<ggml_backend_meta_span> spans;
    for (size_t stripe = offset / stripe_bytes; stripe < n_stripes && stripe * stripe_bytes < offset + size; ++stripe) {
        size_t full = stripe * stripe_bytes, local = stripe * local_stripe;
        for (size_t s = 0; s < split.n_segments; ++s) {
            for (size_t r = 0; r < split.nr[s]; ++r) {
                for (size_t rank = 0; rank < n_bufs; ++rank) {
                    const size_t bytes = piece[s*n_bufs + rank];
                    if (rank == j) {
                        const size_t lo = std::max(full, offset), hi = std::min(full + bytes, offset + size);
                        if (hi > lo) {
                            spans.push_back({ lo, local + lo - full, hi - lo });
                        }
                        local += bytes;
                    }
                    full += bytes;
                }
            }
        }
    }
    return spans;
}

// The meta-tensor stride of spans that one strided (2D) copy can move; 0 when they are uneven.
static size_t ggml_backend_meta_spans_stride(const std::vector<ggml_backend_meta_span> & spans) {
    const size_t stride = spans.size() > 1 ? spans[1].full - spans[0].full : spans[0].size;
    for (size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].size != spans[0].size || spans[i].full - spans[i - 1].full != stride) {
            return 0;
        }
    }
    return stride;
}

// Moves a device's spans as ONE contiguous shard transfer, gathered/scattered through host staging.
static void ggml_backend_meta_spans_staged(ggml_tensor * shard, const std::vector<ggml_backend_meta_span> & spans,
        const void * upload, void * download, size_t offset) {
    const size_t start = spans.front().local;
    std::vector<uint8_t> staging(spans.back().local + spans.back().size - start);
    if (download) {
        ggml_backend_tensor_get(shard, staging.data(), start, staging.size());
    }
    for (const auto & s : spans) {
        uint8_t * stage = staging.data() + s.local - start;
        if (download) {
            memcpy(static_cast<uint8_t *>(download) + s.full - offset, stage, s.size);
        } else {
            memcpy(stage, static_cast<const uint8_t *>(upload) + s.full - offset, s.size);
        }
    }
    if (upload) {
        ggml_backend_tensor_set(shard, staging.data(), start, staging.size());
    }
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            for (size_t j = 0; j < n_bufs; j++) {
                const auto spans = ggml_backend_meta_spans(tensor, split_state, n_bufs, j, offset, size);
                if (!spans.empty()) {
                    ggml_backend_tensor_memset(ggml_backend_meta_buffer_simple_tensor(tensor, j), value,
                        spans.front().local, spans.back().local + spans.back().size - spans.front().local);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
        case GGML_BACKEND_SPLIT_AXIS_DISJOINT: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Each device gets ONE set: backends that repack weights at upload time (e.g. the CUDA
            // Marlin paths) only do so for complete tensors, so pieces are staged rather than strided.
            for (size_t j = 0; j < n_bufs; j++) {
                const auto spans = ggml_backend_meta_spans(tensor, split_state, n_bufs, j, offset, size);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                if (spans.size() == 1) {
                    ggml_backend_tensor_set(simple_tensor, (const char *) data + spans[0].full - offset, spans[0].local, spans[0].size);
                } else if (!spans.empty()) {
                    ggml_backend_meta_spans_staged(simple_tensor, spans, data, nullptr, offset);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            size_t n_contributors = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            for (size_t j = 0; j < n_bufs; j++) {
                const auto spans = ggml_backend_meta_spans(tensor, split_state, n_bufs, j, offset, size);
                if (spans.empty()) {
                    continue;
                }
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t stride = ggml_backend_meta_spans_stride(spans);
                if (stride != 0) {
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + spans[0].full - offset, spans[0].local,
                        spans[0].size, spans.size(), spans[0].size, stride);
                } else {
                    ggml_backend_meta_spans_staged(simple_tensor, spans, nullptr, data, offset);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        if (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_backend_meta_simple_tensor_container stc_static;

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, ggml_backend_meta_params_owned, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

static ggml_backend_buffer_t ggml_backend_meta_alloc_tensors_ext(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors,
        ggml_backend_meta_alloc_simple_t alloc, void * userdata) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static(params_static, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, ggml_backend_meta_params_owned, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (int i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        if (alloc != nullptr) {
            // caller-managed allocation: the callback places every non-view shard tensor
            // (data + buffer); views are finalized below
            meta_buf_ctx->bufs[i].reset(alloc(i, simple_buft, ctx, userdata));
        } else {
            // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
            // For those edge cases, allocate a dummy buffer instead.
            bool any_nonzero_slice = false;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                if (ggml_nelements(t) != 0) {
                    any_nonzero_slice = true;
                    break;
                }
            }
            if (any_nonzero_slice) {
                meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
            } else {
                meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = meta_buf_ctx->bufs[i].get();
                }
            }
        }
        if (!meta_buf_ctx->bufs[i]) {
            // allocation failed (callback refusal or device OOM) — fail like the non-meta
            // allocator does so the caller can surface a load error instead of crashing
            ggml_backend_buffer_free(meta_buf); // frees the already-installed simple buffers
            return nullptr;
        }
        if (alloc != nullptr) {
            // finalize any views the callback left untouched (it only owes the non-view tensors);
            // a callback that finalized its own views is fine — those are skipped by the null check
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                if (t->view_src != nullptr && t->buffer == nullptr) {
                    ggml_backend_view_init(t);
                }
            }
        }
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    return ggml_backend_meta_alloc_tensors_ext(buft, tensors, n_tensors, nullptr, nullptr);
}

ggml_backend_buffer_t ggml_backend_meta_alloc_ctx_tensors_from_buft_ext(
        ggml_context * ctx, ggml_backend_buffer_type_t buft, ggml_backend_meta_alloc_simple_t alloc, void * userdata) {
    std::vector<ggml_tensor *> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        tensors.push_back(t);
    }
    return ggml_backend_meta_alloc_tensors_ext(buft, tensors.data(), int(tensors.size()), alloc, userdata);
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

// One long-lived worker thread per extra device. Spawning 7 std::threads per step launch (or per uncaptured
// compute) costs ≈0.6 ms of skew between the first and last device's launch, which every first all-reduce of
// the step then absorbs; the pool wakes all workers with one notification. Workers sleep between jobs.
struct ggml_backend_meta_thread_pool {
    std::vector<std::thread>             threads;
    std::mutex                           m;
    std::condition_variable              cv;
    const std::function<void(size_t)> * job = nullptr;
    std::atomic<size_t>                  generation{0};
    std::atomic<size_t>                  done{0};
    std::atomic<bool>                    stop{false};

    ~ggml_backend_meta_thread_pool() {
        {
            std::lock_guard<std::mutex> lock(m);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }

    // runs fn(j) for j = 1..n-1 on the workers and fn(0) on the caller; returns when all are done
    void run(size_t n, const std::function<void(size_t)> & fn) {
        while (threads.size() + 1 < n) {
            const size_t j = threads.size() + 1;
            threads.emplace_back([this, j]() {
                size_t seen = 0;
                for (;;) {
                    if (!(stop.load() || generation.load(std::memory_order_acquire) != seen)) {
                        std::unique_lock<std::mutex> lock(m);
                        cv.wait(lock, [&] { return stop.load() || generation.load() != seen; });
                    }
                    if (stop) {
                        return;
                    }
                    seen = generation.load(std::memory_order_acquire);
                    (*job)(j);
                    {
                        std::lock_guard<std::mutex> lock(m);
                        done.fetch_add(1, std::memory_order_release);
                    }
                    cv.notify_all();
                }
            });
        }
        {
            std::lock_guard<std::mutex> lock(m);
            job = &fn;
            done.store(0);
            generation.fetch_add(1, std::memory_order_release);
        }
        cv.notify_all();
        fn(0);
        if (done.load(std::memory_order_acquire) != n - 1) {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return done.load() == n - 1; });
        }
    }
};

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
    // optional per-rank enqueue (one issuer thread per device); returns false when the message needs the collective.
    // `next` is the rank's subgraph that follows the reduce (the backend may prefetch its weights); the proc name
    // carries _v2 since `next` was added, so a core and a backend of mismatched versions fall back to the collective.
    typedef bool (*comm_allreduce_rank_t)(void * comm_ctx, struct ggml_tensor ** tensors, int rank, const struct ggml_cgraph * next);
    comm_allreduce_rank_t comm_allreduce_rank = nullptr;

    // whole-step capture (see ggml_backend_step_*_t): one recorded device graph per backend per
    // ggml graph uid, replayed with a single launch per device instead of a host-driven fan-out
    // of every subgraph and all-reduce
    ggml_backend_step_capturable_t    step_capturable    = nullptr;
    ggml_backend_step_epoch_t         step_epoch         = nullptr;
    ggml_backend_step_wait_uploads_t  step_wait_uploads  = nullptr;
    ggml_backend_step_capture_begin_t step_capture_begin = nullptr;
    ggml_backend_step_capture_end_t   step_capture_end   = nullptr;
    ggml_backend_step_launch_t        step_launch        = nullptr;
    ggml_backend_step_free_t          step_free          = nullptr;
    struct step_record {
        uint64_t              sig = 0;            // structural signature of the graph (0 = free slot)
        bool                  valid = false;      // false = this shape is known not to capture; don't retry
        std::vector<void *>   steps;              // one per backend
        std::vector<uint64_t> epochs;             // per backend, at capture
        int64_t               last_used = 0;
        size_t                nodes = 0;          // original GGML graph size, for retention budgeting
    };
    std::vector<step_record> step_records;
    ggml_backend_meta_thread_pool pool; // per-device launch/issue workers
    uint64_t                 step_last_sig    = 0;
    int                      step_same_sig    = 0;   // consecutive computes with the same signature (warmup)
    // sightings per signature over the whole run: a decode-class shape that recurs non-consecutively (the
    // server's 4-token checkpoint tail after every prompt) is recorded on its second sighting
    std::unordered_map<uint64_t, int> step_sightings;
    // signature per graph uid: a graph the caller reuses unchanged keeps its uid, and the per-layer split graphs
    // alternate, so a single cached entry was rehashed on every call (~70 us per decode split)
    std::unordered_map<uint64_t, uint64_t> step_sig_by_uid;
    size_t                   cur_min_reduce   = 0;   // smallest all-reduce of the last rebuilt graph (bytes)
    bool                     step_graphs      = false;

    void step_record_free(step_record & rec) {
        for (size_t j = 0; j < rec.steps.size(); j++) {
            if (rec.steps[j] != nullptr) {
                step_free(backend_configs[j].backend, rec.steps[j]);
                rec.steps[j] = nullptr;
            }
        }
    }

    void step_records_free() {
        for (auto & rec : step_records) {
            step_record_free(rec);
        }
        step_records.clear();
    }

    void step_record_add(step_record rec) {
        // Bound successful and failed captures alike. Invalid overrides keep
        // the default; zero must not try to evict from an empty cache.
        static const size_t records_override = []() -> size_t {
            if (const char * value = getenv("GGML_META_STEP_RECORDS")) {
                size_t parsed = 0;
                const char * end = value + strlen(value);
                const auto result = std::from_chars(value, end, parsed);
                if (result.ec == std::errc{} && result.ptr == end && parsed > 0) {
                    return parsed;
                }
                GGML_LOG_WARN("invalid GGML_META_STEP_RECORDS value '%s'; using automatic budget\n", value);
            }
            return 0;
        }();
        size_t nodes = rec.nodes;
        for (const auto & existing : step_records) {
            nodes += existing.nodes;
        }
        while (ggml_backend_meta_step_cache_over_budget(step_records.size() + 1, nodes, records_override)) {
            auto oldest = std::min_element(step_records.begin(), step_records.end(),
                    [](const step_record & a, const step_record & b) { return a.last_used < b.last_used; });
            step_record_free(*oldest);
            nodes -= oldest->nodes;
            step_records.erase(oldest);
        }
        rec.last_used = ggml_time_us();
        step_records.push_back(std::move(rec));
    }

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
            comm_allreduce_rank = (comm_allreduce_rank_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor_rank_v2");

            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
            step_capturable    = (ggml_backend_step_capturable_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_step_capturable");
            step_epoch         = (ggml_backend_step_epoch_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_step_epoch");
            step_wait_uploads  = (ggml_backend_step_wait_uploads_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_step_wait_uploads");
            step_capture_begin = (ggml_backend_step_capture_begin_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_step_capture_begin");
            step_capture_end   = (ggml_backend_step_capture_end_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_step_capture_end");
            step_launch        = (ggml_backend_step_launch_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_step_launch");
            step_free          = (ggml_backend_step_free_t)          ggml_backend_reg_get_proc_address(reg, "ggml_backend_step_free");
            step_graphs = step_capturable && step_epoch && step_wait_uploads && step_capture_begin &&
                          step_capture_end && step_launch && step_free && getenv("GGML_META_NO_STEP_GRAPHS") == nullptr;
        }
    }

    ~ggml_backend_meta_context() {
        step_records_free();
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

// An async get/set of a split tensor as one strided copy per device. Pieces that need host staging
// cannot outlive this call, so when any device has uneven pieces nothing is issued and the caller
// synchronizes and takes the sync path instead.
static bool ggml_backend_meta_spans_async(ggml_backend_t backend, const ggml_tensor * tensor,
        const ggml_backend_meta_split_state & split, const void * upload, void * download, size_t offset, size_t size) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    std::vector<std::vector<ggml_backend_meta_span>> spans(n_backends);
    std::vector<size_t> strides(n_backends);
    for (size_t j = 0; j < n_backends; j++) {
        spans[j] = ggml_backend_meta_spans(tensor, split, n_backends, j, offset, size);
        if (!spans[j].empty() && (strides[j] = ggml_backend_meta_spans_stride(spans[j])) == 0) {
            return false;
        }
    }
    for (size_t j = 0; j < n_backends; j++) {
        if (spans[j].empty()) {
            continue;
        }
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
        const ggml_backend_meta_span & s = spans[j][0];
        if (upload) {
            ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) upload + s.full - offset, s.local,
                s.size, spans[j].size(), s.size, strides[j]);
        } else {
            ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) download + s.full - offset, s.local,
                s.size, spans[j].size(), s.size, strides[j]);
        }
    }
    return true;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    const auto split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            if (!ggml_backend_meta_spans_async(backend, tensor, split_state, data, nullptr, offset, size)) {
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_set(tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const auto split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            if (!ggml_backend_meta_spans_async(backend, tensor, split_state, nullptr, data, offset, size)) {
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_get(tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

// Structural signature of a graph: everything a recorded step graph depends on (ops, types, shapes, strides,
// data addresses, op params, flags, view and source addresses). Every graph build gets a fresh uid even when
// nothing changed (a server re-batching its slots, a prompt chunk between decode steps), so recordings are
// keyed by this instead and a recurring shape replays without a re-split or a new capture.
static uint64_t ggml_backend_meta_graph_signature(const ggml_cgraph * cgraph) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void * p, size_t n) {
        // 8 bytes per round (7250 nodes carry ~1.5 MB of fields; a byte-wise hash cost ~5 ms per graph)
        const uint8_t * b = (const uint8_t *) p;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint64_t w;
            memcpy(&w, b + i, 8);
            h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
            h ^= h >> 29;
        }
        for (; i < n; i++) {
            h = (h ^ b[i]) * 1099511628211ULL;
        }
    };
    // Static meta tensors share a placeholder data address, even within the
    // same buffer. Hash their actual shard addresses: buffer bases alone let
    // identical-shaped graphs from different layers replay the wrong weights.
    // This also distinguishes views and scheduler buffer reallocations.
    auto mix_storage = [&](const ggml_tensor * t) {
        if (t != nullptr && ggml_backend_buffer_is_meta(t->buffer)) {
            auto * buf_ctx = (ggml_backend_meta_buffer_context *) t->buffer->context;
            const std::vector<ggml_tensor *> * shards = buf_ctx->find_simple_tensors(t);
            GGML_ASSERT(shards != nullptr);
            for (const ggml_tensor * shard : *shards) {
                mix(&shard->data, sizeof(shard->data));
            }
        } else {
            const void * data = t != nullptr ? t->data : nullptr;
            mix(&data, sizeof(data));
        }
    };
    auto mix_descriptor = [&](const ggml_tensor * t) {
        const int32_t type = t->type;
        const int32_t op   = t->op;
        mix_storage(t);
        mix(&type, sizeof(type));
        mix(&op, sizeof(op));
        mix(t->ne, sizeof(t->ne));
        mix(t->nb, sizeof(t->nb));
        mix(t->op_params, sizeof(t->op_params));
        mix(&t->flags, sizeof(t->flags));
        mix_storage(t->view_src);
        mix(&t->view_offs, sizeof(t->view_offs));
    };
    auto mix_tensor = [&](const ggml_tensor * t) {
        mix_descriptor(t);
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            // Scheduler subgraphs need not list their inputs as leaves. A
            // weight staging address can recur with a different quantization
            // type or layout; addresses alone must not select its old kernel.
            if (t->src[k]) {
                mix_descriptor(t->src[k]);
            } else {
                mix_storage(nullptr);
            }
        }
    };
    mix(&cgraph->n_nodes, sizeof(cgraph->n_nodes));
    mix(&cgraph->n_leafs, sizeof(cgraph->n_leafs));
    for (int i = 0; i < cgraph->n_nodes; i++) {
        mix_tensor(cgraph->nodes[i]);
    }
    for (int i = 0; i < cgraph->n_leafs; i++) {
        mix_tensor(cgraph->leafs[i]);
    }
    return h == 0 ? 1 : h;
}

static uint64_t ggml_backend_meta_step_sig(ggml_backend_meta_context * backend_ctx, const ggml_cgraph * cgraph) {
    auto & cache = backend_ctx->step_sig_by_uid;
    auto it = cache.find(cgraph->uid);
    if (it != cache.end()) {
        return it->second;
    }
    if (cache.size() > 4096) {
        cache.clear(); // uids of graphs that were split again are never seen anymore
    }
    const uint64_t sig = ggml_backend_meta_graph_signature(cgraph);
    cache.emplace(cgraph->uid, sig);
    return sig;
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // A CPU split can leave metadata-only views in this graph, including views
    // of computed embeddings. They have no per-device shards or GPU work.
    const auto is_host_view = [](const ggml_tensor * node) {
        return node->view_src != nullptr && ggml_op_is_empty(node->op) &&
            ggml_backend_buffer_is_host(node->buffer);
    };
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // Launching a large executable graph costs on the order of a millisecond, and every device waits at its
    // first all-reduce for the last device launched, so fan the launches out over one thread per device (the
    // CUDA driver is thread-safe per device/stream). Used for the recording step's own launch as well: a
    // sequential first launch was observed to start device j only after device j-1's stream had drained.
    auto launch_steps = [&](const std::vector<void *> & steps) -> bool {
        std::vector<char> launched_j(n_backends, 1);
        for (size_t j = 0; j < n_backends; j++) {
            backend_ctx->step_wait_uploads(backend_ctx->backend_configs[j].backend);
        }
        auto launch_one = [&](size_t j) {
            launched_j[j] = backend_ctx->step_launch(backend_ctx->backend_configs[j].backend, steps[j]);
        };
        // launch on the caller: step replays cost less than waking the pool, most of all on a busy cpu
        // (GGML_META_SEQ_LAUNCH=0: launch on the pool)
        static const bool seq_launch = [] { const char * env = getenv("GGML_META_SEQ_LAUNCH"); return !env || atoi(env) != 0; }();
        if (n_backends > 2 && !seq_launch) {
            backend_ctx->pool.run(n_backends, launch_one);
        } else {
            for (size_t j = 0; j < n_backends; j++) {
                launch_one(j);
            }
        }
        size_t n_launched = 0;
        for (size_t j = 0; j < n_backends; j++) {
            n_launched += launched_j[j] != 0;
        }
        if (n_launched != 0 && n_launched != n_backends) {
            // Submitted graphs may already have updated recurrent state or be
            // waiting for peers. Neither retrying nor synchronizing is safe.
            GGML_ABORT("meta step graph launch failed after %zu/%zu devices submitted; cannot retry", n_launched, n_backends);
        }
        return n_launched == n_backends;
    };

    const bool step_graphs = backend_ctx->step_graphs && n_backends > 1 && backend_ctx->comm_ctx != nullptr && cgraph->uid != 0;
    uint64_t step_sig = 0;
    if (step_graphs) {
        step_sig = ggml_backend_meta_step_sig(backend_ctx, cgraph);
    }
    if (step_graphs) {
        // replay a recorded step for this shape if its device state has not moved — before any re-split,
        // which a recurring shape does not need
        for (auto & rec : backend_ctx->step_records) {
            if (rec.sig != step_sig) {
                continue;
            }
            if (!rec.valid) {
                break; // known not to capture: fall through to the normal path
            }
            bool stale = false;
            for (size_t j = 0; j < n_backends && !stale; j++) {
                stale = backend_ctx->step_epoch(backend_ctx->backend_configs[j].backend) != rec.epochs[j];
            }
            if (stale) {
                backend_ctx->step_record_free(rec);
                rec.sig = 0;
                break;
            }
            if (launch_steps(rec.steps)) {
                rec.last_used = ggml_time_us();
                return GGML_STATUS_SUCCESS;
            }
            GGML_LOG_WARN("%s: step graph replay failed for signature %016" PRIx64 ", recording again\n", __func__, step_sig);
            rec.sig = 0;
            break;
        }

        if (step_sig == backend_ctx->step_last_sig) {
            backend_ctx->step_same_sig++;
        } else {
            backend_ctx->step_last_sig = step_sig;
            backend_ctx->step_same_sig = 0;
        }
        if (backend_ctx->step_sightings.size() > 4096) {
            backend_ctx->step_sightings.clear();
        }
        backend_ctx->step_sightings[step_sig]++;
    }

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;
        size_t min_reduce   = SIZE_MAX;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_host_view(node)) {
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (!is_host_view(next) &&
                                ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (is_host_view(next)) {
                        continue;
                    }
                    if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        const bool compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        const bool compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_host_view(node)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                    min_reduce   = std::min(min_reduce, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            // Trailing host views still belong to the final graph range.
            if (i_start < cgraph->n_nodes) {
                for (size_t j = 0; j < n_backends; j++) {
                    backend_ctx->backend_configs[j].cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = cgraph->n_nodes;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;
        if (getenv("GGML_META_DEBUG") != nullptr) {
            // one all-reduce (and one launch fan-out over every device) per subgraph — the number that
            // decides tensor-split decode latency
            fprintf(stderr, "ggml_backend_meta: graph of %d nodes split into %zu subgraphs (%zu all-reduces) across %zu devices\n",
                    cgraph->n_nodes, n_subgraphs, n_subgraphs > 0 ? n_subgraphs - 1 : 0, n_backends);
        }

        backend_ctx->cur_min_reduce = min_reduce == SIZE_MAX ? 0 : min_reduce;
        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            // every slot up to the tracked maximum, sized for the largest graph seen: the old context is gone,
            // so a slot left over from a previous graph with more subgraphs would otherwise dangle until the
            // next graph of that shape (e.g. a decode graph after a larger prompt graph rebuilt the context)
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            if (ggml_nelements(node) > 0) {
                ggml_tensor * node_zero = get_node_aux(node);
                node_zero->op = GGML_OP_FILL;
                node_zero->src[0] = node; // only used for the shape, the data is not read
                ggml_set_op_params_f32(node_zero, 0, 0.0f);
                node_zero->data = node->data;
                node_zero->buffer = node->buffer;
                node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = node_zero;
                step_cgraphs[j]->n_nodes = 1;
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    // One pass over the subgraphs: launch every device, then all-reduce the boundary node. Under a
    // step capture, a fallback all-reduce (not a plain backend collective) invalidates the capture.
    auto run_subgraphs = [&](bool capturing, bool & capture_ok) -> ggml_status {
        // One host thread issuing every device's launches (7250 nodes x 8 devices per prompt chunk ≈ 58k
        // launches) costs several hundred ms per compute, uncaptured or captured (a recording is the same
        // launches into capturing streams), so each device's subgraph runs from its own thread; every device
        // captures on its own stream, so the recording threads are independent too.
        std::vector<ggml_status> status_j(n_backends, GGML_STATUS_SUCCESS);

        // A backend's compute only waits on the async input uploads of its calling thread, and the inputs
        // were uploaded from this one; a capture already ordered them before it began.
        if (!capturing && backend_ctx->step_wait_uploads != nullptr) {
            for (size_t j = 0; j < n_backends; j++) {
                backend_ctx->step_wait_uploads(backend_ctx->backend_configs[j].backend);
            }
        }

        // Persistent issuer threads:
        // device j's thread enqueues subgraph i and then its own rank's part of reduce i, back to back over all
        // subgraphs — the one-shot reduce synchronizes the ranks on-device, so there is no thread spawn/join per
        // subgraph and a slow issuer no longer stalls the others. A message the per-rank path cannot take (the
        // collective fallback) makes every thread rendezvous; the last to arrive issues it for all ranks.
        if (n_backends > 2 &&
                backend_ctx->comm_ctx != nullptr && backend_ctx->comm_allreduce_rank != nullptr) {
            const size_t n_sub = backend_ctx->n_subgraphs;
            std::vector<std::vector<ggml_tensor *>> boundary(n_sub);
            for (size_t i = 0; i + 1 < n_sub; i++) {
                boundary[i].resize(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_cgraph * cg = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
                    boundary[i][j] = cg->nodes[cg->n_nodes - 1];
                }
            }
            std::mutex              m;
            std::condition_variable cv;
            size_t                  arrived = 0, generation = 0;
            ggml_status             collective_status = GGML_STATUS_SUCCESS;
            std::atomic<bool>       capture_aborted{false};
            // every thread arrives; the last one issues the collective for reduce i while the rest wait
            auto rendezvous = [&](size_t i) {
                std::unique_lock<std::mutex> lock(m);
                const size_t gen = generation;
                if (++arrived == n_backends) {
                    if (!capture_aborted && collective_status == GGML_STATUS_SUCCESS) {
                        const bool ok = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, boundary[i].data());
                        if (!ok) {
                            if (capturing) {
                                capture_aborted = true; // the comm backend cannot record its all-reduce
                            } else {
                                collective_status = allreduce_fallback(i);
                            }
                        }
                    }
                    arrived = 0;
                    generation++;
                    cv.notify_all();
                } else {
                    cv.wait(lock, [&] { return generation != gen; });
                }
            };
            // The threads meet after every subgraph before enqueueing reduce i: a device-synchronizing host call
            // inside a compute (pool growth on a fresh shape) would otherwise wait behind that device's own
            // spinning reduce kernel while the lagging rank it waits for sits in the same call — a deadlock seen
            // as a crawl. The barrier costs microseconds; the savings are the per-subgraph thread spawn/join.
            // spinning barrier: the threads arrive within microseconds of each other 97 times per compute, and a
            // condvar wake-up (~50-100 µs each) measurably slowed long prompts; spin briefly, then yield
            std::atomic<size_t> spin_arrived{0}, spin_generation{0};
            auto barrier_only = [&]() {
                const size_t gen = spin_generation.load(std::memory_order_acquire);
                if (spin_arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == n_backends) {
                    spin_arrived.store(0, std::memory_order_relaxed);
                    spin_generation.store(gen + 1, std::memory_order_release);
                } else {
                    // waits are sub-millisecond; a yield loop here (7 threads × sched_yield storms) slowed the
                    // issuing thread measurably, so spin on the cache line with pause and yield only if stuck
                    for (unsigned spins = 0; spin_generation.load(std::memory_order_acquire) == gen; spins++) {
#if defined(__x86_64__) || defined(__i386__)
                        __builtin_ia32_pause();
#endif
                        if ((spins & 0xFFFFF) == 0xFFFFF) {
                            std::this_thread::yield();
                        }
                    }
                }
            };
            auto worker = [&](size_t j) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < n_sub; i++) {
                    // after a failure or an aborted capture the thread only keeps the rendezvous protocol alive
                    const bool live = status_j[j] == GGML_STATUS_SUCCESS && !capture_aborted;
                    if (live) {
                        status_j[j] = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
                    }
                    barrier_only();
                    if (i + 1 < n_sub) {
                        // the per-rank decision is a pure function of the message, so all threads take the same branch
                        if (!backend_ctx->comm_allreduce_rank(backend_ctx->comm_ctx, boundary[i].data(), (int) j,
                                                              bcj.cgraphs[i + 1].cgraph_main)) {
                            rendezvous(i);
                        }
                    }
                }
            };
            backend_ctx->pool.run(n_backends, worker);
            if (capture_aborted) {
                // no step graph will ever complete for this comm backend; the caller re-runs uncaptured
                backend_ctx->step_graphs = false;
                capture_ok = false;
                return GGML_STATUS_SUCCESS;
            }
            for (size_t j = 0; j < n_backends; j++) {
                if (status_j[j] != GGML_STATUS_SUCCESS) {
                    return status_j[j];
                }
            }
            return collective_status;
        }
        for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
            if (n_backends > 2) {
                std::vector<std::thread> workers;
                workers.reserve(n_backends - 1);
                for (size_t j = 1; j < n_backends; j++) {
                    workers.emplace_back([&, i, j]() {
                        auto & bcj = backend_ctx->backend_configs[j];
                        status_j[j] = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
                    });
                }
                status_j[0] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[0].backend, backend_ctx->backend_configs[0].cgraphs[i].cgraph_main);
                for (auto & w : workers) {
                    w.join();
                }
                for (size_t j = 0; j < n_backends; j++) {
                    if (status_j[j] != GGML_STATUS_SUCCESS) {
                        return status_j[j];
                    }
                }
            } else {
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
            }

            if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
                bool backend_allreduce_success = false;
                if (backend_ctx->comm_ctx) {
                    std::vector<ggml_tensor *> nodes;
                    nodes.reserve(n_backends);
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                        nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                    }
                    backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
                }

                if (!backend_allreduce_success) {
                    if (capturing) {
                        // the comm backend cannot record its all-reduce: no step graph will ever complete
                        backend_ctx->step_graphs = false;
                        capture_ok = false;
                        return GGML_STATUS_SUCCESS; // abort the capture; the caller re-runs uncaptured
                    }
                    const ggml_status status = allreduce_fallback(i);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    };

    if (step_graphs) {
        bool known_bad = false;
        for (const auto & rec : backend_ctx->step_records) {
            known_bad = known_bad || (rec.sig == step_sig && !rec.valid);
        }

        // Classify by the smallest all-reduce: the large final logits reduction
        // must not classify an otherwise small decode step as prefill.
        constexpr size_t step_max_reduce = 256 * 1024;
        const bool decode_class = backend_ctx->cur_min_reduce <= step_max_reduce;
        // Amortize prefill capture on the third consecutive sighting. Decode
        // shapes capture on their second sighting overall, including checkpoint
        // tails that recur after prompts but not on consecutive computes.
        const bool prompt_capture = !decode_class && backend_ctx->step_same_sig >= 2;
        const bool decode_seen_before = decode_class && backend_ctx->step_sightings[step_sig] >= 2;
        if ((backend_ctx->step_same_sig >= 1 || decode_seen_before) && !known_bad && (decode_class || prompt_capture)) {
            bool capturable = true;
            for (size_t j = 0; j < n_backends && capturable; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_subgraphs && capturable; i++) {
                    capturable = backend_ctx->step_capturable(bcj.backend, bcj.cgraphs[i].cgraph_main);
                }
            }
            if (capturable) {
                size_t n_began = 0;
                for (size_t j = 0; j < n_backends; j++) {
                    backend_ctx->step_wait_uploads(backend_ctx->backend_configs[j].backend);
                    if (!backend_ctx->step_capture_begin(backend_ctx->backend_configs[j].backend)) {
                        break;
                    }
                    n_began++;
                }
                bool capture_ok = n_began == n_backends;
                ggml_status status = GGML_STATUS_SUCCESS;
                if (capture_ok) {
                    status = run_subgraphs(/*capturing =*/ true, capture_ok);
                }
                // ending a capture instantiates and uploads a graph of thousands of nodes: do the devices in parallel
                std::vector<void *> steps(n_backends, nullptr);
                std::vector<std::thread> workers;
                for (size_t j = 1; j < n_began; j++) {
                    workers.emplace_back([&, j]() {
                        steps[j] = backend_ctx->step_capture_end(backend_ctx->backend_configs[j].backend);
                    });
                }
                if (n_began > 0) {
                    steps[0] = backend_ctx->step_capture_end(backend_ctx->backend_configs[0].backend);
                }
                for (auto & w : workers) {
                    w.join();
                }
                for (size_t j = 0; j < n_began; j++) {
                    capture_ok = capture_ok && steps[j] != nullptr;
                }
                ggml_backend_meta_context::step_record rec;
                rec.steps = steps;
                if (status != GGML_STATUS_SUCCESS) {
                    backend_ctx->step_record_free(rec);
                    return status;
                }
                if (capture_ok) {
                    // keep the recent shapes (decode at each KV size, prefill chunks, batches with a slot
                    // missing all alternate in a server); evict the least recently used
                    rec.sig   = step_sig;
                    rec.valid = true;
                    rec.nodes = cgraph->n_nodes;
                    rec.epochs.resize(n_backends);
                    for (size_t j = 0; j < n_backends; j++) {
                        rec.epochs[j] = backend_ctx->step_epoch(backend_ctx->backend_configs[j].backend);
                    }
                    backend_ctx->step_record_add(std::move(rec));
                    if (getenv("GGML_META_DEBUG") != nullptr) {
                        fprintf(stderr, "ggml_backend_meta: recorded step graph for signature %016" PRIx64 " (%zu subgraphs x %zu devices, %zu records, %d GGML nodes)\n",
                                step_sig, backend_ctx->n_subgraphs, n_backends, backend_ctx->step_records.size(), cgraph->n_nodes);
                    }
                    // capture recorded the work without executing it: run this compute via replay
                    if (launch_steps(steps)) {
                        return GGML_STATUS_SUCCESS;
                    }
                    if (getenv("GGML_META_DEBUG") != nullptr) {
                        fprintf(stderr, "ggml_backend_meta: step graph launch FAILED for signature %016" PRIx64 " (running uncaptured from now on)\n", step_sig);
                    }
                    backend_ctx->step_records.back().sig = 0;
                    // fall through: execute uncaptured
                } else {
                    backend_ctx->step_record_free(rec);
                    ggml_backend_meta_context::step_record bad;
                    bad.sig   = step_sig;
                    bad.valid = false;
                    bad.steps.assign(n_backends, nullptr);
                    bad.epochs.assign(n_backends, 0);
                    backend_ctx->step_record_add(std::move(bad));
                    if (getenv("GGML_META_DEBUG") != nullptr) {
                        fprintf(stderr, "ggml_backend_meta: step capture unavailable for signature %016" PRIx64 " (fallback all-reduce or capture error)\n", step_sig);
                    }
                    // the captured attempt executed nothing: run for real below
                }
            }
        }
    }

    bool unused = true;
    return run_subgraphs(/*capturing =*/ false, unused);
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
