// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-backend-moe-cache.h"

// Optional MoE expert cache function table, populated by a supporting backend.
struct ggml_moe_cache_api ggml_moe_cache = {};
const struct ggml_type_traits_cpu * (*ggml_moe_cache_cpu_traits)(enum ggml_type type) = nullptr;
void (*ggml_moe_cache_cpu_rows)(enum ggml_type type, int n, float * const * dst,
        const void * w, size_t stride, int64_t rows, const void * const * act, int nr) = nullptr;

void ggml_moe_cache_unregister(const void * owner) {
    if (ggml_moe_cache.owner == owner) {
        ggml_moe_cache = {};
    }
}

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

static void ggml_backend_moe_cache_invalidate_buffer(
        ggml_backend_buffer_t buffer, const void * address, size_t size) {
    if (!ggml_moe_cache.invalidate || !buffer || !address || size == 0 ||
        buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        !ggml_backend_buffer_is_host(buffer)) {
        return;
    }
    ggml_moe_cache.invalidate(address, size);
}

// Brackets scheduler compute-buffer allocations (see ggml_moe_cache_api::scratch_alloc_scope).
struct moe_cache_scratch_scope {
    void * session;

    explicit moe_cache_scratch_scope(void * session)
        : session(ggml_moe_cache.scratch_alloc_scope ? session : nullptr) {
        if (this->session) {
            ggml_moe_cache.scratch_alloc_scope(this->session, 1);
        }
    }

    ~moe_cache_scratch_scope() {
        if (session) {
            ggml_moe_cache.scratch_alloc_scope(session, 0);
        }
    }
};

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    // Host weight buffers can back queued cache-fill jobs.
    if (buffer->iface.get_base) {
        void * base = ggml_backend_buffer_get_base(buffer);
        if (base) {
            ggml_backend_moe_cache_invalidate_buffer(
                    buffer, base, ggml_backend_buffer_get_size(buffer));
        }
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    if (buffer->iface.get_base) {
        ggml_backend_moe_cache_invalidate_buffer(
                buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
    }
    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    if (usage != buffer->usage && buffer->iface.get_base) {
        ggml_backend_moe_cache_invalidate_buffer(
                buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
    }
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        if (buffer->iface.get_base) {
            ggml_backend_moe_cache_invalidate_buffer(
                    buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
        }
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    if (size == 0) {
        return;
    }
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        ggml_backend_moe_cache_invalidate_buffer(
                buf, (const char *) tensor->data + offset, size);
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    if (size == 0) {
        return;
    }
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset,
            (n_copies - 1)*stride_tensor + size);
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset, size);
    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset,
            (n_copies - 1)*stride_tensor + size);
    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset, size);
    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (ggml_backend_buffer_is_host(src_buf)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst_buf)) {
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 96
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    int prefetch_input;
    int prefetch_from; // split that first reserves and starts the weight prefetch, or -1
    int device_output_input;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    bool fresh_alloc;      // the graph about to be computed was allocated, not reused
    int last_graph_slot;   // slot of the last computed graph, -1 before the first
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ggml_backend_event_t fence_events[GGML_SCHED_MAX_BACKENDS]; // work queued before a fresh allocation
    bool fence_pending;    // copies into the freshly allocated graph must wait for fence_events
    ggml_backend_event_t copy_events[GGML_SCHED_MAX_BACKENDS]; // without copy events: a peer copy's source waits on the split backend
    bool can_fence;        // every backend that can have work in flight has a fence event
    ggml_backend_buffer_t input_stage[GGML_SCHED_MAX_COPIES]; // host copies of the user inputs, uploaded asynchronously
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;
    void * moe_cache_session;
    bool moe_cache_scratch_prepared; // scratch_prepare already ran for the allocated graph

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;

    // allocation record of the active slot's graph
    const struct ggml_cgraph * alloc_src_graph;
    uint32_t alloc_generation;
    size_t alloc_high_water[GGML_SCHED_MAX_BACKENDS];

    // parked per-graph state of the inactive slots; entry cur_graph_slot is always empty
    struct ggml_backend_sched_graph_state * graph_slots; // [GGML_SCHED_MAX_GRAPH_SLOTS]
    int cur_graph_slot;
    int galloc_graph_slot; // slot whose graph the galloc planned last
    size_t graph_size;
};

// the fields of ggml_backend_sched that describe one split and allocated graph
struct ggml_backend_sched_graph_state {
    bool is_reset;
    bool is_alloc;
    struct ggml_hash_set hash_set;
    int * hv_tensor_backend_ids;
    struct ggml_tensor ** hv_tensor_copies;
    int * node_backend_ids;
    int * leaf_backend_ids;
    int * prev_node_backend_ids;
    int * prev_leaf_backend_ids;
    struct ggml_cgraph graph;
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;
    int cur_copy; // the input copies the split graph was bound to
    struct ggml_context * ctx;
    char * context_buffer;
    const struct ggml_cgraph * alloc_src_graph;
    uint32_t alloc_generation;
    size_t alloc_high_water[GGML_SCHED_MAX_BACKENDS];
};

static void ggml_backend_sched_swap_graph_state(ggml_backend_sched_t sched, ggml_backend_sched_graph_state & s) {
    std::swap(sched->is_reset,              s.is_reset);
    std::swap(sched->is_alloc,              s.is_alloc);
    std::swap(sched->hash_set,              s.hash_set);
    std::swap(sched->hv_tensor_backend_ids, s.hv_tensor_backend_ids);
    std::swap(sched->hv_tensor_copies,      s.hv_tensor_copies);
    std::swap(sched->node_backend_ids,      s.node_backend_ids);
    std::swap(sched->leaf_backend_ids,      s.leaf_backend_ids);
    std::swap(sched->prev_node_backend_ids, s.prev_node_backend_ids);
    std::swap(sched->prev_leaf_backend_ids, s.prev_leaf_backend_ids);
    std::swap(sched->graph,                 s.graph);
    std::swap(sched->splits,                s.splits);
    std::swap(sched->n_splits,              s.n_splits);
    std::swap(sched->splits_capacity,       s.splits_capacity);
    std::swap(sched->graph_inputs,          s.graph_inputs);
    std::swap(sched->n_graph_inputs,        s.n_graph_inputs);
    std::swap(sched->graph_inputs_capacity, s.graph_inputs_capacity);
    std::swap(sched->cur_copy,              s.cur_copy);
    std::swap(sched->ctx,                  s.ctx);
    std::swap(sched->context_buffer,        s.context_buffer);
    std::swap(sched->alloc_src_graph,       s.alloc_src_graph);
    std::swap(sched->alloc_generation,      s.alloc_generation);
    std::swap(sched->alloc_high_water,      s.alloc_high_water);
}

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// the moe-cache provider lets backend_id read node's host expert weights in place
// (routes also hold VRAM copies of the expert scales riding as src[3]/src[4])
static bool ggml_backend_sched_moe_routed(ggml_backend_sched_t sched, const struct ggml_tensor * node,
        const struct ggml_tensor * src, int backend_id) {
    const bool routed_src = src == node->src[0] || src == node->src[3] || src == node->src[4];
    return node->op == GGML_OP_MUL_MAT_ID && routed_src && sched->moe_cache_session &&
        ggml_moe_cache.route_supported &&
        ggml_moe_cache.route_supported(sched->moe_cache_session, sched->backends[backend_id], node);
}

// backend already assigned to the op producing t, looking through views and reshapes
static int ggml_backend_sched_input_backend_id(ggml_backend_sched_t sched, const struct ggml_tensor * t) {
    for (int depth = 0; t && depth < 8; ++depth) {
        const int id = tensor_backend_id(const_cast<struct ggml_tensor *>(t));
        if (id != -1) {
            return id;
        }
        t = t->view_src ? t->view_src : t->src[0];
    }
    return -1;
}

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // a moe-cache backend can read host experts in place for small batches
                // with several devices, only the one already running the layer's activations
                // may route it: asking registers the route, which pins the tensor to that device
                if (src_backend_id == sched->n_backends - 1) {
                    const int input_id = ggml_backend_sched_input_backend_id(sched, tensor->src[1]);
                    const bool pinned = input_id >= 0 && input_id < src_backend_id;
                    for (int b = pinned ? input_id : 0; b < (pinned ? input_id + 1 : src_backend_id); b++) {
                        if (ggml_backend_sched_moe_routed(sched, tensor, src, b)) {
                            SET_CAUSE(tensor, "1.moe");
                            return b;
                        }
                    }
                }
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    // Prefer the device already running the layer, so each device streams
                    // its own layers' weights over its own link and reads its own cache
                    static const bool offload_local = [] {
                        const char * v = getenv("GGML_SCHED_OFFLOAD_LOCAL");
                        return v && atoi(v) != 0;
                    }();
                    if (offload_local && tensor->op == GGML_OP_MUL_MAT_ID) {
                        const int b = ggml_backend_sched_input_backend_id(sched, tensor->src[1]);
                        if (b >= 0 && b < src_backend_id &&
                                ggml_backend_supports_op(sched->backends[b], tensor) &&
                                ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.offl");
                            return b;
                        }
                    }
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

// the buffer type the tensor is or will be allocated in, NULL if not known yet
static ggml_backend_buffer_type_t ggml_backend_sched_tensor_buft(ggml_backend_sched_t sched, struct ggml_tensor * t) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (buf) {
        // the tensor is already allocated
        return buf->buft;
    }
    // see if the tensor already has a backend assigned, and use the buffer type of that backend
    int tensor_backend_id = tensor_backend_id(t);
    if (tensor_backend_id == -1 && t->view_src) {
        tensor_backend_id = tensor_backend_id(t->view_src);
    }
    return tensor_backend_id != -1 ? sched->bufts[tensor_backend_id] : NULL;
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_type_t buft = ggml_backend_sched_tensor_buft(sched, t);
    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// the previous split before split i that runs on the same backend, or -1
static int ggml_backend_sched_prev_on_backend(ggml_backend_sched_t sched, int i) {
    for (int j = i - 1; j >= 0; j--) {
        if (sched->splits[j].backend_id == sched->splits[i].backend_id) {
            return j;
        }
    }
    return -1;
}

// the next split after split i that runs on the same backend, or -1
static int ggml_backend_sched_next_on_backend(ggml_backend_sched_t sched, int i) {
    for (int j = i + 1; j < sched->n_splits; j++) {
        if (sched->splits[j].backend_id == sched->splits[i].backend_id) {
            return j;
        }
    }
    return -1;
}

// The splits whose expert weights are prefetched when split i starts: the next few on the same
// backend, so a large upload can hide behind more than one shorter split
// (GGML_SCHED_PREFETCH_DEPTH, default 2). A target can be returned again by a later split; callers
// claim each target once.
static int ggml_backend_sched_prefetch_targets(ggml_backend_sched_t sched, int i, int * targets, int max_targets) {
    static const int depth = [] {
        const char * v = getenv("GGML_SCHED_PREFETCH_DEPTH");
        return v ? std::max(1, atoi(v)) : 2;
    }();
    int n = 0;
    for (int d = 0, j = i; d < depth && (j = ggml_backend_sched_next_on_backend(sched, j)) >= 0; d++) {
        if (sched->splits[j].prefetch_input >= 0 && n < max_targets) targets[n++] = j;
    }
    return n;
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id) &&
                            !ggml_backend_sched_moe_routed(sched, node, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id) &&
                    !ggml_backend_sched_moe_routed(sched, node, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);
        split->graph.pipelined = sched->n_copies > 1;

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
        split->prefetch_input = -1;
        split->prefetch_from = -1;
        split->device_output_input = -1;
        // the weight copy is prefetched while the previous split on the same backend computes
        const int prev_on_backend = ggml_backend_sched_prev_on_backend(sched, i);
        if (prev_on_backend >= 0 && !sched->callback_eval && sched->n_copies == 1 && sched->moe_cache_session &&
                ggml_moe_cache.prefetch_supported && ggml_moe_cache.prefetch_begin && ggml_moe_cache.prefetch_end &&
                split->graph.n_nodes > 0) {
            const auto * node = split->graph.nodes[0];
            if (node->op == GGML_OP_MUL_MAT_ID && node->src[2]->ne[1] >= 2048) {
                for (int j = 0; j < split->n_inputs; ++j) {
                    auto * source = split->inputs[j];
                    if (node->src[0] == tensor_copy(source, split->backend_id, sched->cur_copy) &&
                            ggml_moe_cache.prefetch_supported(sched->backends[split->backend_id], source)) {
                        split->prefetch_input = j;
                        break;
                    }
                }
            }
        }
        if (i > 0 && !sched->callback_eval && sched->n_copies == 1 && sched->moe_cache_session &&
                ggml_moe_cache.output_supported && ggml_moe_cache.output_bind && ggml_moe_cache.output_copy) {
            const auto & previous = sched->splits[i - 1];
            auto * producer_backend = sched->backends[previous.backend_id];
            if (previous.graph.n_nodes > 0 && producer_backend->iface.synchronize == nullptr &&
                    ggml_backend_dev_type(producer_backend->device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                // Collection identifies this output by its row addresses. More
                // than one full FFN in a split could recycle those addresses
                // before the terminal producer runs; keep that split on host.
                const auto projections = std::count_if(previous.graph.nodes,
                        previous.graph.nodes + previous.graph.n_nodes,
                        [](const ggml_tensor * node) { return node->op == GGML_OP_MUL_MAT_ID; });
                auto * source = previous.graph.nodes[previous.graph.n_nodes - 1];
                if ((projections == 3 || projections == 1) &&
                        !(source->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) &&
                        ggml_node_get_use_count(&previous.graph, previous.graph.n_nodes - 1) == 1) {
                    for (int j = 0; j < split->n_inputs; ++j) {
                        if (split->inputs[j] == source &&
                                ggml_moe_cache.output_supported(sched->backends[split->backend_id], source)) {
                            split->device_output_input = j;
                            break;
                        }
                    }
                }
            }
        }
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        auto add_input = [&](ggml_backend_sched_split * owner, int j) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = owner->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, owner->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = owner->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        };
        for (int j = 0; j < split->n_inputs; j++) {
            if (j != split->prefetch_input && j != split->device_output_input) add_input(split, j);
        }
        // Advance both dependencies, not just the upload. This keeps the next
        // weight copies from aliasing anything the current GPU split still uses.
        int targets[64];
        const int n_targets = ggml_backend_sched_prefetch_targets(sched, i, targets, (int) (sizeof(targets) / sizeof(targets[0])));
        for (int t = 0; t < n_targets; t++) {
            auto * next = &sched->splits[targets[t]];
            if (next->prefetch_from < 0) {
                next->prefetch_from = i;
                add_input(next, next->prefetch_input);
            }
        }
        // Cached hits may write this copy while the CPU computes its misses.
        // Reserve it now so it cannot alias storage still live in that split.
        if (i + 1 < sched->n_splits && sched->splits[i + 1].device_output_input >= 0) {
            auto * next = &sched->splits[i + 1];
            add_input(next, next->device_output_input);
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

// Tells the moe-cache how much of each compute buffer the allocated graph uses, once per evaluation:
// before the inputs are set on a fresh allocation, and before a reused graph runs.
// Every slot still holding an allocated graph keeps its part of the buffer.
static void ggml_backend_sched_scratch_prepare(ggml_backend_sched_t sched) {
    if (!sched->moe_cache_session || !ggml_moe_cache.scratch_prepare || sched->moe_cache_scratch_prepared) {
        return;
    }
    const uint32_t generation = ggml_gallocr_get_generation(sched->galloc);
    for (int b = 0; b < sched->n_backends; b++) {
        size_t plan_high_water = 0; // the last planned graph's, not the max over the slots
        ggml_backend_buffer_t buffer = ggml_gallocr_get_plan_buffer(sched->galloc, b, &plan_high_water);
        if (buffer) {
            size_t high_water = sched->alloc_high_water[b];
            if (sched->graph_slots) {
                for (int s = 0; s < GGML_SCHED_MAX_GRAPH_SLOTS; s++) {
                    const ggml_backend_sched_graph_state & state = sched->graph_slots[s];
                    if (state.is_alloc && state.alloc_generation == generation) {
                        high_water = std::max(high_water, state.alloc_high_water[b]);
                    }
                }
            }
            ggml_moe_cache.scratch_prepare(sched->moe_cache_session, sched->backends[b], buffer, high_water);
        }
    }
    sched->moe_cache_scratch_prepared = true;
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    // the galloc plan reused by index only matches graphs of the slot that made it
    bool backend_ids_changed = sched->galloc_graph_slot != sched->cur_graph_slot;
    sched->galloc_graph_slot = sched->cur_graph_slot;
    for (int i = 0; !backend_ids_changed && i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    moe_cache_scratch_scope scratch_scope(sched->moe_cache_session);

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // a new layout in the same buffers is ordered like any freshly allocated graph, but a buffer
        // re-allocation moves memory that graphs in flight still use, so wait for them first (without
        // ggml_backend_sched_synchronize to avoid changing cur_copy)
        if (!ggml_gallocr_reserve_n_in_place(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            for (int i = 0; i < sched->n_backends; i++) {
                ggml_backend_synchronize(sched->backends[i]);
            }
            if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
                GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
                return false;
            }
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    sched->alloc_generation = ggml_gallocr_get_generation(sched->galloc);
    for (int b = 0; b < sched->n_backends; b++) {
        sched->alloc_high_water[b] = 0;
        ggml_gallocr_get_plan_buffer(sched->galloc, b, &sched->alloc_high_water[b]);
    }

    sched->moe_cache_scratch_prepared = false;
    ggml_backend_sched_scratch_prepare(sched);

    return true;
}

struct ggml_backend_sched_staged_input {
    const ggml_tensor * tensor;
    size_t              offset; // in the copy's input_stage buffer
};

// lay out the host user inputs of the splits whose backends can upload asynchronously in one staging
// buffer and return its size; the graph does not have to be allocated yet
static size_t ggml_backend_sched_stage_layout(ggml_backend_sched_t sched,
        std::vector<ggml_backend_sched_staged_input> & staged, ggml_backend_buffer_type_t & host_buft) {
    size_t size = 0;
    for (int s = 0; s < sched->n_splits; s++) {
        const struct ggml_backend_sched_split * split = &sched->splits[s];
        ggml_backend_t backend = sched->backends[split->backend_id];
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(backend->device);
        if (sched->events[split->backend_id][sched->cur_copy] == NULL || backend->iface.set_tensor_async == NULL ||
            buft == NULL || (host_buft != NULL && buft != host_buft)) {
            continue;
        }
        if (host_buft == NULL) {
            host_buft = buft;
        }
        const size_t align = ggml_backend_buft_get_alignment(host_buft);
        for (int i = 0; i < split->n_inputs; i++) {
            const ggml_tensor * input = split->inputs[i];
            // views of user inputs (e.g. a reshaped mask) hold user data too, but lack the flag
            const ggml_tensor * owner = input->view_src ? input->view_src : input;
            ggml_backend_buffer_type_t input_buft = ggml_backend_sched_tensor_buft(sched, split->inputs[i]);
            if (!(owner->flags & GGML_TENSOR_FLAG_INPUT) || input_buft == NULL || !ggml_backend_buft_is_host(input_buft) ||
                std::any_of(staged.begin(), staged.end(), [&](const auto & e) { return e.tensor == input; })) {
                continue;
            }
            staged.push_back({ input, size });
            size += GGML_PAD(ggml_nbytes(input), align);
        }
    }
    return size;
}

// a staging buffer only grows, NULL if that failed. Freeing or allocating pinned host memory waits for
// every device, which drains the pipeline, so reserve grows the buffers of all copies to the worst case
// up front; a failure there is retried when a graph is computed
static ggml_backend_buffer_t ggml_backend_sched_stage_grow(ggml_backend_sched_t sched, int copy,
        ggml_backend_buffer_type_t host_buft, size_t size) {
    ggml_backend_buffer_t & buf = sched->input_stage[copy];
    if (buf == NULL || ggml_backend_buffer_get_size(buf) < size) {
        ggml_backend_buffer_free(buf);
        buf = ggml_backend_buft_alloc_buffer(host_buft, size);
    }
    return buf;
}

// copy the host user inputs of the splits whose backends can upload asynchronously into this copy's
// staging buffer, which stays untouched until this copy's events are waited for again; empty: copy
// the inputs synchronously
static std::vector<ggml_backend_sched_staged_input> ggml_backend_sched_stage_inputs(ggml_backend_sched_t sched) {
    std::vector<ggml_backend_sched_staged_input> staged;
    ggml_backend_buffer_type_t host_buft = NULL;
    const size_t size = ggml_backend_sched_stage_layout(sched, staged, host_buft);
    if (staged.empty()) {
        return staged;
    }
    ggml_backend_buffer_t buf = ggml_backend_sched_stage_grow(sched, sched->cur_copy, host_buft, size);
    if (buf == NULL) {
        staged.clear();
        return staged;
    }
    char * base = (char *) ggml_backend_buffer_get_base(buf);
    for (const auto & e : staged) {
        memcpy(base + e.offset, e.tensor->data, ggml_nbytes(e.tensor));
    }
    return staged;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    struct moe_cache_scope {
        void * session;
        void (*leave)(void *);

        explicit moe_cache_scope(void * session)
            : session(session), leave(ggml_moe_cache.session_leave) {
            auto enter = ggml_moe_cache.session_enter;
            if (enter && leave) {
                enter(session);
            } else {
                leave = nullptr;
            }
        }

        ~moe_cache_scope() {
            if (leave) {
                leave(session);
            }
        }
    } cache_scope(sched->moe_cache_session);

    // weight lookahead jobs, keyed by the split that consumes them
    struct prefetch_scope {
        std::vector<void *> job;
        int (*end)(void *) = ggml_moe_cache.prefetch_end;
        int finish(int split_id) {
            void * pending = job[split_id];
            job[split_id] = nullptr;
            return pending ? end(pending) : 0;
        }
        ~prefetch_scope() {
            for (size_t i = 0; i < job.size(); i++) finish((int) i);
        }
    } prefetch;
    prefetch.job.resize(sched->n_splits);

    std::vector<int32_t> ids;
    const ggml_tensor * ids_route = nullptr; // the route ids holds
    std::vector<ggml_bitset_t> used_ids;
    // routes already downloaded in this graph: expert-parallel stages revisit each device's route
    struct collected_route {
        const ggml_tensor * route;
        int64_t n_expert;
        int32_t lo, n_local;
        std::vector<ggml_bitset_t> used;
    };
    std::vector<collected_route> collected;

    // One route can feed up, gate and down projections. Share its host bitmap
    // between ordinary copies and lookahead rather than downloading it again.
    const auto collect_experts = [&](ggml_backend_t backend, ggml_tensor * route, int64_t n_expert,
                                     const ggml_tensor * node) {
        const int32_t lo = ggml_mmid_window_lo(node);
        const int32_t n_local = ggml_mmid_window_n_local(node);
        for (const auto & c : collected) {
            if (c.route == route && c.n_expert == n_expert && c.lo == lo && c.n_local == n_local) {
                used_ids = c.used;
                return;
            }
        }
        // expert-parallel parts read one route through different windows: download it once
        if (route != ids_route) {
            ids.resize(ggml_nbytes(route) / sizeof(int32_t));
            ggml_backend_tensor_get_async(backend, route, ids.data(), 0, ggml_nbytes(route));
            ggml_backend_synchronize(backend);
            ids_route = route;
        }
        used_ids.assign(ggml_bitset_size(n_expert), 0);
        for (int64_t row = 0; row < route->ne[1]; ++row) {
            for (int64_t col = 0; col < route->ne[0]; ++col) {
                const int32_t raw_id = ids[row * route->nb[1]/sizeof(int32_t) + col * route->nb[0]/sizeof(int32_t)];
                const int32_t id = ggml_mmid_expert_index(raw_id, lo, n_local);
                // Windowed banks zero absent experts instead of reading weights.
                if (n_local != 0 && id < 0) continue;
                GGML_ASSERT(id >= 0 && id < n_expert);
                ggml_bitset_set(used_ids.data(), id);
            }
        }
        if (collected.size() >= 16) collected.clear();
        collected.push_back({route, n_expert, lo, n_local, used_ids});
    };

    // user inputs are copied synchronously into this copy's buffers, which are free once the graph that last
    // used this copy finished on every backend. waiting for that once, up front, keeps a later split's input
    // copy from waiting on an earlier split of this graph (the events are re-recorded per split), which would
    // block the host and serialize pipeline parallelism
    for (int b = 0; b < sched->n_backends; b++) {
        if (sched->events[b][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[b][sched->cur_copy]);
        }
    }

    // a synchronous upload would wait for all the work queued on its backend, which with pipeline parallelism
    // includes earlier splits of this graph on other devices, serializing the pipeline
    const auto staged = ggml_backend_sched_stage_inputs(sched);

    int prev_backend_id = -1;
    const bool fence = sched->fence_pending;
    sched->fence_pending = false;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        bool host_reads[GGML_SCHED_MAX_BACKENDS] = {};
        const bool prefetched = prefetch.finish(split_id) != 0;

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            // a device-side wait orders the GPU work without blocking the host, so later splits and
            // ubatches can still be queued ahead (pipeline parallelism)
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL && split_backend->iface.event_wait != NULL) {
                ggml_backend_event_wait(split_backend, sched->events[prev_backend_id][sched->cur_copy]);
            } else if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // without input copy events, the split backend has to be idle before its inputs are
        // overwritten; once is enough, the copies below do not use the previous inputs
        bool split_backend_idle = false;
        auto sync_split_backend = [&]() {
            if (!split_backend_idle) {
                ggml_backend_synchronize(split_backend);
                split_backend_idle = true;
            }
        };

        // copy the input tensors to the split backend, host sources first: their copies are
        // synchronous and would otherwise wait for the device-to-device copies queued here
        // (and with them the whole previous split on the other device) before the split launches
        for (int input_n = 0; input_n < 2*split->n_inputs; input_n++) {
            const int input_id = input_n % split->n_inputs;
            const bool host_src = ggml_backend_buffer_is_host(split->inputs[input_id]->buffer);
            if (host_src != (input_n < split->n_inputs)) continue;
            if (prefetched && input_id == split->prefetch_input) continue;
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            auto stage = std::find_if(staged.begin(), staged.end(), [&](const auto & e) { return e.tensor == input; });
            if (stage != staged.end()) {
                const char * base = (const char *) ggml_backend_buffer_get_base(sched->input_stage[sched->cur_copy]);
                ggml_backend_tensor_set_async(split_backend, input_cpy, base + stage->offset, 0, ggml_nbytes(input));
                continue;
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                // (with events, the destination was already waited for before the first split, unless freshly allocated)
                if (sched->events[split_backend_id][sched->cur_copy] == NULL || fence) {
                    sync_split_backend();
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                const bool expert_copy = split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    );

                // wait for the split backend to finish using the input before overwriting it
                ggml_backend_event_t copy_event = sched->copy_events[split_backend_id];
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else if (expert_copy) {
                    // the expert uploads are queued on the split backend's stream, behind its earlier work
                } else if (copy_event != NULL && input_id != split->device_output_input &&
                           input_backend != split_backend && input_backend->iface.event_wait != NULL &&
                           !ggml_backend_buffer_is_host(input->buffer)) {
                    // a peer copy runs on the source stream: order it after the split backend's work on the device
                    ggml_backend_event_record(copy_event, split_backend);
                    ggml_backend_event_wait(input_backend, copy_event);
                } else {
                    sync_split_backend();
                }

                if (input_id == split->device_output_input) {
                    const int copied = ggml_moe_cache.output_copy(
                            sched->moe_cache_session, split_backend, input, input_cpy);
                    // Deferred collection errors must stop the graph, not copy
                    // host rows that were deliberately never computed.
                    if (copied < 0) return GGML_STATUS_FAILED;
                    if (copied > 0) continue;
                }

                if (expert_copy) {
                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    if (ggml_nelements(ids_tensor) == 0) {
                        continue;
                    }

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    collect_experts(ids_backend, ids_tensor, n_expert, node);

                    if (ggml_moe_cache.prefill_copy && ids_tensor->ne[1] > 16 &&
                            ggml_moe_cache.prefill_copy(sched->moe_cache_session, split_backend,
                                input, input_cpy, used_ids.data(), used_ids.size())) {
                        continue;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (id < n_expert && !ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    if (id == n_expert) {
                        // A zero-output token batch selects no experts to transfer.
                        continue;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // A synchronous CPU split can enqueue all its device inputs on
                    // their producer streams, then wait once per producer. Separate
                    // blocking tensor reads otherwise drain the same GPU repeatedly.
                    ggml_backend_buffer_t src_buf = input->view_src ? input->view_src->buffer : input->buffer;
                    ggml_backend_buffer_t dst_buf = input_cpy->view_src ? input_cpy->view_src->buffer : input_cpy->buffer;
                    if (ggml_backend_dev_type(split_backend->device) == GGML_BACKEND_DEVICE_TYPE_CPU &&
                        split_backend->iface.synchronize == nullptr &&
                        input_backend->iface.get_tensor_async != nullptr &&
                        src_buf->buft == ggml_backend_get_default_buffer_type(input_backend) &&
                        !ggml_backend_buffer_is_host(src_buf) && ggml_backend_buffer_is_host(dst_buf)) {
                        GGML_ASSERT(ggml_are_same_layout(input, input_cpy));
                        ggml_backend_moe_cache_invalidate_buffer(dst_buf, input_cpy->data, ggml_nbytes(input_cpy));
                        ggml_backend_tensor_get_async(input_backend, input, input_cpy->data, 0, ggml_nbytes(input));
                        host_reads[ggml_backend_sched_backend_id(sched, input_backend)] = true;
                        continue;
                    }
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    // the copy runs on the source stream, which must wait for the destination's earlier graphs
                    ggml_backend_event_t dst_fence = fence ? sched->fence_events[split_backend_id] : NULL;
                    if (dst_fence != NULL && input_backend != split_backend) {
                        if (input_backend->iface.event_wait != NULL) {
                            ggml_backend_event_wait(input_backend, dst_fence);
                        } else {
                            sync_split_backend();
                        }
                    }
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (fence || sched->events[split_backend_id][sched->cur_copy] == NULL) {
                            sync_split_backend();
                        } else {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
            if (host_reads[backend_id]) {
                ggml_backend_synchronize(sched->backends[backend_id]);
            }
        }

        if (!sched->callback_eval) {
            if (split_id + 1 < sched->n_splits) {
                auto * next = &splits[split_id + 1];
                if (next->device_output_input >= 0) {
                    auto * destination_backend = sched->backends[next->backend_id];
                    if (!host_reads[next->backend_id]) ggml_backend_synchronize(destination_backend);
                    auto * source = next->inputs[next->device_output_input];
                    ggml_moe_cache.output_bind(sched->moe_cache_session, destination_backend, source,
                            tensor_copy(source, next->backend_id, sched->cur_copy));
                }
            }
            int targets[64];
            const int n_targets =
                ggml_backend_sched_prefetch_targets(sched, split_id, targets, (int) (sizeof(targets) / sizeof(targets[0])));
            for (int t = 0; t < n_targets; t++) {
                const int j = targets[t];
                auto * next = &splits[j];
                if (next->prefetch_from != split_id) continue;
                auto * source = next->inputs[next->prefetch_input];
                const uint32_t * ready = nullptr;
                auto * current = split->graph.n_nodes ? split->graph.nodes[0] : nullptr;
                auto * route = next->graph.nodes[0]->src[2];
                // The current matmul consumes these exact IDs, so they are
                // already ready before its GPU work starts. Never read a
                // router produced by the current split ahead of execution.
                if (current && current->op == GGML_OP_MUL_MAT_ID &&
                        current->src[2] == route && source->ne[2] <= 512 &&
                        route->type == GGML_TYPE_I32 && route->nb[0] == sizeof(int32_t) &&
                        route->ne[2] == 1 && route->ne[3] == 1) {
                    // a route copied into this split reads from its source, which the
                    // other expert-parallel parts share (one download per route)
                    ggml_backend_t route_backend = split_backend;
                    for (int i = 0; i < split->n_inputs; i++) {
                        if (route == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            route = split->inputs[i];
                            route_backend = ggml_backend_sched_get_tensor_backend(sched, route);
                            break;
                        }
                    }
                    collect_experts(route_backend, route, source->ne[2], next->graph.nodes[0]);
                    ready = used_ids.data();
                }
                prefetch.job[j] = ggml_moe_cache.prefetch_begin(sched->moe_cache_session,
                        sched->backends[next->backend_id], source, tensor_copy(source, next->backend_id, sched->cur_copy),
                        ready, ready ? used_ids.size() : 0);
            }
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    return GGML_STATUS_SUCCESS;
}

// allocates the active per-graph state for graphs of up to sched->graph_size nodes
static void ggml_backend_sched_init_graph_state(ggml_backend_sched_t sched) {
    const size_t graph_size = sched->graph_size;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));
}

static void ggml_backend_sched_free_graph_state(ggml_backend_sched_t sched) {
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
    sched->last_graph_slot = -1;
    sched->can_fence = sched->n_copies > 1;
    sched->graph_size = graph_size;

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    // sched->ctx holds the split-input copies plus their dependency views. A copy is made once per
    // (src, backend), so there are at most 2*graph_size*n_backends inputs (nodes + leafs), each with
    // n_copies copies and one view; graph inputs and alloc-dep views add at most one more per node.
    // The old graph_size*GGML_SCHED_MAX_SPLIT_INPUTS*2 bound reached 84 GB at a 1.3M-node reserve.
    const size_t max_ctx_tensors = graph_size*(2*n_backends*(sched->n_copies + 1) + sched->n_copies + 1);
    sched->context_buffer_size = max_ctx_tensors*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false);

    ggml_backend_sched_init_graph_state(sched);

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
            sched->fence_events[b] = ggml_backend_event_new(backends[b]->device);
            // a backend without a synchronize is synchronous and never has work in flight
            sched->can_fence = sched->can_fence && (backends[b]->iface.synchronize == NULL || sched->fence_events[b] != NULL);
        } else if (backends[b]->iface.event_wait != NULL) {
            sched->copy_events[b] = ggml_backend_event_new(backends[b]->device);
        }
    }

    if (ggml_moe_cache.session_create) {
        void * cache_backends[GGML_SCHED_MAX_BACKENDS];
        for (int b = 0; b < n_backends; b++) {
            cache_backends[b] = backends[b];
        }
        sched->moe_cache_session = ggml_moe_cache.session_create(cache_backends, n_backends, nullptr);
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_set_moe_cache(
        ggml_backend_sched_t sched, enum ggml_moe_cache_mode mode,
        size_t budget_mib, int max_devices, int expert_parallel, int cpu_overlap,
        const char * profile_path) {
    GGML_ASSERT(sched);
    GGML_ASSERT(cpu_overlap >= -2 && cpu_overlap <= 8);
    if (mode == GGML_MOE_CACHE_MODE_UNSPECIFIED && cpu_overlap == -2 && max_devices == 0) {
        return;
    }

    if (sched->moe_cache_session && ggml_moe_cache.session_destroy) {
        ggml_moe_cache.session_destroy(sched->moe_cache_session);
    }
    sched->moe_cache_session = nullptr;
    if (mode == GGML_MOE_CACHE_MODE_OFF ||
        !ggml_moe_cache.query_config || !ggml_moe_cache.session_create) {
        return;
    }

    ggml_moe_cache_config config = {};
    const int automatic = mode == GGML_MOE_CACHE_MODE_UNSPECIFIED ? -1 :
        mode == GGML_MOE_CACHE_MODE_AUTO ? 1 : 0;
    if (!ggml_moe_cache.query_config(automatic, budget_mib, &config)) {
        return;
    }
    if (mode != GGML_MOE_CACHE_MODE_UNSPECIFIED) {
        config.expert_parallel = expert_parallel;
        config.profile_path = profile_path;
    }
    if (cpu_overlap != -2) config.overlap_cpu_rows = cpu_overlap;
    config.max_devices = max_devices;

    void * cache_backends[GGML_SCHED_MAX_BACKENDS];
    for (int index = 0; index < sched->n_backends; index++) {
        cache_backends[index] = sched->backends[index];
    }
    sched->moe_cache_session = ggml_moe_cache.session_create(
            cache_backends, sched->n_backends, &config);
}

void ggml_backend_sched_set_moe_cache_warmup(ggml_backend_sched_t sched, bool warmup) {
    if (sched->moe_cache_session && ggml_moe_cache.session_warmup) {
        ggml_moe_cache.session_warmup(sched->moe_cache_session, warmup);
    }
}

bool ggml_backend_sched_has_moe_cache(ggml_backend_sched_t sched) {
    return sched->moe_cache_session != nullptr;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    if (sched->moe_cache_session && ggml_moe_cache.session_destroy) {
        ggml_moe_cache.session_destroy(sched->moe_cache_session);
        sched->moe_cache_session = NULL;
    }
    ggml_backend_meta_forget_graph_owner(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        ggml_backend_event_free(sched->fence_events[b]);
        ggml_backend_event_free(sched->copy_events[b]);
    }
    for (int c = 0; c < GGML_SCHED_MAX_COPIES; c++) {
        ggml_backend_buffer_free(sched->input_stage[c]);
    }
    ggml_gallocr_free(sched->galloc);
    ggml_backend_sched_free_graph_state(sched);
    for (int s = 0; sched->graph_slots && s < GGML_SCHED_MAX_GRAPH_SLOTS; s++) {
        if (sched->graph_slots[s].context_buffer) {
            ggml_backend_sched_swap_graph_state(sched, sched->graph_slots[s]);
            ggml_backend_sched_free_graph_state(sched);
        }
    }
    free(sched->graph_slots);
    free(sched);
}

bool ggml_backend_sched_set_graph_slot(ggml_backend_sched_t sched, int slot, const struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT(slot >= 0 && slot < GGML_SCHED_MAX_GRAPH_SLOTS);
    if (slot != sched->cur_graph_slot) {
        if (!sched->graph_slots) {
            sched->graph_slots = (ggml_backend_sched_graph_state *) calloc(GGML_SCHED_MAX_GRAPH_SLOTS, sizeof(ggml_backend_sched_graph_state));
        }
        // park the active state in its empty entry, then take the target's, leaving that entry empty
        ggml_backend_sched_swap_graph_state(sched, sched->graph_slots[sched->cur_graph_slot]);
        ggml_backend_sched_swap_graph_state(sched, sched->graph_slots[slot]);
        sched->cur_graph_slot = slot;
        if (!sched->context_buffer) {
            ggml_backend_sched_init_graph_state(sched);
            ggml_backend_sched_reset(sched);
        }
    }
    ggml_backend_meta_set_graph_owner(sched, slot, false);
    return sched->is_alloc && graph && sched->alloc_src_graph == graph &&
           sched->alloc_generation == ggml_gallocr_get_generation(sched->galloc);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    // like ggml_backend_sched_reserve, no reset here: the caller resets before building the graph,
    // and resetting now would drop the backends it pinned with ggml_backend_sched_set_tensor_backend
    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    sched->galloc_graph_slot = sched->cur_graph_slot;
    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    moe_cache_scratch_scope scratch_scope(sched->moe_cache_session);
    sched->galloc_graph_slot = sched->cur_graph_slot;
    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    std::vector<ggml_backend_sched_staged_input> staged;
    ggml_backend_buffer_type_t host_buft = NULL;
    const size_t stage_size = ggml_backend_sched_stage_layout(sched, staged, host_buft);
    if (!staged.empty()) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_sched_stage_grow(sched, c, host_buft, stage_size);
        }
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    // tensors initialized from here on belong to this slot's graph, replacing its previous one
    ggml_backend_meta_set_graph_owner(sched, sched->cur_graph_slot, true);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;
    sched->fresh_alloc = true;
    sched->alloc_src_graph = graph;

    return true;
}

// point a reused graph's split inputs at another set of input copies, so it can be queued again
// while earlier runs of it are still in flight
static void ggml_backend_sched_rebind_copy(ggml_backend_sched_t sched, int copy) {
    // a copy is listed only by the first split of its backend that needs it, later splits there use it too
    std::unordered_map<ggml_tensor *, ggml_tensor *> remap;
    for (int s = 0; s < sched->n_splits; s++) {
        const struct ggml_backend_sched_split * split = &sched->splits[s];
        for (int k = 0; k < split->n_inputs; k++) {
            const size_t id = hash_id(split->inputs[k]);
            remap[tensor_id_copy(id, split->backend_id, sched->cur_copy)] = tensor_id_copy(id, split->backend_id, copy);
        }
    }
    for (int s = 0; s < sched->n_splits; s++) {
        const struct ggml_backend_sched_split * split = &sched->splits[s];
        for (int i = 0; i < split->graph.n_nodes; i++) {
            struct ggml_tensor * node = split->graph.nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (node->src[j] == NULL || !(node->src[j]->flags & GGML_TENSOR_FLAG_INPUT)) {
                    continue;
                }
                auto it = remap.find(node->src[j]);
                if (it != remap.end()) {
                    node->src[j] = it->second;
                }
            }
        }
    }
    sched->cur_copy = copy;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    ggml_backend_meta_set_graph_owner(sched, sched->cur_graph_slot, false);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    if (sched->n_copies > 1) {
        if (sched->fresh_alloc && sched->cur_graph_slot == sched->last_graph_slot && sched->can_fence) {
            // another allocation can overlap buffers that graphs still in flight use. work queued on a
            // backend stays ordered behind them; only copies issued on another backend's stream can
            // overtake, so those wait for the work queued here so far instead of draining every pipeline
            for (int b = 0; b < sched->n_backends; b++) {
                if (sched->fence_events[b] != NULL) {
                    ggml_backend_event_record(sched->fence_events[b], sched->backends[b]);
                }
            }
            sched->fence_pending = true;
        } else if (sched->fresh_alloc || sched->cur_graph_slot != sched->last_graph_slot) {
            for (int b = 0; b < sched->n_backends; b++) {
                ggml_backend_synchronize(sched->backends[b]);
            }
        } else {
            ggml_backend_sched_rebind_copy(sched, sched->next_copy);
            sched->next_copy = (sched->next_copy + 1) % sched->n_copies;
        }
        sched->fresh_alloc = false;
        sched->last_graph_slot = sched->cur_graph_slot;
    }

    ggml_backend_sched_scratch_prepare(sched);
    sched->moe_cache_scratch_prepared = false;

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
