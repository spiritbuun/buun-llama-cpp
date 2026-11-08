#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Fork wire namespace: upstream 8 has different op/type IDs. The high bit
// prevents either peer from treating the other's tensors as compatible.
#define RPC_PROTO_MAJOR_VERSION    (0x80 | 8)
#define RPC_PROTO_MINOR_VERSION    0
// HELLO checks major/minor only. An enum/layout change must bump a checked
// version, not merely this informational patch field.
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 107, "GGML_OP_COUNT changed - review fork RPC wire version");
static_assert(GGML_TYPE_COUNT == 93, "GGML_TYPE_COUNT changed - review fork RPC wire version");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
