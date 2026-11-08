#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

// Expert store reader: a thread pool that reads expert slices from immutable
// weight files (a GGUF, or a repack cache entry) with direct I/O, so a tier
// that streams experts from disk does not grow the page cache. Each read is
// split into pieces, which gives one drive the queue depth it needs.
// (ggml-base cannot reach llama_file, whose direct-I/O path this mirrors.)

#ifdef __cplusplus
extern "C" {
#endif

// Lower values are served first; reads of equal priority are served in order.
enum ggml_moe_store_priority {
    GGML_MOE_STORE_DEMAND  = 0,
    GGML_MOE_STORE_PREFILL = 1,
    GGML_MOE_STORE_HINT    = 2,
    GGML_MOE_STORE_SEED    = 3,
};

struct ggml_moe_store;

// Runs on a reader thread once every piece has landed (ok) or any piece failed.
typedef void (*ggml_moe_store_done_t)(void * user, bool ok);

GGML_API struct ggml_moe_store * ggml_moe_store_init(int n_threads, size_t piece_bytes);
// Waits for the reads already started; queued reads complete as failed.
GGML_API void ggml_moe_store_free(struct ggml_moe_store * store);

// Returns a file id, or -1. Falls back to buffered reads (dropped from the
// page cache after each piece) where the filesystem rejects direct I/O.
GGML_API int ggml_moe_store_open(struct ggml_moe_store * store, const char * path);

// Required alignment of every destination buffer; 4 KiB covers the logical
// block size of every drive direct I/O targets.
#define GGML_MOE_STORE_ALIGN 4096

// Bytes a destination needs to hold `size` bytes at `offset` of a file: the
// direct read covers the enclosing aligned window.
GGML_API size_t ggml_moe_store_span(size_t offset, size_t size);

// Queues a read of [offset, offset + size) into dst, which must be
// GGML_MOE_STORE_ALIGN aligned and hold ggml_moe_store_span(offset, size)
// bytes, and returns where the requested bytes will start within dst. done
// always runs exactly once; when the read is rejected, it runs (not ok) before
// this returns NULL.
GGML_API void * ggml_moe_store_read(
        struct ggml_moe_store * store, int file, uint64_t offset, size_t size,
        void * dst, int priority, ggml_moe_store_done_t done, void * user);

// Registry of read-only file mappings, so a tier can read a mapped tensor's
// bytes from its file instead of faulting them in. map duplicates fd; unmap
// drops any registered bytes in [addr, addr + size).
GGML_API void ggml_moe_store_map(const void * addr, size_t size, int fd);
GGML_API void ggml_moe_store_unmap(const void * addr, size_t size);
// True when [ptr, ptr + size) lies in one registered range: path (re)opens its
// file, file identifies that file for this process's lifetime, and offset is
// ptr's position in it; each may be NULL.
GGML_API bool ggml_moe_store_find(const void * ptr, size_t size, char * path, size_t path_size,
        uint64_t * file, uint64_t * offset);

// Host memory this process could still make unreclaimable, which bounds how much of the
// mapped weights can stay in the page cache: the tightest cgroup v2 limit above it less
// its anonymous, shared and kernel memory, and at most MemAvailable; SIZE_MAX when unknown.
GGML_API size_t ggml_moe_store_host_room(void);

#ifdef __cplusplus
}
#endif
