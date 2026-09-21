// Dynamic VBR (S2, "option C"): CUDA/HIP virtual-memory pool for the KV cache.
//
// One cuMemAddressReserve VA range holds every (layer,side) KV tensor at a FIXED, page-aligned
// offset sized for its MAX tier (F16 x kv_size) — tensor data pointers never move. Physical
// commit chunks are mapped on demand as the write watermark advances and unmapped from a tensor's tail
// after a tier degrade shrinks its byte footprint. Freed pages are fungible across tensors, so
// no relocation/compaction is ever needed. Same-source on ROCm (vendors/hip.h maps cuMem*).
//
// Chunks are tracked at the effective commit granularity. Handles are released immediately
// after mapping (physical is freed by cuMemUnmap), matching ggml_cuda_pool_vmm; per-chunk unmap
// also sidesteps ROCR-Runtime issue #285 (can't unmap one giant range on HIP).

#include "common.cuh"
#include "ggml-cuda.h"
#include "vbr-vmm-policy.h"
#include "ggml-vbr-diagnostic.h"

#include <set>
#include <vector>

#if defined(GGML_USE_VMM)

struct ggml_vbr_vmm_pool {
    int         device;
    CUdeviceptr base    = 0;
    size_t      va_size = 0;
    size_t      gran    = 0;
    uint64_t    residency_epoch = 0;
    std::set<size_t> chunks; // mapped chunk offsets (each gran bytes)
#if defined(GGML_USE_HIP) && defined(__linux__)
    void * mapping_guard = nullptr;
#endif
};

static void vmm_pool_mapping_boundary(ggml_vbr_vmm_pool * pool) {
#if defined(GGML_USE_HIP) && defined(__linux__)
    // ROCr's DRM VMM path can leave stale translations after same-VA remaps. A KFD
    // allocation waits for page-table updates and invalidates the compute TLB.
    // Uncached bypasses ROCr's fragment allocator: an ordinary small hipMalloc
    // can return a suballocation without issuing any kernel mapping operation.
    // Keep a page reserved between boundaries so this also works near VRAM capacity.
    CUDA_CHECK(hipFree(pool->mapping_guard));
    pool->mapping_guard = nullptr;
    CUDA_CHECK(hipExtMallocWithFlags(&pool->mapping_guard, 4096, hipDeviceMallocUncached));
#else
    GGML_UNUSED(pool);
#endif
}

bool ggml_backend_cuda_vmm_available(int device) {
    return device >= 0 && device < ggml_cuda_info().device_count && ggml_cuda_info().devices[device].vmm;
}

size_t ggml_backend_cuda_vmm_granularity(int device) {
    if (!ggml_backend_cuda_vmm_available(device)) {
        return 0;
    }
    return ggml_cuda_vbr_vmm_commit_granularity(
        ggml_cuda_info().devices[device].vmm_granularity,
#if defined(GGML_USE_HIP)
        true
#else
        false
#endif
    );
}

ggml_vbr_vmm_pool * ggml_backend_cuda_vmm_pool_init(int device, size_t va_size) {
    if (!ggml_backend_cuda_vmm_available(device) || va_size == 0) {
        return nullptr;
    }
    auto * pool = new ggml_vbr_vmm_pool;
    pool->device = device;
    pool->gran   = ggml_backend_cuda_vmm_granularity(device);
    pool->va_size = GGML_PAD(va_size, pool->gran);
    CUdeviceptr base = 0;
    if (cuMemAddressReserve(&base, pool->va_size, 0, 0, 0) != CUDA_SUCCESS) {
        delete pool;
        return nullptr;
    }
    pool->base = base;
    ggml_vbr_diag_record(GGML_VBR_DIAG_MAP,
        "pool_init pool=%p device=%d physical=%d base=%llu va_bytes=%zu gran=%zu",
        (void *) pool, device, ggml_cuda_info().devices[device].physical_device,
        (unsigned long long) base, pool->va_size, pool->gran);
#if defined(GGML_USE_HIP) && defined(__linux__)
    ggml_cuda_set_device(device);
    if (hipExtMallocWithFlags(&pool->mapping_guard, 4096, hipDeviceMallocUncached) != hipSuccess) {
        CU_CHECK(cuMemAddressFree(base, pool->va_size));
        delete pool;
        return nullptr;
    }
#endif
    return pool;
}

void * ggml_backend_cuda_vmm_pool_base(ggml_vbr_vmm_pool * pool) {
    return (void *) pool->base;
}

size_t ggml_backend_cuda_vmm_pool_mapped(ggml_vbr_vmm_pool * pool) {
    return pool->chunks.size() * pool->gran;
}

uint64_t ggml_backend_cuda_vmm_pool_residency_epoch(ggml_vbr_vmm_pool * pool) {
    return pool->residency_epoch;
}

size_t ggml_backend_cuda_vmm_pool_mapped_in_range(
        ggml_vbr_vmm_pool * pool, size_t off, size_t len) {
    const size_t g = pool->gran;
    GGML_ASSERT(off % g == 0);
    GGML_ASSERT(len % g == 0);
    GGML_ASSERT(off <= pool->va_size && len <= pool->va_size - off);

    size_t chunks = 0;
    const size_t end = off + len;
    for (auto it = pool->chunks.lower_bound(off); it != pool->chunks.end() && *it < end; ++it) {
        chunks++;
    }
    GGML_ASSERT(chunks <= SIZE_MAX / g);
    return chunks * g;
}

bool ggml_backend_cuda_vmm_pool_map(ggml_vbr_vmm_pool * pool, size_t off, size_t len) {
    if (len == 0) {
        return true;
    }
    GGML_ASSERT(off + len <= pool->va_size);
    ggml_cuda_set_device(pool->device);
    std::vector<size_t> new_chunks;
    const size_t g  = pool->gran;
    const size_t c0 = (off / g) * g;
    const size_t c1 = GGML_PAD(off + len, g);
    if (ggml_vbr_diag_enabled()) {
        ggml_vbr_diag_record(GGML_VBR_DIAG_MAP,
            "map_begin pool=%p device=%d physical=%d base=%llu off=%zu len=%zu gran=%zu mapped=%zu epoch=%llu",
            (void *) pool, pool->device, ggml_cuda_info().devices[pool->device].physical_device,
            (unsigned long long) pool->base, off, len, g, pool->chunks.size() * g,
            (unsigned long long) pool->residency_epoch);
    }
    // Observe the ORIGINAL return value; no retries, error clearing, device
    // queries, or synchronization are introduced around the failing call.
    auto observe = [&](CUresult rc, const char * operation, size_t chunk) {
        if (rc != CUDA_SUCCESS) {
            ggml_vbr_diag_record(GGML_VBR_DIAG_MAP,
                "map_failure operation=%s result=%d pool=%p device=%d physical=%d base=%llu "
                "chunk=%zu gran=%zu request_off=%zu request_len=%zu mapped=%zu new_chunks=%zu epoch=%llu",
                operation, int(rc), (void *) pool, pool->device,
                ggml_cuda_info().devices[pool->device].physical_device, (unsigned long long) pool->base,
                chunk, g, off, len, pool->chunks.size() * g, new_chunks.size(),
                (unsigned long long) pool->residency_epoch);
        }
        return rc;
    };
    auto finish_mapping = [&]() {
        if (new_chunks.empty()) {
            return;
        }
        // Publish mapping updates before even the initialization writes. In particular,
        // a stream/device wait alone does not invalidate stale HIP address translations.
        vmm_pool_mapping_boundary(pool);
        for (size_t c : new_chunks) {
            CUDA_CHECK(cudaMemset((char *) pool->base + c, 0, g));
        }
        // The initialization runs on the legacy stream, while ggml uses non-blocking
        // streams. Also settle a partial allocation before returning it to the caller.
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        GGML_ASSERT(pool->residency_epoch != UINT64_MAX);
        pool->residency_epoch++;
    };
    for (size_t c = c0; c < c1; c += g) {
        if (pool->chunks.count(c)) {
            continue;
        }
        CUmemAllocationProp prop = {};
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        // raw driver-API device id must be PHYSICAL — under GGML_CUDA_DEVICES virtual-device
        // emulation (#25228) pool->device is a ggml (possibly virtual) id (cuMemCreate/cuMemSetAccess
        // don't go through ggml_cuda_set_device's translation).
        prop.location.id   = ggml_cuda_info().devices[pool->device].physical_device;
        CUmemGenericAllocationHandle handle;
        if (observe(cuMemCreate(&handle, g, &prop, 0), "cuMemCreate", c) != CUDA_SUCCESS) {
            ggml_vbr_diag_dump("cuMemCreate_failure_existing_recovery");
            finish_mapping();
            return false; // physical exhausted — caller decides (degrade / abort)
        }
        const CUdeviceptr ptr = (CUdeviceptr)((char *) pool->base + c);
        CU_CHECK(observe(cuMemMap(ptr, g, 0, handle, 0), "cuMemMap", c));
        CU_CHECK(observe(cuMemRelease(handle), "cuMemRelease", c)); // physical is freed when the chunk is unmapped
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id   = ggml_cuda_info().devices[pool->device].physical_device;
        access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(observe(cuMemSetAccess(ptr, g, &access, 1), "cuMemSetAccess", c));
        pool->chunks.insert(c);
        new_chunks.push_back(c);
    }
    finish_mapping();
    ggml_vbr_diag_record(GGML_VBR_DIAG_MAP, "map_end pool=%p new_chunks=%zu mapped=%zu epoch=%llu",
        (void *) pool, new_chunks.size(), pool->chunks.size() * g, (unsigned long long) pool->residency_epoch);
    return true;
}

bool ggml_backend_cuda_vmm_pool_unmap(ggml_vbr_vmm_pool * pool, size_t off, size_t len) {
    // unmap only chunks FULLY inside [off, off+len) — partial chunks stay mapped
    ggml_cuda_set_device(pool->device);
    const size_t g  = pool->gran;
    const size_t c0 = GGML_PAD(off, g);
    const size_t c1 = ((off + len) / g) * g;
    bool changed = false;
    ggml_vbr_diag_record(GGML_VBR_DIAG_MAP, "unmap_begin pool=%p device=%d off=%zu len=%zu mapped=%zu",
        (void *) pool, pool->device, off, len, pool->chunks.size() * g);
    for (size_t c = c0; c < c1; c += g) {
        auto it = pool->chunks.find(c);
        if (it == pool->chunks.end()) {
            continue;
        }
        CU_CHECK(cuMemUnmap((CUdeviceptr)((char *) pool->base + c), g));
        pool->chunks.erase(it);
        changed = true;
    }
    if (changed) {
        vmm_pool_mapping_boundary(pool);
        GGML_ASSERT(pool->residency_epoch != UINT64_MAX);
        pool->residency_epoch++;
    }
    ggml_vbr_diag_record(GGML_VBR_DIAG_MAP, "unmap_end pool=%p mapped=%zu epoch=%llu",
        (void *) pool, pool->chunks.size() * g, (unsigned long long) pool->residency_epoch);
    return true;
}

void ggml_backend_cuda_vmm_pool_clear(ggml_vbr_vmm_pool * pool) {
    ggml_cuda_set_device(pool->device);
    for (size_t c : pool->chunks) {
        CUDA_CHECK(cudaMemset((void *)((char *) pool->base + c), 0, pool->gran));
    }
    if (!pool->chunks.empty()) {
        // order the legacy-stream memsets against the non-blocking ggml streams (see pool_map)
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
}

void ggml_backend_cuda_vmm_pool_free(ggml_vbr_vmm_pool * pool) {
    if (!pool) {
        return;
    }
    ggml_cuda_set_device(pool->device);
    // cuMemUnmap/cuMemAddressFree are host-immediate with no implicit device sync (unlike
    // cudaFree): under -sm layer pipeline parallelism a prior ubatch's kernels can still be
    // reading this VA when the fattn dequant scratch re-reserves mid-decode — settle the
    // device before pulling the mapping out from under them.
    CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t c : pool->chunks) {
        CU_CHECK(cuMemUnmap((CUdeviceptr)((char *) pool->base + c), pool->gran));
    }
    CU_CHECK(cuMemAddressFree(pool->base, pool->va_size));
#if defined(GGML_USE_HIP) && defined(__linux__)
    CUDA_CHECK(hipFree(pool->mapping_guard));
#endif
    delete pool;
}

#else // !GGML_USE_VMM — stubs so llama links regardless of build flags

bool   ggml_backend_cuda_vmm_available(int)                                  { return false;   }
size_t ggml_backend_cuda_vmm_granularity(int)                                { return 0;       }
ggml_vbr_vmm_pool * ggml_backend_cuda_vmm_pool_init(int, size_t)            { return nullptr; }
void * ggml_backend_cuda_vmm_pool_base(ggml_vbr_vmm_pool *)                 { return nullptr; }
size_t ggml_backend_cuda_vmm_pool_mapped(ggml_vbr_vmm_pool *)               { return 0;       }
uint64_t ggml_backend_cuda_vmm_pool_residency_epoch(ggml_vbr_vmm_pool *)    { return 0;       }
size_t ggml_backend_cuda_vmm_pool_mapped_in_range(ggml_vbr_vmm_pool *, size_t, size_t) { return 0; }
bool   ggml_backend_cuda_vmm_pool_map(ggml_vbr_vmm_pool *, size_t, size_t)  { return false;   }
bool   ggml_backend_cuda_vmm_pool_unmap(ggml_vbr_vmm_pool *, size_t, size_t){ return false;   }
void   ggml_backend_cuda_vmm_pool_clear(ggml_vbr_vmm_pool *)                {                 }
void   ggml_backend_cuda_vmm_pool_free(ggml_vbr_vmm_pool *)                 {                 }

#endif // GGML_USE_VMM
