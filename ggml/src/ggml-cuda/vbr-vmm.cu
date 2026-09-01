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

#include <set>

#if defined(GGML_USE_VMM)

struct ggml_vbr_vmm_pool {
    int         device;
    CUdeviceptr base    = 0;
    size_t      va_size = 0;
    size_t      gran    = 0;
    uint64_t    residency_epoch = 0;
    std::set<size_t> chunks; // mapped chunk offsets (each gran bytes)
};

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
    bool zeroed = false;
    const size_t g  = pool->gran;
    const size_t c0 = (off / g) * g;
    const size_t c1 = GGML_PAD(off + len, g);
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
        if (cuMemCreate(&handle, g, &prop, 0) != CUDA_SUCCESS) {
            // Earlier chunks in this same call were zeroed on the legacy stream. A recoverable
            // partial failure retains them, so settle their initialization before exposing them
            // through mapped()/mapped_in_range() or a later idempotent retry.
            if (zeroed) {
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
                GGML_ASSERT(pool->residency_epoch != UINT64_MAX);
                pool->residency_epoch++;
            }
            return false; // physical exhausted — caller decides (degrade / abort)
        }
        const CUdeviceptr ptr = (CUdeviceptr)((char *) pool->base + c);
        CU_CHECK(cuMemMap(ptr, g, 0, handle, 0));
        CU_CHECK(cuMemRelease(handle)); // physical is freed when the chunk is unmapped
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id   = ggml_cuda_info().devices[pool->device].physical_device;
        access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(ptr, g, &access, 1));
        // fresh pages start zeroed: same NaN-in-padding guarantee the eager buffer clear gave
        CUDA_CHECK(cudaMemset((void *) ptr, 0, g));
        pool->chunks.insert(c);
        zeroed = true;
    }
    if (zeroed) {
        // the memsets ran on the legacy stream; ggml streams are non-blocking, so nothing orders
        // them against the compute/side streams that write these pages next — settle them here
        // (rare: only on watermark growth, and the pages are new)
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        GGML_ASSERT(pool->residency_epoch != UINT64_MAX);
        pool->residency_epoch++;
    }
    return true;
}

bool ggml_backend_cuda_vmm_pool_unmap(ggml_vbr_vmm_pool * pool, size_t off, size_t len) {
    // unmap only chunks FULLY inside [off, off+len) — partial chunks stay mapped
    ggml_cuda_set_device(pool->device);
    const size_t g  = pool->gran;
    const size_t c0 = GGML_PAD(off, g);
    const size_t c1 = ((off + len) / g) * g;
    bool changed = false;
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
        GGML_ASSERT(pool->residency_epoch != UINT64_MAX);
        pool->residency_epoch++;
    }
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
