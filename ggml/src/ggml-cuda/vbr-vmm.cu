// Dynamic VBR (S2, "option C"): CUDA/HIP virtual-memory pool for the KV cache.
//
// One cuMemAddressReserve VA range holds every (layer,side) KV tensor at a FIXED, page-aligned
// offset sized for its MAX tier (F16 x kv_size) — tensor data pointers never move. Physical 2MB
// pages are mapped on demand as the write watermark advances and unmapped from a tensor's tail
// after a tier degrade shrinks its byte footprint. Freed pages are fungible across tensors, so
// no relocation/compaction is ever needed. Same-source on ROCm (vendors/hip.h maps cuMem*).
//
// Chunks are tracked at allocation-granularity (typically 2MB). Handles are released immediately
// after mapping (physical is freed by cuMemUnmap), matching ggml_cuda_pool_vmm; per-chunk unmap
// also sidesteps ROCR-Runtime issue #285 (can't unmap one giant range on HIP).

#include "common.cuh"
#include "ggml-cuda.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
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

static bool ggml_vbr_vmm_diagnostics_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_VBR_VMM_DIAGNOSTICS");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

static uint64_t ggml_vbr_vmm_elapsed_us(
        const std::chrono::steady_clock::time_point & begin,
        const std::chrono::steady_clock::time_point & end) {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
}

static std::chrono::steady_clock::time_point ggml_vbr_vmm_diag_now(const bool enabled) {
    return enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
}

bool ggml_backend_cuda_vmm_available(int device) {
    return device >= 0 && device < ggml_cuda_info().device_count && ggml_cuda_info().devices[device].vmm;
}

size_t ggml_backend_cuda_vmm_granularity(int device) {
    return ggml_backend_cuda_vmm_available(device) ? ggml_cuda_info().devices[device].vmm_granularity : 0;
}

ggml_vbr_vmm_pool * ggml_backend_cuda_vmm_pool_init(int device, size_t va_size) {
    if (!ggml_backend_cuda_vmm_available(device) || va_size == 0) {
        return nullptr;
    }
    auto * pool = new ggml_vbr_vmm_pool;
    pool->device = device;
    pool->gran   = ggml_cuda_info().devices[device].vmm_granularity;
    pool->va_size = GGML_PAD(va_size, pool->gran);
    const bool diagnostics = ggml_vbr_vmm_diagnostics_enabled();
    const auto reserve_begin = ggml_vbr_vmm_diag_now(diagnostics);
    CUdeviceptr base = 0;
    if (cuMemAddressReserve(&base, pool->va_size, 0, 0, 0) != CUDA_SUCCESS) {
        if (diagnostics) {
            const auto reserve_end = std::chrono::steady_clock::now();
            std::fprintf(stderr,
                "GGML_VBR_VMM_DIAG version=1 event=pool_init status=failed device=%d physical_device=%d "
                "requested=%zu rounded=%zu granularity=%zu reserve_us=%llu\n",
                device, ggml_cuda_info().devices[device].physical_device, va_size, pool->va_size,
                pool->gran, (unsigned long long) ggml_vbr_vmm_elapsed_us(reserve_begin, reserve_end));
            std::fflush(stderr);
        }
        delete pool;
        return nullptr;
    }
    pool->base = base;
    if (diagnostics) {
        const auto reserve_end = std::chrono::steady_clock::now();
        std::fprintf(stderr,
            "GGML_VBR_VMM_DIAG version=1 event=pool_init status=ok pool=%p device=%d physical_device=%d "
            "requested=%zu rounded=%zu granularity=%zu reserve_us=%llu\n",
            (void *) pool, device, ggml_cuda_info().devices[device].physical_device, va_size, pool->va_size,
            pool->gran, (unsigned long long) ggml_vbr_vmm_elapsed_us(reserve_begin, reserve_end));
        std::fflush(stderr);
    }
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
    const bool diagnostics = ggml_vbr_vmm_diagnostics_enabled();
    const auto call_begin = ggml_vbr_vmm_diag_now(diagnostics);
    const size_t mapped_before = pool->chunks.size() * g;
    size_t new_chunks = 0;
    uint64_t create_us = 0;
    uint64_t map_us = 0;
    uint64_t access_us = 0;
    uint64_t memset_us = 0;
    static std::atomic<uint64_t> call_serial { 0 };
    const uint64_t call = diagnostics ? call_serial.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
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
        const auto create_begin = ggml_vbr_vmm_diag_now(diagnostics);
        if (cuMemCreate(&handle, g, &prop, 0) != CUDA_SUCCESS) {
            const auto create_end = ggml_vbr_vmm_diag_now(diagnostics);
            if (diagnostics) {
                create_us += ggml_vbr_vmm_elapsed_us(create_begin, create_end);
            }
            // Earlier chunks in this same call were zeroed on the legacy stream. A recoverable
            // partial failure retains them, so settle their initialization before exposing them
            // through mapped()/mapped_in_range() or a later idempotent retry.
            if (zeroed) {
                const auto sync_begin = ggml_vbr_vmm_diag_now(diagnostics);
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
                const auto sync_end = ggml_vbr_vmm_diag_now(diagnostics);
                if (diagnostics) {
                    memset_us += ggml_vbr_vmm_elapsed_us(sync_begin, sync_end);
                }
                GGML_ASSERT(pool->residency_epoch != UINT64_MAX);
                pool->residency_epoch++;
            }
            if (diagnostics) {
                const auto call_end = std::chrono::steady_clock::now();
                std::fprintf(stderr,
                    "GGML_VBR_VMM_DIAG version=1 event=map status=failed call=%llu pool=%p device=%d "
                    "requested_off=%zu requested_len=%zu rounded_off=%zu rounded_len=%zu mapped_before=%zu "
                    "mapped_after=%zu new_chunks=%zu create_us=%llu map_us=%llu access_us=%llu "
                    "memset_sync_us=%llu total_us=%llu\n",
                    (unsigned long long) call, (void *) pool, pool->device, off, len, c0, c1 - c0,
                    mapped_before, pool->chunks.size() * g, new_chunks,
                    (unsigned long long) create_us, (unsigned long long) map_us,
                    (unsigned long long) access_us, (unsigned long long) memset_us,
                    (unsigned long long) ggml_vbr_vmm_elapsed_us(call_begin, call_end));
                std::fflush(stderr);
            }
            return false; // physical exhausted — caller decides (degrade / abort)
        }
        const auto create_end = ggml_vbr_vmm_diag_now(diagnostics);
        if (diagnostics) {
            create_us += ggml_vbr_vmm_elapsed_us(create_begin, create_end);
        }
        const CUdeviceptr ptr = (CUdeviceptr)((char *) pool->base + c);
        const auto map_begin = ggml_vbr_vmm_diag_now(diagnostics);
        CU_CHECK(cuMemMap(ptr, g, 0, handle, 0));
        CU_CHECK(cuMemRelease(handle)); // physical is freed when the chunk is unmapped
        const auto map_end = ggml_vbr_vmm_diag_now(diagnostics);
        if (diagnostics) {
            map_us += ggml_vbr_vmm_elapsed_us(map_begin, map_end);
        }
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id   = ggml_cuda_info().devices[pool->device].physical_device;
        access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        const auto access_begin = ggml_vbr_vmm_diag_now(diagnostics);
        CU_CHECK(cuMemSetAccess(ptr, g, &access, 1));
        const auto access_end = ggml_vbr_vmm_diag_now(diagnostics);
        if (diagnostics) {
            access_us += ggml_vbr_vmm_elapsed_us(access_begin, access_end);
        }
        // fresh pages start zeroed: same NaN-in-padding guarantee the eager buffer clear gave
        const auto memset_begin = ggml_vbr_vmm_diag_now(diagnostics);
        CUDA_CHECK(cudaMemset((void *) ptr, 0, g));
        const auto memset_end = ggml_vbr_vmm_diag_now(diagnostics);
        if (diagnostics) {
            memset_us += ggml_vbr_vmm_elapsed_us(memset_begin, memset_end);
        }
        pool->chunks.insert(c);
        new_chunks++;
        zeroed = true;
    }
    if (zeroed) {
        // the memsets ran on the legacy stream; ggml streams are non-blocking, so nothing orders
        // them against the compute/side streams that write these pages next — settle them here
        // (rare: only on watermark growth, and the pages are new)
        const auto sync_begin = ggml_vbr_vmm_diag_now(diagnostics);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        const auto sync_end = ggml_vbr_vmm_diag_now(diagnostics);
        if (diagnostics) {
            memset_us += ggml_vbr_vmm_elapsed_us(sync_begin, sync_end);
        }
        GGML_ASSERT(pool->residency_epoch != UINT64_MAX);
        pool->residency_epoch++;
    }
    if (diagnostics) {
        const auto call_end = std::chrono::steady_clock::now();
        std::fprintf(stderr,
            "GGML_VBR_VMM_DIAG version=1 event=map status=ok call=%llu pool=%p device=%d "
            "requested_off=%zu requested_len=%zu rounded_off=%zu rounded_len=%zu mapped_before=%zu "
            "mapped_after=%zu new_chunks=%zu create_us=%llu map_us=%llu access_us=%llu "
            "memset_sync_us=%llu total_us=%llu\n",
            (unsigned long long) call, (void *) pool, pool->device, off, len, c0, c1 - c0,
            mapped_before, pool->chunks.size() * g, new_chunks,
            (unsigned long long) create_us, (unsigned long long) map_us,
            (unsigned long long) access_us, (unsigned long long) memset_us,
            (unsigned long long) ggml_vbr_vmm_elapsed_us(call_begin, call_end));
        std::fflush(stderr);
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
