#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_runtime.h>
#endif
#include <cstdlib>
#include <cstdio>
#include <cstring>

// GPU cross-attention ring buffer for DFlash speculative decoding.
// Keeps per-layer ring buffers on GPU and interleaves them into the layout
// expected by the drafter's target_hidden tensor, avoiding the CPU round-trip.

struct dflash_cross_ring_gpu {
    int device;               // CUDA device where ring buffers are allocated
    int n_layers;
    int n_embd;
    int ring_size;

    float ** d_layer_rings;   // device: array of n_layers device pointers
    float *  d_staging;       // device: interleaved output [ring_size * n_layers * n_embd]
    float ** h_layer_ptrs;    // host: copy of per-layer device pointers
};

// Interleave kernel: reads per-layer circular ring, writes interleaved output.
// Grid: (cross_len, n_layers), Block: 256
// Each thread block copies one (token, layer) slice of n_embd floats.
__global__ static void k_cross_ring_interleave(
        const float * const * __restrict__ d_rings,
        float * __restrict__ d_out,
        const int ring_size,
        const int read_start,
        const int cross_len,
        const int n_layers,
        const int n_embd) {
    const int t = blockIdx.x; // token index [0, cross_len)
    const int l = blockIdx.y; // layer index [0, n_layers)

    if (t >= cross_len || l >= n_layers) return;

    const int slot = (read_start + t) % ring_size;
    const float * src = d_rings[l] + (size_t)slot * n_embd;
    float * dst = d_out + (size_t)t * n_layers * n_embd + (size_t)l * n_embd;

    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" void * dflash_cross_ring_gpu_alloc(int n_layers, int n_embd, int ring_size) {
    // env var override
    const char * env = getenv("GGML_DFLASH_GPU_RING");
    if (env && atoi(env) == 0) {
        return nullptr;
    }

    auto * ring = new dflash_cross_ring_gpu();
    cudaGetDevice(&ring->device);
    ring->n_layers  = n_layers;
    ring->n_embd    = n_embd;
    ring->ring_size = ring_size;
    // per-layer ring buffers on device
    ring->h_layer_ptrs = new float*[n_layers];
    for (int l = 0; l < n_layers; l++) {
        cudaError_t err = cudaMalloc(&ring->h_layer_ptrs[l], (size_t)ring_size * n_embd * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "dflash gpu ring: cudaMalloc failed for layer %d: %s\n", l, cudaGetErrorString(err));
            for (int j = 0; j < l; j++) cudaFree(ring->h_layer_ptrs[j]);
            delete[] ring->h_layer_ptrs;
            delete ring;
            return nullptr;
        }
        cudaMemset(ring->h_layer_ptrs[l], 0, (size_t)ring_size * n_embd * sizeof(float));
    }

    // device array of layer pointers
    cudaError_t err = cudaMalloc(&ring->d_layer_rings, n_layers * sizeof(float *));
    if (err != cudaSuccess) {
        for (int l = 0; l < n_layers; l++) cudaFree(ring->h_layer_ptrs[l]);
        delete[] ring->h_layer_ptrs;
        delete ring;
        return nullptr;
    }
    cudaMemcpy(ring->d_layer_rings, ring->h_layer_ptrs, n_layers * sizeof(float *), cudaMemcpyHostToDevice);

    // staging buffer for interleaved output
    err = cudaMalloc(&ring->d_staging, (size_t)ring_size * n_layers * n_embd * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(ring->d_layer_rings);
        for (int l = 0; l < n_layers; l++) cudaFree(ring->h_layer_ptrs[l]);
        delete[] ring->h_layer_ptrs;
        delete ring;
        return nullptr;
    }

    size_t total_mb = ((size_t)ring_size * n_embd * sizeof(float) * n_layers +
                       (size_t)ring_size * n_layers * n_embd * sizeof(float)) / (1024 * 1024);
    fprintf(stderr, "dflash gpu ring: allocated %d layers x %d slots x %d embd + staging (~%zu MB)\n",
            n_layers, ring_size, n_embd, total_mb);

    return ring;
}

extern "C" void dflash_cross_ring_gpu_free(void * handle) {
    if (!handle) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    cudaFree(ring->d_staging);
    cudaFree(ring->d_layer_rings);
    for (int l = 0; l < ring->n_layers; l++) {
        cudaFree(ring->h_layer_ptrs[l]);
    }
    delete[] ring->h_layer_ptrs;
    delete ring;
}

// Split a [ring_pos, ring_pos + n_tokens) span into at most two contiguous segments
// (wrap-around) and invoke copy(ring_tok, other_tok, seg_tokens) for each — the one
// place that owns the ring wrap math for write/write_d2d/read below.
template <typename F>
static void ring_span_for_each(int ring_size, int ring_pos, int n_tokens, F && copy) {
    const int pos   = ((ring_pos % ring_size) + ring_size) % ring_size;
    const int first = ring_size - pos;
    if (first >= n_tokens) {
        copy(pos, 0, n_tokens);
    } else {
        copy(pos, 0, first);
        copy(0, first, n_tokens - first);
    }
}

// Upload host data to a specific position in the GPU ring for one layer.
// Handles wrap-around: if ring_pos + n_tokens > ring_size, splits into two copies.
extern "C" void dflash_cross_ring_gpu_write(
        void * handle, int layer, int ring_pos,
        const float * host_data, int n_tokens, int n_embd) {
    if (!handle) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    if (layer < 0 || layer >= ring->n_layers) return;
    if (n_tokens <= 0) return;

    // The GPU ring holds only ring_size tokens (the cross-attention window). A prefill
    // decode chunk can hand us far more (e.g. a >512-token -b batch), and only the last
    // ring_size of them can survive in the ring. Writing the whole run overflows the
    // destination allocation — cudaMemcpyAsync then fails with "invalid argument", and
    // because that error is latched it surfaces later at an unrelated CUDA_CHECK. Keep
    // only the most recent ring_size tokens, advancing ring_pos past the dropped ones.
    if (n_tokens > ring->ring_size) {
        const int skip = n_tokens - ring->ring_size;
        host_data += (size_t)skip * n_embd;
        ring_pos  += skip;
        n_tokens   = ring->ring_size;
    }

    // Ensure cudaStreamPerThread belongs to the ring's device regardless of
    // which GPU the caller (target model decode) last set as current.
    (void)cudaSetDevice(ring->device);

    float * dst = ring->h_layer_ptrs[layer];
    const size_t stride = (size_t)n_embd * sizeof(float);

    ring_span_for_each(ring->ring_size, ring_pos, n_tokens, [&](int ring_tok, int src_tok, int n) {
        cudaMemcpyAsync(dst + (size_t)ring_tok * n_embd, host_data + (size_t)src_tok * n_embd,
                        (size_t)n * stride, cudaMemcpyHostToDevice, cudaStreamPerThread);
    });
}

extern "C" void dflash_cross_ring_gpu_set_tensor(void * d_dst, const void * d_src, size_t offset, size_t size);

// Device-to-device variant of the ring write: source is a device pointer (e.g. the
// target's graph-embedded capture staging), possibly on another GPU. Same clamp and
// wrap-around handling as the host write; peer copies resolved from pointer attributes.
extern "C" void dflash_cross_ring_gpu_write_d2d(
        void * handle, int layer, int ring_pos,
        const void * dev_src, int n_tokens, int n_embd) {
    if (!handle || !dev_src) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    if (layer < 0 || layer >= ring->n_layers) return;
    if (n_tokens <= 0) return;

    const float * src = (const float *)dev_src;
    if (n_tokens > ring->ring_size) {
        const int skip = n_tokens - ring->ring_size;
        src      += (size_t)skip * n_embd;
        ring_pos += skip;
        n_tokens  = ring->ring_size;
    }

    (void)cudaSetDevice(ring->device);

    float * dst = ring->h_layer_ptrs[layer];
    const size_t stride = (size_t)n_embd * sizeof(float);

    ring_span_for_each(ring->ring_size, ring_pos, n_tokens, [&](int ring_tok, int src_tok, int n) {
        dflash_cross_ring_gpu_set_tensor(dst, src + (size_t)src_tok * n_embd,
                                         (size_t)ring_tok * stride, (size_t)n * stride);
    });
}

// Read a token range out of the ring into host memory (checkpoint persistence when the
// CPU ring is not being maintained). Handles wrap-around; synchronous.
extern "C" void dflash_cross_ring_gpu_read(
        void * handle, int layer, int ring_pos,
        float * host_dst, int n_tokens, int n_embd) {
    if (!handle || !host_dst) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    if (layer < 0 || layer >= ring->n_layers) return;
    if (n_tokens <= 0 || n_tokens > ring->ring_size) return;

    (void)cudaSetDevice(ring->device);

    const float * src = ring->h_layer_ptrs[layer];
    const size_t stride = (size_t)n_embd * sizeof(float);

    ring_span_for_each(ring->ring_size, ring_pos, n_tokens, [&](int ring_tok, int dst_tok, int n) {
        cudaMemcpy(host_dst + (size_t)dst_tok * n_embd, src + (size_t)ring_tok * n_embd,
                   (size_t)n * stride, cudaMemcpyDeviceToHost);
    });
}

// Launch interleave kernel. Returns device pointer to interleaved staging buffer.
extern "C" const float * dflash_cross_ring_gpu_interleave(
        void * handle, int write_pos, int filled, int ctx_window) {
    if (!handle) return nullptr;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    int cross_len = filled < ctx_window ? filled : ctx_window;
    if (cross_len <= 0) return nullptr;

    (void)cudaSetDevice(ring->device);

    int read_start = ((write_pos - cross_len) % ring->ring_size + ring->ring_size) % ring->ring_size;

    dim3 grid(cross_len, ring->n_layers);
    dim3 block(256);

    k_cross_ring_interleave<<<grid, block, 0, cudaStreamPerThread>>>(
        (const float * const *)ring->d_layer_rings,
        ring->d_staging,
        ring->ring_size,
        read_start,
        cross_len,
        ring->n_layers,
        ring->n_embd);

    // sync so staging is ready before drafter decode reads it
    cudaStreamSynchronize(cudaStreamPerThread);

    return ring->d_staging;
}

// D2D copy: from device source to device destination (raw pointers).
// Uses peer copy when source and destination are on different devices.
extern "C" void dflash_cross_ring_gpu_set_tensor(
        void * d_dst, const void * d_src, size_t offset, size_t size) {
    if (!d_dst || !d_src || size == 0) return;

    cudaPointerAttributes dst_attr, src_attr;
    cudaPointerGetAttributes(&dst_attr, (const char *)d_dst + offset);
    cudaPointerGetAttributes(&src_attr, d_src);

    if (dst_attr.type == cudaMemoryTypeDevice && src_attr.type == cudaMemoryTypeDevice
            && dst_attr.device != src_attr.device) {
        cudaMemcpyPeerAsync((char *)d_dst + offset, dst_attr.device,
                            d_src, src_attr.device, size, cudaStreamPerThread);
    } else {
        cudaMemcpyAsync((char *)d_dst + offset, d_src, size,
                         cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    }
}
