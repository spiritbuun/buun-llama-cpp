#pragma once
// Route vendor headers (cuda_runtime, cuda_fp16) through the shim for
// CUDA/HIP portability — see ggml-cuda/vendors/{cuda,hip}.h.
#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include "ggml-common.h"
#include "ggml-turbo-meansub.h"
#include <atomic>
#include <mutex>

// === InnerQ per-channel equalization ===
// Scale K channels before L2 norm + FWHT to reduce quantization error on anisotropic distributions.
// Inverse scale applied to Q in FA kernel to preserve dot products.
// Calibration: accumulate per-channel K^2, then set scale[i] = 1/sqrt(mean(K_i^2) * 128).
static __device__ float d_innerq_channel_scale[128];     // per-channel K scale (init to 1.0)
static __device__ float d_innerq_channel_scale_inv[128]; // per-channel Q inverse scale (init to 1.0)
static __device__ float d_innerq_channel_sq[128];        // calibration accumulator: sum of K_i^2
static __device__ float d_innerq_channel_max[128];       // calibration accumulator: max of |K_i| (for paper's formula)
static __device__ int   d_innerq_count;                  // calibration token count
static __device__ int   d_innerq_calibrate;              // 1 = accumulate stats, 0 = apply scales
static __device__ int   d_innerq_is_k;                   // 1 = current set_rows is K cache, 0 = V cache

// === Track B probe: post-FWHT per-bin statistics (KVarN channel-axis gate, #135) ===
// During InnerQ calibration (identity scale), accumulate K-only post-rotation bin stats to test
// whether our FWHT leaves material per-bin std imbalance / DC offset (= KVarN's per-channel lever).
static __device__ float d_postfwht_bin_sum[128];         // sum of post-FWHT bin value (K only) → mean/DC
static __device__ float d_postfwht_bin_sq[128];          // sum of post-FWHT bin value^2 (K only) → std
static __device__ int   d_postfwht_count;                // K group count for the probe

// === PFHEAD probe: RAW per-(layer,channel) K/V mean/std (per-head mean trick gate) ===
// #135 pooled bins across heads+layers, which cancels per-head DC offsets (the TorQuant-cert
// pooled-centering trap). This probe keys channel stats by (kv, layer, channel) so per-head
// means survive, and accumulates PRE-norm PRE-rotation raw values: subtracting a per-head mean
// at encode is softmax-neutral for K only when done in the raw domain (post-norm the missing
// term picks up each token's group norm and stops being a uniform logit shift).
// Enabled by TURBO_PFHEAD_DUMP=<path> during InnerQ calibration (TURBO_INNERQ=1, -ctk/-ctv turbo3).
#define PFHEAD_MAX_L GGML_TURBO_MEANSUB_MAX_L
#define PFHEAD_MAX_C GGML_TURBO_MEANSUB_MAX_C
static __device__ float * d_pfhead_sum = nullptr;        // [2][PFHEAD_MAX_L][PFHEAD_MAX_C] kv-major
static __device__ float * d_pfhead_sq  = nullptr;
static __device__ int   * d_pfhead_cnt = nullptr;        // [2][PFHEAD_MAX_L] row counts
// NOTE: the layer index is passed as a KERNEL ARGUMENT (pf_layer), never a device symbol —
// a per-launch symbol write races with queued kernels on the non-blocking compute stream
// and misattributes samples across layers (discovered 2026-06-12: l63 absorbed 8x rows).

// Forward declaration: fattn compilation unit has its own copy of inverse scales
extern void turbo_innerq_update_fattn_scales(const float * scale_inv);
extern void turbo_innerq_init_fattn();
extern void turbo_q_calibrate_init();
extern void turbo_q_calibrate_finalize();

// TCQ error dump: save post-FWHT normalized values and output symbols for autocorrelation analysis
static __device__ float   * d_tcq_dump_x_buf   = nullptr; // [max_groups][128] original values
static __device__ uint8_t * d_tcq_dump_out_buf  = nullptr; // [max_groups][128] output symbols
static __device__ int       d_tcq_dump_max      = 0;       // max groups to dump (0 = disabled)
static __device__ int       d_tcq_tiehi         = 0;       // TURBO_TCQ_TIEHI tie-break probe

// === Post-FWHT data extraction for empirical codebook computation ===
// Enabled by TURBO_EXTRACT=<max_samples> env var (e.g. TURBO_EXTRACT=2000000)
// Dumps post-rotation normalized values to /tmp/turbo_postrot.bin (float32)
// Device-visible extraction state
static __device__ float * d_extract_buf_ptr = nullptr;
static __device__ int   * d_extract_pos_ptr = nullptr;
static __device__ int     d_extract_max_val = 0;
// Optional per-record tag (layer*1000 + kvsel*100 + group), -1 = untagged source.
// Parallel file /tmp/turbo_postrot_tags.bin, one int32 per 128-float record.
static __device__ int   * d_extract_tag_ptr = nullptr;
// TURBO_EXTRACT_STRIDE=N keeps every Nth record so the corpus spans more text
// (default 1 = old behavior: buffer fills within the first prefill chunk).
static __device__ int   * d_extract_seen_ptr = nullptr;
static __device__ int     d_extract_stride_val = 1;

// Host-side management
static float * h_extract_gpu_buf = nullptr;
static int   * h_extract_gpu_tags = nullptr;
static int   * h_extract_gpu_pos = nullptr;
static int   * h_extract_gpu_seen = nullptr;
static int     h_extract_max = 0;
static int     h_extract_state = 0;  // 0=uninit, 1=collecting, 2=done
static std::once_flag h_extract_init_flag;
static std::mutex     h_extract_check_mutex;

static void turbo_extract_init(int max_samples) {
	int cur_device;
	cudaGetDevice(&cur_device);
	int device_count;
	cudaGetDeviceCount(&device_count);
	cudaMalloc(&h_extract_gpu_buf, (size_t)max_samples * sizeof(float));
	cudaMalloc(&h_extract_gpu_tags, (size_t)(max_samples / 128) * sizeof(int));
	cudaMalloc(&h_extract_gpu_pos, sizeof(int));
	cudaMalloc(&h_extract_gpu_seen, sizeof(int));
	int zero = 0;
	cudaMemcpy(h_extract_gpu_pos, &zero, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(h_extract_gpu_seen, &zero, sizeof(int), cudaMemcpyHostToDevice);
	const char * stride_env = getenv("TURBO_EXTRACT_STRIDE");
	int stride = stride_env ? atoi(stride_env) : 1;
	if (stride < 1) stride = 1;
	for (int id = 0; id < device_count; id++) {
		cudaSetDevice(id);
		cudaMemcpyToSymbol(d_extract_buf_ptr, &h_extract_gpu_buf, sizeof(float *));
		cudaMemcpyToSymbol(d_extract_tag_ptr, &h_extract_gpu_tags, sizeof(int *));
		cudaMemcpyToSymbol(d_extract_pos_ptr, &h_extract_gpu_pos, sizeof(int *));
		cudaMemcpyToSymbol(d_extract_seen_ptr, &h_extract_gpu_seen, sizeof(int *));
		cudaMemcpyToSymbol(d_extract_stride_val, &stride, sizeof(int));
		cudaMemcpyToSymbol(d_extract_max_val, &max_samples, sizeof(int));
		if (id != cur_device) {
			cudaDeviceEnablePeerAccess(cur_device, 0);
		}
	}
	cudaSetDevice(cur_device);
	h_extract_max = max_samples;
	h_extract_state = 1;
	fprintf(stderr, "TURBO_EXTRACT: collecting up to %d post-rotation samples\n", max_samples);
}

static void turbo_extract_check_done() {
	if (h_extract_state != 1) return;
	int pos;
	cudaMemcpy(&pos, h_extract_gpu_pos, sizeof(int), cudaMemcpyDeviceToHost);
	if (pos < h_extract_max) return;
	// Buffer full — dump to disk
	if (pos > h_extract_max) pos = h_extract_max;
	float * host_buf = (float *)malloc((size_t)pos * sizeof(float));
	cudaMemcpy(host_buf, h_extract_gpu_buf, (size_t)pos * sizeof(float), cudaMemcpyDeviceToHost);
	const char * path = "/tmp/turbo_postrot.bin";
	FILE * fp = fopen(path, "wb");
	if (fp) {
		fwrite(host_buf, sizeof(float), pos, fp);
		fclose(fp);
		fprintf(stderr, "TURBO_EXTRACT: wrote %d samples to %s (%.1f MB)\n",
				pos, path, (float)pos * sizeof(float) / (1024*1024));
	}
	free(host_buf);
	{
		const int n_rec = pos / 128;
		int * tag_buf = (int *)malloc((size_t)n_rec * sizeof(int));
		cudaMemcpy(tag_buf, h_extract_gpu_tags, (size_t)n_rec * sizeof(int), cudaMemcpyDeviceToHost);
		FILE * tfp = fopen("/tmp/turbo_postrot_tags.bin", "wb");
		if (tfp) {
			fwrite(tag_buf, sizeof(int), n_rec, tfp);
			fclose(tfp);
			fprintf(stderr, "TURBO_EXTRACT: wrote %d tags to /tmp/turbo_postrot_tags.bin\n", n_rec);
		}
		free(tag_buf);
	}
	// Disable extraction (set device pointers to null) on all devices
	float * null_ptr = nullptr;
	int   * null_iptr = nullptr;
	int     zero_max = 0;
	int cur_dev;
	cudaGetDevice(&cur_dev);
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	for (int id = 0; id < dev_count; id++) {
		cudaSetDevice(id);
		cudaMemcpyToSymbol(d_extract_buf_ptr, &null_ptr, sizeof(float *));
		cudaMemcpyToSymbol(d_extract_tag_ptr, &null_iptr, sizeof(int *));
		cudaMemcpyToSymbol(d_extract_pos_ptr, &null_iptr, sizeof(int *));
		cudaMemcpyToSymbol(d_extract_seen_ptr, &null_iptr, sizeof(int *));
		cudaMemcpyToSymbol(d_extract_max_val, &zero_max, sizeof(int));
	}
	cudaSetDevice(cur_dev);
	cudaFree(h_extract_gpu_buf); h_extract_gpu_buf = nullptr;
	cudaFree(h_extract_gpu_tags); h_extract_gpu_tags = nullptr;
	cudaFree(h_extract_gpu_pos); h_extract_gpu_pos = nullptr;
	cudaFree(h_extract_gpu_seen); h_extract_gpu_seen = nullptr;
	h_extract_state = 2;
	// Also finalize Q² calibration if running
	turbo_q_calibrate_finalize();
}

// Device-side: append 128 post-rotation values to extraction buffer
static __device__ void turbo_extract_append(const float * x, int tag = -1) {
	if (!d_extract_buf_ptr || !d_extract_pos_ptr) return;
	if (d_extract_stride_val > 1 && d_extract_seen_ptr) {
		if (atomicAdd(d_extract_seen_ptr, 1) % d_extract_stride_val != 0) return;
	}
	int base = atomicAdd(d_extract_pos_ptr, 128);
	if (base + 128 <= d_extract_max_val) {
		for (int j = 0; j < 128; j++) d_extract_buf_ptr[base + j] = x[j];
		if (d_extract_tag_ptr) d_extract_tag_ptr[base / 128] = tag;
	}
}

// Host-side init: set identity scales, zero accumulators (all devices)
static void turbo_innerq_init() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    float zeros[128] = {};
    int zero = 0;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale, ones, sizeof(ones));
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv, ones, sizeof(ones));
        cudaMemcpyToSymbol(d_innerq_channel_sq, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_innerq_channel_max, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_innerq_calibrate, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_innerq_is_k, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_postfwht_bin_sum, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_postfwht_bin_sq, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_postfwht_count, &zero, sizeof(zero));
    }
    cudaSetDevice(cur_device);
    turbo_innerq_init_fattn();
    turbo_q_calibrate_init();
}

// Host-side: set K/V flag before kernel launch (called from set-rows.cu)
static void turbo_innerq_set_is_k(int is_k) {
    cudaMemcpyToSymbol(d_innerq_is_k, &is_k, sizeof(int));
}

// PFHEAD probe host state (current device only — probe runs are single-GPU)
static float * h_pfhead_sum_ptr = nullptr;
static float * h_pfhead_sq_ptr  = nullptr;
static int   * h_pfhead_cnt_ptr = nullptr;

// Host-side: allocate + arm the PFHEAD probe if TURBO_PFHEAD_DUMP is set
static void turbo_pfhead_start() {
    const char * path = getenv("TURBO_PFHEAD_DUMP");
    if (!path || !path[0] || h_pfhead_sum_ptr) return;
    const size_t n = (size_t) 2 * PFHEAD_MAX_L * PFHEAD_MAX_C;
    cudaMalloc(&h_pfhead_sum_ptr, n * sizeof(float));
    cudaMalloc(&h_pfhead_sq_ptr,  n * sizeof(float));
    cudaMalloc(&h_pfhead_cnt_ptr, 2 * PFHEAD_MAX_L * sizeof(int));
    cudaMemset(h_pfhead_sum_ptr, 0, n * sizeof(float));
    cudaMemset(h_pfhead_sq_ptr,  0, n * sizeof(float));
    cudaMemset(h_pfhead_cnt_ptr, 0, 2 * PFHEAD_MAX_L * sizeof(int));
    cudaMemcpyToSymbol(d_pfhead_sum, &h_pfhead_sum_ptr, sizeof(h_pfhead_sum_ptr));
    cudaMemcpyToSymbol(d_pfhead_sq,  &h_pfhead_sq_ptr,  sizeof(h_pfhead_sq_ptr));
    cudaMemcpyToSymbol(d_pfhead_cnt, &h_pfhead_cnt_ptr, sizeof(h_pfhead_cnt_ptr));
    fprintf(stderr, "PFHEAD probe: armed, raw per-(layer,channel) K/V stats -> %s\n", path);
}

// === K-mean subtraction (per-head mean trick, K-only) ===
// TURBO_KMEAN_SUB=<PFH1 dump> loads per-(layer,channel) raw K means (sum/cnt, kv=0 slab) and
// subtracts them from K at encode, PRE-norm PRE-rotation. No decode change needed: the missing
// q·mu term is a per-head constant across positions, so softmax is shift-invariant to it.
// Channels from layers with cnt<100 get mu=0 (no-op). V is NOT subtracted (V means survive the
// weighted sum and would need a decode add-back).
static float * h_kmean_dev[GGML_CUDA_MAX_DEVICES][GGML_TURBO_MEANSUB_MAX_MODELS] = {};
static std::atomic<bool> h_kmean_checked[GGML_CUDA_MAX_DEVICES][GGML_TURBO_MEANSUB_MAX_MODELS] = {};
// Dense source slabs are immutable and model-keyed. Serialize first use across devices/models;
// CUDA pointers remain cached independently for each pair.
static std::mutex h_kmean_mutex;
static float * h_vmean_dev[GGML_CUDA_MAX_DEVICES][GGML_TURBO_MEANSUB_MAX_MODELS] = {};
static std::atomic<bool> h_vmean_checked[GGML_CUDA_MAX_DEVICES][GGML_TURBO_MEANSUB_MAX_MODELS] = {};
static std::mutex h_vmean_mutex;

static float * turbo_meansub_upload(int device, cudaStream_t stream, const float * host, size_t bytes) {
    ggml_cuda_set_device(device);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    GGML_ASSERT(capture_status == cudaStreamCaptureStatusNone);
#endif

    float * dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, bytes));
    CUDA_CHECK(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return dev;
}

// Shared PFH1 mean-table loader. kvsel: 0 = K slab (TURBO_KMEAN_SUB), 1 = V slab
// (TURBO_VMEAN_SUB; encode half of the V tap — the graph-level add restores mu_V at decode).
static float * turbo_meansub_load(
        int device, cudaStream_t stream, int model_id, int kvsel, const char * env_name) {
    if (getenv("TURBO_MEANSUB_OFF")) return nullptr;   // explicit disable (A/B + opt-out)
    if (model_id < 0 || model_id >= GGML_TURBO_MEANSUB_MAX_MODELS) return nullptr;
    const char * path = getenv(env_name);
    if (!path || !path[0]) {
        // No env override -> use the binary-baked per-architecture means (tap default-on).
        int bL = 0, bC = 0, blive = 0;
        const float * baked = ggml_turbo_meansub_table(model_id, kvsel, &bL, &bC, &blive);
        if (baked && bL == PFHEAD_MAX_L && bC == PFHEAD_MAX_C && blive > 0) {
            const size_t nk = (size_t) PFHEAD_MAX_L * PFHEAD_MAX_C;
            float * dev = turbo_meansub_upload(device, stream, baked, nk * sizeof(float));
            fprintf(stderr, "TURBO meansub (device %d): %s-mean BAKED table (%d live layers)\n",
                    device, kvsel == 0 ? "K" : "V", blive);
            return dev;
        }
        return nullptr;
    }
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "%s: cannot open %s\n", env_name, path); return nullptr; }
    int32_t hdr[4];
    const size_t n = (size_t) 2 * PFHEAD_MAX_L * PFHEAD_MAX_C;
    if (fread(hdr, sizeof(int32_t), 4, f) != 4 || hdr[0] != 0x50464831 ||
        hdr[1] != 2 || hdr[2] != PFHEAD_MAX_L || hdr[3] != PFHEAD_MAX_C) {
        fprintf(stderr, "%s: bad header in %s\n", env_name, path); fclose(f); return nullptr;
    }
    const size_t nk = (size_t) PFHEAD_MAX_L * PFHEAD_MAX_C;
    float * sums = (float *) malloc(nk * sizeof(float));
    int   * cnts = (int *)   malloc(2 * PFHEAD_MAX_L * sizeof(int));
    bool ok = fseek(f, (long) (16 + (size_t) kvsel * nk * sizeof(float)), SEEK_SET) == 0 &&
              fread(sums, sizeof(float), nk, f) == nk &&
              fseek(f, (long) (16 + 2 * n * sizeof(float)), SEEK_SET) == 0 &&
              fread(cnts, sizeof(int), 2 * PFHEAD_MAX_L, f) == (size_t) 2 * PFHEAD_MAX_L;
    fclose(f);
    if (!ok) { fprintf(stderr, "%s: short read in %s\n", env_name, path); free(sums); free(cnts); return nullptr; }
    float * mu = (float *) calloc(nk, sizeof(float));
    int layers_live = 0;
    for (int l = 0; l < PFHEAD_MAX_L; l++) {
        const int c = cnts[kvsel * PFHEAD_MAX_L + l];
        if (c < 100) continue;
        layers_live++;
        for (int j = 0; j < PFHEAD_MAX_C; j++) {
            mu[(size_t) l * PFHEAD_MAX_C + j] = sums[(size_t) l * PFHEAD_MAX_C + j] / c;
        }
    }
    free(sums); free(cnts);
    float * dev = turbo_meansub_upload(device, stream, mu, nk * sizeof(float));
    free(mu);
    fprintf(stderr, "%s (device %d): %s-mean table loaded from %s (%d live layers)\n",
            env_name, device, kvsel == 0 ? "K" : "V", path, layers_live);
    return dev;
}

// Set true (around a set_rows call) to suppress the encode mean-sub tap. Used by the VBR transcode
// re-encode: its f32 input is already in stored domain (V - mu_V), so re-subtracting mu would double
// it. Defined in set-rows.cu; shared so the choke-point lookups below see it across TUs.
extern bool g_turbo_meansub_suppress;

static float * turbo_kmean_table(int device, cudaStream_t stream, int model_id) {
    if (g_turbo_meansub_suppress) return nullptr;
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES ||
            model_id < 0 || model_id >= GGML_TURBO_MEANSUB_MAX_MODELS) return nullptr;
    if (h_kmean_checked[device][model_id].load(std::memory_order_acquire)) return h_kmean_dev[device][model_id];
    std::lock_guard<std::mutex> lock(h_kmean_mutex);
    if (!h_kmean_checked[device][model_id].load(std::memory_order_relaxed)) {
        h_kmean_dev[device][model_id] = turbo_meansub_load(device, stream, model_id, 0, "TURBO_KMEAN_SUB");
        h_kmean_checked[device][model_id].store(true, std::memory_order_release);
    }
    return h_kmean_dev[device][model_id];
}

static float * turbo_vmean_table_enc(int device, cudaStream_t stream, int model_id) {
    if (g_turbo_meansub_suppress) return nullptr;
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES ||
            model_id < 0 || model_id >= GGML_TURBO_MEANSUB_MAX_MODELS) return nullptr;
    if (h_vmean_checked[device][model_id].load(std::memory_order_acquire)) return h_vmean_dev[device][model_id];
    std::lock_guard<std::mutex> lock(h_vmean_mutex);
    if (!h_vmean_checked[device][model_id].load(std::memory_order_relaxed)) {
        h_vmean_dev[device][model_id] = turbo_meansub_load(device, stream, model_id, 1, "TURBO_VMEAN_SUB");
        h_vmean_checked[device][model_id].store(true, std::memory_order_release);
    }
    return h_vmean_dev[device][model_id];
}

// Host-side: copy back + write the PFHEAD dump (PFH1: hdr, sums, sqs, counts)
static void turbo_pfhead_dump() {
    if (!h_pfhead_sum_ptr) return;
    cudaDeviceSynchronize();   // drain in-flight set_rows kernels before reading accumulators
    const char * path = getenv("TURBO_PFHEAD_DUMP");
    const size_t n = (size_t) 2 * PFHEAD_MAX_L * PFHEAD_MAX_C;
    float * sum = (float *) malloc(n * sizeof(float));
    float * sq  = (float *) malloc(n * sizeof(float));
    int   * cnt = (int *)   malloc(2 * PFHEAD_MAX_L * sizeof(int));
    cudaMemcpy(sum, h_pfhead_sum_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sq,  h_pfhead_sq_ptr,  n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnt, h_pfhead_cnt_ptr, 2 * PFHEAD_MAX_L * sizeof(int), cudaMemcpyDeviceToHost);
    FILE * f = fopen(path, "wb");
    if (f) {
        const int32_t hdr[4] = { 0x50464831 /*PFH1*/, 2, PFHEAD_MAX_L, PFHEAD_MAX_C };
        fwrite(hdr, sizeof(int32_t), 4, f);
        fwrite(sum, sizeof(float), n, f);
        fwrite(sq,  sizeof(float), n, f);
        fwrite(cnt, sizeof(int), 2 * PFHEAD_MAX_L, f);
        fclose(f);
        int layers_seen = 0;
        long total_rows = 0;
        for (int l = 0; l < PFHEAD_MAX_L; l++) {
            if (cnt[l] > 0) { layers_seen++; total_rows += cnt[l]; }
        }
        fprintf(stderr, "PFHEAD probe: wrote %s (%d K layers seen, %ld K rows)\n", path, layers_seen, total_rows);
    } else {
        fprintf(stderr, "PFHEAD probe: FAILED to open %s\n", path);
    }
    free(sum); free(sq); free(cnt);
}

// Host-side: enable calibration mode (all devices)
static void turbo_innerq_start_calibration() {
    float zeros[128] = {};
    int zero = 0, one = 1;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_sq, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_innerq_channel_max, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_postfwht_bin_sum, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_postfwht_bin_sq, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_postfwht_count, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_innerq_calibrate, &one, sizeof(one));
    }
    cudaSetDevice(cur_device);
    turbo_pfhead_start();
}

// Host-side: finalize calibration — compute scales from accumulated stats
static void turbo_innerq_finalize_calibration() {
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);

    int zero = 0;
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_calibrate, &zero, sizeof(zero));
    }
    cudaSetDevice(cur_device);

    float sq[128], ch_max[128];
    int count;
    cudaMemcpyFromSymbol(sq, d_innerq_channel_sq, sizeof(sq));
    cudaMemcpyFromSymbol(ch_max, d_innerq_channel_max, sizeof(ch_max));
    cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(count));

    // Track B probe (#135): report post-FWHT per-bin std imbalance + DC offset (K-only).
    // Flat std (~<1.3x) + zero-mean => KVarN's per-channel machinery (Sinkhorn + affine zp) is null here.
    {
        float pf_sum[128], pf_sq[128];
        int pf_count = 0;
        cudaMemcpyFromSymbol(pf_sum, d_postfwht_bin_sum, sizeof(pf_sum));
        cudaMemcpyFromSymbol(pf_sq, d_postfwht_bin_sq, sizeof(pf_sq));
        cudaMemcpyFromSymbol(&pf_count, d_postfwht_count, sizeof(pf_count));
        if (pf_count > 0) {
            float std_min = 1e30f, std_max = 0.0f, std_sum = 0.0f;
            float max_abs_mean = 0.0f, bin0_mean = 0.0f, bin0_std = 0.0f;
            int imax = -1;
            for (int j = 0; j < 128; j++) {
                float mean = pf_sum[j] / pf_count;
                float ex2  = pf_sq[j] / pf_count;
                float var  = ex2 - mean * mean;
                float sd   = var > 0.0f ? sqrtf(var) : 0.0f;
                std_sum += sd;
                if (sd < std_min) std_min = sd;
                if (sd > std_max) { std_max = sd; imax = j; }
                if (fabsf(mean) > max_abs_mean) max_abs_mean = fabsf(mean);
                if (j == 0) { bin0_mean = mean; bin0_std = sd; }
            }
            float mean_std = std_sum / 128.0f;
            float std_ratio = std_min > 1e-12f ? std_max / std_min : 0.0f;
            fprintf(stderr,
                "PostFWHT probe (#135, K-only, %d groups): per-bin std min=%.5f max=%.5f mean=%.5f "
                "RATIO=%.3f (max bin=%d) | max|mean|=%.5f (vs mean_std %.5f) | bin0 mean=%.5f std=%.5f\n",
                pf_count, std_min, std_max, mean_std, std_ratio, imax,
                max_abs_mean, mean_std, bin0_mean, bin0_std);
        }
    }

    turbo_pfhead_dump();

    if (count == 0) return;

    // Mode: 0=RMS-based (default), 1=max-based (paper's formula: sqrt(max|K_i|))
    static const char * mode_env = getenv("TURBO_INNERQ_MODE");
    int mode = mode_env ? atoi(mode_env) : 0;

    static const char * strength_env = getenv("TURBO_INNERQ_STRENGTH");
    float strength = strength_env ? atof(strength_env) : 0.5f;
    float max_clamp = 2.0f;

    float scale[128], scale_inv[128];
    float max_ratio = 1.0f;

    if (mode == 1) {
        // Paper's formula: scale[i] = 1/sqrt(max(|K_{:,i}|))
        // This normalizes each channel so its max value becomes sqrt(max_val)
        fprintf(stderr, "InnerQ mode=1 (paper's max-based formula)\n");
        for (int i = 0; i < 128; i++) {
            if (ch_max[i] > 1e-10f) {
                float s = 1.0f / sqrtf(ch_max[i]);
                // Normalize so mean scale = 1 (preserve overall magnitude)
                scale[i] = s;
            } else {
                scale[i] = 1.0f;
            }
        }
        // Normalize scales to have geometric mean ≈ 1
        float log_sum = 0.0f;
        for (int i = 0; i < 128; i++) log_sum += logf(scale[i]);
        float geo_mean = expf(log_sum / 128.0f);
        for (int i = 0; i < 128; i++) {
            scale[i] /= geo_mean;
            if (scale[i] > max_clamp) scale[i] = max_clamp;
            if (scale[i] < 1.0f / max_clamp) scale[i] = 1.0f / max_clamp;
            scale_inv[i] = 1.0f / scale[i];
            float ratio = fmaxf(scale[i], 1.0f / scale[i]);
            if (ratio > max_ratio) max_ratio = ratio;
        }
    } else {
        // RMS-based: scale = (mean_rms/channel_rms)^strength
        float total_rms = 0.0f;
        float channel_rms[128];
        for (int i = 0; i < 128; i++) {
            channel_rms[i] = sqrtf(sq[i] / count);
            total_rms += channel_rms[i];
        }
        float mean_rms = total_rms / 128.0f;

        for (int i = 0; i < 128; i++) {
            if (channel_rms[i] > 1e-10f) {
                float raw = mean_rms / channel_rms[i];
                float s = powf(raw, strength);
                if (s > max_clamp) s = max_clamp;
                if (s < 1.0f / max_clamp) s = 1.0f / max_clamp;
                scale[i] = s;
                scale_inv[i] = 1.0f / s;
            } else {
                scale[i] = 1.0f;
                scale_inv[i] = 1.0f;
            }
            float ratio = fmaxf(scale[i], 1.0f / scale[i]);
            if (ratio > max_ratio) max_ratio = ratio;
        }
    }

    fprintf(stderr, "InnerQ calibration: %d tokens, mode=%d, strength=%.2f, max scale ratio: %.3f (clamped to %.1f)\n",
            count, mode, strength, max_ratio, max_clamp);

    // Fail-loud (2026-07-05 audit): the cross-TU scale push reaches set-rows.cu + fattn.cu only.
    // The fused-MMA template TUs and any other decode instantiations keep their compile-time
    // identity copies — the fattn-common.cuh promise to "gate the fused turbo1_tcq path off if
    // calibration is ever revived" was never implemented. Until it is, armed non-identity scales
    // make fused turbo1_tcq decode silently wrong (identity inv-scale on scaled-domain K).
    if (max_ratio >= 1.2f) {
        // (defined in set-rows.cu)
        extern bool g_turbo_innerq_calibrated;
        // the fused turbo1_tcq decode path is NOT scale-aware (per-TU symbol never updated);
        // g_turbo_innerq_calibrated makes the fattn dispatch route turbo1_tcq off the fused
        // path onto the materialize path (which uses the pushed scales) while calibrated
        g_turbo_innerq_calibrated = true;
        fprintf(stderr, "InnerQ: non-identity scales armed — turbo1_tcq decode will use the "
                "materialize path instead of the fused kernel (fused is not scale-aware)\n");
    }

    // Auto-detect: if channels are already well-balanced, InnerQ won't help — skip.
    // TURBO_INNERQ_FORCE=1 bypasses this PPL-era heuristic so low-strength scales
    // (e.g. strength 0.2 → ratio<1.2) can be measured on KLD/hazard/trajectory.
    static const char * force_env = getenv("TURBO_INNERQ_FORCE");
    bool force = force_env && atoi(force_env) > 0;
    if (max_ratio < 1.2f && !force) {
        fprintf(stderr, "InnerQ: max ratio %.3f < 1.2 — channels already balanced, disabling (would hurt quality)\n", max_ratio);
        float ones[128];
        for (int i = 0; i < 128; i++) ones[i] = 1.0f;
        for (int id = 0; id < device_count; id++) {
            cudaSetDevice(id);
            cudaMemcpyToSymbol(d_innerq_channel_scale, ones, sizeof(ones));
            cudaMemcpyToSymbol(d_innerq_channel_scale_inv, ones, sizeof(ones));
        }
        cudaSetDevice(cur_device);
        turbo_innerq_update_fattn_scales(ones);
        return;
    }

    // Print top-5 most affected channels
    float scale_copy[128];
    for (int i = 0; i < 128; i++) scale_copy[i] = scale[i];
    for (int k = 0; k < 5; k++) {
        float best = 0; int best_i = -1;
        for (int i = 0; i < 128; i++) {
            float r = fabsf(logf(scale_copy[i]));
            if (r > best) { best = r; best_i = i; }
        }
        if (best_i >= 0) {
            fprintf(stderr, "  channel %d: scale=%.4f (max=%.6f, rms=%.6f)\n",
                    best_i, scale[best_i], ch_max[best_i], sqrtf(sq[best_i] / count));
            scale_copy[best_i] = 1.0f; // mark as printed
        }
    }

    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale, scale, sizeof(scale));
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv, scale_inv, sizeof(scale_inv));
    }
    cudaSetDevice(cur_device);
    turbo_innerq_update_fattn_scales(scale_inv);
}

// === Shared constants ===
#if TURBO3_SKEW_EXP == 1
// data-retrained symmetric Lloyd-8 (pooled 27B post-FWHT K+V rows, cert_dump_27b)
static __constant__ float d_turbo_centroids_3bit[8] = {
    -0.187204f, -0.117719f, -0.066485f, -0.021602f,
     0.021602f,  0.066485f,  0.117719f,  0.187204f
};
static __constant__ float d_turbo_mid_3bit[7] = {
    -0.152461f, -0.092102f, -0.044044f, 0.0f, 0.044044f, 0.092102f, 0.152461f
};
#elif TURBO3_SKEW_EXP == 2
// skew-sign-aligned asymmetric Lloyd-8 (same calibration, coords aligned by ss tables below)
static __constant__ float d_turbo_centroids_3bit[8] = {
    -0.186344f, -0.117092f, -0.065915f, -0.021100f,
     0.022043f,  0.066992f,  0.118394f,  0.188239f
};
static __constant__ float d_turbo_mid_3bit[7] = {
    -0.151718f, -0.091504f, -0.043508f, 0.000471f, 0.044518f, 0.092693f, 0.153316f
};
#else // 0 = stock, 3 = oracle (stock book, variant-2 code paths)
static __constant__ float d_turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float d_turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};
#endif
#if TURBO3_SKEW_EXP == 2
// per-side global skew signs: sign of median train-split skew per post-FWHT coordinate
static __constant__ float d_turbo3_ss_k[128] = {
    -1,+1,+1,+1,-1,+1,-1,-1,-1,+1,-1,+1,+1,-1,-1,+1,-1,-1,+1,+1,+1,-1,-1,+1,+1,-1,-1,-1,+1,-1,+1,-1,
    -1,+1,-1,-1,+1,+1,+1,+1,+1,+1,+1,+1,-1,+1,+1,+1,-1,-1,+1,-1,+1,+1,-1,-1,+1,-1,-1,-1,-1,-1,-1,-1,
    +1,-1,-1,+1,-1,-1,-1,+1,+1,+1,-1,-1,-1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,-1,+1,-1,-1,-1,-1,-1,
    -1,-1,+1,-1,+1,+1,-1,+1,-1,+1,-1,-1,-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1,+1,+1,+1,+1,-1,+1,+1,-1};
static __constant__ float d_turbo3_ss_v[128] = {
    +1,+1,-1,+1,-1,-1,+1,+1,-1,-1,+1,+1,+1,+1,-1,+1,+1,+1,-1,+1,-1,+1,-1,-1,+1,+1,-1,-1,-1,+1,+1,+1,
    +1,-1,+1,-1,+1,+1,-1,-1,-1,-1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1,+1,-1,+1,+1,-1,+1,+1,-1,
    -1,-1,+1,+1,-1,+1,-1,+1,+1,-1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1,-1,-1,-1,+1,-1,+1,-1,-1,+1,+1,+1,-1,
    -1,+1,-1,-1,-1,-1,+1,-1,+1,+1,+1,+1,-1,-1,-1,+1,+1,-1,+1,+1,-1,-1,+1,-1,-1,+1,+1,-1,+1,-1,-1,-1};
#elif TURBO3_SKEW_EXP == 3
static __constant__ float d_turbo3_ss_k[128] = {
    +1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,
    +1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,
    +1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,
    +1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1};
static __constant__ float d_turbo3_ss_v[128] = {
    +1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,
    +1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,
    +1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,
    +1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1};
#endif

// === TURBO2: 2-bit codebook (Lloyd-Max for N(0, 1/128)) ===
static __constant__ float d_turbo_centroids_2bit[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
static __constant__ float d_turbo_mid_2bit[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// === TURBO4: 4-bit codebook (Lloyd-Max for N(0, 1/sqrt(128))) ===
static __constant__ float d_turbo_centroids_4bit[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
static __constant__ float d_turbo_mid_4bit[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

// Runtime vanilla-book override (encode-side twin of turbo_vanilla_cb_load_fattn; see
// fattn-common.cuh). Overriding turbo8 also arms d_turbo8_cb_override: the stock encoder is
// a uniform-grid round which never consults the book, so a companded book switches encode to
// a binary search over d_turbo_mid_8bit.
static __constant__ float d_turbo_mid_8bit[255] = {};
static __constant__ int   d_turbo8_cb_override = 0;

// === TURBO8: 8-bit codebook (uniform grid centroid[i]=(i-127.5)/127.5 in [-1,1]; per-block absmax scale in norm) ===
static __constant__ float d_turbo_centroids_8bit[256] = {
    -1.00000000f, -0.99215686f, -0.98431373f, -0.97647059f, -0.96862745f, -0.96078431f, -0.95294118f, -0.94509804f,
    -0.93725490f, -0.92941176f, -0.92156863f, -0.91372549f, -0.90588235f, -0.89803922f, -0.89019608f, -0.88235294f,
    -0.87450980f, -0.86666667f, -0.85882353f, -0.85098039f, -0.84313725f, -0.83529412f, -0.82745098f, -0.81960784f,
    -0.81176471f, -0.80392157f, -0.79607843f, -0.78823529f, -0.78039216f, -0.77254902f, -0.76470588f, -0.75686275f,
    -0.74901961f, -0.74117647f, -0.73333333f, -0.72549020f, -0.71764706f, -0.70980392f, -0.70196078f, -0.69411765f,
    -0.68627451f, -0.67843137f, -0.67058824f, -0.66274510f, -0.65490196f, -0.64705882f, -0.63921569f, -0.63137255f,
    -0.62352941f, -0.61568627f, -0.60784314f, -0.60000000f, -0.59215686f, -0.58431373f, -0.57647059f, -0.56862745f,
    -0.56078431f, -0.55294118f, -0.54509804f, -0.53725490f, -0.52941176f, -0.52156863f, -0.51372549f, -0.50588235f,
    -0.49803922f, -0.49019608f, -0.48235294f, -0.47450980f, -0.46666667f, -0.45882353f, -0.45098039f, -0.44313725f,
    -0.43529412f, -0.42745098f, -0.41960784f, -0.41176471f, -0.40392157f, -0.39607843f, -0.38823529f, -0.38039216f,
    -0.37254902f, -0.36470588f, -0.35686275f, -0.34901961f, -0.34117647f, -0.33333333f, -0.32549020f, -0.31764706f,
    -0.30980392f, -0.30196078f, -0.29411765f, -0.28627451f, -0.27843137f, -0.27058824f, -0.26274510f, -0.25490196f,
    -0.24705882f, -0.23921569f, -0.23137255f, -0.22352941f, -0.21568627f, -0.20784314f, -0.20000000f, -0.19215686f,
    -0.18431373f, -0.17647059f, -0.16862745f, -0.16078431f, -0.15294118f, -0.14509804f, -0.13725490f, -0.12941176f,
    -0.12156863f, -0.11372549f, -0.10588235f, -0.09803922f, -0.09019608f, -0.08235294f, -0.07450980f, -0.06666667f,
    -0.05882353f, -0.05098039f, -0.04313725f, -0.03529412f, -0.02745098f, -0.01960784f, -0.01176471f, -0.00392157f,
    0.00392157f, 0.01176471f, 0.01960784f, 0.02745098f, 0.03529412f, 0.04313725f, 0.05098039f, 0.05882353f,
    0.06666667f, 0.07450980f, 0.08235294f, 0.09019608f, 0.09803922f, 0.10588235f, 0.11372549f, 0.12156863f,
    0.12941176f, 0.13725490f, 0.14509804f, 0.15294118f, 0.16078431f, 0.16862745f, 0.17647059f, 0.18431373f,
    0.19215686f, 0.20000000f, 0.20784314f, 0.21568627f, 0.22352941f, 0.23137255f, 0.23921569f, 0.24705882f,
    0.25490196f, 0.26274510f, 0.27058824f, 0.27843137f, 0.28627451f, 0.29411765f, 0.30196078f, 0.30980392f,
    0.31764706f, 0.32549020f, 0.33333333f, 0.34117647f, 0.34901961f, 0.35686275f, 0.36470588f, 0.37254902f,
    0.38039216f, 0.38823529f, 0.39607843f, 0.40392157f, 0.41176471f, 0.41960784f, 0.42745098f, 0.43529412f,
    0.44313725f, 0.45098039f, 0.45882353f, 0.46666667f, 0.47450980f, 0.48235294f, 0.49019608f, 0.49803922f,
    0.50588235f, 0.51372549f, 0.52156863f, 0.52941176f, 0.53725490f, 0.54509804f, 0.55294118f, 0.56078431f,
    0.56862745f, 0.57647059f, 0.58431373f, 0.59215686f, 0.60000000f, 0.60784314f, 0.61568627f, 0.62352941f,
    0.63137255f, 0.63921569f, 0.64705882f, 0.65490196f, 0.66274510f, 0.67058824f, 0.67843137f, 0.68627451f,
    0.69411765f, 0.70196078f, 0.70980392f, 0.71764706f, 0.72549020f, 0.73333333f, 0.74117647f, 0.74901961f,
    0.75686275f, 0.76470588f, 0.77254902f, 0.78039216f, 0.78823529f, 0.79607843f, 0.80392157f, 0.81176471f,
    0.81960784f, 0.82745098f, 0.83529412f, 0.84313725f, 0.85098039f, 0.85882353f, 0.86666667f, 0.87450980f,
    0.88235294f, 0.89019608f, 0.89803922f, 0.90588235f, 0.91372549f, 0.92156863f, 0.92941176f, 0.93725490f,
    0.94509804f, 0.95294118f, 0.96078431f, 0.96862745f, 0.97647059f, 0.98431373f, 0.99215686f, 1.00000000f,
};
// === FWHT rotation sign arrays (from turbo-wht.h, seed=42 rotation, seed=1042 QJL) ===
static __constant__ float d_turbo_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
static __constant__ float d_turbo_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};
// QJL sign arrays removed — turbo4 now uses pure 4-bit PolarQuant (no QJL correction)

// Encode-side runtime vanilla-book override: TURBO_CB_T2/T3/T4/T8 (raw f32, ascending, exact
// count 4/8/16/256). Updates THIS TU's centroid + midpoint __constant__ copies; for turbo8 it
// also uploads 255 mids and arms d_turbo8_cb_override (stock encode is a uniform round that
// never reads the book). static inline per-TU, same pattern as turbo_tcq_load_kv_encode.
static inline void turbo_vanilla_cb_load_encode() {
    static const char * p2 = getenv("TURBO_CB_T2");
    static const char * p3 = getenv("TURBO_CB_T3");
    static const char * p4 = getenv("TURBO_CB_T4");
    static const char * p8 = getenv("TURBO_CB_T8");
    if (!p2 && !p3 && !p4 && !p8) return;
    int dev = 0; cudaGetDevice(&dev);
    static bool done[GGML_CUDA_MAX_DEVICES] = {};
    if (done[dev]) return;
    done[dev] = true;
    auto load = [](const char * p, const void * cent_sym, const void * mid_sym, int n, const char * name) {
        if (!p || !p[0]) return false;
        float buf[256], mids[255];
        FILE * f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "TURBO_CB: cannot open %s\n", p); return false; }
        const bool ok = fread(buf, sizeof(float), n, f) == (size_t) n;
        // optional trailing n-1 EXACT mids (else computed midpoints; a 5e-7 boundary shift is
        // quality-neutral but breaks bit-compat with the compiled tables via KV chaos)
        const bool mids_in_file = ok && fread(mids, sizeof(float), n - 1, f) == (size_t) (n - 1);
        fclose(f);
        if (!ok) { fprintf(stderr, "TURBO_CB: short file %s (need %d f32)\n", p, n); return false; }
        if (!mids_in_file) {
            for (int i = 0; i + 1 < n; i++) mids[i] = 0.5f * (buf[i] + buf[i + 1]);
        }
        cudaMemcpyToSymbol(cent_sym, buf, n * sizeof(float));
        if (mid_sym) cudaMemcpyToSymbol(mid_sym, mids, (n - 1) * sizeof(float));
        fprintf(stderr, "TURBO_CB: %s encode book <- %s (mids %s)\n", name, p, mids_in_file ? "from file" : "computed");
        return true;
    };
    load(p2, d_turbo_centroids_2bit, d_turbo_mid_2bit, 4,  "turbo2");
    load(p3, d_turbo_centroids_3bit, d_turbo_mid_3bit, 8,  "turbo3");
    load(p4, d_turbo_centroids_4bit, d_turbo_mid_4bit, 16, "turbo4");
    if (load(p8, d_turbo_centroids_8bit, d_turbo_mid_8bit, 256, "turbo8")) {
        const int one = 1;
        cudaMemcpyToSymbol(d_turbo8_cb_override, &one, sizeof(int));
    }
}

// === FWHT rotation functions ===
static __device__ __forceinline__
void turbo_fwht_128_cuda(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

// Forward rotation: signs1 → FWHT → signs2
static __device__ __forceinline__
void turbo_rotate_forward_cuda(float * x, const float * s1, const float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    turbo_fwht_128_cuda(x);
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
}

static __device__ __forceinline__
uint8_t turbo_find_nearest_3bit(float val) {
    if      (val < d_turbo_mid_3bit[0]) return 0;
    else if (val < d_turbo_mid_3bit[1]) return 1;
    else if (val < d_turbo_mid_3bit[2]) return 2;
    else if (val < d_turbo_mid_3bit[3]) return 3;
    else if (val < d_turbo_mid_3bit[4]) return 4;
    else if (val < d_turbo_mid_3bit[5]) return 5;
    else if (val < d_turbo_mid_3bit[6]) return 6;
    else                                return 7;
}

static __device__ __forceinline__
uint8_t turbo_find_nearest_4bit(float val) {
    // Binary search over 15 midpoints for 16 centroids
    if (val < d_turbo_mid_4bit[7]) {
        if (val < d_turbo_mid_4bit[3]) {
            if (val < d_turbo_mid_4bit[1]) {
                return val < d_turbo_mid_4bit[0] ? 0 : 1;
            } else {
                return val < d_turbo_mid_4bit[2] ? 2 : 3;
            }
        } else {
            if (val < d_turbo_mid_4bit[5]) {
                return val < d_turbo_mid_4bit[4] ? 4 : 5;
            } else {
                return val < d_turbo_mid_4bit[6] ? 6 : 7;
            }
        }
    } else {
        if (val < d_turbo_mid_4bit[11]) {
            if (val < d_turbo_mid_4bit[9]) {
                return val < d_turbo_mid_4bit[8] ? 8 : 9;
            } else {
                return val < d_turbo_mid_4bit[10] ? 10 : 11;
            }
        } else {
            if (val < d_turbo_mid_4bit[13]) {
                return val < d_turbo_mid_4bit[12] ? 12 : 13;
            } else {
                return val < d_turbo_mid_4bit[14] ? 14 : 15;
            }
        }
    }
}

// === TURBO3: SET_ROWS kernel ===
template<typename idx_t>
static __global__ void k_set_rows_turbo3(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int innerq_is_k, const float * __restrict__ kmean_mu, const int pf_layer,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO3_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO3_GROUP;
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }
    // InnerQ: calibrate from both K and V, apply scaling to both
    if (d_innerq_calibrate) {
        for (int j = 0; j < 128; j++) {
            atomicAdd(&d_innerq_channel_sq[j], x[j] * x[j]);
            float abs_val = fabsf(x[j]);
            // atomicMax for float: CAS loop (no native float atomicMax)
            unsigned int * addr = (unsigned int *)&d_innerq_channel_max[j];
            unsigned int old_val = __float_as_uint(abs_val);
            unsigned int assumed;
            do {
                assumed = *addr;
                if (__uint_as_float(assumed) >= abs_val) break;
            } while (atomicCAS(addr, assumed, old_val) != assumed);
        }
        atomicAdd(&d_innerq_count, 1);
        // PFHEAD probe: raw (pre-norm, pre-rotation) per-(kv,layer,channel) stats
        if (d_pfhead_sum != nullptr && pf_layer >= 0 && pf_layer < PFHEAD_MAX_L && i00 + 128 <= PFHEAD_MAX_C) {
            const int kvsel = innerq_is_k ? 0 : 1;
            const size_t base = ((size_t) kvsel * PFHEAD_MAX_L + pf_layer) * PFHEAD_MAX_C + i00;
            for (int j = 0; j < 128; j++) {
                atomicAdd(&d_pfhead_sum[base + j], x[j]);
                atomicAdd(&d_pfhead_sq[base + j],  x[j] * x[j]);
            }
            if (i00 == 0) atomicAdd(&d_pfhead_cnt[kvsel * PFHEAD_MAX_L + pf_layer], 1);
        }
    }
    // K-mean subtract (raw domain; probe above intentionally sees PRE-subtract values)
    if (kmean_mu != nullptr) {
        for (int j = 0; j < 128; j++) x[j] -= kmean_mu[i00 + j];
    }
    for (int j = 0; j < 128; j++) x[j] *= d_innerq_channel_scale[j];
    norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += x[j] * x[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    // Track B probe (#135): K-only post-FWHT per-bin stats during InnerQ calibration (identity scale).
    if (d_innerq_calibrate && innerq_is_k) {
        for (int j = 0; j < 128; j++) {
            atomicAdd(&d_postfwht_bin_sum[j], x[j]);
            atomicAdd(&d_postfwht_bin_sq[j], x[j] * x[j]);
        }
        atomicAdd(&d_postfwht_count, 1);
    }
    // Post-rotation extraction (if enabled); tag = layer*1000 + kvsel*100 + group
    turbo_extract_append(x, pf_layer >= 0
        ? pf_layer * 1000 + (innerq_is_k ? 0 : 100) + (int)(i00 / 128) : -1);
    // Quantize and accumulate reconstruction norm for correction
#if TURBO3_SKEW_EXP >= 2
    const float * t3ss = innerq_is_k ? d_turbo3_ss_k : d_turbo3_ss_v;
#endif
    float recon_norm_sq = 0.0f;
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo3_0 & blk = dst_row_ptr[grp_idx * blocks_per_group + b];
        const int off = b * QK_TURBO3;
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk.signs[j] = 0;
        for (int j = 0; j < QK_TURBO3; j++) {
#if TURBO3_SKEW_EXP >= 2
            uint8_t idx = turbo_find_nearest_3bit(x[off + j] * t3ss[off + j]);
#else
            uint8_t idx = turbo_find_nearest_3bit(x[off + j]);
#endif
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));
            float c = d_turbo_centroids_3bit[idx];
            recon_norm_sq += c * c;
        }
    }
    // Norm correction: store corrected norm so dequant(x) has exact original L2 norm
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) {
        dst_row_ptr[grp_idx * blocks_per_group + b].norm = __float2half(corrected_norm);
    }
}

// === TURBO3: GET_ROWS dequantize ===
#define QR_TURBO3_0 2
static __device__ __forceinline__
void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
      v.x = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm; }
    { const int j = iqs + 16;
      const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
      v.y = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm; }
}

// === TURBO3: per-128-group quantize (for the ragged roundtrip harness) ===
// Mirrors k_set_rows_turbo3's math on a single 128-coord group (= QK_TURBO3_GROUP =
// head_dim = 4 sub-blocks of QK_TURBO3). No innerq atomics (read-only channel scale,
// identity by default) so it never mutates calibration globals.
static __device__ __forceinline__
void quantize_f32_turbo3_0_block(const float * src, block_turbo3_0 * dst, const int is_k = 1) {
    GGML_UNUSED(is_k);
    // NOTE: no InnerQ channel-scale here. d_innerq_channel_scale is only initialized
    // (to 1.0) by turbo_innerq_init(), which runs only when a turbo KV type is active.
    // The ragged injector runs with -ctk f16, so that global stays zero-initialized;
    // multiplying by it would zero the reconstruction. InnerQ is never calibrated in
    // these sweeps, so identity (no scale) is the faithful turbo3 behavior here.
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j];
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += x[j] * x[j];
    const float grp_norm = sqrtf(norm_sq);
    const float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
#if TURBO3_SKEW_EXP >= 2
    const float * t3ss = is_k ? d_turbo3_ss_k : d_turbo3_ss_v;
#endif
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;
    float recon_norm_sq = 0.0f;
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo3_0 & blk = dst[b];
        const int off = b * QK_TURBO3;
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk.signs[j] = 0;
        for (int j = 0; j < QK_TURBO3; j++) {
#if TURBO3_SKEW_EXP >= 2
            uint8_t idx = turbo_find_nearest_3bit(x[off + j] * t3ss[off + j]);
#else
            uint8_t idx = turbo_find_nearest_3bit(x[off + j]);
#endif
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));
            const float c = d_turbo_centroids_3bit[idx];
            recon_norm_sq += c * c;
        }
    }
    const float recon_norm = sqrtf(recon_norm_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) dst[b].norm = __float2half(corrected_norm);
}

// Temperature scaling for turbo4. KLD-optimal: α=1.0 (any scaling hurts KLD).
// Override via TURBO4_ALPHA env var.
static __constant__ float d_turbo4_alpha = 1.0f;

// === TURBO4: SET_ROWS quantize (4-bit PolarQuant, no QJL) ===
static __device__ __forceinline__
void quantize_f32_turbo4_0_block(const float * src, block_turbo4_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    // Forward FWHT rotation before quantization
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    // Post-rotation extraction (if enabled)
    turbo_extract_append(x);
    // 4-bit quantization: find nearest of 16 centroids, pack 2 per byte
    for (int j = 0; j < 128; j += 2) {
        uint8_t idx0 = turbo_find_nearest_4bit(x[j]);
        uint8_t idx1 = turbo_find_nearest_4bit(x[j + 1]);
        dst->qs[j / 2] = (idx1 << 4) | idx0;
    }
    // Norm correction: compute reconstruction norm in rotated space
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = (j & 1) ? (dst->qs[j / 2] >> 4) : (dst->qs[j / 2] & 0xF);
        float r = d_turbo_centroids_4bit[idx];
        recon_sq += r * r;
    }
    float recon_norm = sqrtf(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
    dst->norm = __float2half(corrected * d_turbo4_alpha);
}

// === TURBO4: GET_ROWS dequantize ===
#define QR_TURBO4_0 2
static __device__ __forceinline__
void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
      v.x = d_turbo_centroids_4bit[idx] * norm; }
    { const int j = iqs + 64;
      uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
      v.y = d_turbo_centroids_4bit[idx] * norm; }
}

// === TURBO8: SET_ROWS quantize (8-bit PolarQuant, no QJL) ===
static __device__ __forceinline__
void quantize_f32_turbo8_0_block(const float * src, block_turbo8_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    // Forward FWHT rotation before quantization
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    // Post-rotation extraction (if enabled)
    turbo_extract_append(x);
    // 8-bit quantization: rotated-Q8 = per-block absmax scale + uniform 256-level grid.
    // The Hadamard rotation suppresses outlier coordinates, so a uniform grid in the
    // rotated domain beats a fixed-Gaussian codebook (no tail clamping). Scale is stored
    // in dst->norm; centroids are the fixed uniform [-1,1] table.
    float absmax = 0.0f;
    for (int j = 0; j < 128; j++) absmax = fmaxf(absmax, fabsf(x[j]));
    float scale = absmax > 1e-10f ? absmax : 1e-10f;
    float inv_scale = 1.0f / scale;
    if (d_turbo8_cb_override) {
        // Companded book (TURBO_CB_T8): nearest centroid via lower-bound over the 255 mids
        // (the uniform round below assumes the stock evenly-spaced grid).
        for (int j = 0; j < 128; j++) {
            const float v = x[j] * inv_scale;
            int idx = 0;
            #pragma unroll
            for (int step = 128; step >= 1; step >>= 1) {
                const int cand = idx + step;
                if (cand <= 255 && d_turbo_mid_8bit[cand - 1] < v) idx = cand;
            }
            dst->qs[j] = (uint8_t)idx;
        }
    } else {
        for (int j = 0; j < 128; j++) {
            int idx = (int)lrintf(x[j] * inv_scale * 127.5f + 127.5f);
            idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
            dst->qs[j] = (uint8_t)idx;
        }
    }
    dst->norm = __float2half(norm * scale);
}

// === TURBO8: GET_ROWS dequantize ===
#define QR_TURBO8_0 2
static __device__ __forceinline__
void dequantize_turbo8_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo8_0 * x = (const block_turbo8_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    v.x = d_turbo_centroids_8bit[x[ib].qs[iqs]]      * norm;
    v.y = d_turbo_centroids_8bit[x[ib].qs[iqs + 64]] * norm;
}

// === RAGGED reconstruct-to-fp16 quality harness (EXP-16) ===
// A static (position, layer, K/V) -> precision-tier schedule, applied at KV write.
// Degraded rows are round-tripped quant->dequant and stored as f16; protected rows
// stay exact f16. No ragged storage / no mixed-dtype fattn — this measures the
// QUALITY (KLD) cost of per-row precision only. tier codes: 16=f16 (lossless),
// 8=turbo8, 4=turbo4, 3=t3, 2=t2, 11=t1tcq, 22=t2tcq, 33=t3tcq.
// kv: 0=any, 1=K-only, 2=V-only. Last matching band wins.
#define RAGGED_MAX_BANDS 768
struct ragged_band { int pos_lo, pos_hi, lay_lo, lay_hi, cb_lo, cb_hi, kv, tier; };
static __constant__ ragged_band d_ragged_bands[RAGGED_MAX_BANDS];
static __constant__ int d_ragged_nbands = 0;
static __constant__ int d_ragged_default_tier = 16;

// Content/token axis (EXP-15d): optional per-(window,position) tier mask in global
// memory. d_ragged_window = current KLD chunk index (set per-chunk from perplexity).
// A non-zero mask byte seeds the tier (overriding default); positional bands below
// still override it (last-match-wins) so wall bands protect sink/recent on top.
static __device__ const unsigned char * d_ragged_cmask = nullptr;
static __constant__ int d_ragged_cmask_nctx = 0;
static __constant__ int d_ragged_window      = 0;

static __device__ __forceinline__
int ragged_lookup_tier(int64_t pos, int layer, int is_k, int cb) {
    int tier = d_ragged_default_tier;
    if (d_ragged_cmask != nullptr && d_ragged_cmask_nctx > 0 && pos < d_ragged_cmask_nctx) {
        const unsigned char ct = d_ragged_cmask[(int64_t) d_ragged_window * d_ragged_cmask_nctx + pos];
        if (ct) tier = ct;
    }
    for (int i = 0; i < d_ragged_nbands; i++) {
        const ragged_band b = d_ragged_bands[i];
        if (pos < b.pos_lo || pos >= b.pos_hi) continue;
        if (layer < b.lay_lo || layer > b.lay_hi) continue;
        if (cb < b.cb_lo || cb >= b.cb_hi) continue;
        if (b.kv == 1 && !is_k) continue;
        if (b.kv == 2 &&  is_k) continue;
        tier = b.tier;
    }
    return tier;
}

// Quantize one 128-coord block to `tier`, then dequantize back to ORIGINAL space
// (inverse FWHT = forward rotation with the sign arrays swapped, since the
// normalized FWHT is an involution). Storing the original-space reconstruction as
// f16 makes a standard f16 Q.K reproduce the real rotated-space turbo score:
// Q . R^-1(norm*c) == (HQ) . (norm*c).
static __device__ __forceinline__
void turbo_roundtrip_block_to_orig(const float * src, float * out, int tier, int is_k) {
    float xr[128];
    float post_norm = 1.0f;
    if (tier == 8) {
        block_turbo8_0 blk;
        quantize_f32_turbo8_0_block(src, &blk);
        post_norm = __half2float(blk.norm);
        for (int j = 0; j < 128; j++) {
            xr[j] = d_turbo_centroids_8bit[blk.qs[j]];
        }
    } else if (tier == 4) {
        block_turbo4_0 blk;
        quantize_f32_turbo4_0_block(src, &blk);
        post_norm = __half2float(blk.norm);
        for (int j = 0; j < 128; j++) {
            const uint8_t idx = (j & 1) ? (blk.qs[j / 2] >> 4) : (blk.qs[j / 2] & 0xF);
            xr[j] = d_turbo_centroids_4bit[idx];
        }
    } else if (tier == 2) { // 2-bit, shared 128-group norm (mirror of the turbo2 set_rows kernel,
                            // inlined here because turbo_find_nearest_2bit is defined further down).
        float x2[128];
        float nsq = 0.0f;
        for (int j = 0; j < 128; j++) { x2[j] = src[j]; nsq += x2[j] * x2[j]; }
        const float gnorm = sqrtf(nsq);
        const float invn = gnorm > 1e-10f ? 1.0f / gnorm : 0.0f;
        for (int j = 0; j < 128; j++) x2[j] *= invn;
        turbo_rotate_forward_cuda(x2, d_turbo_wht_signs1, d_turbo_wht_signs2);
        uint8_t idxs[128];
        float rnsq = 0.0f;
        for (int j = 0; j < 128; j++) {
            const float val = x2[j];
            const uint8_t idx = (val < d_turbo_mid_2bit[0]) ? 0 :
                                (val < d_turbo_mid_2bit[1]) ? 1 :
                                (val < d_turbo_mid_2bit[2]) ? 2 : 3;
            idxs[j] = idx;
            const float c = d_turbo_centroids_2bit[idx];
            rnsq += c * c;
        }
        const float rnorm = sqrtf(rnsq);
        const float cnorm = (rnorm > 1e-10f) ? gnorm / rnorm : gnorm;
        post_norm = cnorm;
        for (int j = 0; j < 128; j++) xr[j] = d_turbo_centroids_2bit[idxs[j]];
    } else { // tier == 3 (4 sub-blocks of QK_TURBO3, shared group norm)
        block_turbo3_0 blk[QK_TURBO3_GROUP / QK_TURBO3];
        quantize_f32_turbo3_0_block(src, blk, is_k);
        post_norm = __half2float(blk[0].norm);
#if TURBO3_SKEW_EXP >= 2
        const float * t3ss = is_k ? d_turbo3_ss_k : d_turbo3_ss_v;
#endif
        for (int ib = 0; ib < QK_TURBO3_GROUP / QK_TURBO3; ib++) {
            for (int j = 0; j < QK_TURBO3; j++) {
                const uint8_t low2 = (blk[ib].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
                const uint8_t hi1  = (blk[ib].signs[j / 8] >> (j % 8)) & 0x1;
#if TURBO3_SKEW_EXP >= 2
                xr[ib * QK_TURBO3 + j] = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * t3ss[ib * QK_TURBO3 + j];
#else
                xr[ib * QK_TURBO3 + j] = d_turbo_centroids_3bit[low2 | (hi1 << 2)];
#endif
            }
        }
    }
    turbo_rotate_forward_cuda(xr, d_turbo_wht_signs2, d_turbo_wht_signs1);
    for (int j = 0; j < 128; j++) {
        const float v = xr[j] * post_norm;
        out[j] = is_k ? v * d_innerq_channel_scale_inv[j] : v;
    }
}

// === TURBO2: find nearest 2-bit centroid ===
static __device__ __forceinline__
uint8_t turbo_find_nearest_2bit(float val) {
    if      (val < d_turbo_mid_2bit[0]) return 0;
    else if (val < d_turbo_mid_2bit[1]) return 1;
    else if (val < d_turbo_mid_2bit[2]) return 2;
    else                                return 3;
}

// === TURBO2: SET_ROWS kernel ===
template<typename idx_t>
static __global__ void k_set_rows_turbo2(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo2_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const float * __restrict__ kmean_mu,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO2_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo2_0 * dst_row_ptr = (block_turbo2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO2_GROUP;
    const int blocks_per_group = QK_TURBO2_GROUP / QK_TURBO2;
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }
    // K-mean subtract (raw domain): re-derive the norm from the centered vector
    if (kmean_mu != nullptr) {
        norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) { x[j] -= kmean_mu[i00 + j]; norm_sq += x[j] * x[j]; }
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    float recon_norm_sq = 0.0f;
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo2_0 & blk = dst_row_ptr[grp_idx * blocks_per_group + b];
        const int off = b * QK_TURBO2;
        for (int j = 0; j < QK_TURBO2 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = turbo_find_nearest_2bit(x[off + j]);
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            float c = d_turbo_centroids_2bit[idx];
            recon_norm_sq += c * c;
        }
    }
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) {
        dst_row_ptr[grp_idx * blocks_per_group + b].norm = __float2half(corrected_norm);
    }
}

// === TURBO2: GET_ROWS dequantize ===
#define QR_TURBO2_0 2
static __device__ __forceinline__
void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_0 * x = (const block_turbo2_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      const uint8_t idx = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      v.x = d_turbo_centroids_2bit[idx] * norm; }
    { const int j = iqs + 16;
      const uint8_t idx = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      v.y = d_turbo_centroids_2bit[idx] * norm; }
}

// Optional SEPARATE V-side codebook (K/V-split). Has its OWN compiled-in default now
// (copy-K->V disabled); TURBO_TCQ_CB_V overrides only V. K-encode reads
// d_turbo3_tcq_codebook, V-encode reads this, selected by innerq_is_k in k_set_rows_turbo3_tcq.
static __constant__ float d_turbo3_tcq_codebook_v[512] = {
    -0.176876873f, -0.118416861f, -0.0761471018f, -0.0447181612f, -0.00392233999f, 0.0489111096f, 0.0902970433f, 0.14635466f,
    -0.0946576968f, -0.0559962243f, -0.0302548725f, -0.00884746574f, -0.00884746574f, 0.0200079326f, 0.0577601604f, 0.103985772f,
    -0.135598078f, -0.0559968352f, 0.00828930736f, 0.0361441784f, 0.0732201859f, 0.0817056298f, 0.127882868f, 0.193429515f,
    -0.236759409f, -0.0868931785f, -0.0432750694f, -0.0303228591f, -0.00696427235f, 0.024344895f, 0.0548498854f, 0.115451649f,
    -0.113649949f, -0.0759628788f, -0.0521383174f, -0.0193102583f, -0.00369788613f, 0.0146498419f, 0.0547891408f, 0.159904227f,
    -0.187377661f, -0.146724939f, -0.11061547f, -0.0791360736f, -0.0440946482f, -0.0144825559f, 0.0290816613f, 0.0884379297f,
    -0.0852011815f, -0.0430183522f, -0.0226644743f, 0.0112192193f, 0.0390752926f, 0.0640700087f, 0.095136717f, 0.13772805f,
    -0.12935853f, -0.0709433183f, -0.0357996412f, -0.00288573466f, -0.00288573466f, 0.028702762f, 0.0645379424f, 0.107880369f,
    -0.122470647f, -0.0739600733f, -0.0341991149f, -0.012296522f, 0.0199557431f, 0.0199557431f, 0.0573284514f, 0.104966536f,
    -0.159187943f, -0.115453534f, -0.0818956494f, -0.0462123491f, 0.024891993f, 0.0522041619f, 0.0975019932f, 0.238296285f,
    -0.105870768f, -0.0860663429f, -0.0606526136f, -0.0557363257f, -0.0218950529f, -0.00846894551f, 0.0232181922f, 0.0627226084f,
    -0.142055362f, -0.0353555679f, 0.00328380987f, 0.00736280065f, 0.0393852405f, 0.0863731205f, 0.117111653f, 0.15658322f,
    -0.21105361f, -0.185308263f, -0.148743793f, -0.124403708f, -0.100041501f, -0.0679302365f, -0.0365559161f, 0.0367332734f,
    -0.0598724149f, -0.0233231168f, 0.00645363238f, 0.032846421f, 0.0556079708f, 0.0674588457f, 0.0976238549f, 0.13645646f,
    -0.158521906f, -0.108558111f, -0.0875888988f, -0.0518983603f, -0.0518983603f, -0.0118839731f, 0.00715786405f, 0.0487830453f,
    -0.261381567f, -0.122490726f, -0.0688513964f, -0.0324650556f, 0.0306476038f, 0.091423966f, 0.145522714f, 0.246422932f,
    -0.0669574514f, -0.0203008801f, 0.0146694202f, 0.0404772274f, 0.0735744461f, 0.103208132f, 0.139817789f, 0.19168137f,
    -0.0362621024f, 0.00491102459f, 0.0346047506f, 0.0732331499f, 0.0920870006f, 0.124441206f, 0.166039392f, 0.166039392f,
    -0.172402173f, -0.0255705249f, -0.0149608441f, -0.000704373408f, 0.0287760161f, 0.0435062163f, 0.075672321f, 0.109859884f,
    -0.118631318f, -0.118631318f, -0.0834067687f, -0.0834067687f, -0.0757180378f, -0.0459860377f, -0.00961863343f, 0.192524537f,
    -0.0451413579f, -0.015006653f, -0.0141126225f, 0.022220308f, 0.0307657998f, 0.0550360456f, 0.090898402f, 0.090898402f,
    -0.102176085f, -0.101714268f, -0.0654631183f, -0.000976806739f, 0.125787318f, 0.138862163f, 0.172410414f, 0.21177949f,
    -0.0304783173f, 0.0170904659f, 0.031891875f, 0.0694930851f, 0.082852751f, 0.109130576f, 0.140266165f, 0.182268769f,
    -0.0764161423f, -0.0254436061f, 0.00673568621f, 0.035249494f, 0.0644336045f, 0.0644336045f, 0.106129512f, 0.158337772f,
    -0.242637336f, -0.182605371f, -0.142921135f, -0.10884326f, -0.0788128898f, -0.0418869741f, -0.00442521507f, 0.0476941764f,
    -0.205798537f, -0.166769832f, -0.134739369f, -0.102670468f, -0.0728637949f, -0.0441084094f, -0.000184274119f, 0.0309528895f,
    -0.231348604f, -0.156730846f, -0.113921493f, -0.0992795825f, -0.0700323954f, -0.0548418686f, -0.0192977265f, 0.141322881f,
    -0.183901057f, 0.00421330146f, 0.0358202942f, 0.0384667479f, 0.0662506521f, 0.0662506521f, 0.0845405012f, 0.0988950357f,
    -0.295811743f, -0.139857203f, -0.0983865187f, -0.0689445361f, -0.053064622f, -0.0200633779f, 0.00281952578f, 0.120041497f,
    -0.136180028f, -0.0714779943f, -0.0366355292f, -0.0366355292f, -1.92951138e-05f, 0.0177245941f, 0.0514235981f, 0.0514235981f,
    -0.206010208f, -0.174232557f, -0.145896226f, -0.102482997f, -0.00960603263f, 0.0195680279f, 0.0480228513f, 0.0852651894f,
    -0.198772699f, -0.163127035f, -0.124294125f, -0.0975355953f, -0.0722151697f, -0.0470228754f, -0.00971428398f, 0.0274044629f,
    -0.0274835415f, 0.0204150788f, 0.0602538288f, 0.0893707722f, 0.121699311f, 0.150838777f, 0.192401767f, 0.244853675f,
    -0.13003552f, -0.0866545662f, -0.050875809f, -0.0198723506f, 0.0427571721f, 0.0759131461f, 0.124128364f, 0.300333202f,
    -0.0694411695f, 0.0122253271f, 0.0534021296f, 0.0731122121f, 0.105984591f, 0.117304564f, 0.159797624f, 0.209410056f,
    -0.159880534f, -0.159880534f, -0.116101854f, -0.0558235161f, -0.0399357565f, 0.126363367f, 0.159986258f, 0.236862764f,
    -0.125637054f, -0.0840563178f, 0.0117021035f, 0.053172119f, 0.0627194121f, 0.0943006948f, 0.163924828f, 0.194789574f,
    -0.026679812f, 0.010760827f, 0.0351526402f, 0.0586721823f, 0.0839044973f, 0.100516967f, 0.130418167f, 0.283241332f,
    -0.107419722f, -0.0670032427f, -0.0468876511f, -0.011866997f, 0.128184676f, 0.155335605f, 0.191104874f, 0.223861858f,
    -0.0353680626f, 0.0153140118f, 0.0500409864f, 0.0821120739f, 0.106105357f, 0.134016007f, 0.179932058f, 0.209855989f,
    -0.185345352f, -0.137568921f, -0.100465529f, -0.0690446496f, -0.0387077071f, -0.00603976008f, 0.0298719648f, 0.0824758336f,
    -0.0753742978f, -0.0435122699f, -0.0184914321f, 0.00652743364f, 0.00652743364f, 0.0312343501f, 0.0476280004f, 0.0772584826f,
    -0.14677681f, -0.110104889f, -0.0493518412f, -0.0281198416f, 0.00044620171f, 0.0177182276f, 0.050046593f, 0.0943839774f,
    -0.211110413f, -0.0794639438f, -0.0290603545f, -0.0290603545f, 0.00660928711f, 0.00660928711f, 0.0268525928f, 0.0589293055f,
    -0.0928101912f, -0.0582146756f, -0.0582146756f, -0.0233178362f, -0.0137845641f, 0.0198776145f, 0.0345822908f, 0.0667714849f,
    -0.236260116f, -0.190895811f, -0.156597689f, -0.127286375f, -0.0949480087f, -0.0502674282f, -0.0116296364f, 0.0799051374f,
    -0.0628373548f, -0.0234580543f, -0.00987345912f, 0.0210197456f, 0.0210197456f, 0.0480055586f, 0.0741760954f, 0.110231586f,
    -0.14602527f, -0.111594357f, -0.0840363055f, -0.0591976494f, -0.0334885493f, -0.00440378161f, 0.0296884067f, 0.0708110705f,
    -0.0918308124f, -0.0466427691f, -0.0114560947f, 0.0151799079f, 0.0415129215f, 0.0692910627f, 0.106995568f, 0.15801689f,
    -0.265835822f, -0.136571661f, -0.0992405638f, -0.0686350614f, -0.0356153361f, -0.00820365362f, 0.0849370956f, 0.129136488f,
    -0.19236508f, -0.0411618203f, 0.0149009284f, 0.0260554794f, 0.057334438f, 0.0668766126f, 0.107671611f, 0.169883981f,
    -0.0867072493f, -0.0503898263f, -0.0503898263f, -0.0183659811f, -0.00998108741f, 0.0322193839f, 0.0626041442f, 0.125008464f,
    -0.166617364f, -0.114416778f, 0.0435903035f, 0.0955699086f, 0.0955699086f, 0.137931198f, 0.137931198f, 0.242053837f,
    -0.0753244832f, -0.039837461f, -0.00976719614f, 0.00926619302f, 0.0321972631f, 0.049129609f, 0.086447306f, 0.174170598f,
    -0.238970727f, -0.137454197f, -0.112297922f, -0.0728462711f, 0.0534135588f, 0.0809852704f, 0.113146149f, 0.145978987f,
    -0.100819319f, -0.0519368574f, -0.0221283212f, 0.00551196467f, 0.00551196467f, 0.0346855707f, 0.0704962984f, 0.11254327f,
    -0.150274724f, -0.0893646479f, -0.0556213371f, -0.027744934f, 0.000275628583f, 0.0316795707f, 0.0674656853f, 0.118501939f,
    -0.105402619f, -0.0134610916f, 0.0170810632f, 0.0476340391f, 0.0772455335f, 0.101922646f, 0.150536776f, 0.208813652f,
    -0.120642103f, -0.0812854692f, -0.064764753f, -0.0622290969f, -0.0327842906f, -0.0209947526f, 0.00940577686f, 0.0488117412f,
    -0.147846326f, -0.104877621f, 0.00389082986f, 0.0148656378f, 0.0483935587f, 0.0897327363f, 0.120465532f, 0.187486157f,
    -0.0633428469f, -0.0316850729f, -0.0209032353f, 0.0179677866f, 0.024466211f, 0.0616848618f, 0.0806632638f, 0.146979511f,
    -0.184542835f, -0.128439516f, -0.0914528146f, -0.0630969554f, -0.0317214839f, -0.00592412194f, 0.0331228226f, 0.109939829f,
    -0.142377883f, -0.0925493315f, -0.0724670216f, -0.0368435569f, -0.0368435569f, 0.00111511303f, 0.0178956445f, 0.0591671504f,
    -0.1816006f, -0.0911702141f, -0.0514943935f, -0.018047763f, 0.0375295356f, 0.0775823295f, 0.121360153f, 0.175192803f
};
// 3-bit TCQ codebook (coord-descent split K=iter374/V=iter500, 512-state bitshift trellis). If you copy these, credit spiritbuun!
// CUDA GLA product-aware training, 100 iters on Qwen3.5-27B FWHT-rotated KV activations. Decode: state_t = read_9_bits(qs, t*3)
static __constant__ float d_turbo3_tcq_codebook[512] = {
    -0.183893934f, -0.121672049f, -0.0791359544f, -0.046731215f, -0.00666991901f, 0.0475080945f, 0.0907080695f, 0.14690879f,
    -0.0979461521f, -0.0580380931f, -0.0316831581f, -0.0105782337f, -0.0105782337f, 0.0184551626f, 0.0560457781f, 0.107396744f,
    -0.140606418f, -0.0601783656f, 0.0111594656f, 0.0375199169f, 0.072943531f, 0.0865838081f, 0.132308871f, 0.19814615f,
    -0.240745157f, -0.0894153044f, -0.0469890535f, -0.0281599164f, -0.00766694034f, 0.02462947f, 0.0570506826f, 0.116590597f,
    -0.124388188f, -0.0831644163f, -0.0537905954f, -0.0246422663f, -0.00278992951f, 0.0167240966f, 0.057573352f, 0.165533811f,
    -0.190235764f, -0.146601588f, -0.108043067f, -0.0730642229f, -0.0402573049f, -0.00733679486f, 0.0344631076f, 0.0915360749f,
    -0.0919815451f, -0.0484762825f, -0.0225543827f, 0.0100657092f, 0.0360163376f, 0.0637554601f, 0.0934289098f, 0.139390543f,
    -0.134466484f, -0.0754663721f, -0.0404217541f, -0.00621252321f, -0.00621252321f, 0.0295492094f, 0.0665665343f, 0.109540321f,
    -0.125093043f, -0.0753392577f, -0.0366914906f, -0.013325939f, 0.0197222829f, 0.0197222829f, 0.058561895f, 0.105032757f,
    -0.160255417f, -0.115106232f, -0.0809243768f, -0.0436738022f, 0.0232018977f, 0.0504039004f, 0.0940148011f, 0.239200741f,
    -0.10970559f, -0.0874195099f, -0.0629666224f, -0.0479206853f, -0.0193739031f, 0.00102260499f, 0.0300720166f, 0.0681033581f,
    -0.144781038f, -0.0420141928f, -0.00362845301f, 0.00888643786f, 0.0370424129f, 0.0860288292f, 0.118368044f, 0.158959836f,
    -0.210958183f, -0.181000009f, -0.145474225f, -0.111148052f, -0.0976673439f, -0.0601001978f, -0.0249350574f, 0.0436672047f,
    -0.0668508559f, -0.0261955205f, 0.0065735993f, 0.0312025063f, 0.054358758f, 0.0674791411f, 0.0976065844f, 0.140383199f,
    -0.161672235f, -0.112715766f, -0.0845386758f, -0.0521860793f, -0.0521860793f, -0.0130863627f, 0.00696631894f, 0.0476341061f,
    -0.252923846f, -0.122392282f, -0.0713471919f, -0.0328238793f, 0.0240218118f, 0.0894976035f, 0.145818293f, 0.24668324f,
    -0.073220551f, -0.0236957576f, 0.0102930684f, 0.0353920236f, 0.0691256076f, 0.0985696614f, 0.137256041f, 0.189614564f,
    -0.0502833091f, -0.00565859396f, 0.0286075752f, 0.066605255f, 0.0871521756f, 0.119602025f, 0.158142954f, 0.168751225f,
    -0.175584257f, -0.0341163017f, -0.0173440408f, 0.00307532516f, 0.0296365824f, 0.0478386357f, 0.0802622214f, 0.114337616f,
    -0.118590161f, -0.118590161f, -0.084049277f, -0.084049277f, -0.0699663013f, -0.0324438028f, 5.29729041e-05f, 0.19294551f,
    -0.0541346818f, -0.0175047275f, -0.0175047275f, 0.0200698338f, 0.0293456614f, 0.0579658262f, 0.0899222642f, 0.0957605168f,
    -0.119241469f, -0.0972932726f, -0.0644499138f, -0.0024833139f, 0.122197151f, 0.139332771f, 0.172300205f, 0.21219936f,
    -0.0331701711f, 0.015225986f, 0.0285493899f, 0.0670408905f, 0.0796692073f, 0.106544077f, 0.137182161f, 0.182562992f,
    -0.0858262032f, -0.0381055102f, -0.00616805023f, 0.0255204923f, 0.0553918295f, 0.0655543655f, 0.106944799f, 0.157700181f,
    -0.242345199f, -0.180416867f, -0.140878379f, -0.108254388f, -0.0727699324f, -0.0386086702f, -0.00359225203f, 0.0489149503f,
    -0.205374017f, -0.167183548f, -0.133294746f, -0.0997761115f, -0.0726001114f, -0.0361344256f, 0.00219192635f, 0.0329590142f,
    -0.234755725f, -0.156393439f, -0.111062549f, -0.0921722874f, -0.0656430721f, -0.042091649f, -0.0131200869f, 0.14400208f,
    -0.187395617f, -0.00631702226f, 0.0312876888f, 0.0370554961f, 0.0649538115f, 0.0649538115f, 0.0851514339f, 0.101010829f,
    -0.296026796f, -0.134109616f, -0.0969388932f, -0.065796487f, -0.0522480346f, -0.016812481f, 0.00972079206f, 0.127023593f,
    -0.148928612f, -0.0699235946f, -0.0353685096f, -0.0353685096f, 0.00128477218f, 0.0220097248f, 0.0484720394f, 0.0579271652f,
    -0.207757875f, -0.176566273f, -0.139580965f, -0.0966144577f, -0.0122343302f, 0.0188486669f, 0.0470502935f, 0.0844931304f,
    -0.192821145f, -0.159601375f, -0.121661752f, -0.0940833539f, -0.0669547096f, -0.032731086f, 0.000731927052f, 0.043238312f,
    -0.0405226275f, 0.0146349678f, 0.0530351996f, 0.0838401541f, 0.117179267f, 0.147012845f, 0.187892273f, 0.237704456f,
    -0.131397679f, -0.0888778716f, -0.0510519631f, -0.0186860263f, 0.0413332731f, 0.0739963576f, 0.119542755f, 0.293748438f,
    -0.0698558614f, 0.00909625459f, 0.0473094396f, 0.0736788809f, 0.102738716f, 0.121018387f, 0.160598114f, 0.206884608f,
    -0.15600495f, -0.15600495f, -0.112845749f, -0.0542854853f, -0.0375399329f, 0.123125635f, 0.159139097f, 0.238998085f,
    -0.12343131f, -0.0788678825f, 0.00950591173f, 0.0485208258f, 0.060881868f, 0.0925532505f, 0.163316205f, 0.19521907f,
    -0.0300476532f, 0.00934905373f, 0.0335868709f, 0.0572843142f, 0.0795197934f, 0.100715578f, 0.130292892f, 0.286933988f,
    -0.111474313f, -0.0695398673f, -0.0446144678f, -0.0120056905f, 0.121348284f, 0.152726233f, 0.190981179f, 0.230802268f,
    -0.0456744209f, 0.00446651457f, 0.0356691591f, 0.0707410946f, 0.0969110951f, 0.130060345f, 0.177368283f, 0.209888905f,
    -0.181429148f, -0.132349089f, -0.096384868f, -0.0670052916f, -0.0331958309f, -0.00406058552f, 0.0322575644f, 0.0839152783f,
    -0.080646269f, -0.0485131182f, -0.020840032f, 0.0067816521f, 0.0067816521f, 0.03301901f, 0.0480648056f, 0.0810775533f,
    -0.15135777f, -0.113699488f, -0.0495930426f, -0.0238340702f, 0.00171267462f, 0.0245221071f, 0.056579724f, 0.0995381325f,
    -0.21709007f, -0.0819376335f, -0.0310494751f, -0.0209883451f, 0.00764906593f, 0.00764906593f, 0.0289887264f, 0.0618672185f,
    -0.0967344418f, -0.0585093424f, -0.0585093424f, -0.0216641277f, -0.0147808893f, 0.0232498553f, 0.0402229242f, 0.0697923899f,
    -0.222648814f, -0.182433069f, -0.146210521f, -0.123411715f, -0.0889251679f, -0.0457589403f, -0.00565404305f, 0.0863953978f,
    -0.0674847141f, -0.025954714f, -0.01161189f, 0.0195248164f, 0.0195248164f, 0.0462095514f, 0.0725923553f, 0.112698667f,
    -0.146840304f, -0.109984159f, -0.0802509785f, -0.0559492074f, -0.028127281f, 0.00246952311f, 0.0392661616f, 0.0840138122f,
    -0.0943086743f, -0.047335308f, -0.0139622008f, 0.014748414f, 0.0405948535f, 0.0672248527f, 0.107435256f, 0.159865171f,
    -0.263280392f, -0.140648499f, -0.0980155542f, -0.0668227002f, -0.0352244489f, -0.0057305824f, 0.082278423f, 0.131931469f,
    -0.19797343f, -0.0447539352f, 0.00710752886f, 0.0287079774f, 0.0556011572f, 0.0737060085f, 0.110791743f, 0.170808941f,
    -0.100491844f, -0.0573539957f, -0.0573539957f, -0.0294787511f, -0.0126911495f, 0.0292611681f, 0.0612234995f, 0.127080083f,
    -0.16950123f, -0.109938413f, 0.0373583473f, 0.0942437127f, 0.0944511443f, 0.138221353f, 0.138221353f, 0.24027732f,
    -0.0775320381f, -0.0381027088f, -0.010444006f, 0.010525831f, 0.032791771f, 0.054374963f, 0.0916863084f, 0.174927294f,
    -0.2494075f, -0.136862621f, -0.104047403f, -0.0639363825f, 0.0494431295f, 0.0747602358f, 0.111924723f, 0.148628786f,
    -0.101804078f, -0.0481099747f, -0.019803565f, 0.0104311146f, 0.0104311146f, 0.0417805426f, 0.0777839571f, 0.121798143f,
    -0.148435131f, -0.0874183774f, -0.0553609543f, -0.0268047657f, 0.00139895745f, 0.0306037441f, 0.0681915879f, 0.119916782f,
    -0.106675662f, -0.0187912453f, 0.0159159284f, 0.0463443957f, 0.07810615f, 0.101121187f, 0.149378836f, 0.208805084f,
    -0.129402503f, -0.0851520896f, -0.0698187202f, -0.0604829006f, -0.034309037f, -0.0159158204f, 0.0163722336f, 0.0539306439f,
    -0.177764267f, -0.111634009f, -0.00221751956f, 0.0118578952f, 0.044615984f, 0.0843759403f, 0.116801053f, 0.188143194f,
    -0.0712072402f, -0.0373023897f, -0.024838673f, 0.0119737033f, 0.0230308585f, 0.0561865382f, 0.0851837099f, 0.148400083f,
    -0.161503494f, -0.120578848f, -0.0822868645f, -0.0552259125f, -0.0194718223f, 0.00621957704f, 0.0467183329f, 0.111418948f,
    -0.140908927f, -0.0911993906f, -0.0639272109f, -0.0318799019f, -0.0318799019f, 0.00381714199f, 0.0225626379f, 0.0646396726f,
    -0.191901982f, -0.0893336609f, -0.0482244268f, -0.0146285985f, 0.0375444815f, 0.0815859213f, 0.12748225f, 0.182862461f
};

// TCQ norm alpha: K alpha = 1.0 (no scaling), V alpha = 1.04 (optimal at 2K context).
// Override via TURBO_TCQ_ALPHA (K) and TURBO_TCQ_ALPHA_V (V) env vars.
// Alpha is applied at encode time (baked into fp16 norm) — this outperforms decode-time application.
static __constant__ float d_tcq_norm_alpha = 1.0f;
static __constant__ float d_tcq_norm_alpha_v = 1.04f;

// Load K and V ENCODE codebooks (this TU's __constant__ copies). Mirrors the decode helper:
// K = TURBO_TCQ_CB_K ?? TURBO_TCQ_CB ?? compiled-in; V = TURBO_TCQ_CB_V ?? TURBO_TCQ_CB ??
// compiled-in. V default captured BEFORE the K override so K-only swaps leave V at the anchor.
#include <sys/stat.h>
static inline void turbo_tcq_load_kv_encode() {
    auto load_file = [](const char * p, float * out) -> bool {
        FILE * f = fopen(p, "rb"); if (!f) return false;
        bool ok = fread(out, sizeof(float), 512, f) == 512; fclose(f); return ok;
    };
    auto file_mtime = [](const char * p) -> long { struct stat st; return (p && stat(p, &st) == 0) ? (long)st.st_mtime : 0; };
    int dev = 0; cudaGetDevice(&dev);
    static int hot = -1; if (hot < 0) hot = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
    static bool init[GGML_CUDA_MAX_DEVICES] = {};
    static long mk[GGML_CUDA_MAX_DEVICES] = {}, mv[GGML_CUDA_MAX_DEVICES] = {};
    const char * cb = getenv("TURBO_TCQ_CB");
    const char * kp = getenv("TURBO_TCQ_CB_K"); if (!kp) kp = cb;
    const char * vp = getenv("TURBO_TCQ_CB_V"); if (!vp) vp = cb;
    const bool first = !init[dev];
    const long nk = (hot || first) ? file_mtime(kp) : mk[dev];
    const long nv = (hot || first) ? file_mtime(vp) : mv[dev];
    const bool do_k = first || (hot && nk != mk[dev]);
    const bool do_v = first || (hot && nv != mv[dev]);
    if (!do_k && !do_v) return;
    float buf[512];
    if (false && first && !vp) {  // V falls back to compiled-in anchor (read before any K overwrite)
        float anchor[512]; cudaMemcpyFromSymbol(anchor, d_turbo3_tcq_codebook, 512*sizeof(float));
        cudaMemcpyToSymbol(d_turbo3_tcq_codebook_v, anchor, 512*sizeof(float)); mv[dev] = nv;
    }
    if (do_k && kp && load_file(kp, buf)) { cudaMemcpyToSymbol(d_turbo3_tcq_codebook,   buf, 512*sizeof(float)); mk[dev] = nk; }
    if (do_v && vp && load_file(vp, buf)) { cudaMemcpyToSymbol(d_turbo3_tcq_codebook_v, buf, 512*sizeof(float)); mv[dev] = nv; }
    init[dev] = true;
    if (first && (getenv("TURBO_TCQ_CB_K") || getenv("TURBO_TCQ_CB_V")))
        fprintf(stderr, "TCQ encode: K/V-split codebooks (K=%s V=%s) hotswap=%d\n", kp?kp:"compiled", vp?vp:"compiled", hot);
}

// TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis.
// One block per 128-element group; block width is selected at the launch site.
// Double-buffered cost arrays + global memory backtrace (128 syncs/group, was 384)
template<int nt, typename idx_t>
// minBlocks=2: the 128-step Viterbi is __syncthreads-latency-bound; a second resident block
// per SM hides the sync stalls (pp512 tax vs turbo4 was ~4% with minBlocks=1).
// Viterbi-encode block width. The kernel does 128 sequential s_barrier-synced ACS steps, so the
// barrier cost scales with waves/block. AMD RDNA (wave32) is barrier-bound here: 512 thr = 16 waves
// = expensive block barriers; 128 thr = 4 waves = cheap -> +24.5% prefill on gfx1151 (BIT-EXACT,
// only argmin tie-break differs). NVIDIA is the opposite: the author tuned 512 on a 3090 and 128
// regresses it (-3.3% pp / -1.7% tg on the Qwen D=128 boxmodel, same-tree A/B 2026-07-08).
// TODO(gate-axis): the AMD branch is validated only on gfx1151/RDNA3.5. CDNA is wave64 (untested,
// different barrier math) and discrete RDNA occupancy may differ -> A/B on the RDNA4 9070XT to decide
// whether to narrow this to RDNA/wave32 (or a runtime warpSize check). See memory project_encode_remap_cuda_gate.
#ifndef TCQ3_ENC_NT
#if defined(__HIP_PLATFORM_AMD__) || defined(GGML_USE_HIP)
#define TCQ3_ENC_NT 128   // AMD RDNA (gfx1151 validated): fewer wave32 waves -> cheaper block barriers
#else
#define TCQ3_ENC_NT 512   // NVIDIA: author-tuned optimum (128 regresses on the 3090)
#endif
#endif
static __global__ void __launch_bounds__(nt) k_set_rows_turbo3_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_tcq * __restrict__ dst, const int64_t ne_total_groups,
        uint8_t * __restrict__ bt_buf,
        const int use_shared_bt,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int innerq_is_k, const float * __restrict__ kvmean_mu,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    static_assert(nt == 128 || nt == 256 || nt == 512, "unsupported TCQ3 encoder width");

    const int64_t group = blockIdx.x;
    if (group >= ne_total_groups) return;

    const int sid = threadIdx.x; // state index 0..511

    // Compute source and destination pointers (same index math as turbo3)
    const int64_t i_base = group * QK_TURBO3_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_tcq * dst_blk = (block_turbo3_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                                  + (i00 / QK_TURBO3_TCQ);

    // Shared memory layout (~5KB, was ~35KB before global bt optimization):
    // x[128]     : rotated+normalized input (also reused as outputs[] after Viterbi)
    // cost[512]  : path costs buffer A (also reused for reductions)
    // Backtrace: one predecessor byte for each of the 64 low-state groups per
    // step. The predecessor is independent of the output bits in sid[8:6], so
    // storing 128×64 bytes is equivalent to the older 128×512 layout.
    extern __shared__ uint8_t bt_shared[];
    __shared__ float x[128];
    __shared__ float cost[512];
    __shared__ int warp_min_idx[16];
    __shared__ float warp_min_cost[16];
    __shared__ float cost_b[512];   // double-buffering for Viterbi
    __shared__ float pred_min_cost[64];
    __shared__ int shared_initial_state;

    if (sid < 128) x[sid] = grp_src[sid];
    __syncthreads();

    if (d_innerq_calibrate && sid < 128) {
        atomicAdd(&d_innerq_channel_sq[sid], x[sid] * x[sid]);
        float abs_val = fabsf(x[sid]);
        unsigned int * addr = (unsigned int *)&d_innerq_channel_max[sid];
        unsigned int old_val = __float_as_uint(abs_val);
        unsigned int assumed;
        do {
            assumed = *addr;
            if (__uint_as_float(assumed) >= abs_val) break;
        } while (atomicCAS(addr, assumed, old_val) != assumed);
        if (sid == 0) atomicAdd(&d_innerq_count, 1);
    }
    // Affine tap: raw-domain mean subtract (mu pre-offset to this layer on host)
    if (kvmean_mu != nullptr && sid < 128) x[sid] -= kvmean_mu[i00 + sid];
    // No sync needed: calibration writes d_innerq_channel_sq/max, scaling reads d_innerq_channel_scale
    if (sid < 128) x[sid] *= d_innerq_channel_scale[sid];
    __syncthreads();

    // Norm reduction
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    // 128-thread remap: only cost[0..127] hold real values; reduce 128 (start at 64, not 512/256)
    for (int stride = 64; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 16);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 8);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 4);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 2);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // FWHT. The first five stages are contained within each 32-lane warp, so
    // use warp shuffles and only synchronize for the two cross-warp stages.
    if (sid < 128) {
        float v = x[sid] * d_turbo_wht_signs1[sid];
        const int lane = sid & 31;
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFFULL, v, h);
            v = (lane & h) ? (other - v) : (v + other);
        }
        x[sid] = v;
    }
    __syncthreads();
    if (sid < 64) {
        const int j = ((sid >> 5) << 6) + (sid & 31);
        float a = x[j], b = x[j + 32];
        x[j] = a + b; x[j + 32] = a - b;
    }
    __syncthreads();
    if (sid < 64) {
        float a = x[sid], b = x[sid + 64];
        x[sid] = a + b; x[sid + 64] = a - b;
    }
    __syncthreads();
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * d_turbo_wht_signs2[sid];
    __syncthreads();

    if (sid == 0) turbo_extract_append(x);
    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();

    float saved_norm = cost[0];

    // Viterbi forward pass: double-buffered cost, two-phase min.
    // ⚠ THREE restructures were tried and REVERTED (2026-07-03, pp512/tg32 27B/3090):
    // (0) hoisting the loop-invariant codebook load out of the loop (the t2/t1 win): pp flat
    //     AND tg 30.3 -> 29.8 — under launch_bounds(512,2) the extra live register tips this
    //     512-thread kernel into spills, and the encoder sits on the DECODE path too (every
    //     generated token is re-encoded). Keep the in-loop load here;
    // (1) every-thread redundant 8-way scan (the t2/t1 shape): 8x redundancy + stride-8 bank
    //     conflicts — not attempted after t2 measured flat with only 4x redundancy;
    // (2) register-resident cost + lexicographic (cost,p) clique shuffle-min + double-buffered
    //     group minima (bit-exact, genuinely one barrier): pp512 1134 -> 1088 (-4%), tg32 30.3
    //     -> 29.8 — the ~6-op shuffle chain sits on the step's critical path and costs more
    //     than the barrier it removes. 512 threads with a 2-warp min phase is the local optimum.
    uint8_t * bt = use_shared_bt ? bt_shared : bt_buf + (int64_t)blockIdx.x * (128 * 64);
    for (int s = sid; s < 512; s += nt) cost[s] = 0.0f;  // 512 states, nt threads
    __syncthreads();

    for (int t = 0; t < 128; t++) {
        // Double-buffer: even steps read cost/write cost_b, odd steps read cost_b/write cost
        float * cost_rd = (t & 1) ? cost_b : cost;
        float * cost_wr = (t & 1) ? cost   : cost_b;

        float xt = x[t];

        // Right-shift trellis: ns = (prev >> 3) | (out << 6). The best
        // predecessor depends only on sid's low 6 bits, so compute those 64
        // minima once instead of repeating the same 8-way scan for each out.
        if (sid < 64) {
            const int base_prev = sid << 3;
            float best = cost_rd[base_prev];
            int best_p = 0;
#pragma unroll
            for (int p = 1; p < 8; p++) {
                float c = cost_rd[base_prev | p];
                // TURBO_TCQ_TIEHI: tie-break probe (Codex knob adjudication 2026-07-04) —
                // on exact float ties prefer the HIGHEST predecessor instead of the lowest.
                if (c < best || (d_tcq_tiehi && c == best)) {
                    best = c;
                    best_p = p;
                }
            }
            pred_min_cost[sid] = best;
            bt[t * 64 + sid] = (uint8_t) best_p;
        }
        __syncthreads();

        // Narrow-block remap: each thread writes its 512/nt states (states sharing low-6 bits
        // = same predecessor min; only the codebook value differs).
        const float * cb_enc = innerq_is_k ? d_turbo3_tcq_codebook : d_turbo3_tcq_codebook_v;
        for (int s = sid; s < 512; s += nt) {
            float dist = xt - cb_enc[s];
            dist = dist * dist;
            cost_wr[s] = pred_min_cost[s & 0x3F] + dist;
        }
        __syncthreads();
    }
    // After 128 steps (even count): final costs are in cost[] (step 127 writes to cost)

    // HIP's legacy encoder uses a 128-lane final-state reduction. Preserve that order
    // when RDNA4 uses a wider forward pass so equal-cost path selection stays identical.
    // CUDA and other backends retain their existing launch-width reduction.
#if defined(GGML_USE_HIP)
    constexpr int reduce_nt = 128;
#else
    constexpr int reduce_nt = nt;
#endif
    if (sid < reduce_nt) {
        float my_cost = 3.4028234663852886e38f;
        int my_idx = 0;
        for (int s = sid; s < 512; s += reduce_nt) {
            float c = cost[s];
            if (c < my_cost) { my_cost = c; my_idx = s; }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFFULL, my_cost, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFFULL, my_idx, offset);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32] = my_idx;
        }
    }
    __syncthreads();
    if (sid < 32) {
        constexpr int NWARPS = reduce_nt / 32;
        float best = (sid < NWARPS) ? warp_min_cost[sid] : 3.4028234663852886e38f;
        int best_idx = (sid < NWARPS) ? warp_min_idx[sid] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_down_sync(0xFFFFFFFFULL, best, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFFULL, best_idx, offset);
            if (other_cost < best) {
                best = other_cost;
                best_idx = other_idx;
            }
        }
        if (sid == 0) {
            shared_initial_state = best_idx; // temporarily: best final state (becomes initial after backtrack)
        }
    }
    __syncthreads();

    // Save x[] to global buffer before backtrack overwrites it
    if (d_tcq_dump_max > 0 && group < d_tcq_dump_max && sid < 128)
        d_tcq_dump_x_buf[group * 128 + sid] = x[sid];

    // Backtrack (inherently sequential, reads global bt)
    uint8_t * outputs = (uint8_t *)x;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6);
            int p = bt[t * 64 + (state & 0x3F)];
            state = ((state & 0x3F) << 3) | p;
        }
        shared_initial_state = state;
    }
    __syncthreads();

    // Save output symbols to global buffer
    if (d_tcq_dump_max > 0 && group < d_tcq_dump_max && sid < 128)
        d_tcq_dump_out_buf[group * 128 + sid] = outputs[sid];

    // Parallel recon norm: t>=2 can compute state directly from 3 outputs (3 shifts of 3 = 9 bits)
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 2) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++)
                cur_state = (cur_state >> 3) | (((int)outputs[t]) << 6);
        } else {
            cur_state = ((int)outputs[sid - 2] & 0x7)
                      | (((int)outputs[sid - 1] & 0x7) << 3)
                      | (((int)outputs[sid]     & 0x7) << 6);
        }
        float c = (innerq_is_k ? d_turbo3_tcq_codebook : d_turbo3_tcq_codebook_v)[cur_state];
        my_recon_sq = c * c;
    }
    cost[sid] = my_recon_sq;
    __syncthreads();
    // 128-thread remap: reduce the 128 real recon values (start at 64)
    for (int stride = 64; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 16);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 8);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 4);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 2);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;
    corrected_norm *= innerq_is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v;

    // Parallel bitpack: qs stores 6 initial-state bits followed by 128 3-bit
    // output symbols. Each byte is independent, so avoid the old serial OR loop.
    if (sid < 49) {
        const int init_bits = (shared_initial_state >> 3) & 0x3F;
        uint8_t packed = 0;
#pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            const int pos = sid * 8 + bit;
            int v = 0;
            if (pos < 6) {
                v = (init_bits >> pos) & 1;
            } else {
                const int sym_bit_pos = pos - 6;
                const int sym_idx = sym_bit_pos / 3;
                if (sym_idx < 128) {
                    v = (outputs[sym_idx] >> (sym_bit_pos % 3)) & 1;
                }
            }
            packed |= (uint8_t)(v << bit);
        }
        dst_blk->qs[sid] = packed;
    }
    if (sid == 0) {
        dst_blk->norm = __float2half(corrected_norm);
    }
}

// TCQ GET_ROWS dequantize (for non-FA paths)
#define QR_TURBO3_TCQ 2
static __device__ __forceinline__
void dequantize_turbo3_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs
    {
        const int t = iqs;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.x = d_turbo3_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64 (stride = half block size)
    {
        const int t = iqs + 64;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.y = d_turbo3_tcq_codebook[state] * norm;
    }
}

// =====================================================================================
// TURBO2_TCQ: 2-bit Trellis-Coded Quantization (k=2, L=8, 256 states, free initial state)
// =====================================================================================

// Optional SEPARATE V-side codebook (K/V-split). Has its OWN compiled-in default now
// (copy-K->V disabled); TURBO_TCQ_CB_V overrides only V. K-encode reads
// d_turbo2_tcq_codebook, V-encode reads this, selected by iq_is_k in k_set_rows_turbo2_tcq.
static __constant__ float d_turbo2_tcq_codebook_v[256] = {
    -0.119564079f, -0.0639450625f, -0.0146701029f, 0.0416402221f, -0.0174125507f, 0.0607083291f, 0.0869926363f, 0.171065256f,
    -0.16671373f, -0.0957040414f, -0.0541641749f, 0.00483020535f, -0.0375655368f, 0.0259719901f, 0.0769131333f, 0.133007482f,
    -0.0324706659f, 0.0333815999f, 0.0984280258f, 0.0984280258f, -0.155279562f, -0.112451389f, -0.0665582195f, -0.0390739292f,
    -0.0346273445f, 0.0182233006f, 0.0490703546f, 0.0933427289f, -0.0894939452f, -0.0381219573f, 0.0113708591f, 0.0437584259f,
    -0.165027022f, -0.0160728171f, 0.0613521039f, 0.135686308f, -0.0833117142f, -0.0365488529f, 0.0164404102f, 0.099056907f,
    -0.121164955f, -0.0040460364f, 0.0630212426f, 0.181239545f, -0.0605864599f, -0.00133762986f, 0.0431564003f, 0.101762854f,
    -0.12510027f, -0.056848336f, 0.00908441842f, 0.0434325337f, -0.0811009184f, -0.012191263f, 0.02857095f, 0.0814263523f,
    -0.0321501978f, -0.0321501978f, 0.029185608f, 0.0595929883f, -0.12000709f, -0.043961253f, 0.0411117785f, 0.122811541f,
    -0.0616433136f, 0.00571473129f, 0.0482196435f, 0.107436121f, -0.107340127f, -0.0429929644f, -0.0429929644f, 0.0221426263f,
    -0.0428249799f, 0.0209775325f, 0.0627604052f, 0.117164612f, -0.230132937f, -0.151602969f, -0.0907785371f, -0.0294134971f,
    -0.248246327f, -0.130108818f, 0.00269244495f, 0.167826846f, -0.0335534066f, -0.00608976046f, 0.0562643558f, 0.069990091f,
    -0.171183065f, -0.12197154f, -0.0781908184f, -0.0404915027f, 0.024088202f, 0.0760850534f, 0.133858576f, 0.142337188f,
    -0.0873355567f, -0.0873355567f, -0.0297851842f, 0.0287425462f, 0.0087178275f, 0.0875995159f, 0.145248801f, 0.247238815f,
    -0.0639663637f, -0.0639663637f, 0.000737096125f, 0.0582280643f, -0.171122551f, -0.106863976f, -0.000283801404f, 0.212037727f,
    -0.0181136485f, 0.0445317663f, 0.106189817f, 0.179431632f, -0.191655919f, -0.107088737f, -0.0591632575f, 0.00430248259f,
    -0.113013029f, 0.0722250044f, 0.111605771f, 0.160289794f, -0.069291912f, -0.00479303487f, -0.00479303487f, 0.011135878f,
    0.00871209241f, 0.0821463615f, 0.119828083f, 0.185098261f, -0.177150086f, -0.107679032f, -0.0194601547f, 0.0539710745f,
    0.0272501465f, 0.101181231f, 0.145318046f, 0.224032938f, -0.111173153f, -0.0449736379f, 0.0124536771f, 0.0649857223f,
    -0.149271756f, -0.072600022f, -0.0625199154f, -0.00819602143f, 0.0521048568f, 0.0701553002f, 0.12849921f, 0.153413758f,
    -0.09441223f, -0.0395641439f, -0.00628664764f, 0.033637695f, -0.0625679642f, -0.00288261333f, 0.048446767f, 0.0824608654f,
    -0.020097293f, 0.0397767387f, 0.0397767387f, 0.096181415f, -0.15045543f, -0.107571267f, -0.0757932365f, -0.0231326856f,
    0.0288167801f, 0.0614108481f, 0.121438771f, 0.121438771f, -0.109789863f, -0.0484966673f, -0.0484966673f, 0.0239409413f,
    -0.206154153f, -0.139580801f, -0.0519063473f, 0.0952233076f, 0.0150455777f, 0.0563047603f, 0.108979754f, 0.108979754f,
    -0.211518392f, -0.123396739f, -0.0610813461f, -0.0347043239f, -0.16440773f, -0.0947239771f, -0.0382283144f, 0.0619030744f,
    -0.156453997f, -0.0912440941f, -0.052928295f, 0.0173273887f, -0.0383094996f, 0.0252818447f, 0.0609354861f, 0.131438211f,
    -0.110849418f, -0.0558453612f, -0.00742872898f, 0.0378554091f, 0.00291320775f, 0.0774993449f, 0.135913149f, 0.200488344f,
    -0.0666170493f, -0.00166794204f, 0.0575314499f, 0.0759283155f, -0.10012687f, -0.0674847737f, -0.00651199976f, 0.00259250682f,
    0.0104604028f, 0.0656453595f, 0.0920356736f, 0.13813594f, -0.150504857f, -0.0939132497f, -0.0484457873f, -0.0145660006f,
    -0.122355804f, -0.0236139596f, 0.125541419f, 0.194417521f, -0.0579938814f, 0.00458693132f, 0.0397750214f, 0.0835727304f,
    -0.191926062f, -0.138371572f, -0.0826586038f, -0.0307260547f, -0.00564225856f, 0.0404783934f, 0.0745785311f, 0.129904017f,
    -0.0882599801f, -0.0324259102f, 0.0363260582f, 0.0363260582f, -0.105927862f, -0.0311713293f, 0.017906744f, 0.163340628f,
    -0.0498079844f, -0.00281904545f, 0.0399033576f, 0.0836533159f, -0.0364821255f, 0.0435362682f, 0.093898572f, 0.178892568f
};
// 2-bit TCQ codebook (product_mono/iter090, 256-state bitshift trellis). If you copy these, credit spiritbuun!
// CUDA GLA product-aware training, 100 iters on Qwen3.5-27B FWHT-rotated KV activations. Decode: state_t = read_8_bits(qs, t*2)
static __constant__ float d_turbo2_tcq_codebook[256] = {
    -0.121293001f, -0.0652255416f, -0.0143092116f, 0.0418524668f, -0.0205144603f, 0.0578708202f, 0.0894252285f, 0.171630561f,
    -0.166073903f, -0.0955754593f, -0.051604785f, 0.00835077371f, -0.0391413756f, 0.0262039863f, 0.0772112086f, 0.13472344f,
    -0.0371244252f, 0.0303828381f, 0.0964954272f, 0.0964954272f, -0.153411731f, -0.109019525f, -0.0586176291f, -0.0293964371f,
    -0.0386296436f, 0.0189001262f, 0.0492552966f, 0.0956836939f, -0.0905035883f, -0.0400135368f, 0.00827013049f, 0.0426574461f,
    -0.163978577f, -0.0164943393f, 0.0604525544f, 0.137080088f, -0.0841633603f, -0.0372650921f, 0.0186997745f, 0.100679226f,
    -0.124819808f, -0.00636629853f, 0.062648125f, 0.180190608f, -0.0643028617f, -0.00309835235f, 0.0422532f, 0.100626528f,
    -0.123252243f, -0.0552402064f, 0.0100476379f, 0.0455638319f, -0.0816477984f, -0.0135007529f, 0.0266381856f, 0.0824232548f,
    -0.0326516367f, -0.0326516367f, 0.0282593723f, 0.0614907183f, -0.120962285f, -0.042986732f, 0.0406342745f, 0.123708166f,
    -0.0633638948f, 0.00444420008f, 0.0480255149f, 0.108578168f, -0.108954027f, -0.0433389395f, -0.0433389395f, 0.0242247004f,
    -0.0466773361f, 0.0198548008f, 0.0617980957f, 0.117236525f, -0.22931838f, -0.149933621f, -0.0888211057f, -0.0269997902f,
    -0.249029279f, -0.131829545f, 0.0050804778f, 0.167578816f, -0.03524689f, -0.00841350388f, 0.0572811142f, 0.0706143305f,
    -0.173972413f, -0.122652858f, -0.0801565647f, -0.0375118926f, 0.0238485113f, 0.0755895823f, 0.132045984f, 0.141673177f,
    -0.0868954882f, -0.0868954882f, -0.0285401735f, 0.0313807204f, 0.00747606717f, 0.0866691098f, 0.145436034f, 0.246695101f,
    -0.0659284219f, -0.0659284219f, 0.00104726967f, 0.0582513548f, -0.172842219f, -0.106118038f, -0.000922924315f, 0.213078246f,
    -0.0186287966f, 0.0460402556f, 0.108218782f, 0.180559739f, -0.19514209f, -0.111467898f, -0.0593740903f, 0.00543067884f,
    -0.110466279f, 0.0715457723f, 0.110809304f, 0.161070481f, -0.0704413429f, -0.00710619334f, -0.00710619334f, 0.0174200684f,
    0.00746762427f, 0.0811579898f, 0.119038679f, 0.185045496f, -0.177901462f, -0.106362857f, -0.0179273244f, 0.0551470444f,
    0.0256535504f, 0.0992645845f, 0.142786577f, 0.223529384f, -0.111267753f, -0.0449991263f, 0.0132676214f, 0.0663414672f,
    -0.148271054f, -0.0723822191f, -0.0606202595f, -0.00630783942f, 0.0497888848f, 0.071064502f, 0.128621861f, 0.154562116f,
    -0.0971549302f, -0.0402041078f, -0.00789874326f, 0.0369283147f, -0.0616744198f, -0.00166254514f, 0.0472205244f, 0.0859320685f,
    -0.0219837818f, 0.0407113731f, 0.0407113731f, 0.0973400399f, -0.149714395f, -0.105322495f, -0.0728028268f, -0.0172174871f,
    0.0268614478f, 0.0596466064f, 0.121133149f, 0.121133149f, -0.112265021f, -0.0490688905f, -0.0490688905f, 0.0245620999f,
    -0.205158055f, -0.138588384f, -0.051692646f, 0.0981862471f, 0.0128893815f, 0.0546906516f, 0.105790317f, 0.105790317f,
    -0.205479771f, -0.12078198f, -0.0628709123f, -0.0326852798f, -0.163441271f, -0.0935551152f, -0.0372627303f, 0.0646881312f,
    -0.155186564f, -0.0900211185f, -0.0496243872f, 0.0199296083f, -0.0397389159f, 0.0244133994f, 0.0621035174f, 0.132506922f,
    -0.111906596f, -0.0571601354f, -0.00674164575f, 0.0384595022f, -0.00602581911f, 0.0733171999f, 0.132919669f, 0.198568419f,
    -0.0653695911f, -0.000432662782f, 0.057027299f, 0.0789022967f, -0.100032941f, -0.0688288137f, -0.001128476f, 0.00303854235f,
    -0.00244383421f, 0.0639961362f, 0.089498803f, 0.141602159f, -0.150657237f, -0.0925111547f, -0.0505311303f, -0.00445175637f,
    -0.124654092f, -0.0240918342f, 0.126107931f, 0.195764229f, -0.0592086203f, 0.00490828836f, 0.0404077731f, 0.0850417614f,
    -0.193506882f, -0.138968095f, -0.081633091f, -0.0242004432f, -0.011047815f, 0.0390194245f, 0.0735576898f, 0.130851239f,
    -0.0880464092f, -0.0321566798f, 0.0365370661f, 0.0365370661f, -0.10624709f, -0.0330201462f, 0.0187416486f, 0.161234185f,
    -0.0509652421f, -0.00451211026f, 0.0393821076f, 0.084524475f, -0.0383864157f, 0.0412525721f, 0.0913410187f, 0.176818803f
};

// Same K/V-split semantics as turbo_tcq_load_kv_encode() but for the 2-bit (256-state) turbo2_tcq
// encode symbols. Separate function because the codebook files are 256 floats (vs 512 for turbo3).
// Defined here (after the turbo2 symbols) so both d_turbo2_tcq_codebook[_v] are in scope.
static inline void turbo2_tcq_load_kv_encode() {
    auto load_file = [](const char * p, float * out) -> bool {
        FILE * f = fopen(p, "rb"); if (!f) return false;
        bool ok = fread(out, sizeof(float), 256, f) == 256; fclose(f); return ok;
    };
    auto file_mtime = [](const char * p) -> long { struct stat st; return (p && stat(p, &st) == 0) ? (long)st.st_mtime : 0; };
    int dev = 0; cudaGetDevice(&dev);
    static int hot = -1; if (hot < 0) hot = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
    static bool init[GGML_CUDA_MAX_DEVICES] = {};
    static long mk[GGML_CUDA_MAX_DEVICES] = {}, mv[GGML_CUDA_MAX_DEVICES] = {};
    const char * cb = getenv("TURBO_TCQ_CB");
    const char * kp = getenv("TURBO_TCQ_CB_K"); if (!kp) kp = cb;
    const char * vp = getenv("TURBO_TCQ_CB_V"); if (!vp) vp = cb;
    const bool first = !init[dev];
    const long nk = (hot || first) ? file_mtime(kp) : mk[dev];
    const long nv = (hot || first) ? file_mtime(vp) : mv[dev];
    const bool do_k = first || (hot && nk != mk[dev]);
    const bool do_v = first || (hot && nv != mv[dev]);
    if (!do_k && !do_v) return;
    float buf[256];
    if (false && first && !vp) {  // V falls back to compiled-in anchor (read before any K overwrite)
        float anchor[256]; cudaMemcpyFromSymbol(anchor, d_turbo2_tcq_codebook, 256*sizeof(float));
        cudaMemcpyToSymbol(d_turbo2_tcq_codebook_v, anchor, 256*sizeof(float)); mv[dev] = nv;
    }
    if (do_k && kp && load_file(kp, buf)) { cudaMemcpyToSymbol(d_turbo2_tcq_codebook,   buf, 256*sizeof(float)); mk[dev] = nk; }
    if (do_v && vp && load_file(vp, buf)) { cudaMemcpyToSymbol(d_turbo2_tcq_codebook_v, buf, 256*sizeof(float)); mv[dev] = nv; }
    init[dev] = true;
    if (first && (getenv("TURBO_TCQ_CB_K") || getenv("TURBO_TCQ_CB_V")))
        fprintf(stderr, "TCQ2 encode: K/V-split codebooks (K=%s V=%s) hotswap=%d\n", kp?kp:"compiled", vp?vp:"compiled", hot);
}

// 2-bit TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis (k=2, L=8)
// Double-buffered cost arrays + global memory backtrace (128 syncs/group, was 384)
template<typename idx_t>
// minBlocks=3: same sync-latency reasoning as turbo3_tcq (256 threads leave headroom for 3).
static __global__ void __launch_bounds__(256, 3) k_set_rows_turbo2_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo2_tcq * __restrict__ dst, const int64_t ne_total_groups,
        uint8_t * __restrict__ bt_buf,
        const int use_shared_bt,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int iq_is_k, const float * __restrict__ kvmean_mu,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int grp = blockIdx.x;
    if (grp >= ne_total_groups) return;
    const int sid = threadIdx.x; // 0..255 = trellis state

    // Compute source and destination pointers (all threads, used by thread 0)
    const int64_t i_base = int64_t(grp) * QK_TURBO2_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo2_tcq * dst_blk = (block_turbo2_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                               + (i00 / QK_TURBO2_TCQ);

    // Backtrace: one predecessor byte per 64 low-state groups per step.
    // The predecessor depends only on sid's low 6 bits (same as turbo3_tcq).
    extern __shared__ uint8_t bt_shared[];
    __shared__ float x[128];
    __shared__ float cost[256];
    __shared__ float cost_b[256];   // double-buffering for Viterbi
    __shared__ int warp_min_idx[8];
    __shared__ float warp_min_cost[8];
    __shared__ int shared_initial_state;

    if (sid < 128) x[sid] = grp_src[sid];
    __syncthreads();

    if (d_innerq_calibrate && sid < 128) {
        atomicAdd(&d_innerq_channel_sq[sid], x[sid] * x[sid]);
        float abs_val = fabsf(x[sid]);
        unsigned int * addr = (unsigned int *)&d_innerq_channel_max[sid];
        unsigned int old_val = __float_as_uint(abs_val);
        unsigned int assumed;
        do {
            assumed = *addr;
            if (__uint_as_float(assumed) >= abs_val) break;
        } while (atomicCAS(addr, assumed, old_val) != assumed);
        if (sid == 0) atomicAdd(&d_innerq_count, 1);
    }
    // Affine tap: raw-domain mean subtract (mu pre-offset to this layer on host)
    if (kvmean_mu != nullptr && sid < 128) x[sid] -= kvmean_mu[i00 + sid];
    if (sid < 128) x[sid] *= d_innerq_channel_scale[sid];
    __syncthreads();

    // Norm reduction
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 16);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 8);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 4);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 2);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // FWHT. The first five stages use warp shuffles, the two cross-warp stages
    // use shared memory (same approach as turbo3_tcq).
    if (sid < 128) {
        float v = x[sid] * d_turbo_wht_signs1[sid];
        const int lane = sid & 31;
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFFULL, v, h);
            v = (lane & h) ? (other - v) : (v + other);
        }
        x[sid] = v;
    }
    __syncthreads();
    if (sid < 64) {
        const int j = ((sid >> 5) << 6) + (sid & 31);
        float a = x[j], b = x[j + 32];
        x[j] = a + b; x[j + 32] = a - b;
    }
    __syncthreads();
    if (sid < 64) {
        float a = x[sid], b = x[sid + 64];
        x[sid] = a + b; x[sid + 64] = a - b;
    }
    __syncthreads();
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * d_turbo_wht_signs2[sid];
    __syncthreads();

    if (sid == 0) turbo_extract_append(x);
    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();

    float saved_norm = cost[0];

    // Viterbi forward pass: double-buffered cost, ONE barrier per step. Every thread scans its
    // own 4-way predecessor group straight from the read buffer (reads cost_rd, writes cost_wr —
    // no intra-step hazard), instead of a 64-thread min phase + barrier + broadcast. The scan is
    // 4x redundant across threads sharing a group, but the loop is barrier-latency-bound, and
    // identical floats in identical compare order keep it bit-exact. The codebook value is
    // hoisted: it is loop-invariant, and the compiler cannot lift a __device__ load across
    // __syncthreads (was 128 redundant global loads per thread).
    uint8_t * bt = use_shared_bt ? bt_shared : bt_buf + (int64_t)blockIdx.x * (128 * 64);
    cost[sid] = 0.0f;
    const float cb_sid = (iq_is_k ? d_turbo2_tcq_codebook : d_turbo2_tcq_codebook_v)[sid];
    __syncthreads();

    for (int t = 0; t < 128; t++) {
        float * cost_rd = (t & 1) ? cost_b : cost;
        float * cost_wr = (t & 1) ? cost   : cost_b;

        float xt = x[t];

        // Right-shift trellis (k=2, L=8): ns = (prev >> 2) | (out << 6); the best predecessor
        // depends only on sid's low 6 bits.
        const int pred_idx = sid & 0x3F;
        const int base_prev = pred_idx << 2;
        float best = cost_rd[base_prev];
        int best_p = 0;
#pragma unroll
        for (int p = 1; p < 4; p++) {
            float c = cost_rd[base_prev | p];
            if (c < best) {
                best = c;
                best_p = p;
            }
        }
        if (sid < 64) {
            bt[t * 64 + sid] = (uint8_t) best_p; // pred_idx == sid here: same byte as the old phase
        }

        float dist = xt - cb_sid;
        dist = dist * dist;

        cost_wr[sid] = best + dist;
        __syncthreads();
    }
    // After 128 steps (even count): final costs in cost[]

    // Warp argmin over 256 costs
    {
        float my_cost = cost[sid];
        int my_idx = sid;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFFULL, my_cost, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFFULL, my_idx, offset);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32] = my_idx;
        }
    }
    __syncthreads();
    if (sid < 32) {
        float best = (sid < 8) ? warp_min_cost[sid] : 3.4028234663852886e38f;
        int best_idx = (sid < 8) ? warp_min_idx[sid] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_down_sync(0xFFFFFFFFULL, best, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFFULL, best_idx, offset);
            if (other_cost < best) {
                best = other_cost;
                best_idx = other_idx;
            }
        }
        if (sid == 0) {
            shared_initial_state = best_idx;
        }
    }
    __syncthreads();

    // Save x[] to global buffer before backtrack overwrites it
    if (d_tcq_dump_max > 0 && grp < d_tcq_dump_max && sid < 128)
        d_tcq_dump_x_buf[grp * 128 + sid] = x[sid];

    // Backtrack (inherently sequential, reads compressed bt)
    uint8_t * outputs = (uint8_t *)x;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6);
            int p = bt[t * 64 + (state & 0x3F)];
            state = ((state & 0x3F) << 2) | p;
        }
        shared_initial_state = state;
    }
    __syncthreads();

    // Save output symbols to global buffer
    if (d_tcq_dump_max > 0 && grp < d_tcq_dump_max && sid < 128)
        d_tcq_dump_out_buf[grp * 128 + sid] = outputs[sid];

    // Parallel recon norm: t>=3 can compute state directly from 4 outputs (4 shifts of 2 = 8 bits)
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 3) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++)
                cur_state = (cur_state >> 2) | (((int)outputs[t]) << 6);
        } else {
            cur_state = ((int)outputs[sid - 3] & 0x3)
                      | (((int)outputs[sid - 2] & 0x3) << 2)
                      | (((int)outputs[sid - 1] & 0x3) << 4)
                      | (((int)outputs[sid]     & 0x3) << 6);
        }
        float c = (iq_is_k ? d_turbo2_tcq_codebook : d_turbo2_tcq_codebook_v)[cur_state];
        my_recon_sq = c * c;
    }
    cost[sid] = my_recon_sq;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 16);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 8);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 4);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 2);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;
    corrected_norm *= iq_is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v;

    // Parallel bitpack: qs stores 6 initial-state bits followed by 128 2-bit
    // output symbols. Each byte is independent (2-bit symbols never cross byte
    // boundaries after the 6-bit prefix), so avoid the serial OR loop.
    if (sid < 33) {
        const int init_bits = (shared_initial_state >> 2) & 0x3F;
        uint8_t packed = 0;
#pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            const int pos = sid * 8 + bit;
            int v = 0;
            if (pos < 6) {
                v = (init_bits >> pos) & 1;
            } else {
                const int sym_bit_pos = pos - 6;
                const int sym_idx = sym_bit_pos / 2;
                if (sym_idx < 128) {
                    v = (outputs[sym_idx] >> (sym_bit_pos % 2)) & 1;
                }
            }
            packed |= (uint8_t)(v << bit);
        }
        dst_blk->qs[sid] = packed;
    }
    if (sid == 0) {
        dst_blk->norm = __float2half(corrected_norm);
    }
}

// 2-bit TCQ GET_ROWS dequantize
#define QR_TURBO2_TCQ 2
static __device__ __forceinline__
void dequantize_turbo2_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs: read 8-bit state via sliding window
    {
        const int t = iqs;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.x = d_turbo2_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64
    {
        const int t = iqs + 64;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.y = d_turbo2_tcq_codebook[state] * norm;
    }
}

// =====================================================================================
// TURBO1_TCQ: 1-bit Trellis-Coded Quantization (k=1, L=8, 256 states, free initial state)
// Encode mirrors k_set_rows_turbo2_tcq but with the k=1 right-shift recurrence
// ns = (prev >> 1) | (out << 7): 2 predecessors sharing sid's low 7 bits, 1 output bit/step.
// Bitstream = 7 init-prefix bits + 128 1-bit outputs = 135 bits = 17 bytes. Separate K/V
// 256-state codebooks (TURBO1_TCQ_CB_K / TURBO1_TCQ_CB_V); cold-start = turbo2_tcq anchor.
// Decode = per-row inverse FWHT (k_turbo1_tcq_dequant_f16 in fattn.cu).
// =====================================================================================
// Baked-in turbo1_tcq codebooks (tail-GLA k=1/L=8, TCQ_TAIL_W=6, trained on real post-FWHT Qwen3.6-27B KV).
// K = r1K_tail_s5/iter300, V = r1V_tail_s11/iter300 (box-2 finalists 2026-07-04; dorei re-confirm 2026-07-12:
// median KLD 0.027913/0.024851/0.024283/0.024715 @2k/8k/16k/32k = -6.8/-7.7/-7.8/-3.3% vs prior s5i135/s1i490).
// Literals are bit-exact round-trips of the .bin sources (9 sig digits).
// TURBO1_TCQ_CB_K/_V env still override these at runtime for future sweeps.
static __constant__ float d_turbo1_tcq_codebook[256]   = {  // K encode
    -0.153683484f, +0.0932431519f, -0.0284221377f, -0.0213323738f, -0.0918863863f, -0.0834758952f, +0.0542073846f, +0.0542073846f,
    -0.128483802f, -0.0949096978f, +0.0436351784f, +0.0444926955f, -0.177975088f, +0.132742643f, -0.0365675129f, -0.00663438998f,
    +0.0619110875f, +0.0619110875f, -0.176081076f, -0.0222552121f, -0.0675581396f, -0.0675581396f, +0.0479601882f, +0.0701695681f,
    +0.0259720646f, +0.0664564744f, +0.0030253143f, +0.167568222f, -0.0674274117f, -0.0624906905f, +0.0593295284f, +0.0716510117f,
    +0.00550817279f, +0.00762284314f, +0.123896576f, +0.14741008f, -0.0566166341f, -0.0527233072f, -0.132352635f, +0.00985434558f,
    -0.0692158043f, -0.0692158043f, +0.054607287f, +0.054607287f, +0.13614215f, +0.19078669f, +0.00156285148f, +0.00209119194f,
    -0.188667893f, -0.0470676497f, -0.116242573f, -0.0126988571f, +0.00894579757f, +0.150882706f, +0.0449662991f, +0.0545337871f,
    +0.0826691911f, +0.0982875451f, -0.0603181981f, -0.0603181981f, +0.0555567369f, +0.0555567369f, -0.110769883f, -0.0770125017f,
    -0.0724448115f, -0.0724448115f, +0.0754566267f, +0.0754566267f, -0.11218828f, -0.0535744801f, +0.0575145707f, +0.0575145707f,
    +0.119879454f, +0.152365059f, -0.0423310474f, -0.0423310474f, -0.034485355f, -0.034485355f, -0.0241647139f, +0.0743408874f,
    +0.0647695139f, +0.0677786916f, -0.0567673519f, -0.0567673519f, +0.0939702094f, +0.142256647f, -0.0456033908f, -0.0456033908f,
    +0.0414441153f, +0.0414441153f, -0.116412848f, -0.0563932136f, +0.0463487506f, +0.118106365f, -0.147341222f, -0.0630168244f,
    -0.0628485903f, -0.0628485903f, +0.0281294063f, +0.0281294063f, +0.157495975f, +0.157495975f, -0.0421977639f, -0.0421977639f,
    +0.0308154635f, +0.0308154635f, -0.0292343032f, -0.0292343032f, -0.125520885f, -0.125520885f, +0.0718865544f, +0.0718865544f,
    +0.0689959824f, +0.0689959824f, -0.081269294f, -0.081269294f, -0.0909179002f, -0.0909179002f, +0.0190163292f, +0.173722342f,
    -0.0671347827f, -0.0671347827f, +0.0800654069f, +0.0800654069f, +0.0816404521f, +0.0816404521f, -0.0624779612f, -0.0624779612f,
    -0.0195930507f, -0.0195930507f, +0.138596535f, +0.138596535f, +0.0451605618f, +0.0459208488f, -0.109351009f, -0.109351009f,
    +0.0349098034f, +0.0685105473f, -0.124482363f, -0.124482363f, -0.0191877317f, -0.0191877317f, +0.0745684579f, +0.186214373f,
    -0.0670050532f, -0.0248501189f, +0.00614000158f, +0.127290398f, +0.0596978553f, +0.0726318583f, -0.0574334636f, -0.0574334636f,
    -0.0218063109f, -0.0218063109f, -0.176524386f, +0.0025834043f, +0.0332166888f, +0.0332166888f, -0.0316858068f, -0.0316858068f,
    -0.16282168f, -0.160597652f, -0.0151454788f, -0.000276063627f, +0.00815063529f, +0.00815063529f, +0.0220920816f, +0.197730258f,
    +0.0771044195f, +0.0771044195f, -0.0734175146f, -0.0734175146f, +0.00178386632f, +0.0152899409f, -0.15694806f, -0.15694806f,
    -0.00759414956f, +0.0315586925f, +0.0653701797f, +0.163669571f, -0.144907251f, +0.00212525437f, -0.048506137f, -0.0196365472f,
    -0.0620619431f, -0.0620619431f, +0.0959848315f, +0.0959848315f, -0.073461324f, -0.0689062104f, +0.0647197068f, +0.0647197068f,
    +0.0735893846f, +0.0735893846f, -0.0616040565f, -0.0610223711f, +0.0637794361f, +0.0996553302f, -0.0651129931f, -0.0651129931f,
    -0.0469228663f, -0.0452931263f, +0.113321088f, +0.113321088f, +0.0841016918f, +0.0841016918f, -0.194901869f, -0.0528107621f,
    -0.0684252232f, -0.0684252232f, +0.0722951591f, +0.0722951591f, -0.042255573f, -0.042255573f, +0.118550412f, +0.118550412f,
    -0.139325529f, -0.0234075822f, +0.0242734887f, +0.0253605284f, -0.146457925f, -0.0543592274f, +0.0258371048f, +0.0723357424f,
    +0.0426197089f, +0.0518061668f, -0.121822171f, -0.121822171f, -0.0367406197f, -0.0367406197f, +0.0920659974f, +0.0920659974f,
    -0.105211377f, -0.0984823778f, +0.0756842345f, +0.0756842345f, +0.0151738664f, +0.0239403863f, -0.0424663946f, -0.0277856514f,
    -0.0687293857f, -0.0687293857f, +0.0550689697f, +0.0550689697f, +0.0518884808f, +0.0518884808f, -0.163815364f, -0.0187361594f,
    +0.06937702f, +0.105992272f, -0.0624594353f, -0.0624594353f, -0.0639030039f, -0.0639030039f, +0.11936225f, +0.11936225f,
};
static __constant__ float d_turbo1_tcq_codebook_v[256] = {  // V encode
    -0.119549245f, -0.119549245f, +0.04884452f, +0.04884452f, +0.112512514f, +0.112512514f, -0.0753565952f, -0.0124196028f,
    -0.0542244501f, -0.0454595797f, +0.113282755f, +0.113282755f, -0.0478299409f, -0.00588486949f, -0.156662762f, +0.0177723076f,
    +0.0502630547f, +0.0751912221f, -0.0732487515f, -0.0363520756f, -0.0878284127f, -0.0833693445f, +0.0655872673f, +0.0655872673f,
    -0.177922264f, +0.147200137f, -0.00281362329f, -0.00281362329f, -0.0236557927f, -0.00629976951f, +0.0262224339f, +0.0262224339f,
    -0.122524828f, -0.121980354f, +0.0355546959f, +0.0355546959f, -0.0892947987f, -0.0783565268f, +0.0634978563f, +0.0634978563f,
    +0.100714654f, +0.116031229f, -0.0382292345f, -0.0382292345f, +0.0700735897f, +0.0700735897f, -0.0744145215f, -0.0744145215f,
    +0.0285345633f, +0.0285345633f, -0.137101039f, -0.0669855997f, +0.0462302938f, +0.0955360904f, -0.0378546566f, -0.0378546566f,
    -0.00299687427f, +0.0354520716f, +0.158731595f, +0.159036174f, -0.00417822506f, -0.00417822506f, -0.169665679f, -0.101989843f,
    +0.0885970294f, +0.174134374f, -0.0419115685f, -0.0374047682f, -0.0866785944f, -0.0649861544f, +0.0656950325f, +0.0656950325f,
    -0.161553636f, -0.145003155f, +0.0255606212f, +0.0260413494f, -0.184533074f, -0.0552854389f, +0.0322727412f, +0.0322727412f,
    -0.0823699087f, -0.0823699087f, +0.0553851724f, +0.0553851724f, +0.0966504067f, +0.0966504067f, -0.0494570695f, -0.0411219671f,
    -0.0437131599f, -0.0437131599f, +0.124873698f, +0.166997984f, +0.0439662524f, +0.0628898293f, -0.169980094f, -0.0406379513f,
    +0.0272835698f, +0.0272835698f, -0.102720782f, -0.0547375381f, -0.104257204f, -0.0429750308f, +0.0265545268f, +0.171791926f,
    -0.0853084773f, -0.0853084773f, +0.0476404428f, +0.0476404428f, +0.0900258943f, +0.0900258943f, -0.054219868f, -0.054219868f,
    -0.0479668193f, -0.0429475419f, +0.131989121f, +0.131989121f, +0.0447537825f, +0.0447537825f, -0.0831433907f, -0.078851141f,
    +0.0472813472f, +0.0472813472f, -0.120939665f, -0.120939665f, -0.146120414f, -0.0148609718f, +0.0725838169f, +0.0725838169f,
    +0.0397478864f, +0.0397478864f, -0.0799675956f, -0.0799675956f, -0.0573435687f, -0.037379805f, +0.0547020584f, +0.169056475f,
    +0.0825577676f, +0.0825577676f, -0.0486915782f, -0.0486915782f, +0.0708708912f, +0.164926067f, -0.0399483964f, -0.0399483964f,
    -0.0932579115f, -0.066275239f, +0.0614030398f, +0.119138196f, +0.0609519258f, +0.0609519258f, -0.087209031f, -0.0788605884f,
    -0.0137468651f, -0.0137468651f, -0.170376986f, +0.0652351379f, +0.0180841498f, +0.0180841498f, -0.0360591412f, -0.0360591412f,
    +0.032522019f, +0.032522019f, -0.157200888f, -0.157200888f, +0.0732334927f, +0.0732334927f, -0.0809531361f, -0.0801029429f,
    -0.0435965359f, -0.0396991074f, +0.124063686f, +0.124063686f, -0.06837219f, -0.06837219f, +0.0630521029f, +0.0643705279f,
    -0.0663868934f, +0.186983824f, +0.00707759568f, +0.0534422286f, -0.0711288378f, -0.0526081622f, +0.0910681486f, +0.0910681486f,
    -0.181568101f, -0.0869468525f, -0.00884238817f, -0.00884238817f, +0.116818011f, +0.116818011f, +0.00558634428f, +0.0387513824f,
    -0.0515811332f, -0.0300675891f, +0.0699146464f, +0.0728352666f, +0.048193451f, +0.0833275244f, -0.0958049595f, -0.0958049595f,
    +0.0172887873f, +0.0292257778f, -0.148785338f, -0.148785338f, -0.000765698263f, +0.0509745441f, +0.0231456049f, +0.172527939f,
    +0.0573678799f, +0.0582933389f, -0.0823828951f, -0.0823828951f, -0.0570469573f, -0.0570469573f, +0.0781005844f, +0.111567542f,
    +0.10178373f, +0.10178373f, -0.0412749536f, -0.0412749536f, -0.0953622833f, -0.0633069724f, +0.0198892411f, +0.124936409f,
    -0.0825958773f, +0.186329335f, +0.0161208287f, +0.0615422241f, +0.0401945598f, +0.0468487293f, -0.164305717f, -0.0166916698f,
    +0.0545916557f, +0.0545916557f, -0.0917867944f, -0.0917867944f, -0.0585476533f, -0.0585476533f, +0.0977485031f, +0.0977485031f,
    +0.0931819826f, +0.0931819826f, -0.0435980707f, -0.0435980707f, -0.0865197331f, -0.0865197331f, +0.0579762831f, +0.0579762831f,
    -0.0325509906f, -0.0325509906f, +0.0279718302f, +0.0279718302f, +0.0218467899f, +0.18708235f, -0.0869560316f, -0.085878171f,
};

static inline void turbo1_tcq_load_kv_encode() {
    auto load_file = [](const char * p, float * out) -> bool {
        FILE * f = fopen(p, "rb"); if (!f) return false;
        bool ok = fread(out, sizeof(float), 256, f) == 256; fclose(f); return ok;
    };
    auto file_mtime = [](const char * p) -> long { struct stat st; return (p && stat(p, &st) == 0) ? (long)st.st_mtime : 0; };
    int dev = 0; cudaGetDevice(&dev);
    static int hot = -1; if (hot < 0) hot = getenv("TURBO1_TCQ_HOTSWAP") ? 1 : 0;
    static bool init[GGML_CUDA_MAX_DEVICES] = {};
    static long mk[GGML_CUDA_MAX_DEVICES] = {}, mv[GGML_CUDA_MAX_DEVICES] = {};
    const char * cb = getenv("TURBO1_TCQ_CB");
    const char * kp = getenv("TURBO1_TCQ_CB_K"); if (!kp) kp = cb;
    const char * vp = getenv("TURBO1_TCQ_CB_V"); if (!vp) vp = cb;
    const bool first = !init[dev];
    const long nk = (hot || first) ? file_mtime(kp) : mk[dev];
    const long nv = (hot || first) ? file_mtime(vp) : mv[dev];
    const bool do_k = first || (hot && nk != mk[dev]);
    const bool do_v = first || (hot && nv != mv[dev]);
    if (!do_k && !do_v) return;
    float buf[256];
    // cold-start default = baked-in best codebook (seed5/iter135 K, seed1/iter490 V); only env overrides it.
    if (do_k && kp && load_file(kp, buf)) { cudaMemcpyToSymbol(d_turbo1_tcq_codebook,   buf, 256*sizeof(float)); mk[dev] = nk; }
    if (do_v && vp && load_file(vp, buf)) { cudaMemcpyToSymbol(d_turbo1_tcq_codebook_v, buf, 256*sizeof(float)); mv[dev] = nv; }
    init[dev] = true;
}

// 1-bit TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis (k=1, L=8).
// Stores FWHT-rotated trellis codes; decode (k_turbo1_tcq_dequant_f16) does the inverse FWHT.
template<typename idx_t>
// minBlocks=3: same sync-latency reasoning as turbo3_tcq (256 threads leave headroom for 3).
static __global__ void __launch_bounds__(256, 3) k_set_rows_turbo1_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo1_tcq * __restrict__ dst, const int64_t ne_total_groups,
        uint8_t * __restrict__ bt_buf,
        const int use_shared_bt,
        const int iq_is_k, const float * __restrict__ kvmean_mu,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int grp = blockIdx.x;
    if (grp >= ne_total_groups) return;
    const int sid = threadIdx.x; // 0..255 = trellis state

    const int64_t i_base = int64_t(grp) * QK_TURBO1_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo1_tcq * dst_blk = (block_turbo1_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                               + (i00 / QK_TURBO1_TCQ);

    extern __shared__ uint8_t bt_shared[];
    __shared__ float x[128];
    __shared__ float cost[256];
    __shared__ float cost_b[256];        // double-buffering for Viterbi
    __shared__ int   warp_min_idx[8];
    __shared__ float warp_min_cost[8];
    __shared__ int   shared_initial_state;

    if (sid < 128) {
        x[sid] = grp_src[sid];
        // Affine tap: raw-domain mean subtract (mu pre-offset to this layer on host);
        // same-thread element, so it folds into the load under one barrier
        if (kvmean_mu != nullptr) {
            x[sid] -= kvmean_mu[i00 + sid];
        }
    }
    __syncthreads();

    // Norm reduction
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 16);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 8);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 4);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 2);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // FWHT: signs1 -> butterfly -> inv_sqrt_128 -> signs2 (same as turbo2_tcq encode).
    if (sid < 128) {
        float v = x[sid] * d_turbo_wht_signs1[sid];
        const int lane = sid & 31;
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFFULL, v, h);
            v = (lane & h) ? (other - v) : (v + other);
        }
        x[sid] = v;
    }
    __syncthreads();
    if (sid < 64) {
        const int j = ((sid >> 5) << 6) + (sid & 31);
        float a = x[j], b = x[j + 32];
        x[j] = a + b; x[j + 32] = a - b;
    }
    __syncthreads();
    if (sid < 64) {
        float a = x[sid], b = x[sid + 64];
        x[sid] = a + b; x[sid + 64] = a - b;
    }
    __syncthreads();
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * d_turbo_wht_signs2[sid];
    __syncthreads();

    if (sid == 0) turbo_extract_append(x, iq_is_k ? 0 : 100);   // post-FWHT harvest, tagged K=0/V=100
    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();
    float saved_norm = cost[0];

    // Viterbi forward pass: double-buffered cost (1 sync/step)
    uint8_t * bt = use_shared_bt ? bt_shared : bt_buf + (int64_t)blockIdx.x * (128 * 128);
    cost[sid] = 0.0f;
    __syncthreads();

    // ONE barrier per step (same transform as k_set_rows_turbo2_tcq): each thread does its own
    // 2-way predecessor scan from the read buffer — bit-exact, half the barriers, and the
    // loop-invariant codebook value is hoisted out of the loop (cbk itself is still used by the
    // recon pass below).
    const float * cbk = iq_is_k ? d_turbo1_tcq_codebook : d_turbo1_tcq_codebook_v;
    const float cb_sid = cbk[sid];
    for (int t = 0; t < 128; t++) {
        float * cost_rd = (t & 1) ? cost_b : cost;
        float * cost_wr = (t & 1) ? cost   : cost_b;

        float xt = x[t];

        // Right-shift trellis (k=1, L=8): ns = (prev >> 1) | (out << 7); the best predecessor
        // depends only on sid's low 7 bits.
        const int pred_idx = sid & 0x7F;
        const int base_prev = pred_idx << 1;
        float best = cost_rd[base_prev];
        int best_p = 0;
        float c1 = cost_rd[base_prev | 1];
        if (c1 < best) { best = c1; best_p = 1; }
        if (sid < 128) {
            bt[t * 128 + sid] = (uint8_t) best_p; // pred_idx == sid here
        }

        float dist = xt - cb_sid;
        dist = dist * dist;

        cost_wr[sid] = best + dist;
        __syncthreads();
    }
    // After 128 steps (even count): final costs in cost[]

    // Warp argmin over 256 costs
    {
        float my_cost = cost[sid];
        int my_idx = sid;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFFULL, my_cost, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFFULL, my_idx, offset);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32] = my_idx;
        }
    }
    __syncthreads();
    if (sid < 32) {
        float best = (sid < 8) ? warp_min_cost[sid] : 3.4028234663852886e38f;
        int best_idx = (sid < 8) ? warp_min_idx[sid] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_down_sync(0xFFFFFFFFULL, best, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFFULL, best_idx, offset);
            if (other_cost < best) { best = other_cost; best_idx = other_idx; }
        }
        if (sid == 0) shared_initial_state = best_idx;
    }
    __syncthreads();

    // Backtrack (sequential): emit 1 output bit/step
    uint8_t * outputs = (uint8_t *)x;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 7);
            int p = bt[t * 128 + (state & 0x7F)];
            state = ((state & 0x7F) << 1) | p;
        }
        shared_initial_state = state;
    }
    __syncthreads();

    // Parallel recon norm: state at step sid = 8 outputs (1 bit each); sid<7 walk from initial
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 7) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++)
                cur_state = (cur_state >> 1) | (((int)outputs[t]) << 7);
        } else {
            cur_state = 0;
#pragma unroll
            for (int b = 0; b < 8; b++)
                cur_state |= ((int)outputs[sid - 7 + b] & 0x1) << b;
        }
        float c = cbk[cur_state];
        my_recon_sq = c * c;
    }
    cost[sid] = my_recon_sq;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 16);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 8);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 4);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 2);
        v += __shfl_down_sync(0xFFFFFFFFULL, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;

    // Parallel bitpack: 7 init-state bits then 128 1-bit output symbols (17 bytes).
    if (sid < 17) {
        const int init_bits = (shared_initial_state >> 1) & 0x7F;  // 7 bits
        uint8_t packed = 0;
#pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            const int pos = sid * 8 + bit;
            int v = 0;
            if (pos < 7) {
                v = (init_bits >> pos) & 1;
            } else {
                const int sym_idx = pos - 7;
                if (sym_idx < 128) v = outputs[sym_idx] & 1;
            }
            packed |= (uint8_t)(v << bit);
        }
        dst_blk->qs[sid] = packed;
    }
    if (sid == 0) dst_blk->norm = __float2half(corrected_norm);
}

// === TURBO1_TCQ: GET_ROWS dequantize (rotated-domain trellis lookup; not used for KV) ===
#define QR_TURBO1_TCQ 2
static __device__ __forceinline__
void dequantize_turbo1_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo1_tcq * x = (const block_turbo1_tcq *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int bp = iqs;      const uint16_t raw = (uint16_t)x[ib].qs[bp>>3] | ((uint16_t)x[ib].qs[(bp>>3)+1] << 8);
      v.x = norm * d_turbo1_tcq_codebook[(raw >> (bp & 7)) & 0xFF]; }
    { const int bp = iqs + 64; const uint16_t raw = (uint16_t)x[ib].qs[bp>>3] | ((uint16_t)x[ib].qs[(bp>>3)+1] << 8);
      v.y = norm * d_turbo1_tcq_codebook[(raw >> (bp & 7)) & 0xFF]; }
}
