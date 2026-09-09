// Quantize one common F32 row with two independent clipping bounds. If the
// markers agree, only the first packed matrix is needed by the paired GEMM.
static __global__ void ffn_dual_pack_kernel(const float *   src,
                                            __nv_fp8_e4m3 * gate,
                                            __nv_fp8_e4m3 * up,
                                            float *         gs,
                                            float *         us,
                                            const int32_t * gm,
                                            const int32_t * um,
                                            int             m) {
    const int row = blockIdx.x;
    float     values[20], maximum = 0;
#pragma unroll
    for (int i = 0; i < 20; ++i) {
        const int col = threadIdx.x + i * 256;
        values[i]     = row < m ? src[int64_t(row) * 5120 + col] : 0.0f;
        maximum       = fmaxf(maximum, fabsf(values[i]));
    }
    __shared__ float maxima[8];
    maximum            = block_reduce<block_reduce_method::MAX, 256>(maximum, maxima);
    const float gbound = __int_as_float(*gm), ubound = __int_as_float(*um);
    const float gmax   = gbound > 0 ? fminf(maximum, gbound) : maximum;
    const float umax   = ubound > 0 ? fminf(maximum, ubound) : maximum;
    const float gscale = gmax / 448.0f, uscale = umax / 448.0f;
    const float gi = gmax == 0 ? 0 : 1.0f / gscale, ui = umax == 0 ? 0 : 1.0f / uscale;
    if (threadIdx.x == 0) {
        gs[row] = gscale;
        us[row] = uscale;
    }
    const bool same = *gm == *um;
#pragma unroll
    for (int i = 0; i < 20; ++i) {
        const int64_t offset = int64_t(row) * 5120 + threadIdx.x + i * 256;
        gate[offset]         = __nv_fp8_e4m3(values[i] * gi);
        if (!same) {
            up[offset] = __nv_fp8_e4m3(values[i] * ui);
        }
    }
}
