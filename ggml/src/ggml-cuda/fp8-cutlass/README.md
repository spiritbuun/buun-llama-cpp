# SM120 native channel-FP8 kernels

These kernels serve native **F32 channel-scale** FP8 projections. BF16-scale
projections use the separate precision-preserving executor in
`../fp8-channel-bf16.cu`; converting their scales to F32 to enter this path
changes numerical behavior and is not an import optimization.

The CUDA backend builds this object automatically when CUDA >= 12.8 and SM120
are selected. It is linked into ggml-cuda; no external provider library or
environment variable is required. Unsupported shapes fall back to cuBLASLt
or the general backend. Windows compilation/runtime remains untested.

Retained implementation choices:

- Two independently accumulated K halves, combined with explicit F32 rounding.
- Register-retained split2 and a small-N tile, with a two-launch split2 fallback.
- Paired gate/up FFN with independent activation clipping markers.
- Paired F32 output stores and fused SwiGLU.
- Row-retained activation packing at the measured widths.

Single-part accumulation, alternative reference arithmetic, tuning switches,
diagnostic entry points, and prefix/checkpoint experiments are not included.

## Source provenance

The wrapper code was developed in this fork. The modified shared-memory
copy/MMA pipeline derives from NVIDIA CUTLASS; its BSD-3-Clause notice is
retained in the corresponding files. `LICENSE` covers the Apache-licensed
wrapper; individual source notices apply where present.

`vendor/include` is the CUTLASS include tree shipped with vLLM 0.28.0 at
`vllm/third_party/fmha_sm100/cutlass/include`. It is isolated to this object
target so it cannot shadow the backend's existing CUTLASS subset. The archived
header bundle used for integration has SHA-256:

```text
b1a6b0e6ec8c15ba8b3a0591fe6878c262361eda740f59fbe1ea04bc8bb378f1
```

This records the actual packaged source used, not an inferred upstream Git
revision. NVIDIA license/copyright notices are preserved in the headers.
