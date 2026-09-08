# Experimental SM120 FP8 epilogue

This provider is outside the normal CMake source glob. It is a research path,
not a new build dependency or public default. Tested with CUDA12.8, SM120a,
and CUTLASS4.3.4 headers. Build it separately:

```sh
nvcc -O3 -std=c++17 -use_fast_math -arch=sm_120a \
  --expt-relaxed-constexpr -Xcompiler=-fPIC -shared \
  -I /path/to/cutlass/include \
  ggml/src/ggml-cuda/experiments/fp8-cutlass-sm120.cu \
  -o /path/to/libfp8-cutlass-sm120.so
```

Set `GGML_CUDA_FP8_CUTLASS_LIBRARY` to that library alongside the existing
`GGML_CUDA_FP8_LT` research gate. The conservative tested configuration uses
`GGML_CUDA_FP8_LT_SPLIT_K=2`. The one-part variant changes model logits and
must not be described as fidelity-neutral. Batches padded to fewer than 1024
rows and unsupported shapes retain the existing path. Broader numerical and
lifecycle qualification and dependency packaging are required before
considering a public default.

The caller queries scratch size before execution, supplies context-owned
storage and the CUDA stream, and keeps both valid through GPU completion.
The provider has no persistent device storage or internal allocation. Its
code remains loaded because captured CUDA graphs may reference its kernels.

Configuration informed by vLLM v0.28.0 (commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`), especially `scaled_mm.cuh` and
`scaled_mm_sm120_fp8_dispatch.cuh` under
`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/`. This wrapper avoids Torch,
writes FP32 outputs, uses native scale order, and supports a two-part reduction
inside the final epilogue. See LICENSE (Apache2.0); external CUTLASS headers
retain their own BSD3-Clause notices.
