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

For two parts and K divisible by256, one kernel runs the original CUTLASS
main loop twice, retaining the first accumulator in registers. This preserves
the two independent accumulations without a global partial matrix. Other
supported K sizes keep the two-launch implementation; the one-part path is
unchanged. This uses the unmodified CUTLASS headers, not a forked main loop.

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

## Paired channel-FP8 FFN experiment

`fp8-cutlass-ffn-sm120.cu` combines two dense gate/up projections with their
SwiGLU epilogue. It reads the original weight tensors directly, with no weight
repack or duplicate resident weights. Build separately against the same
**unmodified** CUTLASS headers:

```sh
nvcc -O3 -std=c++17 -use_fast_math -arch=sm_120a \
  --expt-relaxed-constexpr --cudart shared -Xcompiler=-fPIC -shared \
  -I /path/to/cutlass/include \
  ggml/src/ggml-cuda/experiments/fp8-cutlass-ffn-sm120.cu \
  -o /path/to/libfp8-cutlass-ffn-sm120.so
```

Set `BUUN_PRIVATE_FFN_PAIR_LIBRARY` to this library alongside the FP8 research
settings above. The host matcher currently selects SM120, dense contiguous
F32 inputs, F32 channel scales, dynamic I32 activation markers, K=5120,
split2, and batches padded to at least 1024 rows. Other shapes use the existing
path. Graph use-count and memory-range checks remain mandatory.

The two clipping settings need not match. The packer computes their row scales
independently; the device GEMM uses one activation matrix when the markers
agree and two otherwise. Its transfer byte count follows the same condition.
This works across CUDA-graph replays with changed marker values and introduces
no host scalar cache. It reserves a second packed-activation workspace and
additional shared-memory capacity; these costs are not inherently free.
The unequal-marker path is correct but is not claimed faster on every shape.

On the tested Qwen3.8-27B channel-FP8 / RTX5090 workload, balanced warmed PP2048
rose from 5788.7 to 5849.9 tok/s (~1.06%). The source-packaged recheck gave
5789.7 to 5853.5 tok/s (~1.10%). Exact model-logit repeats, a 1020-token
ubatch boundary, 24 serving token comparisons and 18 changing-marker graph-replay
cells passed. Decode changed by less than 0.25%; measured device occupancy
remained 27456 MiB. These are model-specific research results, not parity with
vLLM or qualification for a public default. No checkpoint feature was disabled.

The provider exports `buun_fp8_cutlass_ffn`; early private prototypes had
different entry points and incompatible argument lists. Do not substitute
those old libraries. Layout-check and F32-reference entry points are retained
for standalone correctness tests. The adapted dual-input MMA routine preserves
the upstream BSD notice; the main single-input path reuses the retained GEMM.

## Prefix-checkpoint state replay

The private `BUUN_PRIVATE_PREFIX_CHECKPOINT` server pilot captures a separate
F32 recurrent state after token 2044 while evaluating a full 2048-token prompt.
It retains the existing checkpoint serializer, frontier, and publication policy.
Only eligible single-slot, fully offloaded Qwen35 hybrid text prompts use it;
other configurations retain the ordinary split. Its extra roughly 150 MiB GPU
plane is not yet integrated into auto-fit, so it is not a public default.

`gdn-prefix-chunk-sm120.py` adapts the pinned vLLM FLA state kernel to preserve
an additional F32 state before chunk 31. The backend can then replay only the
last partial chunk instead of the whole prefix. It does not restart from the
ordinary BF16 chunk-state storage. Generate the optional module on SM120 with
the pinned vLLM/Triton environment used for the existing FLA modules:

```sh
python ggml/src/ggml-cuda/experiments/gdn-prefix-chunk-sm120.py /path/to/output
```

Set `BUUN_PRIVATE_PREFIX_CHUNK_CUBIN` to the generated `prefix_chunk_state.cubin`.
Its ABI is specific to H=48, Hg=16, D=128, BT=BV=64, four warps, three stages,
and chunk 31; the native dispatch additionally requires a 2048-token batch and
a prefix after token 1984. Other prefixes retain full replay. The temporary
F32 plane uses 3 MiB of backend scratch, shared across layer executions.
The experiment preserves full output, final state and prefix state exactly in
component/graph-replay and full-model checkpoint/restore gates. This is not a
generic checkpoint architecture or support for speculative rollback planes.

## Checkpoint host-page experiment

On Linux, `BUUN_PRIVATE_CHECKPOINT_HUGE_PAGES=1` advises transparent huge pages
for newly allocated checkpoint byte buffers of at least 8 MiB, before their
normal zero initialization. It changes neither the serialized format nor
immutable ownership/accounting: invalidated buffers are still freed immediately,
and failed overwrites still leave the old payload intact. Unset the variable for
the ordinary allocation path; failed advice falls back to ordinary pages.

On the same RTX5090 host, a balanced warmed PP2048 run with both prefix experiments
above improved from 6200.7 to 6399.0 tok/s (3.20%). Timers attributed most of the
gain to freeing the previous ~150 MiB checkpoint: ~8–10 ms became ~0.8 ms.
The GPU computations are unchanged. This remains below the matched vLLM result.

This is not a recommended public default: transparent huge-page allocation can
perform synchronous reclaim/compaction under memory pressure, depending on the
host's Linux settings. The experiment does not change those settings or reserve
a persistent host pool. Memory-pressure/tail-latency qualification is still needed.
