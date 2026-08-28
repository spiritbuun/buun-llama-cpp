# Native compressed-tensors packed integers

## Objective

Run compressed-tensors `pack-quantized` WNA16 checkpoints directly from their
safetensors directories with competitive fidelity, resident VRAM, prompt
processing, and token generation. The initial production target is:

- `TelperionAI/Qwen3.8-27B-INT4-AWQ-GPTQ-gdn4`
- revision `dfd2484e16dec819fb95f6f51721556c57b6324e`
- single RTX A6000, compute capability 8.6

The target deliberately exercises mixed contracts and Qwen3.5 recurrent-layout
transforms. It is therefore a useful architecture gate rather than merely a
kernel microbenchmark.

## Source contract

The checkpoint uses compressed-tensors `pack-quantized`, not the legacy
AutoAWQ or AutoGPTQ container layouts. It has 400 quantized projections:

| Contract | Modules | Stored tensors | Shape convention |
| --- | ---: | --- | --- |
| INT4 asymmetric group-32 | 312 | I32 `weight_packed`, BF16 `weight_scale`, I32 `weight_zero_point`, I64 `weight_shape` | weights packed along input columns; zero points packed along output rows |
| INT8 symmetric group-128 | 88 | I32 `weight_packed`, BF16 `weight_scale`, I64 `weight_shape` | weights packed along input columns |

Packing follows compressed-tensors' `pack_to_int32`: consecutive low-to-high
bit lanes, with signed integers offset into the unsigned storage range before
packing. For asymmetric INT4, the common packing offset cancels between the
weight and zero point, so dequantization is `(unsigned_code -
unsigned_zero_point) * scale`. For symmetric INT8, dequantization is
`(unsigned_code - 128) * scale`.

The config declares static activation ordering, not arbitrary GPTQ `g_idx`
activation gathering. The recipe used AWQ smoothing followed by GPTQ weight
calibration; those are calibration methods, not distinct runtime containers.

## Baselines and acceptance gates

Record every retained candidate against the same device, clocks, context, KV
type, and token shapes.

1. vLLM 0.28.0 at source revision
   `2cf0a6915ce544dc493a0990f2ea38d81601128a`, tensor parallel 1. Its source
   selects `CompressedTensorsWNA16` and `MarlinLinearKernel` for this contract.
2. The exact source checkpoint occupies 21.09 GiB. vLLM reports 19.87 GiB of
   model memory in text-only mode before KV allocation.
3. Measure PP from the differential `(p512,o1) - (p1,o1)` and TG from
   `(p1,o128) - (p1,o1)`, five warmed repeats.
4. Measure llama.cpp with `llama-bench` at the same logical shapes and with
   BF16/F16 KV held constant. Report model-buffer VRAM separately from context,
   compute, and VMM reservations.
5. Fidelity gate: dump matching deterministic logits and require near-zero KLD
   against vLLM. Exact identity is not expected when the execution arithmetic
   differs. Any larger divergence must be localized before performance work is
   accepted.
6. Structural gate: scalar-dequantize selected projections from the source and
   compare every canonical row after Qwen3.5 row/column permutations.

## Phase 1 — parse and validate the container

- Add one generic packed-integer quant format with explicit `num_bits`,
  `group_size`, `symmetric`, and activation-order fields.
- Accept only the two contracts above initially. Reject dynamic weights,
  non-null activation quantization, unsupported zero-point dtypes, arbitrary
  group maps, and malformed packed dimensions before model allocation.
- Treat `weight_packed`, `weight_scale`, `weight_zero_point`, and `weight_shape`
  as one binding and require complete consumption.
- Add synthetic fixtures with row-, lane-, group-, scale-, and zero-distinct
  values. Cover padding and malformed shapes.

## Phase 2 — exact reference materialization

- Materialize asymmetric W4 group-32 into Q4_1. Each source group is already
  one Q4_1 block: retain its 4-bit codes, store the BF16 scale as FP16, and fold
  the unsigned zero point into Q4_1's minimum term.
- Materialize symmetric W8 group-128 into four Q8_0 blocks. Decode each packed
  byte to signed INT8 and repeat the group scale for the four 32-value blocks.
- Apply Qwen3.5 recurrent row/column permutations to the canonical quantized
  blocks, not to packed source bytes. Row transforms move complete quantized
  rows; the value-head column transform moves aligned 32-value blocks.
- Run CPU, CUDA, sanitizer, full-model coherence, and differential-logit gates.

This phase intentionally reuses mature Q4_1/Q8_0 execution. It establishes an
honest end-to-end baseline before adding a new kernel. Q4_1 is 4.5 bpw versus
the source INT4 contract's 4.625 bpw. Q8_0 is 8.5 bpw versus the source INT8
contract's 8.125 bpw, so only the INT8 subset has a resident-weight penalty.

## Phase 3 — profile before choosing a native executor

- Compare the reference baseline against vLLM's Marlin result at batch widths
  1, 8, 32, 128, and 512.
- Profile decode and prefill separately. Determine whether Q4_1/Q8_0 matrix
  multiplication, recurrent GDN, graph overhead, or another operation owns the
  remaining gap.
- Record DP4A/tensor-pipe utilization, memory throughput, occupancy, register
  pressure, and launch count for the dominant projections.
- Do not add a new type if existing Q4_1 is already at parity and fidelity is
  acceptable.

## Phase 4 — close measured storage or execution gaps

Candidate A, if INT8 storage matters: add a symmetric Q8 group-128 block type
with one FP16 scale and 128 signed bytes (8.125 bpw). Its CUDA dot product reuses
four Q8 activation sub-blocks under one weight scale. Provide CPU reference and
HIP implementations before presenting it as generally supported.

Candidate B, if W4A8 arithmetic or throughput is the limiting factor: add a
W4A16 group-32 executor. Keep the architecture-independent tensor type and
dispatch contract in ggml; perform any Ampere kernel-layout repack during load,
as vLLM's `MarlinLinearKernel.process_weights_after_loading` does. Preserve a
portable canonical representation for CPU/HIP/offload rather than making an
opaque CUDA-only layout the model's sole storage.

Candidate C, if the gap is primarily prefill: specialize MMQ/GEMM batch shapes
without changing the stored representation. Candidate D, if it is decode:
specialize MMVQ and fuse scale/zero arithmetic into the dot-product loader.

Every candidate must beat the Phase-2 reference in its intended regime and
must not increase total resident VRAM beyond vLLM. Backend-specific repacks must
be owned by the backend and remain valid across views, partial writes, and
teardown.

## Phase 5 — broader coverage

- Generalize group sizes only after a real checkpoint and a measured use case.
- Keep legacy AutoAWQ and non-act-order GPTQ adapters as separate source
  containers feeding shared canonical execution.
- Treat arbitrary GPTQ act order as a separate gather/runtime contract.
- Add tensor splitting, layer splitting, CPU offload, and HIP gates after the
  single-GPU executor is correct and competitive.

## Stop conditions

- No performance patch without a repeatable pre-patch loss and post-patch win.
- No silent conversion to a numerically different quantization contract.
- No claim of vLLM parity unless model memory, PP, and TG are all measured on
  the same GPU under stated settings.
