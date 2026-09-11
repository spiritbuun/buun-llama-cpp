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
2. The text-only source tensors selected by both engines occupy about 19.44 GiB
   in their exact quantization contracts. vLLM reports 19.99 GiB of consumed
   weights plus non-framework allocations after initialization (19.87 GiB in
   its model-memory counter); keep those counters distinct.
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

## Phase 2 — portable reference materialization

- The first reference materialized asymmetric W4 group-32 into Q4_1, converting
  BF16 scale/minimum metadata to FP16. Q4_1 is 5.0 bpw, not 4.5 bpw.
- The first reference materialized symmetric W8 group-128 into four Q8_0
  blocks, repeating the group scale for each 32-value block (8.5 bpw).
- Apply Qwen3.5 recurrent row/column permutations to the canonical quantized
  blocks, not to packed source bytes. Row transforms move complete quantized
  rows; the value-head column transform moves aligned 32-value blocks.
- Run CPU, CUDA, sanitizer, full-model coherence, and differential-logit gates.

This phase intentionally reused mature Q4_1/Q8_0 execution to establish an
end-to-end baseline. On the target model, however, that expanded the 312 W4
modules by 0.895 GiB and the 88 W8 modules by 0.167 GiB: 1.062 GiB total.

The production baseline therefore uses two source-faithful portable types:
W4 asymmetric group-32 in 128-value blocks (4.625 bpw), and W8 symmetric
group-128 (8.125 bpw). Both retain BF16 scales exactly. The W4 block boundary
matches Qwen3.5's 128-wide recurrent value heads, so recurrent row/column
permutations move whole blocks without decoding or changing the quantization.
CUDA initially executes these through generic dequantization plus cuBLAS; the
optimized executor must consume the same canonical allocation rather than
retaining a second permanent packed copy.

Measured on one RTX A6000 with the 27B gdn4 checkpoint:

| executor | model bytes | peak process VRAM | PP512 | TG128 |
| --- | ---: | ---: | ---: | ---: |
| expanded Q4_1/Q8_0 reference | 20.505 GiB | — | 1494.57 t/s | 32.10 t/s |
| compact, native MMVQ decode | 19.443 GiB | 18,728 MiB | 1160.68 t/s | 33.59 t/s |
| vLLM | 19.87 GiB model counter | 19.99 GiB consumed | 1970.40 t/s | 34.01 t/s |

llama.cpp cells report the three-repeat mean; the retained vLLM cell reports
the warmed median from its differential harness.

The native MMVQ result closes decode to within 1.2% of vLLM without a second
packed allocation. The remaining measured gap is prompt processing: compact
weights still take the generic dequantize-plus-cuBLAS path above the MMVQ batch
limit. Phase 4C (native MMQ/GEMM) is therefore the next executor task.

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

Candidate A: optimize the symmetric Q8 group-128 type with one BF16 scale and
128 signed bytes (8.125 bpw). Its CUDA dot product can reuse four Q8 activation
sub-blocks under one weight scale. Provide a HIP implementation before
presenting it as generally supported.

Candidate B: add a W4A16/W4A8 group-32 executor over the compact canonical
type. A backend-private layout is acceptable only if it replaces, rather than
duplicates, the device allocation and remains reconstructible for CPU/HIP
offload. Prefer a kernel that reads canonical blocks directly.

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
