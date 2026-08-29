# Native safetensors architecture bridge

Status: implementation in progress. The tensor-source interface is internal and is not a stable public API.

## 1. Objective

Native safetensors loading should be a source-format adapter, not a second model implementation.

For an architecture already supported through GGUF, adding safetensors support should normally require:

1. mapping Hugging Face configuration fields into the existing llama.cpp model metadata;
2. mapping canonical runtime tensor names to source tensor names;
3. declaring the small number of architecture-specific layout transforms; and
4. selecting an already-supported quantization adapter.

The existing model class must remain the authority for tensor topology and graph construction. Existing CPU, CUDA, HIP, split/offload, VBR, speculative-decoding, and graph-fusion paths should be reused rather than reimplemented in a safetensors importer.

New backend kernels should only be needed for a genuinely new numerical format or operation, not because weights arrived in a safetensors container.

## 2. Current state

The first implementation already reuses the runtime after import:

```text
safetensors directory
    -> safetensors registry and quantization-config parser
    -> metadata-only compatibility GGUF plus registered architecture importer
    -> internal llama_model_tensor_source
    -> existing llama_model_qwen35 tensor creation and graph
    -> existing ggml backend dispatch
```

### Already generic

- Safetensors shard discovery, header parsing, dtype/shape validation, bounds checks, and tensor reads.
- Ordered compressed-tensors rule matching.
- The callback-backed `llama_model_loader` path.
- Device placement and backend-buffer selection after a runtime tensor has been described.
- Backend dispatch by ggml operation, tensor type, shape, and layout.
- The existing Qwen3.5 model graph and its optimized kernels.

### Still coupled to Qwen3.5

- Only one architecture importer is registered so far (Qwen3.5), but selection
  now goes through an ambiguity-checking importer registry.
- The importer manually constructs Qwen-specific GGUF model keys; tokenizer,
  sampling, RoPE parsing, and metadata storage are shared helpers.
- Canonical-to-source name mapping, plain-tensor dtype conversion, and Qwen-specific
  permutations are still mixed in one file. Quant-format binding and validation
  now live in a reusable adapter.
- Model validation is fixed to one Qwen3.5 geometry.
- The quantization, architecture, and metadata seams are separated. Explicit
  transform objects and shared ordinary-tensor naming remain before
  architecture two is cheap.

The duplicated target manifest has been removed. The next structural target is
extracting reusable naming and transform helpers so a second architecture does
not copy Qwen-specific importer machinery.

## 3. Required invariants

These are design constraints, not optional cleanup goals.

1. **One graph implementation.** Safetensors and GGUF for one architecture use the same `llama_model_*` implementation and build the same logical graph.
2. **One tensor-topology authority.** `load_arch_tensors()` and its calls to `create_tensor()` define which runtime tensors exist. An importer must not maintain a second complete target list.
3. **Canonical runtime tensors.** Once admitted, a source tensor has an ordinary canonical llama.cpp name, ggml type, shape, strides, and quantization auxiliaries. Backends must not branch on the original container format.
4. **Quantization is architecture-independent where possible.** FP8 block, FP8 channel, NVFP4, W8A8, AWQ, and GPTQ contracts belong to reusable quant adapters.
5. **Architecture transforms are explicit.** Row/column permutations, offset norms, recurrent-state conventions, and fused source projections remain in the architecture adapter.
6. **No hidden conversion of the whole model.** Loading may transform one destination tensor at a time. It must not create a complete temporary GGUF or a second resident copy of the model.
7. **Backend capability is checked before expensive allocation.** Unsupported type/backend combinations fail with a precise message or use an explicitly supported fallback.
8. **Source files remain read-only.** Repacking occurs in bounded host staging or directly into the selected destination buffer.
9. **The GGUF path does not regress.** The new source abstraction must add no work to ordinary GGUF loading or inference after construction.
10. **Failure is strict.** Missing scales, ambiguous name mappings, unsupported zero points, shape mismatches, and unconsumed required quantized tensors are errors rather than silent coercions.

## 4. Target architecture

```text
                         +--------------------------+
                         | existing llama_model_*   |
                         | hparams, tensors, graph  |
                         +------------+-------------+
                                      |
                              canonical requests
                                      |
                         +------------v-------------+
                         | llama_tensor_source      |
                         | describe / bind / load   |
                         +------------+-------------+
                                      |
                   +------------------+------------------+
                   |                                     |
          +--------v---------+                  +--------v---------+
          | GGUF source      |                  | safetensors      |
          | existing path    |                  | source           |
          +------------------+                  +--------+---------+
                                                            |
                                      +---------------------+--------------------+
                                      |                                          |
                             +--------v---------+                       +--------v---------+
                             | architecture     |                       | quant adapter    |
                             | config/names/    |                       | format contract/ |
                             | exceptions       |                       | materialization  |
                             +------------------+                       +------------------+
```

### 4.1 Generic tensor-source seam

Introduce an internal source interface usable by both the existing GGUF path and callback-backed sources. Exact names can change during implementation, but the responsibilities should resemble:

```cpp
struct llama_source_tensor_desc {
    ggml_type type;
    std::array<int64_t, GGML_MAX_DIMS> ne;
};

class llama_tensor_source {
public:
    virtual bool describe(
        const std::string & canonical_name,
        llama_source_tensor_desc & out) const = 0;

    virtual void load(
        const std::string & canonical_name,
        ggml_tensor & destination) const = 0;

    virtual void validate_complete() const = 0;
    virtual ~llama_tensor_source() = default;
};
```

`llama_model_loader::create_tensor()` must ask the source for type and source-described dimensions when a virtual source is active. This replaces the current requirement that every tensor first be inserted into synthetic GGUF tensor metadata.

The existing model's requested dimensions remain an independent check. A source description does not get to redefine the architecture silently.

`get_tensor_info()` must use the same source seam so optional scale tensors and other architecture-dependent auxiliaries work without a synthetic tensor manifest.

### 4.2 Safetensors architecture adapter

An architecture adapter owns only source naming and genuine architecture semantics:

```cpp
class llama_safetensors_arch_adapter {
public:
    virtual bool probe(const hf_config & config) const = 0;
    virtual void emit_model_metadata(metadata_sink & sink) const = 0;
    virtual source_binding bind(const canonical_tensor_request & request) const = 0;
    virtual void validate_model_contract() const = 0;
};
```

A `source_binding` identifies source tensors and a composable transform chain. It should not directly contain backend code.

Common Hugging Face naming templates should be reusable:

- embeddings and output head;
- input/post-attention norms;
- Q/K/V/O projections;
- dense gate/up/down MLP projections;
- common MoE router and expert tensors.

Architecture adapters override only exceptional names or layouts. Qwen3.5 retains ownership of its linear-attention projections, QKV/Z permutations, output-projection column permutation, offset-norm conversion, `A_log` transform, and MTP naming.

### 4.3 Quantization adapters

Move numerical-format handling out of the Qwen adapter. Each reusable quant adapter must define:

- how it recognizes a producer contract;
- required weight, scale, zero-point, and optional auxiliary tensors;
- legal source dtypes and shapes;
- destination ggml type and dimensions;
- whether the source bytes are already canonical;
- bounded materialization/repack steps;
- supported CPU/CUDA/HIP backends and fallback behavior;
- validation of all correctness-critical auxiliaries.

Initial adapters:

| Adapter | Source contract | Runtime representation |
|---|---|---|
| FP8 channel | E4M3 weight plus BF16 per-output-channel scale | `GGML_TYPE_F8_E4M3` plus canonical scale tensor |
| FP8 block | E4M3 weight plus BF16 128x128 inverse-scale grid | `GGML_TYPE_F8_E4M3` plus canonical block-scale tensor |
| NVFP4 | packed 4-bit values, FP8 group scales, global input/weight scales | `GGML_TYPE_NVFP4` plus canonical auxiliaries |

Future W8A8, AWQ, and GPTQ support should add adapters rather than branches to every architecture importer.

### 4.4 Transform pipeline

Represent materialization as ordered, typed transforms instead of one architecture-specific function containing all cases.

Candidate transform stages:

1. source tensor read or mapped span;
2. source dtype validation;
3. generic rank/shape normalization;
4. architecture row or column permutation;
5. architecture value transform, such as offset-norm `+1` or `-exp(A_log)`;
6. quant adapter repack or scale conversion;
7. destination upload.

Ordering must be explicit because a permutation of packed values may require a corresponding permutation of scales. A binding must either transform every coupled tensor consistently or be rejected.

Transforms should expose whether they can operate in chunks. Full-tensor staging remains acceptable initially, bounded to one destination tensor, but streaming transforms are a later optimization.

### 4.5 Metadata and tokenizer bridge

Keep metadata separate from tensor enumeration.

Provide reusable helpers for:

- standard Hugging Face scalar configuration fields;
- RoPE and MRoPE configuration;
- sampling defaults;
- tokenizer vocabulary, merges, added tokens, and special-token IDs;
- chat-template discovery;
- conversion into the metadata keys already consumed by each model class.

The first refactor may continue emitting an in-memory GGUF metadata context for compatibility. It should contain model/tokenizer metadata only; tensor descriptions should come from `llama_tensor_source`.

Bypassing the in-memory GGUF metadata layer entirely is not required for this project. Reusing the existing hparam parser is lower risk and does not create an on-disk conversion.

### 4.6 Importer registry

Replace the hard-coded Qwen constructor with an ordered internal registry:

1. parse enough `config.json` to identify the architecture;
2. ask registered architecture adapters to probe;
3. require exactly one match;
4. construct the generic safetensors source with that adapter and parsed quantization configuration;
5. invoke the ordinary model-loading path.

Ambiguous matches and unsupported architectures must report the detected `model_type`, `architectures`, and quantization method.

## 5. Implementation phases

Each phase should be independently reviewable and should preserve a runnable Qwen3.6 path.

### Phase 0 — Freeze trustworthy baselines

- [ ] Record the exact Qwen3.6 NVFP4 and official block-FP8 model revisions.
- [ ] Record A6000 PP, TG, peak VRAM, load time, and first-token correctness.
- [ ] Preserve direct projection correctness tests for every supported shape.
- [ ] Preserve same-process reference-vs-reference anchor tests; the anchor must be exactly zero.
- [ ] Disable or reject experimental backend paths that are known to produce incorrect output.
- [ ] Save vLLM commands, versions, and corresponding baselines.

Gate: both source models load and produce coherent greedy output, and the recorded performance is reproducible within the chosen thermal/clock tolerance.

Current known exception: explicitly enabling the experimental Humming NVFP4
path produced incoherent greedy output on the A6000 while the ordinary NVFP4
path remained coherent. The experimental path stays opt-in and must not be
treated as a correctness baseline until separately repaired and revalidated.

### Phase 1 — Introduce `llama_tensor_source` without changing behavior

- [x] Define the internal source interface and ownership/lifetime rules.
- [x] Adapt the direct safetensors loader to implement it while preserving the public callback API.
- [x] Route `describe`, `get_tensor_info`, loading, progress accounting, and errors through one interface.
- [x] Keep synthetic tensor metadata temporarily as the source of descriptions.
- [ ] Add tests for callback failure propagation, cancellation, optional tensors, and source lifetime.

Gate: no tensor bytes, graph nodes, logits, PP, or TG change from the Phase-0 safetensors baseline; GGUF tests remain unchanged.

### Phase 2 — Make existing model tensor requests authoritative

- [x] Teach virtual `create_tensor()` to query `llama_tensor_source::describe()` directly.
- [x] Remove the requirement that virtual tensor descriptions live in GGUF tensor metadata.
- [x] Route `get_tensor_info()` through the same source.
- [x] Track successfully bound and loaded canonical tensors. Optional description probes do not count as bindings.
- [x] Extend `validate_complete()` from bound/load parity to required auxiliaries
  of every bound quantized weight. Dormant optional submodels may remain unbound.
- [x] Delete `qwen35_target_names()` after full-model parity smokes.

Gate: a recorded canonical manifest from `load_arch_tensors()` matches the former synthetic manifest exactly for both NVFP4 and block-FP8 models.

### Phase 3 — Extract generic quant adapters

- [x] Define the complete quant source-binding representation.
- [x] Move FP8-channel contract validation and scale binding out of the Qwen file.
- [x] Move FP8-block 128x128 validation, inverse-scale handling, and scale transpose out of the Qwen file.
- [x] Move NVFP4 packed-weight validation, repack, and scale binding out of the Qwen file.
- [x] Centralize dtype size and dtype-to-ggml-type mapping where it is truly format-generic.
- [x] Add isolated fixtures for missing scales, wrong scale dtype, wrong grid
  shape, zero points, invalid global scalars, dependency consumption, and
  overlapping compressed-tensors rules.

Gate: quant-adapter unit tests cover each accepted and rejected contract; Qwen output and performance remain at baseline.

### Phase 4 — Extract reusable metadata and tokenizer helpers

- [x] Split generic JSON/file helpers from the Qwen adapter.
- [x] Add a metadata sink with strict required/optional field handling.
- [x] Extract tokenizer vocabulary, merges, added-token, special-token, and chat-template conversion.
- [x] Extract common RoPE and sampling-default helpers without weakening architecture validation.
- [x] Keep architecture-specific GGUF key selection in the architecture adapter.

Gate: tokenizer round-trip tests match canonical token IDs and encode/decode results; chat-template output matches the GGUF reference.

### Phase 5 — Reduce Qwen3.5 to architecture-specific policy

- [x] Replace imperative ordinary projection mapping with shared naming templates.
- [x] Keep only recurrent projection names, MTP names, and genuine Qwen exceptions locally.
- [x] Express row/column permutations as explicit transform objects.
- [x] Express offset-norm and `A_log` conversions as explicit transform objects.
- [x] Replace the fixed 27B check with validated geometry where the existing Qwen model implementation supports it.
- [x] Report block-FP8 and NVFP4 source formats distinctly.

The Qwen adapter now accepts the dense model implementation's 24-, 32-, and
64-layer families, derives the recurrent K/V head grouping and dimensions from
`text_config`, and maps MTP after the configured trunk rather than after a
hard-coded layer 64. It rejects multiple MTP blocks, non-integral V/K head
groups, unequal recurrent K/V dimensions (the current graph uses one SSM state
dimension for both), and block-FP8 head layouts that cannot be expressed as an
exact 128-row scale grid. Channel scales and block-grid scales carry distinct
transform plans; in particular, an output channel scale is not permuted with
the input columns of its weight.

Gate: the Qwen adapter contains no quant-format implementation and no complete target manifest.

### Phase 6 — Prove the seam with a second existing architecture

- [x] Select a dense architecture already well supported through GGUF and available in one supported safetensors quant format.
- [x] Add only its configuration/name/layout adapter.
- [x] Do not add a model graph or duplicate backend kernel for the sake of safetensors.
- [x] Document which portions were reused unchanged.

Preferred proof target: a conventional dense Llama- or Qwen3-family model. A second irregular hybrid architecture should wait until the common path is proven.

Gate: the second architecture loads through its existing `llama_model_*` class, matches a clean GGUF/reference KLD panel within the declared near-zero tolerance, and exercises the same backend kernels for equivalent runtime tensor types.

Proof target: `RedHatAI/Qwen3-4B-FP8-dynamic`, a conventional 36-layer
Qwen3 model with channel-scaled E4M3 projections. The adapter adds Qwen3
configuration metadata and root names only. It reuses the ordinary decoder
name mapper, tokenizer/metadata bridge, compressed-tensors parser, generic
tensor materializer, existing `llama_model_qwen3` graph, and existing
F8-E4M3/BF16-scale backend path unchanged. No Qwen3 safetensors graph or
backend kernel was added.

The proof also generalized two producer/container seams discovered by the
second model: older compressed-tensors checkpoints may declare the
single-format `float-quantized` schema and select projections by the `Linear`
module class, and pure channel-FP8 GGUF export must preserve F8 tensors just as
the mixed NVFP4+FP8 exporter already did.

Gate results on one RTX A6000:

- 651/651 canonical tensors matched the GGUF control byte-for-byte
  (5,192,136,704 bytes).
- Both paths exposed 145 F32, 254 BF16, and 252 F8-E4M3 tensors.
- All 16,384 live WikiText logit rows matched bit-for-bit across the full
  151,936-token vocabulary. The resulting KLD is exactly zero.
- The native directory loaded through `llama_model_qwen3` and generated a
  coherent greedy continuation.

The live-logit gate is intentional. `llama-perplexity`'s `.kld` artifact
quantizes saved log probabilities to `uint16_t`, so comparing a model to its
own saved artifact is not an exact-zero anchor. Native and GGUF produced
byte-identical per-position values even through that lossy instrument, but the
live comparison is the proof of exact equality.

### Phase 7 — Loading and memory optimization

- [x] Replace avoidable source-vector copies with read-only shard mappings.
- [x] Read row- and block-local transforms from mappings into one final target buffer.
- [x] Upload identical source layouts directly into the final selected backend buffer.
- [x] Preserve one-final-tensor-at-a-time peak-memory bounds for transformed tensors.
- [x] Investigate zero-copy CPU mappings for identical canonical layouts.
- [x] Measure load time, peak host RAM, steady-state model bytes, and storage reads.

Gate: no complete model duplication, no temporary GGUF, and no throughput regression after load.

The registry now keeps each shard mapped for the importer lifetime. Plain BF16,
F32, and raw F8 tensors use the mapped address in one synchronous full-tensor
backend upload. NVFP4, W8A8, AWQ, and GPTQ transforms read the mapped source
directly and allocate only their final canonical tensor image. A full backend
set is intentional: CPU repack and optional CUDA-private upload transforms
require offset zero and the complete tensor. True mapped CPU tensor ownership
would require a source-aware buffer allocator and lifetime handoff before the
current destination-allocation seam; it was investigated and deferred rather
than emulated with an unsafe view. `--no-mmap` is honored: the same registry
falls back to checked, bounded file reads and never exposes a mapped address.
The registry also preserves `--direct-io` through `llama_file`'s aligned direct
read path (including its existing platform fallback), while `--mlock` and
`mmap+mlock` lock final CPU-resident model buffers rather than the temporary
source mapping.

Measured on one RTX A6000, including model load and a three-repeat PP512/TG128
run:

| source (pinned revision) | canonical model bytes | peak host RSS | wall time | PP512 | TG128 |
|---|---:|---:|---:|---:|---:|
| `nytopop/Qwen3-4B.w8a8@57645321` | 5,417,007,104 | 6,080 MiB | 36.21 s | 8,017 t/s | 121.5 t/s |
| `Qwen/Qwen3-4B-AWQ@74d4bd2b` | 3,049,519,104 | 3,690 MiB | 34.87 s | 7,864 t/s | 169.7 t/s |
| `JunHowie/Qwen3-4B-GPTQ-Int4@8153b802` | 3,049,519,104 | 3,693 MiB | 34.05 s | 7,837 t/s | 169.4 t/s |

The measurement command was `llama-bench -m <directory> -ngl 99 -fa on
-p 512 -n 128 -r 3 -o json`, wrapped by `/usr/bin/time -v` for wall time and
peak RSS.

Warm-cache storage reads were zero for the retained W8A8 and GPTQ runs (the
first W8A8 run reported 496 filesystem input blocks). These are not cold-disk
numbers. A parallel row-repack experiment did not change W8A8 wall time and was
removed; the recurring startup cost is dominated downstream of the source-copy
loop. Steady-state graphs contain only canonical ggml tensors and no source
format dispatch.

### Phase 8 — Add further quant formats independently

- [x] Add W8A8 compressed-tensors support.
- [x] Add AWQ support with an explicit decision between native execution and load-time repack.
- [x] Add non-act-order GPTQ support.
- [x] Treat act-order GPTQ as a separate runtime-kernel project unless an exact canonical representation is designed.
- [x] Record CPU, CUDA, HIP, split/offload, and mmap capabilities for each adapter.

Gate: each new format has contract fixtures, projection correctness, end-to-end KLD, backend capability checks, and independent PP/TG measurements.

The first W8A8 adapter accepts only the measured compressed-tensors contract:
signed INT8 channel weights, BF16 channel scales, and dynamic symmetric INT8
token activations. It preserves every INT8 code and folds each BF16 channel
scale into existing Q8_0 blocks. Against an exactly dequantized BF16 control on
the canonical 16K WikiText panel it measured median KLD 0.000106, mean 0.000564,
p99 0.006248, and 99.219% same-top.

AWQ and non-act-order GPTQ are intentionally reference adapters, not the final
competitive executors. Both unpack their producer-specific INT32 layout and
fold group-128 affine parameters into Q4_1. This preserves the 4-bit codes and
works through existing placement and backend machinery, but costs 5.0 bpw and
uses W4A8 arithmetic. The authoritative AWQ packing permutation is
`[0,4,1,5,2,6,3,7]`; GPTQ uses ordinary nibble order and its v1 stored-zero
`+1` convention. GPTQ additionally requires an identity group map. A real
`desc_act=true` configuration or non-identity `g_idx` fails during adapter
validation, before model allocation.

Qwen3.5 recurrent projections add row/column layout transforms. The reference
AWQ/GPTQ repacks cannot apply those transforms after unpacking without also
moving their group parameters, and W8A8/NVFP4 column transforms need a
block-aware permutation. Those combinations are rejected during tensor
description rather than misusing the packed source shape. Production native
executors must make the coupled transform order explicit before lifting this
restriction.

One 512-token WikiText control per format measured the representation cost
against weights dequantized directly from the same source tensors:

| source | median KLD | mean KLD | p99 KLD | same-top |
|---|---:|---:|---:|---:|
| AWQ W4A16-g128 -> Q4_1 | 0.000684 | 0.003999 | 0.041151 | 97.647% |
| GPTQ W4A16-g128 -> Q4_1 | 0.001656 | 0.014044 | 0.218618 | 97.647% |

Those results are not near-zero and must not be described as transparent native
AWQ/GPTQ execution. Production parity still requires a group-128 W4A16 type
and executor (or an integrated maintained executor such as Marlin). The value
of these adapters is a strict loader/reference path and a reusable correctness
oracle for that kernel work.

| adapter output | CPU | CUDA | HIP | layer/tensor placement and CPU offload | direct source mmap |
|---|---|---|---|---|---|
| raw BF16/F32 and channel-scaled F8 | canonical backend support | canonical backend support | canonical backend support | yes | yes |
| NVFP4 | existing NVFP4 type | existing NVFP4 type | existing NVFP4 type | yes | no (repack) |
| W8A8 -> Q8_0 | existing Q8_0 type | existing Q8_0 type | existing Q8_0 type | yes | no (repack) |
| AWQ/GPTQ -> Q4_1 | existing Q4_1 type | existing Q4_1 type | existing Q4_1 type | yes | no (repack) |

The A6000 gates include CUDA execution for all three new adapters and an
all-CPU GPTQ smoke test (2.0 PP t/s, 1.5 TG t/s). A raw channel-FP8 model also
completed a CUDA `load_mode=none` smoke (PP16 248.3 t/s, TG8 111.9 t/s), proving
that disabling mmap takes the bounded-read/materialization path rather than the
direct mapped upload. HIP and multi-device split inherit established
Q8_0/Q4_1 kernels and placement semantics, but still need real-machine
regression runs before release.

### Rejected experiment — packed sibling projections

The A6000 comparison exposed a graph-topology difference rather than a weak
integer kernel. In one PP512 pass, native Q4 Marlin plus Q8 MMQ consumed about
231.5 ms, while vLLM's Marlin families consumed about 239.6 ms. vLLM packs
attention Q/K/V, dense gate/up, recurrent QKV/Z, and recurrent beta/alpha.

A complete dense gate/up proof concatenated the two canonical Q4-A32 tensors
without increasing steady model bytes, built one matmul followed by typed
half-row views, and added a dedicated Marlin BF16 split-SwiGLU epilogue. This
closed the accidental generic-fallback regression, but the resulting
34,816-row Marlin launch was slower than the established pair of 17,408-row
launches on Ampere:

| A6000, PP512/TG128, same binary | PP512 t/s | TG128 t/s |
|---|---:|---:|
| separate gate/up | 1696.7–1702.9 | 32.71–32.79 |
| packed gate/up + fused split-SwiGLU | 1613.3–1626.8 | 32.25–32.77 |

Packing therefore lost about 4.7% prefill with no decode benefit. The code was
removed. Do not generalize vLLM's packed-module topology onto the current
Ampere Marlin executor; reconsider it only alongside a kernel whose tiling was
designed and benchmarked for the wider projections.

### Retained Ampere recurrent-path fusions

Two exact Qwen3.8 recurrent-prefill fusions were retained for the native
group-affine path on an RTX A6000:

- paired Q/K L2 normalization now runs in the FLA BF16 input packer, preserving
  llama.cpp's reduction order and `rsqrtf(fmaxf(sum, eps*eps))` contract while
  avoiding the intermediate F32 tensors; and
- a Q4-A32 recurrent projection can leave its already-rounded BF16 output in
  the graph destination for the split convolution to consume directly, rather
  than widening it to F32 only for the convolution to read it back as BF16.

The first changes the measured kernels from 1.309 ms of L2 normalization plus
2.242 ms of ordinary packing to 2.383 ms of combined normalization and packing
per PP512 pass. The second removes 48 BF16-to-F32 output conversions and changes
the 48 recurrent convolution launches from 3.632 ms in F32 to 2.299 ms in BF16.
Together, the final five-sample A6000 measurements were 1704.16 PP2048 t/s and
32.66 TG128 t/s. The control, L2-fused, and convolution-deferred 121 MiB logits
files were byte-identical, with SHA-256
`8a2ea9c88341c96c593821f2c9c920844f95e5eb48c8d7aaead5e20ccc181833`.

The residual/RMS epilogue now also omits its F32 normalized output when every
consumer is a supported Q4-A32 Marlin projection and can therefore reuse the
epilogue's persistent BF16 activation. Two order-balanced PP2048 pairs measured
1708.99 versus 1705.75 t/s and 1704.56 versus 1700.54 t/s, a combined gain of
about 0.22%. TG128 remained flat at 32.64 t/s. The resulting logits file was
again byte-identical to the same SHA-256 baseline. The consumer scan is strict:
an output tensor, a non-matmul use, or an unsupported projection preserves the
ordinary F32 materialization.

A larger GDN epilogue experiment also fused the following RMS normalization,
SiLU gate, and multiply. It removed 48 SiLU launches and improved PP2048 from
1703.45 to 1711.49 t/s, but making the independent gate projection available
early changed decode graph ordering and reduced TG128 from 32.66 to 32.36 t/s.
That trade was rejected and its code removed.

Packing the recurrent Q/K/V and Z projections into the same 16,384-row Marlin
launch was also tested because vLLM represents those checkpoint tensors as one
merged projection. The combined Marlin kernel took essentially the same time
as the two original kernels, while splitting its output back into the existing
Q/K/V and Z graph tensors cost about 3 ms per PP512 pass. Steady PP2048 was
1701.06 t/s, a tie/slight loss against 1700.5--1705.8 t/s controls, and the
proof required about 1.9 GiB of duplicate packed weights. The implementation
was removed. A production packed loader would eliminate the duplicate weights,
but not the measured output-split cost, so it is not justified for the current
Ampere Marlin executor.

The two small recurrent beta/alpha BF16 projections share one F32 input. The
ordinary cuBLAS path converted that input independently for both projections.
Reusing the existing per-stream BF16 image for the second projection removes
54 conversion launches per PP512 pass while leaving both GEMMs unchanged. A
stable PP2048 A6000 pair measured 1718.94 versus 1702.50 t/s (+0.97%); TG128
was unchanged. The 121 MiB logits file was byte-identical to the baseline SHA
above. The retained matcher is deliberately narrow (`[48,5120]` BF16 weights,
F32 input, prefill only), because those two consumers do not mutate their
shared source between reads.

A generic version was also measured at 1720.53 versus 1697.40 PP2048 t/s, but
changed the logits file. Tensor identity alone is not a valid general cache
key: graph storage may be updated in place between consumers. The broad form
was rejected as a correctness bug rather than accepted as an additional small
performance gain.

Two smaller prefill launch fusions follow vLLM's fused Qwen GDN preparation
without changing either projection or the embedded FLA math. First, the two
ordinary beta/alpha cuBLAS projections now feed one pointwise epilogue for
add/softplus/multiply and sigmoid. Two order-balanced pairs measured a combined
1721.13 versus 1718.02 PP2048 t/s (+0.18%). Batch-one decode retains its older
specialized dot-and-gate kernel. Second, the existing Q/K/V L2+BF16 pack launch
also copies the already-computed gate, beta, and recurrent state into FLA head
order; its V grid already spans all of those elements. This removes the
separate head-pack launch and measured a combined 1716.28 versus 1713.26 t/s
(+0.18%) in two order-balanced pairs. Both changes are Ampere GDN-prefill-only;
their stacked 121 MiB logits file was byte-identical to the same reference SHA.
The retained stack measured a steady PP512 median of 1728.95 t/s, a PP2048
median of 1704.58 t/s (1706.03 mean), and a TG128 median of 32.75 t/s on the
A6000.

Two further exact handoff fusions did not earn retention. Copying the recurrent
state into graph order inside the following RMS epilogue removed 48 launches
and about 0.54 ms of profiled work, but two order-balanced PP2048 pairs were a
tie: their combined means differed by less than 0.2%, and the combined median
favored the original schedule. Deferring the recurrent Z/gate projection's
already-rounded BF16 output directly into the SiLU-and-multiply consumer also
removed 48 widening launches, but lost 0.28% in both combined mean and median
(1709.97 versus 1714.84 t/s mean). Both implementations were removed.

### Retained W8 MMQ activation reuse

The mixed INT4/INT8 Qwen3.8 checkpoint has 88 W8A8 projections per PP pass.
The ordinary MMQ entry quantized its F32 activation independently for every
projection, even when dense gate/up or attention Q/K/V consume the same graph
tensor. Two exact CUDA changes remove that duplicated work:

- adjacent same-shape Q8_0_G128 projections use the existing paired-MMQ
  executor, extended to ordinary dense matmuls, so gate/up share one Q8_1
  activation image; and
- a graph proof recognizes exactly three direct Q8 matmul consumers of one
  activation, rejects any other consumer or unsafe output-range overlap, and
  lets Q/K/V reuse one grow-only per-stream image. The largest observed image
  is about 2.8 MiB. Ambiguous graphs and non-CUDA backends fall back unchanged.

Together these reduce `quantize_mmq_q8_1` from 88 launches / 2.38 ms to 48
launches / 1.46 ms per PP512 pass. Order-balanced PP2048 gates measured
paired-MMQ at 1714.33 versus 1707.34 t/s (+0.41%), then Q/K/V reuse at 1713.92
versus 1707.59 t/s (+0.37%) with pairing enabled in both arms. Batch-one uses
MMVQ and declines both paths. The combined logits file remained byte-identical
to SHA-256 `8a2ea9c88341c96c593821f2c9c920844f95e5eb48c8d7aaead5e20ccc181833`.

## 6. Test matrix

### Structural tests

- Canonical tensor request set comes solely from the existing model loader.
- Every requested tensor binds exactly once.
- Every required source quant weight has all required auxiliaries.
- No architecture adapter duplicates a complete target manifest.
- GGUF construction and loading do not enter safetensors code.

### Correctness tests

- Registry bounds, gaps, overlaps, duplicate names, dtype sizes, and shard-index validation.
- Quant rule declaration order and anchored regex behavior.
- Per-format scale shape/dtype/value checks.
- Architecture transform fixtures with known row/column indices.
- Direct projection comparison for all retained kernel shapes.
- Greedy coherent-generation smoke tests.
- Reference-vs-reference anchor exactly zero.
- Safetensors-versus-reference KLD near zero with the tolerance and reference named explicitly.
- Full KLD/hazard testing whenever a numerical representation or kernel changes.

### Placement and backend tests

- Single CUDA GPU.
- CUDA layer split and tensor split.
- Partial CPU offload.
- CPU-only or a precise early rejection for unsupported formats.
- HIP/RDNA or a precise early rejection for unsupported formats.
- Model load cancellation and teardown.
- Optional MTP and multimodal sidecars where the architecture supports them.

### Performance tests

- Load time and peak host staging memory.
- Peak VRAM after load.
- PP at representative batch sizes.
- TG at short and deep context.
- Kernel-level comparison against the corresponding GGUF/runtime type.
- External-engine comparison only under equivalent model, quant contract, device count, batching, graph mode, and sampling behavior.

## 7. Review checkpoints

Run a focused design/code review after Phases 2, 3, and 6.

Questions reviewers must answer:

1. Is tensor topology declared in more than one place?
2. Can a quant adapter be reused by another architecture without including Qwen code?
3. Can an architecture adapter accidentally alter backend selection after the tensor becomes canonical?
4. Are all correctness-critical scale and permutation contracts mandatory?
5. Does any fallback silently change numerical format or placement?
6. Does the GGUF hot path perform new virtual dispatch, mapping, or source-format checks?
7. Is the source/importer lifetime valid until the final upload and all asynchronous copies complete?
8. Can cancellation or an exception leave partially registered/repacked tensor state behind?

## 8. Recommended next order

The architecture bridge, bounded loader, second-model proof, and first format
adapters have working implementations. Before describing the branch as
release-ready:

1. run the remaining real-machine HIP and multi-device placement gates;
2. perform the planned simplification and correctness review over the complete
   source/importer/adapter diff;
3. freeze reproducible commands and baselines for the retained NVFP4 and FP8
   executors;
4. implement native group-128 W4A16 execution before presenting AWQ or GPTQ as
   performance-competitive support;
5. add act-order GPTQ only with an exact activation-gather/runtime contract.

## 9. Definition of done

This architecture project is complete when:

- Qwen3.5 safetensors no longer maintains a parallel tensor manifest;
- format-generic quant code is absent from the Qwen adapter;
- at least two existing llama.cpp architectures load native safetensors through their original model classes;
- equivalent GGUF and safetensors tensors reach the same backend kernels;
- unsupported format/backend combinations fail before large allocations;
- no temporary GGUF or whole-model duplicate is created;
- correctness, KLD, placement, load-memory, PP, and TG gates pass; and
- adding a third already-supported architecture demonstrably requires only metadata/name/layout policy unless it introduces a genuinely new operation or quantization format.
