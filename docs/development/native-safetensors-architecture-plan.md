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

Each phase should be independently reviewable and should preserve the runnable
Qwen3.6 path. The initial release targets are Qwen3.8-27B plus the large-MoE
Qwen3.8-Flash-Next (`qwen4exp`) and DeepSeek-V4-Flash architectures.

### Phase 0 — Freeze trustworthy baselines

- [ ] Record exact revisions for the representative Qwen3.8-27B release set:
  official block-FP8, Unsloth NVFP4, AWQ INT4, GPTQ INT4, AutoRound W4A16,
  W8A16 INT8, SmoothQuant W8A8, and BitsAndBytes NF4.
- [x] Pin the official Qwen3.8-Flash-Next and DeepSeek-V4-Flash safetensors
  revisions. Inventory their complete tensor contracts, including Qwen PLE
  shards and DeepSeek's mixed FP4/FP8 expert representation, before describing
  either source format as supported.
- [ ] Record PP, TG, peak VRAM, peak host memory, load time, and first-token
  correctness on the reference hardware used for each external-engine comparison.
- [ ] Preserve direct projection correctness tests for every supported shape.
- [ ] Preserve same-process reference-vs-reference anchor tests; the anchor must be exactly zero.
- [ ] Disable or reject experimental backend paths that are known to produce incorrect output.
- [ ] Save vLLM commands, versions, and corresponding baselines.

Gate: every representative Phase-0 source loads and produces coherent greedy
output, and the recorded performance is reproducible within the chosen
thermal/clock tolerance.

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

AWQ and identity-map GPTQ currently unpack their producer-specific INT32 layout
and fold affine parameters into Q4_1. This preserves the 4-bit codes and works
through existing placement and backend machinery, but costs 5.0 bpw and uses
W4A8 arithmetic. The authoritative AWQ packing permutation is
`[0,4,1,5,2,6,3,7]`; GPTQ uses ordinary nibble order and its v1 stored-zero
`+1` convention.

Act-order GPTQ is a distinct native representation: it keeps packed I4 weights,
unpacked group zero codes, original F16/BF16 scales, and a compact per-column
`g_idx`. CPU and CUDA reference executors index the arbitrary group map during
the dot product. On the real
`chieunq/Qwen3-1.7B-gptq-v2-4bit-gsm8k2048` checkpoint, the prompt `The capital
of France is` matched GPTQModel's argmax and all top-10 tokens with last-logit
KLD 0.000679. The scrambled group map is therefore exercised end to end rather
than inferred from a synthetic identity fixture.

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
| AWQ/identity-GPTQ -> Q4_1 | existing Q4_1 type | existing Q4_1 type | existing Q4_1 type | yes | no (repack) |
| GPTQ act-order | scalar reference | scalar reference | explicit reject | CPU/CUDA placement | yes |
| BitsAndBytes NF4/FP4 | scalar reference | scalar reference | explicit reject | CPU/CUDA placement | yes |
| BitsAndBytes INT8 | channel-INT8 fallback | channel-INT8 executor | explicit reject | CPU/CUDA placement | yes |

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

A GDN epilogue also folds the following RMS normalization, SiLU gate, and
multiply into the BF16 output unpack. The raw gate projection is made available
early only when each sequence contains more than one token, so decode retains
its original graph order. This removes 48 unary-gated launches per PP512 pass;
the fused unpack costs 2.28 ms instead of 1.57 ms, while removing about 2.78 ms
of unary work. Two order-balanced PP2048 pairs on the retained stack measured
1724.72 versus 1718.46 t/s (+0.36%). The earlier decode regression therefore
does not apply to this prefill-only form. Its 121 MiB logits file was
byte-identical to the same reference SHA-256.

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

## 6. Broad quantization coverage

Native loading is not considered support unless the source representation,
every correctness-critical auxiliary tensor, the graph operation, and the
selected backend agree on the same numerical contract. Metadata recognition
alone is not support. Unsupported combinations must fail before model buffers
are allocated rather than silently converting to a different quantizer.

The implementation order is deliberately shared-contract-first:

| Family | Source contract | Current state | Next proof |
|---|---|---|---|
| BF16/F16/F32 | plain tensors | direct upload | second-architecture and placement matrix |
| NVFP4 | packed E2M1, group-16 E4M3 scales, two global scales | native importer and CUDA executor | freeze KLD/perf baselines; HIP is an explicit reject |
| FP8 W8A8 dynamic | E4M3 weights, channel or 128x128 weight scales, dynamic token activations | native channel/block paths; compressed-tensors `naive-quantized`, bounded FBGEMM FP8, and Quark PTPC schemas recognized | retain exact backend capability gates and add HIP implementations |
| Grouped FP8 W8A8 | E4M3 weights and BF16 scales per 32 values, dynamic group-32 activations | native importer plus scalar CPU/CUDA reference execution | real-model KLD and optimized CUDA kernel |
| FP8 W8A8 static | E4M3 weights, tensor weight scale, tensor input scale | native CUDA scalar-scale path; real ModelOpt mixed checkpoint executes | freeze KLD against its framework reference; reject unsupported backends early |
| INT8 W8A8 dynamic | I8 weights, channel scale, dynamic token activations | native CUDA executor; compact/current compressed-tensors schemas and real SmoothQuant+GPTQ and Quark checkpoints execute | complete CPU/HIP/partial-placement policy |
| INT8 W8A8 static | I8 weights, tensor/channel weight scale, tensor input scale, optional activation zero point | symmetric and asymmetric CUDA paths; real asymmetric checkpoint executes | add a public full-model fixture and placement matrix |
| EETQ W8A16 | transposed I8 weights and one FP16 scale per output channel | exact load-time transpose into Q8_0; preserves every code and scale with 6.25% block-scale overhead | real-model logits proof and native weight-only CUDA fast path if demand justifies it |
| Quanto W8A16 | I8 or E4M3 weights and one BF16/F16 scale per output channel (`qint8`, `qfloat8_e4m3fn`) | native raw-code loading with a self-describing scale sidecar; CPU reference and CUDA execution; real qint8/qfloat8 full-model logits proven | add HIP execution and large-model performance gates |
| Quanto W4A16 | packed qint4 data plus per-group scale and shift | load-time Q4_1 repack and CPU/CUDA execution | replace the correctness-first repack with an exact native executor before claiming parity with Quanto |
| AWQ/GPTQ W4A16 | packed I4 plus group scale/zero | exact Q4_1 repack for AWQ and identity-map GPTQ at group sizes divisible by 32, including per-channel; F16/BF16 scales; AutoRound GPTQ sidecar configs and per-module precision exceptions | widen real-model coverage for g32/g64 and architecture transforms |
| W4A8 INT8 | group-128 I4 weights and dynamic/static INT8 activations | exact Q4-A32 repack plus scaled CUDA executor | full-model KLD/perf and unsupported-backend gates |
| W4A8 FP8 | group-128 I4 weights and dynamic FP8 activations | exact repack plus CUDA executor | Hopper/Blackwell real-model KLD/perf |
| MXFP4/MXFP8 | microscaled weights/activations and E8M0 sidecars | native compressed-tensors importer and CUDA execution; Quark MXFP4 schema and real checkpoint proven | real-model KLD/perf on each supported architecture |
| AutoRound/INC MXFP4/MXFP8 | the same group-32 MX values and E8M0 scales, declared by the compact AutoRound schema | maps onto the existing MXFP contracts, including exact per-module FP16 exceptions and the optional `quantization_config.json` sidecar | real small-model full-load proof; current public LLM checkpoints are very large MoEs |
| TorchAO tiled INT4 / unpacked INT8 | `Int4TilePackedTo4dTensor` and `IntxUnpackedToInt8Tensor` safetensors subclasses | exact load-time repack into Q4_1/Q8_0; all tiled lanes and INT8 rows covered | native execution formats if load time or repack precision becomes material |
| Transformers HQQ INT4 | flattened axis-1 `4bit_u8` codes with learned F16 scale/zero per group | exact code-preserving Q4_1 repack for unquantized metadata at group sizes divisible by 32; public Llama-3.2-1B checkpoint loads and generates coherently | external-reference KLD and a native metadata-preserving executor |
| ExecuTorch HQQ experts | packed signed INT4 expert rows plus one BF16 scale per 128 values | exact nibble-preserving Q4_0 repack for the public Qwen3.5-MoE export; every expert/row/lane covered | external-reference KLD and a native scale-preserving executor |
| WNA16 | packed 2--8-bit weights, group scales/zeros | exact 4-bit group-32 and 8-bit group-128 subsets cover the current mainstream releases | defer native 2/3/5/6-bit layouts until a real checkpoint and user demand justify them |
| BitsAndBytes | NF4/FP4 codes with block absmax and optional nested scale quantization; INT8 rows with SCB absmax | native packed 4-bit types and exact nested-scale bundle plus serialized INT8/SCB loading; real NF4 and INT8 models load and generate coherently | optimize 4-bit CUDA, add HIP, and add LLM.int8 outlier decomposition where its threshold materially changes fidelity |
| GPTQ act-order | packed I4 plus non-identity `g_idx` | native packed type and compact scale/zero/group bundle; real-model CPU/CUDA reference execution and external KLD proof | optimize CUDA and add HIP; transformed Qwen3.5 recurrent projections reject early |
| ModelOpt W4A4 NVFP4 | NVFP4 weights and NVFP4 activations | explicitly rejected; W4A16 and mixed W4A16/FP8 are supported | Blackwell activation-quantization executor and real-model fidelity/perf |

Producer names are not treated as numerical formats. Quark W4A16, static FP8,
PTPC FP8, INT8, NVFP4, and MXFP4 all map onto the same canonical contracts as
their compressed-tensors/ModelOpt counterparts. FBGEMM's
`activation_scale_ub` is preserved in the dynamic-FP8 marker and applied before
the row scale is chosen. SmoothQuant checkpoints need no special runtime
operator when smoothing has already been folded into the stored weights; their
remaining INT8 or FP8 tensor contract is handled normally.

That distinction is now backed by a public FP8 checkpoint rather than config
inspection alone. `MLliu6/Qwen3-VL-4B-Instruct-SmoothQuant-W8A8-FP8` loaded
directly, selected the shared channel-FP8 path, and generated a coherent Paris
answer. The smoothing operation had already been folded by the producer; no
runtime smoothing tensor or extra graph multiply was present.

The Quanto `qint8` proof is at the independently dequantized BF16 control
floor: last-token KLD 0.00039238 against the official Quanto execution versus
0.00039069 for the BF16 control, with the same argmax. Quanto `qfloat8_e4m3fn`
measures 0.00020709 against official Quanto on the shared token sequence; the
plain BF16 engine/reference floor is 0.00016884. The qint4 Q4_1 repack improves
substantially over an ordinary Q4_1 interpretation (0.00045605 versus
0.00113866 KLD), but remains above its dequantized-BF16 control floor
(0.00012797), so it is usable coverage rather than exact format parity.

Still outside the release coverage are genuinely different numerical/layout
families rather than config aliases: ModelOpt W4A4 NVFP4 activation execution,
EXL2/EXL3, AQLM, SpQR, VPTQ, HIGGS, SINQ, HQQ 1/2/3/8-bit, axis-0,
float-view, or metadata-quantized variants, Quanto qint2, sparse TorchAO
layouts, and native 2/3/5/6-bit WNA16. They must not be advertised as
supported or silently requantized.

MLX is also intentionally outside this project phase. Although MLX checkpoints
use safetensors as a container, their tensor layouts target the separate MLX and
Metal execution ecosystem. This fork does not yet have Metal implementations of
its KV codec families, so MLX import would not provide a coherent supported
backend. Revisit it only as part of a broader Metal project.

### 6.1 Release scope informed by Qwen3.8-27B

A 2026-09-01 Hugging Face inventory found 933 repositories attached as
quantizations of `Qwen/Qwen3.8-27B`. Name/tag counts overlap, but their relative
prevalence is useful: 342 GGUF, 344 MLX, 110 NVFP4, 76 FP8, 56 GPTQ, 42 AWQ,
41 AutoRound, 24 EXL3, four BitsAndBytes, and one direct SmoothQuant base-model
release. No HQQ or Quanto release for this model surfaced in the same search.

The native-safetensors release boundary is therefore:

1. official block-FP8;
2. mixed NVFP4/FP8/BF16 checkpoints;
3. AWQ, GPTQ, and AutoRound W4A16;
4. W8A16 INT8;
5. SmoothQuant W8A8; and
6. BitsAndBytes NF4.

WNA16 is an execution umbrella, not a checkpoint format that is commonly named
as such. Qwen3.8 has ordinary W4A16 and W8A16 releases, which are in scope, but
only isolated INT5/INT6 AutoRound experiments and effectively no ordinary
W2A16/W3A16 publication ecosystem. Native 2/3/5/6-bit WNA16 is consequently a
demand-driven post-release project, not a release requirement.

EXL3 is a real but separate ecosystem. Its trellis/codebook representation needs
a new executor rather than another metadata alias, so its presence does not
expand the initial native-safetensors release boundary.

Generic ModelOpt W4A4 NVFP4 activation execution remains on the near-term
post-release roadmap. DeepSeek-V4-Flash is a separate release target whose
official configuration declares FP4 experts inside a block-FP8 model. Its exact
stored expert layout and activation contract must be inventoried from the shard
headers and execution reference. If that contract is not already represented
canonically, the narrowly required DeepSeek executor becomes a release blocker;
it must not be assumed to be identical to ModelOpt W4A4 from metadata alone.

The public `SocialLocalMobile/Qwen3.5-35B-A3B-HQQ-INT4` checkpoint provides a
different, fully serialized path: tiled TorchAO INT4 dense projections,
TorchAO unpacked-INT8 embeddings, and scale-only HQQ routed experts under the
flat ExecuTorch Qwen3.5-MoE namespace. The native loader now maps that namespace,
reuses the existing runtime architecture, supports its fused full-attention QKV
projection, and generates coherently. This is not a claim of generic HQQ
support: it is separate from the now-supported Transformers `4bit_u8`/axis-1
contract, and other HQQ bit widths and metadata modes remain distinct.

The generic HQQ proof uses the public
`nm-testing/Llama-3.2-1B-Instruct-HQQ` checkpoint. Its 112 projections retain
the producer's code lanes and learned affine parameters during load-time Q4_1
repacking; with F16 KV (the model's 64-dimensional heads do not satisfy this
fork's default Turbo4 KV block contract), it generated `The capital of France
is Paris.` at 2075.4 prompt and 393.2 generation tokens/s on the A100 test box.
An all-CPU load and generation arm also passed, confirming that the portable
Q4_1 materialization is not CUDA-placement-only.

Architecture coverage is a separate axis. Qwen2, Qwen3, Qwen3.5, Qwen3-VL,
Llama, and classic Mistral currently exercise the generic importer seam. TinyLlama
with a serialized BitsAndBytes INT8 checkpoint provides the Llama-family proof:
154 quantized matrices were recognized, the full model loaded without a whole-
model conversion, and greedy generation returned a coherent Paris answer.
For Mistral-7B-Instruct-v0.3, all 291 native tensors (14,496,579,584 bytes)
matched a fresh BF16 GGUF conversion exactly. Native safetensors and GGUF now
produce bit-identical CUDA logits; against the Transformers BF16 reference the
last-token KLD is 0.0000894 (the CPU GGUF control is 0.0000938). This proof also
caught and closed an unrelated CUDA integration bug: the Qwen-specific hidden-
BF16 GLU handoff had been admitted for Mistral's 14,336-wide FFN even though its
producer contract is the Qwen 17,408-wide path.

### 6.2 Initial large-MoE architecture boundary

The initial architecture boundary includes both current flagship large MoEs:

1. `Qwen/Qwen3.8-Flash-Next`, whose source identifies as `qwen4_exp`; and
2. `deepseek-ai/DeepSeek-V4-Flash`, whose source identifies as `deepseek_v4`.

Both ordinary runtime architectures already exist on current master. Qwen4 is
`LLM_ARCH_QWEN4EXP` with `src/models/qwen4exp.cpp`; it is not the older
`LLM_ARCH_QWEN3NEXT` path. DeepSeek uses `LLM_ARCH_DEEPSEEK4` and
`src/models/deepseek4.cpp`. Native safetensors adapters must therefore translate
source metadata, names, auxiliary quant tensors, and the few required layout
transforms into those existing model classes. They must not duplicate QSA,
PLE, hyper-connection, compressed-attention, MTP, VBR, or multimodal graph
logic inside the importer.

The existing Qwen4 converter is the mapping oracle. In particular, its PLE
hash constants remain exact integers, its 128 n-gram embedding shards remain
bounded/chunked rather than concatenated into a second whole-model allocation,
its combined indexer Q/K projection is split into the canonical runtime tensors,
and its Qwen3-VL vision tower continues through the established vision model.
The direct importer must express the same contracts through bounded source
bindings instead of producing an intermediate GGUF.

DeepSeek-V4 already requests routed expert weights as canonical 3-D
`ffn_{gate,down,up}_exps` tensors. That representation is also a runtime
offload contract: `--cpu-moe` and `--n-cpu-moe` place those tensors in host
memory, while the adaptive cache discovers host `WEIGHTS` buffers by canonical
`_exps`/`_chexps` names, three-dimensional shape, and backend-supported expert
stride. The safetensors path must preserve those allocation roots, names,
dimensions, buffer usage, and per-expert addressability. An opaque importer-
owned packed aggregate that happens to execute on one GPU is not acceptable.

Release proof for each large MoE requires:

- bounded direct load without a temporary GGUF or a second resident model;
- coherent generation and reference KLD for the exact source quant contract;
- layer split and tensor split across multiple GPUs;
- partial expert placement with `--cpu-moe` and `--n-cpu-moe`;
- `--moe-cache auto`, `on`, `soft`, explicit budget, and multi-GPU expert-
  parallel cache paths, including cold/warm correctness and stable residency;
- agreement between fit-time placement/accounting and the final loaded buffers;
- MTP, VBR, long-context, and teardown gates; and
- Qwen4 text and multimodal/PLE paths, including bounded loading of the large
  n-gram table.

Current architecture proof (2026-09-01):

- The source contracts are pinned at Qwen revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540` and DeepSeek revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`.
- The official 149 GiB DeepSeek checkpoint loads directly with all 43 trunk
  layers assigned to CUDA, while its 129 trunk routed-expert roots stay in
  `CUDA_Host` under `-ot 'exps=CPU'`. The final buffers are 7418.32 MiB on CUDA
  and 141362.00 MiB in pinned host storage. Greedy chat generation completed
  coherently (`The capital of France is` -> `Paris.`) through the existing
  DeepSeek graph. This end-to-end proof caught a source-boundary bug in
  `load_arch_hparams()`: its MTP-presence probe used GGUF-only `get_weight()`,
  reset `n_layer_nextn` to zero for a valid direct source, and executed the MTP
  block as a 44th trunk layer. The source-aware `has_tensor()` probe now
  preserves `n_layer=43`, `n_layer_all=44`; the same reusable existence check
  replaces GGUF-only MTP/trunk probes in the other MTP-capable model classes.
  Disabling the optimized block-FP8
  executor did not repair the pre-fix output, which independently excluded that
  kernel as the cause. Parallel mmap-backed expert repacking reduces end-to-end
  load time from roughly 10--11 minutes to about 4.5 minutes on the 40 GiB A100
  proof host.
- The same source passed an initial adaptive-cache gate with
  `--moe-cache auto -ot 'exps=CPU'`: a cold France request and a subsequent
  Germany request both generated correct answers, at 5.60 and 5.74 t/s
  respectively. GPU residency rose from roughly 7.8 GiB without the cache to
  9.34 GiB after the two requests, confirming that the native canonical expert
  roots participate in runtime promotion rather than merely falling through to
  host execution. The wider cache mode, budget, eviction, and multi-GPU matrix
  remains pending.
- Native MTP now passes initialization and generation against the same source.
  Its first production gate generated the correct `Paris.` answer at 8.00 t/s,
  drafting 48 tokens and accepting 26. The gate exposed a backend-contract seam
  in the MTP `e_proj+h_proj` projection: DeepSeek's hyper-connection streams
  presented a 3-D activation to the otherwise valid block-FP8 2-D projection.
  Flattening the independent stream/token axes for that matmul and restoring
  the hidden layout afterward preserves the broadcasted numerical operation
  while making the existing scaled-FP8 executor applicable. The same shape fix
  is applied to Qwen4's equivalent hyper-connection MTP projection.
- A native-source VBR invocation also initializes and generates coherently
  (`Germany` -> `Berlin`), but it does not constitute a dynamic-tier proof.
  The existing DeepSeek-V4 cache explicitly pins its concatenated raw/CSA/HCA
  children to static `q8_0` under `-ctk/-ctv vbr`; runtime logging names the
  missing contract as per-child tier ganging. Native loading reaches the same
  documented fallback as GGUF. Dynamic DeepSeek-V4 VBR remains a runtime-cache
  project rather than an importer defect or completed release gate.
- Partial dense-layer offload also crosses the native source/backend seam
  successfully. With `-ngl 20 -ot 'exps=CPU'`, 20/45 layers occupied a
  3700.14 MiB CUDA model buffer and the remaining weights occupied
  145080.18 MiB of pinned host storage. The mixed CUDA/CPU graph initialized,
  executed prompt and decode tokens through the block-FP8 CPU path, returned
  finite coherent reasoning text, and tore down cleanly. At roughly 0.21 t/s
  this is a placement-correctness proof, not a useful serving configuration.
- The official DeepSeek source now passes both four-GPU placement modes on RTX
  3090s with routed experts held in host memory. Layer split placed 1765.35,
  1653.95, 1633.68, and 2365.33 MiB of model data on the four devices. Tensor
  split held 3547.37 MiB in the meta model buffer and used 4387 MiB per device
  after context/compute allocation. Both generated the exact one-word answer
  `Paris`; the short decode probes measured 3.49 and 5.39 t/s respectively.
  The tensor-split gate found two shared runtime defects rather than an importer
  byte error: source-less `ARANGE` was not classified as mirrored by the meta
  backend, and DeepSeek's block-FP8 Q-B, grouped output, and shared-expert scale
  grids were mirrored instead of sharded along their weights' block axes. The
  focused synthetic DeepSeek architecture test now passes the complete CUDA,
  CPU, and meta-device graph with NMSE `8.71e-08` on CUDA/meta versus CPU.
  One non-fatal checkpoint-publication warning remains on the production tensor-
  split request; checkpoint persistence under tensor split is not claimed by
  this gate.
- Four-GPU adaptive expert caching also consumes the native MXFP4 expert roots
  directly. `auto` selected three-way expert parallelism and allocated 18.09,
  20.46, 20.48, and 19.75 GiB pools. Five cold/warm requests generated correct
  answers or coherent code; repeated code decode rose from 8.11 to 9.25 t/s.
  Clean teardown reported 86.3--87.4% residency-probe hits, 16,521 successful
  fills, and zero fill, dispatch, or collection failures. Explicit `on` also
  passed with two concurrently active slots: both slots completed their short
  reasoning probes at 5.67 t/s each, with 5,296 successful fills and zero fill,
  dispatch, or collection failures. A 4096 MiB/device fixed-budget arm filled
  all four 962-slot pools, produced the same coherent quicksort twice, warmed
  decode from 6.37 to 6.93 t/s, and performed 30 replacements without a cache
  failure. This proves `auto`, `on`, expert-parallel operation, fixed budgets,
  replacement, and multi-slot execution; cache-aware `soft` placement remains
  a separate fit gate.
- The same regression pass restored the legacy `llama_model_init_from_user()`
  callback contract. Callback-backed models synthesize the canonical tensor
  requests because their GGUF metadata is not a complete tensor manifest;
  strict describe/bind/load completeness remains exclusive to real tensor
  sources. This also restores compilation and execution of the architecture
  fixture after the tensor-source interface addition.
- The text-only Qwen4 fixture at revision
  `d49cb5156b0dad4016f33b55e142a390da88e9fa` exercises recurrent attention,
  QSA/indexer splitting, PLE, hyper-connections, separate-expert aggregation,
  and the existing Qwen4 CUDA graph. Its native source path and an independent
  F32 GGUF conversion produced the same 16-token greedy continuation. This
  proof found and fixed a reversed grouped-to-tiled V-head permutation, a
  GGUF-only PLE shape probe, two production PLE dimension assumptions, and
  incorrect canonical indexer suffixes.
- The official Qwen4 FP8 source at revision
  `236dfdf285828023ca3bcd3f37366c58a3469b13` now passes a full four-GPU
  layer-split load with its routed experts and 128-shard PLE table in host
  memory. The source contains 75,264 block-FP8 tensors. Its PLE embedding is a
  raw F8 table plus one mandatory parent scale, so the native path keeps the
  51-GiB table in F8, gathers directly, and applies the scale afterward instead
  of expanding it to F32. Mapped shard upload reduced end-to-end load time from
  11:41 to 10:28 on the proof host. Pure Qwen4 row/column permutations retain
  BF16, while numerical transforms and the CUDA SSM convolution kernel retain
  their required F32 contract; focused fixtures pin all three cases. Final
  model buffers were 1945.82, 1735.39, 1883.13, and 2664.95 MiB on CUDA plus
  167040.86 MiB in pinned host memory. With the cache disabled, deterministic
  probes returned `Paris` and `Berlin` at 16.2--16.5 prompt t/s and 8.3--8.7
  decode t/s. Teardown released all device and host allocations; it also logged
  a non-fatal pre-existing compute-buffer expectation mismatch, so accounting
  silence is not claimed by this gate.
- Qwen4's checkpoint stores each routed expert as a separate 128x128-block-FP8
  matrix, while the current portable `MUL_MAT_ID`, CPU-offload, and MoE-cache
  contract requires one canonical 3-D allocation root without a block-scale
  sidecar. The bounded bridge therefore converts each expert independently to
  `Q8_0_G128`: 8.125 bpw versus 8.001 source bpw (+1.55%), with no whole-model
  duplicate. A two-expert fixture verifies independent values and scale-grid
  orientation. This is a portability bridge, not a fidelity result: the added
  requantization requires reference KLD before production acceptance, and an
  exact block-FP8 `MUL_MAT_ID` path remains a possible later optimization.
- The canonical `Q8_0_G128` expert bridge now participates in adaptive caching.
  The cache admission allowlist and both its dedicated and fused dispatch
  switches accept the type; the underlying MMVQ type traits and dot product
  were already shared with ordinary execution. On the official Qwen source,
  `--moe-cache 4096 --moe-cache-expert-parallel auto` resolved on and allocated
  2,579 slots on each of four RTX 3090s. The first 64-token quicksort request
  generated coherent code at 14.95 t/s; an identical warm request produced the
  same bytes at 23.01 t/s. Eight repeated merge-sort probes then forced 96--135
  replacements per device while sustaining 22.44--23.45 t/s. All four devices
  reported zero fill, dispatch, and collection failures.
- Tensor split exposed two independent Meta-backend enumeration gaps in the
  cache integration. Admission inspected only the top-level Meta device instead
  of its physical children, and session creation accepted only top-level CUDA
  backends even though the scheduler supplies one Meta backend. Both sites now
  use the public Meta child-enumeration APIs and deduplicate physical devices.
  On the official Qwen source, `-sm tensor -ts 1,1,1,1 -np 2` resolved the cache
  on, created one 4,092-MiB/2,579-slot `Q8_0_G128` pool on each RTX 3090, and
  served two simultaneous slots coherently. A warm concurrent probe returned
  `Paris` and a correct factorial implementation while all devices performed
  replacements; fill, dispatch, and collection failures remained zero. This
  closes the tensor-split and multi-slot cache execution gate, not a throughput
  comparison against layer split.
- The full production Qwen multimodal path, cache-aware `soft` placement,
  MTP/VBR, long-context, and KLD gates remain required. The tiny fixture remains
  an architecture/transform proof rather than a quality model.

## 7. Test matrix

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
- Large-MoE `--cpu-moe`, `--n-cpu-moe`, and every retained `--moe-cache`
  policy, including expert-parallel caching on multiple GPUs.
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

## 8. Review checkpoints

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

## 9. Recommended next order

The architecture bridge, bounded loader, second-model proof, and first format
adapters have working implementations. Before describing the branch as
release-ready:

1. finish the current simplification/correctness pass, remove abandoned
   experiments, and split the working tree into reviewable commits;
2. add source adapters for the existing `LLM_ARCH_QWEN4EXP` and
   `LLM_ARCH_DEEPSEEK4` model classes, using their converters only as mapping
   oracles and keeping all graph/runtime policy in the model implementations;
3. run the representative Qwen3.8-27B acceptance matrix from Phase 0, including
   exact-zero anchors, reference KLD, coherent generation, resident VRAM, peak
   host memory, PP, and TG;
4. run the Qwen4/DeepSeek large-MoE matrix from section 6.2 before calling the
   architecture boundary complete;
5. run the real-machine placement matrix for every retained family: single CUDA,
   layer split, tensor split, partial CPU offload, CPU-only or precise early
   rejection, HIP or precise early rejection, cancellation, and teardown;
6. close the measured compact W4A16 prompt-processing gap with a native MMQ/GEMM
   path that consumes the canonical allocation without a second resident copy;
7. freeze reproducible commands and baselines for BF16, NVFP4, FP8, W8A8,
   BitsAndBytes, and normal/act-order GPTQ, then optimize only the paths with a
   measured material gap;
8. verify MTP, VBR, and multimodal integration on representative Qwen3.8 models;
9. publish a user-facing format/backend/architecture support matrix, invocation
   examples, memory behavior, performance baselines, and precise limitations;
10. widen the architecture bridge only when a real checkpoint proves the
   canonical name/layout mapping.

Post-release, in priority order:

1. implement and prove ModelOpt W4A4 NVFP4 activation execution on Blackwell;
2. replace correctness-first BitsAndBytes and act-order GPTQ paths only where
   profiling demonstrates a worthwhile performance gap;
3. add optimized HIP executors for formats that have demonstrated user demand;
4. add new architectures only from real checkpoint fixtures; and
5. consider EXL3 or unusual WNA16 widths only as separately scoped projects.

## 10. Definition of done

This architecture project is complete when:

- Qwen3.5 safetensors no longer maintains a parallel tensor manifest;
- format-generic quant code is absent from the Qwen adapter;
- Qwen3.8-27B, Qwen3.8-Flash-Next/Qwen4, and DeepSeek-V4-Flash load native
  safetensors through their existing model classes;
- equivalent GGUF and safetensors tensors reach the same backend kernels;
- unsupported format/backend combinations fail before large allocations;
- no temporary GGUF or whole-model duplicate is created;
- every format in the Qwen3.8 release boundary has a pinned, reproducible
  correctness, KLD, memory, PP, and TG record;
- single-GPU, split, offload, and unsupported-backend behavior is documented and
  proven rather than inferred;
- both large MoEs pass layer/tensor split plus CPU-expert and adaptive-cache
  gates without changing their canonical expert ownership;
- correctness, KLD, placement, load-memory, PP, and TG gates pass; and
- adding a third already-supported architecture demonstrably requires only metadata/name/layout policy unless it introduces a genuinely new operation or quantization format.
