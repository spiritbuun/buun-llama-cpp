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
- The importer manually constructs every GGUF-compatible model and tokenizer key.
- Canonical-to-source name mapping, plain-tensor dtype conversion, and Qwen-specific
  permutations are still mixed in one file. Quant-format binding and validation
  now live in a reusable adapter.
- Model validation is fixed to one Qwen3.5 geometry.
- The quantization and architecture seams are separated, but the metadata and
  transform helpers still need extraction before architecture two is cheap.

The duplicated target manifest has been removed. The next structural target is
extracting reusable metadata, tokenizer, naming, and transform helpers so a
second architecture does not copy Qwen-specific importer machinery.

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
- [ ] Centralize dtype-to-ggml-type mapping where it is truly format-generic.
- [x] Add isolated fixtures for missing scales, wrong scale dtype, wrong grid
  shape, zero points, invalid global scalars, dependency consumption, and
  overlapping compressed-tensors rules.

Gate: quant-adapter unit tests cover each accepted and rejected contract; Qwen output and performance remain at baseline.

### Phase 4 — Extract reusable metadata and tokenizer helpers

- [ ] Split generic JSON/file helpers from the Qwen adapter.
- [ ] Add a metadata sink with strict required/optional field handling.
- [ ] Extract tokenizer vocabulary, merges, added-token, special-token, and chat-template conversion.
- [ ] Extract common RoPE and sampling-default helpers without weakening architecture validation.
- [ ] Keep architecture-specific GGUF key selection in the architecture adapter.

Gate: tokenizer round-trip tests match canonical token IDs and encode/decode results; chat-template output matches the GGUF reference.

### Phase 5 — Reduce Qwen3.5 to architecture-specific policy

- [ ] Replace imperative ordinary projection mapping with shared naming templates.
- [ ] Keep only recurrent projection names, MTP names, and genuine Qwen exceptions locally.
- [ ] Express row/column permutations as explicit transform objects.
- [ ] Express offset-norm and `A_log` conversions as explicit transform objects.
- [ ] Replace the fixed 27B check with validated geometry where the existing Qwen model implementation supports it.
- [x] Report block-FP8 and NVFP4 source formats distinctly.

Gate: the Qwen adapter contains no quant-format implementation and no complete target manifest.

### Phase 6 — Prove the seam with a second existing architecture

- [ ] Select a dense architecture already well supported through GGUF and available in one supported safetensors quant format.
- [ ] Add only its configuration/name/layout adapter.
- [ ] Do not add a model graph or duplicate backend kernel for the sake of safetensors.
- [ ] Document which portions were reused unchanged.

Preferred proof target: a conventional dense Llama- or Qwen3-family model. A second irregular hybrid architecture should wait until the common path is proven.

Gate: the second architecture loads through its existing `llama_model_*` class, matches a clean GGUF/reference KLD panel within the declared near-zero tolerance, and exercises the same backend kernels for equivalent runtime tensor types.

### Phase 7 — Loading and memory optimization

- [ ] Replace avoidable full-vector copies with bounded reusable staging buffers.
- [ ] Stream transforms that are naturally row- or block-local.
- [ ] Upload directly into the final selected backend buffer when legal.
- [ ] Preserve one-final-tensor-at-a-time peak-memory bounds for non-streamable transforms.
- [ ] Investigate zero-copy CPU mappings only for source layouts already identical to canonical ggml layouts.
- [ ] Measure load time, peak host RAM, peak VRAM, and storage reads.

Gate: no complete model duplication, no temporary GGUF, and no throughput regression after load.

### Phase 8 — Add further quant formats independently

- [ ] Add W8A8 compressed-tensors support.
- [ ] Add AWQ support with an explicit decision between native execution and load-time repack.
- [ ] Add non-act-order GPTQ support.
- [ ] Treat act-order GPTQ as a separate runtime-kernel project unless an exact canonical representation is designed.
- [ ] Require each adapter to declare CPU, CUDA, HIP, split/offload, and mmap capabilities.

Gate: each new format has contract fixtures, projection correctness, end-to-end KLD, backend capability checks, and independent PP/TG measurements.

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

## 8. Recommended immediate order

Phases 1–3 are implemented. Recommended next sequence:

1. freeze the current NVFP4/block-FP8 correctness and performance baselines;
2. extract metadata and tokenizer helpers;
3. replace Qwen's ordinary projection table with shared naming templates and
   explicit transform objects;
4. run the complete regression matrix;
5. add one straightforward dense architecture as proof of the bridge;
6. only then expand numerical-format coverage.

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
