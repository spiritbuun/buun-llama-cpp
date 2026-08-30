# Qwen4 Flash Next VBR support plan

Status: core implementation complete; release validation in progress
Target architecture: `LLM_ARCH_QWEN4EXP`
Last updated: 2026-08-30

## Goal

Give Qwen4 Flash Next the same release-quality VBR product contract as the already-supported
architectures:

- static TurboQuant KV (`t8`, `t4`, `t3`, `t2`, and `t1`);
- dynamic F16-to-T1 VBR with exact fit and runtime accounting;
- dense and QSA-sparse attention correctness across tier changes;
- native Qwen4 MTP compatibility;
- prompt-artifact capture and restore, including QSA index state;
- single- and multi-GPU layer placement;
- measured Qwen4 degradation order and documented telemetry.

The work should reuse the ordinary attention VBR implementation. Qwen4's attention K/V is a
normal attention cache; its QSA index keys are a separate, fixed-layout cache. No Qwen-specific
Turbo codec kernel should be needed.

## Current state

Qwen4 live execution now admits static Turbo and dynamic VBR. The attention child is the sole
representation owner, the QSA index child is fixed F16, and automatic prompt artifacts carry
authenticated attention, recurrent, and QSA-index state as one atomic tree. The remaining release
work is the broader numerical/lifecycle matrix, measured degradation order, and capacity/soak
validation below.

The Qwen4 memory tree is:

```text
llama_memory_hybrid_idx
├── mem_attn  ordinary attention K/V for 12 full-attention layers
├── mem_recr  recurrent state for 36 linear-attention layers
└── mem_idx   raw QSA index keys mirroring the attention-cache cells
```

VBR ownership must be:

```text
mem_attn  dynamic/static Turbo owner
mem_recr  unchanged fixed recurrent state
mem_idx   unchanged fixed F16 index state
```

QSA ranking reads `mem_idx`, while attention reads and writes `mem_attn`. The attention path
already applies the normal rotated K/V cache writes, so the existing Turbo/VBR attention kernels
are structurally applicable. The index cache cannot be reconstructed from attention K/V because
it comes from a different learned projection.

### Reference model geometry

Use the accepted Qwen3.8 Flash Next GGUF as the primary test shape:

- 48 trunk layers: 12 full-attention and 36 linear-attention layers;
- embedding width 2560;
- 24 query heads, 2 KV heads, K/V head dimension 256;
- QSA index dimension 128 and `top_k = 2048`;
- training context 262,144.

At the full training context, attention K/V alone costs approximately:

| Layout | Approximate attention-KV memory |
| --- | ---: |
| F16 | 6.00 GiB |
| T4, 4.125 bits/value | 1.55 GiB |
| T1, 1.25 bits/value | 0.47 GiB |

The current fixed F16 index allocation is approximately 1.50 GiB if both generic K and V
planes are allocated. Making the indexer key-only could halve that cost, but it is a separate
optimization and is not required for VBR correctness.

## Design invariants

These are release blockers, not preferences.

- [x] Only `mem_attn` owns Turbo layouts, VBR epochs, degradation state, and VBR budget.
- [x] `mem_idx` remains F16 and never exposes a Turbo/VBR controller.
- [x] `mem_recr` remains fixed-layout recurrent state.
- [x] Attention and index occupancy, cell identity, sequence ownership, and positions never
      diverge after any mutation or rollback path.
- [x] The fit pass accounts for the indexer's context-linear fixed cost without charging it to
      the VBR budget or scaling it to the VBR floor.
- [x] Dense QSA, sparse QSA, and the dense-to-sparse transition remain correct at every tier.
- [x] A representation-epoch change invalidates or rebuilds every graph assumption that depends
      on the attention cache layout without disturbing the fixed index cache.
- [x] MTP draft memory remains independent and non-VBR; target VBR state remains authoritative.
- [x] Prompt artifacts either carry authenticated QSA index state atomically with the other
      companions or fail closed/live-only. An attention-only artifact must never be adopted.
- [x] Unsupported non-Flash-Attention or device-backend combinations retain the existing typed
      failure behavior; CPU/partial-offload sides retain the generic safe Q8_0 pinning policy.
- [x] No DeepSeek-specific grouped-tier, padding-page, or lockstep-cursor machinery is imported.

## What carries over from the deferred DeepSeek V4 work

The archived `fork-dsv4-vbr` worktree is a design reference only. Do not
cherry-pick it: it is based on an older tree and solves a materially different cache topology.

Reusable concepts:

- fixed F16 ownership for an auxiliary index cache;
- a complete parent-memory forwarding checklist for every VBR API;
- parent-aware separation of controller-managed and fixed memory;
- logical-prefix mapping before artifact restoration;
- measured degradation orders over only the cache units the model actually owns.

Do not port:

- `vbr_group_prepare` or grouped/lockstep tier transitions;
- hard-seal coordination across concatenated caches;
- padding-page and combined-child-epoch machinery;
- the DeepSeek-specific `np = 1` restriction;
- tensor-parallel VBR assumptions from `weekly-runs/exp39_tp_vbr_recon.md`.

DeepSeek V4 concatenates raw/SWA and compressed attention state that must move tiers together.
Qwen4 has ordinary attention K/V plus a separate fixed QSA indexer, so its correct design is
simpler: one existing VBR child and one fixed companion.

## Implementation phases

Each phase has a stop/go gate. Keep the architecture refusal in place until the phase that first
provides a complete live-runtime contract; avoid adding a permanent experimental environment
variable.

### Phase 0 — Freeze the baseline and turn the refusal into a contract

- [ ] Preserve the current Qwen4 Turbo/VBR refusal as a negative test.
- [ ] Add the exact reference geometry and F16/T4/T1 allocation math to a focused Qwen4 VBR test.
- [ ] Record current dense-QSA, sparse-QSA, MTP, fit, and prompt-cache behavior on the accepted
      model and binaries.
- [ ] Add a scoped test fixture that exposes the three memory children without relying on log
      text.
- [ ] Record a source and binary manifest for later A/B and memory comparisons.

Exit criteria:

- The current failure is deterministic and typed.
- Expected attention, recurrent, and index allocation sizes are independently asserted.
- Later phases can prove which child owns each byte and each controller operation.

### Phase 1 — Separate controller-managed memory from fixed context-linear memory

The existing fit path temporarily prices K/V at the VBR floor and scales context memory. That is
wrong for Qwen4's fixed, context-linear QSA index allocation: including it in the scalable total
can over-advertise context or overcommit the VBR pool.

- [ ] Add an internal memory-breakdown authority that identifies bytes managed by the VBR
      controller. Prefer a narrow virtual such as `memory_breakdown_vbr_managed()` over another
      model-specific condition in `common/fit.cpp`.
- [ ] Make ordinary `llama_kv_cache` report its controller-managed context allocation.
- [ ] Make `llama_memory_hybrid` forward only its attention child's managed allocation.
- [ ] Make `llama_memory_hybrid_idx` report the same attention-managed allocation while keeping
      `mem_idx` in the fixed context-linear total.
- [ ] Update fit calculations to price these independently:
  - controller-managed context bytes at entry/floor/fractional-floor cost;
  - fixed context-linear index bytes at their actual type;
  - fixed model/recurrent/compute bytes unchanged.
- [ ] Hand the runtime controller a budget that excludes the fixed index allocation.
- [ ] Add overflow-checked arithmetic for the 262,144-token case and partial offload.
- [ ] Expose separate telemetry for controller-managed KV bytes/value and fixed index bytes/token.
      Keep `kv_bpv()` attention-only rather than silently changing its meaning.

Exit criteria — Gate A, accounting:

- For explicit and automatic context sizes, `managed + fixed-linear + fixed` equals the actual
  allocation within allocator alignment.
- Changing the VBR floor changes only the managed term.
- The fixed indexer is reserved before the VBR budget is derived and cannot be overcommitted.
- Fit tests cover single GPU, layer-split multi-GPU, CPU-pinned layers, and MTP target overhead.

### Phase 2 — Construct live Qwen4 Turbo/VBR memory with strict ownership

- [x] Extend `llama_memory_hybrid_idx` construction to accept the VBR parameters needed by its
      attention child.
- [x] Pass configured K/V types and VBR state only to `mem_attn`.
- [x] Force `mem_idx` to F16 regardless of `-ctk`, `-ctv`, static Turbo tier, or dynamic VBR.
- [x] Leave `mem_recr` unchanged.
- [x] Audit every VBR API on `llama_memory_hybrid` and `llama_memory_hybrid_idx`; forward tier,
      epoch, capture, freeze, preflight, floor, scratch, cotenancy, seal, and reset operations only
      through `mem_attn` unless the API explicitly describes whole-tree state.
- [x] Enable static Turbo first and prove T8/T4/T3/T2/T1 allocation and decode.
- [x] Enable dynamic VBR for live execution after the same ownership tests pass.
- [x] Keep automatic prompt-artifact capture typed live-only at this phase.

Expected core files:

- `src/llama-memory.h`
- `src/llama-kv-cache.{h,cpp}`
- `src/llama-memory-hybrid.{h,cpp}`
- `src/llama-memory-hybrid-idx.{h,cpp}`
- `src/llama-model.cpp`
- `common/fit.cpp`

`src/models/qwen4exp.cpp` should not need codec-specific changes. Any proposed Qwen-specific
Turbo kernel is a design-review trigger.

Exit criteria — Gate B, ownership:

- Runtime inspection proves `mem_attn` enters F16 and can traverse the configured ladder.
- `mem_idx` is always F16, has no Turbo interface, and its bytes do not change during a tier flip.
- Static Turbo and dynamic VBR both fail cleanly on unsupported placements/backends.

### Phase 3 — Make all sequence mutations transactional across three children

`llama_memory_hybrid_idx` already overrides the common sequence mutations, but transient and
try-style operations inherited from `llama_memory_hybrid` can currently omit `mem_idx`. This is a
correctness gap even before VBR and becomes release-critical when tier changes add more rollback
paths.

- [x] Inventory every mutation entry point in `llama_memory_i`, including:
  - `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, and `seq_div`;
  - attention-only and transient removal variants;
  - `try_seq_cp` and `try_seq_cp_transient`;
  - clear/reset/defrag/context-shift-related operations that can change cell identity.
- [x] Override each applicable operation in `llama_memory_hybrid_idx`.
- [x] Apply operations transactionally across attention, recurrent, and index children.
- [x] On a child failure, roll back already-mutated children or leave all children unchanged.
- [x] Add an internal invariant checker for tests: occupied cells, sequence IDs, positions, and
      logical prefixes must agree between attention and index children.
- [ ] Run the checker after server slot reuse, speculative rejection, prompt truncation, and
      sequence copy paths.

Exit criteria — Gate C, alignment:

- Every sequence operation preserves attention/index alignment.
- Injected failure at each child boundary proves rollback.
- Speculative rejection and server slot reuse do not leave orphaned index cells.

### Phase 4 — Validate live CUDA VBR across dense and sparse QSA

- [x] Exercise entry F16 and force each T8/T4/T3/T2/T1 tier through live CUDA decode.
- [x] Test below the QSA sparse-selection boundary, above `top_k`, and across a continuation that
      changes from dense to sparse selection.
- [x] Change tiers after a dense write, before each sparse continuation write, and while the graph
      representation epoch changes.
- [ ] Verify post-retier writes, cell reuse, canaries, and logical slot equality.
- [ ] Test odd lengths, multiple sequences, prompt batches, and one-token decode.
- [ ] Confirm CUDA graph rebuild/reuse decisions include the attention representation epoch but
      do not spuriously treat fixed `mem_idx` as tiered.
- [ ] Confirm `--no-kv-offload`, non-FA, CPU KV, and unsupported mixed placements fail closed.
- [ ] Run a long-context growth test to the configured floor and verify each degrade step releases
      the predicted bytes.

Exit criteria — Gate D, live numerical correctness:

- Dense and sparse QSA logits remain within the existing per-tier Turbo tolerances.
- No tier transition changes QSA-selected cell identities relative to the fixed-index reference.
- Actual allocator usage tracks the Phase 1 prediction throughout degradation.

### Phase 5 — Prove native Qwen4 MTP compatibility

The native Qwen4 draft context owns a separate plain draft attention cache. Its VBR controls
remain disarmed; target VBR is the only dynamic controller.

- [x] Construct target VBR plus the Qwen4 MTP sidecar/context.
- [x] Assert draft memory has no dynamic VBR controller and target memory has exactly one.
- [x] Exercise accepted and rejected draft tokens before, during, and after target tier changes.
- [x] Verify target attention/index alignment after rejection rollback.
- [x] Verify draft graph/caching does not inherit a target degradation policy through a
      process-global channel.
- [ ] Compare acceptance, target tokens, and memory accounting with the non-VBR target.

Exit criteria — Gate E, speculative correctness:

- Accepted and rejected speculation cannot desynchronize target `mem_attn` and `mem_idx`.
- Draft memory remains fixed and independent.
- Long speculative runs survive every target tier transition without stale graph reuse.

### Phase 6 — Add QSA index state to VBR prompt artifacts

Attention-only restoration is invalid for Qwen4 because QSA index keys come from a separate
projection. Full support requires an authenticated fixed-state companion.

- [x] Add a typed artifact companion for QSA index state, `qsa_index`.
- [x] Capture the occupied logical prefix, cell metadata, and fixed F16 index bytes at the same
      frontier as attention and recurrent state.
- [x] Stage and validate attention, recurrent, index, terminal-logit, model identity, adapter
      identity, sequence, and representation metadata before mutating live memory.
- [x] Adopt all companions atomically. Any missing, corrupt, mismatched, or truncated companion
      must leave the live slot untouched.
- [x] Map logical prefixes/cells before copying bytes; do not assume physical cache-cell identity
      survives eviction and restoration.
- [x] Bind artifact authentication to the Qwen4 index geometry and layout.
- [x] Teach the server artifact store to select this companion for
      `llama_memory_hybrid_idx`/Qwen4.
- [x] Until all checks pass, automatic `--cache-ram` capture reports a typed live-only reason,
      while an explicit artifact request must fail clearly rather than silently omitting index
      state.

Expected artifact files:

- `src/llama-vbr-artifact*.{h,cpp}`
- `src/llama-vbr-explicit-capture.{h,cpp}`
- `tools/server/server-vbr-artifact-store.{h,cpp}`
- focused artifact catalog/authentication tests

Exit criteria — Gate F, artifact atomicity:

- Capture, evict, and zero-replay restore above the QSA sparse boundary matches a live slot's
  tokens/logits.
- Missing or corrupt index state is rejected before any child is adopted.
- Failure injection at every stage proves the original live slot remains usable.
- MTP complete-frontier restoration resumes the normal speculative loop.

### Phase 7 — Measure and bake the Qwen4 degradation order

Qwen4 currently has no architecture-specific entry in `src/llama-vbr-degrade-orders.inc`.
The generic order is acceptable for functional bring-up, but not for final quality claims.

- [ ] Reproduce the established KLD measurement methodology on the exact Qwen4 model family.
- [ ] Measure the 12 attention layers, K and V independently, over all five transitions: 120
      priced steps.
- [ ] Use a representative prompt panel that covers dense and sparse QSA, long context, code,
      prose, and retrieval.
- [ ] Confirm the fixed index state is identical across all measurements.
- [ ] Generate the table with `scripts/make-vbr-pricing-tables.py` and preserve raw matrices,
      command lines, model hashes, and generator version.
- [ ] Add `{LLM_ARCH_QWEN4EXP, 48}` to the degradation-order table.
- [ ] Define fallback matching by exact attention-KV layer geometry so an MTP-attached target does
      not accidentally select a different order.
- [ ] Run greedy, perplexity/KLD, and task-quality gates at T4 and T1 floors.

Exit criteria — Gate G, quality:

- The baked order beats or matches the generic order at equal realized bits/value.
- The order contains exactly the model's controller-owned attention units—never recurrent or
  index state.
- Raw measurement evidence is reproducible and checked into the normal receipt location.

### Phase 8 — Capacity, multi-GPU, soak, and release

- [ ] Test explicit and automatic context sizing at F16, T4, fractional floors, and T1.
- [ ] Verify the 262,144-token advertised capacity and actual allocations agree with accounting.
- [ ] Test 1× GPU and 2× layer placement; verify each attention child binds to the correct device
      pool and fixed index state follows the owning layer placement.
- [ ] Treat tensor split as a separate support case. Do not infer it from layer-split success.
- [ ] Run long server soaks with slot reuse, prompt artifacts, MTP, context exhaustion, and reset.
- [ ] Verify empty-cache reset returns attention tiers to entry F16 without reallocating or
      changing the index layout.
- [ ] Add Qwen4 to `docs/vbr.md` support/limitation tables and document fixed index memory in
      `/props` and `/models` telemetry.
- [ ] Remove the architecture refusal only after all required gates pass in the same candidate.

Release acceptance:

- No crashes, allocator overcommit, stale graph reuse, cell misalignment, or silent artifact
  fallback.
- Zero typed backend/controller failures in supported configurations.
- Capacity predictions match observed peak memory within allocator/alignment tolerance.
- Quality gates pass at the documented default floor.
- Performance is measured against fixed F16 and static Turbo controls; VBR bookkeeping does not
  impose material overhead before the first degradation.

## Test matrix

Prefer a new focused `tests/test-qwen4-vbr.cpp` for the cross-child contracts, while keeping model
construction cases in `tests/test-llama-archs.cpp` and generic artifact/epoch assertions in their
existing suites.

| Area | Required cases |
| --- | --- |
| Construction | dynamic VBR, every static Turbo tier, fixed F16 index, fixed recurrent state, unsupported backend/FA/placement refusals |
| Accounting | explicit/auto context, F16/T4/T1/fractional floors, partial offload, 1×/2× layer placement, MTP overhead |
| QSA | dense, sparse, dense-to-sparse continuation, tier flip on each side of transition, graph epoch rebuild |
| Numerics | entry F16 and T8/T4/T3/T2/T1, odd lengths, prompt batch, multi-sequence, canaries |
| Mutation | every sequence method, transient/try variants, injected child failure and rollback |
| MTP | accept/reject, tier flip during speculation, draft controller absent, target/index alignment |
| Artifacts | complete capture/restore, physical-cell remap, missing/corrupt companion, atomic rollback, MTP resume |
| Lifecycle | grow, degrade, empty reset, slot reuse, server soak, context exhaustion |
| Quality | KLD price panel, generic-vs-measured order, T4 default, T1 floor |

Mutation-resistant tests must observe the actual owning child or actual controller action. Output
equivalence alone is insufficient when a fixed path could produce the same result while bypassing
the intended VBR route.

## Recommended landing sequence

Keep commits independently reviewable:

1. managed-vs-fixed accounting API and fit tests;
2. indexed-hybrid ownership plus static Turbo construction;
3. transactional sequence operations;
4. dynamic live VBR plus dense/sparse CUDA tests;
5. Qwen4 MTP tests;
6. QSA index artifact companion and server integration;
7. measured degradation order, telemetry, documentation, and final enablement.

Before each merge boundary:

- run the focused CPU/CUDA tests with instrumentation enabled where needed;
- rebuild production with instrumentation disabled and verify the test symbols/counters are absent;
- run `git diff --check` and a multi-lens simplification/correctness review;
- preserve exact source, binary, model, command, allocation, and quality receipts.

## Deliberate non-goals

- Compressing the QSA index cache.
- Converting the generic index allocation to key-only storage.
- Importing DeepSeek grouped-tier or tensor-parallel VBR machinery.
- Adding a Qwen-specific Turbo codec kernel without evidence that ordinary attention kernels are
  insufficient.
- Enabling context shift/self-extend under dynamic VBR.
- Silently accepting prompt artifacts that omit QSA index state.

These can be evaluated after the full support contract above is stable.

## Progress log

Record evidence here as phases close.

| Date | Phase/gate | Result | Receipt or commit |
| --- | --- | --- | --- |
| 2026-08-30 | Plan | Initial design and release gates recorded | this document |
| 2026-08-30 | Phase 0 / Phase 1 foundation | In progress: Qwen4 ownership/reference-memory contract added; generic representation-managed context accounting implemented; architecture refusal intentionally retained | local `test-llama-archs --arch qwen4exp` and VBR representation-epoch CPU gates |
| 2026-08-30 | Phase 2 / Phase 3 foundation | In progress: VBR parameters are scoped to the attention child, QSA index storage is hard-pinned F16, and indexed attention/transient/copy mutation routes now include rollback-tested index operations | unified two-sequence success/failure mutation fixture in `test-llama-archs` |
| 2026-08-30 | Phase 2 live admission | In progress: architecture-level Turbo/VBR refusal removed; CPU-bound movable Turbo sides follow the generic safe Q8_0 fallback; controller construction is admitted; indexed prompt-artifact topology remains typed live-only/strict-fail through `llama_memory_tree_collect` | CPU Qwen4 construction gate plus indexed-tree refusal contract; CUDA runtime gate pending |
| 2026-08-30 | Gate B / initial Gate D | PASS on RTX 3090: all five static Turbo tiers decode; one dynamic context traverses F16→T8→T4→T3→T2→T1 with decode after every applied production step; QSA index K remains bit-identical during every retier and fixed F16; the same context crosses the 256→512 dense/sparse selection boundary; indexed artifact collection refuses live-only state | `qwen4-vbr-20260830T-current/full-ladder-gate/test-llama-archs.log` |
| 2026-08-30 | Real-model smoke | PASS: shipped four-shard Qwen4 starts with `-ctk/-ctv vbr`, explicit 96 MiB budget and t1 floor; VMM attention pool, fixed recurrent state, and separate fixed-F16 QSA index cache are constructed; 19-token prompt + 64-token decode completes; automatic artifact capture fails closed to live-only | `qwen4-vbr-20260830T-current/real-model-smoke/server.log` |
| 2026-08-30 | Real-model pressure | PASS: shipped GGUF processes a 2,500-token prompt with a feasible explicit 64 MiB attention budget; at the 2,560-cell watermark the production controller applies 18 generic-order steps and maps exactly 64 MiB; an intentionally impossible 16 MiB case exhausts safely and still completes, documenting the 52 MiB CUDA-page floor at the seed watermark | `qwen4-vbr-20260830T-current/real-model-pressure-feasible/` and `real-model-pressure/` |
| 2026-08-30 | FA/CPU fallback and multi-sequence | PASS after fixing an early/late FA-gate mismatch found by the new test: device dynamic VBR now promotes explicit FA-off before memory construction and arms the controller; `offload_kqv=false` stays static host Q8 without spuriously enabling FA; `n_seq_max=2` is forced to one unified stream and both sequence positions stay aligned across attention/index children | `qwen4-vbr-20260830T-current/fa-cpu-fallback-fix/test-llama-archs.log` |
| 2026-08-30 | Gate E live MTP | PASS on the attached Qwen4 MTP artifact: target owns the sole 64 MiB VBR controller, the MTP draft explicitly disarms its controller, and a target F16→T8 transition occurs during generation at the 2,304-cell frontier. The same 800-token completion finishes with 199 accepted and 401 rejected/rollback draft cycles after/beside the retier; target reports 10.09375 realized KV bits/value | `qwen4-vbr-20260830T-current/mtp-retier-during-spec-2/` |
| 2026-08-30 | Phase 8 multi-GPU bring-up | PASS for 2×3090 layer split: two device-local VMM pools receive proportional shares of one 64 MiB controller and 22 steps are applied across both devices during a 2,500-token prompt. Tensor split also constructs mirrored VMM pools and decodes at a feasible 128 MiB budget; its 104 MiB seed/page floor is materially higher than layer split and is tracked as a separate capacity case | `qwen4-vbr-20260830T-current/dual-layer-pressure/`, `dual-tensor-pressure/`, and diagnostic `tensor-split-refusal/` |
| 2026-08-30 | Gate F QSA artifact atomicity | PASS on RTX 3090: a real 676-token Qwen4 prompt captures a 672-token VBR artifact, erases the live slot, atomically restores attention, recurrent, and typed QSA-index companions after physical-cell relocation, and evaluates only the 4-token suffix (`cache_n=672`, `prompt_n=4`). Missing/corrupt companion and rollback paths remain mutation-tested and fail closed | `qwen4-vbr-20260830T-current/qsa-companion/final-real-roundtrip-pass2/` |
