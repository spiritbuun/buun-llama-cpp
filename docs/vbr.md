# VBR: variable-bit-rate KV cache

VBR runs the KV cache on the TurboQuant codec ladder (f16 → turbo8 → turbo4 → turbo3_tcq →
turbo2_tcq → turbo1_tcq, 16 → 1.25 effective bits/value) and, in dynamic mode, changes
tiers **at runtime**: the cache starts at the high-quality entry tier and individual
(layer, K/V) tensors are transcoded down the ladder in place as the context fills, following
a measured per-model price order. You get near-entry-tier quality at short contexts and
bottom-tier capacity at full depth, from one flag.

## Quickstart

```sh
llama-server -m model.gguf -ctk vbr
```

That single flag is the complete product:

- the cache starts at **f16** — full quality and f16 decode speed until there is actual
  budget pressure (layers keep f16 speed until their own degrade step fires);
- a KV VRAM budget is derived automatically from the memory left after weights/compute
  (the fit pass), or falls back to the floor-layout cost of the full context;
- when `-c` is unset, the advertised context is the budget's capacity at the **floor**
  tier (t1 by default), capped at the model's training context;
- as mapped KV approaches the budget, the runtime controller degrades tensors down the
  price order, cheapest quality-per-byte first. With `-v` you can watch the
  `VBR degrade #…` steps fire.

The cache resets losslessly when it empties (tiers return to entry), and fully-degraded
runs stay coherent — the price orders were measured on KLD panels per model.

## Flags

| flag | meaning |
|---|---|
| no cache flags | common CLI tools default to dynamic VBR on both sides with a t4 quality floor; the tensors still enter at F16 and degrade only under pressure. |
| `-ctk vbr` / `-ctv vbr` / `-ct vbr` | explicitly select VBR and, without `--vbr-floor`, open the complete ladder to t1. An explicitly non-VBR side (`-ctv q8_0`) is pinned — it never degrades and counts in the aggregate at its fixed bits/value. |
| `--vbr-budget <tier\|number>` (`--vbr-bits`) | `dynamic` (default) = runtime controller. A tier (`t8/t4/t3/t2/t1`) or a number selects a **fixed** static tier instead — no runtime degrades. |
| `--vbr-entry <tier>` | dynamic entry tier: `f16` (default), `t8`, `t4`, `t3`, `t2`, or `t1`. A lower entry is an explicit quality-for-bandwidth trade; VBR continues degrading from that tier toward the floor. |
| `--vbr-floor <bits\|tier>` (`--vbr-min-bits`) | LITERAL aggregate K+V bits/value floor for dynamic mode. The degrade order stops at the last step whose aggregate stays ≥ the floor — e.g. `4.25` means "t4 layout with a few units held one tier higher", **not** a snap-up to the next tier. Clamped to the reachable ladder range (minimum 1.25; maximum depends on the selected entry and any pinned side). Implicit VBR defaults to t4 (4.125); explicit `-ct vbr` without this flag uses t1 (1.25). |
| `--vbr-vram <SIZE>` | explicit KV VRAM budget (e.g. `8G`). Default `auto` = derived by the fit pass. |
| `--vbr-policy <json>` | fixed mode only: a measured policy ladder; the best rung ≤ the fixed budget (and ≥ the floor) is selected and its per-layer schedule applied. |
| `--cache-ram <MiB>` | on the server, a nonzero value enables automatic projected-artifact host caching for supported dynamic-VBR topologies (default: 8192). `--cache-ram 0` is the zero-cost opt-out. |

Notes:

- The C API mirror is `llama_context_params.vbr_dynamic` / `vbr_vram_budget_bytes` /
  `vbr_min_bits` — that struct is the runtime channel; the CLI flags map onto it.
- A non-tier floor costs slightly more than the tier below it; capacity estimates account
  for this (the advertised context is scaled to the literal floor's cost).
- When both K and V are movable, the floor cannot exceed the selected entry tier. With one
  pinned side, the floor is aggregate and may be higher than the movable side's entry bpv.
- `/props` and `/models` on the server expose a `vbr` object (mode, floor, budget,
  realized/selected bits — `null` where a value is not a fixed number under the dynamic
  controller).

## Requirements and limitations

- **Backend**: turbo-typed KV needs a backend that exports the TurboQuant interface
  (`ggml-vbr.h`) — currently CUDA (and ROCm for the VMM pool). KV on other backends is
  refused at init; layers whose KV lands on the CPU (partial offload, `--no-kv-offload`)
  fall back to q8_0 per layer with a warning.
- **Flash attention** is required and force-enabled (turbo KV stores rotated-space data
  that only the FA paths decode).
- **Unified KV**: dynamic mode needs single-stream KV; with parallel sequences
  (`-np > 1`, perplexity/imatrix multi-sequence batching) unified KV is forced
  automatically, with a warning.
- **Qwen4 Flash Next**: live static Turbo and dynamic VBR are supported for its ordinary
  attention child. The QSA index cache remains a separate fixed-F16 allocation and the
  recurrent child remains fixed-layout. Host prompt artifacts carry both fixed-layout children
  as authenticated companions and adopt the complete tree atomically. Qwen4 currently uses the
  generic degradation order; a measured model-specific order is still pending.
- **Context shift / self-extend** is disabled under dynamic VBR (the shift graph would
  touch unmapped pool pages); generation stops cleanly when the context fills.
- **Host prompt cache**: supported server topologies use authenticated projected VBR
  artifacts under the ordinary `--cache-ram` budget. The fixed-layout state serializer,
  manual slot files, and KV shifting/cache reuse remain disabled because their snapshots
  carry tier-typed KV. If artifact construction is unavailable for a topology, startup
  reports the typed refusal and continues live-only. Context checkpoints stay enabled on
  hybrid models (recurrent state only) but are disabled on SWA models.
- **Exact continuation**: complete-frontier artifacts also authenticate the terminal raw
  vocabulary-logit row. Restoring an exact prompt therefore reports `prompt_n = 0`
  instead of replaying its last token solely to regenerate logits. With MTP or DFlash2,
  that cached first sample feeds the normal speculative loop; drafting is not disabled.
  While VBR prompt caching is enabled, the target keeps a full raw row instead of using
  backend-only sampling or the DFlash target-argmax shortcut.
- Degrade progress lines are `INFO`-level: visible with `-v`. Greedy-identical output does
  NOT mean no degrades happened.
- `--vbr-policy` exports its per-layer schedule through a process-global env: with a
  speculative **draft model** in the same process, the schedule also applies to the
  draft's cache (overriding `-ctkd/-ctvd`). Avoid `--vbr-policy` together with `-md`.

### Automatic host-cache support matrix

The default-on server route is intentionally narrower than the explicit artifact APIs.
It supports text-only, non-aLoRA slots on a dynamic-VBR target whose artifact store can
bind every attention child, device lane, and accounting domain. Ordinary LoRA identity,
attention-only targets, hybrid/recurrent targets, SWA/iSWA trees, MTP, external draft
models including DFlash2, single-GPU placement, and bound multi-GPU placement use that
route. Stateful trees and speculative slots are captured as exact complete-frontier
artifacts with their required companion state. Shortened stem publication remains
attention-only; unsupported stem shapes retain the complete live frontier.

The following automatic cases are typed live-only fallbacks today:

| Case | Typed reason | Behavior |
| --- | --- | --- |
| media prompt/projector state | `media_prompt_unsupported` | that live slot is neither captured nor displaced |
| aLoRA invocation boundary | `alora_invocation_unsupported` | that live slot is neither captured nor displaced |
| missing portable topology or pool/lane binding | `artifact_topology_unavailable` | automatic startup continues live-only |
| incomplete accounting or budget evidence | `accounting_unavailable` | automatic startup continues live-only |
| pinned-ring/store construction refusal | `artifact_store_unavailable` plus the store failure/status | automatic startup continues live-only |
| missing, corrupt, or mismatched Qwen4 QSA-index companion | `required_companion_unavailable` | the artifact is refused before publication; the live slot is retained or the prompt replays cold |

Automatic fallback never clears the live source. An explicit `--vbr-prompt-cache`
request is strict instead: an unsupported global topology or unavailable store fails
startup with the same typed reason. Transient capture/admission failures retain the live
source and retry; permanent per-artifact incompatibilities are terminal only for that
unchanged frontier.

### Host-cache control-plane payloads

The cache control plane uses one payload vocabulary everywhere: `fixed_state` is the
historical serialized target/draft state image and `vbr_artifact` is an authenticated,
immutable projected artifact. `/cache/plan` reports `payload_kind` on every host candidate;
cache lifecycle and destruction receipts use the same field. Lease inspection and event
records derive it from their typed `host_snapshot` or `vbr_reference` subject. Families
apply to both representations and therefore report
`payload_scope: ["fixed_state", "vbr_artifact"]`; they are lineage policy, not a second
payload owner. Live prefixes and checkpoints report a null payload kind because their state
has not entered the host tier.

## Degrade orders

The price order — which (layer, side) to degrade next, per tier band — is measured per
model on KLD panels and baked into the binary for the supported fleet
(`src/llama-vbr-degrade-orders.inc`). Models without a baked order use a **generic**
cross-model order synthesized from normalized-position rank curves averaged over the
measured fleet (dense, MoE-hybrid and SWA-mixed layouts); it is a good hedge but a
measured order is better. Override or supply one with:

```sh
VBR_DEGRADE_ORDER=order.txt llama-server ...   # tokens: <layer><k|v>:<t8|t4|t3|t2|t1>
```

`VBR_FORCE_GENERIC=1` forces the generic order on a supported model (A/B instrument).

## Environment appendix

CLI-managed internals (set nothing yourself; these are developer overrides on top of the
cparams channel): `VBR_VMM` (=0 forces the controller off, =1 on), `VBR_MODE`,
`VBR_BUDGET_MIB`, `VBR_MIN_BITS`.

Product-adjacent knobs: `VBR_DEGRADE_ORDER` (order file), `VBR_LAYER_SCHEDULE`
(per-layer static schedule, `<il0>-<il1>:<k|v>:<tier>;…` or `@file`; `VBR_SCHEDULE_CTX`
sets the discovery window the row coordinates refer to, `VBR_LAYER_STRICT=1` errors on
unsupported bands), `VBR_POLICY_LADDER` (default for `--vbr-policy auto`),
`VBR_STASH_ROWS` (f16 sink-stash rows per tensor, default 128; 0 disables),
`VBR_FORCE_GENERIC`.

Developer instruments (off by default, safe to ignore): `VBR_VRAM_HEADROOM_MIB` (free-VRAM
clamp headroom, default 192), `VBR_PROMOTE` (container promotion kill switch, default on),
`VBR_TRANSCODE_TEST[_N]` (in-binary transcode oracle), `VBR_TRANSCODE_FIDELITY`
(per-transcode error report; slow), `VBR_TRANSCODE_NOTILE` (tiling isolator; unbounded
scratch), `VBR_STASH_CAPTURE_ONLY` (stash ablation).

Telemetry exports for wrapper scripts (written, never read): `VBR_SELECTED_*`,
`VBR_CAPACITY_BITS`, `VBR_VRAM_BUDGET`, `VBR_BUDGET`.
