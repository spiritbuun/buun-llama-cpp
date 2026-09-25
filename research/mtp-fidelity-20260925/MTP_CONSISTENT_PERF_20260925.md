# MTP batch-consistent arithmetic: cost recovery, 2026-09-25

Investigation on `investigate/mtp-fidelity-20260925`, base `0b2789f23`.
Same Dorei mixed-quant 27B as `MTP_FIDELITY_20260925.md`; not Apollo's P100/Q6_K.
No production dispatch changes or public knobs added. Private probes in `bench/`.
This round retained no kernel edits: local tracked tree and Dorei's active
binary directory have been restored to the pre-experiment baseline. Failed
variants are preserved privately for diagnosis, not enabled behind public flags.

## Controls

- `mtp-fixed-fa.cpp`: lean preload, no tensor tracing/synchronization. Selects
  an existing F16 D256 attention tile for this 24-query-head/4-KV-head shape.
- `mtp-consistent-perf.py`: code-only, 6 cold requests per mode. Excludes the
  first request from timing medians. All six have the same text and MTP71/72
  acceptance. Both plain and MTP are measured.
- The unrestricted-clock forward sweep `mtp-perf-a.log` thermally throttled:
  SW Thermal Slowdown active, SM clock falling to1455MHz. Reject its small
  performance differences as confounded.
- Reverse sweep `mtp-perf-locked-b.log` temporarily locks core1200MHz, restores
  unlocked clocks on exit. No power-limit or memory-clock changes. These are
  controlled relative timings, NOT headline full-speed numbers.

| Configuration | Plain t/s | MTP t/s |
|---|---:|---:|
| Native dispatch | 40.731 | 88.798 |
| MMVQ only | 40.796 | 87.602 |
| FA tile4×8 only | 40.766 | 88.719 |
| MMVQ + tile4×8 | 40.766 | 87.656 |
| MMVQ + tile1×8 | 40.756 | 87.738 |
| MMVQ + tile2×8 | 40.788 | 87.747 |

Most cost is projection dispatch, ~1.3% under this controlled clock. Tile1×8
also passes the short64-position exact-logit panel; tile2 timing alone is not
a correctness qualification. Tile1 is not proven invariant at deep context,
where split-K heuristics depend on the number of output tiles.

## Candidate: small-batch MMVQ + same-shape ADD

Current MMVQ fusion supports only single-row dense inputs. Prototype extends
same-shape bias/residual ADD to widths2–4 on SM86, initially IQ2_XS/IQ3_S only.
Reuses the existing MMVQ reduction and per-column bias indexing; rejects output
overlap with projection inputs and noncontiguous biases. No gate/scale fusion
is newly admitted. Single-row arithmetic is unchanged.

Version1 builds and passes all54 short-replay comparisons with exact zero KLD.
But clock-controlled serving regresses: consistent-before87.67 vs fused86.23
and86.26t/s. The profiler and code are being examined before rejecting it.
The general fused kernel carries optional gate machinery, although the new
multi-row matcher only admits ADD. Version2 removes that unused machinery at
compile time; it recovered the regression but supplied no clear gain.

Remote root `/root/bench/vbr-reset-51a215d7e-20260925`:

- `baseline-bin-mtp-consistency/`: untouched pre-candidate binaries/libraries.
- `candidate-bin-mtp-bias-v1/`: first fusion binaries, preserved for profiling.
- `candidate-bin-mtp-bias-v2/`, `v3/`, `v4/` (same prefix): later candidates.
- `build-cuda/bin/`: restored baseline after the experiments. Set **LD_LIBRARY_PATH** explicitly when
  using archived binaries; their RUNPATH points to the original build location.
- `mtp-consistent-before.nsys-rep` and `*-kernels.csv`: only width4 teacher-forced
  decode captured, not model loading/prefill. MMVQ dominates; ADD is1.2% of
  aggregate GPU time (2816 launches /64 token positions).

Private `mtp-bias-unit.cpp` exercises both quants, widths1/2/3/4, even/odd
output-row counts, and both ADD operand orders. It writes deterministic GPU
results for byte comparison across libraries. All32 cases pass, and the version1
and version2 outputs are byte-identical to pre-change GPU results.

### Profile of the regression

IQ3_S, four columns: baseline118 registers/thread; generic fused136; bias-only
specialization126. V1's fused subset averages89.42us; V2 restores it to69.97us.
The combined IQ3_S time goes from92.40ms before, to100.86ms inV1, back to92.95ms
inV2. Captures have the same16 verification batches; absolute timings use
unlocked clocks, so use register counts/launch structure as stronger evidence
than sub-percent differences. V2 clock-controlled serving is87.75/87.71t/s,
essentially neutral versus consistent-before87.75. No independent win claimed.

Next experiment: one output row per block rather than two for these types at
width3/4 in the SM86-compiled kernel. Same warp count, dot products, and reduction
order. Host launch geometry must use the compiled architecture, not merely the
runtime device, so fallback cubins do not disagree with host grid sizing.
V3 (one row) passes all32 isolated tests and all54 short-logit comparisons
exactly. Serving loses:82.34t/s vs native88.79/88.77 in the bounding ABBA arms.
Telemetry stays at1200MHz with thermal throttle inactive; no concurrent compile
in this run. Profiles show IQ3_S total92.40→117.89ms and IQ2_XS34.78→43.39ms,
despite registers falling to80–95. GridX doubles: lower register use did not
pay for the additional blocks and reduced input reuse. Reject V3.

V4 tries the opposite, four rows/block, with an explicit guard against loading
weight rows beyond an odd-sized matrix's end. All32 isolated cases and all54
short-logit comparisons pass exactly. But serving is65.81/65.82t/s versus
native88.78/88.84; reject. The follow-up profile puts combined IQ3_S time at
223.43ms and IQ2_XS at80.00ms. `localMemoryPerThread` is0 in both baseline andV4,
so do **not** label this a proven register-spilling regression. The larger
work-per-block variant is simply much slower on these shapes. Both V3/V4
ABBA sweeps had no concurrent compilation and no observed thermal throttling.

## End state and next useful work

- Exactness remains demonstrated by the diagnostic fixed-dispatch controls,
  not by a shipped batch-invariance feature.
- Attention tiling is nearly free in this short-context screen. Aligning the
  projection path costs about1.3% at a controlled1200MHz. Do not advertise the
  capped-clock88t/s as a replacement for the original~108t/s full-speed result.
- Bias fusion, its lean specialization, and block-row sweeps did not recover
  that cost. No losing/neutral kernel experiments retained in the fork.
- Private `bench/mtp-bias-v4-experiment.patch` holds the final trial, including
  the bounded-load guard. Changing its row count4→1 reproduces the V3 layout;
  removing the row-count override recovers V2. V1 additionally lacks the
  compile-time bias-only specialization. Archived candidate libraries remain
  available for exact reproduction.
- Next focus: keep the fast projection execution while reconciling activation
  scale grouping and accumulation, rather than compensating with ADD fusion.
  Matching activation scales alone does not guarantee matching reduction
  rounding, and the existing FLA-vs-rollback prefill distinction remains a
  separate boundary. Longer contexts, other quants and other architectures are
  not qualified by this short-context screen.

All clocks restored; no server or profiler left running on Dorei. Logs, per-arm
commands/responses, telemetry and kernel summary CSVs copied to the local
`apollo-20260925-results/dorei/` directory. Full Nsight captures stay in the
remote results directory. Nothing committed/pushed from these experiments.

## Follow-up: isolate projection types

Private `mtp-projection-select.cpp` holds all small-width quantized projections
at MMVQ except one selected type, with attention held at 4x8. The short prose
teacher replay (31-token prefix, 64 positions, widths 1/4) gives:

| Projection allowed to use native dispatch | mean KLD | argmax flips |
|---|---:|---:|
| None (consistent control) | 0 | 0 |
| Q2_K | 0.000337772 | 1 |
| Q4_K | 0.000353227 | 1 |
| IQ4_NL | 0 | 0 |

The IQ4_NL result is specific to this model's exercised operations; it is not
proof of general MMQ/MMVQ equivalence. Q2_K and Q4_K independently invalidate
the idea of correcting only Q2_K activation scale grouping.

Nsight captures `mtp-projection-select-prof-{type}` isolate the width4 decode
loop. Aggregate kernel times over the same 64 positions (unlocked clocks):

| Type | MMVQ | MMQ | MMQ fixup |
|---|---:|---:|---:|
| Q2_K, 448 calls | 36.03 ms | 27.57 ms | 1.15 ms |
| Q4_K, 304 calls | 10.90 ms | 11.73 ms | 1.17 ms |

Activation quantization is additional and not included in these columns.
Q2_K dominates the projection cost of enforcing consistent arithmetic; Q4_K
MMQ is not actually faster for the particular shapes in this model.

Next candidate: transpose the fully unrolled row/column traversal for Q2_K
and Q4_K, widths 2–4, SM86 only, no fusion. Reuse a weight fragment across
columns before advancing the output row, without changing each accumulator's
dot products, K order, or reduction. No activation quantizer change.

Row-first result: all 64 isolated cases match byte-for-byte, and all 54 short
replay records are exact. Serving at 1200 MHz: native 88.762 t/s (first arm),
consistent 87.673/87.686, candidate 87.688/87.695. Neutral; do not retain.
Reprofiling confirms no useful shift: Q2_K 36.03 -> 35.94 ms, Q4_K
10.90 -> 11.01 ms; register counts 106 -> 107 and 89 -> 93. These profile
timings are unlocked-clock diagnostics, not evidence of sub-percent speedups.

Further source-level distinction: Q4_K's MMVQ computes its minimum correction
from the quantized activation integers and scale (`vec_dot_q4_K_q8_1_impl_vmmq`),
whereas MMQ consumes a stored sum computed from the original F32 activations
(`quantize_mmq_q8_1`, DS4). Thus this is not merely a different floating-point
reduction tree. Q2_K also has different scale grouping and sum handling.

Next Q2_K candidate pre-scales its packed two-bit weights by the four-bit
integer subscale once per weight fragment before the column loop. Maximum
value 3*15=45: no byte carry, no INT8 overflow; each integer dot product is
identical to multiplying its original result by that subscale. Retains the
same floating accumulation and minima correction. Isolated and whole-model
checks are still required to verify compiler behavior.

Pre-scaling outcome: 64 isolated cases byte-identical, 54/54 short replay
records exact. Profile Q2_K grows 36.03 -> 38.38 ms despite registers falling
106 -> 95 (local memory/thread remains zero). Not retained; private patch
`bench/mtp-q2-prescale-experiment.patch` and archived candidate binaries allow
reproduction. No serving win claimed for this candidate.

Third candidate: use one single-column reduction per (output row, activation
column), interleaving adjacent column blocks in grid.x to encourage weight
cache reuse. Still one launch, Q2_K widths 2–4 only on native SM86. Explicit
logical row count is necessary; output column stride can include padding or
the larger unsharded output, so it must not size this grid. This changes work
placement, not the dot products or the single-token reduction tree.

Independent-column outcome: all 64 isolated cases and 54 short-replay records
are exact. But Q2_K aggregate kernel time rises to 70.58 ms (baseline 36.03),
while the unchanged IQ3_S control is 93.35 vs 92.26 ms. Registers drop to 42,
with zero local memory/thread, but grid.x grows eightfold for width4 (one
row/one column instead of two rows/four columns per block). Reject: reduced
register use does not compensate for extra block work and redundant accesses.
Private patch: `bench/mtp-q2-independent-experiment.patch`.

### End of follow-up

- No new serving improvement. The fully matched short-context projection +
  attention control remains about 1.3% below native MTP at controlled clocks.
- All three trials were removed from the tracked tree. Current source and
  active Dorei binaries are restored to baseline. No MTP production fix is
  committed or shipped by this investigation.
- Archived binaries: `candidate-bin-mtp-kquant-row-first`,
  `candidate-bin-mtp-q2-prescale`, `candidate-bin-mtp-q2-independent`.
- Row-first was serving-tested in forward/reverse order, with all code
  completions identical and 71/72 acceptance throughout. Pre-scale and
  independent-column candidates were rejected at the kernel-profile stage;
  no end-to-end t/s claim is made for either.
- Clocks reset, no server/profiler left running. New small result files and
  serving records copied into the local receipts directory; full profiles
  remain on Dorei.

The useful next research direction is an MMQ-compatible canonical activation
and correction representation plus matched accumulation, not another generic
MMVQ launch-layout sweep. Matching Q2_K scales alone or merely using the same
FP16 datatype is insufficient. This is more substantial kernel work; these
results do not establish a zero-overhead solution or rule one out.
