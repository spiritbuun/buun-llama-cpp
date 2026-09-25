# MTP fidelity exploration — 2026-09-25

Production fixes merged/pushed first: master `0b2789f23` (HIP physical VRAM
accounting); existing physical P100 meta-I/O qualification in
`APOLLO_FOLLOWUP_20260925.md`.

Investigation branch `investigate/mtp-fidelity-20260925`, no production changes.
All new probes live privately in `bench/`. Dorei RTX 3090, target
`/root/models/apollo-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` (embedded MTP).
This is NOT Apollo's P100 Q6_K quant or his unavailable agent-evaluation receipt.
Build: `/root/bench/vbr-reset-51a215d7e-20260925/build-cuda`, CUDA 13.3, sm86,
contains the memory-query fix; target arithmetic unchanged from master51a.

## Questions and controls

1. Does the proposal verifier preserve the intended target distribution?
   Code audit plus `test-sampling`: pass; no intentionally biased MTP acceptance
   found. Nonzero-temperature rejection sampling does not preserve seeded text.
2. Do target logits differ when the same token history is batched differently?
   Yes, even without a drafter or accept/reject logic. Measure this separately.
3. Does enabling recurrent rollback alter target arithmetic before any rejection?
   Yes at the tested 512-token prefill boundary; isolated to FLA dispatch below.
4. Does actual serving repeat, and where does greedy MTP first diverge?
   F16 and Turbo4 tests below; this is not an agent-task quality verdict.

## Target-only teacher-forced probe

`bench/mtp-batch-fidelity.cpp`: fresh context each arm, same 512-token WikiText
prefix, same following 128 tokens. Width 1/2/4, two repeats each. F16 KV.
Context configurations: (rollback snapshots=0, sequence capacity=1), (0,2), (3,2).
Full-vocabulary double-precision softmax/KLD; exact byte comparison separately.
Only final statistics retained, not giant logit archives.

At b/ub512, relative to rs0/seq1/width1:

| Arm | Mean KLD | p99 KLD | Max KLD | Argmax flips / 128 |
|---|---:|---:|---:|---:|
| Same-shape repeated anchor | 0 | 0 | 0 | 0 |
| rs0, seq2, width1 | 0 | 0 | 0 | 0 |
| rs0, seq1, width2 | 0.000150019 | 0.000777617 | 0.001331788 | 0 |
| rs0, seq1, width4 | 0.000253891 | 0.001280982 | 0.003146050 | 0 |
| rs3, seq2, width1 | 0.000193257 | 0.001001142 | 0.001412673 | 0 |
| rs3, seq2, width4 | 0.000306692 | 0.001203372 | 0.001942539 | 0 |

All same-shape repeats are byte-identical, not merely low KLD. This is a small
128-position diagnostic, not a broad quality panel. Large maximum absolute
logit differences in low-probability vocabulary are not the KLD tail metric.

### Causal control for the rollback-sizing difference

`build_recurrent_attn` uses K=1 without rollback and K=n_rs_seq+1 with rollback.
CUDA's FLA prefill route requires `!keep_rs`. At 512 tokens on this 3090 it is
eligible only for the no-rollback arm. Thus enabling MTP can select a different
prefill implementation before any proposal is checked.

- b/ub256: rs0/width1 and rs3/width1 become byte-identical.
- Direct diagnostic LD_PRELOAD override of `ggml_cuda_gdn_fla_ptx_supported`
  returning false, b/ub512 unchanged: rs0/width1 and rs3/width1 also become
  byte-identical. Confirmed override banner in the log.
- The width1-vs-width4 difference remains without FLA: mean 0.000277024,
  p99 0.001409472, max 0.002086194, no argmax flips on this panel.

The preload is private diagnostic code, never linked into production. These
measurements do not say which arithmetic path is more accurate against an
independent higher-precision reference, and do not justify disabling fast FLA.

## Actual greedy serving

`bench/mtp-serving-fidelity.py`: fresh plain and MTP server processes, np1,
c4096, ngl99, FA on, b/ub512, fit off, host cache off, cache_prompt=false.
Temperature0, seed1234, max_tokens192, thinking disabled, explicit no penalties.
Default MTP adaptive maximum3. Each prompt repeated twice per mode; cache_n=0
in returned timings. This avoids mixing cache reuse with the MTP question.

| KV | Code | Prose | Arithmetic | Same-mode repeats |
|---|---|---|---|---|
| F16 | identical | differs after 383 shared characters | identical | all exact |
| Turbo4 | identical | differs after 337 shared characters | identical | all exact |

Code and arithmetic outputs finish; prose hits the 192-token cap. No quality
ranking from these three prompts. Different output text is not by itself proof
of biased acceptance or worse task performance.

F16 prose with logprobs enabled reproduces the same divergence at generated
token index63: plain chooses ` assign`, MTP chooses ` prioritize`. Plain's
logprob gap is 0.0525187. The accepted MTP token has an empty top-logprobs list
and placeholder logprob0 in the API, so that response cannot establish its
actual target margin. Target-only replay of the shared prefix is the next check.

### Shared-prefix replay completed

The harness records `/apply-template` and `/tokenize`; the resulting 31 prompt
tokens equal the server's reported prompt_n. Append the plain response's token
IDs, not re-tokenized text. Diagnostic prefix=31, rows=64.

With no drafter and no accept/reject operations, row62 (predicting generated
token63) flips from token9501 (` assign`) to60445 (` prioritize`) solely when
batch width increases. Width1 favors assign by0.0686779 logit units; width2
favors prioritize by0.0547981 and width4 by0.0119858. This is the same pair and
position as the real serving divergence. The probe's margin is not identical
to the server's0.0525187: this is matched token history, not a reconstruction
of every serving graph/output-row/rollback scheduling choice.

Both sequence-capacity controls and enabling/disabling rollback snapshots are
byte-identical at each fixed width on this short-prompt case. Same-shape repeat
anchors are zero. Width4 versus width1: mean KLD0.000411719, median0.000167789,
p99 (lower empirical order statistic)0.002548342, max0.002850010, 1/64 argmax
flips. Width2: mean0.000204899, 1/64 flips.

This reproduces a concrete greedy divergence without any speculative
acceptance machinery. It does not isolate which projection/attention/recurrent
kernel causes the width-dependent difference, nor prove that rollback and
acceptance have no bugs on other inputs. Next useful diagnostic is a
layer-by-layer first-divergence trace on this saved token history; do not remove
fast kernels or change acceptance policy based only on changed text.

All probes finished; Dorei idle. No additional paid GPU rented for this work.

## Follow-up: dispatch and attention tiling isolated

Private `bench/mtp-layer-trace.cpp` initially used a backend evaluation callback.
That callback changes graph splitting/fusion and changes the logits, so its
intermediate comparisons are not production-path evidence on their own.
`bench/mtp-matmul-trace.cpp` instead interposes exported CUDA host launch
functions, synchronizes, and reads their actual inputs/outputs without splitting
the graph. Its final logits match the unobserved control. This preload is not
part of the fork or a proposed public option.

The model is a mixed-quant GGUF despite its filename. The first observed
projection discrepancy is layer 1's **Q2_K** attention-gate projection: identical
F32 input, maximum output difference 0.065710783. The SM86 MMVQ threshold is two
rows for Q2_K; four rows select MMQ. MMQ's D2S6 activation layout uses scales over
64 values, whereas MMVQ uses 32. This is an existing numerical tradeoff, not
evidence of a sampler accepting an invalid proposal. The relevant dispatch
tuning was introduced in `2479cbfddb`.

Diagnostic `GGML_CUDA_MMVQ_MAX_N=8` keeps these small projections on MMVQ. On the
same 64-position prose replay, width4/width1 mean KLD drops from 0.000411719 to
0.000204899, but the same one argmax flip remains. Disabling all fusion as well
gives 0.000159353 and no argmax flip, but changes the reference too and is not a
production fix.

With MMVQ held fixed, layers 0–2 and the first full-attention layer's Q/K/V
projections agree for the traced first token. Every query head entering layer 3
FlashAttention is identical. Attention output differs in 1497/6144 elements,
maximum 2.38418579e-7, **before** sigmoid-and-gate; this is not first introduced
by the fused gate. Downstream activation quantization amplifies it.

A second private control holds the F16 D256 attention tile at ncols1=4,
ncols2=8 for query batches of at most four, using an already compiled kernel.
Combined with fixed MMVQ dispatch, the first-token trace becomes byte-identical.
More importantly, the complete 64-position replay is **byte-identical, KLD zero**
at widths 1, 2, and 4, for all three rs/sequence-capacity configurations and both
repeats. No acceptance, rollback, weight, or quantizer-code change is needed to
make this particular teacher-forced replay invariant.

This does not prove general MTP correctness, nor that the fixed configuration
is more accurate. It establishes causes for this replay. The fixed attention
tile also changes single-token reference arithmetic.

### Actual serving control

Fresh current-master control (`mtp-serving-current-control`) again reproduces
the same prose split; code and arithmetic agree. Under both private overrides
(`mtp-serving-fixedmath`), all three prompts have identical plain/MTP output,
and both repeats agree within each mode. The plain output also agrees with the
original plain output for all three prompts. Prose now agrees across all 192
generated tokens. This exercises real speculation and rejection rather than
only the teacher-forced replay.

Preliminary server decode speeds, arithmetic mean of two requests each:

| Prompt | Plain current | Plain fixed math | MTP current | MTP fixed math |
|---|---:|---:|---:|---:|
| Code | 48.76 | 48.03 | 108.25 | 105.65 |
| Prose | 48.48 | 47.85 | 66.84 | 67.93 |
| Arithmetic | 48.29 | 47.27 | 99.78 | 96.00 |

Units t/s, same mixed-IQ3 model, F16 KV, RTX 3090. Code has identical output and
71/72 acceptance in both arms; the measured -2.4% is a useful initial cost
estimate, not a qualified production performance verdict. No reverse-order
pairing yet, the diagnostic interposer itself has some overhead, and prose
changes output/acceptance between controls. All requests are cold (`cache_n=0`).
Nothing from this probe has been made a public flag or a production default.

Reproduce from `/root/bench/vbr-reset-51a215d7e-20260925` on Dorei:

```sh
GGML_CUDA_MMVQ_MAX_N=8 MTP_TRACE_FIXED_FA_TILE=1 \
LD_PRELOAD=$PWD/mtp-matmul-trace.so \
python3 mtp-serving-fidelity.py \
  --bin "$PWD/build-cuda/bin/llama-server" \
  --model /root/models/apollo-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
  --out "$PWD/results/mtp-serving-fixedmath"
```

The private override selects tile **4×8**; `=1` above means enable, not tile
width. It only overrides F16/F16, D256 attention with at most four queries.
Use a fresh output directory when repeating. Exact server commands, prompts,
responses, and timing fields are retained in the result directories.

The follow-up WikiText replay also passes: prefix512, following128 positions,
b/ub256 (avoids the separately isolated FLA-vs-rollback prefill switch). All
widths, rs configurations, and repeats are byte-identical with both controls:
54/54 comparison records have changed=0 and KLD=0. Both this and the short-prose
panel are archived locally alongside the serving records in
`apollo-20260925-results/dorei/`. Dorei is idle; production tree remains unchanged.

Next work should assess whether batch-consistent arithmetic can retain the
fast paths, with longer contexts, other quant mixes/backends, and controlled
timing before deployment. These results do not establish the cause of Apollo's
P100/Q6_K agent-task scores or prove all MTP paths lossless.

## Outstanding evidence

- Apollo's exact weights/build/sampling/prompts and paired agent outcomes.
- No broad KLD/task-outcome conclusion and no general proof of losslessness from
  this small panel. Greedy target kernels can vary with verification batch shape.
- Accepted-token logprob reporting is a diagnostic limitation, not evidence of
  a probability-one target prediction.
