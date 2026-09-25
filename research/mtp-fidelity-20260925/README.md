# MTP fidelity investigation checkpoint

Research backup, **not a production fix**. Base: `0b2789f23f5ac10ed9a99aeb40ed10d3ba1c0163`.
No changes to production dispatch, acceptance, kernels, or build targets are
enabled by this checkpoint. Nothing here is included automatically in a build.

## Current result

On the tested RTX 3090 / mixed-quant Qwen3.8-27B model, greedy plain decoding
and MTP diverge because target arithmetic changes with batching. Holding small
projections on MMVQ and F16 D256 attention at a fixed 4x8 tile eliminates the
observed divergence: short prose and WikiText teacher-forced probes have exact
logits, and the tested code/prose/arithmetic serving completions match.

This costs about **1.3% MTP decode throughput** in the clock-controlled code
test. Several attempts to recover that cost were neutral or slower; none was
retained in production. This is neither a general losslessness guarantee nor
a diagnosis of the original P100/Q6_K task-quality report. FLA prefill with
versus without recurrent rollback is a separate arithmetic boundary.

## Inventory

- [Fidelity ledger](MTP_FIDELITY_20260925.md): original reproduction, sampler
  audit, shared-history logits, first-divergence tracing, serving controls.
- [Performance ledger](MTP_CONSISTENT_PERF_20260925.md): clock controls,
  experiments, negative results, profiles, remaining work.
- `bench/`: source for probes, diagnostic preloads, serving/sweep harnesses,
  and explicitly rejected experiment patches. Apply patches individually to
  the base, not cumulatively. The bias-v4 patch is the final rejected variant;
  the ledger explains the earlier variants. The neutral row-first trial is
  described in the ledger, not retained as a source patch.
- `results/measurements.jsonl`: 776 extracted fidelity/performance records,
  each labeled with its original log filename. These are not newly rerun tests.
- `results/*-kernels.csv`: Nsight kernel summaries for projection isolation
  and the final three trials.
- `results/teacher-tokens.txt`: prompt plus generated token IDs used to
  reproduce the short prose disagreement without retokenizing its text.

The ledgers and scripts are historical snapshots. References to “private”,
“not committed”, the research hub, and machine paths describe their original
execution, before this backup commit. Their `bench/` references resolve here;
full original logs remain in the research hub's
`knowledge/server-resume/apollo-20260925-results/dorei/` and the remote bench
directory recorded in the ledgers. Model weights, binary libraries, full
Nsight captures, and giant logits are deliberately not committed.

## Reproduce the consistency control

Linux/CUDA diagnostic only: exported C++ symbol interposition is tied to this
revision. Tested CUDA 13.3, SM86, F16 KV, 24 query heads / 4 KV heads, D256.
The lean preload aborts if its expected specialization is missing. Do not use
the tracing preloads for performance measurements: they synchronize/read back
tensors; the evaluation-callback tracer can also change graph fusion.

From the repository root, with a shared-library CUDA build at `build-cuda`
and `llama-server` built (SM86, Turbo FA enabled; all-quants FA was disabled):

```sh
MTP_REPO="$PWD"
MTP_RESEARCH="$MTP_REPO/research/mtp-fidelity-20260925"
MTP_BUILD="$MTP_REPO/build-cuda"
MTP_MODEL=/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf

g++ -std=c++17 -O2 -Iinclude -Icommon -Iggml/include \
  "$MTP_RESEARCH/bench/mtp-batch-fidelity.cpp" \
  -L"$MTP_BUILD/bin" -Wl,-rpath,"$MTP_BUILD/bin" \
  -lllama-common -lllama -lggml -lggml-base -ldl -pthread \
  -o "$MTP_BUILD/mtp-batch-fidelity"
g++ -std=c++17 -O2 -fPIC -shared -Iggml/include \
  "$MTP_RESEARCH/bench/mtp-fixed-fa.cpp" -ldl \
  -o "$MTP_BUILD/mtp-fixed-fa.so"

GGML_CUDA_MMVQ_MAX_N=8 MTP_FIXED_FA_COLS=4 \
LD_LIBRARY_PATH="$MTP_BUILD/bin" LD_PRELOAD="$MTP_BUILD/mtp-fixed-fa.so" \
MTP_PROBE_PREFIX=31 MTP_PROBE_ROWS=64 \
MTP_PROBE_TOKENS="$MTP_RESEARCH/results/teacher-tokens.txt" \
"$MTP_BUILD/mtp-batch-fidelity" -m "$MTP_MODEL" \
  -ngl 99 -fa on -c 4096 -b 512 -ub 512 -np 1 -ctk f16 -ctv f16
```

Compare against the same command with both arithmetic controls/preloads unset.
The successful consistency control must emit **54** `BATCH_FIDELITY` records,
all with `changed=0` and `mean_kld=0`. Exit status alone does not assert
equivalence; a completed probe may intentionally report nonzero differences.
`MTP_PROBE_PROFILE=1` instead restricts the run to one configuration, one repeat,
widths 1/4, and brackets width4 decoding with the CUDA profiler API.

For actual serving, use a fresh output directory for each run:

```sh
GGML_CUDA_MMVQ_MAX_N=8 MTP_FIXED_FA_COLS=4 \
LD_LIBRARY_PATH="$MTP_BUILD/bin" LD_PRELOAD="$MTP_BUILD/mtp-fixed-fa.so" \
python3 "$MTP_RESEARCH/bench/mtp-serving-fidelity.py" \
  --bin "$MTP_BUILD/bin/llama-server" --model "$MTP_MODEL" \
  --out "$MTP_BUILD/mtp-serving-consistent"
```

The harness records exact commands, prompts, token IDs, responses, and timings.
Use `--prompt code --repeats 6` for the timing screen. Sweep scripts retain the
original remote bench layout and model paths: adapt those before reuse, and
set `LD_LIBRARY_PATH` explicitly when comparing archived libraries. Run arms
serially without concurrent compilation. The controlled 1200 MHz numbers are
relative comparisons, not full-speed headline throughput. Any clock lock must
be reset with `nvidia-smi -rgc` afterwards, including on failure.

## Remaining work

Reconcile MMQ activation scale grouping, minimum-correction sums, and
accumulation with the single-token path while retaining its speed. Matching
only Q2_K scale grouping is insufficient: Q4_K independently causes a flip.
Do not resume the already unsuccessful generic MMVQ block-layout sweep without
a new reason. Deep context, other quants/backends, and production integration
remain unqualified.
