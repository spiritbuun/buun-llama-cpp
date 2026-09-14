# Issue 124: temporary HIP stall diagnostics

Branch: `diag/issue124-hip-stall`. Based on master `ed67e11b8`.
This is instrumentation, not a proposed fix. It does not change cache policy,
GPU graphs, memory limits, kernel selection, or insert device synchronization.
Logging itself can perturb scheduling and throughput. Do not benchmark this run.

## Run

Build this branch using your usual HIP configuration. Keep the same model,
MTP sidecar and server arguments that reproduce the problem. Prefer a
`RelWithDebInfo` build if collecting debugger stacks; do not strip binaries.

Prefix the server command with `BUUN_STALL_TRACE=1`, add `-lv 4`, and redirect
both stdout and stderr directly to a local regular file:

```bash
BUUN_STALL_TRACE=1 ./build/bin/llama-server <your usual arguments> -lv 4 \
  > issue124-server.log 2>&1
```

Avoid a tmux scrollback capture or a pipe to a slow consumer. The trace is
verbose; ensure adequate free disk space. Without the environment variable,
the additional trace is disabled. This variable belongs only to this temporary
branch and is not a new supported performance setting.

Record the exact build commit, local diff, model filenames, complete command,
ROCm version, kernel version and available system RAM. Keep all original server
messages. The diagnostic markers do not contain prompt text, but ordinary
verbose server logs may contain application data: review before sharing publicly.

## What the markers mean

`STALL` lines contain `begin`/`end`, native Linux thread ID, scope ID, phase,
owner pointer, three phase-specific integers, monotonic microseconds and elapsed
microseconds. Pair on `(tid, id, owner, phase)`; do not infer pairing from adjacency
because different threads and nested scopes interleave. `end` means scope exit,
not success; it is also emitted during exception unwinding. Check ordinary errors
and return-path logs too. Idle/condition-variable waits are not automatically bugs.

| Phase | Integers a / b / c |
|---|---|
| `target.decode` | batch tokens / first token position / whether output is requested |
| `target.sync`, `spec.process` | batch tokens |
| `spec.draft` | participating slots |
| `checkpoint.stage`, `.publish`, `.finalize` | slot / task / frontier tokens |
| `checkpoint.target_copy`, `.draft_copy` | slot / task / byte count |
| Backend name, e.g. `CPU`, `ROCm0` | graph node count |
| `gpu.stream_sync` | owner identifies backend context |
| `queue.yield` | owner identifies server queue |
| `idle_capture.batch` | readiness-only boolean |
| `idle_capture.join` | allow-publication / task-arrival booleans |
| `artifact.prepare`, `.transfer`, `.publish` | owner identifies artifact store |
| `artifact.projected_batch` | manifest count / byte limit |
| `artifact.projected_publish` | publication count |
| `moe.fill_job` | pool / cache slot / bytes |
| `moe.host_copy`, `.copy_sync`, `.copy_blocking` | bytes |

Backend compute is NOT necessarily GPU completion: GPU work can be queued
asynchronously, whereas CPU backend execution can finish within the call.
`gpu.stream_sync` wraps only the existing explicit backend wait, not every
implicit wait inside HIP/rocBLAS. A final unmatched `target.decode` therefore
needs a thread stack; it alone does not identify the stalled kernel or driver.

## While the stall is happening

Do not kill the server immediately. Record its PID and collect:

1. Two all-thread stack snapshots, several seconds apart. With GDB installed,
   replace `PID` below with the actual server PID:

   ```bash
   sudo gdb -q -nx -batch -iex 'set auto-load off' \
     -ex 'set pagination off' -ex 'thread apply all bt 20' -ex detach \
     -p PID > issue124-stacks-1.txt 2>&1
   ```

   Repeat with `issue124-stacks-2.txt`. Attaching briefly pauses all threads;
   capture system metrics first. If attach stalls, report that rather than
   repeatedly attaching. Stack traces can reveal paths/arguments; review them.

2. CPU activity per thread (`top -H -p PID`), system RAM/swap and paging
   (`vmstat 1`), and GPU utilization/VRAM/GTT usage using your AMD monitoring tool.
   Include disk activity if expert weights are memory-mapped or swap is active.
3. Kernel messages covering the incident:

   ```bash
   sudo journalctl -k --since '15 minutes ago' > issue124-kernel.log
   ```

Send the direct server log, both stacks, metrics and build/config provenance.
The goal is to distinguish checkpoint/capture CPU work, host paging, model
evaluation, queue waits and HIP/driver waits. Change one feature at a time only
after the trace identifies a candidate; disabling several features upfront may
hide the original problem.

## Qualification

- HIP/gfx1201 server build passed on Dorei (2026-09-14).
- RX 9070 smoke on Toji, Qwen3.5-2B Q4_K_M, dynamic VBR/T4, 4096 context:
  trace-off and trace-on runs produced identical 32-token greedy output.
- Trace-off emitted no markers. Trace-on emitted 1354 markers, all begin/end
  pairs matched, including target evaluation/sync, CPU/HIP backends, checkpoint
  staging/copy/publication/finalization, projected artifact capture/publication
  and queue yields.
- This smoke did not exercise MTP or MoE-cache worker phases. The original
  reported large-model stall has not been reproduced locally; no claim that
  this branch fixes it or reproduces its performance.
