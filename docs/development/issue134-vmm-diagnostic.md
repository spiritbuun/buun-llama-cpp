# Issue #134: Windows long-context VBR mapping diagnostic

Branch: `diagnostic/issue134-vmm`, based on master `08826ad6e`.
This is **instrumentation, not a fix**, and is not intended for merging into master.
It retains the original mapping calls, synchronization, budget decisions, retries,
and fatal-error policy. Do not use diagnostic runs as performance benchmarks.

## First run: reproduce without changing the memory policy

Use the same Windows machine, driver, model files, and full failing request.
First stop/unload the existing model server so two copies do not compete for VRAM.
Do not run through a proxy that automatically restarts the crashed server.
Leave other applications as they were during the original reproduction.

From your normal checkout, create a separate worktree:

```powershell
git fetch origin diagnostic/issue134-vmm
git worktree add ../buun-issue134 origin/diagnostic/issue134-vmm
cd ../buun-issue134
```

Build with the same generator/toolchain you normally use. For example, from your
CUDA/Visual Studio developer shell (both reported GPUs are SM120):

```powershell
cmake -S . -B build-issue134 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_BUILD_TESTS=ON
cmake --build build-issue134 --config Release -j 8 --target llama-server test-vbr-diagnostic
ctest --test-dir build-issue134 -C Release -R '^test-vbr-diagnostic-' --output-on-failure
```

Use the newly built executable **with its newly built DLLs**. Do not copy just
`llama-server.exe` over an older installation. The path below is for a typical
multi-configuration build; a Ninja build may use `build-issue134/bin/llama-server.exe`.

```powershell
.\scripts\issue134-vmm.ps1 `
  -Server .\build-issue134\bin\Release\llama-server.exe `
  -Model 'D:\models\Qwen3.8-27B-NVFP4' `
  -Projector 'D:\models\Qwen3.8-27B-NVFP4' `
  -Draft 'D:\models\Qwen3.8-27B-DFlash2-Q8_0.gguf'
```

Send the **same approximately 195,198-token request** to the server at
`http://127.0.0.1:8099` using your existing client. The script does not generate or
upload a prompt. It uses the issue's reported flags, plus localhost binding,
port 8099, `-lv 5`, and `GGML_VBR_DIAG=1`. `-Port` can change the port.
If the original command had additional relevant options, use
`-ExtraArguments @('--alias', 'your-model-name')` and tell us what was added.

Do not set `CUDA_LAUNCH_BLOCKING`, change headroom, disable graphs, or remove the
drafter/projector for this first run. The script does not change those settings;
selected existing environment overrides are recorded in `command.json`.

The server's failure is still expected to terminate it. The script retains the
logs and does **not** restart it. After a successful response, stop the server
with Ctrl+C. The script stops only its own GPU-monitor process.

## What to send back

The script prints a unique `issue134-Baseline-*` directory containing:

- `server.log`: full stdout/stderr, including `VBR_DIAG_BEGIN` / `VBR_DIAG_END` snapshots.
- `gpu.csv`: one-second per-GPU memory/utilization readings, when `nvidia-smi` is available.
- `nvidia-smi.txt`: driver/device inventory.
- `command.json`, `version.txt`, `result.json`: arguments, selected environment,
  executable hash, version, and exit status.

Send the complete directory after reviewing it for private data. The **new
recorder contains no prompt text or tensor contents**, but existing verbose
server logs, file paths, or additional launch arguments may contain sensitive
information. Redact those as necessary without removing VBR diagnostic lines.
Please confirm the client-side prompt token count and whether the failure stayed
at the same position. A missing `gpu.csv` should be reported, not treated as a
successful telemetry capture.

## Follow-up arms (do not run a whole campaign yet)

After reviewing the baseline, we will choose the most useful next test. The same
script supports exactly one arm per invocation:

- `-Mode NoCheckpoints`: changes only `--ctx-checkpoints` from 2 to 0.
- `-Mode Static`: changes dynamic VBR to static K=t8 / V=t4; checkpoints remain 2.

Use the **same full request and context size** for either arm. The previously
reported 169K static success is below the approximately 175K dynamic failure
boundary and cannot by itself establish that the full request fits.

There is no automatic retry, extra synchronization, or headroom experiment in
this branch. Those would be separate, explicitly chosen probes if the baseline
evidence calls for them.

## Maintainer notes / interpretation

`GGML_VBR_DIAG=1` enables a process-wide recorder in ggml-base, exported so the
server, llama, and CUDA backend DLLs all write the same history. Three independent
rings retain 128 mapping, 128 pressure, and 128 state events, each with a 1024-byte
message bound. Dumps report overwritten counts and truncated messages explicitly.
Sequence numbers merge the rings chronologically. Timestamps are monotonic
microseconds, not wall-clock timestamps.

- Mapping events identify the pool, logical/physical device, VA base, chunk,
  granularity, request range, mapped bytes, and residency epoch. A growth-owner
  event connects the pool/range to the controller, tensor, K/V side, type, and
  watermark. A failed driver call records the exact original numeric result and
  operation; earlier calls in that page's straight-line create/map/release/access
  sequence had succeeded. No CUDA call is added to collect failure telemetry.
- Pressure events include the original free/total-memory observations used by
  the controller, headroom, nominal/effective limits, and pending-unmap ranges.
  `effective_cached` is labeled with its sampling stamp: it may be stale on the
  stable fast path. Diagnostics do not force a new budget sample or alter its
  memoization. Pending bytes are requested ranges, not a claim of immediately
  reclaimable physical memory. GPU polling is supplementary and coarser.
- State events retain checkpoint completion, retier freeze scopes, degradation,
  deferred-unmap activity, and recurrent-position discontinuities. Existing
  `-lv 5` output supplies additional context.
- A sparse snapshot is emitted every 64 VBR prepare boundaries. The CUDA fatal
  handler dumps the rings **before** its original device query/logger/abort.
  The existing recoverable `cuMemCreate` failure also dumps before returning.

All recorder operations are host-only. Enabled tracing still adds bounded
formatting/locking and periodic file I/O, so a disappearing failure is not proof
of a fix. The recorder tests check gating, concurrent writers, wraparound,
cross-channel retention, truncation, ordering, and a simulated failure entry.
They are **not** a reproduction of the reporter's GPU/driver failure.

## Validation before handoff

- Built `llama-server`, `test-vbr-vmm`, and `test-vbr-diagnostic` on Dorei
  (Linux, RTX 3090). Recorder enabled/disabled and the CUDA VMM test all passed.
- A separate host-side fatal fixture called the real CUDA fatal handler with a
  simulated error. The shared history and its end marker reached stderr before
  the original abort (exit 134). This validates reporting, not the driver fault.
- A real Qwen 27B request processed 2,408 prompt tokens and generated 8 tokens
  successfully with dynamic VBR, a 16 MiB test budget, and t4 floor. The trace
  contained budget samples, effective limits, and degradation records.
- Cross-compiled and linked the recorder as a Windows DLL plus an importing test
  executable using clang-cl/MSVC headers. Native Windows CUDA execution remains
  untested; the reporter's baseline is still needed.
- PowerShell 7 launcher smoke tests on Linux verified spaced paths, argument
  forwarding, stdout/stderr capture, recorded exit code 42, and the baseline /
  no-checkpoint / static arm flags. Native Windows PowerShell 5.1 execution has
  not been tested.
