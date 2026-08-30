# buun-llama-cpp

![buunslamma](buunslamma.png)

> **This is a highly experimental fork of llama.cpp. Use at your own discretion.**

A research and development fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), providing unique KV cache codecs, inference techniques, and bleeding edge features.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## VBR — Variable Bit-Rate KV Cache

Why pay 3-bit or 4-bit quality for a context length you only sometimes reach? VBR quantizes the KV cache
*dynamically as your session grows* — giving you the highest-quality cache possible at any given depth.
It quantizes the cache **layer by layer, using our best KV codecs**, from the least-sensitive layers to
the most-sensitive, and only as far down the ladder as your VRAM and context actually require.

The cache starts at **FP16** and stays there until there is real budget pressure; then it degrades one
(layer, side) tensor at a time — first to an aggregate 15.75 bits/value, then 15.51, and so on down the
ladder (`f16 → turbo8 → turbo4 → turbo3_tcq → turbo2_tcq → turbo1_tcq`), following a per-model price
order measured on KLD panels. For Qwen35 (16 attention layers) that is **160** distinct steps from FP16
down to turbo1_tcq; for Gemma4-31B (60 layers), **600** — the finest-grained quality control we can give
you at every moment of a session.

And because VBR always draws from the best codecs on the ladder, **you never have to track KV formats
again**: new codecs and research roll straight into VBR, so the default cache will always use the best
available ladder.

### Recommended usage

On a dedicated GPU, just run:

```sh
llama-server -m model.gguf
```

VBR is the default cache, with a **turbo4 quality floor**. It derives a KV VRAM budget from whatever is
left after weights and compute, advertises the largest context that fits without going below turbo4
(capped at the model's training length), and degrades tiers on the fly as context fills. The cache is
still FP16 until memory pressure actually requires compression.

For maximum context, explicitly use `-ct vbr`. That deliberate opt-in opens the complete ladder down to
turbo1_tcq unless you also set `--vbr-floor`:

```sh
llama-server -m model.gguf -ct vbr
```

Run with `-v` to watch the `VBR degrade #…` steps fire.

### Choose the two things you know

VBR balances three quantities: **context length**, **KV VRAM**, and **minimum quality**. Usually specify
at most two and let VBR solve the third:

| You know | Use | VBR determines |
|---|---|---|
| How many tokens you need and how much KV VRAM you can spare | `-ct vbr -c N --vbr-vram SIZE` | The highest-quality terminal layer mixture that fits. Explicit `-ct vbr` leaves the full ladder available. |
| How many tokens you need and the lowest quality you will accept | `-c N --vbr-floor TIER` | The safe KV VRAM budget available on the machine. |
| How much KV VRAM you can spare and the lowest quality you will accept | `--vbr-vram SIZE --vbr-floor TIER` | The largest fillable context. |

Setting all three usually just over-constrains the same calculation. Pick the two requirements you
actually know; VBR can then optimize the remaining dimension instead of receiving three potentially
contradictory answers.

### Settings

| flag | meaning |
|---|---|
| `-ct vbr` (or `-ctk vbr` / `-ctv vbr`) | VBR is already enabled by default. Explicitly selecting it opens the full ladder to t1 when no `--vbr-floor` is supplied. Explicitly pinning a side (`-ctv q8_0`) holds it at fixed bits and never degrades it. Use `-ct f16` or another concrete type to opt out of VBR. |
| `-c <N>` | Cap the context at N tokens; VBR then spends your whole VRAM budget running *that* window at the highest quality it can, instead of advertising the max floor-tier capacity. E.g. `-c 30000` = the best-quality cache that fits a 30k window. |
| `--vbr-vram <SIZE>` | Explicit KV VRAM budget (e.g. `8G`). Default `auto` = whatever VRAM is left after weights and compute. |
| `--vbr-floor <bits\|tier>` | Literal aggregate bits/value floor for dynamic mode. Implicit VBR defaults to t4 (4.125); explicit `-ct vbr` without this flag uses t1 (1.25). Degrades stop at the last step still ≥ the floor. |
| `--vbr-budget <tier\|number>` | Default `dynamic` (runtime controller). A tier (`t8/t4/t3/t2/t1`) or a number instead selects a **fixed** static tier — no runtime degrades. |

**Requirements:** a CUDA or ROCm backend (turbo-typed KV needs the TurboQuant interface; layers whose KV
lands on the CPU fall back to q8_0). Flash attention is required and force-enabled. Dynamic mode uses
unified KV (forced automatically with `-np > 1`). Context-shift / self-extend and slot/session
save-restore are disabled in dynamic mode (they would snapshot tier-typed KV that can't restore across a
degrade — tier-aware save-restore is planned); context checkpoints stay enabled on hybrid models. Generation stops cleanly when the context
fills. Models without a baked price order use a generic cross-model order.

## TCQ (trellis-coded KV cache)

Standard KV cache quantization treats each value independently. TCQ constrains the quantization
indices to follow a 512-state trellis, giving a much larger effective codebook at the same bit rate
(with FWHT rotation and context-adaptive norm scaling on top). At the 3-bit setting, the trellis cuts
median KL-divergence by **~40% versus scalar quantization** while using slightly *fewer* bits
(3.25 vs 3.50 bpv), and lands perplexity on par with an f16 KV cache.

**Paper**: [Closing the Gap: Trellis-Coded Quantization for KV Cache at 2-3 Bits](https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache)

### Quality (Qwen3.6-27B Q6_K, 16K, RTX 3090)

`turbo3` is scalar 3-bit, `turbo3_tcq` the trellis-coded version at fewer bits — the gap is exactly what
the trellis buys. `pp` = prefill t/s on a 16K prompt, `tg` = decode t/s at 16K depth.

| Codec | bpv | PPL | median KLD | pp (t/s) | tg (t/s) |
|-------|-----|-----|------------|----------|----------|
| turbo3 (scalar) | 3.5 | 5.730 | 0.00279 | 1015 | 28.1 |
| turbo3_tcq | 3.25 | 5.668 | 0.00163 | 818 | 26.6 |

The trellis cuts median KLD ~40% (and edges PPL below f16) at fewer bits. Its cost lands in **prefill** —
the Viterbi re-encode drops prompt processing to ~818 t/s vs ~1015 for scalar turbo3; decode is barely
affected on a dedicated GPU.

### Speed

The trellis cost is compute, so it depends on hardware. On dedicated GPUs (e.g. RTX 3090) the fused
tensor-core decode path keeps generation at essentially vanilla / f16 speed (the `tg` numbers above). On
weaker-compute hardware (e.g. the Strix Halo iGPU) the cost is exposed and the TCQ types can be up to
~40% slower than their scalar counterparts — there, prefer the scalar or higher-bit codecs.

### How it works

1. **FWHT rotation** with random sign flips converts correlated KV vectors into i.i.d. Gaussian entries
2. **Viterbi encoding** on a 512-state (3-bit) or 256-state (2-bit) right-shift trellis finds the globally optimal codeword assignment
3. **O(1) sliding-window decode** -- each value decodes via a bit window lookup, no trellis traversal at inference
4. **Context-adaptive alpha** -- logarithmic norm scaling formula automatically adjusts dequantization scale per context length

### Custom codebooks

Trained codebooks are included in `codebooks/`. The defaults are compiled into the CUDA kernels, but you can override them:

```sh
TURBO_TCQ_CB=codebooks/3bit/product_aware_iter080.bin \
TURBO_TCQ_CB2=codebooks/2bit/product_aware_iter090.bin \
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo3_tcq -ctv turbo3_tcq
```

Codebook training scripts are in `scripts/tcq_train_*.py`.

## Codecs

These are the individual KV cache codecs the fork ships. In practice you rarely pick one by hand —
**dynamic VBR mixes them per layer automatically and is the recommended default**; the codecs below are
mainly for testing and comparison.

Measured on Qwen3.6-27B Q6_K at 16K context (18 chunks of the wikitext-2 test set) on an RTX 3090.
Median KL-divergence is versus an f16 KV cache (0 = identical logits); `pp` is prefill throughput on a
16K prompt and `tg` is decode throughput at 16K depth. Ordered by KL-divergence (quality) — lower is
better.

| Codec | bpv | PPL | median KLD | pp (t/s) | tg (t/s) |
|-------|-----|-----|------------|----------|----------|
| f16 | 16.0 | 5.683 | 0 (ref) | 1104 | 30.7 |
| q8_0 | 8.5 | 5.698 | 0.00020 | 1050 | 27.4 |
| turbo8 | 8.125 | 5.693 | 0.00020 | 1013 | 28.6 |
| turbo4 | 4.125 | 5.723 | 0.00090 | 1017 | 28.4 |
| q4_0 | 4.5 | 5.705 | 0.00115 | 1033 | 26.9 |
| turbo3_tcq | 3.25 | 5.668 | 0.00163 | 818 | 26.6 |
| turbo3 | 3.5 | 5.730 | 0.00279 | 1015 | 28.1 |
| turbo2_tcq | 2.25 | 5.711 | 0.00561 | 985 | 28.5 |
| turbo2 | 2.5 | 5.964 | 0.01083 | 1013 | 28.2 |
| turbo1_tcq | 1.25 | 6.012 | 0.02633 | 991 | 28.5 |

## Speculative decoding + Vision (`--mmproj-gpu-swap`)

On VRAM-constrained GPUs, a speculative context and the vision encoder (mmproj) may not fit in VRAM simultaneously. For example, Qwen3.6-27B Q6_K + MTP uses ~22.6 GiB on a 24 GiB RTX 3090, leaving no room for mmproj's ~1.1 GiB GPU footprint.

`--mmproj-gpu-swap` solves this by keeping mmproj on CPU at startup, then temporarily unloading MTP or an external `draft-dflash` sidecar when an image arrives, loading mmproj to GPU for fast encoding (~1-2s instead of 30-60s on CPU), and restoring speculation afterward. MTP is recreated from the target model; an external DFlash sidecar is reloaded from its GGUF.

```sh
./build/bin/llama-server -m Qwen3.6-27B-Q6_K.gguf \
  --mmproj mmproj.gguf --spec-type draft-mtp \
  --mmproj-gpu-swap -ngl 99
```

When combined with auto-fit (no `-c` flag), the server automatically sizes context to leave room for the swap. With a single slot, it also auto-enables `--kv-unified` to avoid splitting the KV cache into separate streams, which doubles usable per-slot context.

## DeepSeek V4 Flash and Qwen3.8 Flash Next — MoE cache

DeepSeek V4 Flash is much larger than a typical consumer-GPU model, but its routed experts can stay
in system RAM while the hot expert tensors are cached in spare VRAM. On a multi-GPU machine, use
layer splitting: tensor splitting was substantially slower for this CPU-expert workload even with
NVLink.

### Choose the MoE cache mode

Use `--moe-cache auto` with two or more GPUs. Automatic mode is deliberately conservative and
requires two eligible devices. On a single GPU, use `--moe-cache on`; otherwise the cache remains
off and most of the card can sit unused.

### Target model only (two or more GPUs)

```sh
./build/bin/llama-server \
  -m DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 99 -sm layer -fa on -c 8192 -np 1 -ub 4096 \
  -ctk f16 -ctv f16 \
  -ot 'exps=CPU' --moe-cache auto \
  --moe-cache-expert-parallel auto \
  --host 0.0.0.0 --port 8081
```

`-ot 'exps=CPU'` leaves the routed experts in system RAM while keeping the remaining offloaded
weights on GPU. `--moe-cache auto` then fills otherwise spare VRAM with the hottest expert tensors
and adapts their residency as routing changes. Expert-parallel mode divides resident rows within a
layer across the selected cache devices while the CPU computes misses. `auto` uses both devices on
a dual-GPU host and caps larger hosts at three-way dispatch. For one GPU, change the cache mode to
`--moe-cache on` and omit expert parallelism.

### Dual RTX 3090 + DSpark

```sh
GGML_CUDA_MOE_CACHE_RESERVE_MB=1024 \
./build/bin/llama-server \
  -m DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf \
  -md dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
  -ngl 99 -sm layer -fa on -c 8192 -np 1 -ub 4096 \
  -ctk f16 -ctv f16 \
  -ot 'exps=CPU' --moe-cache auto \
  --moe-cache-expert-parallel auto \
  --spec-type draft-dspark -ngld 0 -otd 'exps=CPU' \
  --spec-draft-n-max 3 --spec-draft-p-min 0 \
  -t 22 -tb 22 -td 22 -tbd 22 \
  --host 0.0.0.0 --port 8081
```

Without expert-parallel dispatch, the best measured dual-RTX-3090 configuration for the IQ2_M
target averaged about **41.1 tokens/s** on a 24-core EPYC 7443. With complete miss-row accounting,
the command above produced seven warm 500-token completions at **50.26--52.28 tokens/s**
(**51.48 mean**). A 2 GiB reserve reduced the mean to **49.94 tokens/s** but leaves more allocation
headroom. The expert-parallel cache defaults to admitting up to 16 entries per node and requires 40
fresh misses before replacing a resident entry; these settings reduce expensive cache churn in
both one- and two-slot testing. Two concurrent DSpark slots averaged **59.20 aggregate tokens/s**
over four warm 1,000-token waves and completed every response cleanly.

The 1 GiB reserve is an aggressive performance setting, not the global default. The thread count is
also machine-specific; using all 24 physical cores reduced this host sharply. Prompt processing
measured **326 pp/s at 2,048 tokens** before expert parallelism was added.

On four or more eligible GPUs, `auto` selects three-way dispatch. Fanout and CPU-thread optima are
host-specific, so compare fanouts two, three, and four rather than assuming every card should join
each expert operation.

Although the command requests `-ngld 0`, `--spec-type draft-dspark` lets the server recognize the
CPU-backbone DSpark configuration before model loading. The default GPU assist keeps the large
draft experts on CPU while placing all lightweight draft layers and the Markov/output tail in about
594 MiB of GPU memory. This raised the corrected, cache-tuned dual-3090 result from 48.7 to 51.5
tokens/s. Use `--no-spec-dspark-gpu-assist` when that allocation is more valuable as KV capacity;
use `--spec-draft-device none` to keep the entire drafter on CPU.

### Single RTX 3090 + DSpark

```sh
./build/bin/llama-server \
  -m DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf \
  -md dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
  -ngl 99 -sm layer -fa on -c 8192 -np 1 -ub 4096 \
  -ctk f16 -ctv f16 \
  -ot 'exps=CPU' --moe-cache on \
  --spec-type draft-dspark -ngld 0 -otd 'exps=CPU' \
  --spec-draft-n-max 2 --spec-draft-p-min 0 \
  -t 20 -tb 20 -td 20 -tbd 20 \
  --host 0.0.0.0 --port 8081
```

On an RTX 3090 with a 24-core EPYC 7443, this configuration averaged **31.5 tokens/s** after
warmup. Target-only inference with the same forced cache averaged about 24.0 tokens/s. Depth two
slightly beat depth three and four; depth five was slower. Twenty CPU threads beat 12, 16, 24 and
32 on this host, but the ideal count is machine-specific—benchmark around your number of physical
cores while leaving capacity for cache service and server work. Prompt processing measured
**333 pp/s at 2,048 tokens**.

The CUDA MoE cache normally keeps a 3 GiB safety reserve. Advanced users can try a 2 GiB reserve:

```sh
GGML_CUDA_MOE_CACHE_RESERVE_MB=2048 ./build/bin/llama-server ...
```

That raised the tested single-GPU result only slightly, from 31.5 to 31.8 tokens/s. The dual-GPU
headline above uses 1 GiB, but 2 GiB is the safer starting point. Do not eliminate the reserve:
CUDA graphs, workspaces and transient allocations still need headroom.

### Tuning on another host

Change one group at a time and restart the server between configurations. Use the same prompt,
temperature and output length throughout; discard the first completion after each load and compare
at least three warmed 512-token completions. Record both generation speed and accepted/drafted token
counts—a faster result caused only by a luckier generation is not a reliable configuration win.

1. Select the cache mode first: `--moe-cache on` for one GPU, `--moe-cache auto` for two or more.
2. On two or more GPUs, try `--moe-cache-expert-parallel auto`. The option is deliberately not the
   default: it changes cache placement and the CPU/GPU numerical path, and models other than the
   tested DeepSeek V4 Flash IQ2_M may have a different working-set tradeoff. Compare it against 0.
3. Keep the safe 3 GiB reserve and sweep CPU concurrency. Set all four pools together with
   `-t N -tb N -td N -tbd N`. Start around physical cores minus 8, minus 4, minus 2, and all
   physical cores; avoid assuming SMT threads help.
4. With the best thread count, sweep `--spec-draft-n-max 2`, `3`, and `4`. Depth five was already
   clearly worse on the tested host.
5. Only then try cache reserves of 2560 and 2048 MiB using
   `GGML_CUDA_MOE_CACHE_RESERVE_MB`. On a dedicated, well-characterized host, 1536 and 1024 MiB are
   additional performance points. Keep the larger reserve unless the smaller value wins
   repeatedly, and verify long generation without an allocation failure.
6. Recheck the winner in reverse order against the original configuration to catch temperature,
   power and host-load drift.

The reported PP figures used `-ub 4096`, five distinct 2,048-token prompts per server, no reusable
prefix, and one discarded cold prompt. Two server loads were run in reverse single/dual order; the
eight warmed measurements averaged 332.8 pp/s on one 3090 and 326.3 pp/s on two. Layer splitting
helps decode but adds device handoffs during the already-efficient large-batch prefill path, so a
second 3090 did not improve PP in this configuration.

The tested IQ2_M target plus Q8_0 DSpark sidecar occupied **95.3 GiB of process RSS/PSS**, including
about 94.4 GiB of file-backed model data. **128 GiB of system RAM is recommended.** Around
100–104 GiB usable is the practical floor for keeping this working set resident; a nominal 96 GiB
machine will rely on reclaim/reload or swap and can slow down sharply. More slots, `--no-mmap`, and
other resident models require additional headroom. Context length, KV type, host memory bandwidth,
CPU threads, and expert-cache hit rate all affect the final speed.

## Build

### NVIDIA (CUDA)

```sh
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

### AMD (ROCm / HIP)

Tested on ROCm 7.2 + RDNA3 (`gfx1100`, RX 7900 XTX). Other RDNA3/RDNA4 targets should work by swapping `AMDGPU_TARGETS`.

```sh
cmake -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++

cmake --build build -j$(nproc)
```

#### [For Windows ROCM](https://github.com/lemonade-sdk/llamacpp-rocm/blob/main/docs/manual_instructions.md)
Test on ROCm 7.13 + AMD Radeon AI PRO R9700

## Recommended configurations

**For most use, VBR needs no cache flags** — it picks the best codec per layer automatically, spends
all your spare VRAM on quality, and keeps a t4 quality floor:

```sh
./build/bin/llama-server -m model.gguf -ngl 99
```

Use explicit `-ct vbr` to open the complete ladder to t1 for maximum context. The fixed-tier codecs
below are for pinning a specific tier — benchmarking, a fixed budget, or a backend without VBR support.

### turbo4 (4.125 bpv) -- low divergence, great compression

The safe fixed-tier default. 4-bit KV cache at ~3.8x compression with no speed penalty — higher fidelity (lower KL-divergence from FP16) than the 2–3 bit codecs below.

```sh
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo4 -ctv turbo4
```

### 3-bit TCQ (3.25 bpv) -- best quality at 3-bit

The best-quality 3-bit option — about 40% lower KL-divergence than scalar turbo3 at the same tier. ~5x KV cache compression.

```sh
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo3_tcq -ctv turbo3_tcq
```

### 2-bit TCQ (2.25 bpv) -- maximum compression

~7x KV cache compression. Best for fitting very long contexts in limited VRAM.

```sh
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo2_tcq -ctv turbo2_tcq
```

### Asymmetric 2.75 bpv -- best 2-bit quality

3-bit keys + 2-bit values. 15-17% lower KLD than the reverse, because adaptive alpha already compensates V quantization error.

```sh
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo3_tcq -ctv turbo2_tcq
```

### Scalar turbo3 / turbo2 (3.5 / 2.5 bpv) -- no trellis

Scalar quantization without TCQ. Faster encode, worse quality than TCQ equivalents.

```sh
# 3-bit scalar
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo3 -ctv turbo3

# 2-bit scalar
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo2 -ctv turbo2
```

---

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
