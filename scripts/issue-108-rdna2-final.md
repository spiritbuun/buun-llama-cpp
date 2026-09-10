# Issue 108 final RDNA2 campaign

This branch contains the corrected gfx1030 WGP register accounting and uses a 256 KiB
minimum HIP VMM commit granularity. A larger driver-mandated granularity remains
authoritative; the campaign records the resolved value for each GPU in `vmm-policy.tsv`.
The campaign stops at the first failed preflight or benchmark and still creates an archive,
rather than repeating the same failure across every arm.

Start from a clean checkout so the binary and source receipts cannot disagree:

```bash
git clone --branch debug/issue-108-rdna2-final \
  --single-branch \
  https://github.com/spiritbuun/buun-llama-cpp.git \
  buun-issue108-final
cd buun-issue108-final

HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ \
cmake -S . -B build-hip-gfx1030 \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1030 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-hip-gfx1030 --parallel "$(nproc)" --target \
  llama-bench test-cuda-rdna2-vmm-policy test-vbr-vmm
```

Run the campaign with the same two-GPU tensor split used in the report:

```bash
scripts/issue-108-vmm-batching-campaign.sh \
  build-hip-gfx1030/bin/llama-bench \
  /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  issue-108-rdna2-final-results \
  -- -sm tensor --tensor-split 1/1
```

No diagnostic environment variables are needed; the script sets only the narrowly scoped
ones needed for each child process. Detailed VMM timing is confined to the small preflight;
the benchmark arms do not carry per-map instrumentation. The former 64 KiB arm is a
request, so a driver that requires larger pages may resolve it to the same size as the
default arm; the logs and summary retain both requested and observed sizes. Please attach
both of these files to issue 108:

```text
issue-108-rdna2-final-results.tar.gz
issue-108-rdna2-final-results/SUMMARY.tsv
```

If a preflight fails, also attach `issue-108-rdna2-final-results/FAILURE.txt`. The archive
will already contain the associated log and hardware/build receipts.
