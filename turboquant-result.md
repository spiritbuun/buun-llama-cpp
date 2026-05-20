# Qwen3.6 27B Q5_K_XL 131K TurboQuant Result

Backend: `spiritbuun/buun-llama-cpp` HIP build, 7900 XTX, `turbo4` KV, full offload.

## Result

| Metric | Value |
|---|---:|
| Prompt tokens | 130,052 |
| Completion tokens | 207 |
| Prompt eval | 285.91 tok/s |
| Generation | 10.33 tok/s |
| Peak GPU0 VRAM | 22.94 GiB |
| HTTP status | 200 |
| Output | coherent |

## Working Command

```bash
cd /home/homelabserver/src/llama.cpp-turboquant

env ROCR_VISIBLE_DEVICES=0 \
HIP_VISIBLE_DEVICES=0 \
GPU_DEVICE_ORDINAL=0 \
CUDA_VISIBLE_DEVICES=0 \
build/bin/llama-server \
  -m /home/homelabserver/models/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  -ngl 99 \
  -c 131072 \
  -np 1 \
  -b 512 \
  -ub 128 \
  -ctk turbo4 \
  -ctv turbo4 \
  --cache-ram 0 \
  --no-cache-prompt \
  -ctxcp 0 \
  -cpent -1 \
  --timeout 7200 \
  --temp 0 \
  --reasoning off \
  --host 0.0.0.0 \
  --port 8080 \
  --no-webui
```

## Notes

- This meets the 131K target on the 7900 XTX without dropping Qwen from Q5 to Q4.
- The fork needed one local HIP shim patch: `ggml/src/ggml-cuda/vendors/hip.h` now aliases `cudaPointerAttributes`, `cudaPointerGetAttributes`, and `cudaMemoryTypeDevice` to HIP equivalents.
- Final cleanup returned GPU0 to baseline VRAM.

## Logs

- `/tmp/qwen-longctx/turbohip-q5-131072-ngl99-turbo4-b512-ub128-rerun.log`
- `/tmp/qwen-longctx/turbohip-q5-131072-ngl99-turbo4-b512-ub128-rerun-response.json`
- `/tmp/qwen-longctx/turbohip-q5-131072-ngl99-turbo4-b512-ub128-rerun-vram.log`
