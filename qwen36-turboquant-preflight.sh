#!/usr/bin/env bash
set -euo pipefail

MODEL_PATTERN='llama-server .*Qwen3\.6-27B-UD-Q5_K_XL\.gguf'
PORT=18080

# Only clean stale Qwen3.6 TurboQuant instances. Do not kill the CPU embedder on 8083.
pkill -TERM -f "$MODEL_PATTERN" 2>/dev/null || true
sleep 3
pkill -KILL -f "$MODEL_PATTERN" 2>/dev/null || true

if ss -ltn "sport = :${PORT}" | tail -n +2 | grep -q .; then
  echo "Port ${PORT} is still in use after stale Qwen3.6 cleanup" >&2
  exit 1
fi

/opt/rocm/bin/rocm-smi --showpids --showmeminfo vram
