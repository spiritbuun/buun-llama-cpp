#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/issue-108-rdna2-sweep.sh SERVER REQUEST_JSON OUTPUT_DIR -- SERVER_ARGS...

REQUEST_JSON is posted unchanged to /completion for every run. SERVER_ARGS should contain the
model, tensor split and any drafter/mmproj arguments, but should omit --port, --flash-attn,
--ubatch-size, -ct/--cache-type and --spec-type because this script controls those switches.

Environment controls (space-separated):
  RDNA2_PATHS       default: "auto tile4 tile8 tile16 tile32 tile32-direct vec"
  RDNA2_UBATCHES    default: "2048"
  RDNA2_CACHE_TYPES default: "vbr q8_0"
  RDNA2_SPEC_MODES  default: "configured"
  RDNA2_FLASH_MODES default: "on"
  RDNA2_ALLREDUCE_MODES default: "configured" (also accepts internal/none)
  RDNA2_PORT        default: 18080
  RDNA2_TIMEOUT     default: 900 seconds per request

For a short first pass, use one cache type and the configured drafter. Then benchmark only the
best viable paths with additional ubatch sizes, for example:
  RDNA2_PATHS="auto tile8 tile16 vec" RDNA2_UBATCHES="256 512 1024 2048" \
  RDNA2_CACHE_TYPES="vbr" RDNA2_SPEC_MODES="configured none" \
    scripts/issue-108-rdna2-sweep.sh ./build/bin/llama-server request.json logs -- -m model.gguf ...

Flash Attention off is a separate control:
  RDNA2_PATHS="default" RDNA2_FLASH_MODES="off" RDNA2_CACHE_TYPES="vbr q8_0" ...
EOF
}

if [[ $# -lt 5 ]]; then
    usage >&2
    exit 2
fi

server=$1
request=$2
output_dir=$3
shift 3
if [[ ${1:-} != "--" ]]; then
    usage >&2
    exit 2
fi
shift
server_args=("$@")

if [[ ! -x $server ]]; then
    echo "server is not executable: $server" >&2
    exit 2
fi
if [[ ! -f $request ]]; then
    echo "request JSON does not exist: $request" >&2
    exit 2
fi

paths=${RDNA2_PATHS:-"auto tile4 tile8 tile16 tile32 tile32-direct vec"}
ubatches=${RDNA2_UBATCHES:-"2048"}
cache_types=${RDNA2_CACHE_TYPES:-"vbr q8_0"}
spec_modes=${RDNA2_SPEC_MODES:-"configured"}
flash_modes=${RDNA2_FLASH_MODES:-"on"}
allreduce_modes=${RDNA2_ALLREDUCE_MODES:-"configured"}
port=${RDNA2_PORT:-18080}
timeout_s=${RDNA2_TIMEOUT:-900}
base_url="http://127.0.0.1:${port}"

mkdir -p "$output_dir"
request_hash=$(sha256sum "$request" | awk '{print $1}')
request_bytes=$(wc -c < "$request")

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "request=$request"
    echo "request_sha256=$request_hash"
    echo "request_bytes=$request_bytes"
    echo "server=$server"
    printf 'server_args='; printf '%q ' "${server_args[@]}"; echo
    echo "paths=$paths"
    echo "ubatches=$ubatches"
    echo "cache_types=$cache_types"
    echo "spec_modes=$spec_modes"
    echo "flash_modes=$flash_modes"
    echo "allreduce_modes=$allreduce_modes"
    echo "uname=$(uname -a)"
    echo "hipcc=$(command -v hipcc || true)"
    hipcc --version 2>&1 || true
    server_real=$(readlink -f "$server" 2>/dev/null || echo "$server")
    server_build=$(dirname "$(dirname "$server_real")")
    if [[ -f $server_build/CMakeCache.txt ]]; then
        echo "cmake_cache=$server_build/CMakeCache.txt"
        grep -E '^(CMAKE_BUILD_TYPE|CMAKE_HIP_COMPILER|CMAKE_HIP_FLAGS|CMAKE_CXX_FLAGS|AMDGPU_TARGETS|GGML_HIP|GGML_CUDA)' \
            "$server_build/CMakeCache.txt" || true
    fi
    ldd "$server" 2>&1 || true
    rocminfo 2>&1 | sed -n '1,160p' || true
    rocm-smi --showproductname --showdriverversion --showuniqueid --showmeminfo vram 2>&1 || true
} > "$output_dir/system.txt"

server_pid=
cleanup() {
    if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

run_one() {
    local path=$1 ubatch=$2 cache_type=$3 spec_mode=$4 flash_mode=$5 allreduce_mode=$6
    local label="fa-${flash_mode}_path-${path}_ub-${ubatch}_ct-${cache_type}_spec-${spec_mode}_ar-${allreduce_mode}"
    local log="$output_dir/${label}.server.log"
    local response="$output_dir/${label}.response.json"
    local meta="$output_dir/${label}.meta.txt"
    local -a cmd=("$server" "${server_args[@]}" --port "$port" --flash-attn "$flash_mode" --ubatch-size "$ubatch" -ct "$cache_type")

    if [[ $spec_mode == none ]]; then
        cmd+=(--spec-type none)
    elif [[ $spec_mode != configured ]]; then
        echo "unknown RDNA2_SPEC_MODES value: $spec_mode" >&2
        return 2
    fi

    # The path selector is irrelevant when FA is disabled, but retaining a stable label makes
    # result collation simple. "default" means the production dispatcher with diagnostics only.
    local path_env=
    if [[ $path != default ]]; then
        path_env=$path
    fi

    {
        echo "label=$label"
        echo "request_sha256=$request_hash"
        echo "request_bytes=$request_bytes"
        printf 'command='; printf '%q ' "${cmd[@]}"; echo
        echo "GGML_FATTN_RDNA2_PATH=${path_env:-<unset>}"
    } > "$meta"

    echo "[$(date --iso-8601=seconds)] starting $label"
    local -a env_args=(env
        GGML_FATTN_DIAGNOSTICS=1
        GGML_FATTN_RDNA2_PATH="$path_env"
        GGML_VBR_TOPOLOGY_DIAGNOSTICS=1
        GGML_VBR_VMM_DIAGNOSTICS=1
        GGML_ALLREDUCE_DIAGNOSTICS=1)
    if [[ $allreduce_mode != configured ]]; then
        if [[ $allreduce_mode != internal && $allreduce_mode != none ]]; then
            echo "unknown RDNA2_ALLREDUCE_MODES value: $allreduce_mode" >&2
            return 2
        fi
        env_args+=(GGML_CUDA_ALLREDUCE="$allreduce_mode")
    fi
    "${env_args[@]}" "${cmd[@]}" > "$log" 2>&1 &
    server_pid=$!

    local ready=0
    for _ in $(seq 1 600); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            break
        fi
        if curl -fsS "$base_url/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if [[ $ready != 1 ]]; then
        echo "status=startup_failed" >> "$meta"
        cleanup
        server_pid=
        return 1
    fi

    local start end curl_status
    start=$(date +%s%N)
    timeout "$timeout_s" curl -fsS -H 'Content-Type: application/json' --data-binary "@$request" \
        "$base_url/completion" > "$response"
    curl_status=$?
    end=$(date +%s%N)
    {
        echo "status=$([[ $curl_status == 0 ]] && echo ok || echo request_failed)"
        echo "curl_status=$curl_status"
        echo "wall_ns=$((end - start))"
        echo "response_sha256=$(sha256sum "$response" 2>/dev/null | awk '{print $1}')"
        echo "response_bytes=$(wc -c < "$response" 2>/dev/null || echo 0)"
    } >> "$meta"
    if command -v python3 >/dev/null && [[ -s $response ]]; then
        python3 - "$response" >> "$meta" <<'PY' || true
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        obj = json.load(f)
except Exception as exc:
    print(f"response_parse_error={exc}")
    raise SystemExit(0)

timings = obj.get("timings") or {}
for key in (
    "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
    "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
):
    if key in timings:
        print(f"timings_{key}={timings[key]}")
usage = obj.get("usage") or {}
for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
    if key in usage:
        print(f"usage_{key}={usage[key]}")
PY
    fi

    cleanup
    server_pid=
    return "$curl_status"
}

failures=0
for flash_mode in $flash_modes; do
    for allreduce_mode in $allreduce_modes; do
        for path in $paths; do
            if [[ $flash_mode == off && $path != default ]]; then
                continue
            fi
            for ubatch in $ubatches; do
                for cache_type in $cache_types; do
                    for spec_mode in $spec_modes; do
                        run_one "$path" "$ubatch" "$cache_type" "$spec_mode" "$flash_mode" "$allreduce_mode" || failures=$((failures + 1))
                    done
                done
            done
        done
    done
done

grep -hE 'GGML_(FATTN|VBR_VMM|VBR_TOPOLOGY|ALLREDUCE)_DIAG' "$output_dir"/*.server.log \
    > "$output_dir/diagnostics.txt" 2>/dev/null || true
echo "completed with $failures failed arm(s); results: $output_dir"
exit "$([[ $failures == 0 ]] && echo 0 || echo 1)"
