#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/issue-108-vmm-batching-campaign.sh LLAMA_BENCH MODEL [OUTPUT_DIR] [-- EXTRA_BENCH_ARGS...]

Runs each arm in a fresh process and compares the HIP VMM commit sizes 4, 64, 256, and
2048 KiB in mirrored order, bracketed by static Q8 controls. The default workload matches
issue #108: pp1601, tg64, b32768, ub2048, t8, ngl99, Flash Attention on.

Useful environment overrides:
  ISSUE108_PROMPT_TOKENS       default 1601
  ISSUE108_GENERATE_TOKENS     default 64
  ISSUE108_BATCH               default 32768
  ISSUE108_UBATCH              default 2048
  ISSUE108_THREADS             default 8
  ISSUE108_GPU_LAYERS          default 99

Pass hardware-specific options after --, for example:
  ... -- -sm tensor --tensor-split 12,16

The script controls -ctk/-ctv, -p, -n, -b, -ub, -t, -ngl, -fa, -r, and -o.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi

bench=$1
model=$2
shift 2
if [[ ! -x $bench ]]; then
    echo "llama-bench is not executable: $bench" >&2
    exit 2
fi
if [[ ! -f $model ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi
bench=$(readlink -f "$bench")
model=$(readlink -f "$model")

if (( $# > 0 )) && [[ $1 != -- ]]; then
    output_dir=$1
    shift
else
    output_dir="issue-108-vmm-batching-$(date +%Y%m%d-%H%M%S)"
fi
if (( $# > 0 )); then
    if [[ $1 != -- ]]; then
        usage >&2
        exit 2
    fi
    shift
fi
extra_args=("$@")

for arg in "${extra_args[@]}"; do
    case $arg in
        -ct|-ctk|-ctv|--cache-type|--cache-type-k|--cache-type-v|-p|-n|-b|-ub|-t|-ngl|-fa|-r|-o|--output)
            echo "extra arguments contain campaign-controlled option: $arg" >&2
            exit 2
            ;;
        --cache-type=*|--cache-type-k=*|--cache-type-v=*|--output=*)
            echo "extra arguments contain campaign-controlled option: ${arg%%=*}" >&2
            exit 2
            ;;
    esac
done

mkdir -p "$output_dir"
output_dir=$(readlink -f "$output_dir")
prompt=${ISSUE108_PROMPT_TOKENS:-1601}
generate=${ISSUE108_GENERATE_TOKENS:-64}
batch=${ISSUE108_BATCH:-32768}
ubatch=${ISSUE108_UBATCH:-2048}
threads=${ISSUE108_THREADS:-8}
gpu_layers=${ISSUE108_GPU_LAYERS:-99}

for value in "$prompt" "$generate" "$batch" "$ubatch" "$threads" "$gpu_layers"; do
    if [[ ! $value =~ ^[0-9]+$ ]]; then
        echo "campaign numeric controls must be non-negative integers" >&2
        exit 2
    fi
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
summary="$repo_root/scripts/issue-108-vmm-summary.py"

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "bench=$bench"
    echo "bench_sha256=$(sha256sum "$bench" | awk '{print $1}')"
    echo "model=$model"
    stat -c 'model_size=%s model_mtime=%Y' "$model"
    printf 'extra_args='; printf '%q ' "${extra_args[@]}"; echo
    echo "prompt=$prompt generate=$generate batch=$batch ubatch=$ubatch"
    echo "threads=$threads gpu_layers=$gpu_layers"
    uname -a
    hipcc --version 2>&1 || true
    rocminfo 2>&1 | sed -n '1,180p' || true
    rocm-smi --showproductname --showdriverversion --showmeminfo vram 2>&1 || true
} > "$output_dir/system.txt"

run_one() {
    local label=$1 cache_type=$2 commit_kb=$3
    local stdout="$output_dir/$label.jsonl"
    local stderr="$output_dir/$label.stderr.log"
    local meta="$output_dir/$label.meta.txt"
    local -a cmd=("$bench" -m "$model" -ngl "$gpu_layers" -fa on
        -ctk "$cache_type" -ctv "$cache_type"
        -p "$prompt" -n "$generate" -b "$batch" -ub "$ubatch"
        -t "$threads" -r 1 -o jsonl "${extra_args[@]}")
    {
        echo "label=$label"
        echo "cache_type=$cache_type"
        echo "commit_kb=$commit_kb"
        printf 'command='; printf '%q ' "${cmd[@]}"; echo
    } > "$meta"
    echo "[$(date --iso-8601=seconds)] $label"
    local status=0
    if [[ $cache_type == vbr ]]; then
        env GGML_VBR_VMM_DIAGNOSTICS=1 GGML_VBR_VMM_HIP_COMMIT_KB="$commit_kb" \
            "${cmd[@]}" > "$stdout" 2> "$stderr" || status=$?
    else
        "${cmd[@]}" > "$stdout" 2> "$stderr" || status=$?
    fi
    echo "status=$status" >> "$meta"
    return "$status"
}

failures=0
run_one q8-control-before q8_0 0 || failures=$((failures + 1))
for kb in 4 64 256 2048 2048 256 64 4; do
    run_one "vbr-${kb}k-$(printf '%02d' $(find "$output_dir" -name "vbr-${kb}k-*.meta.txt" | wc -l))" vbr "$kb" || \
        failures=$((failures + 1))
done
run_one q8-control-after q8_0 0 || failures=$((failures + 1))

python3 "$summary" "$output_dir" | tee "$output_dir/SUMMARY.tsv"
tar -czf "$output_dir.tar.gz" -C "$(dirname "$output_dir")" "$(basename "$output_dir")"
echo "results=$output_dir"
echo "archive=$output_dir.tar.gz"
if (( failures != 0 )); then
    exit 1
fi
