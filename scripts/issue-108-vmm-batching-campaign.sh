#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/issue-108-vmm-batching-campaign.sh LLAMA_BENCH MODEL [OUTPUT_DIR] [-- EXTRA_BENCH_ARGS...]

Runs the final issue #108 gfx1030 campaign. Before any timed matrix it:
  1. runs the host RDNA2/VMM policy contract;
  2. validates VMM accounting on GPU 0 and GPU 1; and
  3. runs a Q8 FlashAttention smoke test with the WGP correction receipt enabled.

The timed arms compare static Q8, the production 256 KiB HIP VMM commit size, and the
former 64 KiB policy in mirrored order. Every arm uses a fresh process. The first failure
stops the campaign and still produces a summary/archive, so one defect cannot turn into a
matrix of identical aborts.

Default workload: pp1601, tg64, b32768, ub2048, t8, ngl99, Flash Attention on.

Useful environment overrides:
  ISSUE108_PROMPT_TOKENS       default 1601
  ISSUE108_GENERATE_TOKENS     default 64
  ISSUE108_BATCH               default 32768
  ISSUE108_UBATCH              default 2048
  ISSUE108_THREADS             default 8
  ISSUE108_GPU_LAYERS          default 99

Pass hardware-specific options after --, for example:
  ... -- -sm tensor --tensor-split 1/1

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
    output_dir="issue-108-rdna2-final-$(date +%Y%m%d-%H%M%S)"
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
bin_dir=$(dirname "$bench")
policy_test="$bin_dir/test-cuda-rdna2-vmm-policy"
vmm_test="$bin_dir/test-vbr-vmm"
export LD_LIBRARY_PATH="$bin_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for required in "$policy_test" "$vmm_test"; do
    if [[ ! -x $required ]]; then
        echo "required preflight binary is missing: $required" >&2
        echo "build llama-bench, test-cuda-rdna2-vmm-policy, and test-vbr-vmm from this branch" >&2
        exit 2
    fi
done

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch=$(git -C "$repo_root" branch --show-current 2>/dev/null || echo unknown)"
    echo "bench=$bench"
    echo "bench_sha256=$(sha256sum "$bench" | awk '{print $1}')"
    "$bench" --version 2>&1 || true
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

finalize() {
    python3 "$summary" "$output_dir" | tee "$output_dir/SUMMARY.tsv"
    tar -czf "$output_dir.tar.gz" -C "$(dirname "$output_dir")" "$(basename "$output_dir")"
    echo "results=$output_dir"
    echo "archive=$output_dir.tar.gz"
}

stop_campaign() {
    echo "campaign stopped: $1" | tee "$output_dir/FAILURE.txt" >&2
    finalize
    exit 1
}

echo "[$(date --iso-8601=seconds)] policy preflight"
"$policy_test" > "$output_dir/policy-test.log" 2>&1 || \
    stop_campaign "RDNA2/VMM policy contract failed"
grep -q '^PASS:' "$output_dir/policy-test.log" || \
    stop_campaign "RDNA2/VMM policy test did not report PASS"

for device in 0 1; do
    echo "[$(date --iso-8601=seconds)] VMM device $device preflight"
    GGML_VBR_VMM_DIAGNOSTICS=1 "$vmm_test" "$device" \
        > "$output_dir/vmm-device-$device.log" 2>&1 || \
        stop_campaign "VMM accounting failed on device $device"
    grep -q "^PASS: device $device VMM range accounting (256 KiB pages)" \
        "$output_dir/vmm-device-$device.log" || \
        stop_campaign "device $device did not expose the production 256 KiB HIP VMM policy"
done

run_one() {
    local label=$1 cache_type=$2 commit_kb=$3 arm_prompt=$4 arm_generate=$5
    local stdout="$output_dir/$label.jsonl"
    local stderr="$output_dir/$label.stderr.log"
    local meta="$output_dir/$label.meta.txt"
    local -a cmd=("$bench" -m "$model" -ngl "$gpu_layers" -fa on
        -ctk "$cache_type" -ctv "$cache_type"
        -p "$arm_prompt" -n "$arm_generate" -b "$batch" -ub "$ubatch"
        -t "$threads" -r 1 -o jsonl "${extra_args[@]}")
    {
        echo "label=$label"
        echo "cache_type=$cache_type"
        echo "commit_kb=$commit_kb"
        printf 'command='; printf '%q ' "${cmd[@]}"; echo
    } > "$meta"
    echo "[$(date --iso-8601=seconds)] $label"
    local status=0
    if [[ $cache_type == vbr && $commit_kb != default ]]; then
        env GGML_FATTN_RDNA2_DIAGNOSTICS=1 \
            GGML_VBR_VMM_HIP_COMMIT_KB="$commit_kb" \
            "${cmd[@]}" > "$stdout" 2> "$stderr" || status=$?
    elif [[ $cache_type == vbr ]]; then
        env GGML_FATTN_RDNA2_DIAGNOSTICS=1 \
            "${cmd[@]}" > "$stdout" 2> "$stderr" || status=$?
    else
        env GGML_FATTN_RDNA2_DIAGNOSTICS=1 \
            "${cmd[@]}" > "$stdout" 2> "$stderr" || status=$?
    fi
    echo "status=$status" >> "$meta"
    return "$status"
}

run_checked() {
    local label=$1 cache_type=$2 commit_kb=$3 arm_prompt=$4 arm_generate=$5
    run_one "$label" "$cache_type" "$commit_kb" "$arm_prompt" "$arm_generate" || \
        stop_campaign "$label exited unsuccessfully"
    grep -q '"n_prompt"' "$output_dir/$label.jsonl" || \
        stop_campaign "$label produced no prompt benchmark receipt"
    if grep -q 'GGML_FATTN_RDNA2_WGP.*action=reject' "$output_dir/$label.stderr.log"; then
        stop_campaign "$label rejected the validated RDNA2 WGP kernel"
    fi
}

# Exercise the formerly aborting path once before the timed matrix. A completed benchmark is
# authoritative even when a newer ROCm runtime reports positive occupancy and needs no correction.
run_checked q8-smoke q8_0 0 "$prompt" 1
if grep -q 'GGML_FATTN_RDNA2_WGP.*action=accept' "$output_dir/q8-smoke.stderr.log"; then
    echo "RDNA2 WGP correction accepted the validated kernel"
else
    echo "No zero-occupancy correction was needed during the successful Q8 smoke test"
fi

run_checked q8-control-before q8_0 0 "$prompt" "$generate"
run_checked vbr-default-00 vbr default "$prompt" "$generate"
run_checked vbr-64k-00 vbr 64 "$prompt" "$generate"
run_checked vbr-64k-01 vbr 64 "$prompt" "$generate"
run_checked vbr-default-01 vbr default "$prompt" "$generate"
run_checked q8-control-after q8_0 0 "$prompt" "$generate"

finalize
