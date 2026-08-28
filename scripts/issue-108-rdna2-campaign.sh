#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/issue-108-rdna2-campaign.sh SERVER MODEL [OUTPUT_DIR] [-- EXTRA_SERVER_ARGS...]

Uses an already-built issue-108 RDNA2 diagnostic server, generates one deterministic request
containing exactly 1,601 model-tokenized prompt tokens, runs the diagnostic phases, selects the
fastest valid Flash Attention path and ubatch automatically, and creates one .tar.gz archive.

The built-in server configuration matches the reported RX 6950 XT + RX 6800 workload:
  -ngl 99 -sm tensor -c 131072 -np 1 -b 32768 -ub 2048 -tb 24 -t 8
  --spec-type draft-mtp --spec-draft-n-max 2 --kv-unified --cache-prompt

Use EXTRA_SERVER_ARGS for local necessities such as a tensor split. Do not pass --port,
--flash-attn, --ubatch-size or -ct/--cache-type; the campaign controls those variables.

Examples:
  scripts/issue-108-rdna2-campaign.sh \
      ./build/bin/llama-server \
      /models/Qwen3.8-27B-UD-Q4_K_XL.gguf

  scripts/issue-108-rdna2-campaign.sh ./build/bin/llama-server /models/model.gguf issue-108-results -- \
      --tensor-split 12,16

Environment:
  RDNA2_TARGET_PROMPT_TOKENS default: 1601
  RDNA2_PREDICT_TOKENS       default: 64 (enough to exercise MTP)
  RDNA2_PORT                 default: 18080
  RDNA2_TIMEOUT              default: 900 seconds per measured request
  RDNA2_UBATCH_CANDIDATES    default: "256 512 1024 2048 4096"
  RDNA2_LAYER_SPLITS         optional space-separated extra tensor splits
  RDNA2_UPSTREAM_SERVER      optional comparison llama-server executable

The script stops if the control arm does not report exactly the requested prompt-token count.
Individual unsupported kernels are retained as useful failures; selection ignores failed arms.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
server=$1
model=$2
shift 2

if [[ ! -x $server ]]; then
    echo "server is not executable: $server" >&2
    exit 2
fi
if [[ ! -f $model ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi
server=$(readlink -f "$server")
model=$(readlink -f "$model")

if [[ $# -gt 0 && $1 != -- ]]; then
    output_root=$1
    shift
else
    output_root="issue-108-rdna2-$(date +%Y%m%d-%H%M%S)"
fi
if [[ $# -gt 0 ]]; then
    if [[ $1 != -- ]]; then
        usage >&2
        exit 2
    fi
    shift
fi
extra_server_args=("$@")

for arg in "${extra_server_args[@]}"; do
    case $arg in
        --port|-ct|-ctk|-ctv|--cache-type|--cache-type-k|--cache-type-v|--flash-attn|-ub|--ubatch-size)
            echo "EXTRA_SERVER_ARGS contains campaign-controlled option: $arg" >&2
            exit 2
            ;;
        --port=*|--cache-type=*|--cache-type-k=*|--cache-type-v=*|--flash-attn=*|--ubatch-size=*)
            echo "EXTRA_SERVER_ARGS contains campaign-controlled option: ${arg%%=*}" >&2
            exit 2
            ;;
    esac
done

if [[ $output_root != /* ]]; then
    output_root="$PWD/$output_root"
fi
mkdir -p "$output_root"
output_root=$(readlink -f "$output_root")
exec > >(tee -a "$output_root/campaign.log") 2>&1

target_tokens=${RDNA2_TARGET_PROMPT_TOKENS:-1601}
predict_tokens=${RDNA2_PREDICT_TOKENS:-64}
port=${RDNA2_PORT:-18080}
timeout_s=${RDNA2_TIMEOUT:-900}
ubatch_candidates=${RDNA2_UBATCH_CANDIDATES:-"256 512 1024 2048 4096"}
runner="$repo_root/scripts/issue-108-rdna2-sweep.sh"
request="$output_root/request-${target_tokens}-tokens.json"
prompt_text="$output_root/request-${target_tokens}-tokens.txt"

case $target_tokens in
    ''|*[!0-9]*) echo "RDNA2_TARGET_PROMPT_TOKENS must be a positive integer" >&2; exit 2 ;;
esac
case $predict_tokens in
    ''|*[!0-9]*) echo "RDNA2_PREDICT_TOKENS must be a positive integer" >&2; exit 2 ;;
esac
if (( target_tokens < 2 )); then
    echo "RDNA2_TARGET_PROMPT_TOKENS must be at least 2" >&2
    exit 2
fi
if (( predict_tokens < 1 )); then
    echo "RDNA2_PREDICT_TOKENS must be at least 1" >&2
    exit 2
fi
if [[ ! -x $runner ]]; then
    echo "missing executable sweep runner: $runner" >&2
    exit 2
fi

base_args=(
    -m "$model"
    -ngl 99
    --spec-type draft-mtp
    --spec-draft-n-max 2
    -sm tensor
    -c 131072
    -np 1
    -b 32768
    -tb 24
    -t 8
    --kv-unified
    --cache-prompt
    "${extra_server_args[@]}"
)

echo "Issue 108 RDNA2 diagnostic campaign"
echo "repo=$repo_root"
echo "commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "server=$server"
echo "model=$model"
echo "output=$output_root"
echo "target_prompt_tokens=$target_tokens"
echo "predict_tokens=$predict_tokens"
printf 'extra_server_args='; printf '%q ' "${extra_server_args[@]}"; echo

tokenizer_pid=
campaign_complete=0
stop_tokenizer() {
    if [[ -n ${tokenizer_pid:-} ]] && kill -0 "$tokenizer_pid" 2>/dev/null; then
        kill "$tokenizer_pid" 2>/dev/null || true
        wait "$tokenizer_pid" 2>/dev/null || true
    fi
    tokenizer_pid=
}
cleanup_campaign() {
    local status=$?
    stop_tokenizer
    if [[ $status != 0 && $campaign_complete != 1 ]]; then
        local partial_archive="${output_root%/}.partial.tar.gz"
        tar -czf "$partial_archive" -C "$(dirname "$output_root")" "$(basename "$output_root")" 2>/dev/null || true
        echo "campaign stopped with status $status; partial_archive=$partial_archive" >&2
    fi
}
trap cleanup_campaign EXIT
trap 'exit 130' INT TERM

echo "[$(date --iso-8601=seconds)] generating deterministic model-tokenized request"
tokenizer_log="$output_root/tokenizer.server.log"
"$server" "${base_args[@]}" --port "$port" --flash-attn off --ubatch-size 256 -ct q8_0 \
    --spec-type none > "$tokenizer_log" 2>&1 &
tokenizer_pid=$!
tokenizer_ready=0
for _ in $(seq 1 600); do
    if ! kill -0 "$tokenizer_pid" 2>/dev/null; then
        break
    fi
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        tokenizer_ready=1
        break
    fi
    sleep 1
done
if [[ $tokenizer_ready != 1 ]]; then
    echo "tokenizer server did not become ready; see $tokenizer_log" >&2
    exit 1
fi

python3 - "$port" "$target_tokens" "$predict_tokens" "$request" "$prompt_text" <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
target = int(sys.argv[2])
predict = int(sys.argv[3])
request_path = sys.argv[4]
text_path = sys.argv[5]
base = f"http://127.0.0.1:{port}"

paragraph = (
    "A diagnostic benchmark should be repeatable, explicit about its inputs, and easy to "
    "audit. This document describes a fictional observatory whose engineers compare several "
    "computational routes while preserving the same sequence of symbols. Each record contains "
    "ordinary prose, punctuation, decimal values such as 17.25, and stable labels. The purpose "
    "is not to answer a question but to provide enough natural-language context for a measured "
    "prompt-processing pass. No record depends on the date, the machine, or external data."
)
source = "\n\n".join(f"Benchmark record {i:04d}. {paragraph}" for i in range(1, 1025))
body = json.dumps({"content": source, "add_special": True, "parse_special": True}).encode()
req = urllib.request.Request(base + "/tokenize", data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as response:
    obj = json.load(response)
tokens = obj.get("tokens") if isinstance(obj, dict) else None
if not isinstance(tokens, list) or len(tokens) < target:
    raise SystemExit(f"tokenizer returned {0 if tokens is None else len(tokens)} tokens; need {target}")
tokens = tokens[:target]

request_obj = {
    "prompt": tokens,
    "n_predict": predict,
    "temperature": 0.0,
    "seed": 1,
    "stream": False,
    "cache_prompt": False,
}
with open(request_path, "w", encoding="utf-8") as f:
    json.dump(request_obj, f, separators=(",", ":"))
    f.write("\n")

body = json.dumps({"tokens": tokens}).encode()
req = urllib.request.Request(base + "/detokenize", data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as response:
    detokenized = json.load(response).get("content", "")
with open(text_path, "w", encoding="utf-8") as f:
    f.write(detokenized)

print(f"generated_prompt_tokens={len(tokens)}")
print(f"requested_prediction_tokens={predict}")
print(f"first_token={tokens[0]}")
print(f"last_token={tokens[-1]}")
PY
stop_tokenizer

run_phase() {
    local phase=$1 paths=$2 ubatches=$3 cache_types=$4 spec_modes=$5 flash_modes=$6 allreduce_modes=$7
    shift 7
    local phase_dir="$output_root/$phase"
    mkdir -p "$phase_dir"
    echo "[$(date --iso-8601=seconds)] phase=$phase paths='$paths' ubatches='$ubatches' cache_types='$cache_types' spec_modes='$spec_modes' flash='$flash_modes' allreduce='$allreduce_modes'"
    RDNA2_PATHS="$paths" \
    RDNA2_UBATCHES="$ubatches" \
    RDNA2_CACHE_TYPES="$cache_types" \
    RDNA2_SPEC_MODES="$spec_modes" \
    RDNA2_FLASH_MODES="$flash_modes" \
    RDNA2_ALLREDUCE_MODES="$allreduce_modes" \
    RDNA2_PORT="$port" \
    RDNA2_TIMEOUT="$timeout_s" \
        "$runner" "$server" "$request" "$phase_dir" -- "${base_args[@]}" "$@"
}

validate_phase() {
    local phase=$1 require_success=${2:-1} expected_content_hash=${3:-}
    python3 - "$output_root/$phase" "$target_tokens" "$require_success" "$expected_content_hash" <<'PY'
import glob
import os
import sys

directory, target, require_success, expected_hash = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ok = 0
bad_counts = []
bad_content = []
for path in glob.glob(os.path.join(directory, "*.meta.txt")):
    data = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            key, sep, value = line.rstrip("\n").partition("=")
            if sep:
                data[key] = value
    if data.get("status") != "ok":
        continue
    ok += 1
    try:
        count = int(float(data.get("timings_prompt_n", "-1")))
    except ValueError:
        count = -1
    if count != target:
        bad_counts.append((os.path.basename(path), count))
    if expected_hash and data.get("content_sha256") != expected_hash:
        bad_content.append((os.path.basename(path), data.get("content_sha256", "missing")))
if bad_counts:
    for name, count in bad_counts:
        print(f"ERROR: {name} evaluated {count} prompt tokens; expected {target}", file=sys.stderr)
    raise SystemExit(1)
if bad_content:
    for name, digest in bad_content:
        print(f"ERROR: {name} generated content hash {digest}; expected control {expected_hash}", file=sys.stderr)
    raise SystemExit(1)
if require_success and ok == 0:
    print(f"ERROR: no successful arms in {directory}", file=sys.stderr)
    raise SystemExit(1)
print(f"phase_validated={os.path.basename(directory)} successful_arms={ok} prompt_tokens={target}")
PY
}

first_content_hash() {
    local phase=$1
    python3 - "$output_root/$phase" <<'PY'
import glob
import os
import sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.meta.txt"))):
    data = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            key, sep, value = line.rstrip("\n").partition("=")
            if sep:
                data[key] = value
    if data.get("status") == "ok" and data.get("content_sha256"):
        print(data["content_sha256"])
        raise SystemExit(0)
raise SystemExit("no successful arm with generated-content hash")
PY
}

best_arm() {
    local phase=$1
    python3 - "$output_root/$phase" "$target_tokens" <<'PY'
import glob
import os
import re
import sys

best = None
for path in glob.glob(os.path.join(sys.argv[1], "*.meta.txt")):
    data = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            key, sep, value = line.rstrip("\n").partition("=")
            if sep:
                data[key] = value
    if data.get("status") != "ok" or data.get("timings_prompt_n") != sys.argv[2]:
        continue
    try:
        rate = float(data["timings_prompt_per_second"])
    except (KeyError, ValueError):
        continue
    label = data.get("label", os.path.basename(path))
    path_match = re.search(r"_path-(.+?)_ub-", label)
    ubatch_match = re.search(r"_ub-([0-9]+)_ct-", label)
    if not path_match or not ubatch_match:
        continue
    row = (rate, path_match.group(1), ubatch_match.group(1), label)
    if best is None or row[0] > best[0]:
        best = row
if best is None:
    raise SystemExit("no valid performance result")
print(best[1], best[2], best[0], best[3])
PY
}

# A conservative non-Flash control proves the request itself is valid before testing kernels.
run_phase 00-sanity "default" "2048" "q8_0" "none" "off" "configured"
validate_phase 00-sanity
control_content_hash=$(first_content_hash 00-sanity)
echo "control_content_sha256=$control_content_hash"

# Tile sweep isolates the attention kernel from VBR and speculative decoding.
run_phase 10-kernel-sweep "auto tile4 tile8 tile16 tile32 tile32-direct vec" "2048" "q8_0" "none" "on" "configured" || true
validate_phase 10-kernel-sweep 1 "$control_content_hash"
read -r best_path _ best_path_rate best_path_label < <(best_arm 10-kernel-sweep)
echo "selected_path=$best_path prompt_tps=$best_path_rate arm=$best_path_label"

# Compare cache representation at a fixed kernel, then isolate MTP overhead.
run_phase 20-cache-comparison "$best_path" "2048" "vbr q8_0" "none" "on" "configured"
validate_phase 20-cache-comparison
run_phase 30-spec-comparison "$best_path" "2048" "q8_0" "configured none" "on" "configured"
validate_phase 30-spec-comparison 1 "$control_content_hash"

# Find the best ubatch with the normal MTP configuration.
run_phase 40-ubatch-sweep "$best_path" "$ubatch_candidates" "q8_0" "configured" "on" "configured" || true
validate_phase 40-ubatch-sweep 1 "$control_content_hash"
read -r _ best_ubatch best_ubatch_rate best_ubatch_label < <(best_arm 40-ubatch-sweep)
echo "selected_ubatch=$best_ubatch prompt_tps=$best_ubatch_rate arm=$best_ubatch_label"

# Exercise the reported VBR+MTP combination at the selected settings.
run_phase 50-selected-vbr "$best_path" "$best_ubatch" "vbr" "configured" "on" "configured"
validate_phase 50-selected-vbr

# Multi-GPU collective and non-Flash controls distinguish attention from communication costs.
run_phase 60-allreduce "$best_path" "$best_ubatch" "q8_0" "configured" "on" "configured internal none" || true
validate_phase 60-allreduce 1 "$control_content_hash"
run_phase 70-flash-off-q8 "default" "$best_ubatch" "q8_0" "configured" "off" "configured"
validate_phase 70-flash-off-q8 1 "$control_content_hash"
run_phase 71-flash-off-vbr "default" "$best_ubatch" "vbr" "configured" "off" "configured"
validate_phase 71-flash-off-vbr

if [[ -n ${RDNA2_LAYER_SPLITS:-} ]]; then
    for split in $RDNA2_LAYER_SPLITS; do
        safe_split=${split//[^A-Za-z0-9_.-]/_}
        run_phase "80-layer-split-$safe_split" "$best_path" "$best_ubatch" "q8_0" "configured" "on" "configured" --tensor-split "$split" || true
        validate_phase "80-layer-split-$safe_split" 1 "$control_content_hash"
    done
fi

if [[ -n ${RDNA2_UPSTREAM_SERVER:-} ]]; then
    upstream=$(readlink -f "$RDNA2_UPSTREAM_SERVER")
    if [[ ! -x $upstream ]]; then
        echo "RDNA2_UPSTREAM_SERVER is not executable: $upstream" >&2
        exit 1
    fi
    saved_server=$server
    server=$upstream
    run_phase 90-upstream "default" "$best_ubatch" "q8_0" "none" "on" "configured"
    validate_phase 90-upstream
    server=$saved_server
fi

python3 - "$output_root" <<'PY'
import glob
import os
import sys

root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, "*", "*.meta.txt")):
    data = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            key, sep, value = line.rstrip("\n").partition("=")
            if sep:
                data[key] = value
    rows.append((
        os.path.basename(os.path.dirname(path)),
        data.get("label", os.path.basename(path)),
        data.get("status", "unknown"),
        data.get("timings_prompt_n", ""),
        data.get("timings_prompt_per_second", ""),
        data.get("timings_predicted_per_second", ""),
        data.get("content_sha256", ""),
    ))
rows.sort()
with open(os.path.join(root, "summary.tsv"), "w", encoding="utf-8") as f:
    f.write("phase\tarm\tstatus\tprompt_tokens\tprompt_tps\tgeneration_tps\tcontent_sha256\n")
    for row in rows:
        f.write("\t".join(row) + "\n")
PY

cat > "$output_root/selected.env" <<EOF
RDNA2_BEST_PATH=$best_path
RDNA2_BEST_UBATCH=$best_ubatch
RDNA2_BEST_PATH_PROMPT_TPS=$best_path_rate
RDNA2_BEST_UBATCH_PROMPT_TPS=$best_ubatch_rate
RDNA2_TARGET_PROMPT_TOKENS=$target_tokens
RDNA2_PREDICT_TOKENS=$predict_tokens
EOF

archive="${output_root%/}.tar.gz"
tar -czf "$archive" -C "$(dirname "$output_root")" "$(basename "$output_root")"
campaign_complete=1
echo "[$(date --iso-8601=seconds)] campaign complete"
echo "summary=$output_root/summary.tsv"
echo "selected=$output_root/selected.env"
echo "archive=$archive"
