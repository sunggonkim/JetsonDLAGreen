#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"$repo/results/p9-bless-affinity-$(date -u +%Y%m%dT%H%M%SZ)"}
mig_env=${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}

[[ ! -e "$result_dir" ]] || { echo "result directory exists: $result_dir" >&2; exit 1; }
# shellcheck source=/dev/null
source "$mig_env"
mkdir -p "$result_dir/provenance"

common_env=(
  "CUDA_VISIBLE_DEVICES=$JDG_MIG_SMALL_UUID"
  "CUDA_MPS_PIPE_DIRECTORY=$JDG_MPS_PIPE_DIRECTORY"
  "CUDA_MPS_LOG_DIRECTORY=$JDG_MPS_LOG_DIRECTORY"
  "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100"
)
env "${common_env[@]}" taskset --cpu-list 13 \
  "$repo/build-r39/bless-context-probe" >"$result_dir/context-domain.json"

for pair in 25:2 50:4 75:6 100:8; do
  quota=${pair%%:*}
  sms=${pair##*:}
  env "${common_env[@]}" taskset --cpu-list 13 \
    "$repo/build-r39/bless-trt-affinity-smoke" \
    --affinity-sms "$sms" \
    --engine "$repo/models/engines/mig-1g-q${quota}/distilbert-sst2.engine" \
    --model-name distilbert-sst2 --role benchmark --samples 20 --warmup 5 \
    --include-transfers true --priority default \
    >"$result_dir/q${quota}.json" 2>"$result_dir/q${quota}.stderr"
done

cp "$repo/build-r39/bless-context-probe" "$result_dir/provenance/"
cp "$repo/build-r39/bless-trt-affinity-smoke" "$result_dir/provenance/"
cp "$repo/baselines/bless/context_probe.cpp" "$result_dir/provenance/"
cp "$repo/baselines/bless/trt_affinity_smoke.cpp" "$result_dir/provenance/"
cp "$repo/baselines/bless/verify_affinity_smoke.py" "$result_dir/provenance/"
cp "$repo/scripts/run_p9_bless_affinity_smoke.sh" "$result_dir/provenance/"
python3 "$repo/baselines/bless/verify_affinity_smoke.py" \
  --result-dir "$result_dir" --output "$result_dir/verification.json" >/dev/null
manifest_tmp=$(mktemp)
(
  cd "$result_dir"
  find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"$manifest_tmp"
mv "$manifest_tmp" "$result_dir/MANIFEST.sha256"
printf '%s\n' "$result_dir"
