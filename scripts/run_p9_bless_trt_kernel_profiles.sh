#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"$repo/results/p9-bless-common-kernel-profiles-$(date -u +%Y%m%dT%H%M%SZ)"}
mig_env=${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}
binary="$repo/build-r39/bless-trt-squad-replica-smoke"
intercept="$repo/build-r39/libbless-trt-squad-intercept.so"

[[ ! -e "$result_dir" ]] || { echo "result directory exists: $result_dir" >&2; exit 1; }
# shellcheck source=/dev/null
source "$mig_env"
mkdir -p "$result_dir/raw"

profile_model() {
  local model=$1
  local engine=$2
  local sms=$3
  local output="$result_dir/raw/$model-$sms"
  mkdir -p "$output"
  env \
    "CUDA_VISIBLE_DEVICES=$JDG_MIG_SMALL_UUID" \
    "CUDA_MPS_PIPE_DIRECTORY=$JDG_MPS_PIPE_DIRECTORY" \
    "CUDA_MPS_LOG_DIRECTORY=$JDG_MPS_LOG_DIRECTORY" \
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100" \
    "LD_PRELOAD=$intercept" \
    "BLESS_TRT_FIXED_SMS=$sms" \
    "BLESS_TRT_PROFILE_SYNC=1" \
    timeout 120s taskset --cpu-list 13 \
    "$binary" "$engine" "$output/squad.jsonl" \
    >"$output/result.json" 2>"$output/stderr.txt"
}

for sms in 2 4 6 8; do
  profile_model resnet \
    "$repo/models/engines/mig-1g-q25/resnet10-detection.engine" "$sms"
  profile_model distilbert \
    "$repo/models/engines/mig-1g-q25/distilbert-sst2.engine" "$sms"
done

python3 "$repo/analysis/build_bless_common_schedule.py" \
  --profile-root "$result_dir/raw" --output "$result_dir/schedule.json"
(
  cd "$result_dir"
  find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"$result_dir/MANIFEST.sha256"
printf '%s\n' "$result_dir"
