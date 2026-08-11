#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"$repo/results/p9-bless-resnet-safe-boundary-profile-$(date -u +%Y%m%dT%H%M%SZ)"}
mig_env=${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}
engine_2=${ENGINE_2:-"$repo/models/engines/mig-1g-q25/resnet10-detection.engine"}
engine_4=${ENGINE_4:-"$repo/models/engines/mig-1g-q25/resnet10-detection.engine"}
engine_6=${ENGINE_6:-"$repo/models/engines/mig-1g-q25/resnet10-detection.engine"}
engine_8=${ENGINE_8:-"$repo/models/engines/mig-1g-q25/resnet10-detection.engine"}
intercept="$repo/build-r39/libbless-trt-squad-intercept.so"
binary="$repo/build-r39/bless-trt-squad-replica-smoke"

[[ ! -e "$result_dir" ]] || { echo "result directory exists: $result_dir" >&2; exit 1; }
for engine in "$engine_2" "$engine_4" "$engine_6" "$engine_8"; do
  [[ -f "$engine" ]] || { echo "engine is missing: $engine" >&2; exit 1; }
done
# shellcheck source=/dev/null
source "$mig_env"
mkdir -p "$result_dir/profile" "$result_dir/provenance"

run_operation() {
  local operation=$1
  local output="$result_dir/profile/op-$operation"
  mkdir -p "$output"
  env \
    "CUDA_VISIBLE_DEVICES=$JDG_MIG_SMALL_UUID" \
    "CUDA_MPS_PIPE_DIRECTORY=$JDG_MPS_PIPE_DIRECTORY" \
    "CUDA_MPS_LOG_DIRECTORY=$JDG_MPS_LOG_DIRECTORY" \
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100" \
    "LD_PRELOAD=$intercept" \
    "BLESS_TRT_SWITCH_OPERATION=$operation" \
    "BLESS_TRT_ALLOW_MISMATCH=1" \
    timeout 120s taskset --cpu-list 13 \
    "$binary" "$engine_2" "$engine_4" "$engine_6" "$engine_8" \
    "$output/squad.jsonl" \
    >"$output/result.json" 2>"$output/stderr.txt"
}

run_operation 0
total_launches=$(jq -er '.logical_launches | select(type == "number" and . > 1)' \
  "$result_dir/profile/op-0/result.json")
for ((operation = 1; operation <= total_launches; operation++)); do
  run_operation "$operation"
done

python3 "$repo/analysis/freeze_bless_trt_safe_boundaries.py" \
  --profile-dir "$result_dir/profile" \
  --engine "$engine_2" --engine "$engine_4" --engine "$engine_6" \
  --engine "$engine_8" \
  --source-root "$repo" --output "$result_dir/boundary-lock.json" \
  >"$result_dir/freezer-output.json"

cp "$binary" "$intercept" "$result_dir/provenance/"
for index in 2 4 6 8; do
  variable="engine_$index"
  cp "${!variable}" "$result_dir/provenance/resnet10-${index}sm.engine"
done
for file in \
  baselines/bless/trt_activation_replica_smoke.cpp \
  baselines/bless/trt_squad_intercept.cpp \
  baselines/bless/trt_squad_intercept.hpp \
  analysis/freeze_bless_trt_safe_boundaries.py \
  scripts/run_p9_bless_trt_safe_boundary_profile.sh; do
  cp "$repo/$file" "$result_dir/provenance/"
done
(
  cd "$result_dir"
  find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"$result_dir/MANIFEST.sha256"
printf '%s\n' "$result_dir"
