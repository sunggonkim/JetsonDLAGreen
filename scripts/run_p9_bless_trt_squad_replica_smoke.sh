#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"$repo/results/p9-bless-trt-squad-replica-$(date -u +%Y%m%dT%H%M%SZ)"}
mig_env=${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}
engine_2=${ENGINE_2:-"$repo/models/engines/mig-1g-q25/resnet10-detection.engine"}
engine_4=${ENGINE_4:-"$repo/models/engines/mig-1g-q25/resnet10-detection.engine"}
engine_6=${ENGINE_6:-"$repo/models/engines/mig-1g-q25/resnet10-detection.engine"}
engine_8=${ENGINE_8:-"$repo/models/engines/mig-1g-q25/resnet10-detection.engine"}
intercept="$repo/build-r39/libbless-trt-squad-intercept.so"
boundary_lock=${BOUNDARY_LOCK:?set BOUNDARY_LOCK to the matching engine profile}

[[ ! -e "$result_dir" ]] || { echo "result directory exists: $result_dir" >&2; exit 1; }
# shellcheck source=/dev/null
source "$mig_env"
mkdir -p "$result_dir/provenance"
switch_operation=$(jq -er '.selected_switch_operation' "$boundary_lock")

env \
  "CUDA_VISIBLE_DEVICES=$JDG_MIG_SMALL_UUID" \
  "CUDA_MPS_PIPE_DIRECTORY=$JDG_MPS_PIPE_DIRECTORY" \
  "CUDA_MPS_LOG_DIRECTORY=$JDG_MPS_LOG_DIRECTORY" \
  "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100" \
  "LD_PRELOAD=$intercept" \
  "BLESS_TRT_SWITCH_OPERATION=$switch_operation" \
  timeout 120s taskset --cpu-list 13 \
  "$repo/build-r39/bless-trt-squad-replica-smoke" \
  "$engine_2" "$engine_4" "$engine_6" "$engine_8" \
  "$result_dir/squad.jsonl" >"$result_dir/result.json" \
  2>"$result_dir/stderr.txt"

cp "$repo/build-r39/bless-trt-squad-replica-smoke" "$result_dir/provenance/"
cp "$intercept" "$result_dir/provenance/"
for file in \
  baselines/bless/trt_activation_replica_smoke.cpp \
  baselines/bless/trt_squad_intercept.cpp \
  baselines/bless/trt_squad_intercept.hpp \
  baselines/bless/verify_trt_squad_replica_smoke.py \
  scripts/run_p9_bless_trt_squad_replica_smoke.sh; do
  cp "$repo/$file" "$result_dir/provenance/"
done
python3 "$repo/baselines/bless/verify_trt_squad_replica_smoke.py" \
  --result-dir "$result_dir" --engine "$engine_8" \
  --boundary-lock "$boundary_lock" \
  --output "$result_dir/verification.json"
cp "$boundary_lock" "$result_dir/provenance/boundary-lock.json"
(
  cd "$result_dir"
  find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | \
    xargs -0 sha256sum
) >"$result_dir/MANIFEST.sha256"
printf '%s\n' "$result_dir"
