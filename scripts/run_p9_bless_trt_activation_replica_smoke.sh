#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"$repo/results/p9-bless-trt-activation-replica-$(date -u +%Y%m%dT%H%M%SZ)"}
mig_env=${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}
engine="$repo/models/engines/mig-1g-q25/distilbert-sst2.engine"
capture="$repo/build-r39/liborion-trt-driver-capture.so"

[[ ! -e "$result_dir" ]] || { echo "result directory exists: $result_dir" >&2; exit 1; }
# shellcheck source=/dev/null
source "$mig_env"
mkdir -p "$result_dir/provenance"

env \
  "CUDA_VISIBLE_DEVICES=$JDG_MIG_SMALL_UUID" \
  "CUDA_MPS_PIPE_DIRECTORY=$JDG_MPS_PIPE_DIRECTORY" \
  "CUDA_MPS_LOG_DIRECTORY=$JDG_MPS_LOG_DIRECTORY" \
  "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100" \
  "LD_PRELOAD=$capture" \
  "ORION_TRT_DRIVER_TRACE=$result_dir/driver-launches.jsonl" \
  timeout 60s taskset --cpu-list 13 \
  "$repo/build-r39/bless-trt-activation-replica-smoke" "$engine" \
  >"$result_dir/result.json" 2>"$result_dir/stderr.txt"

cp "$repo/build-r39/bless-trt-activation-replica-smoke" "$result_dir/provenance/"
cp "$capture" "$result_dir/provenance/"
cp "$repo/baselines/bless/trt_activation_replica_smoke.cpp" "$result_dir/provenance/"
cp "$repo/baselines/bless/verify_trt_activation_replica_smoke.py" "$result_dir/provenance/"
cp "$repo/baselines/orion/driver_capture/intercept_driver.cpp" "$result_dir/provenance/"
cp "$repo/scripts/run_p9_bless_trt_activation_replica_smoke.sh" "$result_dir/provenance/"
python3 "$repo/baselines/bless/verify_trt_activation_replica_smoke.py" \
  --result-dir "$result_dir" --engine "$engine" \
  --output "$result_dir/verification.json" >/dev/null
manifest_tmp=$(mktemp)
(
  cd "$result_dir"
  find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"$manifest_tmp"
mv "$manifest_tmp" "$result_dir/MANIFEST.sha256"
printf '%s\n' "$result_dir"
