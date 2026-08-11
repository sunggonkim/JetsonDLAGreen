#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"$repo/results/p9-orion-resnet-control-$(date -u +%Y%m%dT%H%M%SZ)"}
mig_env=${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}
deadline_lock=${DEADLINE_LOCK:?set DEADLINE_LOCK to the frozen ResNet pipeline lock}
requests=${CRITICAL_REQUESTS:-100}
warmup=${WARMUP_REQUESTS:-10}
background_period_us=${BACKGROUND_PERIOD_US:-4000}
binary="$repo/build-r39/jdg-orion-mig-trt-pipeline"
resnet="$repo/models/engines/mig-1g-q100/resnet10-detection.engine"
distilbert="$repo/models/engines/mig-1g-q100/distilbert-sst2.engine"
be_profile="$repo/results/p9-orion-distilbert-operation-profile-20260809T110353Z/profile.json"
hp_profile="$repo/results/p9-orion-resnet10-operation-profile-20260809T1430Z/profile.json"
be_scheduler_profile="$repo/results/p9-orion-distilbert-operation-profile-20260809T110353Z/scheduler-profile.tsv"
hp_scheduler_profile="$repo/results/p9-orion-resnet10-operation-profile-20260809T1430Z/scheduler-profile.tsv"

for path in "$mig_env" "$deadline_lock" "$binary" "$resnet" "$distilbert" \
            "$be_profile" "$hp_profile" "$be_scheduler_profile" \
            "$hp_scheduler_profile"; do
  [[ -f "$path" ]] || { printf 'missing Orion input: %s\n' "$path" >&2; exit 1; }
done
[[ "$requests" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid CRITICAL_REQUESTS\n' >&2; exit 1; }
[[ "$warmup" =~ ^[0-9]+$ ]] || { printf 'invalid WARMUP_REQUESTS\n' >&2; exit 1; }
[[ ! -e "$result_dir" ]] || { printf 'result directory exists: %s\n' "$result_dir" >&2; exit 1; }

deadline_kind=$(jq -er '.kind' "$deadline_lock")
case "$deadline_kind" in
  p9-common-placement-deadline-lock)
    python3 "$repo/analysis/freeze_p9_common_placement_deadline.py" \
      --verify "$deadline_lock" >/dev/null
    ;;
  p9-dependent-pipeline-deadline-lock)
    python3 "$repo/analysis/freeze_p9_pipeline_deadline.py" \
      --verify "$deadline_lock" >/dev/null
    ;;
  *)
    printf 'unsupported Orion deadline lock kind: %s\n' "$deadline_kind" >&2
    exit 1
    ;;
esac
[[ $(jq -er '.contract.workload' "$deadline_lock") == resnet-control ]] || {
  printf 'deadline lock is not for resnet-control\n' >&2; exit 1;
}
deadline_us=$(jq -er '.deadline_us' "$deadline_lock")
max_be_duration_us=${ORION_MAX_BE_DURATION_US:-$(jq -er '
  if .pooled_p99_us then .pooled_p99_us
  else (.deadline_us / (.slo_factor // 1.10))
  end
' "$deadline_lock")}

# shellcheck source=/dev/null
source "$mig_env"
: "${JDG_MIG_SMALL_UUID:?missing JDG_MIG_SMALL_UUID}"
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"
: "${JDG_MPS_PIPE_DIRECTORY:?missing JDG_MPS_PIPE_DIRECTORY}"

mkdir -p "$result_dir/provenance"
cp "$deadline_lock" "$result_dir/deadline-lock.json"
cp "$binary" "$result_dir/provenance/jdg-orion-mig-trt-pipeline"
cp "$resnet" "$result_dir/provenance/resnet10-detection.engine"
cp "$distilbert" "$result_dir/provenance/distilbert-sst2.engine"
cp "$be_profile" "$result_dir/provenance/best-effort-profile.json"
cp "$hp_profile" "$result_dir/provenance/high-priority-profile.json"
cp "$be_scheduler_profile" "$result_dir/provenance/best-effort-scheduler-profile.tsv"
cp "$hp_scheduler_profile" "$result_dir/provenance/high-priority-scheduler-profile.tsv"
cp "$repo/scripts/run_p9_orion_resnet_control_smoke.sh" "$result_dir/provenance/"
cp "$repo/baselines/orion/verify_resnet_control_smoke.py" "$result_dir/provenance/"

jq -n \
  --argjson requests "$requests" --argjson warmup "$warmup" \
  --argjson deadline_us "$deadline_us" \
  --argjson background_period_us "$background_period_us" \
  --argjson max_be_duration_us "$max_be_duration_us" \
  --arg deadline_lock_sha256 "$(sha256sum "$deadline_lock" | cut -d' ' -f1)" \
  '{schema_version:1,kind:"orion-thor-resnet-control-run-contract",
    workload:"resnet-control",requests:$requests,warmup:$warmup,
    deadline_us:$deadline_us,background_period_us:$background_period_us,
    max_be_duration_us:$max_be_duration_us,
    max_be_duration_source:"frozen-isolated-pipeline-p99",
    deadline_lock_sha256:$deadline_lock_sha256}' >"$result_dir/run-contract.json"

taskset --cpu-list 13 "$binary" \
  --producer-engine "$resnet" --producer "$JDG_MIG_SMALL_UUID" \
  --consumer "$JDG_MIG_BIG_UUID" \
  --producer-mps-pipe "$JDG_MPS_PIPE_DIRECTORY" \
  --workload resnet-control --transport registered-direct \
  --deadline-mode wall --deadline-us "$deadline_us" \
  --warmup "$warmup" --iterations "$requests" \
  --trace-csv "$result_dir/pipeline.csv" \
  --checksum-trace-csv "$result_dir/checksums.csv" \
  --orion-profile-aware true --orion-background-engine "$distilbert" \
  --orion-best-effort-profile "$be_scheduler_profile" \
  --orion-high-priority-profile "$hp_scheduler_profile" \
  --orion-decisions "$result_dir/scheduler-events.jsonl" \
  --orion-max-be-duration-us "$max_be_duration_us" \
  --orion-trace-mode events \
  --orion-background-period-us "$background_period_us" \
  >"$result_dir/result.json" 2>"$result_dir/stderr.log"

python3 "$repo/baselines/orion/verify_resnet_control_smoke.py" \
  --result "$result_dir/result.json" --pipeline "$result_dir/pipeline.csv" \
  --checksums "$result_dir/checksums.csv" --events "$result_dir/scheduler-events.jsonl" \
  --best-effort-profile "$be_profile" --high-priority-profile "$hp_profile" \
  --best-effort-scheduler-profile "$be_scheduler_profile" \
  --high-priority-scheduler-profile "$hp_scheduler_profile" \
  --binary "$result_dir/provenance/jdg-orion-mig-trt-pipeline" \
  --expected-requests "$requests" \
  --output "$result_dir/verification.json" >/dev/null

manifest_tmp=$(mktemp)
(
  cd "$result_dir"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) >"$manifest_tmp"
mv "$manifest_tmp" "$result_dir/SHA256SUMS"
printf '%s\n' "$result_dir"
