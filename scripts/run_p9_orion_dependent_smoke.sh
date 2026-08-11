#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"$repo/results/p9-orion-dependent-whisper-$(date -u +%Y%m%dT%H%M%SZ)"}
mig_env=${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}
deadline_lock=${DEADLINE_LOCK:?set DEADLINE_LOCK to the frozen pipeline lock}
requests=${CRITICAL_REQUESTS:-1000}
warmup=${WARMUP_REQUESTS:-20}
background_period_us=${BACKGROUND_PERIOD_US:-4000}
binary="$repo/build-r39/jdg-orion-mig-trt-pipeline"
whisper="$repo/models/engines/mig-1g-q100/whisper-tiny-encoder.engine"
distilbert="$repo/models/engines/mig-1g-q100/distilbert-sst2.engine"
be_profile="$repo/results/p9-orion-distilbert-operation-profile-20260809T110353Z/profile.json"
hp_profile="$repo/results/p9-orion-whisper-operation-profile-20260809T110353Z/profile.json"
be_scheduler_profile="$repo/results/p9-orion-distilbert-operation-profile-20260809T110353Z/scheduler-profile.tsv"
hp_scheduler_profile="$repo/results/p9-orion-whisper-operation-profile-20260809T110353Z/scheduler-profile.tsv"

for path in "$mig_env" "$deadline_lock" "$binary" "$whisper" "$distilbert" \
            "$be_profile" "$hp_profile" "$be_scheduler_profile" \
            "$hp_scheduler_profile"; do
  [[ -f "$path" ]] || { printf 'missing Orion input: %s\n' "$path" >&2; exit 1; }
done
[[ ! -e "$result_dir" ]] || { printf 'result directory exists: %s\n' "$result_dir" >&2; exit 1; }
[[ "$requests" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid CRITICAL_REQUESTS\n' >&2; exit 1; }
[[ "$warmup" =~ ^[0-9]+$ ]] || { printf 'invalid WARMUP_REQUESTS\n' >&2; exit 1; }

python3 "$repo/analysis/freeze_p9_pipeline_deadline.py" \
  --verify "$deadline_lock" >/dev/null
deadline_us=$(jq -er '.deadline_us' "$deadline_lock")
# Upstream Orion sets this bound per HP model. Its published values track the
# isolated HP inference duration; use the frozen pipeline's isolated pooled p99.
max_be_duration_us=${ORION_MAX_BE_DURATION_US:-$(jq -er '.pooled_p99_us' "$deadline_lock")}

# shellcheck source=/dev/null
source "$mig_env"
mkdir -p "$result_dir/provenance"
cp "$deadline_lock" "$result_dir/deadline-lock.json"
cp "$binary" "$result_dir/provenance/jdg-orion-mig-trt-pipeline"
cp "$be_profile" "$result_dir/provenance/best-effort-profile.json"
cp "$hp_profile" "$result_dir/provenance/high-priority-profile.json"
cp "$be_scheduler_profile" "$result_dir/provenance/best-effort-scheduler-profile.tsv"
cp "$hp_scheduler_profile" "$result_dir/provenance/high-priority-scheduler-profile.tsv"
cp "$repo/scripts/run_p9_orion_dependent_smoke.sh" "$result_dir/provenance/"
cp "$repo/baselines/orion/verify_dependent_smoke.py" "$result_dir/provenance/"

jq -n \
  --argjson requests "$requests" --argjson warmup "$warmup" \
  --argjson deadline_us "$deadline_us" \
  --argjson background_period_us "$background_period_us" \
  --argjson max_be_duration_us "$max_be_duration_us" \
  --arg deadline_lock_sha256 "$(sha256sum "$deadline_lock" | cut -d' ' -f1)" \
  --arg be_profile_sha256 "$(sha256sum "$be_profile" | cut -d' ' -f1)" \
  --arg hp_profile_sha256 "$(sha256sum "$hp_profile" | cut -d' ' -f1)" \
  --arg be_scheduler_profile_sha256 "$(sha256sum "$be_scheduler_profile" | cut -d' ' -f1)" \
  --arg hp_scheduler_profile_sha256 "$(sha256sum "$hp_scheduler_profile" | cut -d' ' -f1)" \
  '{schema_version:1,kind:"orion-thor-dependent-run-contract",
    workload:"whisper-projection",requests:$requests,warmup:$warmup,
    deadline_us:$deadline_us,background_period_us:$background_period_us,
    max_be_duration_us:$max_be_duration_us,
    max_be_duration_source:"frozen-isolated-pipeline-p99",
    deadline_lock_sha256:$deadline_lock_sha256,
    best_effort_profile_sha256:$be_profile_sha256,
    high_priority_profile_sha256:$hp_profile_sha256,
    best_effort_scheduler_profile_sha256:$be_scheduler_profile_sha256,
    high_priority_scheduler_profile_sha256:$hp_scheduler_profile_sha256}' \
  >"$result_dir/run-contract.json"

taskset --cpu-list 13 "$binary" \
  --producer-engine "$whisper" --producer "$JDG_MIG_SMALL_UUID" \
  --consumer "$JDG_MIG_BIG_UUID" \
  --producer-mps-pipe "$JDG_MPS_PIPE_DIRECTORY" \
  --workload whisper-projection --transport registered-direct \
  --deadline-mode wall --deadline-us "$deadline_us" \
  --warmup "$warmup" --iterations "$requests" \
  --trace-csv "$result_dir/pipeline.csv" \
  --orion-profile-aware true --orion-background-engine "$distilbert" \
  --orion-best-effort-profile "$be_scheduler_profile" \
  --orion-high-priority-profile "$hp_scheduler_profile" \
  --orion-decisions "$result_dir/scheduler-events.jsonl" \
  --orion-max-be-duration-us "$max_be_duration_us" \
  --orion-trace-mode events \
  --orion-background-period-us "$background_period_us" \
  >"$result_dir/result.json" 2>"$result_dir/stderr.log"

python3 "$repo/baselines/orion/verify_dependent_smoke.py" \
  --result "$result_dir/result.json" --pipeline "$result_dir/pipeline.csv" \
  --events "$result_dir/scheduler-events.jsonl" \
  --best-effort-profile "$be_profile" --high-priority-profile "$hp_profile" \
  --best-effort-scheduler-profile "$be_scheduler_profile" \
  --high-priority-scheduler-profile "$hp_scheduler_profile" \
  --run-contract "$result_dir/run-contract.json" \
  --deadline-lock "$result_dir/deadline-lock.json" \
  --orion-binary "$result_dir/provenance/jdg-orion-mig-trt-pipeline" \
  --repo "$repo" --output "$result_dir/verification.json" >/dev/null

manifest_tmp=$(mktemp)
(
  cd "$result_dir"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) >"$manifest_tmp"
mv "$manifest_tmp" "$result_dir/SHA256SUMS"
printf '%s\n' "$result_dir"
