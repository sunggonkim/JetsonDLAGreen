#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"$repo/results/p9-orion-resnet50-imagenette-gate100-$(date -u +%Y%m%dT%H%M%SZ)"}
mig_env=${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}
deadline_lock=${DEADLINE_LOCK:-"$repo/results/p9-resnet50-imagenette-calibration-r02-20260811/deadline-lock.json"}
common_workload=${COMMON_WORKLOAD:-"$repo/results/p9-resnet50-imagenette-gate100-20260811/common-workload.json"}
model_dir=${MODEL_DIR:-"$repo/results/p9-resnet50-imagenette-model-20260811"}
profile_dir=${PROFILE_DIR:-"$repo/results/p9-orion-resnet50-imagenette-profile-20260811"}
be_profile_dir=${BE_PROFILE_DIR:-"$repo/results/p9-orion-distilbert-operation-profile-20260809T110353Z"}
distilbert=${BACKGROUND_ENGINE:-"$repo/models/engines/mig-1g-q100/distilbert-sst2.engine"}
binary=${ORION_BINARY:-"$repo/build-r39/jdg-orion-mig-trt-pipeline"}
requests=${CRITICAL_REQUESTS:-90}
warmup=${WARMUP_REQUESTS:-10}
background_period_us=${BACKGROUND_PERIOD_US:-4000}
accuracy_deadline_us=${ACCURACY_DEADLINE_US:-}

for path in "$mig_env" "$deadline_lock" "$common_workload" "$binary" \
            "$model_dir/resnet50-imagenette-backbone.engine" \
            "$model_dir/resnet50-imagenette-head.engine" \
            "$model_dir/class-map.json" "$distilbert" \
            "$profile_dir/profile.json" "$profile_dir/scheduler-profile.tsv" \
            "$be_profile_dir/profile.json" "$be_profile_dir/scheduler-profile.tsv"; do
  [[ -f "$path" ]] || { printf 'missing Orion ImageNette input: %s\n' "$path" >&2; exit 1; }
done
[[ ! -e "$result_dir" ]] || { printf 'result directory exists: %s\n' "$result_dir" >&2; exit 1; }
[[ "$requests" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid CRITICAL_REQUESTS\n' >&2; exit 1; }
[[ "$warmup" =~ ^[0-9]+$ ]] || { printf 'invalid WARMUP_REQUESTS\n' >&2; exit 1; }

python3 "$repo/analysis/freeze_p9_pipeline_deadline.py" --verify "$deadline_lock" >/dev/null
deadline_us=$(jq -er '.deadline_us' "$deadline_lock")
accuracy_deadline_us=${accuracy_deadline_us:-$deadline_us}
max_be_duration_us=${ORION_MAX_BE_DURATION_US:-$(jq -er '.pooled_p99_us' "$deadline_lock")}
input_trace=$(jq -er '.producer_input_trace_path' "$common_workload")
arrival_trace=$(jq -er '.operational_arrival_trace_path' "$common_workload")
# The common contract's canonical request manifest is its ordinary arrival trace.
request_manifest=$(jq -er '.arrival_trace_path' "$common_workload")
dataset_manifest=$(jq -er '.dataset_manifest_path' "$common_workload")
reference_dir=$(dirname "$request_manifest")
reference_predictions=${REFERENCE_PREDICTIONS:-}
if [[ -z "$reference_predictions" ]]; then
  if [[ -f "$reference_dir/reference-predictions-current-deadline.jsonl" ]]; then
    reference_predictions="$reference_dir/reference-predictions-current-deadline.jsonl"
  else
    reference_predictions="$reference_dir/reference-predictions.jsonl"
  fi
fi
reference_pipeline=${REFERENCE_PIPELINE_CSV:-}
if [[ -z "$reference_pipeline" ]]; then
  if [[ -f "$reference_dir/reference-current-deadline.csv" ]]; then
    reference_pipeline="$reference_dir/reference-current-deadline.csv"
  else
    reference_pipeline="$reference_dir/reference.csv"
  fi
fi
for path in "$reference_predictions" "$reference_pipeline" "$reference_dir/reference-output.bin"; do
  [[ -f "$path" ]] || { printf 'missing Orion ImageNette reference: %s\n' "$path" >&2; exit 1; }
done

# shellcheck source=/dev/null
source "$mig_env"
mkdir -p "$result_dir/provenance"

taskset --cpu-list 13 "$binary" \
  --producer-engine "$model_dir/resnet50-imagenette-backbone.engine" \
  --consumer-engine "$model_dir/resnet50-imagenette-head.engine" \
  --consumer-input-tensor gpu_0/res4_5_branch2c_bn_2 \
  --producer "$JDG_MIG_SMALL_UUID" --consumer "$JDG_MIG_BIG_UUID" \
  --producer-mps-pipe "$JDG_MPS_PIPE_DIRECTORY" \
  --workload resnet50-classification --transport registered-direct \
  --deadline-mode wall --deadline-us "$deadline_us" \
  --warmup "$warmup" --iterations "$requests" \
  --trace-csv "$result_dir/pipeline.csv" \
  --application-output-trace "$result_dir/application-output.bin" \
  --producer-input-trace "$input_trace" --arrival-trace "$arrival_trace" \
  --orion-profile-aware true --orion-background-engine "$distilbert" \
  --orion-best-effort-profile "$be_profile_dir/scheduler-profile.tsv" \
  --orion-high-priority-profile "$profile_dir/scheduler-profile.tsv" \
  --orion-decisions "$result_dir/scheduler-events.jsonl" \
  --orion-max-be-duration-us "$max_be_duration_us" \
  --orion-trace-mode events --orion-background-period-us "$background_period_us" \
  >"$result_dir/result.json" 2>"$result_dir/stderr.log"

python3 "$repo/baselines/orion/verify_resnet50_imagenette_smoke.py" \
  --result "$result_dir/result.json" --pipeline "$result_dir/pipeline.csv" \
  --events "$result_dir/scheduler-events.jsonl" \
  --best-effort-profile "$be_profile_dir/profile.json" \
  --high-priority-profile "$profile_dir/profile.json" \
  --deadline-lock "$deadline_lock" --common-workload "$common_workload" \
  --binary "$binary" \
  --producer-engine "$model_dir/resnet50-imagenette-backbone.engine" \
  --consumer-engine "$model_dir/resnet50-imagenette-head.engine" \
  --orion-source "$repo/baselines/orion/driver_capture/scheduler.cpp" \
  --repo "$repo" --output "$result_dir/verification.json" >/dev/null

python3 "$repo/analysis/build_application_prediction_trace.py" \
  --output-trace "$result_dir/application-output.bin" \
  --pipeline-csv "$result_dir/pipeline.csv" \
  --request-manifest "$request_manifest" --class-map "$model_dir/class-map.json" \
  --warmup "$warmup" --deadline-us "$accuracy_deadline_us" \
  --prediction-mode argmax --require-input-binding \
  --output "$result_dir/predictions.jsonl" >/dev/null

python3 "$repo/analysis/verify_application_accuracy.py" \
  --reference-trace "$reference_predictions" \
  --candidate-trace "$result_dir/predictions.jsonl" \
  --dataset "$dataset_manifest" \
  --reference-engine "$model_dir/resnet50-imagenette-unsplit.onnx" \
  --candidate-engine "$model_dir/resnet50-imagenette-head.engine" \
  --workload resnet50-classification --task classification \
  --deadline-us "$accuracy_deadline_us" --accuracy-tolerance 0.0 \
  --minimum-accuracy 0.80 \
  --reference-output-trace "$reference_dir/reference-output.bin" \
  --candidate-output-trace "$result_dir/application-output.bin" \
  --output-trace-warmup "$warmup" \
  --reference-pipeline-csv "$reference_pipeline" \
  --candidate-pipeline-csv "$result_dir/pipeline.csv" \
  --pipeline-warmup "$warmup" --require-input-binding --require-output-traces \
  --output "$result_dir/accuracy-gate.json" >/dev/null

cp "$binary" "$result_dir/provenance/jdg-orion-mig-trt-pipeline"
cp "$model_dir/resnet50-imagenette-backbone.engine" "$result_dir/provenance/"
cp "$model_dir/resnet50-imagenette-head.engine" "$result_dir/provenance/"
cp "$profile_dir/profile.json" "$result_dir/provenance/high-priority-profile.json"
cp "$profile_dir/scheduler-profile.tsv" "$result_dir/provenance/high-priority-scheduler-profile.tsv"
cp "$be_profile_dir/profile.json" "$result_dir/provenance/best-effort-profile.json"
cp "$be_profile_dir/scheduler-profile.tsv" "$result_dir/provenance/best-effort-scheduler-profile.tsv"
cp "$repo/baselines/orion/driver_capture/scheduler.cpp" "$result_dir/provenance/"
cp "$repo/baselines/orion/verify_resnet50_imagenette_smoke.py" "$result_dir/provenance/"
cp "$repo/scripts/run_p9_orion_resnet50_imagenette_smoke.sh" "$result_dir/provenance/"
(
  cd "$result_dir"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) >"$result_dir/SHA256SUMS"
printf '%s\n' "$result_dir"
