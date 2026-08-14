#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly BIG_STATE_DIR="${BIG_STATE_DIR:-/tmp/jdg-mps-2g-p9-colocated}"
readonly BIG_PIPE_DIR="${BIG_STATE_DIR}/pipe"
readonly BIG_LOG_DIR="${BIG_STATE_DIR}/log"
readonly COMMON_WORKLOAD="${COMMON_WORKLOAD:-${ROOT_DIR}/results/p9-resnet50-imagenette-gate100-20260811/common-workload.json}"
readonly DEADLINE_LOCK="${DEADLINE_LOCK:-${ROOT_DIR}/results/p9-resnet50-imagenette-calibration-current-r03-20260811/deadline-lock.json}"
readonly MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/results/p9-resnet50-imagenette-model-20260811}"
readonly PRODUCER_ONNX="${PRODUCER_ONNX:-${ROOT_DIR}/results/p9-resnet50-dag-artifacts-20260811/resnet50-backbone.onnx}"
readonly PRODUCER_ENGINE="${PRODUCER_ENGINE:-${ROOT_DIR}/models/engines/mig-2g-q100/resnet50-backbone-fp16.engine}"
readonly BACKGROUND_ENGINE="${BACKGROUND_ENGINE:-${ROOT_DIR}/models/engines/mig-1g-q100/distilbert-sst2.engine}"
readonly TRTEXEC="${TRTEXEC:-/usr/bin/trtexec}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-mig-colocated-imagenette-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="$(realpath -m -- "${RESULT_DIR}")"
readonly RESULT_DIR
readonly WARMUP="${WARMUP:-10}"
readonly BACKGROUND_PERIOD_MS="${BACKGROUND_PERIOD_MS:-4.0}"
readonly ACCURACY_TOLERANCE="${ACCURACY_TOLERANCE:-0.02}"
big_mps_started=0

big_mps_alive() {
  [[ -S "${BIG_PIPE_DIR}/control" ]] &&
    printf 'get_server_list\n' | env \
      CUDA_MPS_PIPE_DIRECTORY="${BIG_PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${BIG_LOG_DIR}" \
      timeout 2s nvidia-cuda-mps-control 2>/dev/null | grep -Eq '^[0-9]+$'
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "${big_mps_started}" -eq 1 ]]; then
    printf 'quit\n' | env \
      CUDA_MPS_PIPE_DIRECTORY="${BIG_PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${BIG_LOG_DIR}" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

for path in \
  "${MIG_ENV}" \
  "${COMMON_WORKLOAD}" \
  "${DEADLINE_LOCK}" \
  "${PRODUCER_ONNX}" \
  "${BACKGROUND_ENGINE}" \
  "${MODEL_DIR}/resnet50-imagenette-head.engine" \
  "${MODEL_DIR}/resnet50-imagenette-unsplit.onnx" \
  "${MODEL_DIR}/class-map.json" \
  "${BUILD_DIR}/jdg-cuda-init-probe" \
  "${BUILD_DIR}/jdg-mig-trt-pipeline" \
  "${BUILD_DIR}/jdg-trt-bench" \
  "${TRTEXEC}"; do
  [[ -f "${path}" ]] || {
    printf 'missing colocated-MIG input: %s\n' "${path}" >&2
    exit 1
  }
done
[[ ! -e "${RESULT_DIR}" ]] || {
  printf 'result directory already exists: %s\n' "${RESULT_DIR}" >&2
  exit 1
}
[[ "${WARMUP}" =~ ^[0-9]+$ ]] || {
  printf 'WARMUP must be a nonnegative integer\n' >&2
  exit 1
}

python3 "${ROOT_DIR}/analysis/freeze_p9_pipeline_deadline.py" \
  --verify "${DEADLINE_LOCK}" >/dev/null

# shellcheck source=/dev/null
source "${MIG_ENV}"
readonly JDG_MIG_BIG_UUID

mapfile -t mig_devices < <(nvidia-smi -L | awk '/MIG /{print $0}')
if [[ "${#mig_devices[@]}" -ne 2 ]] || \
   [[ "${mig_devices[0]} ${mig_devices[1]}" != *"MIG 2g.0gb"* ]] || \
   [[ "${mig_devices[0]} ${mig_devices[1]}" != *"MIG 1g.0gb"* ]]; then
  printf 'expected the Thor 2g+1g MIG layout\n' >&2
  nvidia-smi -L >&2
  exit 1
fi

mkdir -p "${BIG_PIPE_DIR}" "${BIG_LOG_DIR}"
if ! big_mps_alive; then
  # Only stale state in this dedicated, explicit temporary directory is
  # removed.  The resident 1g MPS daemon is left untouched for the BE tenant.
  find "${BIG_PIPE_DIR}" -mindepth 1 -maxdepth 1 -delete
  env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
    LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    timeout 30s "${BUILD_DIR}/jdg-cuda-init-probe" \
      >"${BIG_LOG_DIR}/direct-init.jsonl" 2>&1
  env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
    CUDA_MPS_PIPE_DIRECTORY="${BIG_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${BIG_LOG_DIR}" \
    nvidia-cuda-mps-control -d
  big_mps_started=1
  env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
    CUDA_MPS_PIPE_DIRECTORY="${BIG_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${BIG_LOG_DIR}" \
    LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    timeout 30s "${BUILD_DIR}/jdg-cuda-init-probe" \
      >"${BIG_LOG_DIR}/mps-init.jsonl" 2>&1
fi
big_mps_alive || {
  printf '2g MPS server did not become ready\n' >&2
  exit 1
}

if [[ ! -f "${PRODUCER_ENGINE}" ]]; then
  mkdir -p "$(dirname "${PRODUCER_ENGINE}")"
  env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
    CUDA_MPS_PIPE_DIRECTORY="${BIG_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${BIG_LOG_DIR}" \
    "${TRTEXEC}" \
      --onnx="${PRODUCER_ONNX}" \
      --saveEngine="${PRODUCER_ENGINE}" \
      --fp16 \
      --skipInference \
      >"${BIG_LOG_DIR}/resnet50-backbone-2g-build.log" 2>&1
fi

request_count="$(jq -er '.request_count' "${COMMON_WORKLOAD}")"
input_trace="$(jq -er '.producer_input_trace_path' "${COMMON_WORKLOAD}")"
arrival_trace="$(jq -er '.operational_arrival_trace_path' "${COMMON_WORKLOAD}")"
request_manifest="$(jq -er '.arrival_trace_path' "${COMMON_WORKLOAD}")"
dataset_manifest="$(jq -er '.dataset_manifest_path' "${COMMON_WORKLOAD}")"
deadline_us="$(jq -er '.deadline_us' "${DEADLINE_LOCK}")"

python3 "${ROOT_DIR}/scripts/run_p9_dependent_stress_smoke.py" \
  --repo "${ROOT_DIR}" \
  --mig-env "${MIG_ENV}" \
  --big-mps-pipe "${BIG_PIPE_DIR}" \
  --result-dir "${RESULT_DIR}" \
  --scenario nvidia-mig-colocated-critical-dag \
  --workload resnet50-classification \
  --iterations "${request_count}" \
  --warmup "${WARMUP}" \
  --deadline-reference "${DEADLINE_LOCK}" \
  --background-period-ms "${BACKGROUND_PERIOD_MS}" \
  --producer-engine "${PRODUCER_ENGINE}" \
  --consumer-engine "${MODEL_DIR}/resnet50-imagenette-head.engine" \
  --consumer-input-tensor gpu_0/res4_5_branch2c_bn_2 \
  --producer-input-trace "${input_trace}" \
  --operational-arrival-trace "${arrival_trace}" \
  --require-operational-arrival-trace \
  --common-workload-contract "${COMMON_WORKLOAD}" \
  --require-common-workload \
  --application-output-trace-dir "${RESULT_DIR}/application-outputs"

scenario_dir="${RESULT_DIR}/nvidia-mig-colocated-critical-dag"
output_trace="${RESULT_DIR}/application-outputs/nvidia-mig-colocated-critical-dag/outputs.bin"
accuracy_dir="${RESULT_DIR}/application-accuracy"
mkdir -p "${accuracy_dir}"
reference_dir="$(dirname "${request_manifest}")"
reference_predictions="${REFERENCE_PREDICTIONS:-${reference_dir}/reference-predictions-current-deadline.jsonl}"
reference_pipeline="${REFERENCE_PIPELINE_CSV:-${reference_dir}/reference-current-deadline.csv}"
if [[ ! -f "${reference_predictions}" ]]; then
  reference_predictions="${reference_dir}/reference-predictions.jsonl"
fi
if [[ ! -f "${reference_pipeline}" ]]; then
  reference_pipeline="${reference_dir}/reference.csv"
fi
for path in \
  "${reference_predictions}" \
  "${reference_pipeline}" \
  "${reference_dir}/reference-output.bin" \
  "${scenario_dir}/pipeline.csv" \
  "${output_trace}"; do
  [[ -f "${path}" ]] || {
    printf 'missing accuracy-gate input: %s\n' "${path}" >&2
    exit 1
  }
done

python3 "${ROOT_DIR}/analysis/build_application_prediction_trace.py" \
  --output-trace "${output_trace}" \
  --pipeline-csv "${scenario_dir}/pipeline.csv" \
  --request-manifest "${request_manifest}" \
  --class-map "${MODEL_DIR}/class-map.json" \
  --warmup "${WARMUP}" \
  --deadline-us "${deadline_us}" \
  --prediction-mode argmax \
  --require-input-binding \
  --output "${accuracy_dir}/predictions.jsonl" >/dev/null

python3 "${ROOT_DIR}/analysis/verify_application_accuracy.py" \
  --reference-trace "${reference_predictions}" \
  --candidate-trace "${accuracy_dir}/predictions.jsonl" \
  --dataset "${dataset_manifest}" \
  --reference-engine "${MODEL_DIR}/resnet50-imagenette-unsplit.onnx" \
  --candidate-engine "${MODEL_DIR}/resnet50-imagenette-head.engine" \
  --workload resnet50-classification \
  --task classification \
  --deadline-us "${deadline_us}" \
  --accuracy-tolerance "${ACCURACY_TOLERANCE}" \
  --minimum-accuracy 0.80 \
  --reference-output-trace "${reference_dir}/reference-output.bin" \
  --candidate-output-trace "${output_trace}" \
  --output-trace-warmup "${WARMUP}" \
  --reference-pipeline-csv "${reference_pipeline}" \
  --candidate-pipeline-csv "${scenario_dir}/pipeline.csv" \
  --pipeline-warmup "${WARMUP}" \
  --require-input-binding \
  --require-output-traces \
  --output "${accuracy_dir}/accuracy-gate.json" >/dev/null

sudo -n nvidia-smi mig -lgip | tee "${RESULT_DIR}/mig-profiles.txt" >/dev/null
sudo -n nvidia-smi mig -lgipp | \
  tee "${RESULT_DIR}/mig-possible-placements.txt" >/dev/null
sudo -n nvidia-smi mig -lgi | \
  tee "${RESULT_DIR}/active-mig-instances.txt" >/dev/null
(
  cd "${RESULT_DIR}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) >"${RESULT_DIR}/SHA256SUMS"

printf '%s\n' "${RESULT_DIR}"
