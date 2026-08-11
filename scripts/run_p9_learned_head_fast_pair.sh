#!/usr/bin/env bash
set -euo pipefail

# Short production-wall sanity pair.  It can consume a common-workload/input
# contract for trace binding, but never consumes thermal/deadline locks and is
# never formal.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/p9-real-resnet-head-fast-$(date -u +%Y%m%dT%H%M%SZ)}"
DEADLINE_US="${DEADLINE_US:?set DEADLINE_US to the exploratory wall deadline in microseconds}"
ITERATIONS="${ITERATIONS:-20}"
WARMUP="${WARMUP:-5}"
BACKGROUND_PERIOD_MS="${BACKGROUND_PERIOD_MS:-4}"
CONSUMER_ENGINE="${CONSUMER_ENGINE:-${ROOT_DIR}/results/p9-real-resnet-head-artifacts-20260810/resnet10-detection-head.engine}"
PRODUCER_INPUT_TRACE="${PRODUCER_INPUT_TRACE:-}"
COMMON_WORKLOAD_CONTRACT="${COMMON_WORKLOAD_CONTRACT:-}"
REQUIRE_COMMON_WORKLOAD="${REQUIRE_COMMON_WORKLOAD:-0}"
NO_BACKGROUND="${NO_BACKGROUND:-0}"

for path in "${MIG_ENV}" "${CONSUMER_ENGINE}"; do
  [[ -f "${path}" ]] || { printf 'missing fast-pair input: %s\n' "${path}" >&2; exit 1; }
done
[[ ! -e "${RESULT_ROOT}" ]] || { printf 'result directory exists: %s\n' "${RESULT_ROOT}" >&2; exit 1; }
mkdir -p "${RESULT_ROOT}"

run_arm() {
  local name="$1" scenario="$2"
  local args=(
    python3 "${ROOT_DIR}/scripts/run_p9_dependent_stress_smoke.py"
    --mig-env "${MIG_ENV}" \
    --result-dir "${RESULT_ROOT}/${name}" \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    --deadline-us "${DEADLINE_US}" \
    --background-period-ms "${BACKGROUND_PERIOD_MS}" \
    --scenario "${scenario}" \
    --workload resnet-detection-head \
    --consumer-engine "${CONSUMER_ENGINE}" \
    --consumer-input-tensor Layer6_relu_Y \
    --checksum-mode inline \
    --application-output-trace-dir "${RESULT_ROOT}/${name}/application-outputs"
  )
  if [[ -n "${PRODUCER_INPUT_TRACE}" ]]; then
    args+=(--producer-input-trace "${PRODUCER_INPUT_TRACE}")
  fi
  if [[ -n "${COMMON_WORKLOAD_CONTRACT}" ]]; then
    args+=(--common-workload-contract "${COMMON_WORKLOAD_CONTRACT}")
  fi
  if [[ "${REQUIRE_COMMON_WORKLOAD}" == "1" ]]; then
    args+=(--require-common-workload)
  fi
  if [[ "${NO_BACKGROUND}" == "1" ]]; then
    args+=(--no-background)
  fi
  "${args[@]}"
}

run_arm quiet quiet
run_arm nvidia-mps nvidia-mps-spatial-sharing

printf 'Exploratory learned-head pair written to %s\n' "${RESULT_ROOT}"
printf 'This pair has no thermal, common-workload, or accuracy promotion gate.\n'
