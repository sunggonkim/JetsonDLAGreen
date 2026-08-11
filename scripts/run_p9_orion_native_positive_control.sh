#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly ENGINE="${ENGINE:-${ROOT_DIR}/models/engines/mig-2g/resnet10-detection.engine}"
readonly REQUESTS="${REQUESTS:-4}"
readonly CONTROL_CPU="${CONTROL_CPU:-12}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-orion-native-positive-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly CAPTURE_LIBRARY="${BUILD_DIR}/liborion-trt-driver-capture.so"

for path in "${BUILD_DIR}/orion-trt-native-smoke" "${CAPTURE_LIBRARY}" \
            "${MIG_ENV}" "${ENGINE}"; do
  [[ -f "${path}" ]] || { printf 'missing Orion native input: %s\n' "${path}" >&2; exit 1; }
done
[[ ! -e "${RESULT_DIR}" ]] || { printf 'result directory exists: %s\n' "${RESULT_DIR}" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"

CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
LD_PRELOAD="${CAPTURE_LIBRARY}" \
ORION_TRT_DRIVER_TRACE="${RESULT_DIR}/driver-launches.jsonl" \
timeout 45s taskset --cpu-list "${CONTROL_CPU}" \
  "${BUILD_DIR}/orion-trt-native-smoke" \
  --engine "${ENGINE}" --samples "${REQUESTS}" --output-dir "${RESULT_DIR}"

python3 "${ROOT_DIR}/baselines/orion/verify_native_smoke.py" \
  --result "${RESULT_DIR}/result.json" \
  --decisions "${RESULT_DIR}/scheduler-decisions.jsonl" \
  --launches "${RESULT_DIR}/driver-launches.jsonl" \
  --requests "${REQUESTS}" --mig-uuid "${JDG_MIG_BIG_UUID}" \
  --output "${RESULT_DIR}/verification.json"

sha256sum "${CAPTURE_LIBRARY}" "${BUILD_DIR}/orion-trt-native-smoke" \
  "${BUILD_DIR}/jdg-trt-bench" "${ENGINE}" \
  "${ROOT_DIR}/baselines/orion/driver_capture/intercept_driver.cpp" \
  "${ROOT_DIR}/baselines/orion/driver_capture/scheduler.cpp" \
  "${RESULT_DIR}/result.json" "${RESULT_DIR}/verification.json" \
  "${RESULT_DIR}/scheduler-decisions.jsonl" \
  "${RESULT_DIR}/driver-launches.jsonl" >"${RESULT_DIR}/SHA256SUMS"

printf 'Orion native TensorRT queue positive control: %s\n' "${RESULT_DIR}"
