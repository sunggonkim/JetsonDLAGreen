#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly ENGINE="${ENGINE:-${ROOT_DIR}/models/engines/mig-2g/resnet10-detection.engine}"
readonly REQUESTS="${REQUESTS:-6}"
readonly CONTROL_CPU="${CONTROL_CPU:-12}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-orion-driver-capture-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly CAPTURE_LIBRARY="${BUILD_DIR}/liborion-trt-driver-capture.so"

for path in "${BUILD_DIR}/jdg-trt-bench" "${CAPTURE_LIBRARY}" "${MIG_ENV}" "${ENGINE}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'missing Orion capture input: %s\n' "${path}" >&2
    exit 1
  fi
done
if [[ -e "${RESULT_DIR}" ]]; then
  printf 'refusing to append to existing result directory: %s\n' "${RESULT_DIR}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"

mkdir -p "${RESULT_DIR}"
CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
LD_PRELOAD="${CAPTURE_LIBRARY}" \
ORION_TRT_DRIVER_TRACE="${RESULT_DIR}/driver-launches.jsonl" \
taskset --cpu-list "${CONTROL_CPU}" \
  "${BUILD_DIR}/jdg-trt-bench" \
  --engine "${ENGINE}" \
  --model-name resnet10-detection \
  --role benchmark \
  --samples "${REQUESTS}" \
  --warmup 1 \
  --burst-size 1 \
  --period-ms 20 \
  --deadline-ms 6 \
  --priority high \
  --include-transfers true \
  --trace "${RESULT_DIR}/requests.csv" \
  >"${RESULT_DIR}/benchmark.json"

python3 "${ROOT_DIR}/baselines/orion/verify_driver_capture.py" \
  --trace "${RESULT_DIR}/driver-launches.jsonl" \
  --benchmark "${RESULT_DIR}/benchmark.json" \
  --expected-requests "${REQUESTS}" \
  --expected-mig-uuid "${JDG_MIG_BIG_UUID}" \
  --output "${RESULT_DIR}/compatibility.json"

sha256sum "${CAPTURE_LIBRARY}" \
  "${ROOT_DIR}/baselines/orion/driver_capture/intercept_driver.cpp" \
  "${BUILD_DIR}/jdg-trt-bench" "${ENGINE}" \
  "${RESULT_DIR}/benchmark.json" "${RESULT_DIR}/driver-launches.jsonl" \
  >"${RESULT_DIR}/SHA256SUMS"

printf 'Orion TensorRT driver capture positive control: %s\n' "${RESULT_DIR}"
