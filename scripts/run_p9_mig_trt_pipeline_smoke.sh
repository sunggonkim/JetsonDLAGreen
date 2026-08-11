#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly PRODUCER_ENGINE="${PRODUCER_ENGINE:-${ROOT_DIR}/models/engines/mig-1g-q100/resnet10-detection.engine}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-mig-trt-pipeline-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly WARMUP="${WARMUP:-10}"
readonly ITERATIONS="${ITERATIONS:-100}"
readonly CONTROL_CPU="${CONTROL_CPU:-13}"

for path in "${BUILD_DIR}/jdg-mig-trt-pipeline" "${MIG_ENV}" \
  "${PRODUCER_ENGINE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required artifact: ${path}" >&2
    exit 1
  fi
done

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_SMALL_UUID:?missing JDG_MIG_SMALL_UUID}"
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"

mkdir -p "${RESULT_DIR}"
cp "${MIG_ENV}" "${RESULT_DIR}/mig.env"
nvidia-smi -L >"${RESULT_DIR}/gpu-inventory.txt"
sha256sum "${BUILD_DIR}/jdg-mig-trt-pipeline" \
  "${ROOT_DIR}/benchmarks/mig_trt_pipeline.cpp" "${PRODUCER_ENGINE}" \
  >"${RESULT_DIR}/SHA256SUMS"

taskset --cpu-list "${CONTROL_CPU}" \
  "${BUILD_DIR}/jdg-mig-trt-pipeline" \
  --producer-engine "${PRODUCER_ENGINE}" \
  --producer "${JDG_MIG_SMALL_UUID}" \
  --consumer "${JDG_MIG_BIG_UUID}" \
  --producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY:-}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --trace-csv "${RESULT_DIR}/trace.csv" \
  --checksum-trace-csv "${RESULT_DIR}/checksums.csv" \
  | tee "${RESULT_DIR}/result.json"

python3 "${ROOT_DIR}/analysis/verify_p9_resnet_control_smoke.py" \
  --result-dir "${RESULT_DIR}" \
  --output "${RESULT_DIR}/verification.json"

printf 'Cross-MIG TensorRT pipeline smoke: %s\n' "${RESULT_DIR}"
