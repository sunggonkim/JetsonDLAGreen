#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p1-ring-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly CONTROL_CPU="${CONTROL_CPU:-13}"
readonly PAYLOAD_BYTES="${PAYLOAD_BYTES:-65536}"
readonly REQUESTS="${REQUESTS:-100}"
readonly TIMEOUT_MS="${TIMEOUT_MS:-30000}"
readonly STALE_TIMEOUT_US="${STALE_TIMEOUT_US:-100000}"
readonly DELAY_US="${DELAY_US:-2000}"
readonly FAULT_AFTER="${FAULT_AFTER:-1}"
readonly TIMEOUT_DELAY_US="${TIMEOUT_DELAY_US:-50000}"
readonly TIMEOUT_CASE_MS="${TIMEOUT_CASE_MS:-1}"

if [[ ! -x "${BUILD_DIR}/jdg-mig-sysmem-ring" ]]; then
  echo "missing ${BUILD_DIR}/jdg-mig-sysmem-ring; build it first" >&2
  exit 1
fi
if [[ ! -r "${MIG_ENV}" ]]; then
  echo "missing MIG environment ${MIG_ENV}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_SMALL_UUID:?missing JDG_MIG_SMALL_UUID}"
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"
: "${JDG_MPS_PIPE_DIRECTORY:?missing JDG_MPS_PIPE_DIRECTORY}"

mkdir -p "${RESULT_DIR}"
cp "${MIG_ENV}" "${RESULT_DIR}/mig.env"
nvidia-smi -L >"${RESULT_DIR}/gpu-inventory.txt"
sha256sum "${BUILD_DIR}/jdg-mig-sysmem-ring" \
  "${ROOT_DIR}/benchmarks/mig_sysmem_ring.cu" \
  >"${RESULT_DIR}/SHA256SUMS"
printf 'payload_bytes=%s\nrequests=%s\ntimeout_ms=%s\nstale_timeout_us=%s\n' \
  "${PAYLOAD_BYTES}" "${REQUESTS}" "${TIMEOUT_MS}" "${STALE_TIMEOUT_US}" \
  >"${RESULT_DIR}/run-metadata.txt"

run_ring() {
  local name="$1"
  shift
  taskset --cpu-list "${CONTROL_CPU}" \
    "${BUILD_DIR}/jdg-mig-sysmem-ring" \
    --producer "${JDG_MIG_SMALL_UUID}" \
    --consumer "${JDG_MIG_BIG_UUID}" \
    --mps-pipe "${JDG_MPS_PIPE_DIRECTORY}" \
    --payload-bytes "${PAYLOAD_BYTES}" \
    --requests "${REQUESTS}" \
    --timeout-ms "${TIMEOUT_MS}" \
    --stale-timeout-us "${STALE_TIMEOUT_US}" \
    "$@" >"${RESULT_DIR}/${name}.json"
}

run_ring normal
run_ring delayed --consumer-delay-us "${DELAY_US}"
set +e
run_ring timeout --consumer-delay-us "${TIMEOUT_DELAY_US}" \
  --timeout-ms "${TIMEOUT_CASE_MS}"
timeout_exit_code=$?
set -e
if [[ "${timeout_exit_code}" -eq 0 ]]; then
  echo "timeout control unexpectedly succeeded" >&2
  exit 1
fi
printf 'timeout_exit_code=%s\n' "${timeout_exit_code}" \
  >>"${RESULT_DIR}/run-metadata.txt"
run_ring consumer-death --fail-consumer-after "${FAULT_AFTER}"

printf 'MIG system-memory ring smoke: %s\n' "${RESULT_DIR}"
