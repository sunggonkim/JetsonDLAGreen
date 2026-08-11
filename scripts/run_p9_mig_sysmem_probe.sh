#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-mig-sysmem-probe-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly WARMUP="${WARMUP:-10}"
readonly ITERATIONS="${ITERATIONS:-100}"
readonly SIZES="${SIZES:-64,4096,65536,1048576,8388608}"
readonly CONTROL_CPU="${CONTROL_CPU:-13}"
readonly TRANSPORTS="${TRANSPORTS:-registered-direct}"

if [[ ! -x "${BUILD_DIR}/jdg-mig-sysmem-handoff" ]]; then
  echo "missing ${BUILD_DIR}/jdg-mig-sysmem-handoff; build it first" >&2
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

mkdir -p "${RESULT_DIR}"
cp "${MIG_ENV}" "${RESULT_DIR}/mig.env"
printf 'transport_modes=%s\nsizes=%s\nwarmup=%s\niterations=%s\ncontrol_cpu=%s\n' \
  "${TRANSPORTS}" "${SIZES}" "${WARMUP}" "${ITERATIONS}" "${CONTROL_CPU}" \
  >"${RESULT_DIR}/run-metadata.txt"
nvidia-smi -L >"${RESULT_DIR}/gpu-inventory.txt"
sha256sum "${BUILD_DIR}/jdg-mig-sysmem-handoff" \
  "${ROOT_DIR}/benchmarks/mig_sysmem_handoff.cu" \
  >"${RESULT_DIR}/SHA256SUMS"

IFS=',' read -r -a transport_modes <<<"${TRANSPORTS}"
for transport in "${transport_modes[@]}"; do
  if [[ -z "${transport}" ]]; then
    echo "TRANSPORTS contains an empty mode" >&2
    exit 1
  fi
  taskset --cpu-list "${CONTROL_CPU}" \
    "${BUILD_DIR}/jdg-mig-sysmem-handoff" \
    --producer "${JDG_MIG_SMALL_UUID}" \
    --consumer "${JDG_MIG_BIG_UUID}" \
    --producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY:-}" \
    --transport "${transport}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --sizes "${SIZES}" \
    | tee "${RESULT_DIR}/${transport}.json"
done
if [[ "${#transport_modes[@]}" -eq 1 ]]; then
  cp "${RESULT_DIR}/${transport_modes[0]}.json" "${RESULT_DIR}/result.json"
fi

printf 'Cross-MIG system-memory probe: %s\n' "${RESULT_DIR}"
