#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly WORKLOAD="${WORKLOAD:-whisper-projection}"
readonly PRODUCER_ENGINE="${PRODUCER_ENGINE:-${ROOT_DIR}/models/engines/mig-1g-q100/whisper-tiny-encoder.engine}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-mig-trt-transport-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly WARMUP="${WARMUP:-10}"
readonly ITERATIONS="${ITERATIONS:-100}"
readonly CONTROL_CPU="${CONTROL_CPU:-13}"

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_SMALL_UUID:?missing JDG_MIG_SMALL_UUID}"
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"
: "${JDG_MPS_PIPE_DIRECTORY:?missing JDG_MPS_PIPE_DIRECTORY}"
case "${WORKLOAD}" in
  whisper-projection) : ;;
  *) printf 'transport smoke currently requires WORKLOAD=whisper-projection\n' >&2; exit 1 ;;
esac

mkdir -p "${RESULT_DIR}"
cp "${MIG_ENV}" "${RESULT_DIR}/mig.env"
nvidia-smi -L >"${RESULT_DIR}/gpu-inventory.txt"

run_cross_mig() {
  local transport="$1"
  taskset --cpu-list "${CONTROL_CPU}" \
    "${BUILD_DIR}/jdg-mig-trt-pipeline" \
    --producer-engine "${PRODUCER_ENGINE}" \
    --producer "${JDG_MIG_SMALL_UUID}" \
    --consumer "${JDG_MIG_BIG_UUID}" \
    --producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}" \
    --workload "${WORKLOAD}" --transport "${transport}" --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --application-output-trace "${RESULT_DIR}/${transport}.outputs.bin" \
    >"${RESULT_DIR}/${transport}.json"
}

for transport in pageable-bounce pinned-bounce registered-direct; do
  run_cross_mig "${transport}"
done

taskset --cpu-list "${CONTROL_CPU}" \
  "${BUILD_DIR}/jdg-mig-trt-pipeline" \
  --producer-engine "${PRODUCER_ENGINE}" \
  --producer "${JDG_MIG_SMALL_UUID}" \
  --consumer "${JDG_MIG_SMALL_UUID}" \
  --producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}" \
  --consumer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}" \
  --workload "${WORKLOAD}" --transport registered-direct --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --application-output-trace "${RESULT_DIR}/mps-same-instance.outputs.bin" \
  >"${RESULT_DIR}/mps-same-instance.json"

python3 - "${RESULT_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob("*.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "ok" or result["checksum_failures"] != 0:
        raise SystemExit(f"invalid transport result: {path}")
    trace = path.with_suffix(".outputs.bin")
    if not trace.is_file() or trace.stat().st_size <= 8:
        raise SystemExit(f"missing application output trace: {trace}")
    print(
        f"{path.stem}: handoff p50={result['handoff_us']['p50']:.3f} us, "
        f"p99={result['handoff_us']['p99']:.3f} us, "
        f"end-to-end p99={result['end_to_end_us']['p99']:.3f} us"
    )
PY

find "${RESULT_DIR}" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"${RESULT_DIR}/SHA256SUMS"
sha256sum "${BUILD_DIR}/jdg-mig-trt-pipeline" \
  "${ROOT_DIR}/benchmarks/mig_trt_pipeline.cpp" "${PRODUCER_ENGINE}" \
  >>"${RESULT_DIR}/SHA256SUMS"
printf 'TensorRT transport smoke: %s\n' "${RESULT_DIR}"
