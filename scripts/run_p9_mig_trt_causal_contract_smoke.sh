#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly PRODUCER_ENGINE="${PRODUCER_ENGINE:-${ROOT_DIR}/models/engines/mig-1g-q100/resnet10-detection.engine}"
readonly INPUT_TRACE="${INPUT_TRACE:-${ROOT_DIR}/results/p9-coco8-resnet10-labelled-smoke-20260811-r03/inputs.bin}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-mig-trt-causal-contract-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly WARMUP="${WARMUP:-2}"
readonly ITERATIONS="${ITERATIONS:-19}"
readonly CONTROL_CPU="${CONTROL_CPU:-13}"

for path in "${BUILD_DIR}/jdg-mig-trt-pipeline" "${MIG_ENV}" \
  "${PRODUCER_ENGINE}" "${INPUT_TRACE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required artifact: ${path}" >&2
    exit 1
  fi
done
[[ ! -e "${RESULT_DIR}" ]] || { echo "result directory exists: ${RESULT_DIR}" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_SMALL_UUID:?missing JDG_MIG_SMALL_UUID}"
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"
: "${JDG_MPS_PIPE_DIRECTORY:?missing JDG_MPS_PIPE_DIRECTORY}"

mkdir -p "${RESULT_DIR}"
cp "${MIG_ENV}" "${RESULT_DIR}/mig.env"
cp "${INPUT_TRACE}" "${RESULT_DIR}/producer-input-trace.bin"
python3 "${ROOT_DIR}/scripts/build_operational_arrival_trace.py" \
  --producer-input-trace "${INPUT_TRACE}" \
  --warmup "${WARMUP}" --requests "${ITERATIONS}" --period-us 2000 \
  --output "${RESULT_DIR}/arrival-trace.bin" \
  >"${RESULT_DIR}/arrival-trace.json"
python3 "${ROOT_DIR}/scripts/verify_operational_arrival_trace.py" \
  "${RESULT_DIR}/arrival-trace.bin" >"${RESULT_DIR}/arrival-trace-verified.json"
sha256sum "${BUILD_DIR}/jdg-mig-trt-pipeline" "${ROOT_DIR}/benchmarks/mig_trt_pipeline.cpp" \
  "${PRODUCER_ENGINE}" "${INPUT_TRACE}" "${RESULT_DIR}/arrival-trace.bin" >"${RESULT_DIR}/SHA256SUMS"

common_args=(
  --producer-engine "${PRODUCER_ENGINE}"
  --producer "${JDG_MIG_SMALL_UUID}"
  --producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}"
  --transport registered-direct
  --workload resnet-control
  --warmup "${WARMUP}"
  --iterations "${ITERATIONS}"
  --producer-input-trace "${INPUT_TRACE}"
  --arrival-trace "${RESULT_DIR}/arrival-trace.bin"
)

taskset --cpu-list "${CONTROL_CPU}" \
  "${BUILD_DIR}/jdg-mig-trt-pipeline" "${common_args[@]}" \
  --capture-activation-trace "${RESULT_DIR}/activation-replay.bin" \
  >"${RESULT_DIR}/capture.json"
python3 "${ROOT_DIR}/scripts/verify_activation_replay_trace.py" \
  "${RESULT_DIR}/activation-replay.bin" >"${RESULT_DIR}/activation-replay.json"

run_arm() {
  local mode="$1"
  local arm_dir="${RESULT_DIR}/${mode}"
  mkdir -p "${arm_dir}"
  taskset --cpu-list "${CONTROL_CPU}" \
    "${BUILD_DIR}/jdg-mig-trt-pipeline" "${common_args[@]}" \
    --consumer "${JDG_MIG_BIG_UUID}" \
    --dependency-mode "${mode}" \
    --activation-replay-trace "${RESULT_DIR}/activation-replay.bin" \
    --trace-csv "${arm_dir}/trace.csv" \
    --event-trace-csv "${arm_dir}/events.csv" \
    --checksum-trace-csv "${arm_dir}/checksums.csv" \
    --application-output-trace "${arm_dir}/outputs.bin" \
    >"${arm_dir}/pipeline.json"
}

run_arm dependent
run_arm independent

python3 - "${RESULT_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
dependent = json.loads((root / "dependent/pipeline.json").read_text())
independent = json.loads((root / "independent/pipeline.json").read_text())
for mode, value in (("dependent", dependent), ("independent", independent)):
    if value.get("status") != "ok" or value.get("checksum_failures") != 0:
        raise SystemExit(f"{mode} causal contract failed")
    if value.get("activation_replay_verified_requests") != value["warmup"] + value["iterations"]:
        raise SystemExit(f"{mode} activation replay was not fully verified")
    if value.get("arrival_schedule_mode") != "operational-trace":
        raise SystemExit(f"{mode} did not consume the operational arrival trace")
    events = root / mode / "events.csv"
    if not events.is_file() or len(events.read_text().splitlines()) != value["iterations"] + 1:
        raise SystemExit(f"{mode} event trace is incomplete")
if independent.get("correctness_scope") != "producer-activation-replay-output-oracle":
    raise SystemExit("independent arm still advertises a local synthetic input")
if (root / "dependent/outputs.bin").read_bytes() != (root / "independent/outputs.bin").read_bytes():
    raise SystemExit("dependent and independent output traces differ")
if (root / "dependent/checksums.csv").read_bytes() != (root / "independent/checksums.csv").read_bytes():
    raise SystemExit("dependent and independent activation checksum traces differ")
print("QUIET same-activation causal contract passed")
PY

find "${RESULT_DIR}" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >>"${RESULT_DIR}/SHA256SUMS"
printf 'QUIET causal contract smoke: %s\n' "${RESULT_DIR}"
