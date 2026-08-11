#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly HANDOFF="${BUILD_DIR}/jdg-mig-sysmem-handoff"
readonly RING="${BUILD_DIR}/jdg-mig-sysmem-ring"
readonly PRESSURE_BENCH="${BUILD_DIR}/jdg-bench"
readonly CUDA_LIB="${CUDA_LIB:-/usr/local/cuda-13.2/lib64}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p1-mig-sysmem-characterization-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly SIZES="${SIZES:-64,4096,65536,1048576,8388608}"
readonly WARMUP="${WARMUP:-2}"
readonly ITERATIONS="${ITERATIONS:-8}"
readonly CACHE_FLUSH_BYTES="${CACHE_FLUSH_BYTES:-67108864}"
readonly CONTROL_CPU="${CONTROL_CPU:-13}"
readonly PRESSURE_CPU="${PRESSURE_CPU:-0}"
readonly PRESSURE_SECONDS="${PRESSURE_SECONDS:-6}"
readonly TRANSPORTS="${TRANSPORTS:-registered-direct,pageable-direct-control,pinned-bounce,pageable-bounce,managed-uvm-control,host-materialize-control}"
readonly DIRECTIONS="${DIRECTIONS:-small-to-big,big-to-small}"
readonly CACHE_STATES="${CACHE_STATES:-warm,cold}"
readonly PRESSURE_MODES="${PRESSURE_MODES:-none}"
readonly QUEUE_DEPTHS="${QUEUE_DEPTHS:-1,3}"
PRESSURE_PID=""
export SIZES WARMUP ITERATIONS CACHE_FLUSH_BYTES CONTROL_CPU PRESSURE_CPU
export PRESSURE_SECONDS TRANSPORTS DIRECTIONS CACHE_STATES PRESSURE_MODES QUEUE_DEPTHS

for path in "${HANDOFF}" "${RING}" "${PRESSURE_BENCH}" "${MIG_ENV}"; do
  if [[ ! -x "${path}" && "${path}" != "${MIG_ENV}" ]]; then
    echo "missing executable: ${path}" >&2
    exit 1
  fi
  if [[ "${path}" == "${MIG_ENV}" && ! -r "${path}" ]]; then
    echo "missing MIG environment: ${path}" >&2
    exit 1
  fi
done

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_SMALL_UUID:?missing JDG_MIG_SMALL_UUID}"
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"
: "${JDG_MPS_PIPE_DIRECTORY:?missing JDG_MPS_PIPE_DIRECTORY}"

mkdir -p "${RESULT_DIR}/runs" "${RESULT_DIR}/negative-controls"
cp "${MIG_ENV}" "${RESULT_DIR}/mig.env"
nvidia-smi -L >"${RESULT_DIR}/gpu-inventory.txt"
sha256sum "${HANDOFF}" "${ROOT_DIR}/benchmarks/mig_sysmem_handoff.cu" \
  "${RING}" "${ROOT_DIR}/benchmarks/mig_sysmem_ring.cu" \
  "${PRESSURE_BENCH}" >"${RESULT_DIR}/SHA256SUMS"

python3 - "${RESULT_DIR}/run-metadata.json" <<'PY'
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
output.write_text(json.dumps({
    "schema_version": 1,
    "kind": "p1-mig-sysmem-characterization",
    "transport_description": "full-coherent registered system-memory activation edge",
    "sizes": [int(value) for value in os.environ["SIZES"].split(",")],
    "warmup": int(os.environ["WARMUP"]),
    "iterations": int(os.environ["ITERATIONS"]),
    "cache_flush_bytes": int(os.environ["CACHE_FLUSH_BYTES"]),
    "control_cpu": os.environ["CONTROL_CPU"],
    "pressure_cpu": os.environ["PRESSURE_CPU"],
    "pressure_seconds": float(os.environ["PRESSURE_SECONDS"]),
    "transports": os.environ["TRANSPORTS"].split(","),
    "directions": os.environ["DIRECTIONS"].split(","),
    "cache_states": os.environ["CACHE_STATES"].split(","),
    "pressure_modes": os.environ["PRESSURE_MODES"].split(","),
    "queue_depths": [int(value) for value in os.environ["QUEUE_DEPTHS"].split(",")],
}, indent=2) + "\n", encoding="utf-8")
PY

validate_result() {
  local result="$1"
  python3 - "${result}" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("status") != "ok":
    raise SystemExit(f"transport characterization did not complete: {value.get('status')}")
if value.get("ring_schema") == "JDG_RING1":
    if value.get("queue_depth") != 3:
        raise SystemExit("ring queue depth is not three")
    events = value.get("events", [])
    if not events or any(event.get("mismatches") != 0 for event in events):
        raise SystemExit("ring result has missing events or checksum mismatches")
    if value.get("cache_state") not in {"warm", "cold"}:
        raise SystemExit("ring cache state is missing")
    raise SystemExit(0)
for row in value.get("results", []):
    if row.get("samples", 0) <= 0 or row.get("mismatches") != 0:
        raise SystemExit(f"invalid transport result row: {row}")
if value.get("cache_state") not in {"warm", "cold"}:
    raise SystemExit("cache state is missing from transport result")
PY
}

run_pressure() {
  local mode="$1"
  local uuid="$2"
  local output="$3"
  if [[ "${mode}" == "none" ]]; then
    PRESSURE_PID=""
    return
  fi
  CUDA_VISIBLE_DEVICES="${uuid}" \
    LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    taskset --cpu-list "${PRESSURE_CPU}" \
    "${PRESSURE_BENCH}" --role pressure --background "${mode}" \
    --duration-seconds "${PRESSURE_SECONDS}" >"${output}" 2>&1 &
  PRESSURE_PID=$!
  sleep 0.5
}

stop_pressure() {
  if [[ -z "${PRESSURE_PID}" ]]; then
    return
  fi
  wait "${PRESSURE_PID}"
  PRESSURE_PID=""
}

run_case() {
  local direction="$1"
  local cache_state="$2"
  local pressure_mode="$3"
  local transport="$4"
  local queue_depth="$5"
  local producer="${JDG_MIG_SMALL_UUID}"
  local consumer="${JDG_MIG_BIG_UUID}"
  local mps_pipe_args=(--producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}")
  local ring_mps_pipe_args=(--mps-pipe "${JDG_MPS_PIPE_DIRECTORY}")
  local context_env=()
  if [[ "${direction}" == "big-to-small" ]]; then
    producer="${JDG_MIG_BIG_UUID}"
    consumer="${JDG_MIG_SMALL_UUID}"
    mps_pipe_args=()
    ring_mps_pipe_args=()
    context_env=(env -u JDG_MPS_PIPE_DIRECTORY -u JDG_MPS_LOG_DIRECTORY \
      -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY)
  elif [[ "${direction}" != "small-to-big" ]]; then
    echo "unsupported direction: ${direction}" >&2
    exit 1
  fi
  local case_id="${direction}__${cache_state}__${pressure_mode}__q${queue_depth}__${transport}"
  local case_dir="${RESULT_DIR}/runs/${case_id}"
  mkdir -p "${case_dir}"
  python3 - "${case_dir}/parameters.json" "${direction}" "${cache_state}" \
    "${pressure_mode}" "${transport}" "${producer}" "${consumer}" \
    "${queue_depth}" <<'PY'
import json
import pathlib
import sys

output, direction, cache_state, pressure, transport, producer, consumer, queue_depth = sys.argv[1:]
pathlib.Path(output).write_text(json.dumps({
    "direction": direction,
    "cache_state": cache_state,
    "pressure_mode": pressure,
    "transport_mode": transport,
    "queue_depth": int(queue_depth),
    "producer_uuid": producer,
    "consumer_uuid": consumer,
}, indent=2) + "\n", encoding="utf-8")
PY
  run_pressure "${pressure_mode}" "${consumer}" "${case_dir}/pressure.json"
  set +e
  if [[ "${queue_depth}" == "1" ]]; then
    "${context_env[@]}" taskset --cpu-list "${CONTROL_CPU}" "${HANDOFF}" \
      --producer "${producer}" \
      --consumer "${consumer}" \
      "${mps_pipe_args[@]}" \
      --transport "${transport}" \
      --sizes "${SIZES}" \
      --warmup "${WARMUP}" \
      --iterations "${ITERATIONS}" \
      --cache-state "${cache_state}" \
      --cache-flush-bytes "${CACHE_FLUSH_BYTES}" \
      >"${case_dir}/handoff.json"
  elif [[ "${queue_depth}" == "3" && "${transport}" == "registered-direct" ]]; then
    "${context_env[@]}" taskset --cpu-list "${CONTROL_CPU}" "${RING}" \
      --producer "${producer}" \
      --consumer "${consumer}" \
      "${ring_mps_pipe_args[@]}" \
      --payload-bytes "${SIZES%%,*}" \
      --requests "${ITERATIONS}" \
      --cache-state "${cache_state}" \
      --cache-flush-bytes "${CACHE_FLUSH_BYTES}" \
      >"${case_dir}/handoff.json"
  else
    set -e
    stop_pressure
    return 0
  fi
  local status=$?
  set -e
  stop_pressure
  if [[ "${status}" -ne 0 ]]; then
    echo "characterization failed: ${case_id}" >&2
    exit "${status}"
  fi
  validate_result "${case_dir}/handoff.json"
}

IFS=',' read -r -a direction_values <<<"${DIRECTIONS}"
IFS=',' read -r -a cache_values <<<"${CACHE_STATES}"
IFS=',' read -r -a pressure_values <<<"${PRESSURE_MODES}"
IFS=',' read -r -a queue_values <<<"${QUEUE_DEPTHS}"
IFS=',' read -r -a transport_values <<<"${TRANSPORTS}"

for direction in "${direction_values[@]}"; do
  local_producer="${JDG_MIG_SMALL_UUID}"
  local_consumer="${JDG_MIG_BIG_UUID}"
  if [[ "${direction}" == "big-to-small" ]]; then
    local_producer="${JDG_MIG_BIG_UUID}"
    local_consumer="${JDG_MIG_SMALL_UUID}"
  fi
  negative_mps_pipe_args=(--producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}")
  negative_context_env=()
  if [[ "${direction}" == "big-to-small" ]]; then
    negative_mps_pipe_args=()
    negative_context_env=(env -u JDG_MPS_PIPE_DIRECTORY -u JDG_MPS_LOG_DIRECTORY \
      -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY)
  fi
  "${negative_context_env[@]}" taskset --cpu-list "${CONTROL_CPU}" "${HANDOFF}" \
    --producer "${local_producer}" \
    --consumer "${local_consumer}" \
    "${negative_mps_pipe_args[@]}" \
    --transport p2p-ipc-negative-control \
    >"${RESULT_DIR}/negative-controls/${direction}.json"
done

for direction in "${direction_values[@]}"; do
  for cache_state in "${cache_values[@]}"; do
    for pressure_mode in "${pressure_values[@]}"; do
      for queue_depth in "${queue_values[@]}"; do
        if [[ "${queue_depth}" != "1" && "${queue_depth}" != "3" ]]; then
          echo "unsupported queue depth: ${queue_depth}" >&2
          exit 1
        fi
        for transport in "${transport_values[@]}"; do
          run_case "${direction}" "${cache_state}" "${pressure_mode}" \
            "${transport}" "${queue_depth}"
        done
      done
    done
  done
done

python3 - "${RESULT_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted((root / "runs").glob("*/handoff.json")):
    parameters = json.loads((path.parent / "parameters.json").read_text())
    result = json.loads(path.read_text())
    rows.append({
        **parameters,
        "case_kind": "ring" if result.get("ring_schema") == "JDG_RING1" else "handoff",
        "status": result["status"],
        "transport_description": result["transport_description"],
        "cache_flush_bytes": result["cache_flush_bytes"],
        "results": result.get("results", []),
        "ring_events": len(result.get("events", [])),
        "ring_counters": result.get("counters"),
        "path": str(path.relative_to(root)),
    })
summary = {
    "schema_version": 1,
    "kind": "p1-mig-sysmem-characterization-summary",
    "transport_description": "full-coherent registered system-memory activation edge",
    "case_count": len(rows),
    "cases": rows,
    "negative_controls": [
        json.loads(path.read_text())
        for path in sorted((root / "negative-controls").glob("*.json"))
    ],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

find "${RESULT_DIR}" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >>"${RESULT_DIR}/SHA256SUMS"
printf 'P1 MIG system-memory characterization: %s\n' "${RESULT_DIR}"
