#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly ENGINE_ROOT="${ENGINE_ROOT:-${ROOT_DIR}/models/engines}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p6-full-multimodal-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly MPS_STATE_DIR="${MPS_STATE_DIR:-/tmp/jdg-mps-full}"
readonly CUDA_LIB="/usr/local/cuda-13.2/lib64"
readonly SAMPLES="${SAMPLES:-10000}"
readonly WARMUP="${WARMUP:-500}"
readonly REPEATS="${REPEATS:-3}"
readonly SLO_FACTOR="${SLO_FACTOR:-1.10}"
telemetry_pid=""
mps_started=0
pressure_pids=()

sudo_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

stop_pressures() {
  local pid
  for pid in "${pressure_pids[@]}"; do
    kill -INT "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${pressure_pids[@]}"; do
    wait "${pid}" || true
  done
  pressure_pids=()
}

stop_mps() {
  if [[ "${mps_started}" -eq 1 ]]; then
    printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe" \
      CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
    mps_started=0
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_pressures
  stop_mps
  if [[ -n "${telemetry_pid}" ]]; then
    kill -TERM "${telemetry_pid}" >/dev/null 2>&1 || true
    wait "${telemetry_pid}" >/dev/null 2>&1 || true
  fi
  sudo_cmd systemctl start gdm3 >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

if nvidia-smi -q | awk '/MIG Mode/{seen=1; next} seen && /Current/{print $NF; exit}' | \
    grep -qx Enabled; then
  printf 'full-GPU mode is required; disable MIG and reboot first\n' >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${MPS_STATE_DIR}/pipe" "${MPS_STATE_DIR}/log"
sudo_cmd systemctl stop gdm3
sudo_cmd jetson_clocks
cat /etc/nv_tegra_release >"${RESULT_DIR}/nv_tegra_release.txt"
nvidia-smi -q >"${RESULT_DIR}/nvidia-smi.txt"
sudo_cmd jetson_clocks --show >"${RESULT_DIR}/jetson-clocks.txt" 2>&1 || true
taskset --cpu-list 13 tegrastats --interval 100 \
  >"${RESULT_DIR}/tegrastats.txt" 2>&1 &
telemetry_pid=$!

env CUDA_VISIBLE_DEVICES=0 ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG=full \
  "${ROOT_DIR}/scripts/prepare_models.sh"

critical_args=(
  --engine "${ENGINE_ROOT}/full/resnet10-detection.engine"
  --model-name resnet10-detection
  --role benchmark
  --samples "${SAMPLES}"
  --warmup "${WARMUP}"
  --include-transfers true
)

run_critical() {
  local output="$1"
  local trace="$2"
  local deadline="$3"
  local priority="$4"
  shift 4
  local deadline_args=()
  if [[ -n "${deadline}" ]]; then
    deadline_args=(--deadline-ms "${deadline}")
  fi
  env LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$@" taskset --cpu-list 12 "${BUILD_DIR}/jdg-trt-bench" \
    "${critical_args[@]}" "${deadline_args[@]}" --priority "${priority}" \
    --trace "${trace}" >"${output}.tmp"
  mv "${output}.tmp" "${output}"
}

start_pressure() {
  local model="$1"
  local output="$2"
  local cpu="$3"
  local priority="$4"
  local engine_tag="$5"
  shift 5
  env LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$@" taskset --cpu-list "${cpu}" "${BUILD_DIR}/jdg-trt-bench" \
    --engine "${ENGINE_ROOT}/${engine_tag}/${model}.engine" \
    --model-name "${model}" --role pressure --duration-seconds 3600 \
    --warmup "${WARMUP}" --priority "${priority}" \
    --include-transfers true >"${output}.tmp" &
  pressure_pids+=("$!")
}

run_time_division_pair() {
  local name="$1"
  local repetition="$2"
  local pressure_priority="$3"
  local pressure_engine_tag="$4"
  shift 4
  local pressure_env=(CUDA_VISIBLE_DEVICES=0)
  local critical_env=(CUDA_VISIBLE_DEVICES=0)

  run_critical "${RESULT_DIR}/${name}-r${repetition}-critical.json" \
    "${RESULT_DIR}/${name}-r${repetition}-critical.csv" "${deadline_ms}" \
    default "${critical_env[@]}"

  pressure_pids=()
  start_pressure distilbert-sst2 \
    "${RESULT_DIR}/${name}-r${repetition}-language.json" 0 \
    "${pressure_priority}" "${pressure_engine_tag}" "${pressure_env[@]}"
  start_pressure whisper-tiny-encoder \
    "${RESULT_DIR}/${name}-r${repetition}-audio.json" 1 \
    "${pressure_priority}" "${pressure_engine_tag}" "${pressure_env[@]}"
  sleep 1
  stop_pressures
  mv "${RESULT_DIR}/${name}-r${repetition}-language.json.tmp" \
    "${RESULT_DIR}/${name}-r${repetition}-language.json"
  mv "${RESULT_DIR}/${name}-r${repetition}-audio.json.tmp" \
    "${RESULT_DIR}/${name}-r${repetition}-audio.json"
}

run_pair() {
  local name="$1"
  local repetition="$2"
  local quota="$3"
  local use_mps="$4"
  local critical_priority="$5"
  local pressure_priority="$6"
  local pressure_engine_tag="$7"
  local pressure_env=(CUDA_VISIBLE_DEVICES=0)
  local critical_env=(CUDA_VISIBLE_DEVICES=0)
  if [[ "${use_mps}" == "true" ]]; then
    pressure_env+=(
      CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe"
      CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log"
      CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${quota}"
    )
    critical_env+=(
      CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe"
      CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log"
    )
  fi
  pressure_pids=()
  start_pressure distilbert-sst2 \
    "${RESULT_DIR}/${name}-r${repetition}-language.json" 0 \
    "${pressure_priority}" "${pressure_engine_tag}" "${pressure_env[@]}"
  start_pressure whisper-tiny-encoder \
    "${RESULT_DIR}/${name}-r${repetition}-audio.json" 1 \
    "${pressure_priority}" "${pressure_engine_tag}" "${pressure_env[@]}"
  sleep 1
  run_critical "${RESULT_DIR}/${name}-r${repetition}-critical.json" \
    "${RESULT_DIR}/${name}-r${repetition}-critical.csv" "${deadline_ms}" \
    "${critical_priority}" "${critical_env[@]}"
  stop_pressures
  mv "${RESULT_DIR}/${name}-r${repetition}-language.json.tmp" \
    "${RESULT_DIR}/${name}-r${repetition}-language.json"
  mv "${RESULT_DIR}/${name}-r${repetition}-audio.json.tmp" \
    "${RESULT_DIR}/${name}-r${repetition}-audio.json"
}

for repetition in $(seq 1 "${REPEATS}"); do
  run_critical "${RESULT_DIR}/isolated-r${repetition}-critical.json" \
    "${RESULT_DIR}/isolated-r${repetition}-critical.csv" "" default \
    CUDA_VISIBLE_DEVICES=0
done
deadline_ms="$(python3 - "${RESULT_DIR}" "${SLO_FACTOR}" <<'PY'
import glob
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
factor = float(sys.argv[2])
values = []
for path in glob.glob(str(directory / "isolated-r*-critical.json")):
    with open(path, encoding="utf-8") as source:
        values.append(json.load(source)["release_to_completion"]["p99_ms"])
print(max(values) * factor)
PY
)"
printf '%s\n' "${deadline_ms}" >"${RESULT_DIR}/deadline-ms.txt"

for repetition in $(seq 1 "${REPEATS}"); do
  run_pair native "${repetition}" 100 false default default full
  run_pair native-priority "${repetition}" 100 false high low full
  run_time_division_pair time-division "${repetition}" default full
done

find "${MPS_STATE_DIR}/pipe" -mindepth 1 -maxdepth 1 -delete
CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe" \
CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log" nvidia-cuda-mps-control -d
mps_started=1
for quota in 25 50 100; do
  env CUDA_VISIBLE_DEVICES=0 \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log" \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${quota}" \
    ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG="full-q${quota}" \
    BUILD_MODELS=distilbert-sst2,whisper-tiny-encoder \
    "${ROOT_DIR}/scripts/prepare_models.sh"
done
for quota in 25 50 100; do
  for repetition in $(seq 1 "${REPEATS}"); do
    run_pair "mps-q${quota}" "${repetition}" "${quota}" true default \
      default "full-q${quota}"
  done
done
for repetition in $(seq 1 "${REPEATS}"); do
  run_pair mps-priority-q25 "${repetition}" 25 true high low full-q25
done

python3 "${ROOT_DIR}/analysis/summarize_multimodal.py" "${RESULT_DIR}" \
  >"${RESULT_DIR}/summary.json.tmp"
mv "${RESULT_DIR}/summary.json.tmp" "${RESULT_DIR}/summary.json"
printf 'P6 full-GPU multimodal results: %s\n' "${RESULT_DIR}"
