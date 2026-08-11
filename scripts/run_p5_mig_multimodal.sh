#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly ENGINE_ROOT="${ENGINE_ROOT:-${ROOT_DIR}/models/engines}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p5-mig-multimodal-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly SMALL_STATE_DIR="${SMALL_STATE_DIR:-/tmp/jdg-mps-1g}"
readonly BIG_STATE_DIR="${BIG_STATE_DIR:-/tmp/jdg-mps-2g}"
readonly CUDA_LIB="/usr/local/cuda-13.2/lib64"
readonly SAMPLES="${SAMPLES:-2000}"
readonly WARMUP="${WARMUP:-100}"
readonly REPEATS="${REPEATS:-3}"
readonly SLO_FACTOR="${SLO_FACTOR:-1.10}"
telemetry_pid=""
big_mps_started=0
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
  local failed=0
  for pid in "${pressure_pids[@]}"; do
    kill -INT "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${pressure_pids[@]}"; do
    if ! wait "${pid}"; then
      printf 'pressure process %s failed\n' "${pid}" >&2
      failed=1
    fi
  done
  pressure_pids=()
  return "${failed}"
}

stop_big_mps() {
  if [[ "${big_mps_started}" -eq 1 ]]; then
    printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
      CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
    big_mps_started=0
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_pressures || true
  stop_big_mps
  if [[ -n "${telemetry_pid}" ]]; then
    kill -TERM "${telemetry_pid}" >/dev/null 2>&1 || true
    wait "${telemetry_pid}" >/dev/null 2>&1 || true
  fi
  sudo_cmd systemctl start gdm3 >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

SUDO_PASSWORD="${SUDO_PASSWORD:-}" "${ROOT_DIR}/scripts/configure_thor_mig.sh"
# shellcheck disable=SC1091
source "${SMALL_STATE_DIR}/mig.env"
mkdir -p "${RESULT_DIR}" "${BIG_STATE_DIR}/pipe" "${BIG_STATE_DIR}/log"
sudo_cmd systemctl stop gdm3
sudo_cmd jetson_clocks

cat /etc/nv_tegra_release >"${RESULT_DIR}/nv_tegra_release.txt"
nvidia-smi -q >"${RESULT_DIR}/nvidia-smi.txt"
sudo_cmd nvidia-smi mig -lgip >"${RESULT_DIR}/mig-profiles.txt"
sudo_cmd jetson_clocks --show >"${RESULT_DIR}/jetson-clocks.txt" 2>&1 || true
taskset --cpu-list 13 tegrastats --interval 100 \
  >"${RESULT_DIR}/tegrastats.txt" 2>&1 &
telemetry_pid=$!

env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
  ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG=mig-2g \
  "${ROOT_DIR}/scripts/prepare_models.sh"
for quota in 25 50 100; do
  env CUDA_VISIBLE_DEVICES="${JDG_MIG_SMALL_UUID}" \
    CUDA_MPS_PIPE_DIRECTORY="${JDG_MPS_PIPE_DIRECTORY}" \
    CUDA_MPS_LOG_DIRECTORY="${JDG_MPS_LOG_DIRECTORY}" \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${quota}" \
    ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG="mig-1g-q${quota}" \
    BUILD_MODELS=distilbert-sst2,whisper-tiny-encoder \
    "${ROOT_DIR}/scripts/prepare_models.sh"
done

critical_args=(
  --engine "${ENGINE_ROOT}/mig-2g/resnet10-detection.engine"
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
  if ! env LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$@" taskset --cpu-list 12 "${BUILD_DIR}/jdg-trt-bench" \
    "${critical_args[@]}" "${deadline_args[@]}" --priority "${priority}" \
    --trace "${trace}" >"${output}.tmp" 2>"${output}.err"; then
    mv "${output}.tmp" "${output}.failed.stdout"
    mv "${output}.err" "${output}.failed.stderr"
    return 1
  fi
  mv "${output}.tmp" "${output}"
  if [[ -f "${output}.err" ]]; then
    mv "${output}.err" "${output}.err.txt"
  fi
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
    --model-name "${model}" \
    --role pressure --duration-seconds 3600 --warmup "${WARMUP}" \
    --priority "${priority}" --include-transfers true \
    >"${output}.tmp" 2>"${output}.err" &
  pressure_pids+=("$!")
}

run_pair() {
  local name="$1"
  local repetition="$2"
  local pressure_uuid="$3"
  local quota="$4"
  local pressure_pipe="$5"
  local critical_pipe="$6"
  local critical_priority="$7"
  local pressure_priority="$8"
  local pressure_engine_tag="$9"
  local pressure_env=(CUDA_VISIBLE_DEVICES="${pressure_uuid}")
  local critical_env=(CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}")
  if [[ -n "${pressure_pipe}" ]]; then
    pressure_env+=(
      CUDA_MPS_PIPE_DIRECTORY="${pressure_pipe}"
      CUDA_MPS_LOG_DIRECTORY="${pressure_pipe%/pipe}/log"
      CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${quota}"
    )
  fi
  if [[ -n "${critical_pipe}" ]]; then
    critical_env+=(
      CUDA_MPS_PIPE_DIRECTORY="${critical_pipe}"
      CUDA_MPS_LOG_DIRECTORY="${critical_pipe%/pipe}/log"
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
    CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}"
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
  run_pair same-naive "${repetition}" "${JDG_MIG_BIG_UUID}" 100 "" "" \
    default default mig-2g
done

# The small-instance MPS context already exists, so warming 2g before starting
# its MPS server preserves the R39.2 context ordering requirement.
run_critical "${RESULT_DIR}/big-mps-warm.json" \
  "${RESULT_DIR}/big-mps-warm.csv" "" default \
  CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}"
find "${BIG_STATE_DIR}/pipe" -mindepth 1 -maxdepth 1 -delete
CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" nvidia-cuda-mps-control -d
big_mps_started=1

for quota in 25 50 100; do
  env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
    CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${quota}" \
    ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG="mig-2g-q${quota}" \
    BUILD_MODELS=distilbert-sst2,whisper-tiny-encoder \
    "${ROOT_DIR}/scripts/prepare_models.sh"
done

for quota in 25 50 100; do
  for repetition in $(seq 1 "${REPEATS}"); do
    run_pair "same-mps-q${quota}" "${repetition}" "${JDG_MIG_BIG_UUID}" \
      "${quota}" "${BIG_STATE_DIR}/pipe" "${BIG_STATE_DIR}/pipe" \
      default default "mig-2g-q${quota}"
  done
done
for repetition in $(seq 1 "${REPEATS}"); do
  run_pair same-mps-priority-q25 "${repetition}" "${JDG_MIG_BIG_UUID}" 25 \
    "${BIG_STATE_DIR}/pipe" "${BIG_STATE_DIR}/pipe" high low mig-2g-q25
done
stop_big_mps

for quota in 25 50 100; do
  for repetition in $(seq 1 "${REPEATS}"); do
    run_pair "cross-mps-q${quota}" "${repetition}" "${JDG_MIG_SMALL_UUID}" \
      "${quota}" "${JDG_MPS_PIPE_DIRECTORY}" "" default default \
      "mig-1g-q${quota}"
  done
done

python3 "${ROOT_DIR}/analysis/summarize_multimodal.py" "${RESULT_DIR}" \
  >"${RESULT_DIR}/summary.json.tmp"
mv "${RESULT_DIR}/summary.json.tmp" "${RESULT_DIR}/summary.json"
printf 'P5 MIG multimodal results: %s\n' "${RESULT_DIR}"
