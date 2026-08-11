#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p2-full-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly STATE_DIR="${STATE_DIR:-/tmp/jdg-mps-full}"
readonly PIPE_DIR="${STATE_DIR}/pipe"
readonly LOG_DIR="${STATE_DIR}/log"
readonly CUDA_LIB="/usr/local/cuda-13.2/lib64"
readonly SAMPLES="${SAMPLES:-100000}"
readonly WARMUP="${WARMUP:-1000}"
readonly PRESSURE_SECONDS="${PRESSURE_SECONDS:-8}"
telemetry_pid=""
mps_started=0

sudo_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

stop_mps() {
  if [[ "${mps_started}" -eq 1 ]]; then
    printf 'quit\n' | CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
    mps_started=0
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_mps
  if [[ -n "${telemetry_pid}" ]]; then
    kill -TERM "${telemetry_pid}" >/dev/null 2>&1 || true
    wait "${telemetry_pid}" >/dev/null 2>&1 || true
  fi
  sudo_cmd systemctl unset-environment MIG_DEVICE_UUID >/dev/null 2>&1 || true
  sudo_cmd systemctl start gdm3 >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | \
     grep -qx Enabled; then
  echo "Full GPU mode is required; disable MIG and reboot first" >&2
  exit 1
fi
if [[ ! -x "${BUILD_DIR}/jdg-bench" ]]; then
  echo "missing ${BUILD_DIR}/jdg-bench" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${PIPE_DIR}" "${LOG_DIR}"
sudo_cmd systemctl stop gdm3
sudo_cmd jetson_clocks

cat /etc/nv_tegra_release >"${RESULT_DIR}/nv_tegra_release.txt"
nvidia-smi -q >"${RESULT_DIR}/nvidia-smi.txt"
nvpmodel -q >"${RESULT_DIR}/nvpmodel.txt" 2>&1 || true
sudo_cmd jetson_clocks --show >"${RESULT_DIR}/jetson-clocks.txt" 2>&1 || true
tegrastats --interval 100 >"${RESULT_DIR}/tegrastats.txt" 2>&1 &
telemetry_pid=$!

common_env=(env -u CUDA_VISIBLE_DEVICES
            LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")
critical_env=()
common_args=(--samples "${SAMPLES}" --warmup "${WARMUP}")

run_critical() {
  local output="$1"
  shift
  "${common_env[@]}" "${critical_env[@]}" "${BUILD_DIR}/jdg-bench" \
    "${common_args[@]}" "$@" \
    >"${output}.tmp"
  mv "${output}.tmp" "${output}"
}

run_process_pair() {
  local name="$1"
  local mode="$2"
  shift 2
  "${common_env[@]}" "$@" "${BUILD_DIR}/jdg-bench" --role pressure \
    --background "${mode}" --duration-seconds "${PRESSURE_SECONDS}" \
    >"${RESULT_DIR}/${name}-pressure.json" &
  local pressure_pid=$!
  sleep 1
  run_critical "${RESULT_DIR}/${name}.json" --background none
  wait "${pressure_pid}"
}

run_critical "${RESULT_DIR}/isolated.json" --background none
run_critical "${RESULT_DIR}/green-isolated.json" --background none \
  --isolation green --green-sm-count 8
for mode in compute memory; do
  run_critical "${RESULT_DIR}/same-process-${mode}.json" \
    --background "${mode}"
  run_critical "${RESULT_DIR}/priority-${mode}.json" \
    --background "${mode}" --critical-priority high
  run_critical "${RESULT_DIR}/green-${mode}.json" \
    --background "${mode}" --isolation green --green-sm-count 8
  run_process_pair "naive-${mode}" "${mode}"
done

CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" nvidia-cuda-mps-control -d
mps_started=1
critical_env=(CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}"
              CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}")
for quota in 25 50 75 100; do
  for mode in compute memory; do
    run_process_pair "mps-q${quota}-${mode}" "${mode}" \
      CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
      CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${quota}"
  done
done

python3 "${ROOT_DIR}/analysis/summarize_full_gpu.py" "${RESULT_DIR}" \
  >"${RESULT_DIR}/summary.json.tmp"
mv "${RESULT_DIR}/summary.json.tmp" "${RESULT_DIR}/summary.json"
printf 'P2 Full GPU results: %s\n' "${RESULT_DIR}"
