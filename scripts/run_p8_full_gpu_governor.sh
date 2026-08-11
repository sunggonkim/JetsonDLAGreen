#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly ENGINE_ROOT="${ENGINE_ROOT:-${ROOT_DIR}/models/engines}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p8-full-governor-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly MPS_STATE_DIR="${MPS_STATE_DIR:-/tmp/jdg-mps-full-p8}"
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
    printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe" \
      CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log" \
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
  if [[ "${RESTORE_GDM:-1}" -eq 1 ]]; then
    sudo_cmd systemctl start gdm3 >/dev/null 2>&1 || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | \
    grep -qx Enabled; then
  printf 'full-GPU mode is required; run prepare_full_gpu_reboot.sh first\n' >&2
  exit 1
fi
if [[ ! -x "${BUILD_DIR}/jdg-trt-bench" ]]; then
  printf 'missing benchmark: %s/jdg-trt-bench\n' "${BUILD_DIR}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${MPS_STATE_DIR}/pipe" "${MPS_STATE_DIR}/log"
sudo_cmd systemctl stop gdm3
sudo_cmd jetson_clocks
cat /etc/nv_tegra_release >"${RESULT_DIR}/nv_tegra_release.txt"
nvidia-smi -q >"${RESULT_DIR}/nvidia-smi.txt"
sudo_cmd jetson_clocks --show >"${RESULT_DIR}/jetson-clocks.txt" 2>&1
taskset --cpu-list "${TELEMETRY_CPU:-13}" tegrastats --interval 100 \
  >"${RESULT_DIR}/tegrastats.txt" 2>&1 &
telemetry_pid=$!

if [[ -S "${MPS_STATE_DIR}/pipe/control" ]]; then
  printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log" \
    nvidia-cuda-mps-control >/dev/null 2>&1 || true
  sleep 1
fi
find "${MPS_STATE_DIR}/pipe" -mindepth 1 -maxdepth 1 -delete
CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe" \
CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log" nvidia-cuda-mps-control -d
mps_started=1

for quota in 5 25; do
  env CUDA_VISIBLE_DEVICES=0 \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_STATE_DIR}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_STATE_DIR}/log" \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${quota}" \
    ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG="full-q${quota}" \
    BUILD_MODELS=distilbert-sst2,whisper-tiny-encoder \
    "${ROOT_DIR}/scripts/prepare_models.sh"
done
env CUDA_VISIBLE_DEVICES=0 ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG=full \
  BUILD_MODELS=resnet50-v2 "${ROOT_DIR}/scripts/prepare_models.sh"

governor_args=()
if [[ -n "${GUARD_OVERRIDE_MS:-}" ]]; then
  governor_args+=(--guard-override-ms "${GUARD_OVERRIDE_MS}")
fi

python3 "${ROOT_DIR}/runtime/full_gpu_governor.py" \
  --bench "${BUILD_DIR}/jdg-trt-bench" \
  --engine-root "${ENGINE_ROOT}" \
  --mps-pipe "${MPS_STATE_DIR}/pipe" \
  --mps-log "${MPS_STATE_DIR}/log" \
  --output "${RESULT_DIR}/summary.json" \
  --epochs "${EPOCHS:-12}" \
  --samples "${SAMPLES:-800}" \
  --warmup "${WARMUP:-100}" \
  --burst-size "${BURST_SIZE:-8}" \
  --period-ms "${PERIOD_MS:-12}" \
  --slo-factor "${SLO_FACTOR:-1.10}" \
  --dmr-target "${DMR_TARGET:-0.0005}" \
  --calibration-repeats "${CALIBRATION_REPEATS:-3}" \
  --pressure-startup-seconds "${PRESSURE_STARTUP_SECONDS:-0.25}" \
  --critical-cpu "${CRITICAL_CPU:-12}" \
  --pressure-cpus "${PRESSURE_CPUS:-0-11}" \
  --policy-order "${POLICY_ORDER:-static-q5,static-q25,priority-q25,conservative-guard,profiled-guard,joint-governor}" \
  --experiment-label "${EXPERIMENT_LABEL:-main}" \
  --language-guard-ms "${LANGUAGE_GUARD_MS:-1.5}" \
  --audio-guard-ms "${AUDIO_GUARD_MS:-2}" \
  "${governor_args[@]}"

printf 'P8 full-GPU governor results: %s\n' "${RESULT_DIR}"
