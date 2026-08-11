#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly ENGINE_ROOT="${ENGINE_ROOT:-${ROOT_DIR}/models/engines}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p7-multimodal-governor-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly STATE_DIR="${STATE_DIR:-/tmp/jdg-mps-1g}"
telemetry_pid=""

sudo_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${telemetry_pid}" ]]; then
    kill -TERM "${telemetry_pid}" >/dev/null 2>&1 || true
    wait "${telemetry_pid}" >/dev/null 2>&1 || true
  fi
  sudo_cmd systemctl start gdm3 >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

SUDO_PASSWORD="${SUDO_PASSWORD:-}" "${ROOT_DIR}/scripts/configure_thor_mig.sh"

if ! nvidia-smi -q | awk '/MIG Mode/{seen=1; next} seen && /Current/{print $NF; exit}' | \
    grep -qx Enabled; then
  printf 'MIG mode is still disabled after configure_thor_mig.sh\n' >&2
  exit 1
fi

if [[ ! -f "${STATE_DIR}/mig.env" ]]; then
  printf 'missing MIG environment file: %s/mig.env\n' "${STATE_DIR}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"
sudo_cmd systemctl stop gdm3
sudo_cmd jetson_clocks
cat /etc/nv_tegra_release >"${RESULT_DIR}/nv_tegra_release.txt"
nvidia-smi -q >"${RESULT_DIR}/nvidia-smi.txt"
sudo_cmd jetson_clocks --show >"${RESULT_DIR}/jetson-clocks.txt" 2>&1 || true
taskset --cpu-list "${TELEMETRY_CPU:-13}" tegrastats --interval 100 \
  >"${RESULT_DIR}/tegrastats.txt" 2>&1 &
telemetry_pid=$!

python3 "${ROOT_DIR}/runtime/multimodal_governor.py" \
  --bench "${BUILD_DIR}/jdg-trt-bench" \
  --engine-root "${ENGINE_ROOT}" \
  --mig-env "${STATE_DIR}/mig.env" \
  --output "${RESULT_DIR}/summary.json" \
  --epochs "${EPOCHS:-12}" \
  --samples "${SAMPLES:-5000}" \
  --warmup "${WARMUP:-250}" \
  --slo-factor "${SLO_FACTOR:-1.10}" \
  --pressure-startup-seconds "${PRESSURE_STARTUP_SECONDS:-0.5}" \
  --serial-window-seconds "${SERIAL_WINDOW_SECONDS:-1.0}" \
  --critical-cpu "${CRITICAL_CPU:-12}" \
  --pressure-cpus "${PRESSURE_CPUS:-0-11}" \
  --policy-order "${POLICY_ORDER:-static-q25,static-q100,time-division,profiled,joint-governor}"

printf 'P7 multimodal governor results: %s\n' "${RESULT_DIR}"
