#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p4-governor-$(date -u +%Y%m%dT%H%M%SZ)}"
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
mkdir -p "${RESULT_DIR}"
sudo_cmd systemctl stop gdm3
sudo_cmd jetson_clocks
cat /etc/nv_tegra_release >"${RESULT_DIR}/nv_tegra_release.txt"
nvidia-smi -q >"${RESULT_DIR}/nvidia-smi.txt"
sudo_cmd jetson_clocks --show >"${RESULT_DIR}/jetson-clocks.txt" 2>&1 || true
taskset --cpu-list "${TELEMETRY_CPUS:-13}" \
  tegrastats --interval 100 >"${RESULT_DIR}/tegrastats.txt" 2>&1 &
telemetry_pid=$!

python3 "${ROOT_DIR}/runtime/mig_governor.py" \
  --bench "${BUILD_DIR}/jdg-bench" \
  --mig-env "${STATE_DIR}/mig.env" \
  --output "${RESULT_DIR}/summary.json" \
  --epochs "${EPOCHS:-12}" \
  --samples "${SAMPLES:-10000}" \
  --warmup "${WARMUP:-500}" \
  --pressure-seconds "${PRESSURE_SECONDS:-2}" \
  --slo-factor "${SLO_FACTOR:-1.05}" \
  --critical-cpus "${CRITICAL_CPUS:-12}" \
  --pressure-cpus "${PRESSURE_CPUS:-0-11}" \
  --policy-order "${POLICY_ORDER:-static-q25,static-q100,jdg-governor}"

printf 'P4 governor results: %s\n' "${RESULT_DIR}"
