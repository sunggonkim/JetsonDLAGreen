#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly STATE_DIR="${STATE_DIR:-/tmp/jdg-mps-1g}"
readonly PIPE_DIR="${STATE_DIR}/pipe"
readonly LOG_DIR="${STATE_DIR}/log"
readonly ENV_FILE="${STATE_DIR}/mig.env"
readonly INIT_PROBE="${BUILD_DIR}/jdg-cuda-init-probe"
readonly WARMUP_TIMEOUT_SECONDS="${WARMUP_TIMEOUT_SECONDS:-20}"

ensure_nvidia_driver() {
  if nvidia-smi -L >/dev/null 2>&1; then
    return
  fi
  sudo_cmd modprobe nvidia
  for _ in {1..120}; do
    if nvidia-smi -L >/dev/null 2>&1; then
      return
    fi
    sleep 0.5
  done
  echo "NVIDIA driver did not become ready within 60 seconds" >&2
  return 1
}

sudo_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

run_direct_warmup() {
  local uuid="$1"
  CUDA_VISIBLE_DEVICES="${uuid}" \
  LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    timeout "${WARMUP_TIMEOUT_SECONDS}s" "${INIT_PROBE}" \
      >"${LOG_DIR}/direct-init-probe.jsonl" 2>&1
  CUDA_VISIBLE_DEVICES="${uuid}" \
  LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    timeout "${WARMUP_TIMEOUT_SECONDS}s" "${BUILD_DIR}/jdg-bench" --background none --samples 3 \
      --warmup 1 --critical-elements 65536 --critical-iterations 4 \
      >"${LOG_DIR}/direct-warmup.json" 2>"${LOG_DIR}/direct-warmup.err"
}

run_mps_warmup() {
  local uuid="$1"
  CUDA_VISIBLE_DEVICES="${uuid}" \
  CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
  LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    timeout "${WARMUP_TIMEOUT_SECONDS}s" "${INIT_PROBE}" \
      >"${LOG_DIR}/mps-init-probe.jsonl" 2>&1
  CUDA_VISIBLE_DEVICES="${uuid}" \
  CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
  LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    timeout "${WARMUP_TIMEOUT_SECONDS}s" "${BUILD_DIR}/jdg-bench" --background none --samples 3 \
      --warmup 1 --critical-elements 65536 --critical-iterations 4 \
      >"${LOG_DIR}/mps-warmup.json" 2>"${LOG_DIR}/mps-warmup.err"
}

refresh_mig_uuids() {
  small_uuid="$(nvidia-smi -L | awk '/MIG 1g/{gsub(/[()]/, "", $NF); sub(/^UUID:/, "", $NF); print $NF; exit}')"
  big_uuid="$(nvidia-smi -L | awk '/MIG 2g/{gsub(/[()]/, "", $NF); sub(/^UUID:/, "", $NF); print $NF; exit}')"
  if [[ -z "${small_uuid}" || -z "${big_uuid}" ]]; then
    echo "expected one 1g and one 2g MIG instance" >&2
    return 1
  fi
}

recreate_mig_pair() {
  sudo_cmd nvidia-smi mig -dci
  sudo_cmd nvidia-smi mig -dgi
  sudo_cmd nvidia-smi mig -cgi 83,78 -C
  refresh_mig_uuids
}

if ! grep -q '^# R39 .*REVISION: 2\.' /etc/nv_tegra_release; then
  echo "JetPack 7.2 / L4T R39.2 is required" >&2
  exit 1
fi
if [[ ! -x "${BUILD_DIR}/jdg-bench" || ! -x "${INIT_PROBE}" ]]; then
  echo "missing benchmark binaries in ${BUILD_DIR}; build the project first" >&2
  exit 1
fi

ensure_nvidia_driver

sudo_cmd install -d -m 0755 -o "$(id -u)" -g "$(id -g)" \
  "${STATE_DIR}" "${PIPE_DIR}" "${LOG_DIR}"
sudo_cmd chown -R "$(id -u):$(id -g)" "${STATE_DIR}"

mps_alive=0
if [[ -S "${PIPE_DIR}/control" ]] && \
   printf 'get_server_list\n' | env \
     CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
     CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
     timeout 2s nvidia-cuda-mps-control 2>/dev/null | grep -Eq '^[0-9]+$'; then
  mps_alive=1
fi

if [[ "${mps_alive}" -eq 0 ]]; then
  # Keep the RM driver loaded while the display stack is stopped. Enabling
  # persistence after the isolate is too late: Thor can unload the driver and
  # lose the boot-initialized MIG sysmem scrubber state.
  sudo_cmd nvidia-smi -pm 1
  # NVIDIA's Thor procedure requires isolating multi-user.target. Stopping
  # gdm3 alone can leave a graphics context that poisons the first 1g context.
  if systemctl is-active --quiet graphical.target || \
     systemctl is-active --quiet gdm3; then
    sudo_cmd systemctl isolate multi-user.target
  fi
  sudo_cmd systemctl stop gdm3 || true
  sudo_cmd systemctl stop nvargus-daemon || true
  mig_mode="$(nvidia-smi --query-gpu=mig.mode.current \
    --format=csv,noheader | tr -d '[:space:]')"
  if [[ "${mig_mode}" != "Enabled" ]]; then
    sudo_cmd nvidia-smi -mig 1
  fi
  # R39.2 can clear legacy persistence while transitioning into MIG mode.
  sudo_cmd nvidia-smi -pm 1

  gi_state="$(sudo_cmd nvidia-smi mig -lgi 2>&1 || true)"
  has_big_profile=0
  has_small_profile=0
  if grep -Eq 'MIG 2g\.0gb\+gfx[[:space:]]+83[[:space:]]+1' <<<"${gi_state}"; then
    has_big_profile=1
  fi
  if grep -Eq 'MIG 1g\.0gb\+me[[:space:]]+78[[:space:]]+2' <<<"${gi_state}"; then
    has_small_profile=1
  fi
  existing_count="$(nvidia-smi -L | grep -c 'MIG ' || true)"

  if [[ "${existing_count}" -eq 2 && "${has_big_profile}" -eq 1 && \
        "${has_small_profile}" -eq 1 ]]; then
    echo "reusing verified profiles 83 (2g+gfx) and 78 (1g+me)"
  else
    if [[ "${existing_count}" -gt 0 ]]; then
      sudo_cmd nvidia-smi mig -dci
      sudo_cmd nvidia-smi mig -dgi
      sleep 2
      if nvidia-smi -L | grep -q 'MIG '; then
        echo "MIG instances remained after destroy; refusing a stale configuration" >&2
        nvidia-smi -L >&2
        exit 1
      fi
    fi
    sudo_cmd nvidia-smi mig -cgi 83,78 -C

    gi_state="$(sudo_cmd nvidia-smi mig -lgi 2>&1 || true)"
    if ! grep -Eq 'MIG 2g\.0gb\+gfx[[:space:]]+83[[:space:]]+1' <<<"${gi_state}" || \
       ! grep -Eq 'MIG 1g\.0gb\+me[[:space:]]+78[[:space:]]+2' <<<"${gi_state}"; then
      echo "MIG profile verification failed after creation" >&2
      printf '%s\n' "${gi_state}" >&2
      exit 1
    fi
  fi
fi

small_uuid=""
big_uuid=""
refresh_mig_uuids

if [[ "${mps_alive}" -eq 0 ]]; then
  # Remove a daemon left by a prior failed bootstrap before the direct probe.
  if [[ -f "${PIPE_DIR}/nvidia-cuda-mps-control.pid" ]]; then
    stale_pid="$(cat "${PIPE_DIR}/nvidia-cuda-mps-control.pid" 2>/dev/null || true)"
    if [[ "${stale_pid}" =~ ^[0-9]+$ ]]; then
      kill -TERM "${stale_pid}" >/dev/null 2>&1 || true
      sleep 1
    fi
  fi
  find "${PIPE_DIR}" -mindepth 1 -maxdepth 1 -delete

  # On R39.2, the MPS server can hang while creating the first 1g context.
  # Bootstrap and release one direct context before starting the daemon.
  if ! run_direct_warmup "${small_uuid}"; then
    cp "${LOG_DIR}/direct-init-probe.jsonl" \
      "${LOG_DIR}/direct-init-probe.first-failure.jsonl" || true
    echo "1g CUDA bootstrap failed; recreating the R39.2 MIG pair once" >&2
    recreate_mig_pair
    if ! run_direct_warmup "${small_uuid}"; then
      echo "failed to bootstrap the direct 1g CUDA context after MIG recreation; see ${LOG_DIR}/direct-init-probe.jsonl" >&2
      exit 1
    fi
  fi

  CUDA_VISIBLE_DEVICES="${small_uuid}" \
  CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
    nvidia-cuda-mps-control -d

  if ! run_mps_warmup "${small_uuid}"; then
    printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
    echo "failed to bootstrap the 1g MPS client; see ${LOG_DIR}/mps-init-probe.jsonl" >&2
    exit 1
  fi
fi

sudo_cmd systemctl set-environment MIG_DEVICE_UUID="${big_uuid}"
sudo_cmd systemctl set-default graphical.target
if [[ "${START_GDM:-1}" -eq 1 ]]; then
  sudo_cmd systemctl start gdm3
fi

cat >"${ENV_FILE}" <<EOF
JDG_MIG_BIG_UUID=${big_uuid}
JDG_MIG_SMALL_UUID=${small_uuid}
JDG_MPS_PIPE_DIRECTORY=${PIPE_DIR}
JDG_MPS_LOG_DIRECTORY=${LOG_DIR}
EOF

printf 'MIG configuration ready\n'
printf '  2g critical/graphics: %s\n' "${big_uuid}"
printf '  1g best-effort/MPS:  %s\n' "${small_uuid}"
printf '  environment:         %s\n' "${ENV_FILE}"
