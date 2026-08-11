#!/usr/bin/env bash
set -euo pipefail

readonly STATE_DIR="${STATE_DIR:-/tmp/jdg-mps-1g}"

sudo_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

if ! nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | \
    grep -qx Enabled; then
  printf 'MIG is already disabled; reboot is unnecessary\n'
  exit 0
fi

sudo_cmd systemctl stop gdm3
if [[ -d "${STATE_DIR}/pipe" ]]; then
  printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${STATE_DIR}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="${STATE_DIR}/log" \
    nvidia-cuda-mps-control >/dev/null 2>&1 || true
fi
sudo_cmd nvidia-smi -pm 0
sudo_cmd nvidia-smi mig -dci
sudo_cmd nvidia-smi mig -dgi
sudo_cmd nvidia-smi -mig 0
sudo_cmd systemctl unset-environment MIG_DEVICE_UUID
sudo_cmd systemctl set-default graphical.target
sync
printf 'MIG disable requested; rebooting into full-GPU mode\n'
sudo_cmd systemctl reboot
