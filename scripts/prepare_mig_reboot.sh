#!/usr/bin/env bash
set -euo pipefail

sudo_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

mig_mode="$(nvidia-smi --query-gpu=mig.mode.current \
  --format=csv,noheader | tr -d '[:space:]')"
instance_count="$(nvidia-smi -L | grep -c 'MIG ' || true)"
recovery_action="$(nvidia-smi -q | awk -F: \
  '/GPU Recovery Action/{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}')"
if [[ "${FORCE_REBOOT:-0}" -ne 1 && "${mig_mode}" == "Enabled" && \
      "${instance_count}" -eq 2 && "${recovery_action}" == "None" ]]; then
  echo "MIG mode and both instances are healthy; reboot is unnecessary"
  exit 0
fi

sudo_cmd systemctl set-default multi-user.target
printf 'Rebooting for MIG bootstrap (mode=%s, instances=%s, recovery=%s)\n' \
  "${mig_mode}" "${instance_count}" "${recovery_action:-unknown}"
sudo_cmd systemctl reboot
