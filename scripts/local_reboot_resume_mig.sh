#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: local_reboot_resume_mig.sh --host HOST [options]

Reboot a remote Jetson into MIG bootstrap mode, wait for SSH to return,
then continue with MIG configuration automatically.
Designed to be run on your local machine that opens VS Code Remote SSH.

Required:
  --host HOST                 Remote Jetson hostname or IP.

Options:
  --user USER                 SSH user (default: thor)
  --port PORT                 SSH port (default: 22)
  --repo-dir PATH             Remote repo path (default: ~/skim/JetsonDLAGreen)
  --wait-seconds N            Max wait for SSH recovery (default: 420)
  --resume-cmd CMD            Command run after reconnect
                              (default: ./scripts/configure_thor_mig.sh)
  --vscode-reopen             Reopen VS Code Remote SSH after reconnect.
  --vscode-remote-path PATH   Remote folder path for VS Code reopen
                              (default: /home/<user>/skim/JetsonDLAGreen)
  --skip-reboot               Skip reboot trigger and only wait/reconnect/resume.
  -h, --help                  Show this help.

Auth:
  - If JETSON_PASSWORD is set and sshpass exists, password login is used.
  - Otherwise, regular SSH auth is used (recommended: SSH key).

Sudo:
  - If SUDO_PASSWORD is set, it is forwarded to remote commands.
  - If SUDO_PASSWORD is unset and JETSON_PASSWORD is set, SUDO_PASSWORD uses JETSON_PASSWORD.

Examples:
  JETSON_PASSWORD='<password>' ./scripts/local_reboot_resume_mig.sh --host 192.168.0.42

  JETSON_PASSWORD='<password>' ./scripts/local_reboot_resume_mig.sh \
    --host 192.168.0.42 \
    --vscode-reopen \
    --resume-cmd './scripts/configure_thor_mig.sh && ./scripts/run_p1_mig.sh'
EOF
}

HOST=""
USER_NAME="thor"
PORT="22"
REPO_DIR=""
WAIT_SECONDS="420"
RESUME_CMD="./scripts/configure_thor_mig.sh"
VSCODE_REOPEN=0
VSCODE_REMOTE_PATH=""
SKIP_REBOOT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --user)
      USER_NAME="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="${2:-}"
      shift 2
      ;;
    --wait-seconds)
      WAIT_SECONDS="${2:-}"
      shift 2
      ;;
    --resume-cmd)
      RESUME_CMD="${2:-}"
      shift 2
      ;;
    --vscode-reopen)
      VSCODE_REOPEN=1
      shift
      ;;
    --vscode-remote-path)
      VSCODE_REMOTE_PATH="${2:-}"
      shift 2
      ;;
    --skip-reboot)
      SKIP_REBOOT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${HOST}" ]]; then
  echo "--host is required" >&2
  usage
  exit 2
fi

if ! [[ "${WAIT_SECONDS}" =~ ^[0-9]+$ ]] || [[ "${WAIT_SECONDS}" -le 0 ]]; then
  echo "--wait-seconds must be a positive integer" >&2
  exit 2
fi

JETSON_PASSWORD="${JETSON_PASSWORD:-}"
SUDO_PASSWORD="${SUDO_PASSWORD:-${JETSON_PASSWORD}}"
if [[ -z "${REPO_DIR}" ]]; then
  REPO_DIR="/home/${USER_NAME}/skim/JetsonDLAGreen"
fi
if [[ -z "${VSCODE_REMOTE_PATH}" ]]; then
  VSCODE_REMOTE_PATH="/home/${USER_NAME}/skim/JetsonDLAGreen"
fi

SSH_OPTS=(
  -p "${PORT}"
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=5
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=1
)

ssh_exec() {
  local remote_cmd="$1"
  # shellcheck disable=SC2029
  if [[ -n "${JETSON_PASSWORD}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${JETSON_PASSWORD}" sshpass -e ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "${remote_cmd}"
  else
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "${remote_cmd}"
  fi
}

ssh_probe() {
  if [[ -n "${JETSON_PASSWORD}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${JETSON_PASSWORD}" sshpass -e ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" true >/dev/null 2>&1
  else
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" true >/dev/null 2>&1
  fi
}

echo "[1/4] remote reboot phase"
if [[ "${SKIP_REBOOT}" -eq 0 ]]; then
  remote_reboot_cmd="cd ${REPO_DIR} && SUDO_PASSWORD='${SUDO_PASSWORD}' ./scripts/prepare_mig_reboot.sh"
  set +e
  ssh_exec "${remote_reboot_cmd}" >/tmp/jdg-reboot-trigger.log 2>&1
  reboot_rc=$?
  set -e

  if [[ "${reboot_rc}" -ne 0 && "${reboot_rc}" -ne 255 ]]; then
    echo "reboot trigger failed (exit=${reboot_rc})" >&2
    sed -n '1,120p' /tmp/jdg-reboot-trigger.log >&2 || true
    exit "${reboot_rc}"
  fi
else
  echo "skip-reboot enabled; reboot trigger is skipped"
fi

echo "[2/4] waiting for ssh disconnect"
disconnect_deadline=$((SECONDS + 60))
while ssh_probe; do
  if (( SECONDS >= disconnect_deadline )); then
    echo "ssh did not disconnect within 60s; continue to reconnect wait"
    break
  fi
  sleep 2
done

echo "[3/4] waiting for ssh reconnect"
reconnect_deadline=$((SECONDS + WAIT_SECONDS))
until ssh_probe; do
  if (( SECONDS >= reconnect_deadline )); then
    echo "ssh did not recover within ${WAIT_SECONDS}s" >&2
    exit 1
  fi
  sleep 3
done

echo "[4/4] resume command"
remote_resume_cmd="cd ${REPO_DIR} && SUDO_PASSWORD='${SUDO_PASSWORD}' ${RESUME_CMD}"
ssh_exec "${remote_resume_cmd}"

if [[ "${VSCODE_REOPEN}" -eq 1 ]]; then
  if command -v code >/dev/null 2>&1; then
    remote_uri="vscode-remote://ssh-remote+${HOST}${VSCODE_REMOTE_PATH}"
    echo "reopening VS Code: ${remote_uri}"
    code --folder-uri "${remote_uri}" >/dev/null 2>&1 || \
      echo "warning: VS Code reopen command failed; reconnect manually"
  else
    echo "warning: code CLI not found; skipping VS Code reopen"
  fi
fi

echo "done: reboot/reconnect/resume sequence finished"