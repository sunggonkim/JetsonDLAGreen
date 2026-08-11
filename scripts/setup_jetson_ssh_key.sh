#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup_jetson_ssh_key.sh --host HOST [options]

Install a local SSH public key on a remote Jetson for passwordless login.
Run this on your local machine.

Required:
  --host HOST                 Remote Jetson hostname or IP.

Options:
  --user USER                 SSH user (default: thor)
  --port PORT                 SSH port (default: 22)
  --key-path PATH             Private key path (default: ~/.ssh/id_ed25519)
  --alias NAME                Host alias to write in ~/.ssh/config (optional)
  --overwrite-alias           Replace existing alias block in ~/.ssh/config
  -h, --help                  Show this help.

Auth:
  - Set JETSON_PASSWORD to enable non-interactive bootstrap via sshpass.
  - Without JETSON_PASSWORD, interactive ssh may prompt for password once.

Examples:
  JETSON_PASSWORD='<password>' ./scripts/setup_jetson_ssh_key.sh --host 192.168.0.42 --alias jetson-thor
  ./scripts/setup_jetson_ssh_key.sh --host 192.168.0.42 --user thor --key-path ~/.ssh/jetson_thor_ed25519
EOF
}

HOST=""
USER_NAME="thor"
PORT="22"
KEY_PATH="${HOME}/.ssh/id_ed25519"
ALIAS_NAME=""
OVERWRITE_ALIAS=0

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
    --key-path)
      KEY_PATH="${2:-}"
      shift 2
      ;;
    --alias)
      ALIAS_NAME="${2:-}"
      shift 2
      ;;
    --overwrite-alias)
      OVERWRITE_ALIAS=1
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

KEY_PATH="${KEY_PATH/#\~/${HOME}}"
PUB_PATH="${KEY_PATH}.pub"
SSH_DIR="${HOME}/.ssh"
CONFIG_PATH="${SSH_DIR}/config"

mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"

if [[ ! -f "${KEY_PATH}" ]]; then
  ssh-keygen -t ed25519 -N '' -f "${KEY_PATH}" -C "jetson-${USER_NAME}@${HOST}"
fi

if [[ ! -f "${PUB_PATH}" ]]; then
  echo "missing public key: ${PUB_PATH}" >&2
  exit 1
fi

SSH_OPTS=(
  -p "${PORT}"
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=5
)

ssh_exec() {
  local remote_cmd="$1"
  # shellcheck disable=SC2029
  if [[ -n "${JETSON_PASSWORD:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${JETSON_PASSWORD}" sshpass -e ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "${remote_cmd}"
  else
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "${remote_cmd}"
  fi
}

pub_key="$(cat "${PUB_PATH}")"

remote_install_cmd="umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; grep -Fq '${pub_key}' ~/.ssh/authorized_keys || echo '${pub_key}' >> ~/.ssh/authorized_keys"
ssh_exec "${remote_install_cmd}"

if [[ -n "${ALIAS_NAME}" ]]; then
  touch "${CONFIG_PATH}"
  chmod 600 "${CONFIG_PATH}"

  alias_block=$(cat <<EOF
# >>> jetsondlagreen ${ALIAS_NAME} >>>
Host ${ALIAS_NAME}
  HostName ${HOST}
  User ${USER_NAME}
  Port ${PORT}
  IdentityFile ${KEY_PATH}
  IdentitiesOnly yes
  ServerAliveInterval 5
  ServerAliveCountMax 2
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
# <<< jetsondlagreen ${ALIAS_NAME} <<<
EOF
)

  if grep -Fq "# >>> jetsondlagreen ${ALIAS_NAME} >>>" "${CONFIG_PATH}"; then
    if [[ "${OVERWRITE_ALIAS}" -eq 1 ]]; then
      tmp_cfg="$(mktemp)"
      awk -v begin="# >>> jetsondlagreen ${ALIAS_NAME} >>>" -v end="# <<< jetsondlagreen ${ALIAS_NAME} <<<" '
        $0 == begin {drop=1; next}
        $0 == end {drop=0; next}
        drop != 1 {print}
      ' "${CONFIG_PATH}" >"${tmp_cfg}"
      printf '%s\n' "${alias_block}" >>"${tmp_cfg}"
      mv "${tmp_cfg}" "${CONFIG_PATH}"
    fi
  else
    printf '\n%s\n' "${alias_block}" >>"${CONFIG_PATH}"
  fi
fi

echo "SSH key bootstrap complete"
echo "  host: ${HOST}"
echo "  user: ${USER_NAME}"
echo "  key : ${KEY_PATH}"
if [[ -n "${ALIAS_NAME}" ]]; then
  echo "  alias: ${ALIAS_NAME}"
fi
