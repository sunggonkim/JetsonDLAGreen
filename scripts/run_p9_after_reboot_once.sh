#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly STATUS_DIR="${ROOT_DIR}/results/onboot"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RUN_STAMP
readonly RESULT_DIR="${ROOT_DIR}/results/p9-mig-slack-onboot-${RUN_STAMP}"
readonly LOG_FILE="${STATUS_DIR}/p9-after-reboot-${RUN_STAMP}.log"
readonly STATUS_FILE="${STATUS_DIR}/p9-after-reboot-latest.json"
readonly UNIT_NAME="jdg-p9-after-reboot.service"
readonly OUTPUT_OWNER="${OUTPUT_OWNER:-$(stat -c '%U:%G' "${ROOT_DIR}")}"

mkdir -p "${STATUS_DIR}"

write_status() {
  local state="$1"
  local exit_code="$2"
  cat >"${STATUS_FILE}" <<EOF
{
  "schema_version": 1,
  "state": "${state}",
  "exit_code": ${exit_code},
  "result_dir": "${RESULT_DIR}",
  "log_file": "${LOG_FILE}",
  "timestamp_utc": "${RUN_STAMP}"
}
EOF
}

stop_root_mps() {
  local state_dir
  for state_dir in /tmp/jdg-mps-1g /tmp/jdg-mps-2g-p9; do
    if [[ -S "${state_dir}/pipe/control" ]]; then
      printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${state_dir}/pipe" \
        CUDA_MPS_LOG_DIRECTORY="${state_dir}/log" \
        nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
  done
}

write_status running 0

set +e
(
  cd "${ROOT_DIR}"
  env RESULT_DIR="${RESULT_DIR}" EPOCHS=6 SAMPLES=80 WARMUP=20 \
    CALIBRATION_REPEATS=1 POLICY_ORDER=fixed-borrow,mig-governor \
    RESTORE_GDM=1 \
    ./scripts/run_p9_mig_slack_governor.sh
) >"${LOG_FILE}" 2>&1
run_rc=$?
set -e

stop_root_mps
if [[ "${run_rc}" -eq 0 ]]; then
  write_status succeeded 0
else
  write_status failed "${run_rc}"
fi
systemctl disable "${UNIT_NAME}" >/dev/null 2>&1 || true

chown -R "${OUTPUT_OWNER}" "${STATUS_DIR}" "${RESULT_DIR}" >/dev/null 2>&1 || true
chown -R "${OUTPUT_OWNER}" "${ROOT_DIR}/models/engines/mig-1g-q25" \
  "${ROOT_DIR}/models/engines/mig-1g-q50" \
  "${ROOT_DIR}/models/engines/mig-1g-q100" \
  "${ROOT_DIR}/models/engines/mig-2g" \
  "${ROOT_DIR}/models/engines/mig-2g-q100" >/dev/null 2>&1 || true
exit "${run_rc}"
