#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly STATUS_DIR="${ROOT_DIR}/results/onboot"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RUN_STAMP
readonly RESULT_DIR="${ROOT_DIR}/results/p6-full-multimodal-formal-td-${RUN_STAMP}"
readonly LOG_FILE="${STATUS_DIR}/p6-after-reboot-${RUN_STAMP}.log"
readonly STATUS_FILE="${STATUS_DIR}/p6-after-reboot-latest.json"
readonly UNIT_NAME="jdg-p6-after-reboot.service"
readonly OUTPUT_OWNER="${OUTPUT_OWNER:-$(stat -c '%U:%G' "${ROOT_DIR}")}"

mkdir -p "${STATUS_DIR}"

cleanup_unit() {
  systemctl disable "${UNIT_NAME}" >/dev/null 2>&1 || true
}

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

chown_outputs() {
  chown -R "${OUTPUT_OWNER}" "${STATUS_DIR}" >/dev/null 2>&1 || true
  chown -R "${OUTPUT_OWNER}" "${RESULT_DIR}" >/dev/null 2>&1 || true
}

cleanup_unit
write_status running 0

set +e
(
  cd "${ROOT_DIR}"
  RESULT_DIR="${RESULT_DIR}" REPEATS="3" SAMPLES="10000" WARMUP="500" \
    ./scripts/run_p6_full_multimodal.sh
) >"${LOG_FILE}" 2>&1
run_rc=$?
set -e

if [[ "${run_rc}" -eq 0 ]]; then
  write_status succeeded 0
else
  write_status failed "${run_rc}"
fi

chown_outputs
exit "${run_rc}"
