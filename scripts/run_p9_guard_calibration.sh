#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-guard-calibration-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly THERMAL_LOCK="${THERMAL_LOCK:-}"

if [[ -z "${THERMAL_LOCK}" || ! -f "${THERMAL_LOCK}" ]]; then
  printf 'THERMAL_LOCK must name a frozen thermal pilot lock\n' >&2
  exit 1
fi
mkdir -p "${RESULT_DIR}"
readonly RESULT_THERMAL_LOCK="${RESULT_DIR}/thermal-lock.json"
install -m 0444 "${THERMAL_LOCK}" "${RESULT_THERMAL_LOCK}"
python3 "${ROOT_DIR}/analysis/freeze_p9_thermal.py" \
  --verify "${RESULT_THERMAL_LOCK}"

env RESULT_DIR="${RESULT_DIR}" \
  GUARD_CALIBRATION=1 \
  THERMAL_LOCK="${RESULT_THERMAL_LOCK}" \
  BORROWER_QUOTA=100 \
  PRESSURE_CPUS=0-10 \
  MPS_CPU=11 \
  CRITICAL_CPU=12 \
  TELEMETRY_CPU=13 \
  RESTORE_GDM="${RESTORE_GDM:-1}" \
  "${ROOT_DIR}/scripts/run_p9_mig_slack_governor.sh"

python3 "${ROOT_DIR}/analysis/freeze_p9_guard.py" \
  "${RESULT_DIR}/guard-profile.json" \
  --thermal-lock "${RESULT_THERMAL_LOCK}" \
  --output "${RESULT_DIR}/guard-lock.json"
python3 "${ROOT_DIR}/analysis/freeze_p9_guard.py" \
  --verify "${RESULT_DIR}/guard-lock.json"

printf 'P9 guard profile: %s\n' "${RESULT_DIR}/guard-profile.json"
printf 'P9 frozen guard lock: %s\n' "${RESULT_DIR}/guard-lock.json"
