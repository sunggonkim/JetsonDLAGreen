#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-deadline-calibration-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly THERMAL_LOCK="${THERMAL_LOCK:-}"
readonly GUARD_LOCK="${GUARD_LOCK:-}"

if [[ -z "${THERMAL_LOCK}" || ! -f "${THERMAL_LOCK}" ]]; then
  printf 'THERMAL_LOCK must name a frozen thermal pilot lock\n' >&2
  exit 1
fi
if [[ -z "${GUARD_LOCK}" || ! -f "${GUARD_LOCK}" ]]; then
  printf 'GUARD_LOCK must name a frozen guard calibration lock\n' >&2
  exit 1
fi
mkdir -p "${RESULT_DIR}"
readonly RESULT_THERMAL_LOCK="${RESULT_DIR}/thermal-lock.json"
readonly RESULT_GUARD_LOCK="${RESULT_DIR}/guard-lock.json"
install -m 0444 "${THERMAL_LOCK}" "${RESULT_THERMAL_LOCK}"
install -m 0444 "${GUARD_LOCK}" "${RESULT_GUARD_LOCK}"
python3 "${ROOT_DIR}/analysis/freeze_p9_thermal.py" \
  --verify "${RESULT_THERMAL_LOCK}"
python3 "${ROOT_DIR}/analysis/freeze_p9_guard.py" \
  --verify "${RESULT_GUARD_LOCK}"
thermal_lock_sha="$(sha256sum "${RESULT_THERMAL_LOCK}" | awk '{print $1}')"
guard_lock_sha="$(sha256sum "${RESULT_GUARD_LOCK}" | awk '{print $1}')"
if [[ "$(jq -er '.thermal_lock.sha256' "${RESULT_GUARD_LOCK}")" \
      != "${thermal_lock_sha}" ]]; then
  printf 'guard and thermal locks were not calibrated together\n' >&2
  exit 1
fi
thermal_target_c="$(jq -er '.target_c' "${RESULT_THERMAL_LOCK}")"
thermal_tolerance_c="$(jq -er '.tolerance_c' "${RESULT_THERMAL_LOCK}")"
thermal_window_seconds="$(jq -er '.stability_window_seconds' "${RESULT_THERMAL_LOCK}")"
thermal_max_slope="$(jq -er '.maximum_slope_c_per_minute' "${RESULT_THERMAL_LOCK}")"
thermal_hard_limit_c="$(jq -er '.hard_limit_c' "${RESULT_THERMAL_LOCK}")"
thermal_stability_sensor="$(jq -er '.stability_sensor' "${RESULT_THERMAL_LOCK}")"
thermal_safety_sensor="$(jq -er '.safety_sensor' "${RESULT_THERMAL_LOCK}")"
thermal_handoff_max_ms="$(jq -er '.thermal_handoff_max_ms' "${RESULT_THERMAL_LOCK}")"

env RESULT_DIR="${RESULT_DIR}" \
  CALIBRATION_ONLY=1 \
  CALIBRATION_REPEATS=10 \
  SAMPLES=9600 \
  WARMUP=100 \
  BURST_SIZE=8 \
  PERIOD_MS=20 \
  SLO_FACTOR=1.10 \
  PRESSURE_CPUS=0-10 \
  MPS_CPU=11 \
  CRITICAL_CPU=12 \
  TELEMETRY_CPU=13 \
  THERMAL_TARGET_C="${thermal_target_c}" \
  THERMAL_TOLERANCE_C="${thermal_tolerance_c}" \
  THERMAL_WINDOW_SECONDS="${thermal_window_seconds}" \
  THERMAL_MAX_SLOPE_C_PER_MINUTE="${thermal_max_slope}" \
  THERMAL_HARD_LIMIT_C="${thermal_hard_limit_c}" \
  THERMAL_STABILITY_SENSOR="${thermal_stability_sensor}" \
  THERMAL_SAFETY_SENSOR="${thermal_safety_sensor}" \
  THERMAL_HANDOFF_MAX_MS="${thermal_handoff_max_ms}" \
  THERMAL_LOCK_SHA256="${thermal_lock_sha}" \
  GUARD_LOCK="${RESULT_GUARD_LOCK}" \
  GUARD_LOCK_SHA256="${guard_lock_sha}" \
  EXPERIMENT_LABEL="deadline-calibration" \
  RESTORE_GDM="${RESTORE_GDM:-1}" \
  "${ROOT_DIR}/scripts/run_p9_mig_slack_governor.sh"

python3 "${ROOT_DIR}/analysis/freeze_p9_deadline.py" \
  "${RESULT_DIR}/summary.json" \
  --output "${RESULT_DIR}/deadline-lock.json" \
  --guard-lock "${RESULT_GUARD_LOCK}" \
  --expected-blocks 10 \
  --expected-samples-per-block 9600
python3 "${ROOT_DIR}/analysis/freeze_p9_deadline.py" \
  --verify "${RESULT_DIR}/deadline-lock.json"

printf 'P9 frozen deadline: %s\n' "${RESULT_DIR}/deadline-lock.json"
