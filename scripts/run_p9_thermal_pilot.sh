#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-thermal-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"

env RESULT_DIR="${RESULT_DIR}" \
  THERMAL_PILOT_SECONDS=600 \
  THERMAL_WINDOW_SECONDS=180 \
  THERMAL_STABILITY_SENSOR=soc012 \
  THERMAL_SAFETY_SENSOR=tj \
  THERMAL_HANDOFF_MAX_MS=500 \
  CALIBRATION_REPEATS=1 \
  SAMPLES=160 \
  WARMUP=100 \
  BURST_SIZE=8 \
  PERIOD_MS=20 \
  PRESSURE_CPUS=0-10 \
  MPS_CPU=11 \
  CRITICAL_CPU=12 \
  TELEMETRY_CPU=13 \
  EXPERIMENT_LABEL="thermal-pilot" \
  RESTORE_GDM="${RESTORE_GDM:-1}" \
  "${ROOT_DIR}/scripts/run_p9_mig_slack_governor.sh"

python3 "${ROOT_DIR}/analysis/freeze_p9_thermal.py" \
  "${RESULT_DIR}/summary.json" \
  --output "${RESULT_DIR}/thermal-lock.json"

printf 'P9 thermal pilot: %s\n' "${RESULT_DIR}/summary.json"
printf 'P9 frozen thermal target: %s\n' "${RESULT_DIR}/thermal-lock.json"
