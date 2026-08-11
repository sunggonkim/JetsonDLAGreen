#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/p9-performance-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly POLICY_ORDER="${POLICY_ORDER:-static-mig,resident-full-gate,fixed-full-gate,mig-governor}"
readonly EPOCHS="${EPOCHS:-6}"
readonly SAMPLES="${SAMPLES:-800}"

mkdir -p "${RESULT_ROOT}"

env RESULT_DIR="${RESULT_ROOT}/independent" \
SCENARIO=independent \
POLICY_ORDER="${POLICY_ORDER}" \
EPOCHS="${EPOCHS}" \
SAMPLES="${SAMPLES}" \
CALIBRATION_REPEATS="${CALIBRATION_REPEATS:-3}" \
EXPERIMENT_LABEL=performance-independent \
  "${ROOT_DIR}/scripts/run_p9_mig_slack_governor.sh"

deadline_ms="$(jq -er '.deadline_ms' \
  "${RESULT_ROOT}/independent/summary.json")"

env RESULT_DIR="${RESULT_ROOT}/dependent" \
SCENARIO=dependent \
POLICY_ORDER="${POLICY_ORDER}" \
EPOCHS="${EPOCHS}" \
SAMPLES="${SAMPLES}" \
CALIBRATION_REPEATS="${CALIBRATION_REPEATS:-3}" \
DEADLINE_MS="${deadline_ms}" \
DEADLINE_SOURCE=fixed-explicit \
EXPERIMENT_LABEL=performance-dependent \
  "${ROOT_DIR}/scripts/run_p9_mig_slack_governor.sh"

python3 "${ROOT_DIR}/analysis/summarize_p9_scenarios.py" \
  --independent "${RESULT_ROOT}/independent/summary.json" \
  --dependent "${RESULT_ROOT}/dependent/summary.json" \
  --output "${RESULT_ROOT}/comparison.json"

printf 'QUIET P9 performance scenarios: %s\n' "${RESULT_ROOT}"
