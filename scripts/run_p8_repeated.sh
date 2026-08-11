#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly REPEATS="${REPEATS:-6}"
readonly CAMPAIGN_DIR="${CAMPAIGN_DIR:-${ROOT_DIR}/results/p8-campaign-$(date -u +%Y%m%dT%H%M%SZ)}"

orders=(
  'static-q5,static-q25,priority-q25,conservative-guard,profiled-guard,joint-governor'
  'static-q25,priority-q25,conservative-guard,profiled-guard,joint-governor,static-q5'
  'priority-q25,conservative-guard,profiled-guard,joint-governor,static-q5,static-q25'
  'conservative-guard,profiled-guard,joint-governor,static-q5,static-q25,priority-q25'
  'profiled-guard,joint-governor,static-q5,static-q25,priority-q25,conservative-guard'
  'joint-governor,static-q5,static-q25,priority-q25,conservative-guard,profiled-guard'
)

if [[ "${REPEATS}" -lt 1 || "${REPEATS}" -gt "${#orders[@]}" ]]; then
  printf 'REPEATS must be in [1, %s]\n' "${#orders[@]}" >&2
  exit 1
fi

mkdir -p "${CAMPAIGN_DIR}"
summaries=()
for ((repetition = 1; repetition <= REPEATS; ++repetition)); do
  result_dir="${CAMPAIGN_DIR}/r${repetition}"
  POLICY_ORDER="${orders[repetition - 1]}" RESULT_DIR="${result_dir}" \
    SUDO_PASSWORD="${SUDO_PASSWORD:-}" EPOCHS="${EPOCHS:-12}" \
    SAMPLES="${SAMPLES:-800}" WARMUP="${WARMUP:-100}" \
    CALIBRATION_REPEATS="${CALIBRATION_REPEATS:-3}" \
    "${ROOT_DIR}/scripts/run_p8_full_gpu_governor.sh"
  summaries+=("${result_dir}/summary.json")
done

python3 "${ROOT_DIR}/analysis/summarize_full_gpu_governor.py" \
  "${summaries[@]}" --output "${CAMPAIGN_DIR}/summary.json"
printf 'P8 repeated campaign: %s\n' "${CAMPAIGN_DIR}"
