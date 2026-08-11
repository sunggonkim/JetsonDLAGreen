#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly REPEATS="${REPEATS:-3}"
readonly CAMPAIGN_DIR="${CAMPAIGN_DIR:-${ROOT_DIR}/results/p8-sensitivity-$(date -u +%Y%m%dT%H%M%SZ)}"
summaries=()

sudo_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

restore_gui() {
  sudo_cmd systemctl start gdm3 >/dev/null 2>&1 || true
}
trap restore_gui EXIT INT TERM

run_point() {
  local label=$1
  local value=$2
  local repetition=$3
  local burst_size=$4
  local period_ms=$5
  local policy=$6
  local guard_ms=${7:-}
  local result_dir="${CAMPAIGN_DIR}/${label}-${value}/r${repetition}"
  local -a guard_environment=()
  if [[ -n "${guard_ms}" ]]; then
    guard_environment+=(GUARD_OVERRIDE_MS="${guard_ms}")
  fi
  env "${guard_environment[@]}" \
    SUDO_PASSWORD="${SUDO_PASSWORD:-}" RESTORE_GDM=0 \
    RESULT_DIR="${result_dir}" EXPERIMENT_LABEL="${label}" \
    POLICY_ORDER="${policy}" EPOCHS="${EPOCHS:-12}" \
    SAMPLES="${SAMPLES:-800}" WARMUP="${WARMUP:-100}" \
    CALIBRATION_REPEATS="${CALIBRATION_REPEATS:-3}" \
    BURST_SIZE="${burst_size}" PERIOD_MS="${period_ms}" \
    "${ROOT_DIR}/scripts/run_p8_full_gpu_governor.sh"
  summaries+=("${result_dir}/summary.json")
}

mkdir -p "${CAMPAIGN_DIR}"
guard_orders=(
  '0 0.5 1 1.5 2 3 4 5 6'
  '1.5 2 3 4 5 6 0 0.5 1'
  '6 5 4 3 2 1.5 1 0.5 0'
)
burst_orders=('4 8 16' '8 16 4' '16 4 8')
period_orders=('8 12 16' '12 16 8' '16 8 12')
for ((repetition = 1; repetition <= REPEATS; ++repetition)); do
  order_index=$(((repetition - 1) % 3))
  read -r -a guards <<<"${guard_orders[order_index]}"
  read -r -a bursts <<<"${burst_orders[order_index]}"
  read -r -a periods <<<"${period_orders[order_index]}"
  for guard_ms in "${guards[@]}"; do
    run_point guard "${guard_ms}ms" "${repetition}" 8 12 \
      profiled-guard "${guard_ms}"
  done
  for burst_size in "${bursts[@]}"; do
    period_ms=$((burst_size * 3 / 2))
    run_point burst "${burst_size}" "${repetition}" "${burst_size}" \
      "${period_ms}" joint-governor
  done
  for period_ms in "${periods[@]}"; do
    run_point period "${period_ms}ms" "${repetition}" 8 "${period_ms}" \
      joint-governor
  done
done

python3 "${ROOT_DIR}/analysis/summarize_p8_sensitivity.py" \
  "${summaries[@]}" --output "${CAMPAIGN_DIR}/summary.json"
printf 'P8 sensitivity campaign: %s\n' "${CAMPAIGN_DIR}"
