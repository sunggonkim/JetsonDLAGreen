#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly CAMPAIGN_DIR="${CAMPAIGN_DIR:-${ROOT_DIR}/results/p9-campaign-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly DEADLINE_LOCK="${DEADLINE_LOCK:-}"
readonly THERMAL_LOCK="${THERMAL_LOCK:-}"
readonly GUARD_LOCK="${GUARD_LOCK:-}"
readonly REPEATS="${REPEATS:-14}"
readonly -a WILLIAMS_ORDERS=(
  "static-mig,resident-full-gate,mig-governor,same-mig,fixed-full-gate,uncoordinated-borrow,fixed-borrow"
  "resident-full-gate,same-mig,static-mig,uncoordinated-borrow,mig-governor,fixed-borrow,fixed-full-gate"
  "same-mig,uncoordinated-borrow,resident-full-gate,fixed-borrow,static-mig,fixed-full-gate,mig-governor"
  "uncoordinated-borrow,fixed-borrow,same-mig,fixed-full-gate,resident-full-gate,mig-governor,static-mig"
  "fixed-borrow,fixed-full-gate,uncoordinated-borrow,mig-governor,same-mig,static-mig,resident-full-gate"
  "fixed-full-gate,mig-governor,fixed-borrow,static-mig,uncoordinated-borrow,resident-full-gate,same-mig"
  "mig-governor,static-mig,fixed-full-gate,resident-full-gate,fixed-borrow,same-mig,uncoordinated-borrow"
  "fixed-borrow,uncoordinated-borrow,fixed-full-gate,same-mig,mig-governor,resident-full-gate,static-mig"
  "fixed-full-gate,fixed-borrow,mig-governor,uncoordinated-borrow,static-mig,same-mig,resident-full-gate"
  "mig-governor,fixed-full-gate,static-mig,fixed-borrow,resident-full-gate,uncoordinated-borrow,same-mig"
  "static-mig,mig-governor,resident-full-gate,fixed-full-gate,same-mig,fixed-borrow,uncoordinated-borrow"
  "resident-full-gate,static-mig,same-mig,mig-governor,uncoordinated-borrow,fixed-full-gate,fixed-borrow"
  "same-mig,resident-full-gate,uncoordinated-borrow,static-mig,fixed-borrow,mig-governor,fixed-full-gate"
  "uncoordinated-borrow,same-mig,fixed-borrow,resident-full-gate,fixed-full-gate,static-mig,mig-governor"
)
readonly -a SCHEDULE=(4 7 12 6 3 13 1 11 2 9 5 0 8 10)

if ((REPEATS != 14)); then
  printf 'REPEATS must be exactly 14 for the frozen Williams design\n' >&2
  exit 1
fi

if [[ "${PRINT_POLICY_ORDERS:-0}" -eq 1 ]]; then
  for order_index in "${SCHEDULE[@]}"; do
    printf '%s\n' "${WILLIAMS_ORDERS[${order_index}]}"
  done
  exit 0
fi

if [[ -z "${DEADLINE_LOCK}" || ! -f "${DEADLINE_LOCK}" ]]; then
  printf 'DEADLINE_LOCK must name a frozen calibration lock\n' >&2
  exit 1
fi
if [[ -z "${THERMAL_LOCK}" || ! -f "${THERMAL_LOCK}" ]]; then
  printf 'THERMAL_LOCK must name a frozen thermal pilot lock\n' >&2
  exit 1
fi
if [[ -z "${GUARD_LOCK}" || ! -f "${GUARD_LOCK}" ]]; then
  printf 'GUARD_LOCK must name a frozen guard calibration lock\n' >&2
  exit 1
fi
mkdir -p "${CAMPAIGN_DIR}"
readonly CAMPAIGN_DEADLINE_LOCK="${CAMPAIGN_DIR}/deadline-lock.json"
readonly CAMPAIGN_THERMAL_LOCK="${CAMPAIGN_DIR}/thermal-lock.json"
readonly CAMPAIGN_GUARD_LOCK="${CAMPAIGN_DIR}/guard-lock.json"
install -m 0444 "${DEADLINE_LOCK}" "${CAMPAIGN_DEADLINE_LOCK}"
install -m 0444 "${THERMAL_LOCK}" "${CAMPAIGN_THERMAL_LOCK}"
install -m 0444 "${GUARD_LOCK}" "${CAMPAIGN_GUARD_LOCK}"
python3 "${ROOT_DIR}/analysis/freeze_p9_deadline.py" \
  --verify "${CAMPAIGN_DEADLINE_LOCK}"
python3 "${ROOT_DIR}/analysis/freeze_p9_thermal.py" \
  --verify "${CAMPAIGN_THERMAL_LOCK}"
python3 "${ROOT_DIR}/analysis/freeze_p9_guard.py" \
  --verify "${CAMPAIGN_GUARD_LOCK}"
if ! jq -e '
  .calibration_blocks == 10 and
  .samples_per_block == 9600 and
  .isolated_samples == 96000 and
  (.slo_factor == 1.10)
' "${CAMPAIGN_DEADLINE_LOCK}" >/dev/null; then
  printf 'deadline lock does not match the frozen 10x9600 protocol\n' >&2
  exit 1
fi
deadline_ms="$(jq -er '.deadline_ms' "${CAMPAIGN_DEADLINE_LOCK}")"
slo_factor="$(jq -er '.slo_factor' "${CAMPAIGN_DEADLINE_LOCK}")"
expected_thermal_sha="$(jq -er '.thermal_lock_sha256' "${CAMPAIGN_DEADLINE_LOCK}")"
actual_thermal_sha="$(sha256sum "${CAMPAIGN_THERMAL_LOCK}" | awk '{print $1}')"
actual_guard_sha="$(sha256sum "${CAMPAIGN_GUARD_LOCK}" | awk '{print $1}')"
if [[ "${expected_thermal_sha}" != "${actual_thermal_sha}" ]]; then
  printf 'deadline and thermal locks were not calibrated together\n' >&2
  exit 1
fi
expected_deadline_guard_sha="$(jq -er '.guard_lock_sha256' \
  "${CAMPAIGN_DEADLINE_LOCK}")"
if [[ "${expected_deadline_guard_sha}" != "${actual_guard_sha}" ]]; then
  printf 'deadline and guard locks were not calibrated together\n' >&2
  exit 1
fi
expected_guard_thermal_sha="$(jq -er '.thermal_lock.sha256' \
  "${CAMPAIGN_GUARD_LOCK}")"
if [[ "${expected_guard_thermal_sha}" != "${actual_thermal_sha}" ]]; then
  printf 'guard and thermal locks were not calibrated together\n' >&2
  exit 1
fi
readonly PROVENANCE_DIR="${CAMPAIGN_DIR}/provenance"
deadline_source_summary="$(jq -er '.source_summary' "${CAMPAIGN_DEADLINE_LOCK}")"
thermal_source_summary="$(jq -er '.source_summary' "${CAMPAIGN_THERMAL_LOCK}")"
guard_source_summary="$(jq -er '.source.profile_summary' "${CAMPAIGN_GUARD_LOCK}")"
mkdir -p "${PROVENANCE_DIR}/implementation"
cp -a "$(dirname "${deadline_source_summary}")" \
  "${PROVENANCE_DIR}/deadline-calibration"
cp -a "$(dirname "${thermal_source_summary}")" \
  "${PROVENANCE_DIR}/thermal-pilot"
cp -a "$(dirname "${guard_source_summary}")" \
  "${PROVENANCE_DIR}/guard-calibration"
while IFS= read -r source_file; do
  install -D -m 0444 "${ROOT_DIR}/${source_file}" \
    "${PROVENANCE_DIR}/implementation/${source_file}"
done < <(
  jq -rs 'map(.code_sha256 | keys) | add | unique[]' \
    "${CAMPAIGN_DEADLINE_LOCK}" "${CAMPAIGN_THERMAL_LOCK}"
)
while IFS=$'\t' read -r artifact_name source_file; do
  artifact_name="${artifact_name//:/_}"
  install -D -m 0444 "${source_file}" \
    "${PROVENANCE_DIR}/guard-artifacts/${artifact_name}"
done < <(jq -r '.artifacts | to_entries[] | [.key, .value.path] | @tsv' \
  "${CAMPAIGN_GUARD_LOCK}")
(cd "${CAMPAIGN_DIR}" && \
  find provenance -type f ! -name MANIFEST.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >provenance/MANIFEST.sha256)
thermal_target_c="$(jq -er '.target_c' "${CAMPAIGN_THERMAL_LOCK}")"
thermal_tolerance_c="$(jq -er '.tolerance_c' "${CAMPAIGN_THERMAL_LOCK}")"
thermal_window_seconds="$(jq -er '.stability_window_seconds' "${CAMPAIGN_THERMAL_LOCK}")"
thermal_max_slope="$(jq -er '.maximum_slope_c_per_minute' "${CAMPAIGN_THERMAL_LOCK}")"
thermal_hard_limit_c="$(jq -er '.hard_limit_c' "${CAMPAIGN_THERMAL_LOCK}")"
thermal_stability_sensor="$(jq -er '.stability_sensor' "${CAMPAIGN_THERMAL_LOCK}")"
thermal_safety_sensor="$(jq -er '.safety_sensor' "${CAMPAIGN_THERMAL_LOCK}")"
thermal_handoff_max_ms="$(jq -er '.thermal_handoff_max_ms' "${CAMPAIGN_THERMAL_LOCK}")"
deadline_lock_sha="$(sha256sum "${CAMPAIGN_DEADLINE_LOCK}" | awk '{print $1}')"
thermal_lock_sha="${actual_thermal_sha}"
guard_lock_sha="${actual_guard_sha}"
inputs=()
for repetition in $(seq 1 "${REPEATS}"); do
  python3 "${ROOT_DIR}/analysis/freeze_p9_deadline.py" \
    --verify "${CAMPAIGN_DEADLINE_LOCK}"
  python3 "${ROOT_DIR}/analysis/freeze_p9_thermal.py" \
    --verify "${CAMPAIGN_THERMAL_LOCK}"
  python3 "${ROOT_DIR}/analysis/freeze_p9_guard.py" \
    --verify "${CAMPAIGN_GUARD_LOCK}"
  if [[ "$(sha256sum "${CAMPAIGN_DEADLINE_LOCK}" | awk '{print $1}')" \
        != "${deadline_lock_sha}" || \
        "$(sha256sum "${CAMPAIGN_THERMAL_LOCK}" | awk '{print $1}')" \
        != "${thermal_lock_sha}" || \
        "$(sha256sum "${CAMPAIGN_GUARD_LOCK}" | awk '{print $1}')" \
        != "${guard_lock_sha}" ]]; then
    printf 'campaign lock changed before run %s\n' "${repetition}" >&2
    exit 1
  fi
  schedule_index=$(((repetition - 1) % ${#SCHEDULE[@]}))
  order_index="${SCHEDULE[${schedule_index}]}"
  result_dir="${CAMPAIGN_DIR}/run-${repetition}"
  env RESULT_DIR="${result_dir}" \
    POLICY_ORDER="${WILLIAMS_ORDERS[${order_index}]}" \
    EPOCHS=36 \
    SAMPLES=800 \
    WARMUP=100 \
    BURST_SIZE=8 \
    PERIOD_MS=20 \
    DMR_TARGET=0.0005 \
    CALIBRATION_REPEATS=3 \
    BORROWER_QUOTA=100 \
    LANGUAGE_GUARD_MS=1.5 \
    AUDIO_GUARD_MS=2 \
    MAX_ISOLATED_DRIFT_FRACTION=0.05 \
    PRESSURE_CPUS=0-10 \
    MPS_CPU=11 \
    CRITICAL_CPU=12 \
    TELEMETRY_CPU=13 \
    DEADLINE_MS="${deadline_ms}" \
    DEADLINE_SOURCE="frozen-isolated-p99-factor" \
    DEADLINE_LOCK_SHA256="${deadline_lock_sha}" \
    THERMAL_LOCK_SHA256="${thermal_lock_sha}" \
    GUARD_LOCK="${CAMPAIGN_GUARD_LOCK}" \
    GUARD_LOCK_SHA256="${guard_lock_sha}" \
    SLO_FACTOR="${slo_factor}" \
    THERMAL_TARGET_C="${thermal_target_c}" \
    THERMAL_TOLERANCE_C="${thermal_tolerance_c}" \
    THERMAL_WINDOW_SECONDS="${thermal_window_seconds}" \
    THERMAL_MAX_SLOPE_C_PER_MINUTE="${thermal_max_slope}" \
    THERMAL_HARD_LIMIT_C="${thermal_hard_limit_c}" \
    THERMAL_STABILITY_SENSOR="${thermal_stability_sensor}" \
    THERMAL_SAFETY_SENSOR="${thermal_safety_sensor}" \
    THERMAL_HANDOFF_MAX_MS="${thermal_handoff_max_ms}" \
    EXPERIMENT_LABEL="campaign-r${repetition}" \
    RESTORE_GDM="${RESTORE_GDM:-1}" \
    "${ROOT_DIR}/scripts/run_p9_mig_slack_governor.sh"
  if ! jq -e '.isolated_drift_valid == true' \
    "${result_dir}/summary.json" >/dev/null; then
    printf 'run %s failed isolated drift validation\n' "${repetition}" >&2
    exit 1
  fi
  if ! jq -e --slurpfile lock "${CAMPAIGN_DEADLINE_LOCK}" \
    '.artifacts == $lock[0].calibration_artifacts and
     .hardware == $lock[0].calibration_hardware and
     .mig == $lock[0].calibration_mig and
     .config.cpu_affinity == $lock[0].calibration_cpu_affinity' \
    "${result_dir}/summary.json" >/dev/null; then
    printf 'run %s artifact hashes differ from calibration\n' "${repetition}" >&2
    exit 1
  fi
  inputs+=("${result_dir}/summary.json")
done

python3 "${ROOT_DIR}/analysis/summarize_mig_slack_governor.py" \
  "${inputs[@]}" \
  --deadline-lock "${CAMPAIGN_DEADLINE_LOCK}" \
  --thermal-lock "${CAMPAIGN_THERMAL_LOCK}" \
  --guard-lock "${CAMPAIGN_GUARD_LOCK}" \
  --output "${CAMPAIGN_DIR}/summary.json"
manifest_tmp="$(mktemp)"
(cd "${CAMPAIGN_DIR}" && \
  find . -type f ! -name CAMPAIGN.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >"${manifest_tmp}")
install -m 0444 "${manifest_tmp}" "${CAMPAIGN_DIR}/CAMPAIGN.sha256"
rm -f "${manifest_tmp}"
printf 'P9 repeated campaign: %s\n' "${CAMPAIGN_DIR}"
