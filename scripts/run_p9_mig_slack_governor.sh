#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly ENGINE_ROOT="${ENGINE_ROOT:-${ROOT_DIR}/models/engines}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-mig-slack-$(date -u +%Y%m%dT%H%M%SZ)}"
# tegrastats daemonizes and resolves --logfile after the parent shell exits;
# relative paths therefore do not reliably point back to the result directory.
RESULT_DIR="$(realpath -m -- "${RESULT_DIR}")"
readonly RESULT_DIR
readonly SMALL_STATE_DIR="${SMALL_STATE_DIR:-/tmp/jdg-mps-1g}"
readonly BIG_STATE_DIR="${BIG_STATE_DIR:-/tmp/jdg-mps-2g-p9}"
readonly BORROWER_QUOTA="${BORROWER_QUOTA:-100}"
readonly CRITICAL_CPU="${CRITICAL_CPU:-12}"
readonly PRESSURE_CPUS="${PRESSURE_CPUS:-0-10}"
readonly MPS_CPU="${MPS_CPU:-11}"
readonly TELEMETRY_CPU="${TELEMETRY_CPU:-13}"
readonly TELEMETRY_LOG="${RESULT_DIR}/tegrastats.txt"
readonly SCENARIO="${SCENARIO:-independent}"
telemetry_started=0
big_mps_started=0

sudo_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

stop_big_mps() {
  if [[ "${big_mps_started}" -eq 1 ]]; then
    printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
      CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
    big_mps_started=0
  fi
}

pin_mps_processes() {
  local state_dir="$1"
  local label="$2"
  local control_pid
  local pid
  local role
  local allowed
  local status_file
  local -a pids=()
  if [[ ! -f "${state_dir}/pipe/nvidia-cuda-mps-control.pid" ]]; then
    printf 'missing %s MPS control PID\n' "${label}" >&2
    return 1
  fi
  control_pid="$(tr -d '[:space:]' \
    <"${state_dir}/pipe/nvidia-cuda-mps-control.pid")"
  if [[ ! "${control_pid}" =~ ^[0-9]+$ ]] || \
     [[ ! -r "/proc/${control_pid}/status" ]]; then
    printf 'invalid %s MPS control PID\n' "${label}" >&2
    return 1
  fi
  pids+=("${control_pid}")
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && pids+=("${pid}")
  done < <(pgrep -P "${control_pid}" -f '^nvidia-cuda-mps-server( |$)' || true)
  if ((${#pids[@]} < 2)); then
    printf '%s MPS server is not resident\n' "${label}" >&2
    return 1
  fi
  for pid in "${pids[@]}"; do
    role="server"
    [[ "${pid}" == "${control_pid}" ]] && role="control"
    taskset -apc "${MPS_CPU}" "${pid}" >/dev/null
    for status_file in /proc/"${pid}"/task/*/status; do
      allowed="$(awk '/^Cpus_allowed_list:/{print $2}' "${status_file}")"
      if [[ "${allowed}" != "${MPS_CPU}" ]]; then
        printf 'failed to pin %s MPS task %s to CPU %s\n' \
          "${label}" "${status_file}" "${MPS_CPU}" >&2
        return 1
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${role}" "${pid}" \
        "$(basename "$(dirname "${status_file}")")" "${allowed}" \
        >>"${RESULT_DIR}/mps-thread-affinity.tsv"
    done
    allowed="${MPS_CPU}"
    printf '%s\t%s\t%s\t%s\n' \
      "${label}" "${role}" "${pid}" "${allowed}" \
      >>"${RESULT_DIR}/mps-affinity.tsv"
  done
}

verify_mps_processes() {
  local state_dir="$1"
  local label="$2"
  local output="$3"
  local control_pid
  local pid
  local role
  local allowed
  local status_file
  local command_line
  local -a pids=()
  control_pid="$(tr -d '[:space:]' \
    <"${state_dir}/pipe/nvidia-cuda-mps-control.pid")"
  if [[ ! "${control_pid}" =~ ^[0-9]+$ ]] || \
     [[ ! -r "/proc/${control_pid}/status" ]]; then
    printf '%s MPS control process disappeared\n' "${label}" >&2
    return 1
  fi
  pids+=("${control_pid}")
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && pids+=("${pid}")
  done < <(pgrep -P "${control_pid}" -f '^nvidia-cuda-mps-server( |$)' || true)
  if ((${#pids[@]} < 2)); then
    printf '%s MPS server disappeared\n' "${label}" >&2
    return 1
  fi
  for pid in "${pids[@]}"; do
    role="server"
    [[ "${pid}" == "${control_pid}" ]] && role="control"
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    if [[ "${role}" == "control" && \
          "${command_line}" != nvidia-cuda-mps-control* ]] || \
       [[ "${role}" == "server" && \
          "${command_line}" != nvidia-cuda-mps-server* ]]; then
      printf '%s MPS PID %s changed identity\n' "${label}" "${pid}" >&2
      return 1
    fi
    if ! awk -F '\t' -v label="${label}" -v role="${role}" -v pid="${pid}" \
      '$1 == label && $2 == role && $3 == pid { found = 1 } END { exit !found }' \
      "${RESULT_DIR}/mps-affinity.tsv"; then
      printf '%s MPS PID set changed during the experiment\n' "${label}" >&2
      return 1
    fi
    for status_file in /proc/"${pid}"/task/*/status; do
      allowed="$(awk '/^Cpus_allowed_list:/{print $2}' "${status_file}")"
      if [[ "${allowed}" != "${MPS_CPU}" ]]; then
        printf '%s MPS task %s left CPU %s\n' \
          "${label}" "${status_file}" "${MPS_CPU}" >&2
        return 1
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${role}" "${pid}" \
        "$(basename "$(dirname "${status_file}")")" "${allowed}" >>"${output}"
    done
  done
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_big_mps
  if [[ "${telemetry_started}" -eq 1 ]]; then
    sudo_cmd tegrastats --stop >/dev/null 2>&1 || true
    telemetry_started=0
  fi
  if [[ "${RESTORE_GDM:-1}" -eq 1 ]]; then
    sudo_cmd systemctl start gdm3 >/dev/null 2>&1 || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
SUDO_PASSWORD="${SUDO_PASSWORD:-}" START_GDM=0 \
  "${ROOT_DIR}/scripts/configure_thor_mig.sh"

if ! nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | \
    tr -d '[:space:]' | grep -qx Enabled; then
  printf 'MIG mode is required after configure_thor_mig.sh\n' >&2
  exit 1
fi
if [[ ! -f "${SMALL_STATE_DIR}/mig.env" ]]; then
  printf 'missing MIG environment: %s/mig.env\n' "${SMALL_STATE_DIR}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${SMALL_STATE_DIR}/mig.env"

mkdir -p "${RESULT_DIR}"
install -m 0444 "${SMALL_STATE_DIR}/mig.env" "${RESULT_DIR}/mig.env"
sudo_cmd install -d -m 0755 -o "$(id -u)" -g "$(id -g)" \
  "${BIG_STATE_DIR}" "${BIG_STATE_DIR}/pipe" "${BIG_STATE_DIR}/log"
sudo_cmd chown -R "$(id -u):$(id -g)" "${BIG_STATE_DIR}"
sudo_cmd systemctl stop gdm3
sudo_cmd jetson_clocks --fan
cat /etc/nv_tegra_release >"${RESULT_DIR}/nv_tegra_release.txt"
nvidia-smi -q >"${RESULT_DIR}/nvidia-smi.txt"
sudo_cmd nvidia-smi mig -lgip >"${RESULT_DIR}/mig-profiles.txt"
sudo_cmd nvidia-smi mig -lgi >"${RESULT_DIR}/active-mig-instances.txt"
nvidia-smi -L >"${RESULT_DIR}/gpu-inventory.txt"
sudo_cmd jetson_clocks --show >"${RESULT_DIR}/jetson-clocks.txt" 2>&1 || true
nvpmodel -q >"${RESULT_DIR}/nvpmodel.txt" 2>&1
if ! grep -q 'FAN Dynamic Speed Control=disabled .*pwm1=255' \
  "${RESULT_DIR}/jetson-clocks.txt"; then
  printf 'maximum fixed fan speed was not established\n' >&2
  exit 1
fi
if ! grep -q 'gpu-gpc-0 MinFreq=1575000000 MaxFreq=1575000000' \
  "${RESULT_DIR}/jetson-clocks.txt" || \
   ! grep -q 'EMC MinFreq=4266000000 MaxFreq=4266000000' \
  "${RESULT_DIR}/jetson-clocks.txt"; then
  printf 'GPU or EMC clocks were not locked\n' >&2
  exit 1
fi
if ! grep -q 'NV Power Mode: MAXN' "${RESULT_DIR}/nvpmodel.txt"; then
  printf 'MAXN power mode is required\n' >&2
  exit 1
fi
sudo_cmd tegrastats --stop >/dev/null 2>&1 || true
sudo_cmd taskset --cpu-list "${TELEMETRY_CPU}" \
  tegrastats --interval 75 --readall --logfile "${TELEMETRY_LOG}" --start
telemetry_started=1

if [[ -S "${BIG_STATE_DIR}/pipe/control" ]]; then
  printf 'quit\n' | env CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" \
    nvidia-cuda-mps-control >/dev/null 2>&1 || true
  sleep 1
fi
find "${BIG_STATE_DIR}/pipe" -mindepth 1 -maxdepth 1 -delete

# R39.2 needs a direct context before a second MIG-local MPS server starts.
env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
  LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  timeout 30s "${BUILD_DIR}/jdg-cuda-init-probe" \
  >"${BIG_STATE_DIR}/log/direct-init.jsonl" 2>&1

CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" nvidia-cuda-mps-control -d
big_mps_started=1
env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
  CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
  CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" \
  LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  timeout 30s "${BUILD_DIR}/jdg-cuda-init-probe" \
  >"${BIG_STATE_DIR}/log/mps-init.jsonl" 2>&1

: >"${RESULT_DIR}/mps-affinity.tsv"
: >"${RESULT_DIR}/mps-thread-affinity.tsv"
pin_mps_processes "${SMALL_STATE_DIR}" "resident-1g"
pin_mps_processes "${BIG_STATE_DIR}" "critical-2g"

env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
  CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
  CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
  ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG=mig-2g \
  BUILD_MODELS=resnet50-v2 "${ROOT_DIR}/scripts/prepare_models.sh"

env CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
  CUDA_MPS_PIPE_DIRECTORY="${BIG_STATE_DIR}/pipe" \
  CUDA_MPS_LOG_DIRECTORY="${BIG_STATE_DIR}/log" \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${BORROWER_QUOTA}" \
  ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG="mig-2g-q${BORROWER_QUOTA}" \
  BUILD_MODELS=distilbert-sst2,whisper-tiny-encoder \
  "${ROOT_DIR}/scripts/prepare_models.sh"

for quota in 25 50 100; do
  env CUDA_VISIBLE_DEVICES="${JDG_MIG_SMALL_UUID}" \
    CUDA_MPS_PIPE_DIRECTORY="${JDG_MPS_PIPE_DIRECTORY}" \
    CUDA_MPS_LOG_DIRECTORY="${JDG_MPS_LOG_DIRECTORY}" \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${quota}" \
    ENGINE_ROOT="${ENGINE_ROOT}" ENGINE_TAG="mig-1g-q${quota}" \
    BUILD_MODELS=distilbert-sst2,whisper-tiny-encoder \
    "${ROOT_DIR}/scripts/prepare_models.sh"
done

if [[ "${GUARD_CALIBRATION:-0}" -eq 1 ]]; then
  if [[ -z "${THERMAL_LOCK:-}" || ! -f "${THERMAL_LOCK}" ]]; then
    printf 'THERMAL_LOCK is required for guard calibration\n' >&2
    exit 1
  fi
  profile_args=(
    --bench "${BUILD_DIR}/jdg-trt-bench"
    --engine-root "${ENGINE_ROOT}"
    --mig-env "${RESULT_DIR}/mig.env"
    --thermal-lock "${THERMAL_LOCK}"
    --telemetry-log "${TELEMETRY_LOG}"
    --resident-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}"
    --resident-mps-log "${JDG_MPS_LOG_DIRECTORY}"
    --big-mps-pipe "${BIG_STATE_DIR}/pipe"
    --big-mps-log "${BIG_STATE_DIR}/log"
    --output "${RESULT_DIR}/guard-profile.json"
  )
  if [[ -n "${GUARD_CALIBRATION_MODE:-}" ]]; then
    profile_args+=(--mode "${GUARD_CALIBRATION_MODE}")
  fi
  taskset --cpu-list "${TELEMETRY_CPU}" \
    python3 "${ROOT_DIR}/runtime/profile_p9_guard.py" "${profile_args[@]}"
else
governor_args=()
if [[ -n "${GUARD_OVERRIDE_MS:-}" ]]; then
  governor_args+=(--guard-override-ms "${GUARD_OVERRIDE_MS}")
fi
if [[ -n "${DEADLINE_MS:-}" ]]; then
  governor_args+=(--deadline-ms "${DEADLINE_MS}")
fi
if [[ -n "${DEADLINE_SOURCE:-}" ]]; then
  governor_args+=(--deadline-source "${DEADLINE_SOURCE}")
fi
if [[ -n "${DEADLINE_LOCK_SHA256:-}" ]]; then
  governor_args+=(--deadline-lock-sha256 "${DEADLINE_LOCK_SHA256}")
fi
if [[ -n "${THERMAL_LOCK_SHA256:-}" ]]; then
  governor_args+=(--thermal-lock-sha256 "${THERMAL_LOCK_SHA256}")
fi
if [[ -n "${GUARD_LOCK:-}" ]]; then
  governor_args+=(--guard-lock "${GUARD_LOCK}")
fi
if [[ -n "${GUARD_LOCK_SHA256:-}" ]]; then
  governor_args+=(--guard-lock-sha256 "${GUARD_LOCK_SHA256}")
fi
if [[ "${CALIBRATION_ONLY:-0}" -eq 1 ]]; then
  governor_args+=(--calibration-only)
fi
if [[ -n "${THERMAL_PILOT_SECONDS:-}" ]]; then
  governor_args+=(--thermal-pilot-seconds "${THERMAL_PILOT_SECONDS}")
fi
if [[ -n "${THERMAL_TARGET_C:-}" ]]; then
  governor_args+=(--thermal-target-c "${THERMAL_TARGET_C}")
fi
if [[ -n "${THERMAL_TOLERANCE_C:-}" ]]; then
  governor_args+=(--thermal-tolerance-c "${THERMAL_TOLERANCE_C}")
fi
if [[ -n "${THERMAL_WINDOW_SECONDS:-}" ]]; then
  governor_args+=(--thermal-window-seconds "${THERMAL_WINDOW_SECONDS}")
fi
if [[ -n "${THERMAL_MAX_SLOPE_C_PER_MINUTE:-}" ]]; then
  governor_args+=(
    --thermal-max-slope-c-per-minute \
    "${THERMAL_MAX_SLOPE_C_PER_MINUTE}"
  )
fi
if [[ -n "${THERMAL_TIMEOUT_SECONDS:-}" ]]; then
  governor_args+=(--thermal-timeout-seconds "${THERMAL_TIMEOUT_SECONDS}")
fi
if [[ -n "${THERMAL_HARD_LIMIT_C:-}" ]]; then
  governor_args+=(--thermal-hard-limit-c "${THERMAL_HARD_LIMIT_C}")
fi
if [[ -n "${THERMAL_STABILITY_SENSOR:-}" ]]; then
  governor_args+=(--thermal-stability-sensor "${THERMAL_STABILITY_SENSOR}")
fi
if [[ -n "${THERMAL_SAFETY_SENSOR:-}" ]]; then
  governor_args+=(--thermal-safety-sensor "${THERMAL_SAFETY_SENSOR}")
fi
if [[ -n "${THERMAL_HANDOFF_MAX_MS:-}" ]]; then
  governor_args+=(--thermal-handoff-max-ms "${THERMAL_HANDOFF_MAX_MS}")
fi
if [[ -n "${MAX_ISOLATED_DRIFT_FRACTION:-}" ]]; then
  governor_args+=(
    --max-isolated-drift-fraction "${MAX_ISOLATED_DRIFT_FRACTION}"
  )
fi

taskset --cpu-list "${TELEMETRY_CPU}" \
  python3 "${ROOT_DIR}/runtime/mig_slack_governor.py" \
  --bench "${BUILD_DIR}/jdg-trt-bench" \
  --engine-root "${ENGINE_ROOT}" \
  --mig-env "${SMALL_STATE_DIR}/mig.env" \
  --big-mps-pipe "${BIG_STATE_DIR}/pipe" \
  --big-mps-log "${BIG_STATE_DIR}/log" \
  --telemetry-log "${TELEMETRY_LOG}" \
  --output "${RESULT_DIR}/summary.json" \
  --scenario "${SCENARIO}" \
  --epochs "${EPOCHS:-12}" \
  --samples "${SAMPLES:-800}" \
  --warmup "${WARMUP:-100}" \
  --burst-size "${BURST_SIZE:-8}" \
  --period-ms "${PERIOD_MS:-20}" \
  --pressure-rps-per-tenant "${PRESSURE_RPS_PER_TENANT:-0}" \
  --slo-factor "${SLO_FACTOR:-1.10}" \
  --dmr-target "${DMR_TARGET:-0.0005}" \
  --calibration-repeats "${CALIBRATION_REPEATS:-3}" \
  --readiness-timeout-seconds "${READINESS_TIMEOUT_SECONDS:-120}" \
  --critical-cpu "${CRITICAL_CPU}" \
  --pressure-cpus "${PRESSURE_CPUS}" \
  --mps-cpu "${MPS_CPU}" \
  --telemetry-cpu "${TELEMETRY_CPU}" \
  --policy-order "${POLICY_ORDER:-static-mig,resident-full-gate,same-mig,uncoordinated-borrow,fixed-borrow,fixed-full-gate,mig-governor}" \
  --borrower-quota "${BORROWER_QUOTA}" \
  --language-guard-ms "${LANGUAGE_GUARD_MS:-1.5}" \
  --audio-guard-ms "${AUDIO_GUARD_MS:-2}" \
  --experiment-label "${EXPERIMENT_LABEL:-main}" \
  "${governor_args[@]}"
fi

mkdir -p "${RESULT_DIR}/post-platform"
cat /etc/nv_tegra_release >"${RESULT_DIR}/post-platform/nv_tegra_release.txt"
nvidia-smi -q >"${RESULT_DIR}/post-platform/nvidia-smi.txt"
sudo_cmd nvidia-smi mig -lgi \
  >"${RESULT_DIR}/post-platform/active-mig-instances.txt"
nvidia-smi -L >"${RESULT_DIR}/post-platform/gpu-inventory.txt"
sudo_cmd jetson_clocks --show \
  >"${RESULT_DIR}/post-platform/jetson-clocks.txt" 2>&1 || true
nvpmodel -q >"${RESULT_DIR}/post-platform/nvpmodel.txt" 2>&1
: >"${RESULT_DIR}/post-platform/mps-thread-affinity.tsv"
verify_mps_processes "${SMALL_STATE_DIR}" "resident-1g" \
  "${RESULT_DIR}/post-platform/mps-thread-affinity.tsv"
verify_mps_processes "${BIG_STATE_DIR}" "critical-2g" \
  "${RESULT_DIR}/post-platform/mps-thread-affinity.tsv"
if ! cmp -s "${RESULT_DIR}/active-mig-instances.txt" \
    "${RESULT_DIR}/post-platform/active-mig-instances.txt" || \
   ! cmp -s "${RESULT_DIR}/gpu-inventory.txt" \
    "${RESULT_DIR}/post-platform/gpu-inventory.txt" || \
   ! grep -q 'FAN Dynamic Speed Control=disabled .*pwm1=255' \
    "${RESULT_DIR}/post-platform/jetson-clocks.txt" || \
   ! grep -q 'gpu-gpc-0 MinFreq=1575000000 MaxFreq=1575000000' \
    "${RESULT_DIR}/post-platform/jetson-clocks.txt" || \
   ! grep -q 'EMC MinFreq=4266000000 MaxFreq=4266000000' \
    "${RESULT_DIR}/post-platform/jetson-clocks.txt" || \
   ! diff -u \
     <(grep -E '^(Online CPUs:|cpu[0-9]+:)' \
       "${RESULT_DIR}/jetson-clocks.txt") \
     <(grep -E '^(Online CPUs:|cpu[0-9]+:)' \
       "${RESULT_DIR}/post-platform/jetson-clocks.txt") >/dev/null || \
   ! grep -q 'NV Power Mode: MAXN' \
    "${RESULT_DIR}/post-platform/nvpmodel.txt"; then
  printf 'platform state changed during the experiment\n' >&2
  exit 1
fi

printf 'P9 MIG slack-governor results: %s\n' "${RESULT_DIR}"
