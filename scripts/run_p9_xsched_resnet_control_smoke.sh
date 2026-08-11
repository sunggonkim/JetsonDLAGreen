#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly XSCHED_ROOT="${XSCHED_ROOT:-/tmp/quiet-xsched-1786268346828168599}"
readonly XSCHED_OUTPUT="${XSCHED_OUTPUT:-${XSCHED_ROOT}/output-thor}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly SERVER_PORT="${SERVER_PORT:-50006}"
readonly DEADLINE_LOCK="${DEADLINE_LOCK:?set DEADLINE_LOCK to a verified pipeline deadline lock}"
readonly CRITICAL_REQUESTS="${CRITICAL_REQUESTS:-100}"
readonly BE_REQUESTS="${BE_REQUESTS:-5000}"
readonly BACKGROUND_PERIOD_MS="${BACKGROUND_PERIOD_MS:-4}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-xsched-resnet-control-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly SERVER="${XSCHED_OUTPUT}/bin/xserver"
readonly SHIM="${XSCHED_OUTPUT}/lib/libshimcuda.so"
readonly PIPELINE="${ROOT_DIR}/build-r39/jdg-mig-trt-pipeline"
readonly BENCH="${ROOT_DIR}/build-r39/jdg-trt-bench"
readonly CUDA_DRIVER="/opt/nvidia/l4t-gpu-libs/openrm/libcuda.so.1.1"
readonly RESNET="${ROOT_DIR}/models/engines/mig-1g-q100/resnet10-detection.engine"
readonly DISTILBERT="${ROOT_DIR}/models/engines/mig-1g-q100/distilbert-sst2.engine"
readonly PATCH="${ROOT_DIR}/baselines/xsched/patches/thor-cuda13-tensorrt.patch"
APPLICATION_OUTPUT_TRACE="${APPLICATION_OUTPUT_TRACE:-}"
COMMON_WORKLOAD_CONTRACT="${COMMON_WORKLOAD_CONTRACT:-}"
REQUIRE_COMMON_WORKLOAD="${REQUIRE_COMMON_WORKLOAD:-0}"

for path in "${SERVER}" "${SHIM}" "${MIG_ENV}" "${PIPELINE}" "${BENCH}" \
            "${CUDA_DRIVER}" "${RESNET}" "${DISTILBERT}" "${PATCH}" "${DEADLINE_LOCK}"; do
  [[ -e "${path}" ]] || { printf 'missing XSched input: %s\n' "${path}" >&2; exit 1; }
done
if [[ -n "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  [[ -f "${COMMON_WORKLOAD_CONTRACT}" ]] || { printf 'missing common workload contract: %s\n' "${COMMON_WORKLOAD_CONTRACT}" >&2; exit 1; }
fi
if [[ "${REQUIRE_COMMON_WORKLOAD}" == 1 && -z "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  printf 'REQUIRE_COMMON_WORKLOAD requires COMMON_WORKLOAD_CONTRACT\n' >&2; exit 1
fi
DEADLINE_KIND="$(jq -er '.kind' "${DEADLINE_LOCK}")"
readonly DEADLINE_KIND
case "${DEADLINE_KIND}" in
  p9-common-placement-deadline-lock)
    python3 "${ROOT_DIR}/analysis/freeze_p9_common_placement_deadline.py" \
      --verify "${DEADLINE_LOCK}" >/dev/null
    ;;
  p9-dependent-pipeline-deadline-lock)
    python3 "${ROOT_DIR}/analysis/freeze_p9_pipeline_deadline.py" \
      --verify "${DEADLINE_LOCK}" >/dev/null
    ;;
  *)
    printf 'unsupported XSched deadline-lock kind: %s\n' "${DEADLINE_KIND}" >&2
    exit 1
    ;;
esac
DEADLINE_US="$(jq -er '.deadline_us' "${DEADLINE_LOCK}")"
readonly DEADLINE_US
[[ "${CRITICAL_REQUESTS}" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid CRITICAL_REQUESTS\n' >&2; exit 1; }
awk "BEGIN { exit !(${BACKGROUND_PERIOD_MS} > 0) }" || { printf 'invalid BACKGROUND_PERIOD_MS\n' >&2; exit 1; }
[[ ! -e "${RESULT_DIR}" ]] || { printf 'result directory exists: %s\n' "${RESULT_DIR}" >&2; exit 1; }
mkdir -p "${RESULT_DIR}/provenance"
if [[ -n "${APPLICATION_OUTPUT_TRACE}" ]]; then
  result_root="$(realpath -m "${RESULT_DIR}")"
  trace_path="$(realpath -m "${APPLICATION_OUTPUT_TRACE}")"
  case "${trace_path}" in
    "${result_root}"/*) APPLICATION_OUTPUT_TRACE="${trace_path}" ;;
    *) printf 'APPLICATION_OUTPUT_TRACE must be inside RESULT_DIR for provenance\n' >&2; exit 1 ;;
  esac
fi
readonly APPLICATION_OUTPUT_TRACE
cp "${DEADLINE_LOCK}" "${RESULT_DIR}/deadline-lock.json"
cp "${PIPELINE}" "${RESULT_DIR}/provenance/jdg-mig-trt-pipeline"
cp "${SERVER}" "${RESULT_DIR}/provenance/xserver"
cp "${SHIM}" "${RESULT_DIR}/provenance/libshimcuda.so"
cp "${PATCH}" "${RESULT_DIR}/provenance/thor-cuda13-tensorrt.patch"
cp "${RESNET}" "${RESULT_DIR}/provenance/resnet10-detection.engine"
cp "${DISTILBERT}" "${RESULT_DIR}/provenance/distilbert-sst2.engine"
git -C "${XSCHED_ROOT}" rev-parse HEAD >"${RESULT_DIR}/xsched-commit.txt"

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_SMALL_UUID:?missing JDG_MIG_SMALL_UUID}"
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"
: "${JDG_MPS_PIPE_DIRECTORY:?missing JDG_MPS_PIPE_DIRECTORY}"

server_pid=""
be_pid=""
stop_server() {
  [[ -n "${server_pid}" ]] || return 0
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${server_pid}" 2>/dev/null || break
      sleep 0.05
    done
  fi
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${server_pid}" 2>/dev/null || break
      sleep 0.05
    done
  fi
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -KILL "${server_pid}" 2>/dev/null || true
  fi
  wait "${server_pid}" 2>/dev/null || true
  server_pid=""
}
cleanup() {
  local status=$?
  if [[ -n "${be_pid}" ]] && kill -0 "${be_pid}" 2>/dev/null; then
    kill -INT "${be_pid}" 2>/dev/null || true
    wait "${be_pid}" 2>/dev/null || true
  fi
  stop_server
  exit "${status}"
}
trap cleanup EXIT INT TERM

taskset --cpu-list 13 stdbuf -oL -eL "${SERVER}" HPF "${SERVER_PORT}" \
  >"${RESULT_DIR}/server.log" 2>&1 &
server_pid=$!
sleep 1
kill -0 "${server_pid}"

common_env=(
  "LD_LIBRARY_PATH=${XSCHED_OUTPUT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  "XSCHED_CUDA_LIB=${CUDA_DRIVER}"
  "XSCHED_SCHEDULER=GLB"
  "XSCHED_AUTO_XQUEUE=ON"
  "XSCHED_AUTO_XQUEUE_LEVEL=1"
  "XSCHED_TRT_USER_STREAM_ONLY=ON"
  "XSCHED_DEFER_INIT_UNTIL_FIRST_EVENT=ON"
)

env "${common_env[@]}" CUDA_VISIBLE_DEVICES="${JDG_MIG_SMALL_UUID}" \
  XSCHED_AUTO_XQUEUE_PRIORITY=0 XSCHED_AUTO_XQUEUE_THRESHOLD=1 \
  XSCHED_AUTO_XQUEUE_BATCH_SIZE=1 taskset --cpu-list 0 "${BENCH}" \
  --engine "${DISTILBERT}" --model-name distilbert-sst2 --role benchmark \
  --samples "${BE_REQUESTS}" --warmup 20 --burst-size 1 --period-ms "${BACKGROUND_PERIOD_MS}" \
  --priority low --include-transfers true --trace "${RESULT_DIR}/be.csv" \
  >"${RESULT_DIR}/be.json" 2>"${RESULT_DIR}/be.log" &
be_pid=$!
for _ in $(seq 1 100); do
  grep -q 'using global scheduler' "${RESULT_DIR}/be.log" 2>/dev/null && break
  kill -0 "${be_pid}" 2>/dev/null || { printf 'XSched BE exited early\n' >&2; exit 1; }
  sleep 0.05
done
grep -q 'using global scheduler' "${RESULT_DIR}/be.log"
sleep 0.5

pipeline_args=(
  --producer-engine "${RESNET}" --producer "${JDG_MIG_SMALL_UUID}"
  --consumer "${JDG_MIG_BIG_UUID}"
  --producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}"
  --workload resnet-control --transport registered-direct
  --deadline-mode wall --deadline-us "${DEADLINE_US}"
  --warmup 10 --iterations "${CRITICAL_REQUESTS}"
  --trace-csv "${RESULT_DIR}/pipeline.csv"
  --checksum-trace-csv "${RESULT_DIR}/checksums.csv"
)
if [[ -n "${APPLICATION_OUTPUT_TRACE}" ]]; then
  mkdir -p "$(dirname "${APPLICATION_OUTPUT_TRACE}")"
  pipeline_args+=(--application-output-trace "${APPLICATION_OUTPUT_TRACE}")
fi
env "${common_env[@]}" XSCHED_AUTO_XQUEUE_PRIORITY=10 \
  XSCHED_AUTO_XQUEUE_THRESHOLD=1 XSCHED_AUTO_XQUEUE_BATCH_SIZE=1 \
  taskset --cpu-list 10-12 "${PIPELINE}" "${pipeline_args[@]}" \
  >"${RESULT_DIR}/result.json" 2>"${RESULT_DIR}/result.log"

wait "${be_pid}"
be_pid=""
stop_server
trap - EXIT INT TERM

verify_args=(
  --result "${RESULT_DIR}/result.json"
  --pipeline "${RESULT_DIR}/pipeline.csv"
  --checksums "${RESULT_DIR}/checksums.csv"
  --be-result "${RESULT_DIR}/be.json" --be-trace "${RESULT_DIR}/be.csv"
  --server-log "${RESULT_DIR}/server.log"
  --client-log "${RESULT_DIR}/be.log" --client-log "${RESULT_DIR}/result.log"
  --pipeline-binary "${RESULT_DIR}/provenance/jdg-mig-trt-pipeline"
  --producer-engine "${RESULT_DIR}/provenance/resnet10-detection.engine"
  --background-engine "${RESULT_DIR}/provenance/distilbert-sst2.engine"
  --xsched-server "${RESULT_DIR}/provenance/xserver"
  --xsched-shim "${RESULT_DIR}/provenance/libshimcuda.so"
  --xsched-patch "${RESULT_DIR}/provenance/thor-cuda13-tensorrt.patch"
  --xsched-commit "${RESULT_DIR}/xsched-commit.txt"
  --deadline-lock "${RESULT_DIR}/deadline-lock.json"
  --deadline-us "${DEADLINE_US}" --expected-requests "${CRITICAL_REQUESTS}"
  --background-period-ms "${BACKGROUND_PERIOD_MS}"
)
if [[ -n "${APPLICATION_OUTPUT_TRACE}" ]]; then
  verify_args+=(--application-output-trace "${APPLICATION_OUTPUT_TRACE}")
fi
if [[ -n "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  verify_args+=(--common-workload-contract "${COMMON_WORKLOAD_CONTRACT}")
  [[ "${REQUIRE_COMMON_WORKLOAD}" == 1 ]] && verify_args+=(--require-common-workload)
fi
python3 "${ROOT_DIR}/baselines/xsched/verify_resnet_control_smoke.py" \
  "${verify_args[@]}" --output "${RESULT_DIR}/verification.json" >/dev/null

python3 "${ROOT_DIR}/baselines/xsched/normalize_resnet_control_smoke.py" \
  --verification "${RESULT_DIR}/verification.json" \
  --deadline-lock "${RESULT_DIR}/deadline-lock.json" \
  --background-period-ms "${BACKGROUND_PERIOD_MS}" \
  --output "${RESULT_DIR}/summary.json" >/dev/null

find "${RESULT_DIR}" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"${RESULT_DIR}/SHA256SUMS"
printf 'XSched ResNet control numeric smoke: %s\n' "${RESULT_DIR}"
