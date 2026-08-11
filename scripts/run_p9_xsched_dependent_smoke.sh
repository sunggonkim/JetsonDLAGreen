#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly XSCHED_ROOT="${XSCHED_ROOT:-/tmp/quiet-xsched-1786268346828168599}"
readonly XSCHED_OUTPUT="${XSCHED_OUTPUT:-${XSCHED_ROOT}/output-thor}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly DEADLINE_LOCK="${DEADLINE_LOCK:?set DEADLINE_LOCK to a verified pipeline deadline lock}"
readonly SERVER_PORT="${SERVER_PORT:-50006}"
readonly CRITICAL_REQUESTS="${CRITICAL_REQUESTS:-1000}"
readonly BE_REQUESTS="${BE_REQUESTS:-5000}"
readonly WARMUP="${WARMUP:-20}"
readonly WORKLOAD="${WORKLOAD:-whisper-projection}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-xsched-dependent-${WORKLOAD}-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly SERVER="${XSCHED_OUTPUT}/bin/xserver"
readonly SHIM="${XSCHED_OUTPUT}/lib/libshimcuda.so"
readonly PIPELINE="${ROOT_DIR}/build-r39/jdg-mig-trt-pipeline"
readonly BENCH="${ROOT_DIR}/build-r39/jdg-trt-bench"
readonly CUDA_DRIVER="/opt/nvidia/l4t-gpu-libs/openrm/libcuda.so.1.1"
readonly WHISPER="${ROOT_DIR}/models/engines/mig-1g-q100/whisper-tiny-encoder.engine"
readonly RESNET_BACKBONE="${ROOT_DIR}/models/engines/mig-1g-q100/resnet10-backbone.engine"
readonly RESNET50_BACKBONE="${ROOT_DIR}/models/engines/mig-1g-q100/resnet50-backbone.engine"
readonly CONSUMER_ENGINE="${CONSUMER_ENGINE:-}"
OPERATIONAL_ARRIVAL_TRACE="${OPERATIONAL_ARRIVAL_TRACE:-}"
CONSUMER_INPUT_TENSOR="${CONSUMER_INPUT_TENSOR:-features}"
readonly DISTILBERT="${ROOT_DIR}/models/engines/mig-1g-q100/distilbert-sst2.engine"
readonly PATCH="${ROOT_DIR}/baselines/xsched/patches/thor-cuda13-tensorrt.patch"
APPLICATION_OUTPUT_TRACE="${APPLICATION_OUTPUT_TRACE:-}"
PRODUCER_INPUT_TRACE="${PRODUCER_INPUT_TRACE:-}"
COMMON_WORKLOAD_CONTRACT="${COMMON_WORKLOAD_CONTRACT:-}"
REQUIRE_COMMON_WORKLOAD="${REQUIRE_COMMON_WORKLOAD:-0}"

case "${WORKLOAD}" in
  whisper-projection) PRODUCER_ENGINE="${PRODUCER_ENGINE:-${WHISPER}}"; DEFAULT_INPUT="features" ;;
  resnet-detection-head) PRODUCER_ENGINE="${PRODUCER_ENGINE:-${RESNET_BACKBONE}}"; DEFAULT_INPUT="Layer6_relu_Y" ;;
  resnet50-classification) PRODUCER_ENGINE="${PRODUCER_ENGINE:-${RESNET50_BACKBONE}}"; DEFAULT_INPUT="gpu_0/res4_5_branch2c_bn_2" ;;
  *) printf 'unsupported XSched workload: %s\n' "${WORKLOAD}" >&2; exit 1 ;;
esac
if [[ "${CONSUMER_INPUT_TENSOR}" == "features" && "${WORKLOAD}" == "resnet-detection-head" ]]; then
  CONSUMER_INPUT_TENSOR="${DEFAULT_INPUT}"
fi
if [[ "${CONSUMER_INPUT_TENSOR}" == "features" && "${WORKLOAD}" == "resnet50-classification" ]]; then
  CONSUMER_INPUT_TENSOR="${DEFAULT_INPUT}"
fi
readonly PRODUCER_ENGINE CONSUMER_INPUT_TENSOR
for path in "${SERVER}" "${SHIM}" "${MIG_ENV}" "${DEADLINE_LOCK}" \
            "${PIPELINE}" "${BENCH}" "${CUDA_DRIVER}" "${PRODUCER_ENGINE}" \
            "${DISTILBERT}" "${PATCH}"; do
  [[ -e "${path}" ]] || { printf 'missing XSched input: %s\n' "${path}" >&2; exit 1; }
done
if [[ -n "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  [[ -f "${COMMON_WORKLOAD_CONTRACT}" ]] || { printf 'missing common workload contract: %s\n' "${COMMON_WORKLOAD_CONTRACT}" >&2; exit 1; }
fi
if [[ "${REQUIRE_COMMON_WORKLOAD}" == 1 && -z "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  printf 'REQUIRE_COMMON_WORKLOAD requires COMMON_WORKLOAD_CONTRACT\n' >&2; exit 1
fi
[[ ! -e "${RESULT_DIR}" ]] || { printf 'result directory exists: %s\n' "${RESULT_DIR}" >&2; exit 1; }
mkdir -p "${RESULT_DIR}"
if [[ -n "${APPLICATION_OUTPUT_TRACE}" ]]; then
  result_root="$(realpath -m "${RESULT_DIR}")"
  trace_path="$(realpath -m "${APPLICATION_OUTPUT_TRACE}")"
  case "${trace_path}" in
    "${result_root}"/*) APPLICATION_OUTPUT_TRACE="${trace_path}" ;;
    *) printf 'APPLICATION_OUTPUT_TRACE must be inside RESULT_DIR for provenance\n' >&2; exit 1 ;;
  esac
fi
readonly APPLICATION_OUTPUT_TRACE
if [[ -n "${PRODUCER_INPUT_TRACE}" ]]; then
  [[ -f "${PRODUCER_INPUT_TRACE}" ]] || { printf 'missing producer input trace: %s\n' "${PRODUCER_INPUT_TRACE}" >&2; exit 1; }
fi
if [[ -n "${OPERATIONAL_ARRIVAL_TRACE}" ]]; then
  [[ -f "${OPERATIONAL_ARRIVAL_TRACE}" ]] || { printf 'missing operational arrival trace: %s\n' "${OPERATIONAL_ARRIVAL_TRACE}" >&2; exit 1; }
fi

python3 "${ROOT_DIR}/analysis/freeze_p9_pipeline_deadline.py" \
  --verify "${DEADLINE_LOCK}" >/dev/null
runtime_lock_args=(
  python3 "${ROOT_DIR}/analysis/verify_p9_runtime_lock.py"
  --lock "${DEADLINE_LOCK}" --repo "${ROOT_DIR}"
  --producer-engine "${PRODUCER_ENGINE}"
)
if [[ -n "${CONSUMER_ENGINE}" ]]; then
  runtime_lock_args+=(--consumer-engine "${CONSUMER_ENGINE}")
fi
"${runtime_lock_args[@]}" >/dev/null
DEADLINE_US="$(jq -er '.deadline_us' "${DEADLINE_LOCK}")"
readonly DEADLINE_US
cp "${DEADLINE_LOCK}" "${RESULT_DIR}/deadline-lock.json"
mkdir -p "${RESULT_DIR}/provenance"
cp "${PIPELINE}" "${RESULT_DIR}/provenance/jdg-mig-trt-pipeline"
cp "${SERVER}" "${RESULT_DIR}/provenance/xserver"
cp "${SHIM}" "${RESULT_DIR}/provenance/libshimcuda.so"
cp "${PATCH}" "${RESULT_DIR}/provenance/thor-cuda13-tensorrt.patch"
cp "${PRODUCER_ENGINE}" "${RESULT_DIR}/provenance/$(basename "${PRODUCER_ENGINE}")"
if [[ -n "${PRODUCER_INPUT_TRACE}" ]]; then
  cp "${PRODUCER_INPUT_TRACE}" "${RESULT_DIR}/provenance/producer-input-trace.bin"
  PRODUCER_INPUT_TRACE="${RESULT_DIR}/provenance/producer-input-trace.bin"
fi
if [[ -n "${OPERATIONAL_ARRIVAL_TRACE}" ]]; then
  cp "${OPERATIONAL_ARRIVAL_TRACE}" "${RESULT_DIR}/provenance/operational-arrival-trace.bin"
  OPERATIONAL_ARRIVAL_TRACE="${RESULT_DIR}/provenance/operational-arrival-trace.bin"
fi
if [[ -n "${CONSUMER_ENGINE}" ]]; then
  [[ -f "${CONSUMER_ENGINE}" ]] || { printf 'missing consumer engine: %s\n' "${CONSUMER_ENGINE}" >&2; exit 1; }
  cp "${CONSUMER_ENGINE}" "${RESULT_DIR}/provenance/$(basename "${CONSUMER_ENGINE}")"
fi

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
    kill -TERM "${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "${server_pid}" 2>/dev/null || break
      sleep 0.05
    done
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill -KILL "${server_pid}" 2>/dev/null || true
    fi
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
  --samples "${BE_REQUESTS}" --warmup 20 --burst-size 1 --period-ms 4 \
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
  --producer-engine "${PRODUCER_ENGINE}" --producer "${JDG_MIG_SMALL_UUID}"
  --consumer "${JDG_MIG_BIG_UUID}"
  --producer-mps-pipe "${JDG_MPS_PIPE_DIRECTORY}"
  --workload "${WORKLOAD}" --transport registered-direct
  --consumer-input-tensor "${CONSUMER_INPUT_TENSOR}"
  --deadline-mode wall --deadline-us "${DEADLINE_US}"
  --warmup "${WARMUP}" --iterations "${CRITICAL_REQUESTS}"
  --trace-csv "${RESULT_DIR}/pipeline.csv"
)
if [[ -n "${CONSUMER_ENGINE}" ]]; then
  pipeline_args+=(--consumer-engine "${CONSUMER_ENGINE}")
fi
if [[ -n "${APPLICATION_OUTPUT_TRACE}" ]]; then
  mkdir -p "$(dirname "${APPLICATION_OUTPUT_TRACE}")"
  pipeline_args+=(--application-output-trace "${APPLICATION_OUTPUT_TRACE}")
fi
if [[ -n "${PRODUCER_INPUT_TRACE}" ]]; then
  pipeline_args+=(--producer-input-trace "${PRODUCER_INPUT_TRACE}")
fi
if [[ -n "${OPERATIONAL_ARRIVAL_TRACE}" ]]; then
  pipeline_args+=(--arrival-trace "${OPERATIONAL_ARRIVAL_TRACE}")
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
  --be-result "${RESULT_DIR}/be.json" --be-trace "${RESULT_DIR}/be.csv"
  --server-log "${RESULT_DIR}/server.log"
  --client-log "${RESULT_DIR}/be.log" --client-log "${RESULT_DIR}/result.log"
  --deadline-lock "${RESULT_DIR}/deadline-lock.json"
  --pipeline-binary "${RESULT_DIR}/provenance/jdg-mig-trt-pipeline"
  --xsched-server "${RESULT_DIR}/provenance/xserver"
  --xsched-shim "${RESULT_DIR}/provenance/libshimcuda.so"
  --xsched-patch "${RESULT_DIR}/provenance/thor-cuda13-tensorrt.patch"
  --workload "${WORKLOAD}"
  --producer-engine "${RESULT_DIR}/provenance/$(basename "${PRODUCER_ENGINE}")"
)
if [[ -n "${CONSUMER_ENGINE}" ]]; then
  verify_args+=(--consumer-engine "${RESULT_DIR}/provenance/$(basename "${CONSUMER_ENGINE}")")
fi
if [[ -n "${APPLICATION_OUTPUT_TRACE}" ]]; then
  verify_args+=(--application-output-trace "${APPLICATION_OUTPUT_TRACE}")
fi
if [[ -n "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  verify_args+=(--common-workload-contract "${COMMON_WORKLOAD_CONTRACT}")
  [[ "${REQUIRE_COMMON_WORKLOAD}" == 1 ]] && verify_args+=(--require-common-workload)
fi
python3 "${ROOT_DIR}/baselines/xsched/verify_dependent_smoke.py" \
  "${verify_args[@]}" --repo "${ROOT_DIR}" \
  --output "${RESULT_DIR}/verification.json" >/dev/null

git -C "${XSCHED_ROOT}" rev-parse HEAD >"${RESULT_DIR}/xsched-commit.txt"
find "${RESULT_DIR}" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"${RESULT_DIR}/SHA256SUMS"
printf 'XSched dependent numeric smoke: %s\n' "${RESULT_DIR}"
