#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly XSCHED_ROOT="${XSCHED_ROOT:-/tmp/quiet-xsched-1786268346828168599}"
readonly XSCHED_OUTPUT="${XSCHED_OUTPUT:-${XSCHED_ROOT}/output-thor}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly ENGINE="${ENGINE:-${ROOT_DIR}/models/engines/mig-2g/resnet10-detection.engine}"
readonly BE_SECONDS="${BE_SECONDS:-10}"
readonly HP_REQUESTS="${HP_REQUESTS:-100}"
readonly CONTROL_CPU="${CONTROL_CPU:-13}"
readonly BE_CPU="${BE_CPU:-0}"
readonly HP_CPU="${HP_CPU:-12}"
readonly SERVER_PORT="${SERVER_PORT:-50000}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-xsched-native-positive-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly SERVER="${XSCHED_OUTPUT}/bin/xserver"
readonly BENCH="${BUILD_DIR}/jdg-trt-bench"
readonly CUDA_DRIVER="/opt/nvidia/l4t-gpu-libs/openrm/libcuda.so.1.1"

for path in "${SERVER}" "${BENCH}" "${MIG_ENV}" "${ENGINE}" \
            "${XSCHED_OUTPUT}/lib/libshimcuda.so" "${CUDA_DRIVER}"; do
  [[ -e "${path}" ]] || { printf 'missing XSched native input: %s\n' "${path}" >&2; exit 1; }
done
[[ ! -e "${RESULT_DIR}" ]] || { printf 'result directory exists: %s\n' "${RESULT_DIR}" >&2; exit 1; }
mkdir -p "${RESULT_DIR}"

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"

server_pid=""
be_pid=""
cleanup() {
  local status=$?
  if [[ -n "${be_pid}" ]] && kill -0 "${be_pid}" 2>/dev/null; then
    kill -INT "${be_pid}" 2>/dev/null || true
    wait "${be_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

taskset --cpu-list "${CONTROL_CPU}" \
  stdbuf -oL -eL "${SERVER}" HPF "${SERVER_PORT}" \
  >"${RESULT_DIR}/server.log" 2>&1 &
server_pid=$!
sleep 1
kill -0 "${server_pid}" 2>/dev/null || {
  printf 'XSched server failed to start\n' >&2
  exit 1
}

run_client() {
  local priority=$1
  local threshold=$2
  local batch_size=$3
  shift 3
  env \
    CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}" \
    LD_LIBRARY_PATH="${XSCHED_OUTPUT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    XSCHED_CUDA_LIB="${CUDA_DRIVER}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY="${priority}" \
    XSCHED_AUTO_XQUEUE_THRESHOLD="${threshold}" \
    XSCHED_AUTO_XQUEUE_BATCH_SIZE="${batch_size}" \
    XSCHED_TRT_USER_STREAM_ONLY=ON \
    "$@"
}

run_client 0 16 8 \
  timeout 20s taskset --cpu-list "${BE_CPU}" "${BENCH}" \
    --engine "${ENGINE}" --model-name resnet10-detection \
    --role pressure --warmup 20 --duration-seconds "${BE_SECONDS}" \
    --priority low --include-transfers true \
    --trace "${RESULT_DIR}/be.csv" \
    >"${RESULT_DIR}/be.json" 2>"${RESULT_DIR}/be.log" &
be_pid=$!

# The pressure client must already own a global XQueue before HP arrives.
for _ in $(seq 1 100); do
  grep -q 'using global scheduler' "${RESULT_DIR}/be.log" 2>/dev/null && break
  kill -0 "${be_pid}" 2>/dev/null || { printf 'BE client exited before HP launch\n' >&2; exit 1; }
  sleep 0.05
done
grep -q 'using global scheduler' "${RESULT_DIR}/be.log" || {
  printf 'BE client did not connect to the global scheduler\n' >&2
  exit 1
}
sleep 0.5

run_client 10 1 1 \
  timeout 20s taskset --cpu-list "${HP_CPU}" "${BENCH}" \
    --engine "${ENGINE}" --model-name resnet10-detection \
    --role benchmark --samples "${HP_REQUESTS}" --warmup 20 \
    --priority high --include-transfers true \
    --trace "${RESULT_DIR}/hp.csv" \
    >"${RESULT_DIR}/hp.json" 2>"${RESULT_DIR}/hp.log"

wait "${be_pid}"
be_pid=""
sleep 0.2

python3 "${ROOT_DIR}/baselines/xsched/verify_native_smoke.py" \
  --be "${RESULT_DIR}/be.json" --hp "${RESULT_DIR}/hp.json" \
  --server-log "${RESULT_DIR}/server.log" \
  --be-log "${RESULT_DIR}/be.log" --hp-log "${RESULT_DIR}/hp.log" \
  --output "${RESULT_DIR}/verification.json"

git -C "${XSCHED_ROOT}" rev-parse HEAD >"${RESULT_DIR}/xsched-commit.txt"
sha256sum "${BENCH}" "${SERVER}" "${ENGINE}" \
  "${ROOT_DIR}/baselines/xsched/patches/thor-cuda13-tensorrt.patch" \
  "${RESULT_DIR}/be.json" "${RESULT_DIR}/hp.json" \
  "${RESULT_DIR}/be.csv" "${RESULT_DIR}/hp.csv" \
  "${RESULT_DIR}/server.log" "${RESULT_DIR}/verification.json" \
  >"${RESULT_DIR}/SHA256SUMS"

trap - EXIT INT TERM
kill -INT "${server_pid}" 2>/dev/null || true
wait "${server_pid}" 2>/dev/null || true
server_pid=""
printf 'XSched native two-client positive control: %s\n' "${RESULT_DIR}"
