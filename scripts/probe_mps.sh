#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: probe_mps.sh JDG_BENCH OUTPUT_DIRECTORY" >&2
  exit 2
fi

readonly BENCHMARK="$1"
readonly OUTPUT_DIR="$2"

if [[ ! -x "${BENCHMARK}" ]] || \
   ! command -v nvidia-cuda-mps-control >/dev/null || \
   ! command -v nvidia-cuda-mps-server >/dev/null; then
  echo "status=unavailable"
  echo "reason=required executable is missing"
  exit 0
fi

if pgrep -f '[n]vidia-cuda-mps-(control|server)' >/dev/null; then
  echo "status=skipped"
  echo "reason=an existing MPS process must not be disrupted"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}/pipe" "${OUTPUT_DIR}/log"
export CUDA_MPS_PIPE_DIRECTORY="${OUTPUT_DIR}/pipe"
export CUDA_MPS_LOG_DIRECTORY="${OUTPUT_DIR}/log"

started=0
stop_daemon() {
  if [[ "${started}" -eq 1 ]]; then
    printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}
trap stop_daemon EXIT INT TERM

if ! nvidia-cuda-mps-control -d; then
  echo "status=unavailable"
  echo "reason=control daemon failed to start"
  exit 0
fi
started=1

if ! "${BENCHMARK}" --background none --samples 3 --warmup 1 \
     >"${OUTPUT_DIR}/client.json" 2>"${OUTPUT_DIR}/client.stderr"; then
  echo "status=unavailable"
  echo "reason=CUDA client failed through MPS"
  exit 0
fi

server_list="$(printf 'get_server_list\n' | nvidia-cuda-mps-control)"
if [[ -z "${server_list}" ]]; then
  echo "status=unavailable"
  echo "reason=MPS did not create a server for the CUDA client"
  exit 0
fi

echo "status=supported"
echo "server_ids=${server_list//$'\n'/,}"
