#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p0-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly SAMPLES="${SAMPLES:-500}"
readonly WARMUP="${WARMUP:-50}"
readonly CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-native}"
clock_state_file=""
clocks_pinned=0

restore_clocks() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "${clocks_pinned}" -eq 1 ]]; then
    sudo jetson_clocks --restore "${clock_state_file}" || true
  fi
  exit "${status}"
}
trap restore_clocks EXIT INT TERM

capture_tegrastats() {
  local output="$1"
  local status=0
  timeout --signal=INT 0.35s tegrastats --interval 100 >"${output}" 2>&1 || status=$?
  if [[ "${status}" -ne 0 && "${status}" -ne 124 && "${status}" -ne 130 ]]; then
    return "${status}"
  fi
}

mkdir -p "${RESULT_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

if [[ "${PIN_CLOCKS:-0}" == "1" ]]; then
  clock_state_file="${RESULT_DIR}/jetson-clocks-before.conf"
  sudo jetson_clocks --store "${clock_state_file}"
  sudo jetson_clocks
  clocks_pinned=1
fi

"${BUILD_DIR}/jdg-probe" >"${RESULT_DIR}/capabilities.json.tmp"
mv "${RESULT_DIR}/capabilities.json.tmp" "${RESULT_DIR}/capabilities.json"
"${ROOT_DIR}/scripts/probe_mps.sh" "${BUILD_DIR}/jdg-bench" \
  "${RESULT_DIR}/mps" >"${RESULT_DIR}/mps-runtime.txt" 2>&1

nvpmodel -q >"${RESULT_DIR}/nvpmodel.txt" 2>&1 || true
nvidia-smi -q >"${RESULT_DIR}/nvidia-smi.txt" 2>&1 || true
capture_tegrastats "${RESULT_DIR}/tegrastats-before.txt" || true

for background in none compute memory; do
  output="${RESULT_DIR}/${background}.json"
  "${BUILD_DIR}/jdg-bench" \
    --background "${background}" \
    --samples "${SAMPLES}" \
    --warmup "${WARMUP}" >"${output}.tmp"
  mv "${output}.tmp" "${output}"
done

capture_tegrastats "${RESULT_DIR}/tegrastats-after.txt" || true

python3 "${ROOT_DIR}/analysis/summarize_p0.py" "${RESULT_DIR}" \
  >"${RESULT_DIR}/summary.json.tmp"
mv "${RESULT_DIR}/summary.json.tmp" "${RESULT_DIR}/summary.json"

printf 'P0 results: %s\n' "${RESULT_DIR}"
