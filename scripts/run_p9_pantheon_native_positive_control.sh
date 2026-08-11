#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly PANTHEON_SRC="${PANTHEON_SRC:-/tmp/quiet-pantheon-1786270298}"
readonly PANTHEON_BUILD="${PANTHEON_BUILD:-${PANTHEON_SRC}/build-thor}"
readonly PANTHEON_PYTHON="${PANTHEON_PYTHON:-/tmp/pantheon-venv/bin/python}"
readonly NVPL_DIR="${NVPL_DIR:-/tmp/pantheon-nvpl}"
readonly CUDSS_DIR="${CUDSS_DIR:-/tmp/pantheon-venv/lib/python3.12/site-packages/nvidia/cu13/lib}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly CPU="${CPU:-12}"
readonly RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/p9-pantheon-native-positive-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly RUNTIME="${PANTHEON_BUILD}/runtime"

for path in "${RUNTIME}" "${PANTHEON_PYTHON}" "${MIG_ENV}"; do
  [[ -e "${path}" ]] || { printf 'missing Pantheon runtime input: %s\n' "${path}" >&2; exit 1; }
done
[[ ! -e "${RESULT_DIR}" ]] || { printf 'result directory exists: %s\n' "${RESULT_DIR}" >&2; exit 1; }
mkdir -p "${RESULT_DIR}"

set -a
# shellcheck disable=SC1090
source "${MIG_ENV}"
set +a
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"
readonly TORCH_LIB="/tmp/pantheon-venv/lib/python3.12/site-packages/torch/lib"
export CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}"
export LD_LIBRARY_PATH="${TORCH_LIB}:${NVPL_DIR}:${CUDSS_DIR}:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${PANTHEON_PYTHON}" "${ROOT_DIR}/baselines/pantheon/make_native_smoke_assets.py" \
  --output "${RESULT_DIR}/assets" \
  --proto-dir "${PANTHEON_SRC}/online/proto" \
  --mig-uuid "${JDG_MIG_BIG_UUID}"

timeout 30s taskset --cpu-list "${CPU}" "${RUNTIME}" \
  "${RESULT_DIR}/assets/model-repository" \
  "${RESULT_DIR}/assets/workload.pb" \
  "${RESULT_DIR}/runtime.log" >"${RESULT_DIR}/runtime.stdout" 2>"${RESULT_DIR}/runtime.stderr"

python3 "${ROOT_DIR}/baselines/pantheon/verify_native_smoke.py" \
  --log "${RESULT_DIR}/runtime.log" \
  --environment "${RESULT_DIR}/assets/environment.json" \
  --output "${RESULT_DIR}/verification.json"

git -C "${PANTHEON_SRC}" rev-parse HEAD >"${RESULT_DIR}/pantheon-commit.txt"
sha256sum "${RUNTIME}" "${RESULT_DIR}/runtime.log" \
  "${RESULT_DIR}/assets/environment.json" "${RESULT_DIR}/assets/workload.pb" \
  "${RESULT_DIR}/assets/model-repository/pantheon-smoke-cnn/model_files/"*.pth \
  "${RESULT_DIR}/verification.json" >"${RESULT_DIR}/SHA256SUMS"
printf 'Pantheon native scheduler positive control: %s\n' "${RESULT_DIR}"
