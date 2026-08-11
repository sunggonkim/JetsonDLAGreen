#!/usr/bin/env bash
set -euo pipefail

readonly PANTHEON_SRC="${PANTHEON_SRC:-/tmp/quiet-pantheon-1786270298}"
readonly PANTHEON_BUILD="${PANTHEON_BUILD:-${PANTHEON_SRC}/build-thor}"
readonly PANTHEON_PYTHON="${PANTHEON_PYTHON:-/tmp/pantheon-venv/bin/python}"
readonly NVPL_DIR="${NVPL_DIR:-/tmp/pantheon-nvpl}"
readonly CUDSS_DIR="${CUDSS_DIR:-/tmp/pantheon-venv/lib/python3.12/site-packages/nvidia/cu13/lib}"
readonly EXPECTED_COMMIT="1caa4321fe9f9902ffacb78978f11a32a7a62f64"

[[ "$(git -C "${PANTHEON_SRC}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  printf 'Pantheon upstream commit mismatch\n' >&2
  exit 1
}
for path in "${PANTHEON_PYTHON}" "${NVPL_DIR}/libnvpl_lapack_lp64_gomp.so.0" \
            "${NVPL_DIR}/libnvpl_blas_lp64_gomp.so.0" "${CUDSS_DIR}/libcudss.so.0"; do
  [[ -e "${path}" ]] || { printf 'missing Pantheon build dependency: %s\n' "${path}" >&2; exit 1; }
done

TORCH_CMAKE="$(LD_LIBRARY_PATH="${NVPL_DIR}:${CUDSS_DIR}:/usr/local/cuda/lib64" \
  "${PANTHEON_PYTHON}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
readonly TORCH_CMAKE
readonly SYSTEM_INCLUDE="${PANTHEON_BUILD}/system-include"
mkdir -p "${SYSTEM_INCLUDE}"
ln -sfn /usr/include/google "${SYSTEM_INCLUDE}/google"
ln -sfn /usr/include/fmt "${SYSTEM_INCLUDE}/fmt"

protoc --proto_path="${PANTHEON_SRC}/online/proto" \
  --cpp_out="${PANTHEON_SRC}/online/proto" \
  "${PANTHEON_SRC}/online/proto/model_config.proto" \
  "${PANTHEON_SRC}/online/proto/workload.proto"

cmake -S "${PANTHEON_SRC}/online/apps/runtime" -B "${PANTHEON_BUILD}" -G Ninja \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-I${SYSTEM_INCLUDE}" \
  -DCMAKE_EXE_LINKER_FLAGS="${NVPL_DIR}/libnvpl_lapack_lp64_gomp.so.0 ${NVPL_DIR}/libnvpl_blas_lp64_gomp.so.0 ${CUDSS_DIR}/libcudss.so.0 -Wl,-rpath,${NVPL_DIR} -Wl,-rpath,${CUDSS_DIR}"
cmake --build "${PANTHEON_BUILD}" -j2
printf 'Pantheon Thor runtime: %s\n' "${PANTHEON_BUILD}/runtime"
