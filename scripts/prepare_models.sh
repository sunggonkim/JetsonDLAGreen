#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT_DIR}/models/cache}"
readonly ENGINE_ROOT="${ENGINE_ROOT:-${ROOT_DIR}/models/engines}"
readonly ENGINE_TAG="${ENGINE_TAG:-default}"
readonly ENGINE_DIR="${ENGINE_ROOT}/${ENGINE_TAG}"
readonly TRTEXEC="${TRTEXEC:-trtexec}"
readonly RESNET_SOURCE="/usr/src/jetson_multimedia_api/data/Model/resnet10/resnet10_dynamic_batch.onnx"
readonly BUILD_MODELS="${BUILD_MODELS:-resnet10-detection,distilbert-sst2,whisper-tiny-encoder}"

mkdir -p "${CACHE_DIR}" "${ENGINE_DIR}"

verify() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'checksum mismatch for %s: expected %s, got %s\n' \
      "${path}" "${expected}" "${actual}" >&2
    return 1
  fi
}

download() {
  local url="$1"
  local expected="$2"
  local path="$3"
  if [[ -f "${path}" ]] && verify "${expected}" "${path}"; then
    return
  fi
  local temporary="${path}.part"
  curl --fail --location --retry 5 --silent --show-error \
    "${url}" --output "${temporary}"
  verify "${expected}" "${temporary}"
  mv "${temporary}" "${path}"
}

download_archive_member() {
  local url="$1"
  local archive_expected="$2"
  local member="$3"
  local expected="$4"
  local path="$5"
  if [[ -f "${path}" ]] && verify "${expected}" "${path}"; then
    return
  fi
  local archive="${path}.tar.gz.part"
  local temporary="${path}.part"
  curl --fail --location --retry 5 --silent --show-error \
    "${url}" --output "${archive}"
  verify "${archive_expected}" "${archive}"
  tar -xOzf "${archive}" "${member}" >"${temporary}"
  verify "${expected}" "${temporary}"
  mv "${temporary}" "${path}"
  rm "${archive}"
}

if [[ ! -f "${CACHE_DIR}/resnet10-detection.onnx" ]]; then
  cp --reflink=auto "${RESNET_SOURCE}" "${CACHE_DIR}/resnet10-detection.onnx"
fi
verify "48c26511600dd95a0b9c15282360ea7678452f8016d287b0680d8ee00d2086dc" \
  "${CACHE_DIR}/resnet10-detection.onnx"
download \
  'https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx?download=true' \
  '252cf7048af94a1599019fef35961b2bd3d6db13df0b0a4b032b92baeae31939' \
  "${CACHE_DIR}/distilbert-sst2.onnx"
download \
  'https://huggingface.co/KitsuMate/whisper-tiny-onnx/resolve/main/onnx/encoder_model.onnx?download=true' \
  '6642befb640f950d4a8cbbd17834d59e7e75f575b81ccf213e06b050623ab1dd' \
  "${CACHE_DIR}/whisper-tiny-encoder.onnx"
download_archive_member \
  'https://download.onnxruntime.ai/onnx/models/resnet50.tar.gz' \
  'd9170b1239e33fbba9c2e3ae1022c8916c1f9df14fe3ce4d833b3d7f247e9449' \
  'resnet50/model.onnx' \
  '78eecdb9354e71364b9df6f3b5824ecc48710938d5b4ea23724b9a2e9edbc4a6' \
  "${CACHE_DIR}/resnet50-v2.onnx"

if [[ -f /tmp/jdg-mps-1g/mig.env && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Build on the critical partition so tactic selection matches its SM budget.
  # shellcheck disable=SC1091
  source /tmp/jdg-mps-1g/mig.env
  export CUDA_VISIBLE_DEVICES="${JDG_MIG_BIG_UUID}"
fi
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

build_engine() {
  local name="$1"
  local shapes="$2"
  local onnx="${CACHE_DIR}/${name}.onnx"
  local engine="${ENGINE_DIR}/${name}.engine"
  local log="${ENGINE_DIR}/${name}.build.log"
  if [[ -s "${engine}" && "${FORCE_REBUILD:-0}" != "1" ]]; then
    return
  fi
  "${TRTEXEC}" \
    --onnx="${onnx}" \
    --saveEngine="${engine}" \
    --minShapes="${shapes}" \
    --optShapes="${shapes}" \
    --maxShapes="${shapes}" \
    --fp16 \
    --builderOptimizationLevel=5 \
    --memPoolSize=workspace:2048 \
    --profilingVerbosity=detailed \
    --skipInference >"${log}" 2>&1
  grep -q 'PASSED TensorRT.trtexec' "${log}"
}

build_static_engine() {
  local name="$1"
  local onnx="${CACHE_DIR}/${name}.onnx"
  local engine="${ENGINE_DIR}/${name}.engine"
  local log="${ENGINE_DIR}/${name}.build.log"
  if [[ -s "${engine}" && "${FORCE_REBUILD:-0}" != "1" ]]; then
    return
  fi
  "${TRTEXEC}" \
    --onnx="${onnx}" \
    --saveEngine="${engine}" \
    --fp16 \
    --builderOptimizationLevel=5 \
    --memPoolSize=workspace:2048 \
    --profilingVerbosity=detailed \
    --skipInference >"${log}" 2>&1
  grep -q 'PASSED TensorRT.trtexec' "${log}"
}

IFS=',' read -r -a selected_models <<<"${BUILD_MODELS}"
for model in "${selected_models[@]}"; do
  case "${model}" in
    resnet50-v2)
      build_static_engine "${model}"
      ;;
    resnet10-detection)
      build_engine "${model}" 'data:1x3x368x640'
      ;;
    distilbert-sst2)
      build_engine "${model}" 'input_ids:1x128,attention_mask:1x128'
      ;;
    whisper-tiny-encoder)
      build_engine "${model}" 'input_features:1x80x3000'
      ;;
    *)
      printf 'unknown model in BUILD_MODELS: %s\n' "${model}" >&2
      exit 1
      ;;
  esac
done

sha256sum "${CACHE_DIR}"/*.onnx "${ENGINE_DIR}"/*.engine \
  >"${ENGINE_DIR}/SHA256SUMS"
{
  printf 'engine_tag=%s\n' "${ENGINE_TAG}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'mps_active_thread_percentage=%s\n' \
    "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-unset}"
  nvidia-smi -L
} >"${ENGINE_DIR}/build-environment.txt"
printf 'TensorRT engines ready in %s\n' "${ENGINE_DIR}"
