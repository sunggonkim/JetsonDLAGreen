#!/usr/bin/env bash
set -euo pipefail

# Prepare the real Whisper ASR split used by the application gate.  The
# decoder engines are deliberately FP32: the FP16 TensorRT plan changes greedy
# token choices on the frozen labelled LibriSpeech subset.  Engine files are
# platform artifacts; the build logs and SHA256SUMS are retained with them.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT_DIR}/models/cache}"
readonly ENGINE_ROOT="${ENGINE_ROOT:-${ROOT_DIR}/models/engines}"
readonly ENCODER_ENGINE_DIR="${ENCODER_ENGINE_DIR:-${ENGINE_ROOT}/mig-1g-q100}"
readonly DECODER_ENGINE_DIR="${DECODER_ENGINE_DIR:-${ENGINE_ROOT}/mig-2g-q100}"
readonly TRTEXEC="${TRTEXEC:-trtexec}"

mkdir -p "${CACHE_DIR}" "${ENCODER_ENGINE_DIR}" "${DECODER_ENGINE_DIR}"

verify() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    printf 'checksum mismatch for %s: expected %s, got %s\n' \
      "${path}" "${expected}" "${actual}" >&2
    return 1
  }
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

download \
  'https://huggingface.co/KitsuMate/whisper-tiny-onnx/resolve/main/onnx/encoder_model.onnx?download=true' \
  '6642befb640f950d4a8cbbd17834d59e7e75f575b81ccf213e06b050623ab1dd' \
  "${CACHE_DIR}/whisper-tiny-encoder.onnx"
download \
  'https://huggingface.co/KitsuMate/whisper-tiny-onnx/resolve/main/onnx/decoder_model.onnx?download=true' \
  'ab79e3f2a9a3d98f159f853a3172120a38af7eb5f7863d706aa7d39c228f009e' \
  "${CACHE_DIR}/whisper-tiny-decoder.onnx"
download \
  'https://huggingface.co/KitsuMate/whisper-tiny-onnx/resolve/main/onnx/decoder_with_past_model.onnx?download=true' \
  '0485135066eb1d36dcb04dbabd0cc1141c7cd8c442217abd0798d55fe2bc6bed' \
  "${CACHE_DIR}/whisper-tiny-decoder-with-past.onnx"
download \
  'https://huggingface.co/KitsuMate/whisper-tiny-onnx/resolve/main/onnx/mel.onnx?download=true' \
  '0fb448b82bde665099d8532502dd1d7f95751f3afcd3760e7b30a94ca0bddebf' \
  "${CACHE_DIR}/whisper-tiny-mel.onnx"
download \
  'https://huggingface.co/KitsuMate/whisper-tiny-onnx/resolve/main/tokenizer.json?download=true' \
  '27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566' \
  "${CACHE_DIR}/whisper-tiny-tokenizer.json"

export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

build() {
  local device="$1"
  local name="$2"
  local onnx="$3"
  local output="$4"
  shift 4
  local log="${output}.build.log"
  if [[ -s "${output}" && "${FORCE_REBUILD:-0}" != "1" ]]; then
    return
  fi
  env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY \
    CUDA_VISIBLE_DEVICES="${device}" "${TRTEXEC}" \
    --onnx="${onnx}" --saveEngine="${output}" "$@" \
    --builderOptimizationLevel=5 --memPoolSize=workspace:4096 \
    --profilingVerbosity=detailed --skipInference >"${log}" 2>&1
  grep -q 'PASSED TensorRT.trtexec' "${log}"
  printf '%s\n' "${name}" >>"${output}.components"
}

small_device="${JDG_MIG_SMALL_UUID:-${CUDA_VISIBLE_DEVICES:-}}"
big_device="${JDG_MIG_BIG_UUID:-${CUDA_VISIBLE_DEVICES:-}}"
[[ -n "${small_device}" && -n "${big_device}" ]] || {
  echo 'set JDG_MIG_SMALL_UUID and JDG_MIG_BIG_UUID (or CUDA_VISIBLE_DEVICES)' >&2
  exit 1
}

build "${small_device}" whisper-tiny-encoder \
  "${CACHE_DIR}/whisper-tiny-encoder.onnx" \
  "${ENCODER_ENGINE_DIR}/whisper-tiny-encoder-fp32.engine" \
  --minShapes=input_features:1x80x3000 \
  --optShapes=input_features:1x80x3000 \
  --maxShapes=input_features:1x80x3000

build "${big_device}" whisper-tiny-decoder-initial-4 \
  "${CACHE_DIR}/whisper-tiny-decoder.onnx" \
  "${DECODER_ENGINE_DIR}/whisper-tiny-decoder-initial-4-fp32.engine" \
  --minShapes=input_ids:1x4,encoder_hidden_states:1x1500x384 \
  --optShapes=input_ids:1x4,encoder_hidden_states:1x1500x384 \
  --maxShapes=input_ids:1x4,encoder_hidden_states:1x1500x384

past_shapes="past_key_values.0.decoder.key:1x6x%sx64,past_key_values.0.decoder.value:1x6x%sx64,past_key_values.0.encoder.key:1x6x1500x64,past_key_values.0.encoder.value:1x6x1500x64,past_key_values.1.decoder.key:1x6x%sx64,past_key_values.1.decoder.value:1x6x%sx64,past_key_values.1.encoder.key:1x6x1500x64,past_key_values.1.encoder.value:1x6x1500x64,past_key_values.2.decoder.key:1x6x%sx64,past_key_values.2.decoder.value:1x6x%sx64,past_key_values.2.encoder.key:1x6x1500x64,past_key_values.2.encoder.value:1x6x1500x64,past_key_values.3.decoder.key:1x6x%sx64,past_key_values.3.decoder.value:1x6x%sx64,past_key_values.3.encoder.key:1x6x1500x64,past_key_values.3.encoder.value:1x6x1500x64"
min_past="$(printf "${past_shapes}" 1 1 1 1 1 1 1 1)"
opt_past="$(printf "${past_shapes}" 32 32 32 32 32 32 32 32)"
max_past="$(printf "${past_shapes}" 224 224 224 224 224 224 224 224)"
build "${big_device}" whisper-tiny-decoder-with-past \
  "${CACHE_DIR}/whisper-tiny-decoder-with-past.onnx" \
  "${DECODER_ENGINE_DIR}/whisper-tiny-decoder-with-past-fp32.engine" \
  --minShapes="input_ids:1x1,${min_past}" \
  --optShapes="input_ids:1x1,${opt_past}" \
  --maxShapes="input_ids:1x1,${max_past}"

sha256sum "${CACHE_DIR}/whisper-tiny-encoder.onnx" \
  "${CACHE_DIR}/whisper-tiny-decoder.onnx" \
  "${CACHE_DIR}/whisper-tiny-decoder-with-past.onnx" \
  "${CACHE_DIR}/whisper-tiny-mel.onnx" \
  "${CACHE_DIR}/whisper-tiny-tokenizer.json" \
  "${ENCODER_ENGINE_DIR}/whisper-tiny-encoder-fp32.engine" \
  "${DECODER_ENGINE_DIR}/whisper-tiny-decoder-initial-4-fp32.engine" \
  "${DECODER_ENGINE_DIR}/whisper-tiny-decoder-with-past-fp32.engine" \
  >"${DECODER_ENGINE_DIR}/whisper-asr-fp32-SHA256SUMS"
printf 'Whisper ASR FP32 engines ready in %s and %s\n' \
  "${ENCODER_ENGINE_DIR}" "${DECODER_ENGINE_DIR}"
