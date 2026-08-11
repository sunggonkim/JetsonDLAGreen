#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-r39}"
readonly MIG_ENV="${MIG_ENV:-/tmp/jdg-mps-1g/mig.env}"
readonly SAMPLE_DIR="${SAMPLE_DIR:-${ROOT_DIR}/results/p9-real-whisper-asr-lex12-20260811}"
readonly RESULT_DIR="${RESULT_DIR:-${SAMPLE_DIR}}"
readonly CONTROL_CPU="${CONTROL_CPU:-13}"
readonly WARMUP="${WARMUP:-2}"
readonly ITERATIONS="${ITERATIONS:-10}"
readonly MAX_TOKENS="${MAX_TOKENS:-128}"
readonly DEADLINE_US="${DEADLINE_US:-1000000}"

for path in \
  "${BUILD_DIR}/jdg-mig-whisper-asr" "${MIG_ENV}" \
  "${SAMPLE_DIR}/inputs.bin" "${SAMPLE_DIR}/requests.jsonl" \
  "${ROOT_DIR}/models/engines/mig-1g-q100/whisper-tiny-encoder-fp32.engine" \
  "${ROOT_DIR}/models/engines/mig-2g-q100/whisper-tiny-decoder-initial-4-fp32.engine" \
  "${ROOT_DIR}/models/engines/mig-2g-q100/whisper-tiny-decoder-with-past-fp32.engine" \
  "${ROOT_DIR}/models/cache/whisper-tiny-tokenizer.json"; do
  [[ -f "${path}" ]] || { echo "missing required artifact: ${path}" >&2; exit 1; }
done

# shellcheck disable=SC1090
source "${MIG_ENV}"
: "${JDG_MIG_SMALL_UUID:?missing JDG_MIG_SMALL_UUID}"
: "${JDG_MIG_BIG_UUID:?missing JDG_MIG_BIG_UUID}"
: "${JDG_MPS_PIPE_DIRECTORY:?missing JDG_MPS_PIPE_DIRECTORY}"
mkdir -p "${RESULT_DIR}"
cp "${MIG_ENV}" "${RESULT_DIR}/mig.env"

export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"
taskset --cpu-list "${CONTROL_CPU}" \
  "${BUILD_DIR}/jdg-mig-whisper-asr" \
  --encoder-engine "${ROOT_DIR}/models/engines/mig-1g-q100/whisper-tiny-encoder-fp32.engine" \
  --decoder-initial-engine "${ROOT_DIR}/models/engines/mig-2g-q100/whisper-tiny-decoder-initial-4-fp32.engine" \
  --decoder-with-past-engine "${ROOT_DIR}/models/engines/mig-2g-q100/whisper-tiny-decoder-with-past-fp32.engine" \
  --input-trace "${SAMPLE_DIR}/inputs.bin" \
  --output-trace "${RESULT_DIR}/asr-output.bin" \
  --trace-csv "${RESULT_DIR}/pipeline.csv" \
  --warmup "${WARMUP}" --iterations "${ITERATIONS}" \
  --max-tokens "${MAX_TOKENS}" --deadline-us "${DEADLINE_US}" \
  --producer "${JDG_MIG_SMALL_UUID}" --consumer "${JDG_MIG_BIG_UUID}" \
  --mps-pipe "${JDG_MPS_PIPE_DIRECTORY}" \
  >"${RESULT_DIR}/run.json"

python3 "${ROOT_DIR}/analysis/read_application_output_trace.py" \
  --trace "${RESULT_DIR}/asr-output.bin" \
  --output "${RESULT_DIR}/asr-output.json" >/dev/null
python3 "${ROOT_DIR}/analysis/decode_whisper_token_trace.py" \
  --trace "${RESULT_DIR}/asr-output.bin" \
  --tokenizer "${ROOT_DIR}/models/cache/whisper-tiny-tokenizer.json" \
  --output "${RESULT_DIR}/transcript-map.json" >/dev/null
python3 "${ROOT_DIR}/analysis/build_application_prediction_trace.py" \
  --output-trace "${RESULT_DIR}/asr-output.bin" \
  --pipeline-csv "${RESULT_DIR}/pipeline.csv" \
  --request-manifest "${SAMPLE_DIR}/requests.jsonl" \
  --prediction-mode asr --transcript-map "${RESULT_DIR}/transcript-map.json" \
  --warmup "${WARMUP}" --deadline-us "${DEADLINE_US}" \
  --asr-max-wer 0.20 --require-input-binding \
  --output "${RESULT_DIR}/application-trace.jsonl" >/dev/null

sha256sum \
  "${BUILD_DIR}/jdg-mig-whisper-asr" \
  "${ROOT_DIR}/benchmarks/mig_whisper_asr.cpp" \
  "${ROOT_DIR}/analysis/read_application_output_trace.py" \
  "${ROOT_DIR}/analysis/decode_whisper_token_trace.py" \
  "${ROOT_DIR}/models/engines/mig-1g-q100/whisper-tiny-encoder-fp32.engine" \
  "${ROOT_DIR}/models/engines/mig-2g-q100/whisper-tiny-decoder-initial-4-fp32.engine" \
  "${ROOT_DIR}/models/engines/mig-2g-q100/whisper-tiny-decoder-with-past-fp32.engine" \
  "${ROOT_DIR}/models/cache/whisper-tiny-tokenizer.json" \
  "${SAMPLE_DIR}/inputs.bin" "${SAMPLE_DIR}/requests.jsonl" \
  >"${RESULT_DIR}/SHA256SUMS"
printf 'Real Whisper ASR split: %s\n' "${RESULT_DIR}"
