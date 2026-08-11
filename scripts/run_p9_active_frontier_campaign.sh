#!/usr/bin/env bash
set -euo pipefail

# Fast, replay-verified exploratory loop for the executable matrix: NVIDIA MPS,
# XSched (Thor port), and QUIET. Only MPS and QUIET are currently eligible for
# a numeric frontier; XSched remains a gated candidate. This script does not
# claim formal SLO or thermal evidence.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/p9-active-frontier-$(date -u +%Y%m%dT%H%M%SZ)}"
DEADLINE_LOCK="${DEADLINE_LOCK:-}"
QUIET_PLAN="${QUIET_PLAN:-}"
WORKLOAD="${WORKLOAD:-resnet-control}"
COMMON_WORKLOAD_CONTRACT="${COMMON_WORKLOAD_CONTRACT:-}"
PRODUCER_INPUT_TRACE="${PRODUCER_INPUT_TRACE:-}"
PRODUCER_ENGINE="${PRODUCER_ENGINE:-}"
CONSUMER_ENGINE="${CONSUMER_ENGINE:-}"
APPLICATION_ACCURACY_GATE="${APPLICATION_ACCURACY_GATE:-}"
APPLICATION_ACCURACY_REFERENCE_TRACE="${APPLICATION_ACCURACY_REFERENCE_TRACE:-}"
APPLICATION_ACCURACY_REFERENCE_OUTPUT_TRACE="${APPLICATION_ACCURACY_REFERENCE_OUTPUT_TRACE:-}"
APPLICATION_ACCURACY_REFERENCE_PIPELINE_CSV="${APPLICATION_ACCURACY_REFERENCE_PIPELINE_CSV:-}"
APPLICATION_ACCURACY_REFERENCE_ENGINE="${APPLICATION_ACCURACY_REFERENCE_ENGINE:-}"
APPLICATION_ACCURACY_CLASS_MAP="${APPLICATION_ACCURACY_CLASS_MAP:-}"
APPLICATION_ACCURACY_DEADLINE_US="${APPLICATION_ACCURACY_DEADLINE_US:-}"
APPLICATION_ACCURACY_MINIMUM="${APPLICATION_ACCURACY_MINIMUM:-0.80}"
APPLICATION_ACCURACY_TOLERANCE="${APPLICATION_ACCURACY_TOLERANCE:-0.0}"
REQUESTS="${REQUESTS:-100}"
WARMUP="${WARMUP:-10}"
BACKGROUND_PERIOD_MS="${BACKGROUND_PERIOD_MS:-4.0}"
SEQUENCE_INDICES="${SEQUENCE_INDICES:-0,1,2}"

usage() {
  cat <<'EOF'
Usage: run_p9_active_frontier_campaign.sh

Exploratory matrix: NVIDIA MPS, XSched (Thor port), and QUIET.
Numeric frontier eligibility: NVIDIA MPS and QUIET only.

Required environment:
  DEADLINE_LOCK=/path/to/current-deadline-lock.json
  QUIET_PLAN=/path/to/selected-quiet-plan.json

Optional: RESULT_ROOT, WORKLOAD (resnet-control|resnet-detection-head|resnet50-classification|whisper-projection),
COMMON_WORKLOAD_CONTRACT, PRODUCER_INPUT_TRACE, PRODUCER_ENGINE, CONSUMER_ENGINE, REQUESTS, WARMUP, BACKGROUND_PERIOD_MS,
APPLICATION_ACCURACY_GATE, APPLICATION_ACCURACY_REFERENCE_TRACE, APPLICATION_ACCURACY_REFERENCE_OUTPUT_TRACE,
APPLICATION_ACCURACY_REFERENCE_PIPELINE_CSV, APPLICATION_ACCURACY_REFERENCE_ENGINE, APPLICATION_ACCURACY_CLASS_MAP,
APPLICATION_ACCURACY_DEADLINE_US, APPLICATION_ACCURACY_MINIMUM, APPLICATION_ACCURACY_TOLERANCE,
SEQUENCE_INDICES (default 0,1,2).

The output is exploratory until application accuracy, independent sessions,
and thermal normalization are attached. It never runs BOER or relabels an
internal policy as a published system.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ -z "${DEADLINE_LOCK}" || -z "${QUIET_PLAN}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "${DEADLINE_LOCK}" || ! -f "${QUIET_PLAN}" ]]; then
  printf 'DEADLINE_LOCK and QUIET_PLAN must point to existing files\n' >&2
  exit 2
fi
lock_kind="$(jq -er '.kind' "${DEADLINE_LOCK}")"
case "${lock_kind}" in
  p9-dependent-pipeline-deadline-lock)
    python3 "${ROOT_DIR}/analysis/freeze_p9_pipeline_deadline.py" \
      --verify "${DEADLINE_LOCK}" >/dev/null
    ;;
  p9-common-placement-deadline-lock)
    python3 "${ROOT_DIR}/analysis/freeze_p9_common_placement_deadline.py" \
      --verify "${DEADLINE_LOCK}" >/dev/null
    ;;
  *)
    printf 'unsupported deadline lock kind: %s\n' "${lock_kind}" >&2
    exit 2
    ;;
esac
if [[ -z "${APPLICATION_ACCURACY_DEADLINE_US}" ]]; then
  APPLICATION_ACCURACY_DEADLINE_US="$(jq -er '.deadline_us' "${DEADLINE_LOCK}")"
fi
case "${WORKLOAD}" in
  resnet-control|resnet-detection-head|resnet50-classification|whisper-projection) ;;
  *) printf 'unsupported WORKLOAD: %s\n' "${WORKLOAD}" >&2; exit 2 ;;
esac
if [[ "${WORKLOAD}" == resnet-detection-head && -z "${CONSUMER_ENGINE}" ]]; then
  printf 'CONSUMER_ENGINE is required for resnet-detection-head\n' >&2
  exit 2
fi
if [[ "${WORKLOAD}" == resnet50-classification && -z "${CONSUMER_ENGINE}" ]]; then
  printf 'CONSUMER_ENGINE is required for resnet50-classification\n' >&2
  exit 2
fi
if [[ "${WORKLOAD}" == resnet-detection-head || "${WORKLOAD}" == resnet50-classification ]]; then
  if [[ -z "${PRODUCER_INPUT_TRACE}" ]]; then
    printf 'PRODUCER_INPUT_TRACE is required for learned ResNet workloads\n' >&2
    exit 2
  fi
  if [[ -z "${APPLICATION_ACCURACY_GATE}" ]]; then
    printf 'APPLICATION_ACCURACY_GATE is required for learned workloads\n' >&2
    exit 2
  fi
fi
if [[ -n "${COMMON_WORKLOAD_CONTRACT}" && ! -f "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  printf 'COMMON_WORKLOAD_CONTRACT does not exist\n' >&2
  exit 2
fi
if [[ -n "${CONSUMER_ENGINE}" && ! -f "${CONSUMER_ENGINE}" ]]; then
  printf 'CONSUMER_ENGINE does not exist\n' >&2
  exit 2
fi
if [[ -n "${PRODUCER_INPUT_TRACE}" && ! -f "${PRODUCER_INPUT_TRACE}" ]]; then
  printf 'PRODUCER_INPUT_TRACE does not exist\n' >&2
  exit 2
fi
if [[ -n "${PRODUCER_ENGINE}" && ! -f "${PRODUCER_ENGINE}" ]]; then
  printf 'PRODUCER_ENGINE does not exist\n' >&2
  exit 2
fi
if [[ -n "${APPLICATION_ACCURACY_GATE}" && ! -f "${APPLICATION_ACCURACY_GATE}" ]]; then
  printf 'APPLICATION_ACCURACY_GATE does not exist\n' >&2
  exit 2
fi
if [[ -n "${APPLICATION_ACCURACY_GATE}" ]]; then
  python3 - "${APPLICATION_ACCURACY_GATE}" "${WORKLOAD}" "${COMMON_WORKLOAD_CONTRACT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
workload = sys.argv[2]
contract_path = Path(sys.argv[3]).resolve() if sys.argv[3] else None
try:
    value = json.loads(path.read_bytes())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"application accuracy gate is unreadable: {error}")
if (
    value.get("kind") != "p9-application-accuracy-gate"
    or value.get("status") != "passed"
    or value.get("numeric_comparison_allowed") is not True
    or value.get("application_input_binding_contract") != "passed"
    or value.get("workload") != workload
):
    raise SystemExit("application accuracy gate is not a passed bound gate for this workload")
if contract_path is not None:
    try:
        contract = json.loads(contract_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"common workload contract is unreadable: {error}")
    if contract.get("workload_id") != workload:
        raise SystemExit("common workload workload_id differs from requested workload")
    gate_dataset = Path(value.get("dataset_manifest_path", "")).resolve()
    contract_dataset = Path(contract.get("dataset_manifest_path", "")).resolve()
    if (
        gate_dataset != contract_dataset
        or value.get("dataset_manifest_sha256") != contract.get("dataset_manifest_sha256")
    ):
        raise SystemExit("application accuracy gate is bound to a different dataset manifest")
minimum = value.get("minimum_accuracy")
reference = value.get("reference_accuracy")
candidate = value.get("candidate_accuracy")
if (
    not all(isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in (minimum, reference, candidate))
    or float(minimum) < 0.80
    or float(reference) < float(minimum)
    or float(candidate) < float(minimum)
):
    raise SystemExit("application accuracy gate does not meet the frozen accuracy floor")
PY
fi
if [[ "${WORKLOAD}" == resnet-detection-head || "${WORKLOAD}" == resnet50-classification ]]; then
  if [[ -z "${COMMON_WORKLOAD_CONTRACT}" ]]; then
    printf 'learned active accuracy binding requires COMMON_WORKLOAD_CONTRACT\n' >&2
    exit 2
  fi
  request_manifest_default="$(jq -er '.arrival_trace_path' "${COMMON_WORKLOAD_CONTRACT}")"
  reference_dir_default="$(dirname "${request_manifest_default}")"
  : "${APPLICATION_ACCURACY_REFERENCE_TRACE:=${reference_dir_default}/reference-predictions.jsonl}"
  : "${APPLICATION_ACCURACY_REFERENCE_OUTPUT_TRACE:=${reference_dir_default}/reference-output.bin}"
  : "${APPLICATION_ACCURACY_REFERENCE_PIPELINE_CSV:=${reference_dir_default}/reference.csv}"
  : "${APPLICATION_ACCURACY_REFERENCE_ENGINE:=$(dirname "${CONSUMER_ENGINE}")/resnet50-imagenette-unsplit.onnx}"
  : "${APPLICATION_ACCURACY_CLASS_MAP:=$(dirname "${CONSUMER_ENGINE}")/class-map.json}"
  accuracy_task=classification
  accuracy_prediction_mode=argmax
  if [[ "${WORKLOAD}" == resnet-detection-head ]]; then
    accuracy_task=object-detection
    accuracy_prediction_mode=resnet10-detection
  fi
  for path in "${APPLICATION_ACCURACY_REFERENCE_TRACE}" \
              "${APPLICATION_ACCURACY_REFERENCE_OUTPUT_TRACE}" \
              "${APPLICATION_ACCURACY_REFERENCE_PIPELINE_CSV}" \
              "${APPLICATION_ACCURACY_REFERENCE_ENGINE}" \
              "${APPLICATION_ACCURACY_CLASS_MAP}"; do
    [[ -f "${path}" ]] || {
      printf 'missing active accuracy reference input: %s\n' "${path}" >&2
      exit 2
    }
  done
fi
if [[ ! "${REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'REQUESTS must be a positive integer\n' >&2
  exit 2
fi
if [[ ! "${WARMUP}" =~ ^[0-9]+$ ]]; then
  printf 'WARMUP must be a nonnegative integer\n' >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"

bind_active_accuracy() {
  local output="$1"
  local evidence
  if [[ -f "${output}/summary.json" ]]; then
    evidence="${output}/summary.json"
  else
    evidence="${output}/verification.json"
  fi
  python3 "${ROOT_DIR}/analysis/bind_p9_active_accuracy.py" \
    --evidence "${evidence}" \
    --output-dir "${output}/application-accuracy" \
    --request-manifest "${request_manifest_default}" \
    --dataset "$(jq -er '.dataset_manifest_path' "${COMMON_WORKLOAD_CONTRACT}")" \
    --class-map "${APPLICATION_ACCURACY_CLASS_MAP}" \
    --reference-trace "${APPLICATION_ACCURACY_REFERENCE_TRACE}" \
    --reference-output-trace "${APPLICATION_ACCURACY_REFERENCE_OUTPUT_TRACE}" \
    --reference-pipeline-csv "${APPLICATION_ACCURACY_REFERENCE_PIPELINE_CSV}" \
    --reference-engine "${APPLICATION_ACCURACY_REFERENCE_ENGINE}" \
    --candidate-engine "${CONSUMER_ENGINE}" \
    --workload "${WORKLOAD}" --task "${accuracy_task}" \
    --prediction-mode "${accuracy_prediction_mode}" \
    --deadline-us "${APPLICATION_ACCURACY_DEADLINE_US}" \
    --warmup "${WARMUP}" \
    --minimum-accuracy "${APPLICATION_ACCURACY_MINIMUM}" \
    --accuracy-tolerance "${APPLICATION_ACCURACY_TOLERANCE}" \
    >/dev/null
}

bind_sequence_accuracy() {
  local output="$1"
  local system_dir
  for system_dir in "${output}"/*-nvidia-mps "${output}"/*-xsched "${output}"/*-quiet; do
    [[ -d "${system_dir}" ]] || continue
    bind_active_accuracy "${system_dir}"
  done
}

inputs=()
IFS=',' read -r -a indices <<<"${SEQUENCE_INDICES}"
if [[ "${#indices[@]}" -eq 0 ]]; then
  printf 'SEQUENCE_INDICES is empty\n' >&2
  exit 2
fi
for index in "${indices[@]}"; do
  if [[ ! "${index}" =~ ^[0-2]$ ]]; then
    printf 'SEQUENCE_INDICES must contain only 0, 1, or 2\n' >&2
    exit 2
  fi
  output="${RESULT_ROOT}/sequence-${index}"
  if [[ -f "${output}/summary.json" ]]; then
    if [[ "${WORKLOAD}" == resnet-detection-head || "${WORKLOAD}" == resnet50-classification ]]; then
      bind_sequence_accuracy "${output}"
    fi
    inputs+=("${output}/summary.json")
    continue
  fi
  args=(
    python3 "${ROOT_DIR}/scripts/run_p9_common_sota_williams.py"
    --repo "${ROOT_DIR}" --result-dir "${output}"
    --deadline-lock "${DEADLINE_LOCK}" --quiet-plan "${QUIET_PLAN}"
    --sequence-index "${index}" --requests "${REQUESTS}"
    --warmup "${WARMUP}"
    --background-period-ms "${BACKGROUND_PERIOD_MS}"
    --workload "${WORKLOAD}" --active-only
  )
  if [[ -n "${COMMON_WORKLOAD_CONTRACT}" ]]; then
    args+=(--common-workload-contract "${COMMON_WORKLOAD_CONTRACT}")
  fi
  if [[ -n "${CONSUMER_ENGINE}" ]]; then
    args+=(--consumer-engine "${CONSUMER_ENGINE}")
  fi
  if [[ -n "${PRODUCER_INPUT_TRACE}" ]]; then
    args+=(--producer-input-trace "${PRODUCER_INPUT_TRACE}")
  fi
  if [[ -n "${PRODUCER_ENGINE}" ]]; then
    args+=(--producer-engine "${PRODUCER_ENGINE}")
  fi
  "${args[@]}"
  if [[ "${WORKLOAD}" == resnet-detection-head || "${WORKLOAD}" == resnet50-classification ]]; then
    bind_sequence_accuracy "${output}"
  fi
  inputs+=("${output}/summary.json")
done

if [[ "${#inputs[@]}" -eq 3 ]]; then
  analyzer_args=(
    python3 "${ROOT_DIR}/analysis/summarize_p9_active_williams_repeats.py"
    --output "${RESULT_ROOT}/frontier.json"
  )
  for input in "${inputs[@]}"; do
    analyzer_args+=(--input "${input}")
  done
  "${analyzer_args[@]}"
else
  # A one-sequence smoke is useful for fast hardware feedback, but it must
  # never masquerade as the three-sequence session aggregate. Preserve the
  # complete sequence summaries and expose an explicit non-rankable artifact.
  python3 - "${RESULT_ROOT}/frontier.json" "${inputs[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
inputs = [Path(path).resolve() for path in sys.argv[2:]]
if not inputs:
    raise SystemExit("partial active frontier has no sequence inputs")
sequences = []
for path in inputs:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise SystemExit(f"sequence summary is not newline-complete: {path}")
    value = json.loads(raw)
    if (
        value.get("kind") != "p9-common-sota-williams-sequence"
        or value.get("proposed_system") != "QUIET"
        or value.get("active_only") is not True
        or value.get("active_exploratory_systems") != ["NVIDIA MPS", "XSched", "QUIET"]
        or value.get("numeric_frontier_systems") != ["NVIDIA MPS", "QUIET"]
    ):
        raise SystemExit(f"sequence is not the active QUIET matrix: {path}")
    results = value.get("results")
    evidence = value.get("inputs")
    if (
        not isinstance(results, list)
        or [row.get("system") for row in results] != value.get("execution_order")
        or set(row.get("system") for row in results) != {"NVIDIA MPS", "XSched", "QUIET"}
        or not isinstance(evidence, list)
        or len(evidence) != 3
    ):
        raise SystemExit(f"active sequence rows differ: {path}")
    for item in evidence:
        source = Path(item.get("path", "")).resolve()
        expected_sha = item.get("sha256")
        if not source.is_file() or not isinstance(expected_sha, str):
            raise SystemExit(f"active sequence evidence is missing: {source}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:
            raise SystemExit(f"active sequence evidence SHA differs: {source}")
    sequences.append({
        "sequence_index": value["sequence_index"],
        "execution_order": value["execution_order"],
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "requests_per_system": value["requests_per_system"],
        "workload": value["workload"],
        "results": value["results"],
        "inputs": value["inputs"],
    })
output.write_text(json.dumps({
    "schema_version": 1,
    "kind": "p9-active-williams-production-wall-smoke",
    "proposed_system": "QUIET",
    "systems": ["NVIDIA MPS", "XSched", "QUIET"],
    "numeric_frontier_systems": ["NVIDIA MPS", "QUIET"],
    "sequence_count": len(sequences),
    "sequences": sequences,
    "formal": False,
    "ranking_allowed": False,
    "scope": "single-sequence-exploratory-production-wall; no-session-aggregate",
    "claim_guard": "Run three sequences through summarize_p9_active_williams_repeats.py before reporting paired statistics.",
}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"sequence_count": len(sequences), "ranking_allowed": False}, indent=2))
PY
fi
printf 'QUIET active frontier (exploratory): %s\n' "${RESULT_ROOT}/frontier.json"
