#!/usr/bin/env bash
set -euo pipefail

# Run the conservative same-SLO baseline without presenting it as a proposed
# system. The underlying runner implements the stop-all pipeline scope; this
# wrapper adds an explicit reference-contract check and stable artifact.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR=""
DEADLINE_LOCK=""
REFERENCE_SUMMARY=""
WORKLOAD="resnet-control"
REQUESTS="100"
BACKGROUND_PERIOD_MS="4.0"
PLACEMENT_VARIANT="fixed-1g-producer-2g-consumer"
CONSUMER_ENGINE=""
COMMON_WORKLOAD_CONTRACT=""

usage() {
  cat <<'EOF'
Usage: run_p9_static_full_gate_smoke.sh --result-dir DIR --deadline-lock FILE

Runs the fixed all-worker stop baseline on the production-wall harness.
The output is deliberately non-rankable until the same-session formal analyzer
promotes it; this wrapper never labels the baseline as QUIET or as a SOTA system.

Required: --result-dir DIR, --deadline-lock FILE
Optional: --reference-summary FILE, --workload NAME, --requests N,
  --background-period-ms MS, --placement-variant NAME,
  --consumer-engine FILE, --common-workload-contract FILE.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-dir) RESULT_DIR="${2:?missing value}"; shift 2 ;;
    --deadline-lock) DEADLINE_LOCK="${2:?missing value}"; shift 2 ;;
    --reference-summary) REFERENCE_SUMMARY="${2:?missing value}"; shift 2 ;;
    --workload) WORKLOAD="${2:?missing value}"; shift 2 ;;
    --requests) REQUESTS="${2:?missing value}"; shift 2 ;;
    --background-period-ms) BACKGROUND_PERIOD_MS="${2:?missing value}"; shift 2 ;;
    --placement-variant) PLACEMENT_VARIANT="${2:?missing value}"; shift 2 ;;
    --consumer-engine) CONSUMER_ENGINE="${2:?missing value}"; shift 2 ;;
    --common-workload-contract) COMMON_WORKLOAD_CONTRACT="${2:?missing value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ -z "${RESULT_DIR}" || -z "${DEADLINE_LOCK}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "${DEADLINE_LOCK}" ]]; then
  printf 'deadline lock is not a regular file: %s\n' "${DEADLINE_LOCK}" >&2
  exit 2
fi
if [[ -n "${REFERENCE_SUMMARY}" && ! -f "${REFERENCE_SUMMARY}" ]]; then
  printf 'reference summary is not a regular file: %s\n' "${REFERENCE_SUMMARY}" >&2
  exit 2
fi
if [[ -n "${CONSUMER_ENGINE}" && ! -f "${CONSUMER_ENGINE}" ]]; then
  printf 'consumer engine is not a regular file: %s\n' "${CONSUMER_ENGINE}" >&2
  exit 2
fi
if [[ -n "${COMMON_WORKLOAD_CONTRACT}" && ! -f "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  printf 'common workload contract is not a regular file: %s\n' "${COMMON_WORKLOAD_CONTRACT}" >&2
  exit 2
fi
if [[ ! "${REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'requests must be a positive integer\n' >&2
  exit 2
fi

mkdir -p "${RESULT_DIR}"
args=(
  python3 "${ROOT_DIR}/scripts/run_p9_dependent_stress_smoke.py"
  --repo "${ROOT_DIR}"
  --result-dir "${RESULT_DIR}"
  --iterations "${REQUESTS}"
  --warmup 10
  --deadline-lock "${DEADLINE_LOCK}"
  --background-period-ms "${BACKGROUND_PERIOD_MS}"
  --scenario static-full-gate
  --placement-variant "${PLACEMENT_VARIANT}"
  --workload "${WORKLOAD}"
  --deadline-mode wall
  --checksum-mode inline
  --application-output-trace-dir "${RESULT_DIR}/application-outputs"
)
if [[ -n "${CONSUMER_ENGINE}" ]]; then
  args+=(--consumer-engine "${CONSUMER_ENGINE}")
fi
if [[ -n "${COMMON_WORKLOAD_CONTRACT}" ]]; then
  args+=(--common-workload-contract "${COMMON_WORKLOAD_CONTRACT}" --require-common-workload)
fi
"${args[@]}" >"${RESULT_DIR}/runner.stdout.json"

python3 - "${RESULT_DIR}" "${DEADLINE_LOCK}" "${REFERENCE_SUMMARY}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1]).resolve()
deadline_lock = Path(sys.argv[2]).resolve()
reference = Path(sys.argv[3]).resolve() if sys.argv[3] else None
summary_path = result_dir / "summary.json"
summary = json.loads(summary_path.read_bytes())
if summary.get("kind") != "p9-dependent-small-stress-smoke":
    raise SystemExit("unexpected runner summary kind")
if summary.get("execution_order") != ["Static full gating"]:
    raise SystemExit("static baseline did not run exactly one scenario")
rows = summary.get("results")
if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("system") != "Static full gating":
    raise SystemExit("static baseline result row is missing")
row = rows[0]
if row.get("latency_contract") != "production-wall-arrival-to-completion":
    raise SystemExit("static baseline is not production-wall measured")
if row.get("checksum_mode") != "inline" or row.get("correctness_validated") is not True:
    raise SystemExit("static baseline lacks inline correctness")
if not isinstance(row.get("request_trace"), dict):
    raise SystemExit("static baseline lacks request trace evidence")

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

artifact = {
    "schema_version": 1,
    "kind": "p9-static-full-gating-production-wall-smoke",
    "proposed_system": "QUIET",
    "baseline_system": "Static full gating",
    "numeric_comparison_allowed": False,
    "formal": False,
    "ranking_allowed": False,
    "claim_status": "same-slo-baseline-exploratory-non-rankable",
    "reason": "This is a conservative stop-all baseline; promotion requires the same session, accuracy, and thermal gates as QUIET.",
    "workload": summary.get("workload"),
    "placement_variant": summary.get("placement_variant"),
    "requests": row.get("pipeline_requests"),
    "deadline_misses": row.get("deadline_misses"),
    "wall_p99_us": row.get("wall_pipeline_p99_us"),
    "deadline_p99_us": row.get("pipeline_p99_us"),
    "background_goodput_rps": row.get("background_goodput_rps"),
    "deadline_lock": {"path": str(deadline_lock), "sha256": digest(deadline_lock)},
    "summary": {"path": str(summary_path), "sha256": digest(summary_path)},
    "request_trace": row["request_trace"],
    "application_output_trace": row.get("application_output_trace"),
}
if reference is not None:
    ref = json.loads(reference.read_bytes())
    if ref.get("workload") != summary.get("workload"):
        raise SystemExit("reference workload differs")
    if ref.get("placement_variant") != summary.get("placement_variant"):
        raise SystemExit("reference placement differs")
    if ref.get("deadline_lock", {}).get("sha256") != artifact["deadline_lock"]["sha256"]:
        raise SystemExit("reference deadline lock differs")
    artifact["reference_summary"] = {"path": str(reference), "sha256": digest(reference)}
artifact_path = result_dir / "static-full-gate.json"
artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "artifact": str(artifact_path),
    "system": artifact["baseline_system"],
    "ranking_allowed": False,
    "deadline_misses": artifact["deadline_misses"],
    "wall_p99_us": artifact["wall_p99_us"],
    "background_goodput_rps": artifact["background_goodput_rps"],
}, indent=2))
PY
