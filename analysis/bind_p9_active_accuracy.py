#!/usr/bin/env python3
"""Bind a current comparator row to the application accuracy gate.

The comparator launchers record raw evidence for MIG, MPS, XSched, and QUIET
in two slightly different schemas.  This adapter normalizes only those
schemas and delegates prediction decoding and all correctness checks to the
existing application-accuracy tools.  It never invents labels, timing, or
output bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _path_record(value: Any, label: str) -> tuple[Path, str | None]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{label} path record is missing")
    path = Path(value["path"]).resolve()
    declared = value.get("sha256")
    if declared is not None and (
        not isinstance(declared, str) or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if declared is not None and sha256(path) != declared:
        raise ValueError(f"{label} SHA-256 differs: {path}")
    return path, declared


def candidate_paths(evidence: Path) -> dict[str, Any]:
    """Extract the one candidate output and wall CSV from a raw row artifact."""
    value = json.loads(evidence.resolve().read_bytes())
    if not isinstance(value, dict):
        raise ValueError("active evidence must be a JSON object")

    if isinstance(value.get("results"), list):
        rows = value["results"]
        if len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("dependent active evidence must contain one result row")
        row = rows[0]
        system = row.get("system")
        request = row.get("request_trace")
        output = row.get("application_output_trace")
    else:
        row = value
        system = value.get("system")
        request = value.get("request_trace")
        output = value.get("application_output_trace")

    if system == "XSched (Thor port)":
        system = "XSched"
    if system not in {"NVIDIA MIG", "NVIDIA MPS", "XSched", "QUIET"}:
        raise ValueError(f"unsupported active system: {system!r}")
    output_path, output_sha = _path_record(output, f"{system} application output trace")

    if system == "XSched":
        pipeline_path = evidence.resolve().parent / "pipeline.csv"
        if not pipeline_path.is_file():
            raise ValueError(f"XSched pipeline CSV is missing: {pipeline_path}")
        pipeline_sha = sha256(pipeline_path)
    else:
        pipeline_path, pipeline_sha = _path_record(request, f"{system} pipeline CSV")

    return {
        "system": system,
        "evidence": str(evidence.resolve()),
        "evidence_sha256": sha256(evidence),
        "candidate_output_trace": str(output_path),
        "candidate_output_trace_sha256": output_sha or sha256(output_path),
        "candidate_pipeline_csv": str(pipeline_path),
        "candidate_pipeline_csv_sha256": pipeline_sha or sha256(pipeline_path),
    }


def bind(
    evidence: Path,
    output_dir: Path,
    *,
    request_manifest: Path,
    dataset: Path,
    class_map: Path,
    reference_trace: Path,
    reference_output_trace: Path,
    reference_pipeline_csv: Path,
    reference_engine: Path,
    candidate_engine: Path,
    workload: str,
    task: str,
    prediction_mode: str,
    deadline_us: float,
    warmup: int,
    minimum_accuracy: float,
    accuracy_tolerance: float,
    output_index: int = 0,
) -> dict[str, Any]:
    """Create and verify one row's byte-bound application gate."""
    checked = candidate_paths(evidence)
    for label, path in (
        ("request manifest", request_manifest),
        ("dataset manifest", dataset),
        ("class map", class_map),
        ("reference prediction trace", reference_trace),
        ("reference output trace", reference_output_trace),
        ("reference pipeline CSV", reference_pipeline_csv),
        ("reference engine", reference_engine),
        ("candidate engine", candidate_engine),
    ):
        if not path.resolve().is_file():
            raise ValueError(f"{label} is missing: {path.resolve()}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_trace = output_dir / "predictions.jsonl"
    gate_path = output_dir / "accuracy-gate.json"
    build_command = [
        sys.executable,
        str(ROOT / "analysis/build_application_prediction_trace.py"),
        "--output-trace", checked["candidate_output_trace"],
        "--pipeline-csv", checked["candidate_pipeline_csv"],
        "--request-manifest", str(request_manifest.resolve()),
        "--class-map", str(class_map.resolve()),
        "--warmup", str(warmup), "--deadline-us", str(deadline_us),
        "--output-index", str(output_index),
        "--prediction-mode", prediction_mode,
        "--require-input-binding", "--output", str(candidate_trace),
    ]
    subprocess.run(build_command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    verify_command = [
        sys.executable,
        str(ROOT / "analysis/verify_application_accuracy.py"),
        "--reference-trace", str(reference_trace.resolve()),
        "--candidate-trace", str(candidate_trace),
        "--dataset", str(dataset.resolve()),
        "--reference-engine", str(reference_engine.resolve()),
        "--candidate-engine", str(candidate_engine.resolve()),
        "--workload", workload, "--task", task,
        "--deadline-us", str(deadline_us),
        "--accuracy-tolerance", str(accuracy_tolerance),
        "--minimum-accuracy", str(minimum_accuracy),
        "--reference-output-trace", str(reference_output_trace.resolve()),
        "--candidate-output-trace", checked["candidate_output_trace"],
        "--output-trace-warmup", str(warmup),
        "--reference-pipeline-csv", str(reference_pipeline_csv.resolve()),
        "--candidate-pipeline-csv", checked["candidate_pipeline_csv"],
        "--pipeline-warmup", str(warmup),
        "--require-input-binding", "--require-output-traces",
        "--output", str(gate_path),
    ]
    subprocess.run(verify_command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    gate = json.loads(gate_path.read_bytes())
    if (
        gate.get("kind") != "p9-application-accuracy-gate"
        or gate.get("status") != "passed"
        or gate.get("application_input_binding_contract") != "passed"
        or gate.get("application_output_trace_contract") != "passed"
    ):
        raise ValueError("application accuracy verifier did not emit a passed bound gate")
    return {
        "system": checked["system"],
        "evidence": checked,
        "prediction_trace": {"path": str(candidate_trace), "sha256": sha256(candidate_trace)},
        "accuracy_gate": {"path": str(gate_path), "sha256": sha256(gate_path)},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--reference-output-trace", type=Path, required=True)
    parser.add_argument("--reference-pipeline-csv", type=Path, required=True)
    parser.add_argument("--reference-engine", type=Path, required=True)
    parser.add_argument("--candidate-engine", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--task", default="classification")
    parser.add_argument("--prediction-mode", choices=("argmax", "resnet10-detection"), default="argmax")
    parser.add_argument("--deadline-us", type=float, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--minimum-accuracy", type=float, default=0.80)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.0)
    parser.add_argument("--output-index", type=int, default=0)
    args = parser.parse_args(argv)
    result = bind(
        args.evidence, args.output_dir,
        request_manifest=args.request_manifest, dataset=args.dataset,
        class_map=args.class_map, reference_trace=args.reference_trace,
        reference_output_trace=args.reference_output_trace,
        reference_pipeline_csv=args.reference_pipeline_csv,
        reference_engine=args.reference_engine, candidate_engine=args.candidate_engine,
        workload=args.workload, task=args.task, prediction_mode=args.prediction_mode,
        deadline_us=args.deadline_us, warmup=args.warmup,
        minimum_accuracy=args.minimum_accuracy, accuracy_tolerance=args.accuracy_tolerance,
        output_index=args.output_index,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
