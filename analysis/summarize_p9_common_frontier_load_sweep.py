#!/usr/bin/env python3
"""Build a strict same-deadline production-wall load frontier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta

try:
    from .p9_frontier_evidence import validate_correctness
except ImportError:  # direct CLI execution
    from p9_frontier_evidence import validate_correctness


CORE_SYSTEMS = {"QUIET", "NVIDIA MPS", "XSched"}
OPTIONAL_SYSTEMS = {"Static full gating"}
SYSTEMS = CORE_SYSTEMS | OPTIONAL_SYSTEMS
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DMR_TARGET = 0.0005
PRODUCTION_WALL_DEFINITION = (
    "arrival-to-consumer-completion-excludes-correctness-validation"
)
CORRECTNESS_PLACEMENT = "post-completion"
COMMON_WORKLOAD_KEYS = (
    "schema_version", "workload_id", "topology", "placement", "input_tensor",
    "payload_bytes", "arrival_trace_path", "arrival_trace_sha256",
    "dataset_manifest_path", "dataset_manifest_sha256", "contract_path",
    "contract_sha256",
)


def _cp95_upper(misses: int, requests: int) -> float:
    """One-sided exact 95% binomial upper bound for the request DMR."""
    if misses == requests:
        return 1.0
    return float(beta.ppf(0.95, misses + 1, requests - misses))


def _parse_output_trace(path: Path) -> dict[str, Any]:
    module_path = Path(__file__).with_name("read_application_output_trace.py")
    spec = importlib.util.spec_from_file_location("p9_output_trace_reader_frontier", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("application output trace parser is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse(path)


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def _has_bound_application_accuracy(
    value: dict[str, Any], common_workload: dict[str, Any],
    *, require_output_traces: bool = False,
) -> bool:
    record = value.get("application_accuracy_gate")
    if not isinstance(record, dict):
        return False
    path_value = record.get("path")
    expected = record.get("sha256")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        return False
    path = Path(path_value).resolve()
    try:
        raw = path.read_bytes()
        gate = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return False
    pipeline_bound = True
    for prefix in ("reference", "candidate"):
        record = gate.get(f"{prefix}_pipeline_csv") if isinstance(gate, dict) else None
        if not isinstance(record, dict):
            pipeline_bound = False
            break
        csv_path, csv_sha = record.get("path"), record.get("sha256")
        if not isinstance(csv_path, str) or not csv_path or not isinstance(csv_sha, str) or len(csv_sha) != 64:
            pipeline_bound = False
            break
        try:
            pipeline_bound = hashlib.sha256(Path(csv_path).resolve().read_bytes()).hexdigest() == csv_sha
        except OSError:
            pipeline_bound = False
        if not pipeline_bound:
            break
    dataset_path = gate.get("dataset_manifest_path") if isinstance(gate, dict) else None
    dataset_sha = gate.get("dataset_manifest_sha256") if isinstance(gate, dict) else None
    expected_dataset_path = Path(common_workload["dataset_manifest_path"]).resolve()
    dataset_bound = (
        isinstance(dataset_path, str)
        and Path(dataset_path).resolve() == expected_dataset_path
        and isinstance(dataset_sha, str)
        and dataset_sha == common_workload["dataset_manifest_sha256"]
    )
    return (
        hashlib.sha256(raw).hexdigest() == expected
        and isinstance(gate, dict)
        and gate.get("kind") == "p9-application-accuracy-gate"
        and gate.get("status") == "passed"
        and gate.get("numeric_comparison_allowed") is True
        and gate.get("application_input_binding_required") is True
        and gate.get("application_input_binding_contract") == "passed"
        and gate.get("workload") == common_workload["workload_id"]
        and dataset_bound
        and isinstance(gate.get("minimum_accuracy"), (int, float))
        and not isinstance(gate.get("minimum_accuracy"), bool)
        and math.isfinite(float(gate["minimum_accuracy"]))
        and 0.0 <= float(gate["minimum_accuracy"]) <= 1.0
        and isinstance(gate.get("reference_accuracy"), (int, float))
        and not isinstance(gate.get("reference_accuracy"), bool)
        and math.isfinite(float(gate["reference_accuracy"]))
        and float(gate["reference_accuracy"]) >= float(gate["minimum_accuracy"])
        and isinstance(gate.get("candidate_accuracy"), (int, float))
        and not isinstance(gate.get("candidate_accuracy"), bool)
        and math.isfinite(float(gate["candidate_accuracy"]))
        and float(gate["candidate_accuracy"]) >= float(gate["minimum_accuracy"])
        and pipeline_bound
        and (
            not require_output_traces
            or (
                gate.get("application_output_trace_required") is True
                and gate.get("application_output_trace_contract") == "passed"
                and _bound_output_trace(gate, "reference_output_trace")
                and _bound_output_trace(gate, "candidate_output_trace")
            )
        )
    )


def _bound_output_trace(gate: dict[str, Any], key: str) -> bool:
    trace = gate.get(key)
    if not isinstance(trace, dict):
        return False
    path_value = trace.get("path")
    expected = trace.get("sha256")
    count = trace.get("record_count")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected, str)
        or len(expected) != 64
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or trace.get("capture_boundary") != "post-completion"
    ):
        return False
    try:
        raw = Path(path_value).resolve().read_bytes()
    except OSError:
        return False
    if hashlib.sha256(raw).hexdigest() != expected:
        return False
    try:
        parsed = _parse_output_trace(Path(path_value))
    except (OSError, ValueError):
        return False
    return parsed.get("record_count") == count


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _validate_common_workload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(key not in value for key in COMMON_WORKLOAD_KEYS):
        raise ValueError("common workload contract is malformed")
    if value.get("schema_version") != 1:
        raise ValueError("common workload contract schema differs")
    for path_key, digest_key in (
        ("arrival_trace_path", "arrival_trace_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("contract_path", "contract_sha256"),
    ):
        path = Path(value[path_key]).resolve()
        digest = value[digest_key]
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"common workload evidence SHA differs: {path_key}")
    return dict(value)


def summarize(
    paths: Iterable[Path], *, require_application_accuracy: bool = False,
    require_output_traces: bool = False,
) -> dict[str, Any]:
    if require_output_traces and not require_application_accuracy:
        raise ValueError("--require-output-traces requires application accuracy")
    points: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    contract: tuple[Any, ...] | None = None
    common_workload: dict[str, Any] | None = None
    for path in paths:
        value, digest = _read(path)
        if (
            value.get("kind") != "p9-dependent-small-stress-smoke"
            or value.get("workload") != "resnet-control"
            or value.get("latency_contract") != "production-wall-arrival-to-completion"
            or value.get("deadline_mode") != "wall"
            or value.get("checksum_mode") != "inline"
        ):
            raise ValueError(f"{path} is outside the common load contract")
        if value.get("claim_status") == "diagnostic-only-plan-violation":
            raise ValueError(f"{path} is diagnostic-only and cannot enter a frontier")
        current_common = value.get("common_workload")
        if current_common is None:
            if require_application_accuracy:
                raise ValueError(
                    f"{path}: formal accuracy frontier requires common workload evidence"
                )
            if common_workload is not None:
                raise ValueError(f"{path}: common workload evidence is incomplete")
        else:
            checked_common = _validate_common_workload(current_common)
            if common_workload is None:
                common_workload = checked_common
            elif checked_common != common_workload:
                raise ValueError(f"{path}: common workload contract differs")
        period = _finite(value.get("background_period_ms"), "period")
        offered = _finite(value.get("background_offered_rps"), "offered load")
        if period <= 0 or not math.isclose(offered, 1000.0 / period, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(f"{path} period and offered load disagree")
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError(f"{path} must contain one result")
        row = rows[0]
        system = row.get("system")
        if system not in SYSTEMS:
            raise ValueError(f"{path} lacks an executable system row")
        if require_application_accuracy and not _has_bound_application_accuracy(
            value, common_workload, require_output_traces=require_output_traces
        ):
            raise ValueError(
                f"{path}: {system} lacks a passed byte-bound application accuracy gate"
            )
        correctness = validate_correctness(value, row, path)
        lock = value.get("deadline_lock")
        if not isinstance(lock, dict) or not isinstance(lock.get("sha256"), str):
            raise ValueError(f"{path} lacks deadline-lock provenance")
        current = (
            value.get("workload"), _finite(value.get("deadline_us"), "deadline"),
            value.get("iterations"), value.get("placement_variant"), value.get("checksum_mode"),
            lock["sha256"],
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"{path} differs from common workload/deadline contract")
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            raise ValueError(f"{path} has invalid requests")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError(f"{path} has invalid misses")
        key = (system, offered)
        if key in seen:
            raise ValueError(f"duplicate system/load point: {system} at {offered:g}")
        seen.add(key)
        points.append({
            "system": system, "offered_rps": offered, "period_ms": period,
            "requests": requests, "deadline_misses": misses, "dmr": misses / requests,
            "cp95_upper_dmr": _cp95_upper(misses, requests),
            "p99_us": _finite(row.get("wall_pipeline_p99_us", row.get("pipeline_p99_us")), "p99"),
            "background_goodput_rps": _finite(row.get("background_goodput_rps"), "goodput"),
            "slo_qualified": misses == 0,
            "cp95_slo_qualified": _cp95_upper(misses, requests) <= DEFAULT_DMR_TARGET,
            "deadline_lock_sha256": lock["sha256"],
            "correctness_evidence": correctness,
            "input": {"path": str(path.resolve()), "sha256": digest},
        })
    if contract is None:
        raise ValueError("empty sweep")
    loads = sorted({point["offered_rps"] for point in points})
    for load in loads:
        present = {point["system"] for point in points if point["offered_rps"] == load}
        if not CORE_SYSTEMS.issubset(present):
            raise ValueError(f"load {load:g} does not contain all systems")
    optional_present = {point["system"] for point in points} - CORE_SYSTEMS
    if optional_present and optional_present != OPTIONAL_SYSTEMS:
        raise ValueError("optional baselines must cover the complete load sweep")
    manifest = json.loads((ROOT / "docs/p9-comparator-manifest.json").read_text())
    paper_policy = manifest.get("paper_table_policy")
    if not isinstance(paper_policy, dict) or paper_policy.get("proposed_system") != "QUIET":
        raise ValueError("comparator manifest lacks the QUIET paper-table policy")
    numeric_frontier_systems = paper_policy.get("numeric_frontier_order")
    if (
        not isinstance(numeric_frontier_systems, list)
        or not numeric_frontier_systems
        or any(not isinstance(system, str) for system in numeric_frontier_systems)
    ):
        raise ValueError("comparator manifest numeric frontier order is invalid")
    numeric_frontier_systems = [
        system for system in numeric_frontier_systems
        if system in CORE_SYSTEMS | optional_present
    ]
    exploratory_systems = sorted(
        (CORE_SYSTEMS | optional_present) - set(numeric_frontier_systems)
    )
    frontier: dict[str, Any] = {}
    for system in sorted(CORE_SYSTEMS | optional_present):
        rows = sorted((point for point in points if point["system"] == system), key=lambda point: point["offered_rps"])
        manifest_row = manifest["rows"].get(
            system, manifest.get("same_slo_baselines", {}).get(system, {})
        )
        qualified = [
            row for row in rows
            if row["cp95_slo_qualified"] and manifest_row.get("numeric_comparison_allowed") is True
        ]
        best = max(qualified, key=lambda row: row["offered_rps"], default=None)
        frontier[system] = {
            "points": rows,
            "numeric_comparison_allowed": (
                manifest_row.get("numeric_comparison_allowed", False)
                and common_workload is not None
            ),
            "comparison_status": manifest_row.get("status", "unknown"),
            "max_slo_qualified_offered_rps": best["offered_rps"] if best else None,
            "max_slo_qualified_goodput_rps": best["background_goodput_rps"] if best else None,
        }
    return {
        "schema_version": 1, "kind": "p9-common-production-wall-load-frontier",
        "proposed_system": "QUIET", "workload": contract[0], "deadline_us": contract[1],
        "common_workload": common_workload,
        "placement_variant": contract[3], "offered_loads_rps": loads,
        "systems": sorted(CORE_SYSTEMS | optional_present),
        "numeric_frontier_systems": numeric_frontier_systems,
        "exploratory_systems": exploratory_systems,
        "frontier": frontier, "formal": False,
        "production_wall_definition": PRODUCTION_WALL_DEFINITION,
        "correctness_validation_placement": CORRECTNESS_PLACEMENT,
        "ranking_allowed": False,
        "scope": "exploratory-same-deadline-load-frontier; no-thermal-normalization",
        "application_accuracy_required": require_application_accuracy,
        "application_output_traces_required": require_output_traces,
        "dmr_target": DEFAULT_DMR_TARGET,
        "slo_rule": "one-sided exact CP95 upper DMR <= 0.0005; frontier remains exploratory until formal gates pass",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-application-accuracy",
        action="store_true",
        help="reject any point without a passed byte-bound task-accuracy gate",
    )
    parser.add_argument(
        "--require-output-traces",
        action="store_true",
        help="with formal accuracy, require raw post-completion output traces",
    )
    args = parser.parse_args(argv)
    value = summarize(
        args.input, require_application_accuracy=args.require_application_accuracy,
        require_output_traces=args.require_output_traces,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
