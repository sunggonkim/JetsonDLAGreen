#!/usr/bin/env python3
"""Build an SLO-qualified goodput frontier from common Williams aggregates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_output_trace(path: Path) -> dict[str, Any]:
    module_path = Path(__file__).with_name("read_application_output_trace.py")
    spec = importlib.util.spec_from_file_location("p9_output_trace_reader_sota", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("application output trace parser is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse(path)


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def load_comparator_manifest() -> tuple[dict[str, Any], str]:
    """Load the repository comparator contract once at the trust boundary."""
    path = Path(__file__).resolve().parents[1] / "docs" / "p9-comparator-manifest.json"
    value = load(path)
    rows = value.get("rows")
    if value.get("proposed_system") != "QUIET" or not isinstance(rows, dict):
        raise ValueError("comparator manifest is malformed")
    return value, digest(path)


def _output_trace_bound(gate: dict[str, Any], key: str) -> bool:
    trace = gate.get(key)
    if not isinstance(trace, dict):
        return False
    path_value = trace.get("path")
    expected = trace.get("sha256")
    count = trace.get("record_count")
    if (
        not isinstance(path_value, str) or not path_value
        or not isinstance(expected, str) or len(expected) != 64
        or not isinstance(count, int) or isinstance(count, bool) or count <= 0
        or trace.get("capture_boundary") != "post-completion"
    ):
        return False
    try:
        path = Path(path_value).resolve()
        raw = path.read_bytes()
        parsed = _parse_output_trace(path)
    except (OSError, ValueError):
        return False
    return hashlib.sha256(raw).hexdigest() == expected and parsed.get("record_count") == count


def _pipeline_csv_bound(gate: dict[str, Any], key: str) -> bool:
    record = gate.get(key)
    if not isinstance(record, dict):
        return False
    path_value, expected = record.get("path"), record.get("sha256")
    if (
        not isinstance(path_value, str) or not path_value
        or not isinstance(expected, str) or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        return False
    try:
        return hashlib.sha256(Path(path_value).resolve().read_bytes()).hexdigest() == expected
    except OSError:
        return False


def application_accuracy_bound(
    value: dict[str, Any], *, require_output_traces: bool = False,
    gate_record: dict[str, Any] | None = None,
) -> bool:
    """Require a passed, byte-bound application gate for a numeric row."""
    record = gate_record if gate_record is not None else value.get("application_accuracy_gate")
    if not isinstance(record, dict):
        return False
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str) or len(expected) != 64:
        return False
    try:
        path = Path(path_value).resolve()
        raw = path.read_bytes()
        gate = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return False
    passed = (
        hashlib.sha256(raw).hexdigest() == expected
        and isinstance(gate, dict)
        and gate.get("kind") == "p9-application-accuracy-gate"
        and gate.get("status") == "passed"
        and gate.get("numeric_comparison_allowed") is True
        and gate.get("application_input_binding_required") is True
        and gate.get("application_input_binding_contract") == "passed"
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
        and _pipeline_csv_bound(gate, "reference_pipeline_csv")
        and _pipeline_csv_bound(gate, "candidate_pipeline_csv")
    )
    if not passed or not require_output_traces:
        return passed
    return (
        gate.get("application_output_trace_required") is True
        and gate.get("application_output_trace_contract") == "passed"
        and _output_trace_bound(gate, "reference_output_trace")
        and _output_trace_bound(gate, "candidate_output_trace")
    )


def _load_gate_binding(path: Path) -> dict[str, str]:
    """Read a gate once and retain its content address for the output bundle."""
    resolved = path.resolve()
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"accuracy gate is not newline-complete: {path}")
    try:
        gate = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"accuracy gate is invalid JSON: {path}") from error
    if not isinstance(gate, dict) or gate.get("kind") != "p9-application-accuracy-gate":
        raise ValueError(f"accuracy gate has the wrong kind: {path}")
    return {"path": str(resolved), "sha256": hashlib.sha256(raw).hexdigest()}


def summarize(
    paths: list[Path], *, require_output_traces: bool = False,
    accuracy_gates: dict[str, Path] | None = None,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one aggregate is required")
    manifest, manifest_sha256 = load_comparator_manifest()
    manifest_rows = manifest["rows"]
    external_gate_bindings = {
        system: _load_gate_binding(path)
        for system, path in (accuracy_gates or {}).items()
    }
    points: list[dict[str, Any]] = []
    offered: set[float] = set()
    workload: str | None = None
    deadline: float | None = None
    systems: set[str] | None = None
    for path in paths:
        value = load(path.resolve())
        if (
            value.get("kind") != "p9-common-sota-williams-aggregate"
            or value.get("proposed_system") != "QUIET"
            or value.get("deadline_mode") != "wall"
        ):
            raise ValueError(f"aggregate is not a production-wall QUIET result: {path}")
        if value.get("claim_status") == "diagnostic-only-plan-violation":
            raise ValueError(f"{path} is diagnostic-only and cannot enter a frontier")
        rate = number(value.get("background_offered_rps"), "offered rps")
        if rate in offered:
            raise ValueError("frontier has duplicate offered load")
        offered.add(rate)
        current_workload = value.get("workload")
        if not isinstance(current_workload, str) or not current_workload:
            raise ValueError("aggregate workload is missing")
        if workload is None:
            workload = current_workload
        elif workload != current_workload:
            raise ValueError("frontier workloads differ")
        current_deadline = number(value.get("deadline_lock", {}).get("deadline_us", value.get("deadline_us")), "deadline")
        if deadline is None:
            deadline = current_deadline
        elif not math.isclose(deadline, current_deadline, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("frontier deadlines differ")
        # Schema-v2+ aggregates expose the canonical paper view.  Fall back to
        # the legacy raw replay map for historical artifacts only.
        rows = value.get("headline_systems", value.get("systems"))
        if not isinstance(rows, dict) or not rows:
            raise ValueError("aggregate lacks system rows")
        current_systems = set(rows)
        systems = current_systems if systems is None else systems & current_systems
        for system, row in rows.items():
            if not isinstance(row, dict):
                raise ValueError("aggregate system row is malformed")
            # Functional-only rows (for example Pantheon before its common
            # accuracy adapter) remain visible in the aggregate but do not
            # fabricate a numeric point or enter the frontier.
            if "background_goodput_rps_mean" not in row:
                points.append({
                    "offered_rps": rate,
                    "system": system,
                    "numeric_comparison_allowed": False,
                    "slo_confidence_qualified": False,
                    "comparison_status": row.get("comparison_status", "functional-only"),
                })
                continue
            allowed = row.get("numeric_comparison_allowed")
            qualified = row.get("slo_confidence_qualified")
            if not isinstance(allowed, bool) or not isinstance(qualified, bool):
                raise ValueError("aggregate lacks comparability/SLO contract")
            contract = manifest_rows.get(system)
            if not isinstance(contract, dict):
                raise ValueError(f"system {system!r} is absent from comparator manifest")
            manifest_allowed = contract.get("numeric_comparison_allowed")
            if not isinstance(manifest_allowed, bool):
                raise ValueError(f"manifest comparison contract is invalid for {system}")
            # A run cannot promote a functional-only or topology-mismatched
            # comparator by editing its aggregate JSON.  The manifest is the
            # single publication-facing authority for numeric eligibility.
            if allowed and not manifest_allowed:
                allowed = False
                qualified = False
            accuracy_pending = False
            gate_record: dict[str, Any] | None = None
            if system in external_gate_bindings:
                gate_record = external_gate_bindings[system]
            elif system == "QUIET":
                candidate_gate = value.get("application_accuracy_gate")
                if isinstance(candidate_gate, dict):
                    gate_record = candidate_gate
            else:
                gate_map = value.get("application_accuracy_gates")
                if isinstance(gate_map, dict):
                    candidate_gate = gate_map.get(system)
                    if isinstance(candidate_gate, dict):
                        gate_record = candidate_gate
                if gate_record is None and isinstance(row.get("application_accuracy_gate"), dict):
                    gate_record = row["application_accuracy_gate"]
            if allowed and qualified and not application_accuracy_bound(
                value,
                require_output_traces=require_output_traces,
                gate_record=gate_record,
            ):
                allowed = False
                qualified = False
                accuracy_pending = True
            points.append({
                "offered_rps": rate,
                "system": system,
                "numeric_comparison_allowed": allowed,
                "slo_confidence_qualified": qualified,
                "comparison_status": (
                    "application-accuracy-gate-pending"
                    if accuracy_pending else row.get("comparison_status", "numeric-or-structural")
                ),
                "background_goodput_rps_mean": nonnegative(
                    row.get("background_goodput_rps_mean"), "goodput"
                ),
                "dmr_cp95_upper": nonnegative(
                    row.get("dmr_cp95_upper", 1e-12), "dmr cp95 upper"
                ),
                "pooled_p99_us": number(row.get("pooled_p99_us"), "p99"),
            })
    if systems is None or not systems:
        raise ValueError("no common system rows")
    frontier: dict[str, Any] = {}
    for system in sorted(systems):
        rows = [row for row in points if row["system"] == system]
        eligible = [
            row for row in rows
            if row["numeric_comparison_allowed"] and row["slo_confidence_qualified"]
        ]
        best = max(eligible, key=lambda row: row["offered_rps"], default=None)
        frontier[system] = {
            "numeric_comparison_allowed": rows[0]["numeric_comparison_allowed"],
            "comparison_status": rows[0].get("comparison_status", "numeric-or-structural"),
            "points": sorted(rows, key=lambda row: row["offered_rps"]),
            "max_slo_qualified_offered_rps": (
                best["offered_rps"] if best is not None else None
            ),
            "max_slo_qualified_goodput_rps": (
                best["background_goodput_rps_mean"] if best is not None else None
            ),
        }
    numeric_frontier_systems = [
        name for name in manifest.get("paper_table_policy", {}).get(
            "numeric_frontier_order", []
        ) if name in frontier
    ]
    exploratory_systems = sorted(set(frontier) - set(numeric_frontier_systems))
    return {
        "schema_version": 1,
        "kind": "p9-common-sota-goodput-frontier",
        "proposed_system": "QUIET",
        "workload": workload,
        "deadline_mode": "wall",
        "deadline_us": deadline,
        "offered_loads_rps": sorted(offered),
        "systems": frontier,
        "numeric_frontier_systems": numeric_frontier_systems,
        "exploratory_systems": exploratory_systems,
        "ranking_allowed": bool(
            numeric_frontier_systems
            and all(
                frontier[name]["max_slo_qualified_offered_rps"] is not None
                for name in numeric_frontier_systems
            )
        ),
        "application_accuracy_gates": external_gate_bindings,
        "inputs": [
            {"path": str(path.resolve()), "sha256": digest(path)} for path in paths
        ],
        "comparator_manifest": {
            "path": str((Path(__file__).resolve().parents[1] / "docs" / "p9-comparator-manifest.json")),
            "sha256": manifest_sha256,
        },
        "claim_guard": (
            "Only numeric_comparison_allowed and CP95 SLO-qualified points enter "
            "the maximum-goodput frontier; structural controls remain visible but excluded."
        ),
        "application_output_traces_required": require_output_traces,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-output-traces", action="store_true",
        help="require raw post-completion traces in the QUIET accuracy gate",
    )
    parser.add_argument(
        "--accuracy-gate", action="append", default=[], metavar="SYSTEM=PATH",
        help="bind a passed application-accuracy gate to a numeric system row",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    accuracy_gates: dict[str, Path] = {}
    for spec in args.accuracy_gate:
        if "=" not in spec:
            raise ValueError("--accuracy-gate expects SYSTEM=PATH")
        system, raw_path = spec.split("=", 1)
        if not system or not raw_path or system in accuracy_gates:
            raise ValueError("--accuracy-gate contains a duplicate or empty system/path")
        accuracy_gates[system] = Path(raw_path)
    result = summarize(
        args.input,
        require_output_traces=args.require_output_traces,
        accuracy_gates=accuracy_gates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
