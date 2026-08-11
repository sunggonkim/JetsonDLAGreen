#!/usr/bin/env python3
"""Replay the current three-system production-wall Williams repeats.

This analyzer is intentionally separate from the historical six-system
aggregator. It accepts the executable exploratory matrix (NVIDIA MPS, XSched,
and QUIET), verifies each source evidence byte hash, and reports descriptive
session statistics. Only NVIDIA MPS and QUIET are currently numeric-frontier
eligible; XSched remains a gated candidate until its common-workload accuracy,
thermal, and session gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta, t as student_t

try:
    from scripts.run_p9_common_sota_williams import active_williams_orders
except ModuleNotFoundError:  # direct execution from analysis/
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_p9_common_sota_williams import active_williams_orders


SYSTEMS = ("NVIDIA MPS", "XSched", "QUIET")
NUMERIC_FRONTIER_SYSTEMS = ("NVIDIA MPS", "QUIET")
ACTIVE_WORKLOADS = {
    "resnet-control",
    "resnet-detection-head",
    "resnet50-classification",
    "whisper-projection",
}
PRODUCTION_WALL_DEFINITION = (
    "arrival-to-consumer-completion-excludes-correctness-validation"
)


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _cp95(misses: int, requests: int) -> float:
    return 1.0 if misses == requests else float(beta.ppf(0.95, misses + 1, requests - misses))


def _t95(values: list[float]) -> dict[str, Any] | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    critical = float(student_t.ppf(0.975, len(values) - 1))
    half = critical * sd / math.sqrt(len(values))
    return {
        "method": "paired-session-t-interval",
        "confidence": 0.95,
        "unit": "Williams-sequence-pair",
        "n": len(values),
        "mean": mean,
        "sample_sd": sd,
        "t_critical": critical,
        "lower": mean - half,
        "upper": mean + half,
    }


def _application_accuracy_binding(
    system: str,
    evidence_path: Path,
    workload: str,
    requests: int,
    pipeline_path: Path | None,
    output_path: Path | None,
) -> dict[str, Any] | None:
    """Replay-bind an optional per-row application gate to raw output/timing."""
    gate_path = evidence_path.parent / "application-accuracy" / "accuracy-gate.json"
    if not gate_path.is_file():
        return None
    gate, gate_sha = _read(gate_path)
    if (
        gate.get("kind") != "p9-application-accuracy-gate"
        or gate.get("status") != "passed"
        or gate.get("numeric_comparison_allowed") is not True
        or gate.get("workload") != workload
        or gate.get("requests") != requests
        or gate.get("application_input_binding_contract") != "passed"
        or gate.get("application_output_trace_contract") != "passed"
    ):
        raise ValueError(f"{system} application accuracy contract differs")
    candidate_output = gate.get("candidate_output_trace")
    candidate_pipeline = gate.get("candidate_pipeline_csv")
    if (
        not isinstance(candidate_output, dict)
        or not isinstance(candidate_pipeline, dict)
        or not isinstance(candidate_output.get("path"), str)
        or not isinstance(candidate_output.get("sha256"), str)
        or not isinstance(candidate_pipeline.get("path"), str)
        or not isinstance(candidate_pipeline.get("sha256"), str)
    ):
        raise ValueError(f"{system} application accuracy evidence paths are missing")
    bound_output = Path(candidate_output["path"]).resolve()
    bound_pipeline = Path(candidate_pipeline["path"]).resolve()
    if (
        output_path is None
        or pipeline_path is None
        or bound_output != output_path.resolve()
        or bound_pipeline != pipeline_path.resolve()
        or not bound_output.is_file()
        or not bound_pipeline.is_file()
        or sha256(bound_output) != candidate_output["sha256"]
        or sha256(bound_pipeline) != candidate_pipeline["sha256"]
    ):
        raise ValueError(f"{system} application accuracy evidence is not bound to raw traces")
    return {
        "path": str(gate_path.resolve()),
        "sha256": gate_sha,
        "candidate_accuracy": _finite(gate.get("candidate_accuracy"), f"{system}.candidate_accuracy"),
        "accuracy_delta": _finite(gate.get("accuracy_delta"), f"{system}.accuracy_delta"),
        "candidate_output_trace_sha256": candidate_output["sha256"],
        "candidate_pipeline_sha256": candidate_pipeline["sha256"],
    }


def _source_row(
    system: str, evidence_path: Path, expected_sha: str, expected_lock_sha: str,
    workload: str,
) -> dict[str, Any]:
    value, actual_sha = _read(evidence_path)
    if actual_sha != expected_sha:
        raise ValueError(f"{system} evidence SHA differs: {evidence_path}")
    source_lock = value.get("deadline_lock")
    source_inputs = value.get("inputs")
    source_lock_sha = (
        source_lock.get("sha256") if isinstance(source_lock, dict)
        else source_inputs.get("deadline_lock_sha256")
        if isinstance(source_inputs, dict) else None
    )
    if source_lock_sha != expected_lock_sha:
        raise ValueError(f"{system} evidence is not bound to the sequence deadline lock")
    if system in {"NVIDIA MPS", "QUIET"}:
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError(f"{system} evidence lacks one result row")
        row = rows[0]
        if row.get("system") != system:
            raise ValueError(f"{system} evidence row name differs")
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        p99 = row.get("pipeline_p99_us", row.get("wall_pipeline_p99_us"))
        goodput = row.get("background_goodput_rps")
        correct = row.get("correctness_validated")
        request_trace = row.get("request_trace")
        application_output = row.get("application_output_trace")
        pipeline_path = (
            Path(request_trace["path"]).resolve()
            if isinstance(request_trace, dict) and isinstance(request_trace.get("path"), str)
            else None
        )
        output_path = (
            Path(application_output["path"]).resolve()
            if isinstance(application_output, dict) and isinstance(application_output.get("path"), str)
            else None
        )
    elif system == "XSched":
        expected_kind = f"xsched-dependent-{workload}-numeric-smoke-verification"
        legacy_control_kind = "xsched-thor-resnet-control-numeric-smoke-verification"
        if not (
            value.get("kind") == expected_kind
            and value.get("workload") == workload
            or workload == "resnet-control"
            and value.get("kind") == legacy_control_kind
        ):
            raise ValueError("XSched evidence is not the native verifier output")
        requests = value.get("requests")
        misses = value.get("misses")
        p99 = value.get("p99_us")
        goodput = value.get("background_goodput_rps")
        if goodput is None:
            goodput = value.get("background_window", {}).get("completion_goodput_rps")
        pipeline_path = evidence_path.parent / "pipeline.csv"
        output_trace = value.get("application_output_trace")
        output_path = (
            Path(output_trace["path"]).resolve()
            if isinstance(output_trace, dict) and isinstance(output_trace.get("path"), str)
            else None
        )
        correct = (
            value.get("correctness_validated") is True
            or (
                value.get("checksum_failures") == 0
                and isinstance(value.get("unique_payload_checksums"), int)
                and value["unique_payload_checksums"] >= 2
                and isinstance(value.get("unique_policy_output_checksums"), int)
                and value["unique_policy_output_checksums"] >= 2
                and value.get("token_only") is False
            )
            or (
                value.get("functional_gate_passed") is True
                and value.get("numeric_smoke_valid") is True
            )
        )
    else:  # pragma: no cover - SYSTEMS is closed
        raise ValueError(f"unsupported system: {system}")
    requests = _count(requests, f"{system}.requests")
    if isinstance(misses, bool) or not isinstance(misses, int) or not 0 <= misses <= requests:
        raise ValueError(f"{system}.misses is invalid")
    if correct is not True:
        raise ValueError(f"{system} correctness evidence did not pass")
    accuracy_binding = _application_accuracy_binding(
        system, evidence_path, workload, requests, pipeline_path, output_path
    )
    result = {
        "requests": requests,
        "misses": misses,
        "p99_us": _finite(p99, f"{system}.p99_us"),
        "background_goodput_rps": _finite(goodput, f"{system}.background_goodput_rps"),
        "correctness_validated": True,
        "path": str(evidence_path.resolve()),
        "sha256": actual_sha,
    }
    if accuracy_binding is not None:
        result["application_accuracy"] = accuracy_binding
    return result


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in paths]
    expected_orders = active_williams_orders()
    if len(paths) != len(expected_orders):
        raise ValueError(f"active Williams aggregate requires exactly {len(expected_orders)} sequences")
    seen: set[int] = set()
    contract: tuple[Any, ...] | None = None
    rows: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    source_seen: set[str] = set()
    sequence_inputs: list[dict[str, Any]] = []
    for path in paths:
        run, run_sha = _read(path)
        index = run.get("sequence_index")
        order = tuple(run.get("execution_order", ()))
        if (
            run.get("kind") != "p9-common-sota-williams-sequence"
            or run.get("proposed_system") != "QUIET"
            or isinstance(index, bool) or not isinstance(index, int)
            or index not in range(len(expected_orders)) or index in seen
            or order != expected_orders[index]
            or run.get("active_only") is not True
            or run.get("workload") not in ACTIVE_WORKLOADS
            or run.get("placement_variant") != "fixed-1g-producer-2g-consumer"
            or run.get("deadline_mode") != "wall"
            or run.get("active_exploratory_systems") != list(SYSTEMS)
            or run.get("numeric_frontier_systems") != list(NUMERIC_FRONTIER_SYSTEMS)
        ):
            raise ValueError(f"{path} differs from the active Williams contract")
        seen.add(index)
        requests = _count(run.get("requests_per_system"), "requests_per_system")
        offered = _finite(run.get("background_offered_rps"), "background_offered_rps")
        lock = run.get("deadline_lock")
        plan = run.get("quiet_plan")
        if not isinstance(lock, dict) or not isinstance(plan, dict):
            raise ValueError(f"{path} lacks lock/plan provenance")
        lock_path = Path(lock.get("path", ""))
        plan_path = Path(plan.get("path", ""))
        if not lock_path.is_file() or not plan_path.is_file():
            raise ValueError(f"{path} references missing lock/plan")
        if hashlib.sha256(lock_path.read_bytes()).hexdigest() != lock.get("sha256"):
            raise ValueError(f"{path} lock SHA differs")
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan.get("sha256"):
            raise ValueError(f"{path} plan SHA differs")
        current = (
            run.get("workload"), run.get("placement_variant"), run.get("deadline_mode"),
            offered, requests,
            _finite(run.get("background_period_ms"), "background_period_ms")
            if run.get("background_period_ms") is not None else 1000.0 / offered,
            lock.get("sha256"), plan.get("sha256"),
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"{path} differs from the shared lock/workload contract")
        results = run.get("results")
        inputs = run.get("inputs")
        if not isinstance(results, list) or not isinstance(inputs, list):
            raise ValueError(f"{path} lacks result/input arrays")
        if tuple(row.get("system") for row in results) != order or tuple(item.get("system") for item in inputs) != order:
            raise ValueError(f"{path} result order differs from Williams order")
        by_system: dict[str, dict[str, Any]] = {}
        for row, item in zip(results, inputs, strict=True):
            system = row.get("system")
            if system not in SYSTEMS or system in by_system:
                raise ValueError(f"{path} contains invalid/duplicate active system")
            if (
                row.get("requests") != requests
                or row.get("deadline_mode") != "wall"
                or row.get("latency_contract") != "production-wall-arrival-to-completion"
                or row.get("production_wall_definition") != PRODUCTION_WALL_DEFINITION
                or row.get("correctness_validated") is not True
            ):
                raise ValueError(f"{path} row contract differs for {system}")
            evidence_path = Path(item.get("path", ""))
            evidence_sha = item.get("sha256")
            if not isinstance(evidence_sha, str) or not evidence_path.is_file():
                raise ValueError(f"{path} lacks evidence path/SHA for {system}")
            evidence = _source_row(
                system, evidence_path, evidence_sha, lock["sha256"], run["workload"]
            )
            if evidence["requests"] != requests or evidence["misses"] != row.get("misses"):
                raise ValueError(f"{path} summary disagrees with raw {system} evidence")
            if not math.isclose(evidence["p99_us"], _finite(row.get("p99_us"), f"{system}.p99"), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{path} summary disagrees with raw {system} p99")
            if evidence["sha256"] in source_seen:
                raise ValueError(f"{path} reuses source evidence")
            source_seen.add(evidence["sha256"])
            by_system[system] = evidence
        sequence_inputs.append({"sequence_index": index, "path": str(path), "sha256": run_sha, "execution_order": list(order)})
        for system, evidence in by_system.items():
            rows[system].append({"sequence_index": index, **evidence})
    if seen != set(range(len(expected_orders))) or contract is None:
        raise ValueError("active Williams sequence set is incomplete")
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        values = sorted(rows[system], key=lambda row: row["sequence_index"])
        misses = sum(row["misses"] for row in values)
        requests = sum(row["requests"] for row in values)
        systems[system] = {
            "repeat_count": len(values),
            "repeats": values,
            "total_requests": requests,
            "total_deadline_misses": misses,
            "observed_dmr": misses / requests,
            "cp95_upper_dmr": _cp95(misses, requests),
            "all_repeats_zero_miss": all(row["misses"] == 0 for row in values),
            "p99_us": {
                "mean": statistics.fmean(row["p99_us"] for row in values),
                "min": min(row["p99_us"] for row in values),
                "max": max(row["p99_us"] for row in values),
            },
            "background_goodput_rps": {
                "mean": statistics.fmean(row["background_goodput_rps"] for row in values),
                "min": min(row["background_goodput_rps"] for row in values),
                "max": max(row["background_goodput_rps"] for row in values),
            },
        }
    quiet = {row["sequence_index"]: row for row in rows["QUIET"]}
    paired: dict[str, Any] = {}
    for system in ("NVIDIA MPS", "XSched"):
        baseline = {row["sequence_index"]: row for row in rows[system]}
        p99_delta = [quiet[index]["p99_us"] - baseline[index]["p99_us"] for index in sorted(quiet)]
        goodput_delta = [quiet[index]["background_goodput_rps"] - baseline[index]["background_goodput_rps"] for index in sorted(quiet)]
        paired[system] = {
            "sequence_indices": sorted(quiet),
            "p99_delta_us_quiet_minus_baseline": {"per_sequence": p99_delta, "t95": _t95(p99_delta)},
            "goodput_delta_rps_quiet_minus_baseline": {"per_sequence": goodput_delta, "t95": _t95(goodput_delta)},
            "claim_guard": "descriptive only; no thermal normalization, application accuracy, or formal SLO certification",
        }
    return {
        "schema_version": 1,
        "kind": "p9-active-williams-production-wall-repeats",
        "proposed_system": "QUIET",
        "systems": systems,
        "exploratory_systems": list(SYSTEMS),
        "numeric_frontier_systems": list(NUMERIC_FRONTIER_SYSTEMS),
        "paired_session_statistics": paired,
        "sequence_inputs": sequence_inputs,
        "workload": contract[0],
        "placement_variant": contract[1],
        "deadline_mode": contract[2],
        "background_offered_rps": contract[3],
        "requests_per_system": contract[4],
        "deadline_lock_sha256": contract[6],
        "quiet_plan_sha256": contract[7],
        "formal": False,
        "ranking_allowed": False,
        "scope": "exploratory-active-three-system-production-wall; no-thermal-normalization",
        "statistical_unit": "Williams sequence session",
        "claim_guard": "Session statistics are descriptive; CP95 does not certify the 0.05% DMR target and thermal/application gates remain open.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
