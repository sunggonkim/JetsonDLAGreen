#!/usr/bin/env python3
"""Summarize the current labelled ImageNette offered-load frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import beta


SYSTEMS = ("NVIDIA MPS", "QUIET")
PERIOD_TO_LOAD = {8.0: 125.0, 4.0: 250.0, 2.6666667: 374.99999531250006}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def cp95(misses: int, requests: int) -> float:
    if misses == requests:
        return 1.0
    return float(beta.ppf(0.95, misses + 1, requests - misses))


def bound(path: Path, expected: str | None, label: str) -> dict[str, str]:
    if not path.is_file() or (expected is not None and sha256(path) != expected):
        raise ValueError(f"{label} provenance differs")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def summarize(inputs: list[Path], thermal_summary: Path) -> dict[str, Any]:
    if len(inputs) != 18:
        raise ValueError("current load frontier requires 18 independent point rows")
    rows: list[dict[str, Any]] = []
    contract: tuple[Any, ...] | None = None
    seen_evidence: set[str] = set()
    for path in inputs:
        value = json.loads(path.read_bytes())
        if (
            value.get("kind") != "p9-dependent-small-stress-smoke"
            or value.get("workload") != "resnet50-classification"
            or value.get("iterations") != 1100
            or value.get("deadline_mode") != "wall"
            or value.get("production_wall_definition")
                != "arrival-to-consumer-completion-excludes-correctness-validation"
            or value.get("checksum_mode") != "inline"
        ):
            raise ValueError(f"load frontier row contract differs: {path}")
        result_rows = value.get("results")
        if not isinstance(result_rows, list) or len(result_rows) != 1:
            raise ValueError(f"load frontier result row differs: {path}")
        row = result_rows[0]
        system = row.get("system")
        if system not in SYSTEMS or row.get("pipeline_requests") != 1100:
            raise ValueError(f"load frontier system/request row differs: {path}")
        period = finite(value.get("background_period_ms"), "background period")
        load = next(
            (offered for known, offered in PERIOD_TO_LOAD.items() if math.isclose(period, known, abs_tol=1e-6)),
            None,
        )
        if load is None:
            raise ValueError(f"unsupported load frontier period: {period}")
        lock = value.get("deadline_lock")
        common = value.get("common_workload")
        if not isinstance(lock, dict) or not isinstance(common, dict):
            raise ValueError("load frontier provenance is incomplete")
        current_contract = (
            lock.get("sha256"), common.get("contract_sha256"),
            common.get("producer_input_trace_sha256"), common.get("operational_arrival_trace_sha256"),
        )
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError("load frontier current contract differs")
        trace = row.get("request_trace")
        trace_bound = bound(Path(trace.get("path", "")), trace.get("sha256"), "load frontier request trace") if isinstance(trace, dict) else None
        if trace_bound is None or trace_bound["sha256"] in seen_evidence:
            raise ValueError("load frontier request trace is missing or duplicated")
        seen_evidence.add(trace_bound["sha256"])
        accuracy_path = path.parent / "application-accuracy" / "accuracy-gate.json"
        accuracy = json.loads(accuracy_path.read_bytes())
        if (
            accuracy.get("kind") != "p9-application-accuracy-gate"
            or accuracy.get("status") != "passed"
            or accuracy.get("numeric_comparison_allowed") is not True
            or accuracy.get("candidate_accuracy", 0.0) < 0.8
            or accuracy.get("accuracy_delta") != 0.0
            or accuracy.get("application_input_binding_contract") != "passed"
            or accuracy.get("application_output_trace_contract") != "passed"
        ):
            raise ValueError(f"load frontier application gate differs: {accuracy_path}")
        rows.append({
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "system": system,
            "period_ms": period,
            "offered_load_rps": load,
            "requests": 1100,
            "misses": row["deadline_misses"],
            "p99_us": finite(row["pipeline_p99_us"], "load frontier p99"),
            "background_goodput_rps": finite(row["background_goodput_rps"], "load frontier goodput"),
            "request_trace": trace_bound,
            "application_accuracy": {
                "path": str(accuracy_path.resolve()),
                "sha256": sha256(accuracy_path),
                "candidate_accuracy": accuracy["candidate_accuracy"],
                "delta": accuracy["accuracy_delta"],
            },
            "quiet_plan_violation": value.get("quiet_plan_violation"),
        })
    if contract is None:
        raise ValueError("load frontier contract is empty")
    grouped: dict[str, dict[float, list[dict[str, Any]]]] = {
        system: {} for system in SYSTEMS
    }
    for row in rows:
        grouped[row["system"]].setdefault(row["offered_load_rps"], []).append(row)
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        points = []
        if set(grouped[system]) != set(PERIOD_TO_LOAD.values()):
            raise ValueError(f"{system} load point set differs")
        for load in sorted(grouped[system]):
            repetitions = grouped[system][load]
            if len(repetitions) != 3:
                raise ValueError(f"{system} load point lacks three repetitions: {load}")
            requests = sum(row["requests"] for row in repetitions)
            misses = sum(row["misses"] for row in repetitions)
            point = {
                "offered_load_rps": load,
                "period_ms": repetitions[0]["period_ms"],
                "sessions": len(repetitions),
                "requests": requests,
                "misses": misses,
                "observed_dmr": misses / requests,
                "dmr_cp95_upper": cp95(misses, requests),
                "cp95_slo_qualified": misses == 0 and cp95(misses, requests) <= 0.0005,
                "p99_us": statistics.fmean(row["p99_us"] for row in repetitions),
                "p99_us_by_session": [row["p99_us"] for row in repetitions],
                "background_goodput_rps": statistics.fmean(row["background_goodput_rps"] for row in repetitions),
                "application_accuracy_min": min(row["application_accuracy"]["candidate_accuracy"] for row in repetitions),
                "repetitions": repetitions,
            }
            points.append(point)
        systems[system] = {
            "points": points,
            "cp95_qualified_points": [point["offered_load_rps"] for point in points if point["cp95_slo_qualified"]],
        }
    thermal = json.loads(thermal_summary.read_bytes())
    if (
        thermal.get("kind") != "p9-current-imagenette-thermal-formal-production-wall-aggregate"
        or thermal.get("thermal_normalized") is not True
        or thermal.get("thermal_claim_allowed") is not True
        or thermal.get("systems", {}).get("QUIET", {}).get("slo_confidence_qualified") is not True
    ):
        raise ValueError("thermal anchor is not a qualified current QUIET gate")
    thermal_bound = bound(thermal_summary, None, "thermal anchor")
    return {
        "schema_version": 1,
        "kind": "p9-current-imagenette-dmr-goodput-load-frontier",
        "proposed_system": "QUIET",
        "workload": "resnet50-classification",
        "scope": "current-labelled-imagenette-fixed-placement-three-load-descriptive-frontier",
        "formal": False,
        "thermal_normalized": False,
        "ranking_allowed": False,
        "application_accuracy_bound": True,
        "statistical_unit": "request-level DMR with three independent sessions per offered load",
        "dmr_target": 0.0005,
        "systems": systems,
        "common_contract": {
            "deadline_lock_sha256": contract[0],
            "common_workload_sha256": contract[1],
            "producer_input_trace_sha256": contract[2],
            "operational_arrival_trace_sha256": contract[3],
        },
        "thermal_anchor": thermal_bound,
        "claim_guard": "Load points are descriptive because three-session CP95 bounds do not qualify; the six-session thermal anchor is the qualified 250-rps QUIET point.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--thermal-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize([path.resolve() for path in args.input], args.thermal_summary.resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
