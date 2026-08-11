#!/usr/bin/env python3
"""Replay the frozen-plan QUIET held-out background-load sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.summarize_p9_common_sota_williams import (  # noqa: E402
    cp95_upper, percentile, replay_trace,
)


EXPECTED_RPS = (125.0, 375.0, 500.0, 600.0, 650.0, 700.0, 750.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str, owner: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    return candidate.resolve() if candidate.exists() else (owner.parent / path).resolve()


def summarize(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(EXPECTED_RPS):
        raise ValueError("held-out sweep requires exactly seven load points")
    rows: dict[float, dict[str, Any]] = {}
    lock: dict[str, str] | None = None
    plan: dict[str, str] | None = None
    workload: str | None = None
    inputs = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        offered = float(summary.get("background_offered_rps"))
        nominal = min(EXPECTED_RPS, key=lambda value: abs(value - offered))
        current_workload = summary.get("workload")
        if workload is None and current_workload in {"resnet-control", "whisper-projection"}:
            workload = current_workload
        if (
            summary.get("kind") != "p9-dependent-small-stress-smoke"
            or current_workload != workload
            or summary.get("quiet_gate_scope") != "producer"
            or abs(nominal - offered) > 1e-6
            or nominal in rows
        ):
            raise ValueError("held-out load contract differs")
        if lock is None:
            lock, plan = summary.get("deadline_lock"), summary.get("quiet_plan")
        elif summary.get("deadline_lock") != lock or summary.get("quiet_plan") != plan:
            raise ValueError("held-out lock or plan differs")
        result = summary.get("results")
        if not isinstance(result, list) or len(result) != 1 or result[0].get("system") != "QUIET":
            raise ValueError("held-out result must contain only QUIET")
        row = result[0]
        trace_info = row.get("request_trace")
        if not isinstance(trace_info, dict):
            raise ValueError("held-out trace provenance is missing")
        trace = resolve(trace_info["path"], path)
        latency_field = (
            "validation_excluded_end_to_end_us"
            if workload == "whisper-projection"
            else "wall_end_to_end_us"
        )
        replay = replay_trace(
            trace,
            trace_info["sha256"],
            float(summary["deadline_us"]),
            latency_field=latency_field,
        )
        if (
            replay["requests"] != row.get("pipeline_requests")
            or replay["misses"] != row.get("deadline_misses")
            or not math.isclose(
                percentile(replay["latencies"], 0.99),
                float(row.get("pipeline_p99_us")), abs_tol=0.01,
            )
        ):
            raise ValueError("held-out raw replay differs")
        requests, misses = replay["requests"], replay["misses"]
        rows[nominal] = {
            "offered_rps": nominal,
            "background_goodput_rps": row["background_goodput_rps"],
            "requests": requests,
            "misses": misses,
            "dmr_cp95_upper": cp95_upper(misses, requests),
            "p99_us": percentile(replay["latencies"], 0.99),
            "p999_us": percentile(replay["latencies"], 0.999),
            "maximum_us": max(replay["latencies"]),
            "trace": {"path": str(trace), "sha256": trace_info["sha256"]},
        }
        inputs.append({"path": str(path), "sha256": sha256(path)})
    if (
        set(rows) != set(EXPECTED_RPS)
        or not isinstance(lock, dict)
        or not isinstance(plan, dict)
        or workload is None
    ):
        raise ValueError("held-out sweep is incomplete")
    for value, label in ((lock, "deadline lock"), (plan, "QUIET plan")):
        source = Path(value["path"])
        if sha256(source) != value["sha256"]:
            raise ValueError(f"{label} hash differs")
    zero_miss_loads = [value for value in EXPECTED_RPS if rows[value]["misses"] == 0]
    failure_loads = [value for value in EXPECTED_RPS if rows[value]["misses"] > 0]
    monotone_failure_frontier = not failure_loads or not zero_miss_loads or (
        max(zero_miss_loads) < min(failure_loads)
    )
    return {
        "schema_version": 1,
        "kind": "p9-quiet-frozen-plan-heldout-load-sweep",
        "scope": "raw-replayed-nonthermal-heldout-characterization",
        "proposed_system": "QUIET",
        "workload": workload,
        "deadline_lock": lock,
        "quiet_plan": plan,
        "inputs": inputs,
        "loads": [rows[value] for value in EXPECTED_RPS],
        "zero_miss_offered_rps": zero_miss_loads,
        "failure_offered_rps": failure_loads,
        "monotone_failure_frontier_observed": monotone_failure_frontier,
        "maximum_zero_miss_offered_rps": max(zero_miss_loads) if zero_miss_loads else None,
        "first_observed_failure_rps": min(failure_loads) if failure_loads else None,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize([path.resolve() for path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
