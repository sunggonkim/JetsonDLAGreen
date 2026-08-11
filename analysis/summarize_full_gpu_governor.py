#!/usr/bin/env python3
"""Aggregate repeated full-GPU drain-aware governor experiments."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
POLICIES = {
    "static-q5",
    "static-q25",
    "priority-q25",
    "conservative-guard",
    "profiled-guard",
    "joint-governor",
}


def confidence(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.fmean(values)
    if count == 1:
        return {"n": 1, "mean": mean, "stdev": 0.0, "ci95": 0.0}
    stdev = statistics.stdev(values)
    return {
        "n": count,
        "mean": mean,
        "stdev": stdev,
        "ci95": T95.get(count - 1, 1.96) * stdev / math.sqrt(count),
    }


def policy_metrics(policy: dict[str, Any], burst_size: int) -> dict[str, float]:
    amortized_gate_ms = float(policy["gate_overhead_mean_ms"])
    return {
        "deadline_miss_rate": float(policy["deadline_miss_rate"]),
        "violation_epoch_rate": float(policy["violation_epoch_rate"]),
        "critical_p99_ms_max": float(policy["critical_p99_ms_max"]),
        "pressure_goodput_per_second": float(
            policy["pressure_goodput_per_second"]
        ),
        "language_goodput_per_second": float(
            policy["goodput_by_modality"]["language"]
        ),
        "audio_goodput_per_second": float(
            policy["goodput_by_modality"]["audio"]
        ),
        "rejected_tenants": float(policy["rejected_tenants"]),
        "gate_overhead_amortized_per_request_ms": amortized_gate_ms,
        "gate_overhead_per_burst_ms": amortized_gate_ms * burst_size,
    }


def relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return candidate / baseline - 1.0


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    copy = dict(config)
    copy["policy_order"] = None
    return copy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if any(run.get("schema_version") != 1 for run in runs):
        raise SystemExit("all inputs must use schema version 1")
    reference = comparable_config(runs[0]["config"])
    for run in runs:
        if comparable_config(run["config"]) != reference:
            raise SystemExit("all inputs must have the same workload configuration")
        names = {policy["name"] for policy in run["policies"]}
        if names != POLICIES or len(run["policies"]) != len(POLICIES):
            raise SystemExit("each input must contain every policy exactly once")

    aggregate: dict[str, Any] = {}
    for name in sorted(POLICIES):
        per_run = [
            policy_metrics(
                next(policy for policy in run["policies"] if policy["name"] == name),
                int(run["config"]["burst_size"]),
            )
            for run in runs
        ]
        aggregate[name] = {
            metric: confidence([sample[metric] for sample in per_run])
            for metric in per_run[0]
        }

    governor = aggregate["joint-governor"]
    comparisons: dict[str, Any] = {}
    for baseline_name in sorted(POLICIES - {"joint-governor"}):
        baseline = aggregate[baseline_name]
        comparisons[baseline_name] = {
            "deadline_miss_rate_change": relative_change(
                governor["deadline_miss_rate"]["mean"],
                baseline["deadline_miss_rate"]["mean"],
            ),
            "goodput_change": relative_change(
                governor["pressure_goodput_per_second"]["mean"],
                baseline["pressure_goodput_per_second"]["mean"],
            ),
        }

    isolated_p99 = [
        statistics.fmean(float(value) for value in run["isolated_p99_ms"])
        for run in runs
    ]
    output = {
        "schema_version": 1,
        "config": runs[0]["config"],
        "input_files": [str(path) for path in args.inputs],
        "policy_orders": [run["config"]["policy_order"] for run in runs],
        "isolated_p99_ms": confidence(isolated_p99),
        "deadline_ms": confidence([float(run["deadline_ms"]) for run in runs]),
        "policies": aggregate,
        "governor_vs_baseline": comparisons,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
