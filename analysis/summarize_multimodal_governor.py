#!/usr/bin/env python3
"""Aggregate repeated TensorRT multimodal-governor experiments."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


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


def metrics(policy: dict[str, Any]) -> dict[str, float]:
    return {
        "deadline_miss_rate": policy["deadline_miss_rate"],
        "violation_epoch_rate": policy["violation_epoch_rate"],
        "critical_p99_ms_max": policy["critical_p99_ms_max"],
        "pressure_goodput_per_second": policy["pressure_goodput_per_second"],
        "language_goodput_per_second": policy["goodput_by_modality"]["language"],
        "audio_goodput_per_second": policy["goodput_by_modality"]["audio"],
        "rejected_tenants": float(policy["rejected_tenants"]),
    }


def relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return candidate / baseline - 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    reference = runs[0]["config"] | {"policy_order": None}
    for run in runs[1:]:
        if (run["config"] | {"policy_order": None}) != reference:
            raise SystemExit("all inputs must have the same workload configuration")
    names = [policy["name"] for policy in runs[0]["policies"]]
    aggregate: dict[str, Any] = {}
    for name in names:
        per_run = [
            metrics(next(policy for policy in run["policies"] if policy["name"] == name))
            for run in runs
        ]
        aggregate[name] = {
            metric: confidence([sample[metric] for sample in per_run])
            for metric in per_run[0]
        }
    governor = aggregate["joint-governor"]
    comparisons = {}
    for baseline_name in (
        "static-q25",
        "static-q100",
        "time-division",
        "profiled",
    ):
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
    output = {
        "schema_version": 1,
        "config": runs[0]["config"],
        "input_files": [str(path) for path in args.inputs],
        "policy_orders": [run["config"]["policy_order"] for run in runs],
        "isolated_p99_ms": confidence(
            [run["isolated"]["release_to_completion"]["p99_ms"] for run in runs]
        ),
        "deadline_ms": confidence([run["deadline_ms"] for run in runs]),
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
