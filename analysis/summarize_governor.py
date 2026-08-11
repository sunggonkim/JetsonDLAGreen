#!/usr/bin/env python3
"""Aggregate repeated governor experiments without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any


# Two-sided 95% Student-t critical values for the small experiment counts used here.
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def confidence(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.fmean(values)
    if count == 1:
        return {"n": 1, "mean": mean, "stdev": 0.0, "ci95": 0.0}
    stdev = statistics.stdev(values)
    critical = T95.get(count - 1, 1.96)
    return {
        "n": count,
        "mean": mean,
        "stdev": stdev,
        "ci95": critical * stdev / math.sqrt(count),
    }


def policy_metrics(policy: dict[str, Any]) -> dict[str, float]:
    epochs = policy["epochs"]
    return {
        "deadline_miss_rate": policy["deadline_miss_rate"],
        "pressure_goodput_per_second": policy["pressure_goodput_per_second"],
        "violation_epoch_rate": policy["violation_epoch_rate"],
        "p99_ms_max": max(epoch["p99_ms"] for epoch in epochs),
        "memory_p99_ms_max": policy["by_mode"]["memory"]["p99_ms_max"],
        "compute_p99_ms_max": policy["by_mode"]["compute"]["p99_ms_max"],
        "memory_pressure_launches": policy["by_mode"]["memory"][
            "pressure_launches"
        ],
        "compute_pressure_launches": policy["by_mode"]["compute"][
            "pressure_launches"
        ],
    }


def main() -> int:
    args = parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    reference = runs[0]["config"]
    comparable_reference = reference | {"policy_order": None}
    for run in runs[1:]:
        comparable = run["config"] | {"policy_order": None}
        if comparable != comparable_reference:
            raise SystemExit("all inputs must use an identical experiment config")

    names = [policy["name"] for policy in runs[0]["policies"]]
    aggregate: dict[str, Any] = {}
    for name in names:
        samples = []
        for run in runs:
            policy = next(item for item in run["policies"] if item["name"] == name)
            samples.append(policy_metrics(policy))
        aggregate[name] = {
            metric: confidence([sample[metric] for sample in samples])
            for metric in samples[0]
        }

    governor = aggregate["jdg-governor"]
    comparisons = {}
    for baseline in ("static-q25", "static-q100"):
        base = aggregate[baseline]
        comparisons[baseline] = {
            "miss_rate_reduction": 1.0
            - governor["deadline_miss_rate"]["mean"]
            / base["deadline_miss_rate"]["mean"],
            "goodput_improvement": governor["pressure_goodput_per_second"]["mean"]
            / base["pressure_goodput_per_second"]["mean"]
            - 1.0,
        }

    output = {
        "schema_version": 1,
        "config": reference,
        "policy_orders": [run["config"]["policy_order"] for run in runs],
        "input_files": [str(path) for path in args.inputs],
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
