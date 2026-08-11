#!/usr/bin/env python3
"""Aggregate repeated P8 guard, burst, and arrival-period sweeps."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from collections import defaultdict
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


def point_key(run: dict[str, Any]) -> tuple[str, float]:
    config = run["config"]
    label = str(config["experiment_label"])
    if label == "guard":
        guard = config.get("guard_override_ms")
        if guard is None:
            raise ValueError("guard sweep input is missing guard_override_ms")
        return label, float(guard)
    if label == "burst":
        return label, float(config["burst_size"])
    if label == "period":
        return label, float(config["period_ms"])
    raise ValueError(f"unsupported experiment label: {label}")


def metrics(run: dict[str, Any]) -> dict[str, float]:
    if len(run["policies"]) != 1:
        raise ValueError("sensitivity inputs must contain exactly one policy")
    policy = run["policies"][0]
    amortized_gate_ms = float(policy["gate_overhead_mean_ms"])
    return {
        "deadline_miss_rate": float(policy["deadline_miss_rate"]),
        "critical_p99_ms_max": float(policy["critical_p99_ms_max"]),
        "pressure_goodput_per_second": float(
            policy["pressure_goodput_per_second"]
        ),
        "gate_overhead_amortized_per_request_ms": amortized_gate_ms,
        "gate_overhead_per_burst_ms": amortized_gate_ms
        * int(run["config"]["burst_size"]),
        "deadline_ms": float(run["deadline_ms"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    input_files: list[str] = []
    for path in args.inputs:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("schema_version") != 1:
            raise SystemExit(f"unsupported schema in {path}")
        try:
            grouped[point_key(run)].append(run)
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"invalid sensitivity input {path}: {error}") from error
        input_files.append(str(path))

    experiments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (label, x_value), runs in sorted(grouped.items()):
        samples = [metrics(run) for run in runs]
        configs = [run["config"] for run in runs]
        policy_names = {run["policies"][0]["name"] for run in runs}
        if len(policy_names) != 1:
            raise SystemExit(f"mixed policies at {label}={x_value}")
        point = {
            "x": x_value,
            "policy": next(iter(policy_names)),
            "burst_size": configs[0]["burst_size"],
            "period_ms": configs[0]["period_ms"],
            "guard_override_ms": configs[0].get("guard_override_ms"),
            "metrics": {
                name: confidence([sample[name] for sample in samples])
                for name in samples[0]
            },
        }
        experiments[label].append(point)

    output = {
        "schema_version": 1,
        "input_files": input_files,
        "experiments": dict(experiments),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
