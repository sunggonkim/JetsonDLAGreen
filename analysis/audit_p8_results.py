#!/usr/bin/env python3
"""Recompute P8 latency metrics from raw CSV traces and audit summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def read_trace(path: pathlib.Path, expected: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
    required = {
        "request",
        "release_to_completion_ms",
        "gpu_service_ms",
        "queue_delay_ms",
        "gate_overhead_ms",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError(f"{path}: unexpected CSV columns")
    for index, row in enumerate(rows):
        if int(row["request"]) != index:
            raise ValueError(f"{path}: non-contiguous request index")
    latency = [float(row["release_to_completion_ms"]) for row in rows]
    gate = [float(row["gate_overhead_ms"]) for row in rows]
    if any(not math.isfinite(value) or value < 0.0 for value in latency + gate):
        raise ValueError(f"{path}: invalid timing sample")
    return {
        "p99_ms": percentile(latency, 0.99),
        "p999_ms": percentile(latency, 0.999),
        "gate_mean_ms": statistics.fmean(gate),
        "latency": latency,
    }


def close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{context}: recomputed {actual}, summary {expected}")


def audit(path: pathlib.Path) -> dict[str, int]:
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema")
    config = run["config"]
    samples = int(config["samples_per_epoch"])
    epochs = int(config["epochs"])
    raw = path.parent / "raw"
    trace_count = 0
    request_count = 0

    calibrations = int(config["calibration_repeats"])
    if len(run["isolated"]) != calibrations:
        raise ValueError(f"{path}: isolated calibration count mismatch")
    for repeat, summary in enumerate(run["isolated"], 1):
        trace = read_trace(raw / f"isolated-r{repeat}.csv", samples)
        close(
            trace["p99_ms"],
            float(summary["release_to_completion"]["p99_ms"]),
            f"{path}: isolated r{repeat} p99",
        )
        trace_count += 1
        request_count += samples

    deadline = float(run["deadline_ms"])
    if len(run["policies"]) != len(config["policy_order"]):
        raise ValueError(f"{path}: policy count mismatch")
    if [policy["name"] for policy in run["policies"]] != config["policy_order"]:
        raise ValueError(f"{path}: policy order mismatch")
    for policy in run["policies"]:
        if len(policy["epochs"]) != epochs:
            raise ValueError(f"{path}: {policy['name']} epoch count mismatch")
        total_misses = 0
        gate_means: list[float] = []
        for epoch in policy["epochs"]:
            index = int(epoch["epoch"])
            trace_path = raw / f"{policy['name']}-e{index}.csv"
            trace = read_trace(trace_path, samples)
            misses = sum(value > deadline for value in trace["latency"])
            close(trace["p99_ms"], float(epoch["critical_p99_ms"]), f"{trace_path}: p99")
            close(trace["p999_ms"], float(epoch["critical_p999_ms"]), f"{trace_path}: p999")
            close(trace["gate_mean_ms"], float(epoch["gate_overhead_mean_ms"]), f"{trace_path}: gate")
            if misses != int(epoch["deadline_misses"]):
                raise ValueError(f"{trace_path}: deadline miss count mismatch")
            total_misses += misses
            gate_means.append(trace["gate_mean_ms"])
            trace_count += 1
            request_count += samples
        close(
            total_misses / (epochs * samples),
            float(policy["deadline_miss_rate"]),
            f"{path}: {policy['name']} DMR",
        )
        close(
            statistics.fmean(gate_means),
            float(policy["gate_overhead_mean_ms"]),
            f"{path}: {policy['name']} gate mean",
        )
    return {"traces": trace_count, "requests": request_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    total_traces = 0
    total_requests = 0
    for path in args.inputs:
        result = audit(path)
        total_traces += result["traces"]
        total_requests += result["requests"]
        print(f"audited {path}: {result['traces']} traces, {result['requests']} requests")
    print(f"audit complete: {total_traces} traces, {total_requests} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
