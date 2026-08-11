#!/usr/bin/env python3
"""Replay and aggregate repeated payload-valid dependent smokes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta


SYSTEM_ORDER = (
    "NVIDIA MIG",
    "NVIDIA MPS",
    "Process-stop ablation",
    "QUIET",
)
WILLIAMS_4 = (
    ("NVIDIA MIG", "NVIDIA MPS", "QUIET", "Process-stop ablation"),
    ("NVIDIA MPS", "Process-stop ablation", "NVIDIA MIG", "QUIET"),
    ("Process-stop ablation", "QUIET", "NVIDIA MPS", "NVIDIA MIG"),
    ("QUIET", "NVIDIA MIG", "Process-stop ablation", "NVIDIA MPS"),
)
TRACE_COLUMNS = (
    "request",
    "producer_compute_us",
    "producer_copy_us",
    "producer_validation_us",
    "notification_us",
    "consumer_validation_us",
    "consumer_copy_us",
    "edge_transport_us",
    "consumer_compute_us",
    "output_verification_us",
    "validation_excluded_end_to_end_us",
    "wall_end_to_end_us",
    "deadline_miss",
)
DMR_TARGET = 0.0005


def clopper_pearson_upper(misses: int, requests: int, confidence: float = 0.95) -> float:
    if not 0 <= misses <= requests or requests <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid exact-binomial confidence input")
    if misses == requests:
        return 1.0
    return float(beta.ppf(confidence, misses + 1, requests - misses))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile input is empty")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def replay_trace(path: Path, expected_hash: str, deadline_us: float) -> dict[str, Any]:
    if sha256(path) != expected_hash:
        raise ValueError(f"trace hash differs: {path}")
    latencies: list[float] = []
    misses = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
            raise ValueError(f"trace schema differs: {path}")
        for index, row in enumerate(reader):
            if int(row["request"]) < 0:
                raise ValueError("request index must be nonnegative")
            latency = float(row["validation_excluded_end_to_end_us"])
            if not math.isfinite(latency) or latency < 0.0:
                raise ValueError("trace latency must be finite and nonnegative")
            recorded = int(row["deadline_miss"])
            if recorded not in (0, 1) or recorded != int(latency > deadline_us):
                raise ValueError("trace deadline decision differs")
            latencies.append(latency)
            misses += recorded
    return {"requests": len(latencies), "misses": misses, "latencies": latencies}


def summarize(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two repeated summaries are required")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    first = runs[0]
    contract_keys = (
        "deadline_us",
        "iterations",
        "background_period_ms",
        "background_offered_rps",
        "producer_quota_percent",
        "background_quota_percent",
        "workload",
        "quiet_gate_scope",
        "deadline_source",
        "deadline_lock",
    )
    contract = {key: first.get(key) for key in contract_keys}
    if (
        first.get("kind") != "p9-dependent-small-stress-smoke"
        or contract["workload"] != "whisper-projection"
        or contract["quiet_gate_scope"] != "producer"
    ):
        raise ValueError("unexpected repeated-smoke contract")

    per_system: dict[str, list[dict[str, Any]]] = {name: [] for name in SYSTEM_ORDER}
    inputs: list[dict[str, str]] = []
    execution_orders: list[tuple[str, ...]] = []
    for summary_path, run in zip(paths, runs, strict=True):
        if {key: run.get(key) for key in contract_keys} != contract:
            raise ValueError("repeated-smoke contracts differ")
        rows = run.get("results")
        actual_order = tuple(row.get("system") for row in rows) if isinstance(rows, list) else ()
        recorded_order = tuple(run.get("execution_order", actual_order))
        if (
            set(actual_order) != set(SYSTEM_ORDER)
            or len(actual_order) != len(SYSTEM_ORDER)
            or recorded_order != actual_order
        ):
            raise ValueError("system execution order differs")
        execution_orders.append(actual_order)
        inputs.append({"path": str(summary_path.resolve()), "sha256": sha256(summary_path)})
        for row in rows:
            trace = row.get("request_trace")
            if not isinstance(trace, dict):
                raise ValueError("result lacks request trace provenance")
            trace_path = Path(trace["path"])
            if not trace_path.is_absolute():
                trace_path = (Path.cwd() / trace_path).resolve()
            replay = replay_trace(trace_path, trace["sha256"], float(contract["deadline_us"]))
            if (
                replay["requests"] != row.get("pipeline_requests")
                or replay["misses"] != row.get("deadline_misses")
                or not math.isclose(
                    percentile(replay["latencies"], 0.99),
                    finite(row.get("pipeline_p99_us"), "pipeline p99"),
                    rel_tol=1e-9,
                    abs_tol=0.01,
                )
            ):
                raise ValueError("trace replay differs from summary")
            per_system[row["system"]].append(
                {
                    "latencies": replay["latencies"],
                    "misses": replay["misses"],
                    "requests": replay["requests"],
                    "background_goodput_rps": finite(
                        row.get("background_goodput_rps"), "background goodput"
                    ),
                    "trace": {"path": str(trace_path), "sha256": trace["sha256"]},
                }
            )

    aggregates: dict[str, dict[str, Any]] = {}
    for name, samples in per_system.items():
        latencies = [value for sample in samples for value in sample["latencies"]]
        requests = sum(sample["requests"] for sample in samples)
        misses = sum(sample["misses"] for sample in samples)
        goodputs = [sample["background_goodput_rps"] for sample in samples]
        aggregates[name] = {
            "runs": len(samples),
            "requests": requests,
            "misses": misses,
            "observed_dmr": misses / requests,
            "dmr_target": DMR_TARGET,
            "dmr_cp95_upper": clopper_pearson_upper(misses, requests),
            "confidence_qualified": (
                clopper_pearson_upper(misses, requests) <= DMR_TARGET
            ),
            "pooled_p99_us": percentile(latencies, 0.99),
            "pooled_p999_us": percentile(latencies, 0.999),
            "maximum_us": max(latencies),
            "background_goodput_rps_mean": statistics.fmean(goodputs),
            "background_goodput_rps_range": [min(goodputs), max(goodputs)],
            "trace_inputs": [sample["trace"] for sample in samples],
        }
    order_design = "unbalanced-or-exploratory"
    if len(execution_orders) == 4:
        if set(execution_orders) != set(WILLIAMS_4):
            raise ValueError("four-run execution order is not the frozen Williams design")
        order_design = "four-treatment-williams"
    return {
        "schema_version": 1,
        "kind": "p9-dependent-whisper-repeated-smoke",
        "scope": "repeated-functional-smoke-with-exact-binomial-screen",
        "proposed_system": "QUIET",
        "contract": contract,
        "inputs": inputs,
        "order_design": order_design,
        "execution_orders": [list(order) for order in execution_orders],
        "systems": {
            name: aggregates[name] for name in ("NVIDIA MIG", "NVIDIA MPS", "QUIET")
        },
        "ablations": {"Process-stop ablation": aggregates["Process-stop ablation"]},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize([path.resolve() for path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
