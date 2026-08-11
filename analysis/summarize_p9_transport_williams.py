#!/usr/bin/env python3
"""Raw-replay four balanced Whisper transport ablation sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import t


TREATMENTS = (
    "cross-mig-registered", "cross-mig-pinned",
    "cross-mig-pageable", "same-instance-registered",
)
EXPECTED_HEADER = (
    "request", "producer_compute_us", "producer_copy_us", "producer_validation_us",
    "notification_us", "consumer_validation_us", "consumer_copy_us",
    "edge_transport_us", "consumer_compute_us", "output_verification_us",
    "validation_excluded_end_to_end_us", "wall_end_to_end_us", "deadline_miss",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile input is empty")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def replay(path: Path, expected_rows: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
            raise ValueError(f"unexpected trace schema: {path}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError(f"trace row count differs: {path}")
    if [int(row["request"]) for row in rows] != list(range(100, 100 + expected_rows)):
        raise ValueError(f"trace request sequence differs: {path}")
    metrics: dict[str, list[float]] = {
        "edge": [], "pipeline": [], "producer_copy": [], "consumer_copy": [], "notification": []
    }
    columns = {
        "edge": "edge_transport_us",
        "pipeline": "validation_excluded_end_to_end_us",
        "producer_copy": "producer_copy_us",
        "consumer_copy": "consumer_copy_us",
        "notification": "notification_us",
    }
    for row in rows:
        for name, column in columns.items():
            value = float(row[column])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {column} in {path}")
            metrics[name].append(value)
    return metrics


def mean_t95(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("paired interval requires at least two values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = float(t.ppf(0.975, len(values) - 1)) * math.sqrt(variance / len(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def summarize(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 4:
        raise ValueError("exactly four sequence summaries are required")
    sequences: dict[int, dict[str, dict[str, Any]]] = {}
    artifact_contract: dict[str, Any] | None = None
    inputs: list[dict[str, str]] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        index = raw.get("sequence_index")
        if raw.get("kind") != "p9-whisper-transport-williams-sequence" or index not in range(4):
            raise ValueError(f"invalid sequence summary: {path}")
        if index in sequences or raw.get("payload_bytes") != 2_304_000:
            raise ValueError("duplicate sequence or wrong payload")
        if artifact_contract is None:
            artifact_contract = raw["artifacts"]
        elif raw["artifacts"] != artifact_contract:
            raise ValueError("sequence artifact contracts differ")
        order = raw.get("execution_order")
        runs = raw.get("runs")
        if not isinstance(order, list) or not isinstance(runs, list) or len(runs) != 4:
            raise ValueError("invalid sequence layout")
        if order != [item.get("treatment") for item in runs] or set(order) != set(TREATMENTS):
            raise ValueError("execution order differs from run order")
        expected_rows = raw.get("iterations_per_treatment")
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows <= 0:
            raise ValueError("invalid iteration count")
        sequence: dict[str, dict[str, Any]] = {}
        for position, item in enumerate(runs):
            if item.get("position") != position:
                raise ValueError("run position differs")
            result_path = Path(item["result"]["path"]).resolve()
            trace_path = Path(item["trace"]["path"]).resolve()
            if sha256(result_path) != item["result"]["sha256"] or sha256(trace_path) != item["trace"]["sha256"]:
                raise ValueError("run evidence hash differs")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("status") != "ok" or result.get("iterations") != expected_rows
                or result.get("payload_bytes") != 2_304_000 or result.get("checksum_failures") != 0
                or result.get("unique_payload_checksums", 0) < 2
                or result.get("unique_policy_output_checksums", 0) < 2
            ):
                raise ValueError("invalid pipeline result")
            metrics = replay(trace_path, expected_rows)
            sequence[item["treatment"]] = {
                "metrics": metrics,
                "trace": {"path": str(trace_path), "sha256": sha256(trace_path)},
            }
        sequences[index] = sequence
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
    if set(sequences) != set(range(4)):
        raise ValueError("sequence indices are incomplete")

    rows: dict[str, Any] = {}
    per_sequence_edge_p99: dict[str, list[float]] = {name: [] for name in TREATMENTS}
    for treatment in TREATMENTS:
        pooled = {name: [] for name in ("edge", "pipeline", "producer_copy", "consumer_copy", "notification")}
        traces = []
        for index in range(4):
            current = sequences[index][treatment]
            for name in pooled:
                pooled[name].extend(current["metrics"][name])
            per_sequence_edge_p99[treatment].append(percentile(current["metrics"]["edge"], 0.99))
            traces.append(current["trace"])
        rows[treatment] = {
            "requests": len(pooled["edge"]),
            "edge_p50_us": percentile(pooled["edge"], 0.50),
            "edge_p99_us": percentile(pooled["edge"], 0.99),
            "pipeline_p99_us": percentile(pooled["pipeline"], 0.99),
            "producer_copy_p99_us": percentile(pooled["producer_copy"], 0.99),
            "consumer_copy_p99_us": percentile(pooled["consumer_copy"], 0.99),
            "notification_p99_us": percentile(pooled["notification"], 0.99),
            "trace_inputs": traces,
        }
    registered = per_sequence_edge_p99["cross-mig-registered"]
    comparisons = {}
    for treatment in ("same-instance-registered", "cross-mig-pinned", "cross-mig-pageable"):
        differences = [
            per_sequence_edge_p99[treatment][index] - registered[index]
            for index in range(4)
        ]
        comparisons[f"{treatment}_minus_cross_mig_registered_edge_p99_us"] = {
            "paired_differences": differences,
            "mean_t95": mean_t95(differences),
        }
    return {
        "schema_version": 1,
        "kind": "p9-whisper-transport-williams-aggregate",
        "scope": "balanced-performance-ablation-not-thermal-formal",
        "payload_bytes": 2_304_000,
        "systems": rows,
        "paired_comparisons": comparisons,
        "artifacts": artifact_contract,
        "inputs": inputs,
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
