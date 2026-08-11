#!/usr/bin/env python3
"""Replay independent pipeline traces and freeze the 1.10x p99 deadline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


TRACE_COLUMNS = (
    "request", "producer_compute_us", "producer_copy_us",
    "input_sha256",
    "producer_validation_us", "notification_us", "consumer_validation_us",
    "consumer_copy_us", "edge_transport_us", "consumer_compute_us",
    "output_verification_us", "validation_excluded_end_to_end_us",
    "wall_end_to_end_us", "deadline_miss",
)

WORKLOADS = {
    "whisper-projection": {
        "pipeline": "whisper-last-hidden-state-to-projection-mlp",
        "payload_bytes": 2304000,
        "deadline_modes": {
            "validation-excluded": {
                "trace_column": "validation_excluded_end_to_end_us",
                "summary_object": "stage_latency_us",
                "summary_key": "validation_excluded_end_to_end_p99",
            },
            "wall": {
                "trace_column": "wall_end_to_end_us",
                "summary_object": "end_to_end_us",
                "summary_key": "p99",
            },
        },
    },
    "resnet-control": {
        "pipeline": "resnet10-layer7-cov-to-control-mlp",
        "payload_bytes": 14720,
        "deadline_modes": {
            "wall": {
                "trace_column": "wall_end_to_end_us",
                "summary_object": "end_to_end_us",
                "summary_key": "p99",
            },
        },
    },
    "resnet-detection-head": {
        "pipeline": "resnet10-backbone-to-learned-detection-head",
        "payload_bytes": 1884160,
        "deadline_modes": {
            "wall": {
                "trace_column": "wall_end_to_end_us",
                "summary_object": "end_to_end_us",
                "summary_key": "p99",
            },
        },
    },
    "resnet50-classification": {
        "pipeline": "resnet50-backbone-to-classification-head",
        "payload_bytes": 802816,
        "deadline_modes": {
            "wall": {
                "trace_column": "wall_end_to_end_us",
                "summary_object": "end_to_end_us",
                "summary_key": "p99",
            },
        },
    },
}

PLACEMENTS = {
    "fixed-1g-producer-2g-consumer": ("mig-1g-q100", "mig-2g-q100", "1g", "2g"),
    "fixed-2g-producer-1g-consumer": ("mig-2g-q100", "mig-1g-q100", "2g", "1g"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty calibration trace")
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def replay_trace(path: Path, digest: str, trace_column: str) -> list[float]:
    if sha256(path) != digest:
        raise ValueError("calibration trace hash differs")
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
            raise ValueError("calibration trace schema differs")
        for row in reader:
            value = float(row[trace_column])
            if not math.isfinite(value) or value <= 0.0 or int(row["deadline_miss"]) != 0:
                raise ValueError("invalid deadline-disabled calibration row")
            values.append(value)
    return values


def build_lock(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    workload = WORKLOADS.get(config.get("workload"))
    deadline_mode = config.get("deadline_mode")
    metric = workload.get("deadline_modes", {}).get(deadline_mode) if workload else None
    placement_variant = config.get("placement_variant", "fixed-1g-producer-2g-consumer")
    placement = PLACEMENTS.get(placement_variant)
    if (
        summary.get("kind") != "p9-dependent-pipeline-deadline-calibration"
        or workload is None
        or config.get("payload_bytes") != workload["payload_bytes"]
        or metric is None
        or config.get("slo_factor") != 1.10
        or config.get("producer_quota_percent") != 100
        or config.get("producer_uuid") == config.get("consumer_uuid")
        or placement is None
        or config.get("producer_profile", placement[0]) != placement[0]
        or config.get("consumer_profile", placement[1]) != placement[1]
        or config.get("producer_role", placement[2]) != placement[2]
        or config.get("consumer_role", placement[3]) != placement[3]
    ):
        raise ValueError("invalid pipeline calibration contract")
    root = summary_path.parent
    pooled: list[float] = []
    evidence: list[dict[str, Any]] = []
    blocks = summary.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != config.get("blocks") or len(blocks) < 2:
        raise ValueError("calibration block count differs")
    seen_hashes: set[str] = set()
    for expected_index, block in enumerate(blocks):
        if block.get("index") != expected_index:
            raise ValueError("calibration block order differs")
        result_path = (root / block["result_path"]).resolve()
        trace_path = (root / block["trace_path"]).resolve()
        if sha256(result_path) != block["result_sha256"]:
            raise ValueError("calibration result hash differs")
        if block["trace_sha256"] in seen_hashes:
            raise ValueError("calibration traces are not independent files")
        seen_hashes.add(block["trace_sha256"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        values = replay_trace(
            trace_path, block["trace_sha256"], str(metric["trace_column"])
        )
        if (
            result.get("status") != "ok"
            or result.get("pipeline") != workload["pipeline"]
            or result.get("transport") != config.get("transport")
            or result.get("payload_bytes") != config.get("payload_bytes")
            or result.get("iterations") != config.get("samples_per_block")
            or result.get("producer_quota") != 100
            or result.get("checksum_failures") != 0
            or result.get("deadline_us") is not None
            or result.get("deadline_misses") != 0
            or result.get("deadline_mode") != deadline_mode
            or len(values) != config.get("samples_per_block")
            or not math.isclose(
                percentile(values, 0.99),
                float(result[metric["summary_object"]][metric["summary_key"]]),
                abs_tol=0.01,
            )
        ):
            raise ValueError("calibration result differs from raw trace")
        pooled.extend(values)
        evidence.append(
            {
                "index": expected_index,
                "result_path": str(result_path),
                "result_sha256": block["result_sha256"],
                "trace_path": str(trace_path),
                "trace_sha256": block["trace_sha256"],
                "samples": len(values),
                "p99_us": percentile(values, 0.99),
            }
        )
    pooled_p99 = percentile(pooled, 0.99)
    return {
        "schema_version": 1,
        "kind": "p9-dependent-pipeline-deadline-lock",
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": sha256(summary_path),
        "contract": config,
        "pooled_samples": len(pooled),
        "pooled_p99_us": pooled_p99,
        "slo_factor": 1.10,
        "deadline_us": pooled_p99 * 1.10,
        "artifacts": summary.get("artifacts"),
        "evidence": evidence,
    }


def verify_artifacts(lock: dict[str, Any]) -> None:
    """Rehash every calibration artifact at the lock trust boundary."""
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("deadline lock lacks artifact provenance")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"deadline artifact {name} is invalid")
        path_value, expected = artifact.get("path"), artifact.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"deadline artifact {name} lacks path")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"deadline artifact {name} lacks SHA-256")
        path = Path(path_value)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("deadline artifact changed after calibration")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        lock = json.loads(args.verify.read_text(encoding="utf-8"))
        rebuilt = build_lock(Path(lock["source_summary"]))
        if lock != rebuilt:
            raise ValueError("pipeline deadline lock differs from source evidence")
        verify_artifacts(lock)
        print(json.dumps(lock, indent=2))
        return 0
    if args.summary is None or args.output is None:
        parser.error("--summary and --output are required when not verifying")
    lock = build_lock(args.summary.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
