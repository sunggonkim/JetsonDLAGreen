#!/usr/bin/env python3
"""Verify Orion on the real ResNet10 Layer7_cov dependent pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "orion"))
from verify_dependent_smoke import (  # noqa: E402
    TRACE_COLUMNS, load_json, load_profile_bundle, percentile, replay_events, sha256,
)

# Historical Orion/XSched artifacts predate the input-hash column.  Keep them
# replayable as legacy evidence while requiring the current column for new
# production-wall captures.
LEGACY_TRACE_COLUMNS = tuple(
    column for column in TRACE_COLUMNS if column != "input_sha256"
)


def replay_checksums(path: Path, result: dict[str, Any]) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["request", "payload_checksum", "output_checksum"]:
            raise ValueError("Orion checksum trace schema differs")
        rows = list(reader)
    if len(rows) != result.get("iterations"):
        raise ValueError("Orion checksum trace count differs")
    payloads: set[int] = set()
    outputs: set[int] = set()
    for index, row in enumerate(rows):
        if int(row["request"]) != result["warmup"] + index:
            raise ValueError("Orion checksum request sequence differs")
        payload, output = int(row["payload_checksum"]), int(row["output_checksum"])
        if payload <= 0 or output <= 0:
            raise ValueError("Orion checksum value differs")
        payloads.add(payload)
        outputs.add(output)
    if (len(payloads) != result.get("unique_payload_checksums") or
            len(outputs) != result.get("unique_policy_output_checksums")):
        raise ValueError("Orion checksum cardinality differs")
    return {"payloads": len(payloads), "outputs": len(outputs)}


def replay_wall_pipeline(
    path: Path, result: dict[str, Any], deadline_us: float
) -> dict[str, Any]:
    latencies: list[float] = []
    misses = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) not in {TRACE_COLUMNS, LEGACY_TRACE_COLUMNS}:
            raise ValueError("Orion ResNet pipeline trace schema differs")
        for index, row in enumerate(reader):
            if int(row["request"]) != result["warmup"] + index:
                raise ValueError("Orion ResNet request sequence differs")
            latency = float(row["wall_end_to_end_us"])
            if not math.isfinite(latency) or latency <= 0.0:
                raise ValueError("Orion ResNet latency differs")
            miss = latency > deadline_us
            if int(row["deadline_miss"]) != int(miss):
                raise ValueError("Orion ResNet deadline classification differs")
            misses += int(miss)
            latencies.append(latency)
    if len(latencies) != result.get("iterations") or misses != result.get(
        "deadline_misses"
    ):
        raise ValueError("Orion ResNet request totals differ")
    p99 = percentile(latencies, 0.99)
    if not math.isclose(
        p99, float(result["end_to_end_us"]["p99"]), rel_tol=1e-12, abs_tol=1e-6
    ):
        raise ValueError("Orion ResNet p99 differs")
    return {"requests": len(latencies), "misses": misses, "p99_us": p99}


def verify(
    result_path: Path, pipeline_path: Path, checksum_path: Path,
    events_path: Path, best_effort_profile_path: Path,
    high_priority_profile_path: Path, best_effort_scheduler_path: Path,
    high_priority_scheduler_path: Path, binary_path: Path,
    expected_requests: int | None = None,
) -> dict[str, Any]:
    result = load_json(result_path)
    orion = result.get("orion")
    deadline = result.get("deadline_us")
    iterations = result.get("iterations")
    if (
        result.get("schema_version") != 1 or result.get("status") != "ok"
        or result.get("pipeline") != "resnet10-layer7-cov-to-control-mlp"
        or result.get("transport") != "registered-shared-sysmem-direct-binding"
        or result.get("payload_bytes") != 14_720
        or result.get("payload_shape") != [1, 4, 23, 40]
        or result.get("producer_output_tensor") != "Layer7_cov"
        or result.get("consumer_input_tensor") != "features"
        or result.get("consumer_output_tensor") != "policy_output"
        or isinstance(iterations, bool) or not isinstance(iterations, int)
        or iterations <= 0
        or (expected_requests is not None and iterations != expected_requests)
        or result.get("checksum_failures") != 0
        or not isinstance(deadline, (int, float)) or isinstance(deadline, bool)
        or not math.isfinite(float(deadline)) or float(deadline) <= 0.0
        or not isinstance(orion, dict) or orion.get("enabled") is not True
        or orion.get("status") != 0 or orion.get("measured_background_completed", 0) <= 0
        or orion.get("measured_background_goodput_rps", 0.0) <= 0.0
    ):
        raise ValueError("Orion ResNet control result contract differs")
    best_effort = load_profile_bundle(best_effort_profile_path, best_effort_scheduler_path)
    high_priority = load_profile_bundle(high_priority_profile_path, high_priority_scheduler_path)
    if (best_effort.get("model") != "distilbert-sst2"
            or high_priority.get("model") != "resnet10-detection"
            or high_priority.get("operations_per_inference") != 18):
        raise ValueError("Orion ResNet control profile identity differs")
    pipeline = replay_wall_pipeline(pipeline_path, result, float(deadline))
    checksums = replay_checksums(checksum_path, result)
    events = replay_events(events_path, [best_effort, high_priority], orion["scheduler"])
    return {
        "schema_version": 1,
        "kind": "orion-thor-resnet-control-numeric-smoke-verification",
        "system": "Orion (Thor port)",
        "status": "passed-smoke",
        "scope": "same-workload-smoke-not-formal-statistics",
        "workload": "resnet10-layer7-cov-to-control-mlp",
        "payload_bytes": 14_720,
        "requests": pipeline["requests"],
        "misses": pipeline["misses"],
        "dmr": pipeline["misses"] / pipeline["requests"],
        "p99_us": pipeline["p99_us"],
        "deadline_us": float(deadline),
        "background_goodput_rps": orion["measured_background_goodput_rps"],
        "checksum_failures": 0,
        "unique_payload_checksums": checksums["payloads"],
        "unique_policy_output_checksums": checksums["outputs"],
        "scheduler": {
            "arrivals": orion["scheduler"]["arrivals"],
            "decisions": orion["scheduler"]["decisions"],
            "complementary_admissions": events["complementary"],
            "reordered_decisions": events["reordered"],
            "event_records": events["records"],
        },
        "inputs_sha256": {
            "result": sha256(result_path), "pipeline": sha256(pipeline_path),
            "checksums": sha256(checksum_path), "events": sha256(events_path),
            "best_effort_profile": sha256(best_effort_profile_path),
            "high_priority_profile": sha256(high_priority_profile_path),
            "best_effort_scheduler_profile": sha256(best_effort_scheduler_path),
            "high_priority_scheduler_profile": sha256(high_priority_scheduler_path),
            "binary": sha256(binary_path),
        },
        "token_only": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "pipeline", "checksums", "events",
                 "best-effort-profile", "high-priority-profile",
                 "best-effort-scheduler-profile", "high-priority-scheduler-profile",
                 "binary"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    args = parser.parse_args()
    summary = verify(
        args.result.resolve(), args.pipeline.resolve(), args.checksums.resolve(),
        args.events.resolve(), args.best_effort_profile.resolve(),
        args.high_priority_profile.resolve(), args.best_effort_scheduler_profile.resolve(),
        args.high_priority_scheduler_profile.resolve(), args.binary.resolve(),
        args.expected_requests,
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
