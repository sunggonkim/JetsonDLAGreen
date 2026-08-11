#!/usr/bin/env python3
"""Verify the gpulet Thor ResNet-control numeric smoke from raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "3c1c2aad3b33edcef20e549d5093c43af497e6ae"
PARTITIONS = ((10, 90), (25, 75), (50, 50), (75, 25), (90, 10))
WORKLOADS = {
    "resnet-control": {
        "pipeline": "resnet10-layer7-cov-to-control-mlp",
        "producer_output": "Layer7_cov",
        "shape": [1, 4, 23, 40],
        "payload_bytes": 14_720,
        "deadline_mode": "wall",
        "trace_latency": "wall_end_to_end_us",
    },
    "whisper-projection": {
        "pipeline": "whisper-last-hidden-state-to-projection-mlp",
        "producer_output": "last_hidden_state",
        "shape": [1, 1500, 384],
        "payload_bytes": 2_304_000,
        "deadline_mode": "validation-excluded",
        "trace_latency": "validation_excluded_end_to_end_us",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def verify_summary(
    summary_path: Path, producer_quota: int, background_quota: int,
    requests: int, deadline_us: float, workload: str,
) -> dict[str, Any]:
    contract = WORKLOADS[workload]
    summary = load(summary_path)
    rows = summary.get("results")
    if (
        summary.get("kind") != "p9-dependent-small-stress-smoke"
        or summary.get("workload") != workload
        or summary.get("deadline_source") != "frozen-independent-pipeline-p99-factor"
        or not math.isclose(float(summary.get("deadline_us", -1)), deadline_us)
        or summary.get("background_offered_rps") != 250.0
        or summary.get("producer_quota_percent") != producer_quota
        or summary.get("background_quota_percent") != background_quota
        or not isinstance(rows, list)
        or len(rows) != 1
    ):
        raise ValueError("gpulet common-workload summary differs")
    row = rows[0]
    if (
        row.get("system") != "Partition-only planning"
        or row.get("pipeline_requests") != requests
        or row.get("deadline_mode") != contract["deadline_mode"]
        or row.get("payload_bytes") != contract["payload_bytes"]
        or row.get("unique_payload_checksums", 0) < 2
        or row.get("unique_policy_output_checksums", 0) < 2
        or row.get("producer_quota_percent") != producer_quota
        or row.get("background_quota_percent") != background_quota
    ):
        raise ValueError("gpulet common-workload row differs")
    scenario = summary_path.parent / "gpulet"
    pipeline_path = scenario / "pipeline.json"
    trace_path = scenario / "pipeline.csv"
    pipeline = load(pipeline_path)
    if (
        pipeline.get("status") != "ok"
        or pipeline.get("pipeline") != contract["pipeline"]
        or pipeline.get("producer_output_tensor") != contract["producer_output"]
        or pipeline.get("consumer_input_tensor") != "features"
        or pipeline.get("payload_shape") != contract["shape"]
        or pipeline.get("payload_bytes") != contract["payload_bytes"]
        or pipeline.get("transport")
        != "registered-shared-sysmem-direct-binding"
        or pipeline.get("iterations") != requests
        or pipeline.get("producer_quota") != producer_quota
        or pipeline.get("deadline_mode") != contract["deadline_mode"]
        or not math.isclose(float(pipeline.get("deadline_us", -1)), deadline_us)
        or pipeline.get("checksum_failures") != 0
        or pipeline.get("unique_payload_checksums", 0) < 2
        or pipeline.get("unique_policy_output_checksums", 0) < 2
    ):
        raise ValueError("gpulet pipeline evidence differs")
    latencies: list[float] = []
    misses = 0
    with trace_path.open(newline="", encoding="utf-8") as source:
        for trace_row in csv.DictReader(source):
            latency = float(trace_row[contract["trace_latency"]])
            miss = int(trace_row["deadline_miss"])
            if not math.isfinite(latency) or latency <= 0 or miss != (latency > deadline_us):
                raise ValueError("gpulet raw request trace differs")
            latencies.append(latency)
            misses += miss
    raw_p99 = percentile(latencies, 0.99)
    if (
        len(latencies) != requests
        or misses != pipeline.get("deadline_misses")
        or misses != row.get("deadline_misses")
        or not math.isclose(raw_p99, float(row.get("pipeline_p99_us", -1)), abs_tol=0.01)
    ):
        raise ValueError("gpulet raw trace summary differs")
    return {
        "requests": requests,
        "misses": misses,
        "p99_us": raw_p99,
        "background_goodput_rps": float(row["background_goodput_rps"]),
        "unique_payload_checksums": pipeline["unique_payload_checksums"],
        "unique_policy_output_checksums": pipeline["unique_policy_output_checksums"],
        "summary_sha256": sha256(summary_path),
        "pipeline_sha256": sha256(pipeline_path),
        "trace_sha256": sha256(trace_path),
    }


def verify(root: Path, expected_requests: int | None = None) -> dict[str, Any]:
    result_path = root / "result.json"
    result = load(result_path)
    upstream = result.get("upstream")
    adaptation = result.get("adaptation")
    profiles = result.get("profiles")
    decisions = result.get("scheduler_decisions")
    if (
        result.get("kind") != "gpulet-thor-dependent-evaluation"
        or not isinstance(upstream, dict)
        or upstream.get("commit") != UPSTREAM_COMMIT
        or upstream.get("venue") != "USENIX ATC 2022"
        or not isinstance(adaptation, dict)
        or adaptation.get("workload") not in WORKLOADS
        or adaptation.get("candidate_partitions") != [list(item) for item in PARTITIONS]
        or adaptation.get("profile_and_evaluation_requests_disjoint") is not True
        or not isinstance(profiles, list)
        or len(profiles) != len(PARTITIONS)
        or not isinstance(decisions, list)
        or len(decisions) != len(PARTITIONS)
    ):
        raise ValueError("gpulet top-level contract differs")
    workload = adaptation["workload"]
    contract = WORKLOADS[workload]
    lock_path = Path(result["deadline_lock"]["path"])
    if sha256(lock_path) != result["deadline_lock"]["sha256"]:
        raise ValueError("gpulet deadline lock hash differs")
    lock = load(lock_path)
    deadline_us = float(lock["deadline_us"])
    if (
        lock.get("contract", {}).get("workload") != workload
        or lock.get("contract", {}).get("deadline_mode") != contract["deadline_mode"]
        or lock.get("contract", {}).get("payload_bytes") != contract["payload_bytes"]
    ):
        raise ValueError("gpulet deadline lock contract differs")
    for profile, decision, partition in zip(profiles, decisions, PARTITIONS, strict=True):
        producer, background = partition
        if (
            (profile.get("producer_quota"), profile.get("background_quota")) != partition
            or (decision.get("producer_quota"), decision.get("background_quota")) != partition
            or sha256(Path(profile["path"])) != profile["sha256"]
        ):
            raise ValueError("gpulet profile provenance differs")
        latency_ok = float(profile["critical_p99_ms"]) * 1000.0 <= deadline_us
        rate_ok = float(profile["background_rps"]) >= 237.5
        if decision != {
            "producer_quota": producer,
            "background_quota": background,
            "latency_ok": latency_ok,
            "background_rate_ok": rate_ok,
            "schedulable": latency_ok and rate_ok,
        }:
            raise ValueError("gpulet scheduler decision differs from profiles")
    feasible = [row for row in decisions if row["schedulable"]]
    if bool(feasible) != result.get("spatial_schedule_feasible"):
        raise ValueError("gpulet feasibility decision differs")
    action = result.get("selected_action")
    if not isinstance(action, dict):
        raise ValueError("gpulet selected action is missing")
    if feasible:
        expected = min(feasible, key=lambda row: row["producer_quota"])
        semantics = "gpulet-best-fit"
    else:
        expected = {"producer_quota": 90, "background_quota": 10}
        semantics = "diagnostic-largest-critical-partition"
    if (
        action.get("producer_quota") != expected["producer_quota"]
        or action.get("background_quota") != expected["background_quota"]
        or action.get("semantics") != semantics
    ):
        raise ValueError("gpulet selected action differs")
    evaluation_path = Path(result["evaluation"]["path"])
    if sha256(evaluation_path) != result["evaluation"]["sha256"]:
        raise ValueError("gpulet evaluation hash differs")
    recorded_requests = adaptation.get("evaluation_iterations")
    if recorded_requests is None:
        recorded_requests = expected_requests if expected_requests is not None else 100
    if (
        isinstance(recorded_requests, bool)
        or not isinstance(recorded_requests, int)
        or recorded_requests <= 0
        or (expected_requests is not None and recorded_requests != expected_requests)
    ):
        raise ValueError("gpulet evaluation request contract differs")
    evaluation = verify_summary(
        evaluation_path, action["producer_quota"], action["background_quota"],
        recorded_requests, deadline_us, workload,
    )
    return {
        "schema_version": 1,
        "kind": "gpulet-thor-dependent-numeric-smoke-verification",
        "system": "gpulet (Thor port)",
        "status": "passed-smoke",
        "upstream_commit": UPSTREAM_COMMIT,
        "workload": contract["pipeline"],
        "payload_bytes": contract["payload_bytes"],
        "deadline_us": deadline_us,
        "spatial_schedule_feasible": bool(feasible),
        "selected_action": action,
        **evaluation,
        "dmr": evaluation["misses"] / evaluation["requests"],
        "checksum_failures": 0,
        "token_only": False,
        "result_sha256": sha256(result_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    args = parser.parse_args(argv)
    value = verify(args.result_dir.resolve(), args.expected_requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
