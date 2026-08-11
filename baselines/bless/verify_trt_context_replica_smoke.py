#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SMS = (2, 4, 6, 8)
DRIVER_APIS = {
    "cuLaunchKernel",
    "cuLaunchKernel_ptsz",
    "cuLaunchKernelEx",
    "cuLaunchKernelEx_ptsz",
}
DRIVER_KEYS = {
    "schema_version", "sequence", "api", "tid", "start_monotonic_ns",
    "end_monotonic_ns", "function", "stream", "grid", "block",
    "shared_mem_bytes", "attributes", "result",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def verify(directory: Path, engine: Path) -> dict[str, Any]:
    result_path = directory / "result.json"
    stderr_path = directory / "stderr.txt"
    result = load(result_path)
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "bless-thor-trt-context-replica-smoke"
        or result.get("replica_rounds") != 2
    ):
        raise ValueError("BLESS TensorRT replica wrapper differs")
    replicas = result.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 8:
        raise ValueError("BLESS TensorRT replica count differs")

    contexts: dict[int, int] = {}
    p99_ms: dict[int, list[float]] = {sms: [] for sms in EXPECTED_SMS}
    windows: dict[tuple[int, int], tuple[int, int]] = {}
    seen: set[tuple[int, int]] = set()
    for replica in replicas:
        if not isinstance(replica, dict):
            raise ValueError("BLESS TensorRT replica must be an object")
        round_index = replica.get("round")
        requested = replica.get("requested_sms")
        actual = replica.get("actual_sms")
        context_id = replica.get("context_id")
        benchmark = replica.get("benchmark")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index not in (0, 1)
            or requested not in EXPECTED_SMS
            or actual != requested
            or isinstance(context_id, bool)
            or not isinstance(context_id, int)
            or context_id <= 0
            or not isinstance(benchmark, dict)
        ):
            raise ValueError("BLESS TensorRT replica identity differs")
        identity = (round_index, requested)
        if identity in seen:
            raise ValueError("BLESS TensorRT replica identity is duplicated")
        seen.add(identity)
        previous = contexts.setdefault(requested, context_id)
        if previous != context_id:
            raise ValueError("BLESS affinity context was not reused")
        environment = benchmark.get("execution_environment")
        gpu = benchmark.get("gpu")
        config = benchmark.get("config")
        latency = benchmark.get("release_to_completion")
        if (
            benchmark.get("schema_version") != 1
            or benchmark.get("model") != "distilbert-sst2"
            or benchmark.get("role") != "benchmark"
            or benchmark.get("engine") != str(engine)
            or benchmark.get("completed_requests") != 20
            or not isinstance(environment, dict)
            or environment.get("mps_active_thread_percentage") != 100
            or not isinstance(gpu, dict)
            or gpu.get("name") != "NVIDIA Thor MIG 1g.0gb"
            or gpu.get("multiprocessors") != 8
            or not isinstance(config, dict)
            or config.get("include_transfers") is not True
            or not isinstance(latency, dict)
        ):
            raise ValueError("BLESS TensorRT replica benchmark differs")
        measurement_start = benchmark.get("measurement_start_monotonic_ns")
        measurement_end = benchmark.get("measurement_end_monotonic_ns")
        if (
            isinstance(measurement_start, bool)
            or not isinstance(measurement_start, int)
            or isinstance(measurement_end, bool)
            or not isinstance(measurement_end, int)
            or measurement_end <= measurement_start
        ):
            raise ValueError("BLESS TensorRT replica measurement clock differs")
        windows[identity] = (measurement_start, measurement_end)
        p99_ms[requested].append(
            positive_finite(latency.get("p99_ms"), "BLESS replica p99")
        )
    if seen != {(round_index, sms) for round_index in (0, 1) for sms in EXPECTED_SMS}:
        raise ValueError("BLESS TensorRT replica matrix is incomplete")
    if len(set(contexts.values())) != len(EXPECTED_SMS):
        raise ValueError("BLESS affinity contexts are not distinct")

    trace_path = directory / "driver-launches.jsonl"
    raw_trace = trace_path.read_bytes()
    if not raw_trace or not raw_trace.endswith(b"\n"):
        raise ValueError("BLESS TensorRT driver trace is empty or truncated")
    launches_by_replica: Counter[tuple[int, int]] = Counter()
    signatures_by_replica: dict[tuple[int, int], list[tuple[Any, ...]]] = {
        identity: [] for identity in windows
    }
    api_counts: Counter[str] = Counter()
    previous_end = -1
    trace_records = raw_trace.splitlines()
    for sequence, line in enumerate(trace_records):
        record = json.loads(line)
        if (
            not isinstance(record, dict)
            or set(record) != DRIVER_KEYS
            or record.get("schema_version") != 1
            or record.get("sequence") != sequence
            or record.get("api") not in DRIVER_APIS
            or record.get("result") != 0
        ):
            raise ValueError("BLESS TensorRT driver launch differs")
        start = record.get("start_monotonic_ns")
        end = record.get("end_monotonic_ns")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end < start
            or start < previous_end
        ):
            raise ValueError("BLESS TensorRT driver clocks differ")
        previous_end = end
        api_counts[record["api"]] += 1
        matches = [
            identity
            for identity, (window_start, window_end) in windows.items()
            if window_start <= start <= end <= window_end
        ]
        if len(matches) > 1:
            raise ValueError("BLESS TensorRT driver launch matches multiple replicas")
        if matches:
            launches_by_replica[matches[0]] += 1
            signatures_by_replica[matches[0]].append(
                (
                    record["api"],
                    tuple(record["grid"]),
                    tuple(record["block"]),
                    record["shared_mem_bytes"],
                    record["attributes"],
                )
            )
    if any(launches_by_replica[identity] <= 0 for identity in windows):
        raise ValueError("BLESS TensorRT replica lacks measured driver launches")
    if api_counts["cuLaunchKernelEx"] + api_counts["cuLaunchKernelEx_ptsz"] <= 0:
        raise ValueError("BLESS TensorRT trace lacks extended driver launches")
    canonical_identity = (0, EXPECTED_SMS[0])
    canonical_signatures = signatures_by_replica[canonical_identity]
    if any(
        signatures != canonical_signatures
        for signatures in signatures_by_replica.values()
    ):
        raise ValueError("BLESS TensorRT replica launch sequences differ")

    stderr = stderr_path.read_text(encoding="utf-8")
    lowered = stderr.lower()
    if (
        "error:" in lowered
        or "launch failure" in lowered
        or "launchfailure" in lowered
        or "check(" in lowered
    ):
        raise ValueError("BLESS TensorRT replica stderr reports a failure")
    return {
        "schema_version": 1,
        "kind": "bless-thor-trt-context-replica-functional-gate",
        "status": "passed",
        "numeric_comparison_allowed": False,
        "engine": {"path": str(engine), "sha256": sha256(engine)},
        "affinity_domain_sms": list(EXPECTED_SMS),
        "replica_rounds": 2,
        "contexts_precreated_and_reused": True,
        "driver_launch_records": len(trace_records),
        "driver_api_counts": dict(sorted(api_counts.items())),
        "launches_per_replica": len(canonical_signatures),
        "replica_launch_sequences_identical": True,
        "runs": [
            {"sms": sms, "context_id": contexts[sms], "p99_ms": p99_ms[sms]}
            for sms in EXPECTED_SMS
        ],
        "evidence_sha256": {
            "result.json": sha256(result_path),
            "stderr.txt": sha256(stderr_path),
            "driver-launches.jsonl": sha256(trace_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.result_dir.resolve(), args.engine.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
