#!/usr/bin/env python3
"""Validate the TensorRT driver-launch surface required by a native Orion port."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


DRIVER_APIS = {
    "cuLaunchKernel",
    "cuLaunchKernel_ptsz",
    "cuLaunchKernelEx",
    "cuLaunchKernelEx_ptsz",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(
    trace: Path,
    benchmark: Path | None = None,
    expected_requests: int | None = None,
    expected_mig_uuid: str | None = None,
) -> dict[str, object]:
    raw = trace.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("driver trace is empty or truncated")
    records = [json.loads(line) for line in raw.splitlines()]
    counts: Counter[str] = Counter()
    functions: set[str] = set()
    previous_end = -1
    for expected, record in enumerate(records):
        if set(record) != {
            "schema_version", "sequence", "api", "tid", "start_monotonic_ns",
            "end_monotonic_ns", "function", "stream", "grid", "block",
            "shared_mem_bytes", "attributes", "result",
        }:
            raise ValueError("driver trace record schema differs")
        if record["schema_version"] != 1 or record["sequence"] != expected:
            raise ValueError("driver trace sequence differs")
        api = record["api"]
        if api not in DRIVER_APIS or record["result"] != 0:
            raise ValueError("driver launch failed or has an unknown API")
        start = record["start_monotonic_ns"]
        end = record["end_monotonic_ns"]
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise ValueError("driver launch clocks are invalid")
        if start < previous_end:
            raise ValueError("driver launch records regress")
        previous_end = end
        if any(not isinstance(value, int) or value <= 0 for value in record["grid"]):
            raise ValueError("driver launch grid is invalid")
        if any(not isinstance(value, int) or value <= 0 for value in record["block"]):
            raise ValueError("driver launch block is invalid")
        counts[api] += 1
        functions.add(record["function"])
    ex_launches = counts["cuLaunchKernelEx"] + counts["cuLaunchKernelEx_ptsz"]
    if ex_launches == 0:
        raise ValueError("TensorRT did not expose cuLaunchKernelEx to the capture layer")
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "orion-tensorrt-driver-capture-positive-control",
        "status": "captured",
        "numeric_comparison_allowed": False,
        "records": len(records),
        "unique_function_handles": len(functions),
        "api_counts": dict(sorted(counts.items())),
        "trace_sha256": sha256(trace),
        "next_gate": "connect captured operations to Orion software queues and scheduler",
    }
    if benchmark is not None:
        summary = json.loads(benchmark.read_text(encoding="utf-8"))
        if summary.get("schema_version") != 1 or summary.get("model") != "resnet10-detection":
            raise ValueError("capture positive control used the wrong benchmark")
        if expected_requests is None or summary.get("completed_requests") != expected_requests:
            raise ValueError("capture positive control request count differs")
        environment = summary.get("execution_environment", {})
        if expected_mig_uuid is None or environment.get("cuda_visible_devices") != expected_mig_uuid:
            raise ValueError("capture positive control MIG UUID differs")
        if summary.get("gpu", {}).get("multiprocessors") != 12:
            raise ValueError("capture positive control did not run on the 2g instance")
        result["benchmark"] = {
            "sha256": sha256(benchmark),
            "completed_requests": summary["completed_requests"],
            "mig_uuid": environment["cuda_visible_devices"],
            "multiprocessors": summary["gpu"]["multiprocessors"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--expected-requests", type=int)
    parser.add_argument("--expected-mig-uuid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.benchmark is None) != (args.expected_requests is None):
        parser.error("--benchmark and --expected-requests must be supplied together")
    result = verify(
        args.trace.resolve(),
        args.benchmark.resolve() if args.benchmark is not None else None,
        args.expected_requests,
        args.expected_mig_uuid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
