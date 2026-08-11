#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED = ((25, 2), (50, 4), (75, 6), (100, 8))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def verify(directory: Path) -> dict[str, Any]:
    domain_path = directory / "context-domain.json"
    domain = load(domain_path)
    if (
        domain.get("schema_version") != 1
        or domain.get("kind") != "bless-thor-context-domain"
        or domain.get("device") != "NVIDIA Thor MIG 1g.0gb"
        or domain.get("multiprocessors") != 8
        or domain.get("exec_affinity_supported") is not True
    ):
        raise ValueError("BLESS context domain metadata differs")
    requests = domain.get("requests")
    if not isinstance(requests, list) or len(requests) != 8:
        raise ValueError("BLESS context domain is incomplete")
    actual_domain: set[int] = set()
    for index, request in enumerate(requests, 1):
        if (
            not isinstance(request, dict)
            or request.get("requested_sms") != index
            or request.get("create_result") != "CUDA_SUCCESS"
            or request.get("query_result") != "CUDA_SUCCESS"
            or request.get("destroy_result") != "CUDA_SUCCESS"
            or not isinstance(request.get("create_ns"), int)
            or request["create_ns"] <= 0
        ):
            raise ValueError("BLESS context creation evidence differs")
        actual = request.get("actual_sms")
        if not isinstance(actual, int) or actual <= 0:
            raise ValueError("BLESS actual affinity is invalid")
        actual_domain.add(actual)
    if actual_domain != {2, 4, 6, 8}:
        raise ValueError("BLESS Thor affinity domain differs")

    runs: list[dict[str, Any]] = []
    evidence = {"context-domain.json": sha256(domain_path)}
    for quota, sms in EXPECTED:
        path = directory / f"q{quota}.json"
        stderr_path = directory / f"q{quota}.stderr"
        result = load(path)
        benchmark = result.get("benchmark")
        if (
            result.get("schema_version") != 1
            or result.get("kind") != "bless-thor-trt-affinity-smoke"
            or result.get("requested_sms") != sms
            or result.get("actual_sms") != sms
            or not isinstance(benchmark, dict)
        ):
            raise ValueError(f"BLESS q{quota} wrapper evidence differs")
        environment = benchmark.get("execution_environment")
        gpu = benchmark.get("gpu")
        latency = benchmark.get("release_to_completion")
        engine = benchmark.get("engine")
        if (
            benchmark.get("schema_version") != 1
            or benchmark.get("model") != "distilbert-sst2"
            or benchmark.get("role") != "benchmark"
            or not isinstance(engine, str)
            or f"mig-1g-q{quota}/distilbert-sst2.engine" not in engine
            or benchmark.get("completed_requests") != 20
            or not isinstance(environment, dict)
            or environment.get("mps_active_thread_percentage") != 100
            or not isinstance(gpu, dict)
            or gpu.get("name") != "NVIDIA Thor MIG 1g.0gb"
            or gpu.get("multiprocessors") != 8
            or not isinstance(latency, dict)
        ):
            raise ValueError(f"BLESS q{quota} TensorRT contract differs")
        p99 = latency.get("p99_ms")
        if (
            isinstance(p99, bool)
            or not isinstance(p99, (int, float))
            or not math.isfinite(float(p99))
            or float(p99) <= 0
        ):
            raise ValueError(f"BLESS q{quota} latency is invalid")
        stderr = stderr_path.read_text(encoding="utf-8")
        if "error:" in stderr.lower() or "launch failure" in stderr.lower():
            raise ValueError(f"BLESS q{quota} stderr reports a failure")
        evidence[path.name] = sha256(path)
        evidence[stderr_path.name] = sha256(stderr_path)
        runs.append({"quota": quota, "sms": sms, "p99_ms": float(p99)})

    return {
        "schema_version": 1,
        "kind": "bless-thor-affinity-functional-gate",
        "status": "passed",
        "numeric_comparison_allowed": False,
        "affinity_domain_sms": [2, 4, 6, 8],
        "runs": runs,
        "evidence_sha256": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.result_dir.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
