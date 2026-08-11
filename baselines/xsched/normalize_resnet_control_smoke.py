#!/usr/bin/env python3
"""Normalize verified XSched ResNet evidence into the common smoke schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def normalize(verification_path: Path, deadline_lock_path: Path, background_period_ms: float) -> dict[str, Any]:
    value = json.loads(verification_path.read_text(encoding="utf-8"))
    lock_bytes = deadline_lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if (
        value.get("kind") != "xsched-thor-resnet-control-numeric-smoke-verification"
        or value.get("status") != "passed-smoke"
        or value.get("workload") != "resnet10-layer7-cov-to-control-mlp"
        or value.get("deadline_lock", {}).get("sha256")
        != hashlib.sha256(lock_bytes).hexdigest()
    ):
        raise ValueError("XSched verification/lock provenance differs")
    if lock.get("contract", {}).get("workload") != "resnet-control":
        raise ValueError("XSched lock workload differs")
    result = {
        "schema_version": 1,
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-control",
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "deadline_us": value["deadline_us"],
        "iterations": value["requests"],
        "background_period_ms": background_period_ms,
        "background_offered_rps": 1000.0 / background_period_ms,
        "producer_quota_percent": 100,
        "background_quota_percent": 100,
        "latency_contract": "production-wall-arrival-to-completion",
        "deadline_mode": "wall",
        "checksum_mode": "inline",
        "execution_order": ["XSched"],
        "deadline_lock": {
            "path": str(deadline_lock_path.resolve()),
            "sha256": hashlib.sha256(lock_bytes).hexdigest(),
        },
        "application_output_trace": value.get("application_output_trace"),
        "common_workload": value.get("common_workload"),
        "results": [{
            "system": "XSched",
            "placement_variant": "fixed-1g-producer-2g-consumer",
            "pipeline_requests": value["requests"],
            "deadline_misses": value["misses"],
            "pipeline_p99_us": value["p99_us"],
            "wall_pipeline_p99_us": value["p99_us"],
            "deadline_mode": "wall",
            "latency_contract": "production-wall-arrival-to-completion",
            "checksum_mode": "inline",
            "correctness_validated": value["checksum_failures"] == 0,
            "background_goodput_rps": value["background_goodput_rps"],
            "best_effort_admitted": True,
            "best_effort_status": "completed",
            "gate_p99_us": None,
            "gate_scope": None,
            "producer_quota_percent": 100,
            "background_quota_percent": 100,
            "sota_verification": {
                "path": str(verification_path.resolve()),
                "sha256": hashlib.sha256(verification_path.read_bytes()).hexdigest(),
                "scheduler": value.get("scheduler"),
                "application_output_trace": value.get("application_output_trace"),
            },
        }],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--background-period-ms", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.background_period_ms <= 0:
        raise ValueError("background period must be positive")
    value = normalize(args.verification, args.deadline_lock, args.background_period_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
