#!/usr/bin/env python3
"""Summarize the BLESS TensorRT ResNet compatibility boundary on Thor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    q100_engine: Path,
    q100_failure_stderr: Path,
    boundary_lock_path: Path,
    heldout_path: Path,
    schedule_path: Path,
) -> dict[str, Any]:
    stderr = q100_failure_stderr.read_text(encoding="utf-8")
    boundary = load(boundary_lock_path)
    heldout = load(heldout_path)
    schedule = load(schedule_path)
    if "Myelin" not in stderr or "kErrorCuda" not in stderr:
        raise ValueError("q100 restricted-context failure signature differs")
    if (
        boundary.get("kind") != "bless-thor-tensorrt-safe-boundary-lock"
        or boundary.get("status") != "frozen"
        or boundary.get("total_logical_launches") != 18
        or boundary.get("safe_switch_operations") != [0, 6, 9, 15, 18]
        or boundary.get("selected_switch_operation") != 9
        or len(boundary.get("affinity_engines", [])) != 4
        or len({item.get("sha256") for item in boundary["affinity_engines"]}) != 1
    ):
        raise ValueError("ResNet q25 boundary lock differs")
    q25_sha = boundary["engine"]["sha256"]
    if q25_sha == sha256(q100_engine):
        raise ValueError("q25 replica plan unexpectedly equals the common q100 plan")
    if (
        heldout.get("kind") != "bless-thor-trt-squad-replica-functional-gate"
        or heldout.get("status") != "passed"
        or heldout.get("logical_launches") != 18
        or heldout.get("physical_launches") != 18
        or heldout.get("shadow_launches") != 54
        or heldout.get("safe_switch_operation") != 9
        or heldout.get("activation_copies") != 1
        or heldout.get("engine", {}).get("sha256") != q25_sha
        or heldout.get("boundary_lock", {}).get("sha256")
        != sha256(boundary_lock_path)
    ):
        raise ValueError("ResNet q25 held-out gate differs")
    if (
        schedule.get("kind")
        != "bless-thor-common-tensorrt-profile-and-first-squad"
        or schedule.get("status") != "profiled"
        or schedule.get("models", {}).get("resnet", {}).get("logical_launches")
        != 18
        or schedule.get("models", {}).get("distilbert", {}).get(
            "logical_launches"
        )
        != 47
        or schedule.get("numeric_comparison_allowed") is not False
    ):
        raise ValueError("BLESS common profile differs")
    return {
        "schema_version": 1,
        "kind": "bless-thor-resnet-tensorrt-compatibility-boundary",
        "status": "structural-incompatibility-characterized",
        "numeric_comparison_allowed": False,
        "common_q100_plan": {
            "path": str(q100_engine),
            "sha256": sha256(q100_engine),
            "restricted_replica_result": "failed",
            "failure": "TensorRT Myelin kErrorCuda in 2-SM execution-affinity context",
        },
        "executable_q25_replica_plan": {
            "sha256": q25_sha,
            "logical_launches": 18,
            "safe_switch_operations": [0, 6, 9, 15, 18],
            "selected_switch_operation": 9,
            "held_out_physical_launches": 18,
            "held_out_shadow_launches": 54,
            "held_out_activation_copies": 1,
        },
        "common_profile": {
            "resnet_launches": 18,
            "distilbert_launches": 47,
            "first_squad": schedule["squad"],
            "first_configuration": schedule["configuration"],
        },
        "exclusion_reason": (
            "BLESS cannot consume the same q100 TensorRT producer plan used by "
            "the common campaign in its required restricted context; substituting "
            "the q25 plan would change the executable workload"
        ),
        "inputs": {
            "q100_failure_stderr": {
                "path": str(q100_failure_stderr),
                "sha256": sha256(q100_failure_stderr),
            },
            "boundary_lock": {
                "path": str(boundary_lock_path),
                "sha256": sha256(boundary_lock_path),
            },
            "heldout": {"path": str(heldout_path), "sha256": sha256(heldout_path)},
            "schedule": {"path": str(schedule_path), "sha256": sha256(schedule_path)},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q100-engine", type=Path, required=True)
    parser.add_argument("--q100-failure-stderr", type=Path, required=True)
    parser.add_argument("--boundary-lock", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.q100_engine.resolve(),
        args.q100_failure_stderr.resolve(),
        args.boundary_lock.resolve(),
        args.heldout.resolve(),
        args.schedule.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
