#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("BLESS squad trace must be nonempty and newline complete")
    records = [json.loads(line) for line in raw.splitlines()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("BLESS squad trace records must be objects")
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory: Path, engine: Path, boundary_lock_path: Path) -> dict[str, Any]:
    result = load_json(directory / "result.json")
    trace = load_jsonl(directory / "squad.jsonl")
    boundary_lock = load_json(boundary_lock_path)
    selected_switch = boundary_lock.get("selected_switch_operation")
    if (
        boundary_lock.get("schema_version") != 1
        or boundary_lock.get("kind") != "bless-thor-tensorrt-safe-boundary-lock"
        or boundary_lock.get("status") != "frozen"
        or boundary_lock.get("held_out_validation_required") is not True
        or not isinstance(selected_switch, int)
        or selected_switch not in boundary_lock.get("safe_switch_operations", [])
        or boundary_lock.get("engine", {}).get("sha256") != sha256(engine)
    ):
        raise ValueError("BLESS TensorRT boundary lock differs")
    expected_launches = result.get("logical_launches")
    checksums = result.get("output_checksums")
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "bless-thor-trt-squad-replica-smoke"
        or result.get("status") != "passed"
        or result.get("affinity_domain_sms") != [2, 4, 6, 8]
        or not isinstance(expected_launches, int)
        or expected_launches <= 0
        or result.get("physical_launches") != expected_launches
        or result.get("shadow_launches") != expected_launches * 3
        or result.get("signature_mismatches") != 0
        or result.get("restricted_launches") != selected_switch
        or result.get("unrestricted_launches") != expected_launches - selected_switch
        or result.get("activation_copies") != 1
        or result.get("last_selected_sms") != 8
        or result.get("selected_output_matches") is not True
        or not isinstance(checksums, list)
        or len(checksums) != 4
        or len(set(checksums)) != 1
        or result.get("selected_output_checksum") != checksums[0]
    ):
        raise ValueError("BLESS TensorRT squad result differs")
    if len(trace) != expected_launches:
        raise ValueError("BLESS TensorRT squad trace count differs")
    previous_end = 0
    for index, record in enumerate(trace):
        expected_sms = 2 if index < selected_switch else 8
        if (
            record.get("schema_version") != 1
            or record.get("operation") != index
            or record.get("selected_sms") != expected_sms
            or record.get("activation_copy") is not (index == selected_switch)
            or record.get("api") not in {"cuLaunchKernel", "cuLaunchKernelEx"}
            or record.get("result") != 0
            or not isinstance(record.get("grid"), list)
            or not isinstance(record.get("block"), list)
            or not isinstance(record.get("start_monotonic_ns"), int)
            or not isinstance(record.get("end_monotonic_ns"), int)
            or record["start_monotonic_ns"] < previous_end
            or record["end_monotonic_ns"] < record["start_monotonic_ns"]
        ):
            raise ValueError("BLESS TensorRT squad trace differs")
        previous_end = record["end_monotonic_ns"]
    stderr = (directory / "stderr.txt").read_text(encoding="utf-8")
    if "error:" in stderr.lower() or "launch failure" in stderr.lower():
        raise ValueError("BLESS TensorRT squad stderr reports a failure")
    return {
        "schema_version": 1,
        "kind": "bless-thor-trt-squad-replica-functional-gate",
        "status": "passed",
        "engine": {"path": str(engine.resolve()), "sha256": sha256(engine)},
        "logical_launches": expected_launches,
        "physical_launches": result["physical_launches"],
        "shadow_launches": result["shadow_launches"],
        "safe_switch_operation": selected_switch,
        "boundary_lock": {
            "path": str(boundary_lock_path),
            "sha256": sha256(boundary_lock_path),
        },
        "activation_copies": result["activation_copies"],
        "output_checksum": result["selected_output_checksum"],
        "numeric_comparison_allowed": False,
        "remaining_gate": (
            "drive safe-boundary selection from an independent profile and run "
            "the BLESS relative-progress squad policy on the common workload"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--boundary-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = verify(
        args.result_dir.resolve(),
        args.engine.resolve(),
        args.boundary_lock.resolve(),
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
