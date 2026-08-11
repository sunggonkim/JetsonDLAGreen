#!/usr/bin/env python3
"""Freeze correctness-preserving TensorRT context-switch boundaries for BLESS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def replay_switch(
    directory: Path, operation: int, total_launches: int
) -> dict[str, Any]:
    result_path = directory / f"op-{operation}" / "result.json"
    trace_path = directory / f"op-{operation}" / "squad.jsonl"
    result = load_object(result_path)
    raw = trace_path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"switch {operation} trace is empty or truncated")
    trace = [json.loads(line) for line in raw.splitlines()]
    expected_copies = int(0 < operation < total_launches)
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "bless-thor-trt-squad-replica-smoke"
        or result.get("status") != "passed"
        or result.get("logical_launches") != total_launches
        or result.get("physical_launches") != total_launches
        or result.get("shadow_launches") != total_launches * 3
        or result.get("restricted_launches") != operation
        or result.get("unrestricted_launches") != total_launches - operation
        or result.get("activation_copies") != expected_copies
        or result.get("signature_mismatches") != 0
        or not isinstance(result.get("selected_output_matches"), bool)
        or len(trace) != total_launches
    ):
        raise ValueError(f"switch {operation} result differs")
    for index, record in enumerate(trace):
        expected_sms = 2 if index < operation else 8
        if (
            not isinstance(record, dict)
            or record.get("operation") != index
            or record.get("selected_sms") != expected_sms
            or record.get("activation_copy") is not (
                expected_copies == 1 and index == operation
            )
            or record.get("result") != 0
        ):
            raise ValueError(f"switch {operation} trace differs")
    return {
        "operation": operation,
        "output_matches": result["selected_output_matches"],
        "result_sha256": sha256(result_path),
        "trace_sha256": sha256(trace_path),
    }


def build_lock(
    directory: Path, engines: Path | list[Path], source_root: Path
) -> dict[str, Any]:
    engine_paths = [engines] if isinstance(engines, Path) else list(engines)
    if len(engine_paths) not in {1, 4}:
        raise ValueError("BLESS boundary profile requires one or four engines")
    first = load_object(directory / "op-0" / "result.json")
    total_launches = first.get("logical_launches")
    if not isinstance(total_launches, int) or isinstance(total_launches, bool):
        raise ValueError("BLESS boundary profile launch count is not an integer")
    if total_launches <= 1:
        raise ValueError("BLESS boundary profile needs at least two launches")
    expected_directories = {f"op-{operation}" for operation in range(total_launches + 1)}
    observed_directories = {path.name for path in directory.iterdir() if path.is_dir()}
    if observed_directories != expected_directories:
        raise ValueError("BLESS boundary profile operation set differs")
    evidence = [
        replay_switch(directory, operation, total_launches)
        for operation in range(total_launches + 1)
    ]
    safe = [item["operation"] for item in evidence if item["output_matches"]]
    interior = [operation for operation in safe if 0 < operation < total_launches]
    if not interior or 0 not in safe or total_launches not in safe:
        raise ValueError("BLESS boundary profile lacks fixed-context controls")
    midpoint = total_launches / 2.0
    selected = min(interior, key=lambda operation: (abs(operation - midpoint), operation))
    implementation_paths = [
        source_root / "baselines/bless/trt_squad_intercept.cpp",
        source_root / "baselines/bless/trt_squad_intercept.hpp",
        source_root / "baselines/bless/trt_activation_replica_smoke.cpp",
        source_root / "analysis/freeze_bless_trt_safe_boundaries.py",
        source_root / "scripts/run_p9_bless_trt_safe_boundary_profile.sh",
    ]
    return {
        "schema_version": 1,
        "kind": "bless-thor-tensorrt-safe-boundary-lock",
        "status": "frozen",
        "selection_rule": "correct-interior-boundary-nearest-launch-midpoint",
        "total_logical_launches": total_launches,
        "safe_switch_operations": safe,
        "selected_switch_operation": selected,
        "unsafe_switch_operations": [
            item["operation"] for item in evidence if not item["output_matches"]
        ],
        "held_out_validation_required": True,
        "engine": {
            "path": str(engine_paths[-1]), "sha256": sha256(engine_paths[-1])
        },
        "affinity_engines": [
            {"sms": sms, "path": str(path), "sha256": sha256(path)}
            for sms, path in zip(
                ([8] if len(engine_paths) == 1 else [2, 4, 6, 8]), engine_paths
            )
        ],
        "profile_directory": str(directory),
        "profile_evidence": evidence,
        "implementation_sha256": {
            str(path.relative_to(source_root)): sha256(path)
            for path in implementation_paths
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = build_lock(
        args.profile_dir.resolve(),
        [path.resolve() for path in args.engine],
        args.source_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
