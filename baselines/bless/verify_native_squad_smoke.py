#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("BLESS squad trace must be nonempty and newline complete")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("BLESS squad trace records must be objects")
    return records


def verify(directory: Path) -> dict[str, Any]:
    result_path = directory / "native-squad.json"
    trace_path = directory / "native-squad.jsonl"
    stderr_path = directory / "native-squad.stderr"
    result = load_object(result_path)
    trace = load_jsonl(trace_path)

    if (
        result.get("schema_version") != 1
        or result.get("kind") != "bless-thor-native-squad-smoke"
        or result.get("algorithm") != "relative-progress-kernel-squads"
        or result.get("maximum_squad_kernels") != 6
        or result.get("restricted_fraction") != 0.5
        or result.get("affinity_domain_sms") != [2, 4, 6, 8]
        or result.get("requests") != 2
        or result.get("kernels_per_request") != 12
        or result.get("status") != "passed"
    ):
        raise ValueError("BLESS native squad summary differs")
    checksums = result.get("checksums")
    expected = result.get("expected_checksums")
    if (
        not isinstance(checksums, list)
        or len(checksums) != 2
        or checksums != expected
        or any(isinstance(value, bool) or not isinstance(value, int) for value in checksums)
    ):
        raise ValueError("BLESS native squad checksum differs")
    if result.get("squads") != len(trace) or not trace:
        raise ValueError("BLESS native squad count differs")

    previous_cursor = [0, 0]
    total_kernels = 0
    estimators: set[str] = set()
    for sequence, record in enumerate(trace):
        if record.get("schema_version") != 1 or record.get("sequence") != sequence:
            raise ValueError("BLESS squad trace sequence differs")
        kernel_count = record.get("kernel_count")
        shares = record.get("shares")
        cursor = record.get("cursor")
        predicted_us = record.get("predicted_us")
        estimator = record.get("estimator")
        if (
            isinstance(kernel_count, bool)
            or not isinstance(kernel_count, int)
            or not 1 <= kernel_count <= 6
            or estimator not in {"interference-free", "workload-equivalence"}
            or not isinstance(shares, list)
            or len(shares) != 2
            or any(share not in {2, 4, 6, 8} for share in shares)
            or not isinstance(cursor, list)
            or len(cursor) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in cursor)
            or any(not old <= new <= 12 for old, new in zip(previous_cursor, cursor))
            or sum(cursor[index] - previous_cursor[index] for index in range(2))
            != kernel_count
            or isinstance(predicted_us, bool)
            or not isinstance(predicted_us, (int, float))
            or not math.isfinite(float(predicted_us))
            or float(predicted_us) <= 0
        ):
            raise ValueError("BLESS squad trace record differs")
        if estimator == "interference-free" and sum(shares) != 8:
            raise ValueError("BLESS strict shares do not fill the 1g affinity domain")
        if estimator == "workload-equivalence" and shares != [8, 8]:
            raise ValueError("BLESS unrestricted shares differ")
        previous_cursor = cursor
        total_kernels += kernel_count
        estimators.add(estimator)

    if previous_cursor != [12, 12] or total_kernels != 24:
        raise ValueError("BLESS native workload did not complete")
    if not estimators:
        raise ValueError("BLESS did not select a squad configuration")
    stderr = stderr_path.read_text(encoding="utf-8")
    if stderr.strip():
        raise ValueError("BLESS native squad stderr is not empty")

    return {
        "schema_version": 1,
        "kind": "bless-thor-native-squad-functional-gate",
        "status": "passed",
        "numeric_comparison_allowed": False,
        "squads": len(trace),
        "kernels": total_kernels,
        "estimators": sorted(estimators),
        "evidence_sha256": {
            result_path.name: sha256(result_path),
            trace_path.name: sha256(trace_path),
            stderr_path.name: sha256(stderr_path),
        },
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
