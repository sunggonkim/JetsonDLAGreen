#!/usr/bin/env python3
"""Aggregate a strict production-wall Whisper dependent smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from .p9_frontier_evidence import validate_correctness
except ImportError:
    from p9_frontier_evidence import validate_correctness


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    lock_sha: str | None = None
    contract: tuple[Any, ...] | None = None
    for path in paths:
        value, digest = _read(path)
        if (
            value.get("kind") != "p9-dependent-small-stress-smoke"
            or value.get("workload") != "whisper-projection"
            or value.get("latency_contract") != "production-wall-arrival-to-completion"
            or value.get("deadline_mode") != "wall"
            or value.get("checksum_mode") != "inline"
        ):
            raise ValueError(f"{path} is outside the Whisper wall contract")
        system_rows = value.get("results")
        if not isinstance(system_rows, list) or len(system_rows) != 1 or not isinstance(system_rows[0], dict):
            raise ValueError(f"{path} must contain exactly one result")
        row = system_rows[0]
        system = row.get("system")
        if not isinstance(system, str) or not system:
            raise ValueError(f"{path} has no system label")
        lock = value.get("deadline_lock")
        current_lock = lock.get("sha256") if isinstance(lock, dict) else None
        if not isinstance(current_lock, str) or len(current_lock) != 64:
            raise ValueError(f"{path} lacks deadline-lock provenance")
        if lock_sha is None:
            lock_sha = current_lock
        elif current_lock != lock_sha:
            raise ValueError(f"{path} uses a different deadline lock")
        current = (
            value.get("iterations"), value.get("background_period_ms"),
            _finite(value.get("deadline_us"), "deadline"), value.get("placement_variant"),
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"{path} differs from the common Whisper contract")
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            raise ValueError(f"{path} has invalid request count")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError(f"{path} has invalid miss count")
        if row.get("payload_bytes") != 2_304_000:
            raise ValueError(f"{path} is not the 2.304-MB dependent workload")
        correctness = validate_correctness(value, row, path)
        rows.append({
            "system": system,
            "path": str(path.resolve()),
            "sha256": digest,
            "deadline_lock_sha256": current_lock,
            "requests": requests,
            "deadline_misses": misses,
            "dmr": misses / requests,
            "wall_p99_us": _finite(row.get("wall_pipeline_p99_us", row.get("pipeline_p99_us")), "wall p99"),
            "background_goodput_rps": _finite(row.get("background_goodput_rps"), "goodput"),
            "correctness_evidence": correctness,
        })
    if not rows or lock_sha is None or contract is None:
        raise ValueError("at least one Whisper smoke is required")
    systems = {row["system"] for row in rows}
    if len(systems) != len(rows):
        raise ValueError("each Whisper smoke system must be unique")
    return {
        "schema_version": 1,
        "kind": "p9-whisper-production-wall-smoke",
        "proposed_system": "QUIET",
        "workload": "whisper-projection",
        "payload_bytes": 2_304_000,
        "deadline_us": contract[2],
        "deadline_lock_sha256": lock_sha,
        "placement_variant": contract[3],
        "offered_rps": 1000.0 / float(contract[1]),
        "formal": False,
        "scope": "exploratory-production-wall-dependent-smoke; no-thermal-normalization",
        "rows": sorted(rows, key=lambda row: row["system"]),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
