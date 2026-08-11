#!/usr/bin/env python3
"""Compare two production-wall dependent-pipeline smoke summaries.

This is a fast validation tool, not a formal analyzer. It rejects
incomparable summaries before reporting a paired signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path, system: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "p9-dependent-small-stress-smoke":
        raise ValueError(f"{path}: unsupported summary kind")
    results = value.get("results")
    if not isinstance(results, list) or not results or not all(isinstance(item, dict) for item in results):
        raise ValueError(f"{path}: expected result rows")
    if system is not None:
        matches = [item for item in results if item.get("system") == system]
        if len(matches) != 1:
            raise ValueError(f"{path}: expected exactly one {system!r} result row")
        row = matches[0]
    elif len(results) == 1:
        row = results[0]
    else:
        raise ValueError(f"{path}: multiple result rows require a system selector")
    required = ("system", "pipeline_requests", "deadline_misses", "pipeline_p99_us",
                "background_goodput_rps", "deadline_mode", "latency_contract",
                "checksum_mode")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{path}: missing row fields {missing}")
    # Keep the selected row at index zero so the rest of the comparison and
    # the emitted report use the exact same runner evidence.
    value = dict(value)
    value["results"] = [row]
    return value


def _lock_sha(value: dict[str, Any]) -> str | None:
    lock = value.get("deadline_lock")
    return lock.get("sha256") if isinstance(lock, dict) else None


def compare(
    left_path: Path,
    right_path: Path,
    left_system: str | None = None,
    right_system: str | None = None,
) -> dict[str, Any]:
    left, right = _load(left_path, left_system), _load(right_path, right_system)
    left_row, right_row = left["results"][0], right["results"][0]
    contracts = {
        "workload": (left.get("workload"), right.get("workload")),
        "deadline_mode": (left.get("deadline_mode"), right.get("deadline_mode")),
        "latency_contract": (left.get("latency_contract"), right.get("latency_contract")),
        "checksum_mode": (left.get("checksum_mode"), right.get("checksum_mode")),
        "deadline_us": (left.get("deadline_us"), right.get("deadline_us")),
        "deadline_lock_sha256": (_lock_sha(left), _lock_sha(right)),
        "background_period_ms": (left.get("background_period_ms"), right.get("background_period_ms")),
        "background_offered_rps": (left.get("background_offered_rps"), right.get("background_offered_rps")),
        "iterations": (left.get("iterations"), right.get("iterations")),
    }
    mismatches = {key: values for key, values in contracts.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"incomparable wall smokes: {mismatches}")
    if left_row["system"] == right_row["system"]:
        raise ValueError("paired summaries must contain different systems")
    for row in (left_row, right_row):
        if row["latency_contract"] != "production-wall-arrival-to-completion":
            raise ValueError("headline comparison requires production wall latency")
        if row["checksum_mode"] == "off":
            raise ValueError("headline comparison requires correctness-enabled checksum mode")

    quiet = next((row for row in (left_row, right_row) if row["system"] == "QUIET"), None)
    baseline = None
    if quiet is not None:
        baseline = right_row if quiet is left_row else left_row
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "p9-paired-wall-smoke-comparison",
        "formal": False,
        "contract": {key: values[0] for key, values in contracts.items()},
        "systems": [left_row["system"], right_row["system"]],
        "rows": [
            {
                "system": row["system"],
                "requests": row["pipeline_requests"],
                "deadline_misses": row["deadline_misses"],
                "p99_us": row["pipeline_p99_us"],
                "background_goodput_rps": row["background_goodput_rps"],
                "correctness_validated": row.get("correctness_validated"),
            }
            for row in (left_row, right_row)
        ],
    }
    if quiet is not None and baseline is not None:
        report["quiet_delta_vs_baseline"] = {
            "baseline": baseline["system"],
            "p99_us_delta": quiet["pipeline_p99_us"] - baseline["pipeline_p99_us"],
            "deadline_miss_delta": quiet["deadline_misses"] - baseline["deadline_misses"],
            "background_goodput_rps_delta": quiet["background_goodput_rps"] - baseline["background_goodput_rps"],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-system")
    parser.add_argument("--right-system")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.left, args.right, args.left_system, args.right_system)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
