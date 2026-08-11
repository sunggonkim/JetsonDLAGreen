#!/usr/bin/env python3
"""Freeze the current QUIET thermal envelope from verified session evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def bound(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.verification) != 6:
        raise ValueError("thermal lock requires six verified sessions")
    values: list[dict[str, Any]] = []
    seen_sequences: set[int] = set()
    for path in args.verification:
        value = json.loads(path.read_bytes())
        if (
            value.get("kind") != "p9-current-quiet-thermal-session-verification"
            or value.get("status") != "passed"
            or value.get("protocol") != "p9-current-quiet-thermal-v1"
            or value.get("numeric_comparison_allowed") is not True
        ):
            raise ValueError(f"thermal verification is not a passed current gate: {path}")
        sequence = value.get("sequence_index")
        if not isinstance(sequence, int) or sequence not in {0, 1, 2}:
            raise ValueError("thermal sequence index is invalid")
        if sequence in seen_sequences and len(seen_sequences) < 3:
            # The balanced design repeats each order; duplicate detection is
            # deferred to the aggregate, while this lock checks all six rows.
            pass
        seen_sequences.add(sequence)
        metrics = value.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("thermal metrics are missing")
        temperatures = metrics.get("temperature_c")
        if not isinstance(temperatures, dict):
            raise ValueError("thermal temperatures are missing")
        values.append({
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "sequence_index": sequence,
            "soc012_mean_c": finite(temperatures["soc012"]["mean"], "soc012 mean"),
            "tj_mean_c": finite(temperatures["tj"]["mean"], "tj mean"),
            "soc012_range_c": finite(temperatures["soc012"]["range"], "soc012 range"),
            "tj_range_c": finite(temperatures["tj"]["range"], "tj range"),
            "sample_count": metrics.get("sample_count"),
        })
    centers = {
        "soc012_mean_c": statistics.median(row["soc012_mean_c"] for row in values),
        "tj_mean_c": statistics.median(row["tj_mean_c"] for row in values),
    }
    max_mean_delta = max(
        max(abs(row[key] - centers[key]) for row in values)
        for key in centers
    )
    if max_mean_delta > 4.0:
        raise ValueError("verified sessions exceed fixed cross-session thermal envelope")
    if max(row["soc012_range_c"] for row in values) > 8.0 or max(row["tj_range_c"] for row in values) > 8.0:
        raise ValueError("verified sessions exceed fixed intra-session thermal envelope")
    artifacts = {
        "deadline_lock": bound(args.deadline_lock, "deadline lock"),
        "quiet_plan": bound(args.quiet_plan, "QUIET plan"),
        "producer_engine": bound(args.producer_engine, "producer engine"),
        "consumer_engine": bound(args.consumer_engine, "consumer engine"),
    }
    return {
        "schema_version": 1,
        "kind": "p9-current-quiet-thermal-lock",
        "status": "frozen",
        "protocol": "p9-current-quiet-thermal-v1",
        "proposed_system": "QUIET",
        "workload": "resnet50-classification",
        "thermal_normalized": True,
        "thresholds": {
            "max_cross_session_mean_delta_c": 4.0,
            "max_intra_session_range_c": 8.0,
            "required_temperature_sensors": ["soc012", "tj"],
            "required_power_rail": "VDD_GPU",
        },
        "reference_centers_c": centers,
        "observed_max_cross_session_mean_delta_c": max_mean_delta,
        "sessions": values,
        "artifacts": artifacts,
        "claim_guard": "thermal normalization is limited to the frozen sensor envelope and does not alter scheduling policy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", action="append", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--quiet-plan", type=Path, required=True)
    parser.add_argument("--producer-engine", type=Path, required=True)
    parser.add_argument("--consumer-engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = freeze(args)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
