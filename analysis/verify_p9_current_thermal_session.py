#!/usr/bin/env python3
"""Verify one current ImageNette production-wall session's thermal stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.tegrastats_telemetry import parse_tegrastats_line


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def require_file(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verify(args: argparse.Namespace) -> dict[str, Any]:
    session = json.loads(args.session_summary.read_bytes())
    if (
        session.get("kind") != "p9-current-quiet-thermal-session"
        or session.get("protocol") != "p9-current-quiet-thermal-v1"
        or session.get("return_code") != 0
        or session.get("requests") != 1100
    ):
        raise ValueError("thermal launcher session contract differs")
    lock = json.loads(args.deadline_lock.read_bytes())
    plan = json.loads(args.quiet_plan.read_bytes())
    if lock.get("kind") != "p9-dependent-pipeline-deadline-lock":
        raise ValueError("thermal session deadline lock kind differs")
    if plan.get("proposed_system") != "QUIET" or plan.get("status") != "selected":
        raise ValueError("thermal session QUIET plan kind differs")

    records = [
        json.loads(line)
        for line in args.telemetry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    markers = [record for record in records if record.get("record_type") == "marker"]
    samples = [record for record in records if record.get("record_type") == "sample"]
    by_name: dict[str, dict[str, Any]] = {}
    for marker in markers:
        name = marker.get("name")
        if not isinstance(name, str) or name in by_name:
            raise ValueError("thermal markers are missing or duplicated")
        by_name[name] = marker
    names = ("thermal_prepare", "execution_start", "execution_end", "thermal_end")
    if set(by_name) != set(names):
        raise ValueError("thermal marker set differs")
    timestamps = [by_name[name]["monotonic_ns"] for name in names]
    if any(
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int)
        for timestamp in timestamps
    ) or timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError("thermal marker timestamps are not strictly ordered")
    start_ns, end_ns = timestamps[1], timestamps[2]
    execution_samples = [
        record
        for record in samples
        if start_ns <= record.get("monotonic_ns", -1) <= end_ns
    ]
    if len(execution_samples) < 20:
        raise ValueError("thermal execution interval has too few samples")
    temperatures: dict[str, list[float]] = {"soc012": [], "tj": []}
    gpu_power: list[float] = []
    emc_utilization: list[float] = []
    gr3d_utilization: list[float] = []
    for index, record in enumerate(execution_samples):
        raw = record.get("raw")
        if not isinstance(raw, str):
            raise ValueError(f"thermal sample {index} lacks raw line")
        parsed = parse_tegrastats_line(raw)
        normalized_parsed = json.loads(json.dumps(parsed.to_dict()))
        if record.get("parsed") != normalized_parsed:
            raise ValueError(f"thermal sample {index} parsed payload differs from raw line")
        temperatures["soc012"].append(finite(parsed.temperatures_c["soc012"], "soc012"))
        temperatures["tj"].append(finite(parsed.temperatures_c["tj"], "tj"))
        gpu_power.append(finite(parsed.power["VDD_GPU"].current_mw, "VDD_GPU"))
        if parsed.emc is not None and parsed.emc.utilization_pct is not None:
            emc_utilization.append(parsed.emc.utilization_pct)
        if parsed.gr3d is not None and parsed.gr3d.utilization_pct is not None:
            gr3d_utilization.append(parsed.gr3d.utilization_pct)

    metrics = {
        "sample_count": len(execution_samples),
        "duration_s": (end_ns - start_ns) / 1e9,
        "temperature_c": {
            name: {
                "mean": statistics.fmean(values),
                "p50": percentile(values, 0.50),
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
            }
            for name, values in temperatures.items()
        },
        "power_mw": {
            "VDD_GPU": {
                "mean": statistics.fmean(gpu_power),
                "p50": percentile(gpu_power, 0.50),
                "max": max(gpu_power),
            }
        },
        "emc_utilization_pct": (
            statistics.fmean(emc_utilization) if emc_utilization else None
        ),
        "gr3d_utilization_pct": (
            statistics.fmean(gr3d_utilization) if gr3d_utilization else None
        ),
    }
    condition = {
        "required_fields": ["temperature:soc012", "temperature:tj", "power:VDD_GPU"],
        "optional_fields": ["emc_utilization", "gr3d_utilization"],
        "max_intra_session_temperature_range_c": 8.0,
        "passed": all(item["range"] <= 8.0 for item in metrics["temperature_c"].values()),
    }
    if not condition["passed"]:
        raise ValueError("thermal execution temperature range exceeds frozen protocol")
    bound = {
        "session_summary": require_file(args.session_summary, "thermal session summary"),
        "telemetry": require_file(args.telemetry, "thermal telemetry"),
        "deadline_lock": require_file(args.deadline_lock, "thermal deadline lock"),
        "quiet_plan": require_file(args.quiet_plan, "thermal QUIET plan"),
    }
    return {
        "schema_version": 1,
        "kind": "p9-current-quiet-thermal-session-verification",
        "status": "passed",
        "numeric_comparison_allowed": True,
        "protocol": "p9-current-quiet-thermal-v1",
        "sequence_index": session.get("sequence_index"),
        "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
        "deadline_us": finite(lock["deadline_us"], "deadline lock deadline"),
        "deadline_lock_sha256": sha256(args.deadline_lock),
        "quiet_plan_sha256": sha256(args.quiet_plan),
        "markers": {name: by_name[name]["monotonic_ns"] for name in names},
        "metrics": metrics,
        "thermal_condition": condition,
        "evidence": bound,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-summary", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--quiet-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
