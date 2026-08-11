#!/usr/bin/env python3
"""Summarize common production-wall smoke points into an exploratory frontier.

This intentionally does not compute confidence intervals or thermal-normalized
claims.  It only prevents cross-system/load cherry-picking by requiring one
common contract per point and reporting every supplied point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SYSTEMS = {"NVIDIA MPS", "XSched", "QUIET"}
ROOT = Path(__file__).resolve().parents[1]


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0.0):
        raise ValueError(f"{label} must be finite")
    return value


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def _manifest() -> tuple[dict[str, Any], str]:
    path = ROOT / "docs" / "p9-comparator-manifest.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    if (
        not raw.endswith(b"\n")
        or not isinstance(value, dict)
        or value.get("proposed_system") != "QUIET"
        or not isinstance(value.get("rows"), dict)
    ):
        raise ValueError("comparator manifest is malformed")
    return value, hashlib.sha256(raw).hexdigest()


def summarize(paths: list[Path], *, expected_systems: set[str] = SYSTEMS) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one smoke summary is required")
    manifest, manifest_sha256 = _manifest()
    points: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    contract: tuple[Any, ...] | None = None
    for path in paths:
        value, sha = _read(path)
        if value.get("kind") != "p9-dependent-small-stress-smoke":
            raise ValueError(f"{path} is not a common smoke summary")
        if value.get("deadline_mode") != "wall" or value.get("checksum_mode") != "inline":
            raise ValueError(f"{path} is not production-wall inline evidence")
        if value.get("latency_contract") != "production-wall-arrival-to-completion":
            raise ValueError(f"{path} has an unsupported latency contract")
        lock = value.get("deadline_lock")
        if not isinstance(lock, dict) or not isinstance(lock.get("sha256"), str):
            raise ValueError(f"{path} lacks deadline-lock provenance")
        workload = value.get("workload")
        if not isinstance(workload, str) or not workload:
            raise ValueError(f"{path} lacks workload")
        period = _finite(value.get("background_period_ms"), "background period")
        if period <= 0:
            raise ValueError("background period must be positive")
        offered = _finite(value.get("background_offered_rps"), "offered rps", nonnegative=True)
        expected_offered = 1000.0 / period
        if not math.isclose(offered, expected_offered, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{path} period and offered load differ")
        lock_deadline = lock.get("deadline_us", value.get("deadline_us"))
        current = (
            workload, value.get("deadline_mode"), value.get("checksum_mode"),
            _finite(lock_deadline, "deadline"),
            lock.get("sha256"), value.get("latency_contract"), value.get("iterations"),
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"{path} differs from the frozen sweep contract")
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError(f"{path} must contain exactly one system result")
        row = rows[0]
        system = row.get("system")
        if system not in expected_systems:
            raise ValueError(f"unsupported system label: {system}")
        manifest_row = manifest["rows"].get(system)
        if not isinstance(manifest_row, dict) or not isinstance(
            manifest_row.get("numeric_comparison_allowed"), bool
        ):
            raise ValueError(f"system {system} lacks manifest contract")
        key = (system, offered)
        if key in seen:
            raise ValueError(f"duplicate system/load point: {system} at {offered}")
        seen.add(key)
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            raise ValueError("pipeline request count must be positive")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError("deadline misses are invalid")
        p99 = _finite(row.get("pipeline_p99_us"), "pipeline p99", nonnegative=True)
        goodput = _finite(row.get("background_goodput_rps"), "background goodput", nonnegative=True)
        dmr = misses / requests
        points.append({
            "system": system, "offered_rps": offered, "period_ms": period,
            "numeric_comparison_allowed": manifest_row["numeric_comparison_allowed"],
            "comparison_status": manifest_row.get("status", "unknown"),
            "requests": requests, "deadline_misses": misses, "dmr": dmr,
            "p99_us": p99, "background_goodput_rps": goodput,
            "slo_qualified": misses == 0,
            "input": {"path": str(path.resolve()), "sha256": sha},
        })
    if contract is None:
        raise ValueError("empty sweep")
    systems = {point["system"] for point in points}
    missing = sorted(expected_systems - systems)
    if missing:
        raise ValueError(f"sweep is missing systems: {', '.join(missing)}")
    loads = sorted({point["offered_rps"] for point in points})
    for offered in loads:
        present = {
            point["system"] for point in points if point["offered_rps"] == offered
        }
        missing_at_load = sorted(expected_systems - present)
        if missing_at_load:
            raise ValueError(
                f"sweep load {offered:g} is missing systems: "
                + ", ".join(missing_at_load)
            )
    output: dict[str, Any] = {}
    for system in sorted(systems):
        rows = sorted((p for p in points if p["system"] == system), key=lambda p: p["offered_rps"])
        qualified = [
            p for p in rows
            if p["numeric_comparison_allowed"] and p["slo_qualified"]
        ]
        best = max(qualified, key=lambda p: p["offered_rps"], default=None)
        output[system] = {
            "points": rows,
            "max_slo_qualified_offered_rps": best["offered_rps"] if best else None,
            "max_slo_qualified_goodput_rps": best["background_goodput_rps"] if best else None,
        }
    deadline_us = contract[3]
    return {
        "schema_version": 1,
        "kind": "p9-production-wall-load-sweep",
        "proposed_system": "QUIET",
        "systems": sorted(systems),
        "workload": contract[0],
        "deadline_mode": "wall",
        "checksum_mode": "inline",
        "deadline_us": deadline_us,
        "deadline_lock_sha256": contract[4],
        "comparator_manifest": {
            "path": str((ROOT / "docs/p9-comparator-manifest.json").resolve()),
            "sha256": manifest_sha256,
        },
        "formal": False,
        "scope": "exploratory-production-wall-no-thermal-normalization",
        "slo_rule": "deadline_misses == 0; exploratory only, no confidence claim",
        "offered_loads_rps": loads,
        "frontier": output,
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
