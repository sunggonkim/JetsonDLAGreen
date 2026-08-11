#!/usr/bin/env python3
"""Replay a session-level QUIET placement/quota characterization matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta


PLACEMENTS = (
    "fixed-1g-producer-2g-consumer",
    "fixed-2g-producer-1g-consumer",
)
QUOTAS = (50, 75, 100)
REPEATS = 3
PRODUCTION_WALL = "arrival-to-consumer-completion-excludes-correctness-validation"


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _cp95(misses: int, requests: int) -> float:
    if misses == requests:
        return 1.0
    return float(beta.ppf(0.95, misses + 1, requests - misses))


def summarize(paths: Iterable[Path], *, dmr_target: float = 0.0005) -> dict[str, Any]:
    paths = [path.resolve() for path in paths]
    if len(paths) != len(PLACEMENTS) * len(QUOTAS) * REPEATS:
        raise ValueError("candidate frontier requires exactly 18 session summaries")
    target = _finite(dmr_target, "dmr_target")
    if target >= 1.0:
        raise ValueError("dmr_target must be below one")
    rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    contract: tuple[Any, ...] | None = None
    for path in paths:
        value, digest = _read(path)
        production_wall_ok = value.get("production_wall_definition") == PRODUCTION_WALL or (
            value.get("latency_contract") == "production-wall-arrival-to-completion"
            and value.get("deadline_mode") == "wall"
        )
        if (
            value.get("kind") != "p9-dependent-small-stress-smoke"
            or value.get("workload") != "resnet-control"
            or value.get("latency_contract") != "production-wall-arrival-to-completion"
            or value.get("deadline_mode") != "wall"
            or not production_wall_ok
            or value.get("checksum_mode") != "inline"
            or value.get("placement_variant") not in PLACEMENTS
            or value.get("producer_quota_percent") != 100
            or value.get("background_quota_percent") not in QUOTAS
        ):
            raise ValueError(f"{path} is outside the candidate frontier contract")
        rows_value = value.get("results")
        if not isinstance(rows_value, list) or len(rows_value) != 1:
            raise ValueError(f"{path} must contain one result")
        row = rows_value[0]
        key = (value["placement_variant"], value["background_quota_percent"])
        if row.get("system") != "QUIET" or row.get("correctness_validated") is not True:
            raise ValueError(f"{path} lacks QUIET inline correctness evidence")
        if (
            row.get("placement_variant") != key[0]
            or row.get("producer_quota_percent") != 100
            or row.get("background_quota_percent") != key[1]
        ):
            raise ValueError(f"{path} row quota/placement differs from summary")
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests != 100:
            raise ValueError(f"{path} must contain exactly 100 requests")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError(f"{path} has invalid misses")
        trace = row.get("request_trace")
        if not isinstance(trace, dict):
            raise ValueError(f"{path} lacks request trace provenance")
        trace_path = trace.get("path")
        trace_sha = trace.get("sha256")
        if not isinstance(trace_path, str) or not isinstance(trace_sha, str):
            raise ValueError(f"{path} request trace lacks path/SHA")
        trace_file = Path(trace_path).resolve()
        if not trace_file.is_file() or hashlib.sha256(trace_file.read_bytes()).hexdigest() != trace_sha:
            raise ValueError(f"{path} request trace SHA differs")
        current = (
            value.get("deadline_us"), value.get("background_period_ms"),
            value.get("deadline_lock", {}).get("sha256") if isinstance(value.get("deadline_lock"), dict) else None,
            value.get("checksum_mode"), value.get("workload"),
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"{path} differs from the frozen candidate contract")
        rows.setdefault(key, []).append({
            "path": str(path), "sha256": digest, "requests": requests,
            "deadline_misses": misses, "wall_p99_us": _finite(
                row.get("wall_pipeline_p99_us", row.get("pipeline_p99_us")), "wall p99"
            ), "background_goodput_rps": _finite(
                row.get("background_goodput_rps"), "background goodput"
            ), "trace": {"path": str(trace_file), "sha256": trace_sha},
        })
    expected = {(placement, quota) for placement in PLACEMENTS for quota in QUOTAS}
    if set(rows) != expected or any(len(values) != REPEATS for values in rows.values()):
        raise ValueError("candidate frontier lacks exactly three repeats per point")
    points: list[dict[str, Any]] = []
    for placement in PLACEMENTS:
        for quota in QUOTAS:
            sessions = sorted(rows[(placement, quota)], key=lambda item: item["path"])
            requests = sum(item["requests"] for item in sessions)
            misses = sum(item["deadline_misses"] for item in sessions)
            p99 = [item["wall_p99_us"] for item in sessions]
            goodput = [item["background_goodput_rps"] for item in sessions]
            points.append({
                "placement_variant": placement,
                "producer_quota_percent": 100,
                "background_quota_percent": quota,
                "sessions": sessions,
                "session_count": len(sessions),
                "requests": requests,
                "deadline_misses": misses,
                "observed_dmr": misses / requests,
                "cp95_upper_dmr": _cp95(misses, requests),
                "cp95_slo_qualified": _cp95(misses, requests) <= target,
                "descriptive_zero_miss": misses == 0,
                "p99_us": {"mean": statistics.fmean(p99), "min": min(p99), "max": max(p99)},
                "background_goodput_rps": {
                    "mean": statistics.fmean(goodput), "min": min(goodput), "max": max(goodput)
                },
            })
    qualified = [point for point in points if point["cp95_slo_qualified"]]
    selected = max(qualified, key=lambda point: point["background_goodput_rps"]["mean"]) if qualified else None
    return {
        "schema_version": 1,
        "kind": "p9-quiet-placement-quota-session-frontier",
        "proposed_system": "QUIET",
        "workload": "resnet-control",
        "deadline_us": contract[0],
        "dmr_target": target,
        "formal": False,
        "thermal_normalized": False,
        "scope": "exploratory-session-level-placement-quota-frontier; no-thermal-normalization",
        "statistical_unit": "session",
        "points": points,
        "selected_cp95_slo_point": selected,
        "claim_guard": (
            "CP95 is evaluated over three 100-request sessions per point; "
            "no point is promoted without thermal normalization and application accuracy."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
