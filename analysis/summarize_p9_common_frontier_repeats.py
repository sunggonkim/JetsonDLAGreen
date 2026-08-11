#!/usr/bin/env python3
"""Aggregate same-deadline production-wall repeats for executable comparators."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import t as student_t

try:
    from .p9_frontier_evidence import validate_correctness
except ImportError:  # direct CLI execution
    from p9_frontier_evidence import validate_correctness


CORE_SYSTEMS = {"QUIET", "NVIDIA MPS", "XSched"}
OPTIONAL_SYSTEMS = {"Static full gating"}
SYSTEMS = CORE_SYSTEMS | OPTIONAL_SYSTEMS
NUMERIC_FRONTIER_SYSTEMS = ["NVIDIA MPS", "QUIET"]
PRODUCTION_WALL_DEFINITION = (
    "arrival-to-consumer-completion-excludes-correctness-validation"
)
CORRECTNESS_PLACEMENT = "post-completion"


def _paired_t_interval(values: list[float]) -> dict[str, Any] | None:
    if len(values) < 2:
        return None
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / (len(values) ** 0.5)
    t_critical = float(student_t.ppf(0.975, len(values) - 1))
    mean = statistics.fmean(values)
    half_width = t_critical * standard_error
    return {
        "method": "paired-session-t-interval",
        "confidence": 0.95,
        "unit": "session-pair",
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "t_critical": t_critical,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _session_key(path: Path, fallback: int) -> str:
    match = re.search(r"(?:^|[-_])r(\d+)(?:[-_.]|$)", str(path))
    return f"r{int(match.group(1)):03d}" if match else f"index-{fallback:03d}"


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
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def summarize(paths: Iterable[Path], *, minimum_repeats: int = 3) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {system: [] for system in CORE_SYSTEMS}
    contract: tuple[Any, ...] | None = None
    for path in paths:
        value, digest = _read(path)
        if (
            value.get("kind") != "p9-dependent-small-stress-smoke"
            or value.get("workload") != "resnet-control"
            or value.get("latency_contract") != "production-wall-arrival-to-completion"
            or value.get("deadline_mode") != "wall"
            or value.get("checksum_mode") != "inline"
        ):
            raise ValueError(f"{path} is outside the common frontier contract")
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError(f"{path} must contain one result")
        row = rows[0]
        system = row.get("system")
        if system not in SYSTEMS:
            raise ValueError(f"unsupported system: {system}")
        grouped.setdefault(system, [])
        correctness = validate_correctness(value, row, path)
        lock = value.get("deadline_lock")
        if not isinstance(lock, dict) or not isinstance(lock.get("sha256"), str):
            raise ValueError(f"{path} lacks deadline lock provenance")
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            raise ValueError(f"{path} has invalid request count")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError(f"{path} has invalid miss count")
        current = (
            value.get("iterations"), value.get("background_period_ms"),
            value.get("background_offered_rps"), _finite(value.get("deadline_us"), "deadline"),
            value.get("workload"), value.get("checksum_mode"), lock["sha256"],
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"{path} differs from the common contract")
        grouped[system].append({
            "session_key": _session_key(path, len(grouped[system])),
            "path": str(path.resolve()), "sha256": digest,
            "deadline_lock_sha256": lock["sha256"],
            "wall_p99_us": _finite(row.get("wall_pipeline_p99_us", row.get("pipeline_p99_us")), "p99"),
            "background_goodput_rps": _finite(row.get("background_goodput_rps"), "goodput"),
            "requests": requests, "deadline_misses": misses,
            "dmr": misses / requests,
            "unique_payload_checksums": row.get("unique_payload_checksums"),
            "unique_policy_output_checksums": row.get("unique_policy_output_checksums"),
            "correctness_evidence": correctness,
        })
    if contract is None or any(
        len(grouped[system]) < minimum_repeats for system in CORE_SYSTEMS
    ):
        raise ValueError(f"each system needs at least {minimum_repeats} repeats")
    optional_present = set(grouped) - CORE_SYSTEMS
    if optional_present and any(
        len(grouped[system]) < minimum_repeats for system in optional_present
    ):
        raise ValueError(f"optional baseline needs at least {minimum_repeats} repeats")
    summaries: dict[str, Any] = {}
    for system, rows in grouped.items():
        rows.sort(key=lambda row: row["path"])
        p99s = [row["wall_p99_us"] for row in rows]
        goodputs = [row["background_goodput_rps"] for row in rows]
        summaries[system] = {
            "repeat_count": len(rows), "repeats": rows,
            "p99_us": {"mean": statistics.fmean(p99s), "median": statistics.median(p99s), "min": min(p99s), "max": max(p99s)},
            "background_goodput_rps": {"mean": statistics.fmean(goodputs), "median": statistics.median(goodputs), "min": min(goodputs), "max": max(goodputs)},
            "total_requests": sum(row["requests"] for row in rows),
            "total_deadline_misses": sum(row["deadline_misses"] for row in rows),
            "qualified_repeat_count": sum(row["deadline_misses"] == 0 for row in rows),
            "all_repeats_slo_qualified": all(row["deadline_misses"] == 0 for row in rows),
        }
    # Pair by the recorded session/repeat key rather than by aggregate order.
    # This keeps uncertainty tied to the experimental unit and rejects silently
    # unbalanced sessions in the evidence artifact.
    quiet_rows = {row["session_key"]: row for row in grouped["QUIET"]}
    paired_statistics: dict[str, Any] = {}
    for system in sorted(CORE_SYSTEMS - {"QUIET"}):
        control_rows = {row["session_key"]: row for row in grouped[system]}
        keys = sorted(set(quiet_rows) & set(control_rows))
        if set(quiet_rows) != set(control_rows):
            paired_statistics[system] = {
                "status": "unbalanced-session-keys",
                "quiet_keys": sorted(quiet_rows),
                "control_keys": sorted(control_rows),
            }
            continue
        p99_deltas = [quiet_rows[key]["wall_p99_us"] - control_rows[key]["wall_p99_us"] for key in keys]
        goodput_deltas = [quiet_rows[key]["background_goodput_rps"] - control_rows[key]["background_goodput_rps"] for key in keys]
        paired_statistics[system] = {
            "status": "descriptive",
            "session_keys": keys,
            "p99_delta_us_quiet_minus_baseline": {
                "per_session": p99_deltas,
                "t95": _paired_t_interval(p99_deltas),
            },
            "background_goodput_delta_rps_quiet_minus_baseline": {
                "per_session": goodput_deltas,
                "t95": _paired_t_interval(goodput_deltas),
            },
            "claim_guard": "Descriptive session interval only; no thermal normalization or formal SLO certification.",
        }
    return {
        "schema_version": 1,
        "kind": "p9-common-production-wall-frontier-repeats",
        "proposed_system": "QUIET",
        "workload": contract[4], "deadline_us": contract[3],
        "offered_rps": contract[2], "formal": False,
        "numeric_frontier_systems": NUMERIC_FRONTIER_SYSTEMS,
        "exploratory_systems": sorted(set(grouped) - set(NUMERIC_FRONTIER_SYSTEMS)),
        "production_wall_definition": PRODUCTION_WALL_DEFINITION,
        "correctness_validation_placement": CORRECTNESS_PLACEMENT,
        "ranking_allowed": False,
        "scope": "exploratory-same-deadline-session-frontier; no-thermal-normalization",
        "repeat_unit": "one production-wall run with inline correctness",
        "systems": summaries,
        "paired_session_statistics": paired_statistics,
        "statistical_unit": "paired-session",
        "notes": ["SLO qualification here is descriptive per-run zero misses; no CP or population claim is made."],
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
