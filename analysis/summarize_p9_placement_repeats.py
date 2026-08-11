#!/usr/bin/env python3
"""Aggregate paired QUIET placement repeats under one common lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import t as student_t


PLACEMENTS = (
    "fixed-1g-producer-2g-consumer",
    "fixed-2g-producer-1g-consumer",
)


def _paired_t_interval(values: list[float]) -> dict[str, Any] | None:
    if len(values) < 2:
        return None
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / (len(values) ** 0.5)
    t_critical = float(student_t.ppf(0.975, len(values) - 1))
    mean = statistics.mean(values)
    half_width = t_critical * standard_error
    return {
        "method": "paired-session-t-interval",
        "confidence": 0.95,
        "unit": "placement-session-pair",
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "t_critical": t_critical,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


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


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {placement: [] for placement in PLACEMENTS}
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
            raise ValueError(f"{path} is outside the common placement contract")
        placement = value.get("placement_variant")
        if placement not in grouped:
            raise ValueError(f"unsupported placement: {placement}")
        lock = value.get("deadline_lock")
        if not isinstance(lock, dict) or not isinstance(lock.get("sha256"), str):
            raise ValueError(f"{path} lacks common-lock provenance")
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError(f"{path} must contain one result")
        row = rows[0]
        if row.get("system") != "QUIET" or row.get("correctness_validated") is not True:
            raise ValueError(f"{path} lacks QUIET correctness evidence")
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            raise ValueError(f"{path} has invalid request count")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError(f"{path} has invalid misses")
        current = (
            value.get("iterations"), value.get("background_period_ms"),
            value.get("background_offered_rps"), _finite(value.get("deadline_us"), "deadline"),
            lock["sha256"], value.get("workload"), value.get("checksum_mode"),
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"{path} differs from the frozen common contract")
        grouped[placement].append({
            "path": str(path.resolve()), "sha256": digest,
            "wall_p99_us": _finite(row.get("wall_pipeline_p99_us", row.get("pipeline_p99_us")), "p99"),
            "background_goodput_rps": _finite(row.get("background_goodput_rps"), "goodput"),
            "requests": requests, "deadline_misses": misses,
            "dmr": misses / requests,
            "unique_payload_checksums": row.get("unique_payload_checksums"),
            "unique_policy_output_checksums": row.get("unique_policy_output_checksums"),
        })
    if contract is None or any(not grouped[placement] for placement in PLACEMENTS):
        raise ValueError("both placements need at least one repeat")
    if len(grouped[PLACEMENTS[0]]) != len(grouped[PLACEMENTS[1]]):
        raise ValueError("placement repeat counts differ")
    for rows in grouped.values():
        rows.sort(key=lambda row: row["path"])
    summaries: dict[str, Any] = {}
    for placement, rows in grouped.items():
        p99s = [row["wall_p99_us"] for row in rows]
        goodputs = [row["background_goodput_rps"] for row in rows]
        summaries[placement] = {
            "repeats": rows,
            "repeat_count": len(rows),
            "p99_us": {
                "mean": statistics.fmean(p99s), "median": statistics.median(p99s),
                "min": min(p99s), "max": max(p99s),
            },
            "background_goodput_rps": {
                "mean": statistics.fmean(goodputs), "median": statistics.median(goodputs),
                "min": min(goodputs), "max": max(goodputs),
            },
            "total_requests": sum(row["requests"] for row in rows),
            "total_deadline_misses": sum(row["deadline_misses"] for row in rows),
            "all_repeats_slo_qualified": all(row["deadline_misses"] == 0 for row in rows),
        }
    paired = []
    for index, (forward, reverse) in enumerate(zip(grouped[PLACEMENTS[0]], grouped[PLACEMENTS[1]])):
        paired.append({
            "pair_index": index,
            "p99_us_delta_reverse_minus_forward": reverse["wall_p99_us"] - forward["wall_p99_us"],
            "goodput_rps_delta_reverse_minus_forward": reverse["background_goodput_rps"] - forward["background_goodput_rps"],
            "forward_input_sha256": forward["sha256"], "reverse_input_sha256": reverse["sha256"],
        })
    p99_deltas = [row["p99_us_delta_reverse_minus_forward"] for row in paired]
    goodput_deltas = [row["goodput_rps_delta_reverse_minus_forward"] for row in paired]
    return {
        "schema_version": 1,
        "kind": "p9-quiet-placement-repeat-summary",
        "proposed_system": "QUIET",
        "workload": contract[5],
        "deadline_us": contract[3],
        "formal": False,
        "scope": "exploratory-session-level-placement-characterization; no-thermal-normalization",
        "repeat_unit": "one production-wall run with inline correctness",
        "placements": summaries,
        "paired_deltas": paired,
        "paired_session_statistics": {
            "p99_us_reverse_minus_forward": {
                "t95": _paired_t_interval(p99_deltas),
            },
            "goodput_rps_reverse_minus_forward": {
                "t95": _paired_t_interval(goodput_deltas),
            },
            "claim_guard": (
                "Descriptive placement-session interval only; no thermal "
                "normalization, formal SLO certification, or population claim."
            ),
        },
        "statistical_unit": "paired-placement-session",
        "notes": ["No confidence or population claim is made from three nonthermal repeats."],
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
