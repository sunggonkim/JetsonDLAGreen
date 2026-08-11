#!/usr/bin/env python3
"""Aggregate repeated paired production-wall smoke summaries.

The output is explicitly exploratory: it reports block-level means and
request-level miss counts, but does not claim formal confidence or thermal
normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from scipy.stats import t as student_t


PRODUCTION_WALL_DEFINITION_V2 = (
    "arrival-to-consumer-completion-excludes-correctness-validation"
)
CORRECTNESS_PLACEMENT_V2 = "post-completion"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "p9-dependent-small-stress-smoke":
        raise ValueError(f"{path}: unsupported summary kind")
    rows = value.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{path}: expected one result row")
    row = rows[0]
    required = ("system", "pipeline_requests", "deadline_misses", "pipeline_p99_us",
                "background_goodput_rps", "deadline_mode", "latency_contract",
                "checksum_mode", "producer_uuid", "consumer_uuid",
                "producer_sms", "consumer_sms")
    if any(key not in row for key in required):
        raise ValueError(f"{path}: missing smoke result fields")
    if value.get("latency_contract") != "production-wall-arrival-to-completion":
        raise ValueError(f"{path}: not a production-wall summary")
    if value.get("production_wall_definition") != PRODUCTION_WALL_DEFINITION_V2:
        raise ValueError(f"{path}: stale production-wall contract")
    if value.get("correctness_validation_placement") != CORRECTNESS_PLACEMENT_V2:
        raise ValueError(f"{path}: stale correctness-validation placement")
    if (
        not isinstance(row["producer_uuid"], str) or not row["producer_uuid"]
        or not isinstance(row["consumer_uuid"], str) or not row["consumer_uuid"]
        or isinstance(row["producer_sms"], bool)
        or not isinstance(row["producer_sms"], int) or row["producer_sms"] <= 0
        or isinstance(row["consumer_sms"], bool)
        or not isinstance(row["consumer_sms"], int) or row["consumer_sms"] <= 0
    ):
        raise ValueError(f"{path}: observed MIG topology is missing or invalid")
    return value, row


def _contract(value: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    lock = value.get("deadline_lock")
    return {
        "workload": value.get("workload"),
        "deadline_mode": value.get("deadline_mode"),
        "latency_contract": value.get("latency_contract"),
        "production_wall_definition": value.get("production_wall_definition"),
        "correctness_validation_placement": value.get(
            "correctness_validation_placement"
        ),
        "checksum_mode": value.get("checksum_mode"),
        "deadline_us": value.get("deadline_us"),
        "deadline_lock_sha256": lock.get("sha256") if isinstance(lock, dict) else None,
        "background_period_ms": value.get("background_period_ms"),
        "background_offered_rps": value.get("background_offered_rps"),
        "iterations": value.get("iterations"),
        "producer_uuid": row["producer_uuid"],
        "consumer_uuid": row["consumer_uuid"],
        "producer_sms": row["producer_sms"],
        "consumer_sms": row["consumer_sms"],
    }


def _system(paths: Sequence[Path], expected: str) -> dict[str, Any]:
    if not paths:
        raise ValueError(f"{expected}: no summaries supplied")
    values, rows = zip(*(_load(path) for path in paths))
    contracts = [_contract(value, row) for value, row in zip(values, rows, strict=True)]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError(f"{expected}: repeated summaries have different contracts")
    if any(row["system"] != expected for row in rows):
        raise ValueError(f"{expected}: system label differs")
    requests = [int(row["pipeline_requests"]) for row in rows]
    if len(set(requests)) != 1 or requests[0] <= 0:
        raise ValueError(f"{expected}: request count differs")
    if any(row["checksum_mode"] == "off" or row.get("correctness_validated") is not True for row in rows):
        raise ValueError(f"{expected}: correctness validation is not enabled")
    p99 = [float(row["pipeline_p99_us"]) for row in rows]
    misses = [int(row["deadline_misses"]) for row in rows]
    goodput = [float(row["background_goodput_rps"]) for row in rows]
    return {
        "system": expected,
        "runs": len(rows),
        "requests_per_run": requests[0],
        "requests": sum(requests),
        "deadline_misses": sum(misses),
        "deadline_miss_rate": sum(misses) / sum(requests),
        "p99_us": {
            "mean": statistics.mean(p99),
            "median": statistics.median(p99),
            "min": min(p99),
            "max": max(p99),
            "per_run": p99,
        },
        "background_goodput_rps": {
            "mean": statistics.mean(goodput),
            "median": statistics.median(goodput),
            "min": min(goodput),
            "max": max(goodput),
            "per_run": goodput,
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": _digest(path)}
            for path in paths
        ],
        "contract": contracts[0],
    }


def _paired_t_interval(values: Sequence[float]) -> dict[str, Any] | None:
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
        "unit": "session-pair",
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "t_critical": t_critical,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def summarize(quiet_paths: Sequence[Path], baseline_paths: Sequence[Path], baseline: str) -> dict[str, Any]:
    quiet = _system(quiet_paths, "QUIET")
    control = _system(baseline_paths, baseline)
    if quiet["runs"] != control["runs"]:
        raise ValueError("paired systems have different repetition counts")
    if quiet["contract"] != control["contract"]:
        raise ValueError("paired systems have different contracts")
    p99_deltas = [
        q - b
        for q, b in zip(
            quiet["p99_us"]["per_run"], control["p99_us"]["per_run"], strict=True
        )
    ]
    goodput_deltas = [
        q - b
        for q, b in zip(
            quiet["background_goodput_rps"]["per_run"],
            control["background_goodput_rps"]["per_run"],
            strict=True,
        )
    ]
    return {
        "schema_version": 1,
        "kind": "p9-paired-wall-smoke-repeats",
        "formal": False,
        "scope": "exploratory-production-wall-no-thermal-normalization",
        "baseline": baseline,
        "contract": quiet["contract"],
        "systems": {baseline: control, "QUIET": quiet},
        "quiet_delta_vs_baseline": {
            "p99_mean_us": quiet["p99_us"]["mean"] - control["p99_us"]["mean"],
            "deadline_miss_delta": quiet["deadline_misses"] - control["deadline_misses"],
            "deadline_miss_rate_delta": quiet["deadline_miss_rate"] - control["deadline_miss_rate"],
            "background_goodput_mean_rps": quiet["background_goodput_rps"]["mean"] - control["background_goodput_rps"]["mean"],
        },
        "paired_session_statistics": {
            "p99_delta_us_quiet_minus_baseline": {
                "per_pair": p99_deltas,
                "t95": _paired_t_interval(p99_deltas),
            },
            "background_goodput_delta_rps_quiet_minus_baseline": {
                "per_pair": goodput_deltas,
                "t95": _paired_t_interval(goodput_deltas),
            },
            "claim_guard": (
                "Descriptive session interval only; no thermal normalization or "
                "formal SLO certification is implied."
            ),
        },
        "statistical_unit": "paired-session",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-name", default="NVIDIA MPS")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.quiet, args.baseline, args.baseline_name)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
