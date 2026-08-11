#!/usr/bin/env python3
"""Aggregate the non-thermal P9 QUIET/BOER offered-load campaign."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any


SYSTEM_POLICY = {"QUIET": "mig-governor", "BOER": "uncoordinated-borrow"}


def binomial_log_cdf(successes: int, trials: int, probability: float) -> float:
    if successes < 0 or trials <= 0 or successes > trials:
        raise ValueError("invalid binomial dimensions")
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 0.0 if successes == trials else -math.inf
    terms = []
    for value in range(successes + 1):
        terms.append(
            math.lgamma(trials + 1)
            - math.lgamma(value + 1)
            - math.lgamma(trials - value + 1)
            + value * math.log(probability)
            + (trials - value) * math.log1p(-probability)
        )
    maximum = max(terms)
    return maximum + math.log(sum(math.exp(term - maximum) for term in terms))


def clopper_pearson_upper(
    misses: int, requests: int, confidence_level: float = 0.95
) -> float:
    if misses < 0 or requests <= 0 or misses > requests:
        raise ValueError("invalid binomial dimensions")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid confidence level")
    if misses == requests:
        return 1.0
    target = math.log1p(-confidence_level)
    low = misses / requests
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if binomial_log_cdf(misses, requests, midpoint) > target:
            low = midpoint
        else:
            high = midpoint
    return high


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def replay_entry(entry: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    system = entry.get("system")
    scenario = entry.get("scenario")
    rate = entry.get("offered_rps_per_tenant")
    if system not in SYSTEM_POLICY or scenario not in {"independent", "dependent"}:
        raise ValueError("campaign entry has an invalid system or scenario")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError("campaign entry has an invalid offered rate")
    summary_path = (root / str(entry.get("summary"))).resolve()
    summary = load_json(summary_path)
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("campaign summary lacks config")
    expected = {
        "scenario": scenario,
        "pressure_rps_per_tenant": float(rate),
        "epochs": 8,
        "samples_per_epoch": 800,
        "dmr_target": 0.0005,
        "critical_placement": "2g",
        "resident_placement": "1g",
        "borrower_placement": "2g",
        "borrower_quota": 100 if system == "QUIET" else (25 if scenario == "independent" else 50),
        "guard_override_ms": 10.0 if system == "QUIET" else None,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"campaign config differs: {key}")
    policies = summary.get("policies")
    matches = (
        [policy for policy in policies if policy.get("name") == SYSTEM_POLICY[system]]
        if isinstance(policies, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError("campaign summary lacks the expected system result")
    policy = matches[0]
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("campaign summary lacks artifact provenance")
    benchmark_sha256 = artifacts.get("benchmark_sha256")
    implementation_sha256 = artifacts.get("implementation_sha256")
    if (
        not isinstance(benchmark_sha256, str)
        or len(benchmark_sha256) != 64
        or not isinstance(implementation_sha256, dict)
        or not implementation_sha256
    ):
        raise ValueError("campaign artifact provenance is invalid")
    requests = policy.get("critical_requests")
    misses = policy.get("deadline_misses")
    goodput = policy.get("pressure_goodput_per_second")
    p99 = policy.get("critical_p99_ms_max")
    if (
        not isinstance(requests, int)
        or requests != 6400
        or not isinstance(misses, int)
        or isinstance(misses, bool)
        or not 0 <= misses <= requests
        or not isinstance(goodput, (int, float))
        or isinstance(goodput, bool)
        or not math.isfinite(float(goodput))
        or not isinstance(p99, (int, float))
        or isinstance(p99, bool)
        or not math.isfinite(float(p99))
    ):
        raise ValueError("campaign policy metrics are invalid")
    rate_value = policy.get("deadline_miss_rate")
    if not isinstance(rate_value, (int, float)) or not math.isclose(
        float(rate_value), misses / requests, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("campaign deadline miss rate differs from counts")
    return {
        "system": system,
        "scenario": scenario,
        "offered_rps_per_tenant": rate,
        "repeat": entry.get("repeat"),
        "position": entry.get("position"),
        "summary": str(summary_path),
        "critical_requests": requests,
        "deadline_misses": misses,
        "pressure_goodput_per_second": float(goodput),
        "critical_p99_ms": float(p99),
        "deadline_ms": float(summary["deadline_ms"]),
        "benchmark_sha256": benchmark_sha256,
        "implementation_sha256": implementation_sha256,
    }


def summarize(manifest: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    if manifest.get("quiet_guard_ms") != 10:
        raise ValueError("campaign QUIET guard differs from the frozen performance value")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("campaign manifest has no runs")
    replayed = [replay_entry(entry, root) for entry in entries]
    deadlines = {row["deadline_ms"] for row in replayed}
    if len(deadlines) != 1:
        raise ValueError("campaign runs used different deadlines")
    benchmark_hashes = {row["benchmark_sha256"] for row in replayed}
    implementation_encodings = {
        json.dumps(row["implementation_sha256"], sort_keys=True)
        for row in replayed
    }
    if len(benchmark_hashes) != 1 or len(implementation_encodings) != 1:
        raise ValueError("campaign implementation changed between runs")
    rows = []
    cells = sorted(
        {
            (row["scenario"], row["offered_rps_per_tenant"], row["system"])
            for row in replayed
        }
    )
    for scenario, offered_rps, system in cells:
        samples = [
            row
            for row in replayed
            if (row["scenario"], row["offered_rps_per_tenant"], row["system"])
            == (scenario, offered_rps, system)
        ]
        requests = sum(row["critical_requests"] for row in samples)
        misses = sum(row["deadline_misses"] for row in samples)
        upper = clopper_pearson_upper(misses, requests)
        rows.append(
            {
                "scenario": scenario,
                "offered_rps_per_tenant": offered_rps,
                "system": system,
                "repeats": len(samples),
                "critical_requests": requests,
                "deadline_misses": misses,
                "deadline_miss_rate": misses / requests,
                "dmr_cp95_upper": upper,
                "slo_certified": misses == 0 and upper <= 0.0005,
                "pressure_goodput_mean": statistics.fmean(
                    row["pressure_goodput_per_second"] for row in samples
                ),
                "critical_p99_ms_max": max(
                    row["critical_p99_ms"] for row in samples
                ),
            }
        )
    return {
        "schema_version": 1,
        "proposed_system": "QUIET",
        "competitor": "BOER (Thor port)",
        "scope": "fixed-2g+1g-nonthermal-performance",
        "deadline_ms": deadlines.pop(),
        "dmr_target": 0.0005,
        "confidence": "one-sided exact Clopper-Pearson 95% under binomial sensitivity",
        "benchmark_sha256": benchmark_hashes.pop(),
        "implementation_sha256": replayed[0]["implementation_sha256"],
        "rows": rows,
        "runs": replayed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    result = summarize(manifest, args.manifest.parent)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
