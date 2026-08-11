#!/usr/bin/env python3
"""Run the GSLICE Algorithm-1 resource controller on Thor TensorRT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable


UPSTREAM = {
    "system": "GSLICE",
    "venue": "ACM SoCC 2020",
    "doi": "10.1145/3419111.3421284",
    "algorithm": "GPU Resource Tuning Algorithm (Algorithm 1)",
    "artifact": "not-public-paper-algorithm-reimplementation",
}
QUOTAS = (10, 25, 50, 75, 90)
DEADBAND_PERCENT = 5.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def compute_demand(
    current_percent: float,
    slo_ms: float,
    average_latency_ms: float,
    arrival_rps: float,
    average_throughput_rps: float,
) -> float:
    """Reproduce GSLICE Algorithm 1's proportional update for one IF."""
    current = finite(current_percent, "current percentage")
    slo = finite(slo_ms, "SLO")
    latency = finite(average_latency_ms, "average latency")
    arrival = finite(arrival_rps, "arrival rate")
    throughput = finite(average_throughput_rps, "average throughput")
    residual_latency = slo - latency
    residual_throughput = throughput - arrival
    latency_difference = abs(residual_latency / latency) * 100.0
    throughput_difference = abs(residual_throughput / arrival) * 100.0
    violation_factors = []
    if residual_latency < 0.0 and latency_difference >= DEADBAND_PERCENT:
        violation_factors.append(abs(residual_latency / latency))
    if residual_throughput < 0.0 and throughput_difference >= DEADBAND_PERCENT:
        violation_factors.append(abs(residual_throughput / throughput))
    if violation_factors:
        factor = max(violation_factors)
        return min(100.0, current + current * factor)

    headroom_factors = []
    if residual_latency > 0.0 and latency_difference >= DEADBAND_PERCENT:
        headroom_factors.append(abs(residual_latency / latency))
    if residual_throughput > 0.0 and throughput_difference >= DEADBAND_PERCENT:
        headroom_factors.append(abs(residual_throughput / throughput))
    if headroom_factors:
        factor = max(headroom_factors)
        return max(1.0, current - current * factor)
    return current


def max_min_pair(first_demand: float, second_demand: float) -> tuple[float, float]:
    demands = [(0, min(100.0, max(1.0, first_demand))), (1, min(100.0, max(1.0, second_demand)))]
    demands.sort(key=lambda item: item[1])
    allocation = [0.0, 0.0]
    smaller_index, smaller = demands[0]
    larger_index, larger = demands[1]
    if smaller <= 50.0:
        allocation[smaller_index] = smaller
        allocation[larger_index] = min(larger, 100.0 - smaller)
    else:
        allocation[smaller_index] = 50.0
        allocation[larger_index] = 50.0
    return allocation[0], allocation[1]


def snap_pair(producer: float, background: float) -> tuple[int, int]:
    candidates = [
        (left, right)
        for left in QUOTAS
        for right in QUOTAS
        if left + right <= 100
    ]
    return min(
        candidates,
        key=lambda pair: (
            (pair[0] - producer) ** 2 + (pair[1] - background) ** 2,
            -(pair[0] + pair[1]),
            pair,
        ),
    )


def next_allocation(
    current: tuple[int, int],
    *,
    deadline_ms: float,
    pipeline_p50_ms: float,
    background_period_ms: float,
    background_mean_ms: float,
    background_throughput_rps: float,
) -> tuple[int, int]:
    critical_arrival = 1000.0 / finite(deadline_ms, "deadline")
    critical_throughput = 1000.0 / finite(pipeline_p50_ms, "pipeline p50")
    background_arrival = 1000.0 / finite(background_period_ms, "background period")
    producer_demand = compute_demand(
        current[0], deadline_ms, pipeline_p50_ms, critical_arrival, critical_throughput
    )
    background_demand = compute_demand(
        current[1],
        background_period_ms,
        background_mean_ms,
        background_arrival,
        background_throughput_rps,
    )
    return snap_pair(*max_min_pair(producer_demand, background_demand))


def load_round(path: Path) -> dict[str, float]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("system") != "GSLICE":
        raise ValueError("GSLICE tuning round has the wrong result")
    stage = rows[0].get("stage_latency_us")
    background = json.loads((path / "gslice" / "background.json").read_text(encoding="utf-8"))
    service = background.get("gpu_service")
    if not isinstance(stage, dict) or not isinstance(service, dict):
        raise ValueError("GSLICE tuning round lacks latency evidence")
    return {
        "pipeline_p50_ms": finite(stage["validation_excluded_end_to_end_p50"], "pipeline p50") / 1000.0,
        "pipeline_p99_ms": finite(stage["validation_excluded_end_to_end_p99"], "pipeline p99") / 1000.0,
        "background_mean_ms": finite(service["mean_ms"], "background mean"),
        "background_throughput_rps": finite(rows[0]["background_goodput_rps"], "background throughput"),
        "deadline_misses": int(rows[0]["deadline_misses"]),
    }


def run_common(
    repo: Path,
    output: Path,
    deadline_lock: Path,
    iterations: int,
    period_ms: float,
    allocation: tuple[int, int],
) -> None:
    subprocess.run(
        [
            "python3",
            str(repo / "scripts/run_p9_dependent_stress_smoke.py"),
            "--repo", str(repo),
            "--result-dir", str(output),
            "--iterations", str(iterations),
            "--deadline-lock", str(deadline_lock),
            "--background-period-ms", str(period_ms),
            "--workload", "whisper-projection",
            "--producer-quota", str(allocation[0]),
            "--background-quota", str(allocation[1]),
            "--scenario", "gslice",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--background-period-ms", type=float, default=2.0)
    parser.add_argument("--profile-iterations", type=int, default=300)
    parser.add_argument("--evaluation-iterations", type=int, default=1500)
    parser.add_argument("--maximum-rounds", type=int, default=6)
    args = parser.parse_args(argv)
    if args.profile_iterations <= 0 or args.evaluation_iterations <= 0 or args.maximum_rounds <= 0:
        raise ValueError("iteration and round counts must be positive")
    repo = args.repo.resolve()
    root = args.result_dir.resolve()
    root.mkdir(parents=True)
    lock = json.loads(args.deadline_lock.resolve().read_text(encoding="utf-8"))
    deadline_ms = finite(lock["deadline_us"], "deadline") / 1000.0
    allocation = (50, 50)
    rounds: list[dict[str, Any]] = []
    for index in range(args.maximum_rounds):
        round_dir = root / f"tune-{index + 1:02d}-q{allocation[0]}-q{allocation[1]}"
        run_common(repo, round_dir, args.deadline_lock.resolve(), args.profile_iterations, args.background_period_ms, allocation)
        observed = load_round(round_dir)
        updated = next_allocation(
            allocation,
            deadline_ms=deadline_ms,
            pipeline_p50_ms=observed["pipeline_p50_ms"],
            background_period_ms=args.background_period_ms,
            background_mean_ms=observed["background_mean_ms"],
            background_throughput_rps=observed["background_throughput_rps"],
        )
        rounds.append(
            {
                "round": index + 1,
                "allocation": {"producer": allocation[0], "background": allocation[1]},
                "observed": observed,
                "next_allocation": {"producer": updated[0], "background": updated[1]},
                "summary": str((round_dir / "summary.json").resolve()),
                "summary_sha256": sha256(round_dir / "summary.json"),
            }
        )
        if updated == allocation:
            break
        allocation = updated
    evaluation_dir = root / f"evaluation-q{allocation[0]}-q{allocation[1]}"
    run_common(repo, evaluation_dir, args.deadline_lock.resolve(), args.evaluation_iterations, args.background_period_ms, allocation)
    output = {
        "schema_version": 1,
        "kind": "gslice-thor-dependent-evaluation",
        "upstream": UPSTREAM,
        "controller": {"deadband_percent": DEADBAND_PERCENT, "maximum_rounds": args.maximum_rounds},
        "deadline_lock": {"path": str(args.deadline_lock.resolve()), "sha256": sha256(args.deadline_lock.resolve())},
        "tuning_rounds": rounds,
        "selected_allocation": {"producer": allocation[0], "background": allocation[1]},
        "evaluation": {
            "path": str((evaluation_dir / "summary.json").resolve()),
            "sha256": sha256(evaluation_dir / "summary.json"),
        },
    }
    (root / "result.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
