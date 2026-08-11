#!/usr/bin/env python3
"""Profile and execute gpulet elastic partitioning on Thor TensorRT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "3c1c2aad3b33edcef20e549d5093c43af497e6ae"
UPSTREAM = {
    "system": "gpulet",
    "venue": "USENIX ATC 2022",
    "artifact": "https://github.com/casys-kaist/glet",
    "commit": UPSTREAM_COMMIT,
    "algorithm": "elastic partitioning with interference-aware best fit",
}
PARTITION_PAIRS = ((10, 90), (25, 75), (50, 50), (75, 25), (90, 10))


@dataclass(frozen=True)
class Profile:
    producer_quota: int
    background_quota: int
    critical_p50_ms: float
    critical_p99_ms: float
    background_mean_ms: float
    background_rps: float
    deadline_misses: int
    path: str
    sha256: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value: object, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def verify_upstream(source: Path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("gpulet source is not the pinned upstream commit")
    text = (source / "src/scheduler/scheduler_incremental.cpp").read_text(
        encoding="utf-8"
    )
    for symbol in ("getMaxReturnPart", "getMinPart", "findBestFit"):
        if symbol not in text:
            raise ValueError(f"pinned gpulet scheduler lacks {symbol}")


def load_profile(root: Path, producer: int, background: int) -> Profile:
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("results")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("system") != "Partition-only planning"
    ):
        raise ValueError("gpulet profile has the wrong public result row")
    row = rows[0]
    stage = row.get("stage_latency_us")
    service = json.loads(
        (root / "gpulet/background.json").read_text(encoding="utf-8")
    ).get("gpu_service")
    if not isinstance(stage, dict) or not isinstance(service, dict):
        raise ValueError("gpulet profile lacks stage or service evidence")
    return Profile(
        producer,
        background,
        positive(stage["validation_excluded_end_to_end_p50"], "critical p50") / 1000.0,
        positive(stage["validation_excluded_end_to_end_p99"], "critical p99") / 1000.0,
        positive(service["mean_ms"], "background mean"),
        positive(row["background_goodput_rps"], "background throughput"),
        int(row["deadline_misses"]),
        str(summary_path.resolve()),
        sha256(summary_path),
    )


def select_partition(
    profiles: list[Profile], *, deadline_ms: float, background_target_rps: float
) -> tuple[Profile, bool, list[dict[str, Any]]]:
    """Apply gpulet's SLO/rate feasibility test and smallest-best-fit order."""
    if not profiles:
        raise ValueError("gpulet requires at least one partition profile")
    decisions: list[dict[str, Any]] = []
    feasible: list[Profile] = []
    for profile in sorted(profiles, key=lambda item: item.producer_quota):
        latency_ok = profile.critical_p99_ms <= deadline_ms
        rate_ok = profile.background_rps >= 0.95 * background_target_rps
        decisions.append(
            {
                "producer_quota": profile.producer_quota,
                "background_quota": profile.background_quota,
                "latency_ok": latency_ok,
                "background_rate_ok": rate_ok,
                "schedulable": latency_ok and rate_ok,
            }
        )
        if latency_ok and rate_ok:
            feasible.append(profile)
    if feasible:
        return min(feasible, key=lambda item: item.producer_quota), True, decisions
    # The paper scheduler rejects this spatial placement. Execute its maximum
    # critical partition only as numeric diagnostic evidence.
    return max(profiles, key=lambda item: item.producer_quota), False, decisions


def run_common(
    repo: Path, output: Path, deadline_lock: Path, iterations: int,
    period_ms: float, allocation: tuple[int, int], workload: str,
) -> None:
    subprocess.run(
        [
            "python3", str(repo / "scripts/run_p9_dependent_stress_smoke.py"),
            "--repo", str(repo), "--result-dir", str(output),
            "--iterations", str(iterations), "--deadline-lock", str(deadline_lock),
            "--background-period-ms", str(period_ms),
            "--workload", workload, "--scenario", "gpulet",
            "--producer-quota", str(allocation[0]),
            "--background-quota", str(allocation[1]),
        ],
        cwd=repo, check=True, stdout=subprocess.DEVNULL,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--background-period-ms", type=float, default=2.0)
    parser.add_argument("--profile-iterations", type=int, default=200)
    parser.add_argument("--evaluation-iterations", type=int, default=1500)
    parser.add_argument(
        "--workload",
        choices=("whisper-projection", "resnet-control"),
        default="whisper-projection",
    )
    args = parser.parse_args(argv)
    if args.profile_iterations <= 0 or args.evaluation_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    repo = args.repo.resolve()
    source = args.source.resolve()
    root = args.result_dir.resolve()
    lock_path = args.deadline_lock.resolve()
    root.mkdir(parents=True)
    verify_upstream(source)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("contract", {}).get("workload") != args.workload:
        raise ValueError("gpulet deadline lock differs from requested workload")
    deadline_ms = positive(lock["deadline_us"], "deadline") / 1000.0
    target_rps = 1000.0 / positive(args.background_period_ms, "background period")
    profiles: list[Profile] = []
    for producer, background in PARTITION_PAIRS:
        path = root / f"profile-q{producer}-q{background}"
        run_common(repo, path, lock_path, args.profile_iterations,
                   args.background_period_ms, (producer, background), args.workload)
        profiles.append(load_profile(path, producer, background))
    selected, schedulable, decisions = select_partition(
        profiles, deadline_ms=deadline_ms, background_target_rps=target_rps
    )
    evaluation = root / f"evaluation-q{selected.producer_quota}-q{selected.background_quota}"
    run_common(repo, evaluation, lock_path, args.evaluation_iterations,
               args.background_period_ms,
               (selected.producer_quota, selected.background_quota), args.workload)
    result = {
        "schema_version": 1,
        "kind": "gpulet-thor-dependent-evaluation",
        "upstream": UPSTREAM,
        "adaptation": {
            "framework": "TensorRT",
            "workload": args.workload,
            "fixed_topology": "Thor 2g+1g",
            "batch_size": 1,
            "candidate_partitions": [list(pair) for pair in PARTITION_PAIRS],
            "profile_and_evaluation_requests_disjoint": True,
            "profile_iterations_per_partition": args.profile_iterations,
            "evaluation_iterations": args.evaluation_iterations,
        },
        "deadline_lock": {"path": str(lock_path), "sha256": sha256(lock_path)},
        "profiles": [asdict(profile) for profile in profiles],
        "scheduler_decisions": decisions,
        "spatial_schedule_feasible": schedulable,
        "selected_action": {
            "producer_quota": selected.producer_quota,
            "background_quota": selected.background_quota,
            "semantics": "gpulet-best-fit" if schedulable else "diagnostic-largest-critical-partition",
        },
        "evaluation": {
            "path": str((evaluation / "summary.json").resolve()),
            "sha256": sha256(evaluation / "summary.json"),
        },
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
