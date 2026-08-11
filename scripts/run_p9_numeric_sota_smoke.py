#!/usr/bin/env python3
"""Run six frozen mechanism treatments on the common P9 dependent DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class System:
    public_name: str
    scenario: str
    producer_quota: int
    background_quota: int
    action_source: str


SYSTEMS = (
    System("NVIDIA MIG", "nvidia-mig-isolation", 100, 100, "vendor primitive"),
    System("Quota-only provisioning", "gslice", 90, 10, "frozen quota ablation"),
    System("QUIET", "quiet", 100, 100, "proposed system"),
    System("NVIDIA MPS", "nvidia-mps-spatial-sharing", 100, 100, "vendor primitive"),
    System("Full-DAG quiescence", "orion", 100, 100, "frozen local gating ablation"),
    System("Partition-only planning", "gpulet", 90, 10, "frozen placement ablation"),
)
SYSTEM_BY_NAME = {system.public_name: system for system in SYSTEMS}
CANONICAL_NAMES = tuple(system.public_name for system in SYSTEMS)


def williams_orders() -> tuple[tuple[str, ...], ...]:
    """Return the balanced six-treatment Williams design."""
    count = len(CANONICAL_NAMES)
    indices = [0]
    for offset in range(1, count // 2 + 1):
        indices.append(offset)
        if len(indices) < count:
            indices.append(count - offset)
    base = indices[:count]
    return tuple(
        tuple(CANONICAL_NAMES[(index + rotation) % count] for index in base)
        for rotation in range(count)
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--background-period-ms", type=float, default=2.0)
    parser.add_argument(
        "--sequence-index", type=int,
        help="zero-based sequence in the frozen six-treatment Williams design",
    )
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    repo = args.repo.resolve()
    root = args.result_dir.resolve()
    lock = args.deadline_lock.resolve()
    root.mkdir(parents=True)
    if args.sequence_index is None:
        ordered_systems = SYSTEMS
        order_design = "exploratory-single-order"
    else:
        orders = williams_orders()
        if args.sequence_index < 0 or args.sequence_index >= len(orders):
            raise ValueError("sequence-index is outside the Williams design")
        ordered_systems = tuple(
            SYSTEM_BY_NAME[name] for name in orders[args.sequence_index]
        )
        order_design = "six-treatment-williams"
    rows = []
    inputs = []
    for index, system in enumerate(ordered_systems, start=1):
        run_dir = root / f"{index:02d}-{system.scenario}"
        subprocess.run(
            [
                "python3", str(repo / "scripts/run_p9_dependent_stress_smoke.py"),
                "--repo", str(repo), "--result-dir", str(run_dir),
                "--iterations", str(args.iterations),
                "--deadline-lock", str(lock),
                "--background-period-ms", str(args.background_period_ms),
                "--workload", "whisper-projection", "--scenario", system.scenario,
                "--producer-quota", str(system.producer_quota),
                "--background-quota", str(system.background_quota),
            ],
            cwd=repo, check=True, stdout=subprocess.DEVNULL,
        )
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = summary.get("results")
        if not isinstance(result, list) or len(result) != 1:
            raise ValueError(f"{system.public_name} has the wrong result count")
        row = result[0]
        if row.get("system") != system.public_name:
            raise ValueError(f"{system.public_name} public label differs")
        row["action_source"] = system.action_source
        rows.append(row)
        inputs.append({"path": str(summary_path), "sha256": sha256(summary_path)})
    output = {
        "schema_version": 1,
        "kind": "p9-numeric-mechanism-smoke",
        "scope": "execution-path validation; not formal paper statistics",
        "published_system_comparison": False,
        "workload": "Whisper producer -> 2.304MB coherent edge -> projection consumer",
        "background": "DistilBERT at 500 offered requests/s",
        "deadline_lock": {"path": str(lock), "sha256": sha256(lock)},
        "order_design": order_design,
        "sequence_index": args.sequence_index,
        "execution_order": [system.public_name for system in ordered_systems],
        "iterations_per_system": args.iterations,
        "results": rows,
        "inputs": inputs,
    }
    output_path = root / "summary.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
