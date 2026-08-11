#!/usr/bin/env python3
"""Execute Orion's conservative managed-client policy on TensorRT."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


UPSTREAM_COMMIT = "20f9469764fb96d94ce23a8e70615196e9ce4ba1"
UPSTREAM = {
    "system": "Orion",
    "venue": "EuroSys 2024",
    "doi": "10.1145/3627703.3629578",
    "artifact": "https://github.com/eth-easl/orion",
    "commit": UPSTREAM_COMMIT,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_upstream(source: Path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Orion source is not the pinned upstream commit")
    scheduler = (source / "src/scheduler/scheduler_eval.cpp").read_text(
        encoding="utf-8"
    )
    for token in ("sm_threshold", "hp_limit", "op_info_0.duration"):
        if token not in scheduler:
            raise ValueError(f"pinned Orion scheduler lacks {token}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--background-period-ms", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=1500)
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    repo = args.repo.resolve()
    source = args.source.resolve()
    result_dir = args.result_dir.resolve()
    lock = args.deadline_lock.resolve()
    verify_upstream(source)
    subprocess.run(
        [
            "python3", str(repo / "scripts/run_p9_dependent_stress_smoke.py"),
            "--repo", str(repo), "--result-dir", str(result_dir),
            "--iterations", str(args.iterations), "--deadline-lock", str(lock),
            "--background-period-ms", str(args.background_period_ms),
            "--workload", "whisper-projection", "--scenario", "orion",
        ],
        cwd=repo, check=True, stdout=subprocess.DEVNULL,
    )
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("results")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("Orion evaluation has the wrong result count")
    result = {
        "schema_version": 1,
        "kind": "orion-tensorrt-managed-client-evaluation",
        "upstream": UPSTREAM,
        "fidelity": {
            "port": "managed-client-policy",
            "operation_granularity": "one opaque TensorRT DAG request",
            "high_priority_operation": "Whisper producer plus dependent projection consumer",
            "best_effort_operation": "one DistilBERT TensorRT request",
            "policy": "defer best-effort submission while high-priority operation is active",
            "reason": "opaque TensorRT cuLaunchKernelEx calls lack Orion kernel profiles",
            "not_a_native_interceptor_result": True,
        },
        "deadline_lock": {"path": str(lock), "sha256": sha256(lock)},
        "evaluation": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "result": rows[0],
    }
    output = result_dir / "orion-result.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
