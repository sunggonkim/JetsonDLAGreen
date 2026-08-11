#!/usr/bin/env python3
"""Evaluate one BOER point with the P9 TensorRT runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import time
from typing import Any


SUPPORTED_QUOTAS = {25, 50, 100}


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def metrics_from_summary(
    summary: dict[str, Any], deadline_ms: float, dmr_target: float
) -> dict[str, float]:
    policies = summary.get("policies")
    if not isinstance(policies, list) or len(policies) != 1:
        raise ValueError("BOER candidate run must contain exactly one policy")
    policy = policies[0]
    if not isinstance(policy, dict) or policy.get("name") != "uncoordinated-borrow":
        raise ValueError("BOER candidate run used the wrong execution policy")
    by_modality = policy.get("goodput_by_modality")
    if not isinstance(by_modality, dict):
        raise ValueError("BOER candidate lacks per-modality goodput")
    audio = finite(by_modality.get("audio"), "audio goodput")
    language = finite(by_modality.get("language"), "language goodput")
    miss_rate = finite(policy.get("deadline_miss_rate"), "deadline miss rate")
    p99 = finite(policy.get("critical_p99_ms_max"), "critical p99")
    return {
        # BOER's online pruning uses its p99 QoS test. The stricter QUIET DMR
        # target is retained as an evaluation metric and applied to the final
        # replay, not injected into BOER's published search rule.
        "feasible": float(p99 <= deadline_ms),
        "slo_limit_ms": deadline_ms,
        "worst_p99_ms": p99,
        "served_rps_0": audio,
        "served_rps_1": language,
        "deadline_miss_rate": miss_rate,
        "dmr_target": dmr_target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    parser.add_argument("--scenario", choices=("independent", "dependent"), required=True)
    parser.add_argument("--deadline-ms", type=float, required=True)
    parser.add_argument("--dmr-target", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--calibration-repeats", type=int, default=1)
    args = parser.parse_args()

    candidate_id = os.environ.get("BOER_CANDIDATE_ID")
    quota = int(os.environ["BOER_SM_PERCENT"])
    offered_rps = int(os.environ["BOER_OFFERED_RPS"])
    if not candidate_id or quota not in SUPPORTED_QUOTAS or offered_rps <= 0:
        raise ValueError("invalid BOER candidate environment")
    run_dir = args.result_root.resolve() / (
        f"{args.scenario}-{candidate_id}-{time.time_ns()}"
    )
    run_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "RESULT_DIR": str(run_dir),
            "SCENARIO": args.scenario,
            "POLICY_ORDER": "uncoordinated-borrow",
            "BORROWER_QUOTA": str(quota),
            "PRESSURE_RPS_PER_TENANT": str(offered_rps),
            "DEADLINE_MS": str(args.deadline_ms),
            "DEADLINE_SOURCE": "fixed-explicit",
            "DMR_TARGET": str(args.dmr_target),
            "EPOCHS": str(args.epochs),
            "SAMPLES": str(args.samples),
            "CALIBRATION_REPEATS": str(args.calibration_repeats),
            "EXPERIMENT_LABEL": f"boer-{args.scenario}-{candidate_id}",
        }
    )
    command = [str(args.repo / "scripts" / "run_p9_mig_slack_governor.sh")]
    with (run_dir / "runner.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=args.repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = metrics_from_summary(summary, args.deadline_ms, args.dmr_target)
    metrics["result_dir"] = str(run_dir)  # type: ignore[assignment]
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
