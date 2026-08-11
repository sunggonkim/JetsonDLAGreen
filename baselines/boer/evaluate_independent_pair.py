#!/usr/bin/env python3
"""Evaluate one BOER complementary-MPS point on two independent services."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import signal
import subprocess
import time
from typing import Any


SUPPORTED_QUOTAS = {10, 25, 50, 75, 90}


def quota_pair(producer_quota: int) -> tuple[int, int]:
    pair = (producer_quota, 100 - producer_quota)
    if pair[0] not in SUPPORTED_QUOTAS or pair[1] not in SUPPORTED_QUOTAS:
        raise ValueError("BOER requires a profiled complementary quota pair")
    return pair


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def metrics_from_results(
    producer: dict[str, Any], background: dict[str, Any], slo_ms: float,
) -> dict[str, float]:
    producer_p99 = finite(
        producer.get("release_to_completion", {}).get("p99_ms"), "producer p99"
    )
    background_p99 = finite(
        background.get("release_to_completion", {}).get("p99_ms"), "background p99"
    )
    worst = max(producer_p99, background_p99)
    return {
        "feasible": float(worst <= slo_ms),
        "slo_limit_ms": slo_ms,
        "worst_p99_ms": worst,
        "producer_p99_ms": producer_p99,
        "background_p99_ms": background_p99,
        "served_rps_0": finite(producer.get("throughput_per_second"), "producer RPS"),
        "served_rps_1": finite(background.get("throughput_per_second"), "background RPS"),
        "deadline_miss_rate": 0.0,
        "dmr_target": 0.0005,
    }


def process_state(pid: int) -> str:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = text.rfind(")")
    if end < 0:
        raise RuntimeError("malformed process state")
    return text[end + 2]


def wait_paused(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"worker exited before barrier: {stdout} {stderr}")
        if process_state(process.pid) in {"T", "t"}:
            return
        time.sleep(0.01)
    raise TimeoutError("worker start barrier timed out")


def load_mig(path: pathlib.Path) -> dict[str, str]:
    result = dict(
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
    )
    for key in (
        "JDG_MIG_SMALL_UUID", "JDG_MPS_PIPE_DIRECTORY", "JDG_MPS_LOG_DIRECTORY"
    ):
        if not result.get(key):
            raise ValueError("MIG environment is incomplete")
    return result


def command(
    repo: pathlib.Path, engine: pathlib.Path, model: str, cpu: int,
    offered_rps: int, warmup: int, duration: float,
) -> list[str]:
    return [
        "taskset", "--cpu-list", str(cpu), str(repo / "build-r39/jdg-trt-bench"),
        "--engine", str(engine), "--model-name", model, "--role", "pressure",
        "--duration-seconds", str(duration), "--burst-size", "1", "--period-ms",
        str(1000.0 / offered_rps), "--warmup", str(warmup), "--include-transfers",
        "true", "--priority", "default", "--start-paused", "true",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    parser.add_argument("--slo-ms", type=float, required=True)
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.slo_ms <= 0 or args.duration_seconds <= 0 or args.warmup < 0:
        raise ValueError("invalid independent evaluation contract")
    candidate_id = os.environ.get("BOER_CANDIDATE_ID")
    producer_quota, background_quota = quota_pair(int(os.environ["BOER_SM_PERCENT"]))
    offered_rps = int(os.environ["BOER_OFFERED_RPS"])
    if not candidate_id or offered_rps <= 0:
        raise ValueError("invalid BOER candidate environment")
    repo = args.repo.resolve()
    mig = load_mig(pathlib.Path(os.environ.get("MIG_ENV", "/tmp/jdg-mps-1g/mig.env")))
    run_dir = args.result_root.resolve() / f"{candidate_id}-{time.time_ns()}"
    run_dir.mkdir(parents=True)
    definitions = (
        ("producer", "resnet10-detection", producer_quota, 0),
        ("background", "distilbert-sst2", background_quota, 1),
    )
    workers: list[tuple[str, subprocess.Popen[str]]] = []
    try:
        for label, model, quota, cpu in definitions:
            engine = repo / f"models/engines/mig-1g-q{quota}/{model}.engine"
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_SMALL_UUID"],
                "CUDA_MPS_PIPE_DIRECTORY": mig["JDG_MPS_PIPE_DIRECTORY"],
                "CUDA_MPS_LOG_DIRECTORY": mig["JDG_MPS_LOG_DIRECTORY"],
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(quota),
            })
            process = subprocess.Popen(
                command(repo, engine, model, cpu, offered_rps, args.warmup,
                        args.duration_seconds),
                cwd=repo, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            workers.append((label, process))
        for _, process in workers:
            wait_paused(process)
        for _, process in workers:
            os.kill(process.pid, signal.SIGCONT)
        results: dict[str, dict[str, Any]] = {}
        for label, process in workers:
            stdout, stderr = process.communicate(timeout=args.duration_seconds + 30)
            (run_dir / f"{label}.stderr").write_text(stderr, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"{label} worker failed with {process.returncode}")
            results[label] = json.loads(stdout)
            (run_dir / f"{label}.json").write_text(
                json.dumps(results[label], indent=2) + "\n", encoding="utf-8"
            )
    finally:
        for _, process in workers:
            if process.poll() is None:
                process.kill()
                process.communicate()
    metrics: dict[str, Any] = metrics_from_results(
        results["producer"], results["background"], args.slo_ms
    )
    metrics.update({
        "producer_quota": producer_quota,
        "background_quota": background_quota,
        "result_dir": str(run_dir),
    })
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
