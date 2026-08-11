#!/usr/bin/env python3
"""Evaluate a BOER point on the actual ResNet-to-policy dependent pipeline."""

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


def workload_contract(workload: str) -> tuple[str, str]:
    if workload == "resnet-control":
        return "resnet10-detection", "wall"
    if workload == "whisper-projection":
        return "whisper-tiny-encoder", "validation-excluded"
    raise ValueError("unsupported BOER dependent workload")


def quota_pair(producer_quota: int) -> tuple[int, int]:
    background_quota = 100 - producer_quota
    if (
        producer_quota not in SUPPORTED_QUOTAS
        or background_quota not in SUPPORTED_QUOTAS
    ):
        raise ValueError("BOER candidate must have a profiled complementary quota")
    return producer_quota, background_quota


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def process_state(pid: int) -> str:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = text.rfind(")")
    if end < 0 or end + 2 >= len(text):
        raise RuntimeError(f"malformed process state for PID {pid}")
    return text[end + 2]


def wait_paused(process: subprocess.Popen[str], timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"background exited {process.returncode}: {stdout} {stderr}"
            )
        if process_state(process.pid) in {"T", "t"}:
            return
        time.sleep(0.02)
    raise TimeoutError("background did not reach the post-warmup barrier")


def stop_background(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        try:
            os.kill(process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        try:
            os.kill(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


def metrics_from_results(
    pipeline: dict[str, Any],
    background: dict[str, Any],
    deadline_us: float,
    iterations: int,
) -> dict[str, float]:
    if iterations <= 0 or not math.isfinite(deadline_us) or deadline_us <= 0:
        raise ValueError("invalid evaluation contract")
    if pipeline.get("status") != "ok" or pipeline.get("checksum_failures") != 0:
        raise ValueError("pipeline correctness failed")
    if pipeline.get("iterations") != iterations:
        raise ValueError("pipeline iteration count differs from the contract")
    deadline_mode = pipeline.get("deadline_mode", "wall")
    p99_value = (
        pipeline.get("stage_latency_us", {}).get(
            "validation_excluded_end_to_end_p99"
        )
        if deadline_mode == "validation-excluded"
        else pipeline.get("end_to_end_us", {}).get("p99")
    )
    p99_us = finite(p99_value, "pipeline p99")
    misses = finite(pipeline.get("deadline_misses"), "deadline misses")
    if misses < 0 or misses > iterations:
        raise ValueError("deadline miss count is invalid")
    return {
        # Preserve BOER's published p99 feasibility rule. DMR is reported but
        # does not alter BOER's search decision.
        "feasible": float(p99_us <= deadline_us),
        "slo_limit_ms": deadline_us / 1000.0,
        "worst_p99_ms": p99_us / 1000.0,
        "served_rps_0": finite(pipeline.get("pipeline_rps"), "pipeline rps"),
        "served_rps_1": finite(
            background.get("throughput_per_second"), "background rps"
        ),
        "deadline_miss_rate": misses / float(iterations),
        "dmr_target": 0.0005,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    parser.add_argument("--deadline-us", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--workload",
        choices=("resnet-control", "whisper-projection"),
        default="resnet-control",
    )
    args = parser.parse_args()
    if not math.isfinite(args.deadline_us) or args.deadline_us <= 0:
        raise ValueError("deadline-us must be positive and finite")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iteration counts are invalid")

    candidate_id = os.environ.get("BOER_CANDIDATE_ID")
    producer_quota, background_quota = quota_pair(
        int(os.environ["BOER_SM_PERCENT"])
    )
    offered_rps = int(os.environ["BOER_OFFERED_RPS"])
    if not candidate_id or offered_rps <= 0:
        raise ValueError("invalid BOER candidate environment")

    mig_env_path = pathlib.Path(
        os.environ.get("MIG_ENV", "/tmp/jdg-mps-1g/mig.env")
    )
    mig: dict[str, str] = {}
    for line in mig_env_path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        mig[key] = value
    required = (
        "JDG_MIG_SMALL_UUID",
        "JDG_MIG_BIG_UUID",
        "JDG_MPS_PIPE_DIRECTORY",
        "JDG_MPS_LOG_DIRECTORY",
    )
    if any(not mig.get(key) for key in required):
        raise ValueError("MIG environment is incomplete")

    run_dir = args.result_root.resolve() / f"{candidate_id}-{time.time_ns()}"
    run_dir.mkdir(parents=True)
    producer_model, deadline_mode = workload_contract(args.workload)
    producer_engine = (
        args.repo
        / "models"
        / "engines"
        / f"mig-1g-q{producer_quota}"
        / f"{producer_model}.engine"
    )
    background_engine = (
        args.repo
        / "models"
        / "engines"
        / f"mig-1g-q{background_quota}"
        / "distilbert-sst2.engine"
    )
    for path in (background_engine, producer_engine):
        if not path.is_file():
            raise FileNotFoundError(path)

    background_env = os.environ.copy()
    background_env.update(
        {
            "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_SMALL_UUID"],
            "CUDA_MPS_PIPE_DIRECTORY": mig["JDG_MPS_PIPE_DIRECTORY"],
            "CUDA_MPS_LOG_DIRECTORY": mig["JDG_MPS_LOG_DIRECTORY"],
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(background_quota),
        }
    )
    background_command = [
        "taskset",
        "--cpu-list",
        "0",
        str(args.repo / "build-r39" / "jdg-trt-bench"),
        "--engine",
        str(background_engine),
        "--model-name",
        "distilbert-sst2",
        "--role",
        "pressure",
        "--duration-seconds",
        "3600",
        "--period-ms",
        str(1000.0 / offered_rps),
        "--warmup",
        str(args.warmup),
        "--include-transfers",
        "true",
        "--priority",
        "default",
        "--start-paused",
        "true",
    ]
    background = subprocess.Popen(
        background_command,
        cwd=args.repo,
        env=background_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    background_stdout = ""
    background_stderr = ""
    pipeline: dict[str, Any] | None = None
    pipeline_failure: str | None = None
    try:
        wait_paused(background)
        os.kill(background.pid, signal.SIGCONT)
        pipeline_command = [
            "taskset",
            "--cpu-list",
            "13",
            str(args.repo / "build-r39" / "jdg-mig-trt-pipeline"),
            "--producer-engine",
            str(producer_engine),
            "--producer",
            mig["JDG_MIG_SMALL_UUID"],
            "--consumer",
            mig["JDG_MIG_BIG_UUID"],
            "--producer-mps-pipe",
            mig["JDG_MPS_PIPE_DIRECTORY"],
            "--producer-quota",
            str(producer_quota),
            "--transport",
            "registered-direct",
            "--deadline-us",
            str(args.deadline_us),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--workload",
            args.workload,
            "--deadline-mode",
            deadline_mode,
            "--trace-csv",
            str(run_dir / "pipeline.csv"),
        ]
        completed = subprocess.run(
            pipeline_command,
            cwd=args.repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        (run_dir / "pipeline.stderr").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode == 0:
            pipeline = json.loads(completed.stdout)
            (run_dir / "pipeline.json").write_text(
                json.dumps(pipeline, indent=2) + "\n", encoding="utf-8"
            )
        else:
            pipeline_failure = f"pipeline-exit-{completed.returncode}"
            (run_dir / "pipeline.stdout").write_text(
                completed.stdout, encoding="utf-8"
            )
    finally:
        background_stdout, background_stderr = stop_background(background)
        (run_dir / "background.stdout").write_text(
            background_stdout, encoding="utf-8"
        )
        (run_dir / "background.stderr").write_text(
            background_stderr, encoding="utf-8"
        )

    background_result: dict[str, Any] | None = None
    if background.returncode == 0:
        background_result = json.loads(background_stdout)
        (run_dir / "background.json").write_text(
            json.dumps(background_result, indent=2) + "\n", encoding="utf-8"
        )
    else:
        pipeline_failure = pipeline_failure or f"background-exit-{background.returncode}"

    if pipeline is None or background_result is None:
        failure = {
            "candidate_id": candidate_id,
            "producer_quota": producer_quota,
            "background_quota": background_quota,
            "offered_rps": offered_rps,
            "failure_reason": pipeline_failure or "hardware-execution-failure",
        }
        (run_dir / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "feasible": 0.0,
                    "slo_limit_ms": args.deadline_us / 1000.0,
                    "worst_p99_ms": args.deadline_us / 100.0,
                    "served_rps_0": 1e-9,
                    "served_rps_1": 1e-9,
                    "deadline_miss_rate": 1.0,
                    "dmr_target": 0.0005,
                    "result_dir": str(run_dir),
                }
            )
        )
        return 0

    metrics: dict[str, Any] = metrics_from_results(
        pipeline, background_result, args.deadline_us, args.iterations
    )
    metrics["producer_quota"] = producer_quota
    metrics["background_quota"] = background_quota
    metrics["workload"] = args.workload
    metrics["deadline_mode"] = deadline_mode
    metrics["result_dir"] = str(run_dir)
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
