#!/usr/bin/env python3
"""Run deadline-aware policies on real TensorRT multimodal workloads."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any


MODEL_BY_MODALITY = {
    "language": "distilbert-sst2",
    "audio": "whisper-tiny-encoder",
}
PROFILED_QUOTA = {"language": 50, "audio": 100}
TRACE = (
    ("language",),
    ("audio",),
    ("language", "audio"),
    ("language", "audio", "language", "audio"),
    ("audio", "audio", "audio", "audio", "audio", "audio"),
    ("language", "language", "language", "language", "language", "language"),
)
POLICIES = (
    "static-q25",
    "static-q100",
    "time-division",
    "profiled",
    "joint-governor",
)


@dataclass
class FeedbackState:
    admission_limit: int = 4
    safe_epochs: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=pathlib.Path, required=True)
    parser.add_argument("--engine-root", type=pathlib.Path, required=True)
    parser.add_argument("--mig-env", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--slo-factor", type=float, default=1.10)
    parser.add_argument("--pressure-startup-seconds", type=float, default=0.5)
    parser.add_argument("--serial-window-seconds", type=float, default=1.0)
    parser.add_argument("--critical-cpu", default="12")
    parser.add_argument("--pressure-cpus", default="0-11")
    parser.add_argument("--policy-order", default=",".join(POLICIES))
    return parser.parse_args()


def load_env(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def expand_cpu_list(specification: str) -> list[int]:
    cpus: list[int] = []
    for item in specification.split(","):
        bounds = item.split("-", 1)
        if len(bounds) == 1:
            cpus.append(int(bounds[0]))
        else:
            first, last = (int(value) for value in bounds)
            if first > last:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.extend(range(first, last + 1))
    if not cpus or len(cpus) != len(set(cpus)):
        raise ValueError("CPU list must be non-empty and contain no duplicates")
    return cpus


def policy_action(
    policy: str, offered: tuple[str, ...], state: FeedbackState
) -> list[tuple[str, int]]:
    if policy == "static-q25":
        return [(modality, 25) for modality in offered]
    if policy in {"static-q100", "time-division"}:
        return [(modality, 100) for modality in offered]
    if policy == "profiled":
        return [(modality, PROFILED_QUOTA[modality]) for modality in offered[:2]]
    if policy == "joint-governor":
        return [
            (modality, PROFILED_QUOTA[modality])
            for modality in offered[: state.admission_limit]
        ]
    raise ValueError(f"unknown policy: {policy}")


def update_feedback(
    state: FeedbackState,
    *,
    violated: bool,
    maximum_limit: int = 6,
) -> None:
    if violated:
        state.admission_limit = max(0, state.admission_limit - 1)
        state.safe_epochs = 0
        return
    state.safe_epochs += 1
    if state.safe_epochs >= 2 and state.admission_limit < maximum_limit:
        state.admission_limit += 1
        state.safe_epochs = 0


def pinned(cpu: int | str, command: list[str]) -> list[str]:
    return ["taskset", "--cpu-list", str(cpu), *command]


def engine_path(root: pathlib.Path, tag: str, model: str) -> pathlib.Path:
    path = root / tag / f"{model}.engine"
    if not path.is_file():
        raise FileNotFoundError(f"missing TensorRT engine: {path}")
    return path


def critical_command(
    args: argparse.Namespace, deadline_ms: float | None, trace: pathlib.Path
) -> list[str]:
    command = [
        str(args.bench),
        "--engine",
        str(engine_path(args.engine_root, "mig-2g", "resnet10-detection")),
        "--model-name",
        "resnet10-detection",
        "--role",
        "benchmark",
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--include-transfers",
        "true",
        "--trace",
        str(trace),
    ]
    if deadline_ms is not None:
        command.extend(("--deadline-ms", str(deadline_ms)))
    return pinned(args.critical_cpu, command)


def pressure_command(
    args: argparse.Namespace, modality: str, quota: int, cpu: int
) -> list[str]:
    model = MODEL_BY_MODALITY[modality]
    return pinned(
        cpu,
        [
            str(args.bench),
            "--engine",
            str(engine_path(args.engine_root, f"mig-1g-q{quota}", model)),
            "--model-name",
            model,
            "--role",
            "pressure",
            "--duration-seconds",
            "3600",
            "--warmup",
            str(args.warmup),
            "--include-transfers",
            "true",
        ],
    )


def run_json(
    command: list[str], env: dict[str, str], timeout: float = 120.0
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return json.loads(completed.stdout)


def start_pressures(
    actions: list[tuple[str, int]],
    args: argparse.Namespace,
    base_env: dict[str, str],
    mig: dict[str, str],
) -> list[tuple[str, int, subprocess.Popen[str]]]:
    cpus = expand_cpu_list(args.pressure_cpus)
    processes: list[tuple[str, int, subprocess.Popen[str]]] = []
    for index, (modality, quota) in enumerate(actions):
        environment = base_env | {
            "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_SMALL_UUID"],
            "CUDA_MPS_PIPE_DIRECTORY": mig["JDG_MPS_PIPE_DIRECTORY"],
            "CUDA_MPS_LOG_DIRECTORY": mig["JDG_MPS_LOG_DIRECTORY"],
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(quota),
        }
        process = subprocess.Popen(
            pressure_command(args, modality, quota, cpus[index % len(cpus)]),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((modality, quota, process))
    return processes


def stop_pressures(
    processes: list[tuple[str, int, subprocess.Popen[str]]],
) -> list[dict[str, Any]]:
    for _, _, process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    outputs: list[dict[str, Any]] = []
    try:
        for modality, quota, process in processes:
            stdout, stderr = process.communicate(timeout=30.0)
            if process.returncode != 0:
                raise RuntimeError(
                    f"{modality} q{quota} failed ({process.returncode}): {stderr}"
                )
            result = json.loads(stdout)
            result["modality"] = modality
            result["quota_percent"] = quota
            outputs.append(result)
    except Exception:
        for _, _, process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
        raise
    return outputs


def pressure_completed(result: dict[str, Any]) -> int:
    if "completed_launches" in result:
        return int(result["completed_launches"])
    if "completed_requests" in result:
        return int(result["completed_requests"])
    raise KeyError("pressure result missing completed_launches/completed_requests")


def run_policy(
    policy: str,
    args: argparse.Namespace,
    base_env: dict[str, str],
    mig: dict[str, str],
    deadline_ms: float,
) -> dict[str, Any]:
    state = FeedbackState()
    epochs: list[dict[str, Any]] = []
    policy_start = time.monotonic()
    for epoch_index in range(args.epochs):
        offered = TRACE[epoch_index % len(TRACE)]
        actions = policy_action(policy, offered, state)
        epoch_start = time.monotonic()
        trace = args.output.parent / "raw" / f"{policy}-e{epoch_index}.csv"
        trace.parent.mkdir(parents=True, exist_ok=True)
        critical_env = base_env | {
            "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_BIG_UUID"]
        }
        pressures: list[dict[str, Any]]
        if policy == "time-division":
            critical = run_json(
                critical_command(args, deadline_ms, trace), critical_env
            )
            running = start_pressures(actions, args, base_env, mig)
            time.sleep(args.serial_window_seconds)
            pressures = stop_pressures(running)
        else:
            running = start_pressures(actions, args, base_env, mig)
            try:
                if running:
                    time.sleep(args.pressure_startup_seconds)
                critical = run_json(
                    critical_command(args, deadline_ms, trace), critical_env
                )
            finally:
                pressures = stop_pressures(running)

        latency = critical["release_to_completion"]
        miss_rate = float(critical["deadline_miss_rate"] or 0.0)
        violated = latency["p99_ms"] > deadline_ms or miss_rate > 0.01
        if policy == "joint-governor":
            update_feedback(state, violated=violated)
        by_modality = {
            modality: sum(
                pressure_completed(item)
                for item in pressures
                if item["modality"] == modality
            )
            for modality in MODEL_BY_MODALITY
        }
        epochs.append(
            {
                "epoch": epoch_index,
                "offered_modalities": list(offered),
                "actions": [
                    {"modality": modality, "quota_percent": quota}
                    for modality, quota in actions
                ],
                "offered_tenants": len(offered),
                "admitted_tenants": len(actions),
                "rejected_tenants": len(offered) - len(actions),
                "critical_p50_ms": latency["p50_ms"],
                "critical_p99_ms": latency["p99_ms"],
                "critical_p999_ms": latency["p999_ms"],
                "critical_max_ms": latency["max_ms"],
                "deadline_misses": critical["deadline_misses"],
                "deadline_miss_rate": miss_rate,
                "violated": violated,
                "completed_by_modality": by_modality,
                "elapsed_seconds": time.monotonic() - epoch_start,
            }
        )
    elapsed = time.monotonic() - policy_start
    completions = {
        modality: sum(epoch["completed_by_modality"][modality] for epoch in epochs)
        for modality in MODEL_BY_MODALITY
    }
    total_misses = sum(epoch["deadline_misses"] for epoch in epochs)
    return {
        "name": policy,
        "deadline_miss_rate": total_misses / (args.epochs * args.samples),
        "violation_epoch_rate": sum(epoch["violated"] for epoch in epochs)
        / args.epochs,
        "critical_p99_ms_max": max(epoch["critical_p99_ms"] for epoch in epochs),
        "pressure_completed": sum(completions.values()),
        "pressure_goodput_per_second": sum(completions.values()) / elapsed,
        "goodput_by_modality": {
            modality: count / elapsed for modality, count in completions.items()
        },
        "rejected_tenants": sum(epoch["rejected_tenants"] for epoch in epochs),
        "elapsed_seconds": elapsed,
        "epochs": epochs,
    }


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.samples <= 0 or args.warmup < 0:
        raise SystemExit("epochs and samples must be positive")
    requested = args.policy_order.split(",")
    if len(requested) != len(POLICIES) or set(requested) != set(POLICIES):
        raise SystemExit("policy-order must contain each policy exactly once")
    mig = load_env(args.mig_env)
    base_env = os.environ.copy()
    base_env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.2/lib64:" + base_env.get(
        "LD_LIBRARY_PATH", ""
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "raw").mkdir(parents=True, exist_ok=True)
    calibration_env = base_env | {
        "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_BIG_UUID"]
    }
    calibration = run_json(
        critical_command(args, None, args.output.parent / "raw" / "isolated.csv"),
        calibration_env,
    )
    isolated_p99 = calibration["release_to_completion"]["p99_ms"]
    deadline_ms = isolated_p99 * args.slo_factor
    policies = [
        run_policy(policy, args, base_env, mig, deadline_ms)
        for policy in requested
    ]
    output = {
        "schema_version": 1,
        "config": {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples,
            "warmup": args.warmup,
            "slo_factor": args.slo_factor,
            "policy_order": requested,
            "trace": [list(epoch) for epoch in TRACE],
            "includes_transfers": True,
        },
        "isolated": calibration,
        "deadline_ms": deadline_ms,
        "policies": policies,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
