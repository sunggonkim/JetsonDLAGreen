#!/usr/bin/env python3
"""Evaluate drain-aware multimodal serving on a full, non-MIG GPU."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any


MODEL_BY_MODALITY = {
    "language": "distilbert-sst2",
    "audio": "whisper-tiny-encoder",
}
DEFAULT_PROFILE_GUARD_MS = {"language": 1.5, "audio": 2.0}
TRACE = (
    ("language",),
    ("audio",),
    ("language", "audio"),
    ("language", "audio", "language", "audio"),
    ("audio", "audio", "audio", "audio", "audio", "audio"),
    ("language", "language", "language", "language", "language", "language"),
)
POLICIES = (
    "static-q5",
    "static-q25",
    "priority-q25",
    "conservative-guard",
    "profiled-guard",
    "joint-governor",
)


@dataclass
class FeedbackState:
    admission_limit: int = 6
    guard_adjustment_ms: float = 0.25
    safe_epochs: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=pathlib.Path, required=True)
    parser.add_argument("--engine-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--mps-pipe", type=pathlib.Path, required=True)
    parser.add_argument("--mps-log", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--period-ms", type=float, default=12.0)
    parser.add_argument("--slo-factor", type=float, default=1.10)
    parser.add_argument("--dmr-target", type=float, default=0.0005)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--pressure-startup-seconds", type=float, default=0.25)
    parser.add_argument("--critical-cpu", default="12")
    parser.add_argument("--pressure-cpus", default="0-11")
    parser.add_argument("--policy-order", default=",".join(POLICIES))
    parser.add_argument(
        "--guard-override-ms",
        type=float,
        help="replace the temporal guard for guarded policies (sensitivity only)",
    )
    parser.add_argument("--experiment-label", default="main")
    parser.add_argument(
        "--language-guard-ms",
        type=float,
        default=DEFAULT_PROFILE_GUARD_MS["language"],
    )
    parser.add_argument(
        "--audio-guard-ms",
        type=float,
        default=DEFAULT_PROFILE_GUARD_MS["audio"],
    )
    return parser.parse_args()


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


def engine_path(root: pathlib.Path, tag: str, model: str) -> pathlib.Path:
    path = root / tag / f"{model}.engine"
    if not path.is_file():
        raise FileNotFoundError(f"missing TensorRT engine: {path}")
    return path


def run_json(
    command: list[str], env: dict[str, str], timeout: float
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    result = json.loads(completed.stdout)
    if result.get("schema_version") != 1:
        raise RuntimeError("benchmark returned an unsupported schema")
    return result


def critical_command(
    args: argparse.Namespace,
    trace: pathlib.Path,
    deadline_ms: float | None,
    priority: str,
    gate_pids: list[int] | None = None,
    guard_ms: float = 0.0,
) -> list[str]:
    command = [
        "taskset",
        "--cpu-list",
        args.critical_cpu,
        str(args.bench),
        "--engine",
        str(engine_path(args.engine_root, "full", "resnet50-v2")),
        "--model-name",
        "resnet50-v2",
        "--role",
        "benchmark",
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--burst-size",
        str(args.burst_size),
        "--period-ms",
        str(args.period_ms),
        "--include-transfers",
        "true",
        "--priority",
        priority,
        "--trace",
        str(trace),
    ]
    if deadline_ms is not None:
        command.extend(("--deadline-ms", str(deadline_ms)))
    if gate_pids:
        command.extend(
            ("--gate-pids", ",".join(str(pid) for pid in gate_pids))
        )
        command.extend(("--guard-ms", str(guard_ms)))
    return command


def pressure_command(
    args: argparse.Namespace,
    modality: str,
    quota: int,
    cpu: int,
    priority: str,
) -> list[str]:
    model = MODEL_BY_MODALITY[modality]
    return [
        "taskset",
        "--cpu-list",
        str(cpu),
        str(args.bench),
        "--engine",
        str(engine_path(args.engine_root, f"full-q{quota}", model)),
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
        "--priority",
        priority,
    ]


def start_pressures(
    actions: list[tuple[str, int]],
    args: argparse.Namespace,
    base_env: dict[str, str],
    priority: str,
) -> list[tuple[str, int, subprocess.Popen[str]]]:
    cpus = expand_cpu_list(args.pressure_cpus)
    processes: list[tuple[str, int, subprocess.Popen[str]]] = []
    for index, (modality, quota) in enumerate(actions):
        environment = base_env | {
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(quota),
        }
        process = subprocess.Popen(
            pressure_command(args, modality, quota, cpus[index % len(cpus)], priority),
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
            process.send_signal(signal.SIGCONT)
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
            if result.get("schema_version") != 1:
                raise RuntimeError("pressure worker returned an unsupported schema")
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


def policy_action(
    policy: str, offered: tuple[str, ...], state: FeedbackState
) -> list[tuple[str, int]]:
    quota = 5 if policy == "static-q5" else 25
    limit = state.admission_limit if policy == "joint-governor" else len(offered)
    return [(modality, quota) for modality in offered[:limit]]


def guard_for(
    policy: str,
    actions: list[tuple[str, int]],
    state: FeedbackState,
    profile_guard_ms: dict[str, float] | None = None,
    override_ms: float | None = None,
) -> float:
    profile = profile_guard_ms or DEFAULT_PROFILE_GUARD_MS
    if override_ms is not None and policy in {
        "conservative-guard",
        "profiled-guard",
        "joint-governor",
    }:
        return override_ms
    if policy == "conservative-guard":
        return 6.0
    if policy in {"profiled-guard", "joint-governor"}:
        profiled = max(profile[modality] for modality, _ in actions)
        adjustment = state.guard_adjustment_ms if policy == "joint-governor" else 0.5
        return profiled + adjustment
    return 0.0


def update_feedback(
    state: FeedbackState,
    *,
    violated: bool,
    critical_p99_ms: float,
    deadline_ms: float,
) -> None:
    if violated:
        state.guard_adjustment_ms = min(2.0, state.guard_adjustment_ms + 0.5)
        state.admission_limit = max(1, state.admission_limit - 1)
        state.safe_epochs = 0
        return
    if critical_p99_ms < deadline_ms * 0.95:
        state.safe_epochs += 1
        if state.safe_epochs >= 2:
            state.guard_adjustment_ms = max(0.0, state.guard_adjustment_ms - 0.25)
            state.admission_limit = min(6, state.admission_limit + 1)
            state.safe_epochs = 0
    else:
        state.safe_epochs = 0


def parse_dmon(text: str) -> dict[str, float | int | None]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            fields = stripped.split()
            if len(fields) >= 6:
                rows.append(fields)

    def values(index: int) -> list[float]:
        parsed: list[float] = []
        for row in rows:
            if row[index] != "-":
                parsed.append(float(row[index]))
        return parsed

    def mean_or_none(items: list[float]) -> float | None:
        return statistics.fmean(items) if items else None

    def max_or_none(items: list[float]) -> float | None:
        return max(items) if items else None

    power = values(1)
    temperature = values(2)
    sm = values(4)
    memory = values(5)
    return {
        "samples": len(rows),
        "power_w_mean": mean_or_none(power),
        "power_w_max": max_or_none(power),
        "temperature_c_max": max_or_none(temperature),
        "sm_utilization_mean": mean_or_none(sm),
        "memory_utilization_mean": mean_or_none(memory),
    }


def run_policy(
    policy: str,
    args: argparse.Namespace,
    base_env: dict[str, str],
    deadline_ms: float,
) -> dict[str, Any]:
    state = FeedbackState()
    epochs: list[dict[str, Any]] = []
    policy_start = time.monotonic()
    timeout = args.samples / args.burst_size * args.period_ms / 1000.0 + 60.0
    for epoch_index in range(args.epochs):
        offered = TRACE[epoch_index % len(TRACE)]
        actions = policy_action(policy, offered, state)
        guard_ms = guard_for(
            policy,
            actions,
            state,
            {
                "language": args.language_guard_ms,
                "audio": args.audio_guard_ms,
            },
            args.guard_override_ms,
        )
        pressure_priority = "low" if policy == "priority-q25" else "default"
        critical_priority = "high" if policy == "priority-q25" else "default"
        running = start_pressures(actions, args, base_env, pressure_priority)
        telemetry = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        trace = args.output.parent / "raw" / f"{policy}-e{epoch_index}.csv"
        epoch_start = time.monotonic()
        try:
            if running:
                time.sleep(args.pressure_startup_seconds)
            critical = run_json(
                critical_command(
                    args,
                    trace,
                    deadline_ms,
                    critical_priority,
                    [process.pid for _, _, process in running] if guard_ms > 0.0 else None,
                    guard_ms,
                ),
                base_env,
                timeout,
            )
        finally:
            pressures = stop_pressures(running)
            telemetry.terminate()
            telemetry_stdout, _ = telemetry.communicate(timeout=10.0)
        epoch_elapsed = time.monotonic() - epoch_start
        latency = critical["release_to_completion"]
        miss_rate = float(critical["deadline_miss_rate"] or 0.0)
        violated = (
            latency["p99_ms"] > deadline_ms or miss_rate > args.dmr_target
        )
        if policy == "joint-governor":
            update_feedback(
                state,
                violated=violated,
                critical_p99_ms=float(latency["p99_ms"]),
                deadline_ms=deadline_ms,
            )
        completions = {
            modality: sum(
                int(item["completed_requests"])
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
                "guard_ms": guard_ms,
                "critical_p50_ms": latency["p50_ms"],
                "critical_p99_ms": latency["p99_ms"],
                "critical_p999_ms": latency["p999_ms"],
                "deadline_misses": critical["deadline_misses"],
                "deadline_miss_rate": miss_rate,
                "gate_overhead_mean_ms": critical["gate_overhead"]["mean_ms"],
                "queue_delay_p99_ms": critical["queue_delay"]["p99_ms"],
                "violated": violated,
                "completed_by_modality": completions,
                "telemetry": parse_dmon(telemetry_stdout),
                "elapsed_seconds": epoch_elapsed,
            }
        )
    elapsed = time.monotonic() - policy_start
    completions = {
        modality: sum(epoch["completed_by_modality"][modality] for epoch in epochs)
        for modality in MODEL_BY_MODALITY
    }
    total_samples = args.epochs * args.samples
    return {
        "name": policy,
        "deadline_miss_rate": sum(epoch["deadline_misses"] for epoch in epochs)
        / total_samples,
        "violation_epoch_rate": sum(epoch["violated"] for epoch in epochs)
        / args.epochs,
        "critical_p99_ms_max": max(epoch["critical_p99_ms"] for epoch in epochs),
        "pressure_completed": sum(completions.values()),
        "pressure_goodput_per_second": sum(completions.values()) / elapsed,
        "goodput_by_modality": {
            modality: count / elapsed for modality, count in completions.items()
        },
        "rejected_tenants": sum(epoch["rejected_tenants"] for epoch in epochs),
        "gate_overhead_mean_ms": statistics.fmean(
            epoch["gate_overhead_mean_ms"] for epoch in epochs
        ),
        "elapsed_seconds": elapsed,
        "epochs": epochs,
    }


def main() -> int:
    args = parse_args()
    if (
        args.epochs <= 0
        or args.samples <= 0
        or args.warmup < 0
        or args.burst_size <= 0
        or args.samples % args.burst_size != 0
        or args.period_ms <= 0.0
        or args.dmr_target <= 0.0
        or args.dmr_target >= 1.0
        or args.calibration_repeats <= 0
        or (args.guard_override_ms is not None and args.guard_override_ms < 0.0)
        or args.language_guard_ms <= 0.0
        or args.audio_guard_ms <= 0.0
        or max(args.language_guard_ms, args.audio_guard_ms) + 2.0
        >= args.period_ms
        or (
            args.guard_override_ms is not None
            and args.guard_override_ms >= args.period_ms
        )
    ):
        raise SystemExit("invalid experiment dimensions")
    requested = args.policy_order.split(",")
    if (
        not requested
        or len(requested) != len(set(requested))
        or not set(requested).issubset(POLICIES)
    ):
        raise SystemExit("policy-order must contain unique, known policies")
    base_env = os.environ.copy() | {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_MPS_PIPE_DIRECTORY": str(args.mps_pipe),
        "CUDA_MPS_LOG_DIRECTORY": str(args.mps_log),
    }
    base_env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.2/lib64:" + base_env.get(
        "LD_LIBRARY_PATH", ""
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "raw").mkdir(parents=True, exist_ok=True)
    timeout = args.samples / args.burst_size * args.period_ms / 1000.0 + 60.0
    calibrations = [
        run_json(
            critical_command(
                args,
                args.output.parent / "raw" / f"isolated-r{repeat}.csv",
                None,
                "default",
            ),
            base_env,
            timeout,
        )
        for repeat in range(1, args.calibration_repeats + 1)
    ]
    isolated_p99_values = [
        float(item["release_to_completion"]["p99_ms"]) for item in calibrations
    ]
    deadline_ms = max(isolated_p99_values) * args.slo_factor
    policies = [
        run_policy(policy, args, base_env, deadline_ms) for policy in requested
    ]
    output = {
        "schema_version": 1,
        "config": {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples,
            "warmup": args.warmup,
            "burst_size": args.burst_size,
            "period_ms": args.period_ms,
            "slo_factor": args.slo_factor,
            "dmr_target": args.dmr_target,
            "calibration_repeats": args.calibration_repeats,
            "policy_order": requested,
            "guard_override_ms": args.guard_override_ms,
            "experiment_label": args.experiment_label,
            "profile_guard_ms": {
                "language": args.language_guard_ms,
                "audio": args.audio_guard_ms,
            },
            "trace": [list(epoch) for epoch in TRACE],
            "includes_transfers": True,
            "mig_enabled": False,
        },
        "isolated": calibrations,
        "isolated_p99_ms": isolated_p99_values,
        "deadline_ms": deadline_ms,
        "policies": policies,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
