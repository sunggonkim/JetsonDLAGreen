#!/usr/bin/env python3
"""Evaluate deadline-safe slack borrowing across Thor MIG instances."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

try:
    from tegrastats_telemetry import JsonlTelemetryWriter, TegrastatsMonitor
except ModuleNotFoundError:
    from runtime.tegrastats_telemetry import (
        JsonlTelemetryWriter,
        TegrastatsMonitor,
    )


MODEL_BY_MODALITY = {
    "language": "distilbert-sst2",
    "audio": "whisper-tiny-encoder",
}
RESIDENT_QUOTAS = (25, 50, 100)
TRACE = (
    ("language",),
    ("audio",),
    ("language", "audio"),
    ("language", "audio", "language", "audio"),
    ("audio", "audio", "audio", "audio", "audio", "audio"),
    ("language", "language", "language", "language", "language", "language"),
)
MULTIMODAL_SCENARIO_TRACE = (
    ("audio", "language"),
    ("audio", "language", "audio", "language"),
    ("audio", "language", "audio", "language", "audio", "language"),
    ("audio", "language"),
    ("audio", "language", "audio", "language"),
    ("audio", "language", "audio", "language", "audio", "language"),
)
SCENARIO_TRACES = {
    "independent": MULTIMODAL_SCENARIO_TRACE,
    # Each language request consumes the preceding audio result.  The
    # benchmark enforces this with a one-byte completion token pipe.
    "dependent": MULTIMODAL_SCENARIO_TRACE,
}
POLICIES = (
    "static-mig",
    "resident-full-gate",
    "same-mig",
    "uncoordinated-borrow",
    "fixed-borrow",
    "fixed-full-gate",
    "mig-governor",
)
TELEMETRY_REQUIRED_FIELDS = (
    "ram",
    "mem_available",
    "cpu",
    "temperature:soc012",
    "temperature:tj",
    "power:VIN",
)
TEGRASTATS_REQUESTED_INTERVAL_MS = 75.0
TELEMETRY_INTERVAL_MS = 100.0
TELEMETRY_REQUIRED_FRACTION = 0.8
TELEMETRY_STALE_AFTER_MS = 300.0
THERMAL_PILOT_MINIMUM_SECONDS = 600.0
THERMAL_PILOT_MAXIMUM_SECONDS = 900.0
THERMAL_PILOT_WINDOW_SECONDS = 180.0
THERMAL_STABILITY_CHECKPOINT_SECONDS = 30.0
THERMAL_STABILITY_CHECKPOINT_MAX_LATENESS_SECONDS = 1.0
THERMAL_REQUIRED_STABLE_CHECKPOINTS = 3
THERMAL_MAXIMUM_SLOPE_C_PER_MINUTE = 0.2
THERMAL_STABILITY_SENSOR = "soc012"
THERMAL_SAFETY_SENSOR = "tj"
THERMAL_HANDOFF_MAX_MS = 500.0
THERMAL_HANDOFF_BOUNDARY = "thermal_measurement_end"
THERMAL_ACTIVE_STABLE_ENDPOINTS = 3
THERMAL_ACTIVE_STABLE_SPACING_SECONDS = 1.0
THERMAL_QUALIFICATION_MAX_ATTEMPTS = 3
THERMAL_PROTOCOL_SCHEMA_VERSION = 4
GUARD_LOCK_SCHEMA_VERSION = 3
IMPLEMENTATION_FILES = (
    "analysis/freeze_p9_deadline.py",
    "analysis/freeze_p9_guard.py",
    "analysis/freeze_p9_thermal.py",
    "analysis/summarize_mig_slack_governor.py",
    "benchmarks/trt_inference.cpp",
    "include/jetson_dla_green/stats.hpp",
    "runtime/mig_slack_governor.py",
    "runtime/profile_p9_guard.py",
    "runtime/tegrastats_telemetry.py",
    "scripts/configure_thor_mig.sh",
    "scripts/run_p9_deadline_calibration.sh",
    "scripts/run_p9_guard_calibration.sh",
    "scripts/run_p9_mig_slack_governor.sh",
    "scripts/run_p9_repeated.sh",
    "scripts/run_p9_thermal_pilot.sh",
)


@dataclass(frozen=True)
class WorkerAction:
    tenant_id: int
    modality: str
    placement: str
    quota_percent: int


@dataclass(frozen=True)
class PlacementPlan:
    residents: tuple[WorkerAction, ...]
    borrowers: tuple[WorkerAction, ...]


@dataclass
class FeedbackState:
    resident_admission_limit: int = 6
    resident_quota_index: int = len(RESIDENT_QUOTAS) - 1
    borrower_limit: int = 6
    guard_adjustment_ms: float = 0.0
    safe_epochs: int = 0

    @property
    def resident_quota_percent(self) -> int:
        return RESIDENT_QUOTAS[self.resident_quota_index]


@dataclass
class RunningWorker:
    action: WorkerAction
    process: subprocess.Popen[str]
    cpu: int
    device: str
    warmup: int
    period_ms: float = 0.0


@dataclass(frozen=True)
class DependencyPipe:
    upstream_tenant_id: int
    downstream_tenant_id: int
    read_fd: int
    write_fd: int


@dataclass
class TelemetrySession:
    monitor: TegrastatsMonitor
    tail_process: subprocess.Popen[str]
    reader_thread: threading.Thread
    reader_errors: list[str]


def offered_for_epoch(
    epoch_index: int, scenario: str = "independent"
) -> tuple[str, ...]:
    if epoch_index < 0:
        raise ValueError("epoch index must be non-negative")
    try:
        trace = SCENARIO_TRACES[scenario]
    except KeyError as error:
        raise ValueError(f"unknown workload scenario: {scenario}") from error
    offered = trace[epoch_index % len(trace)]
    return offered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=pathlib.Path, required=True)
    parser.add_argument("--engine-root", type=pathlib.Path, required=True)
    parser.add_argument("--mig-env", type=pathlib.Path, required=True)
    parser.add_argument("--big-mps-pipe", type=pathlib.Path, required=True)
    parser.add_argument("--big-mps-log", type=pathlib.Path, required=True)
    parser.add_argument("--telemetry-log", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIO_TRACES),
        default="independent",
        help="independent concurrent modalities or audio->language dependency",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--period-ms", type=float, default=20.0)
    parser.add_argument(
        "--pressure-rps-per-tenant",
        type=float,
        default=0.0,
        help="offered pressure rate per tenant; zero keeps the saturated workload",
    )
    parser.add_argument("--slo-factor", type=float, default=1.10)
    parser.add_argument(
        "--deadline-ms",
        type=float,
        help="fixed deadline (otherwise pooled isolated p99 times slo-factor)",
    )
    parser.add_argument(
        "--deadline-source",
        choices=("application", "frozen-isolated-p99-factor", "fixed-explicit"),
        help="provenance label required for formal fixed-deadline runs",
    )
    parser.add_argument("--deadline-lock-sha256")
    parser.add_argument("--thermal-lock-sha256")
    parser.add_argument("--guard-lock", type=pathlib.Path)
    parser.add_argument("--guard-lock-sha256")
    parser.add_argument("--dmr-target", type=float, default=0.0005)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--readiness-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--critical-cpu", default="12")
    parser.add_argument("--pressure-cpus", default="0-10")
    parser.add_argument("--mps-cpu", default="11")
    parser.add_argument("--telemetry-cpu", default="13")
    parser.add_argument("--policy-order", default=",".join(POLICIES))
    parser.add_argument("--borrower-quota", type=int, default=25)
    parser.add_argument("--language-guard-ms", type=float, default=1.5)
    parser.add_argument("--audio-guard-ms", type=float, default=2.0)
    parser.add_argument(
        "--guard-override-ms",
        type=float,
        help="replace the fixed and adaptive guard (sensitivity only)",
    )
    parser.add_argument("--experiment-label", default="main")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--thermal-pilot-seconds", type=float)
    parser.add_argument("--thermal-target-c", type=float)
    parser.add_argument("--thermal-tolerance-c", type=float, default=1.0)
    parser.add_argument("--thermal-window-seconds", type=float, default=60.0)
    parser.add_argument(
        "--thermal-max-slope-c-per-minute",
        type=float,
        default=THERMAL_MAXIMUM_SLOPE_C_PER_MINUTE,
    )
    parser.add_argument(
        "--thermal-timeout-seconds",
        type=float,
        default=THERMAL_PILOT_MAXIMUM_SECONDS,
    )
    parser.add_argument("--thermal-hard-limit-c", type=float)
    parser.add_argument(
        "--thermal-stability-sensor",
        choices=(THERMAL_STABILITY_SENSOR,),
        default=THERMAL_STABILITY_SENSOR,
    )
    parser.add_argument(
        "--thermal-safety-sensor",
        choices=(THERMAL_SAFETY_SENSOR,),
        default=THERMAL_SAFETY_SENSOR,
    )
    parser.add_argument(
        "--thermal-handoff-max-ms",
        type=float,
        default=THERMAL_HANDOFF_MAX_MS,
    )
    parser.add_argument("--max-isolated-drift-fraction", type=float, default=0.05)
    return parser.parse_args()


def load_env(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    required = {
        "JDG_MIG_BIG_UUID",
        "JDG_MIG_SMALL_UUID",
        "JDG_MPS_PIPE_DIRECTORY",
        "JDG_MPS_LOG_DIRECTORY",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"MIG environment is missing: {', '.join(missing)}")
    return values


def guard_profile_from_lock(
    lock: dict[str, Any],
) -> dict[str, dict[int, dict[str, float]]]:
    if (
        lock.get("schema_version") != GUARD_LOCK_SCHEMA_VERSION
        or lock.get("kind") != "p9-quota-aware-guard-lock"
    ):
        raise ValueError("guard lock has an unsupported schema")
    raw_guards = lock.get("guards")
    expected_quotas = {
        "resident-1g": (25, 50, 100),
        "borrower-2g": (100,),
    }
    if not isinstance(raw_guards, dict) or set(raw_guards) != set(expected_quotas):
        raise ValueError("guard lock has invalid placements")
    profile: dict[str, dict[int, dict[str, float]]] = {}
    for placement, quotas in expected_quotas.items():
        raw_placement = raw_guards.get(placement)
        if not isinstance(raw_placement, dict) or set(raw_placement) != {
            str(quota) for quota in quotas
        }:
            raise ValueError(f"guard lock has invalid quotas for {placement}")
        profile[placement] = {}
        for quota in quotas:
            raw_modalities = raw_placement[str(quota)]
            if not isinstance(raw_modalities, dict) or set(raw_modalities) != set(
                MODEL_BY_MODALITY
            ):
                raise ValueError(
                    f"guard lock has invalid modalities for {placement}/q{quota}"
                )
            profile[placement][quota] = {}
            for modality in MODEL_BY_MODALITY:
                evidence = raw_modalities[modality]
                if not isinstance(evidence, dict):
                    raise ValueError("guard lock profile evidence must be an object")
                guard_ms = float(evidence.get("guard_ms", math.nan))
                if not math.isfinite(guard_ms) or guard_ms <= 0.0:
                    raise ValueError("guard lock contains a non-positive guard")
                profile[placement][quota][modality] = guard_ms
    return profile


def load_guard_lock(
    path: pathlib.Path, expected_sha256: str | None
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, float]]], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("guard lock SHA-256 differs from the requested artifact")
    try:
        lock = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"guard lock is invalid JSON: {error}") from error
    if not isinstance(lock, dict):
        raise ValueError("guard lock root must be an object")
    return lock, guard_profile_from_lock(lock), digest


def legacy_guard_profile(
    language_ms: float, audio_ms: float
) -> dict[str, dict[int, dict[str, float]]]:
    modalities = {"language": language_ms, "audio": audio_ms}
    return {
        "resident-1g": {
            quota: dict(modalities) for quota in RESIDENT_QUOTAS
        },
        "borrower-2g": {100: dict(modalities)},
    }


def guard_value_ms(
    profile: dict[str, Any], action: WorkerAction
) -> float:
    # Retain the flat form only for exploratory/unit compatibility. Formal P9
    # always consumes a quota-aware frozen lock.
    flat = profile.get(action.modality)
    if isinstance(flat, (int, float)) and not isinstance(flat, bool):
        value = float(flat)
    else:
        try:
            value = float(
                profile[action.placement][action.quota_percent][action.modality]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "guard profile does not cover "
                f"{action.placement}/q{action.quota_percent}/{action.modality}"
            ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("guard profile values must be finite and positive")
    return value


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
    if (
        not cpus
        or any(cpu < 0 for cpu in cpus)
        or len(cpus) != len(set(cpus))
    ):
        raise ValueError("CPU list must be non-empty and contain no duplicates")
    return cpus


def format_cpu_list(cpus: list[int]) -> str:
    ordered = sorted(cpus)
    ranges: list[str] = []
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def engine_path(root: pathlib.Path, tag: str, model: str) -> pathlib.Path:
    path = root / tag / f"{model}.engine"
    if not path.is_file():
        raise FileNotFoundError(f"missing TensorRT engine: {path}")
    return path


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_field(text: str, name: str) -> str:
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == name:
            return value.strip()
    raise ValueError(f"platform snapshot lacks {name}")


def hardware_fingerprint(directory: pathlib.Path) -> dict[str, Any]:
    required = {
        name: directory / name
        for name in (
            "nv_tegra_release.txt",
            "nvidia-smi.txt",
            "active-mig-instances.txt",
            "gpu-inventory.txt",
            "jetson-clocks.txt",
            "mps-affinity.tsv",
            "nvpmodel.txt",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing platform snapshots: " + ", ".join(sorted(missing))
        )
    release = required["nv_tegra_release.txt"].read_text(encoding="utf-8")
    smi = required["nvidia-smi.txt"].read_text(encoding="utf-8")
    active_mig = required["active-mig-instances.txt"].read_text(
        encoding="utf-8"
    )
    inventory = required["gpu-inventory.txt"].read_text(encoding="utf-8")
    clocks = required["jetson-clocks.txt"].read_text(encoding="utf-8")
    mps_affinity_text = required["mps-affinity.tsv"].read_text(encoding="utf-8")
    power = required["nvpmodel.txt"].read_text(encoding="utf-8")
    mig_match = re.search(r"MIG Mode\s+Current\s*:\s*(\S+)", smi)
    gpu_clock = re.search(
        r"gpu-gpc-0 MinFreq=(\d+) MaxFreq=(\d+)", clocks
    )
    emc_clock = re.search(r"EMC MinFreq=(\d+) MaxFreq=(\d+)", clocks)
    fan = re.search(
        r"FAN Dynamic Speed Control=(\w+).*?pwm1=(\d+)", clocks
    )
    online_match = re.search(r"^Online CPUs:\s*([^,]+),", clocks, re.MULTILINE)
    cpu_clocks = {
        int(cpu): {"min_khz": int(minimum), "max_khz": int(maximum)}
        for cpu, minimum, maximum in re.findall(
            r"^cpu(\d+):\s+.*?MinFreq=(\d+) MaxFreq=(\d+)",
            clocks,
            re.MULTILINE,
        )
    }
    active_instances = [
        {
            "gpu": int(gpu),
            "profile_name": profile_name,
            "profile_id": int(profile_id),
            "instance_id": int(instance_id),
            "placement_start": int(placement_start),
            "placement_size": int(placement_size),
        }
        for (
            gpu,
            profile_name,
            profile_id,
            instance_id,
            placement_start,
            placement_size,
        ) in re.findall(
            r"^\|\s+(\d+)\s+MIG\s+(\S+)\s+(\d+)\s+(\d+)\s+"
            r"(\d+):(\d+)\s+\|$",
            active_mig,
            re.MULTILINE,
        )
    ]
    mig_devices = {
        profile: uuid
        for profile, _device, uuid in re.findall(
            r"^\s+MIG\s+(\S+)\s+Device\s+(\d+):\s+"
            r"\(UUID:\s+(MIG-[^)]+)\)$",
            inventory,
            re.MULTILINE,
        )
    }
    mps_affinity: list[dict[str, Any]] = []
    for line in mps_affinity_text.splitlines():
        placement, role, pid, cpu_spec = line.split("\t")
        if role not in {"control", "server"} or not pid.isdigit():
            raise ValueError("invalid MPS affinity snapshot")
        mps_affinity.append(
            {
                "placement": placement,
                "role": role,
                "cpus": expand_cpu_list(cpu_spec),
            }
        )
    if mig_match is None or gpu_clock is None or emc_clock is None or fan is None:
        raise ValueError("platform snapshots lack MIG, clock, or fan state")
    if (
        online_match is None
        or not cpu_clocks
        or not active_instances
        or not mig_devices
        or not mps_affinity
    ):
        raise ValueError("platform snapshots lack CPU or active MIG state")
    online_cpus = expand_cpu_list(online_match.group(1).strip())
    return {
        "l4t_release": release.splitlines()[0],
        "driver_version": snapshot_field(smi, "Driver Version"),
        "cuda_version": snapshot_field(smi, "CUDA Version"),
        "gpu_product_name": snapshot_field(smi, "Product Name"),
        "gpu_architecture": snapshot_field(smi, "Product Architecture"),
        "gpu_uuid": snapshot_field(smi, "GPU UUID"),
        "mig_mode": mig_match.group(1),
        "active_mig_instances": sorted(
            active_instances, key=lambda item: int(item["profile_id"])
        ),
        "mig_device_uuid_by_profile": dict(sorted(mig_devices.items())),
        "power_mode": snapshot_field(power, "NV Power Mode"),
        "online_cpus": online_cpus,
        "cpu_clocks_khz": dict(sorted(cpu_clocks.items())),
        "mps_affinity": sorted(
            mps_affinity,
            key=lambda item: (str(item["placement"]), str(item["role"])),
        ),
        "gpu_clock_min_hz": int(gpu_clock.group(1)),
        "gpu_clock_max_hz": int(gpu_clock.group(2)),
        "emc_clock_min_hz": int(emc_clock.group(1)),
        "emc_clock_max_hz": int(emc_clock.group(2)),
        "fan_dynamic_control": fan.group(1),
        "fan_pwm": int(fan.group(2)),
    }


def artifact_hashes(args: argparse.Namespace) -> dict[str, Any]:
    root = pathlib.Path(__file__).resolve().parents[1]
    engines = {
        "critical-2g-resnet50-v2": engine_path(
            args.engine_root, "mig-2g", "resnet50-v2"
        )
    }
    for quota in RESIDENT_QUOTAS:
        for model in MODEL_BY_MODALITY.values():
            engines[f"resident-1g-q{quota}-{model}"] = engine_path(
                args.engine_root, f"mig-1g-q{quota}", model
            )
    for model in MODEL_BY_MODALITY.values():
        engines[f"borrower-2g-q{args.borrower_quota}-{model}"] = engine_path(
            args.engine_root,
            f"mig-2g-q{args.borrower_quota}",
            model,
        )
    return {
        "benchmark_sha256": file_sha256(args.bench),
        "engines_sha256": {
            name: file_sha256(path) for name, path in sorted(engines.items())
        },
        "implementation_sha256": {
            name: file_sha256(root / name) for name in IMPLEMENTATION_FILES
        },
    }


def plan_for(
    policy: str,
    offered: tuple[str, ...],
    state: FeedbackState,
    borrower_quota: int = 25,
) -> PlacementPlan:
    admitted = min(len(offered), state.resident_admission_limit)
    resident_quota = 100
    borrower_count = 0
    if policy == "same-mig":
        borrower_count = admitted
    elif policy in {
        "uncoordinated-borrow",
        "fixed-borrow",
        "fixed-full-gate",
    }:
        borrower_count = admitted // 2
    elif policy == "mig-governor":
        resident_quota = state.resident_quota_percent
        borrower_count = min(admitted // 2, state.borrower_limit)
    elif policy not in {"static-mig", "resident-full-gate"}:
        raise ValueError(f"unknown policy: {policy}")

    resident_count = admitted - borrower_count
    residents = tuple(
        WorkerAction(index, modality, "resident-1g", resident_quota)
        for index, modality in enumerate(offered[:resident_count])
    )
    borrowers = tuple(
        WorkerAction(index, offered[index], "borrower-2g", borrower_quota)
        for index in range(resident_count, admitted)
    )
    return PlacementPlan(residents, borrowers)


def guard_for(
    policy: str,
    plan: PlacementPlan,
    state: FeedbackState,
    profile_guard_ms: dict[str, Any],
    override_ms: float | None = None,
) -> float:
    gated = gated_placements(policy)
    actions = tuple(
        action
        for action in (*plan.residents, *plan.borrowers)
        if action.placement in gated
    )
    if not actions:
        return 0.0
    if override_ms is not None:
        return override_ms
    per_placement: dict[str, float] = {}
    for action in actions:
        per_placement[action.placement] = (
            per_placement.get(action.placement, 0.0)
            + guard_value_ms(profile_guard_ms, action)
        )
    # MIG instances execute concurrently, while MPS clients inside one instance
    # can serialize. Bound each instance by the sum of its one-in-flight work.
    profiled = max(per_placement.values())
    return profiled


def gated_placements(policy: str) -> frozenset[str]:
    if policy == "fixed-borrow":
        return frozenset({"borrower-2g"})
    if policy == "resident-full-gate":
        return frozenset({"resident-1g"})
    if policy in {"fixed-full-gate", "mig-governor"}:
        return frozenset({"resident-1g", "borrower-2g"})
    return frozenset()


def update_feedback(
    state: FeedbackState,
    *,
    violated: bool,
    critical_p99_ms: float,
    deadline_ms: float,
    drain_near_overrun: bool = False,
    residual_pressure: bool = False,
    maximum_tenants: int = 6,
) -> str:
    if drain_near_overrun:
        state.borrower_limit = max(0, state.borrower_limit - 1)
        state.resident_admission_limit = max(
            1, state.resident_admission_limit - 1
        )
        state.safe_epochs = 0
        return "drain-reclaim"

    if violated or residual_pressure:
        state.borrower_limit = max(0, state.borrower_limit - 1)
        if state.resident_quota_index > 0:
            state.resident_quota_index -= 1
        else:
            state.resident_admission_limit = max(
                1, state.resident_admission_limit - 1
            )
        state.safe_epochs = 0
        return "residual-reclaim"

    if critical_p99_ms >= deadline_ms * 0.90:
        state.safe_epochs = 0
        return "hold-near-deadline"
    state.safe_epochs += 1
    if state.safe_epochs < 3:
        return "hold-hysteresis"
    # Restore quota before admission so q25/q50 never exceed the calibrated
    # three-resident envelope. At q100, the six-resident held-out case applies.
    if state.resident_quota_index < len(RESIDENT_QUOTAS) - 1:
        state.resident_quota_index += 1
        action = "recover-resident-quota"
    elif state.resident_admission_limit < maximum_tenants:
        state.resident_admission_limit += 1
        action = "recover-admission"
    elif state.borrower_limit < maximum_tenants:
        state.borrower_limit += 1
        action = "recover-borrower"
    else:
        action = "hold-full-capacity"
    state.safe_epochs = 0
    return action


def fail_closed_feedback(state: FeedbackState) -> None:
    state.resident_admission_limit = 1
    state.resident_quota_index = 0
    state.borrower_limit = 0
    state.safe_epochs = 0


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


def percentile(values: list[float], quantile: float) -> float:
    if not values or quantile < 0.0 or quantile > 1.0:
        raise ValueError("percentile requires samples and a quantile in [0, 1]")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def pooled_trace_p99(paths: list[pathlib.Path]) -> tuple[float, int]:
    values: list[float] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if "release_to_completion_ms" not in (reader.fieldnames or ()):
                raise ValueError(f"trace lacks release latency: {path}")
            for row in reader:
                value = float(row["release_to_completion_ms"])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"trace contains invalid latency: {path}")
                values.append(value)
    return percentile(values, 0.99), len(values)


def collect_json(
    process: subprocess.Popen[str], timeout: float
) -> dict[str, Any]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"benchmark process {process.pid} failed ({process.returncode}): "
            f"{stderr}"
        )
    result = json.loads(stdout)
    if result.get("schema_version") != 1:
        raise RuntimeError("benchmark returned an unsupported schema")
    return result


def start_telemetry_session(
    source: pathlib.Path,
    destination: pathlib.Path,
    timeout_seconds: float = 5.0,
) -> TelemetrySession:
    deadline = time.monotonic() + timeout_seconds
    while not source.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"telemetry log was not created: {source}")
        time.sleep(0.01)
    # The collector runs at roughly 8--10 Hz on Thor.  Flushing every sample
    # can delay visibility of a causal post-cleanup sample by a full collector
    # period, consuming the strict thermal handoff budget.  Keep markers and
    # close durable while batching raw samples in small bounded groups.
    writer = JsonlTelemetryWriter(destination, flush=False, flush_every=16)
    monitor = TegrastatsMonitor(writer)
    tail_process = subprocess.Popen(
        ["tail", "-n", "0", "-F", str(source)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if tail_process.stdout is None:
        tail_process.terminate()
        monitor.close()
        raise RuntimeError("tail did not expose telemetry stdout")
    reader_errors: list[str] = []

    def read_lines() -> None:
        try:
            monitor.record_stream(tail_process.stdout)
        except Exception as error:
            reader_errors.append(f"{type(error).__name__}: {error}")

    reader_thread = threading.Thread(
        target=read_lines,
        name="tegrastats-reader",
        daemon=True,
    )
    reader_thread.start()
    session = TelemetrySession(
        monitor,
        tail_process,
        reader_thread,
        reader_errors,
    )
    while not monitor.samples():
        if reader_errors:
            close_telemetry_session(session)
            raise RuntimeError(f"telemetry reader failed: {reader_errors[0]}")
        if tail_process.poll() is not None:
            stderr = tail_process.stderr.read() if tail_process.stderr else ""
            close_telemetry_session(session)
            raise RuntimeError(f"telemetry tail exited early: {stderr}")
        if time.monotonic() >= deadline:
            close_telemetry_session(session)
            raise TimeoutError("telemetry did not produce a sample")
        time.sleep(0.01)
    monitor.mark("collector_ready", {"source": str(source)})
    return session


def close_telemetry_session(session: TelemetrySession) -> list[str]:
    errors = session.reader_errors
    if session.tail_process.poll() is None:
        session.tail_process.terminate()
    try:
        session.tail_process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        session.tail_process.kill()
        session.tail_process.wait()
        errors.append("telemetry tail required SIGKILL")
    session.reader_thread.join(timeout=2.0)
    if session.reader_thread.is_alive():
        errors.append("telemetry reader thread did not stop")
    if session.tail_process.returncode not in {0, -signal.SIGTERM}:
        stderr = (
            session.tail_process.stderr.read()
            if session.tail_process.stderr is not None
            else ""
        )
        errors.append(
            f"telemetry tail exited {session.tail_process.returncode}: {stderr}"
        )
    session.monitor.close()
    return errors


def mark_event(
    monitor: TegrastatsMonitor | None,
    name: str,
    metadata: dict[str, Any],
) -> int:
    if monitor is not None:
        return monitor.mark(name, metadata).monotonic_ns
    return time.monotonic_ns()


def minimum_telemetry_samples(
    start_ns: int,
    end_ns: int,
    interval_ms: float = TELEMETRY_INTERVAL_MS,
    required_fraction: float = TELEMETRY_REQUIRED_FRACTION,
) -> int:
    duration_ns = end_ns - start_ns
    if duration_ns <= 0 or interval_ms <= 0.0:
        raise ValueError("telemetry interval must be positive")
    expected = max(1, math.floor(duration_ns / (interval_ms * 1_000_000.0)))
    return max(1, math.floor(expected * required_fraction))


def thermal_stability_checkpoint_is_timely(
    actual_elapsed_seconds: float,
    scheduled_elapsed_seconds: float,
) -> bool:
    return (
        math.isfinite(actual_elapsed_seconds)
        and math.isfinite(scheduled_elapsed_seconds)
        and scheduled_elapsed_seconds > 0.0
        and scheduled_elapsed_seconds <= actual_elapsed_seconds
        and actual_elapsed_seconds - scheduled_elapsed_seconds
        <= THERMAL_STABILITY_CHECKPOINT_MAX_LATENESS_SECONDS
    )


def thermal_window_summary(
    monitor: TegrastatsMonitor,
    sensor: str,
    window_seconds: float,
    reference_ns: int | None = None,
    not_before_ns: int | None = None,
) -> dict[str, float | int] | None:
    if window_seconds <= 0.0:
        raise ValueError("thermal window must be positive")
    reference = time.monotonic_ns() if reference_ns is None else reference_ns
    start = reference - int(window_seconds * 1_000_000_000)
    if not_before_ns is not None:
        start = max(start, not_before_ns)
    sample_window = getattr(monitor, "sample_window", None)
    samples = (
        sample_window(start, reference, end_inclusive=True).samples
        if callable(sample_window)
        else tuple(
            sample
            for sample in monitor.samples()
            if start <= sample.monotonic_ns <= reference
        )
    )
    points = [
        (sample.monotonic_ns, sample.parsed.temperatures_c[sensor])
        for sample in samples
        if sensor in sample.parsed.temperatures_c
    ]
    if len(points) < 2:
        return None
    times = [(timestamp - points[0][0]) / 1_000_000_000.0 for timestamp, _ in points]
    values = [value for _, value in points]
    mean_time = statistics.fmean(times)
    mean_value = statistics.fmean(values)
    denominator = sum((value - mean_time) ** 2 for value in times)
    slope = (
        sum(
            (sample_time - mean_time) * (value - mean_value)
            for sample_time, value in zip(times, values, strict=True)
        )
        / denominator
        * 60.0
        if denominator > 0.0
        else 0.0
    )
    return {
        "samples": len(values),
        "window_seconds": window_seconds,
        "observed_span_seconds": times[-1] - times[0],
        "mean_c": mean_value,
        "min_c": min(values),
        "max_c": max(values),
        "latest_c": values[-1],
        "slope_c_per_minute": slope,
    }


def thermal_window_maximum_gap_seconds(
    monitor: TegrastatsMonitor,
    sensor: str,
    window_seconds: float,
    reference_ns: int,
    not_before_ns: int | None = None,
) -> float | None:
    if window_seconds <= 0.0:
        raise ValueError("thermal window must be positive")
    start_ns = reference_ns - int(window_seconds * 1_000_000_000)
    if not_before_ns is not None:
        start_ns = max(start_ns, not_before_ns)
    sample_window = getattr(monitor, "sample_window", None)
    samples = (
        sample_window(start_ns, reference_ns, end_inclusive=True).samples
        if callable(sample_window)
        else tuple(
            sample
            for sample in monitor.samples()
            if start_ns <= sample.monotonic_ns <= reference_ns
        )
    )
    timestamps = [
        sample.monotonic_ns
        for sample in samples
        if sensor in sample.parsed.temperatures_c
    ]
    if not timestamps:
        return None
    gaps_ns = [timestamps[0] - start_ns, reference_ns - timestamps[-1]]
    gaps_ns.extend(
        current - previous
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
    )
    return max(gaps_ns) / 1_000_000_000.0


def thermal_pilot_checkpoint_is_stable(
    summary: dict[str, float | int] | None,
    telemetry: dict[str, Any],
    *,
    hard_limit_c: float,
    window_seconds: float,
    maximum_slope_c_per_minute: float,
    safety_sensor: str = THERMAL_SAFETY_SENSOR,
) -> bool:
    if summary is None:
        return False
    health = telemetry.get("health")
    safety = telemetry.get("temperatures_c", {}).get(safety_sensor)
    maximum_gap_seconds = summary.get("maximum_gap_seconds")
    if not isinstance(health, dict) or not isinstance(safety, dict):
        return False
    required_samples = minimum_telemetry_samples(
        0, int(window_seconds * 1_000_000_000)
    )
    return (
        health.get("healthy") is True
        and health.get("required_fields") == list(TELEMETRY_REQUIRED_FIELDS)
        and int(telemetry.get("valid_samples", 0)) >= required_samples
        and int(summary["samples"]) >= required_samples
        and float(summary["observed_span_seconds"]) >= window_seconds * 0.99
        and maximum_gap_seconds is not None
        and math.isfinite(float(maximum_gap_seconds))
        and float(maximum_gap_seconds)
        <= TELEMETRY_STALE_AFTER_MS / 1000.0
        and abs(float(summary["slope_c_per_minute"]))
        <= maximum_slope_c_per_minute
        and float(safety["max"]) < hard_limit_c
    )


def thermal_window_is_stable(
    summary: dict[str, float | int] | None,
    *,
    target_c: float,
    tolerance_c: float,
    window_seconds: float,
    maximum_slope_c_per_minute: float,
) -> bool:
    if summary is None:
        return False
    maximum_gap_seconds = summary.get("maximum_gap_seconds")
    required_samples = minimum_telemetry_samples(
        0, int(window_seconds * 1_000_000_000)
    )
    return (
        int(summary["samples"]) >= required_samples
        and float(summary["observed_span_seconds"]) >= window_seconds * 0.99
        and abs(float(summary["mean_c"]) - target_c) <= tolerance_c
        and abs(float(summary["latest_c"]) - target_c) <= tolerance_c
        and abs(float(summary["slope_c_per_minute"]))
        <= maximum_slope_c_per_minute
        and maximum_gap_seconds is not None
        and math.isfinite(float(maximum_gap_seconds))
        and float(maximum_gap_seconds)
        <= TELEMETRY_STALE_AFTER_MS / 1000.0
    )


def require_live_thermal_telemetry(
    monitor: TegrastatsMonitor,
    start_ns: int,
    reference_ns: int,
    hard_limit_c: float,
    label: str,
    safety_sensor: str = THERMAL_SAFETY_SENSOR,
) -> dict[str, Any]:
    live_start_ns = max(start_ns, reference_ns - 1_000_000_000)
    telemetry = monitor.aggregate(
        live_start_ns,
        reference_ns,
        required_fields=TELEMETRY_REQUIRED_FIELDS,
        minimum_valid_samples=1,
        reference_ns=reference_ns,
        end_inclusive=True,
        stale_after_ns=int(TELEMETRY_STALE_AFTER_MS * 1_000_000.0),
        maximum_valid_gap_ns=int(
            TELEMETRY_STALE_AFTER_MS * 1_000_000.0
        ),
    )
    if not telemetry["health"]["healthy"]:
        raise RuntimeError(
            f"thermal telemetry became unhealthy during {label}: "
            f"{telemetry['health']}"
        )
    live_safety = telemetry["temperatures_c"][safety_sensor]
    if float(live_safety["max"]) >= hard_limit_c:
        raise RuntimeError(
            f"thermal hard limit reached during {label}: {live_safety}"
        )
    return telemetry


def validate_thermal_handoff(
    precondition: dict[str, Any],
    measurement_release_ns: int,
    measurement_start_ns: int,
    maximum_ms: float = THERMAL_HANDOFF_MAX_MS,
) -> dict[str, Any]:
    if not math.isfinite(maximum_ms) or maximum_ms <= 0.0:
        raise ValueError("thermal handoff bound must be positive and finite")
    precondition_end_ns = int(precondition["measurement_end_monotonic_ns"])
    if not (
        precondition_end_ns
        <= measurement_release_ns
        <= measurement_start_ns
    ):
        raise RuntimeError("thermal handoff clocks are inconsistent")
    end_to_release_ms = (
        measurement_release_ns - precondition_end_ns
    ) / 1_000_000.0
    end_to_measurement_start_ms = (
        measurement_start_ns - precondition_end_ns
    ) / 1_000_000.0
    if (
        end_to_release_ms >= maximum_ms
        or end_to_measurement_start_ms >= maximum_ms
    ):
        raise RuntimeError(
            "thermal handoff exceeded the strict bound: "
            f"release={end_to_release_ms:.3f} ms, "
            f"measurement_start={end_to_measurement_start_ms:.3f} ms, "
            f"bound={maximum_ms:.3f} ms"
        )
    return {
        "precondition_measurement_end_monotonic_ns": precondition_end_ns,
        "measurement_release_monotonic_ns": measurement_release_ns,
        "measurement_start_monotonic_ns": measurement_start_ns,
        "end_to_release_ms": end_to_release_ms,
        "end_to_measurement_start_ms": end_to_measurement_start_ms,
        "maximum_ms": maximum_ms,
        "strictly_within_bound": True,
    }


def replay_thermal_start(
    args: argparse.Namespace,
    monitor: TegrastatsMonitor,
    *,
    label: str,
    reference_ns: int,
    not_before_ns: int,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    stability_sensor = args.thermal_stability_sensor
    safety_sensor = args.thermal_safety_sensor
    summary = thermal_window_summary(
        monitor,
        stability_sensor,
        args.thermal_window_seconds,
        reference_ns,
        not_before_ns,
    )
    if summary is None:
        raise RuntimeError(f"{label} thermal start has no stability window")
    summary = dict(summary)
    maximum_gap_seconds = thermal_window_maximum_gap_seconds(
        monitor,
        stability_sensor,
        args.thermal_window_seconds,
        reference_ns,
        not_before_ns,
    )
    summary["maximum_gap_seconds"] = maximum_gap_seconds
    window_start_ns = max(
        not_before_ns,
        reference_ns - int(args.thermal_window_seconds * 1_000_000_000),
    )
    telemetry = monitor.aggregate(
        window_start_ns,
        reference_ns,
        required_fields=TELEMETRY_REQUIRED_FIELDS,
        minimum_valid_samples=minimum_telemetry_samples(
            window_start_ns, reference_ns
        ),
        reference_ns=reference_ns,
        end_inclusive=True,
        stale_after_ns=int(TELEMETRY_STALE_AFTER_MS * 1_000_000.0),
        maximum_valid_gap_ns=int(
            TELEMETRY_STALE_AFTER_MS * 1_000_000.0
        ),
    )
    if not telemetry["health"]["healthy"]:
        raise RuntimeError(
            f"{label} thermal start telemetry is unhealthy: "
            f"{telemetry['health']}"
        )
    safety = telemetry["temperatures_c"][safety_sensor]
    if float(safety["max"]) >= args.thermal_hard_limit_c:
        raise RuntimeError(
            f"thermal hard limit reached during {label}: {safety}"
        )
    if (
        maximum_gap_seconds is None
        or maximum_gap_seconds > TELEMETRY_STALE_AFTER_MS / 1000.0
        or not thermal_window_is_stable(
            summary,
            target_c=args.thermal_target_c,
            tolerance_c=args.thermal_tolerance_c,
            window_seconds=args.thermal_window_seconds,
            maximum_slope_c_per_minute=(
                args.thermal_max_slope_c_per_minute
            ),
        )
    ):
        raise RuntimeError(f"{label} thermal start is unstable: {summary}")
    return summary, telemetry


def _thermal_start_evidence(
    args: argparse.Namespace,
    monitor: TegrastatsMonitor,
    *,
    label: str,
    reference_ns: int,
    window_not_before_ns: int,
) -> tuple[
    dict[str, float | int] | None,
    dict[str, Any] | None,
    str | None,
]:
    """Evaluate one exact thermal-start timestamp without throwing away evidence."""

    stability_sensor = args.thermal_stability_sensor
    safety_sensor = args.thermal_safety_sensor
    summary = thermal_window_summary(
        monitor,
        stability_sensor,
        args.thermal_window_seconds,
        reference_ns,
        window_not_before_ns,
    )
    if summary is not None:
        summary = dict(summary)
        summary["maximum_gap_seconds"] = thermal_window_maximum_gap_seconds(
            monitor,
            stability_sensor,
            args.thermal_window_seconds,
            reference_ns,
            window_not_before_ns,
        )
    window_start_ns = max(
        window_not_before_ns,
        reference_ns - int(args.thermal_window_seconds * 1_000_000_000),
    )
    if reference_ns <= window_start_ns:
        return summary, None, f"{label} thermal start has no stability window"
    telemetry = monitor.aggregate(
        window_start_ns,
        reference_ns,
        required_fields=TELEMETRY_REQUIRED_FIELDS,
        minimum_valid_samples=minimum_telemetry_samples(
            window_start_ns, reference_ns
        ),
        reference_ns=reference_ns,
        end_inclusive=True,
        stale_after_ns=int(TELEMETRY_STALE_AFTER_MS * 1_000_000.0),
        maximum_valid_gap_ns=int(
            TELEMETRY_STALE_AFTER_MS * 1_000_000.0
        ),
    )
    if summary is None:
        return summary, telemetry, f"{label} thermal start has no stability window"
    if not telemetry["health"]["healthy"]:
        return (
            summary,
            telemetry,
            f"{label} thermal start telemetry is unhealthy: "
            f"{telemetry['health']}",
        )
    safety = telemetry["temperatures_c"].get(safety_sensor)
    if not isinstance(safety, dict):
        return (
            summary,
            telemetry,
            f"{label} thermal start telemetry lacks {safety_sensor}",
        )
    if float(safety["max"]) >= args.thermal_hard_limit_c:
        return (
            summary,
            telemetry,
            f"thermal hard limit reached during {label}: {safety}",
        )
    if not thermal_window_is_stable(
        summary,
        target_c=args.thermal_target_c,
        tolerance_c=args.thermal_tolerance_c,
        window_seconds=args.thermal_window_seconds,
        maximum_slope_c_per_minute=args.thermal_max_slope_c_per_minute,
    ):
        return summary, telemetry, f"{label} thermal start is unstable: {summary}"
    return summary, telemetry, None


def qualify_thermal_start(
    args: argparse.Namespace,
    monitor: TegrastatsMonitor,
    *,
    label: str,
    attempt: int,
    boundary_ns: int,
    cleanup_end_ns: int,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Check the first causal post-cleanup sample while workloads stay paused."""

    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 1 <= attempt <= THERMAL_QUALIFICATION_MAX_ATTEMPTS
    ):
        raise ValueError("thermal qualification attempt is out of range")
    if boundary_ns < 0 or cleanup_end_ns < 0:
        raise ValueError("thermal qualification clocks must be non-negative")
    if cleanup_end_ns <= boundary_ns:
        raise ValueError("thermal cleanup must follow the active boundary")

    maximum_age_ns = int(TELEMETRY_STALE_AFTER_MS * 1_000_000.0)
    deadline_ns = cleanup_end_ns + maximum_age_ns
    sample = None
    while True:
        now_ns = clock_ns()
        query_end_ns = max(now_ns, cleanup_end_ns + 1)
        query = monitor.sample_window(
            cleanup_end_ns + 1,
            query_end_ns,
            end_inclusive=True,
            reverse=False,
            limit=None,
        )
        if not query.interval_complete:
            raise RuntimeError("thermal qualification telemetry was truncated")
        if query.samples:
            sample = query.samples[0]
            break
        if now_ns >= deadline_ns:
            break
        sleep(0.005)

    sample_ns = int(sample.monotonic_ns) if sample is not None else None
    marker = monitor.mark(
        "thermal_start_qualification",
        {
            "label": label,
            "attempt": attempt,
            "boundary_monotonic_ns": boundary_ns,
            "cleanup_end_monotonic_ns": cleanup_end_ns,
            "sample_monotonic_ns": sample_ns,
        },
    )
    qualification_ns = int(marker.monotonic_ns)
    telemetry: dict[str, Any] | None = None
    sample_age_ms: float | None = None
    stability_value: float | None = None
    safety_value: float | None = None
    failure_reason: str | None = None
    if sample is None:
        failure_reason = f"{label} observed no causal post-cleanup telemetry sample"
    else:
        sample_age_ms = (qualification_ns - sample_ns) / 1_000_000.0
        telemetry = monitor.aggregate(
            sample_ns - 1,
            sample_ns,
            required_fields=TELEMETRY_REQUIRED_FIELDS,
            minimum_valid_samples=1,
            require_all_samples_valid=True,
            reference_ns=qualification_ns,
            end_inclusive=True,
            stale_after_ns=maximum_age_ns,
            maximum_valid_gap_ns=maximum_age_ns,
        )
        stability_value = sample.parsed.temperatures_c.get(
            args.thermal_stability_sensor
        )
        safety_value = sample.parsed.temperatures_c.get(args.thermal_safety_sensor)
        sample_gap_ms = (sample_ns - cleanup_end_ns) / 1_000_000.0
        if not telemetry["health"]["healthy"]:
            failure_reason = (
                f"{label} causal telemetry is unhealthy: {telemetry['health']}"
            )
        elif sample_gap_ms > TELEMETRY_STALE_AFTER_MS:
            failure_reason = f"{label} causal telemetry arrived too late"
        elif sample_age_ms > TELEMETRY_STALE_AFTER_MS:
            failure_reason = f"{label} causal telemetry is stale"
        elif (
            stability_value is None
            or not math.isfinite(float(stability_value))
            or abs(float(stability_value) - args.thermal_target_c)
            > args.thermal_tolerance_c
        ):
            failure_reason = f"{label} stability sensor is outside the target band"
        elif (
            safety_value is None
            or not math.isfinite(float(safety_value))
            or float(safety_value) >= args.thermal_hard_limit_c
        ):
            failure_reason = f"{label} thermal hard limit reached"
    passed = failure_reason is None
    result = {
        "attempt": attempt,
        "passed": passed,
        "boundary": THERMAL_HANDOFF_BOUNDARY,
        "boundary_monotonic_ns": boundary_ns,
        "cleanup_end_monotonic_ns": cleanup_end_ns,
        "qualification_monotonic_ns": qualification_ns,
        "sample_monotonic_ns": sample_ns,
        "sample_age_ms": sample_age_ms,
        "stability_sensor": args.thermal_stability_sensor,
        "stability_value_c": stability_value,
        "safety_sensor": args.thermal_safety_sensor,
        "safety_value_c": safety_value,
        "target_c": args.thermal_target_c,
        "tolerance_c": args.thermal_tolerance_c,
        "telemetry": telemetry,
        "failure_reason": None if passed else failure_reason,
    }
    validate_thermal_qualification_evidence(result, expected_attempt=attempt)
    return result


def validate_thermal_qualification_evidence(
    qualification: Mapping[str, Any],
    *,
    expected_attempt: int | None = None,
) -> None:
    """Validate the exact qualification schema for successful and failed attempts."""

    expected_keys = {
        "attempt",
        "passed",
        "boundary",
        "boundary_monotonic_ns",
        "cleanup_end_monotonic_ns",
        "qualification_monotonic_ns",
        "sample_monotonic_ns",
        "sample_age_ms",
        "stability_sensor",
        "stability_value_c",
        "safety_sensor",
        "safety_value_c",
        "target_c",
        "tolerance_c",
        "telemetry",
        "failure_reason",
    }
    if set(qualification) != expected_keys:
        raise RuntimeError("thermal qualification evidence has an invalid schema")
    attempt = qualification.get("attempt")
    passed = qualification.get("passed")
    boundary_ns = qualification.get("boundary_monotonic_ns")
    cleanup_end_ns = qualification.get("cleanup_end_monotonic_ns")
    qualification_ns = qualification.get("qualification_monotonic_ns")
    sample_ns = qualification.get("sample_monotonic_ns")
    sample_age_ms = qualification.get("sample_age_ms")
    failure_reason = qualification.get("failure_reason")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 1 <= attempt <= THERMAL_QUALIFICATION_MAX_ATTEMPTS
        or (expected_attempt is not None and attempt != expected_attempt)
        or not isinstance(passed, bool)
        or qualification.get("boundary") != THERMAL_HANDOFF_BOUNDARY
        or isinstance(boundary_ns, bool)
        or not isinstance(boundary_ns, int)
        or boundary_ns < 0
        or isinstance(cleanup_end_ns, bool)
        or not isinstance(cleanup_end_ns, int)
        or isinstance(qualification_ns, bool)
        or not isinstance(qualification_ns, int)
        or not boundary_ns < cleanup_end_ns < qualification_ns
        or qualification.get("stability_sensor") != THERMAL_STABILITY_SENSOR
        or qualification.get("safety_sensor") != THERMAL_SAFETY_SENSOR
        or not isinstance(qualification.get("target_c"), (int, float))
        or not math.isfinite(float(qualification["target_c"]))
        or not isinstance(qualification.get("tolerance_c"), (int, float))
        or not math.isfinite(float(qualification["tolerance_c"]))
        or float(qualification["tolerance_c"]) < 0.0
    ):
        raise RuntimeError("thermal qualification evidence is malformed")
    if sample_ns is not None and (
        isinstance(sample_ns, bool)
        or not isinstance(sample_ns, int)
        or not cleanup_end_ns < sample_ns <= qualification_ns
    ):
        raise RuntimeError("thermal qualification sample clock is malformed")
    if passed:
        if (
            failure_reason is not None
            or sample_ns is None
            or not isinstance(sample_age_ms, (int, float))
            or not math.isfinite(float(sample_age_ms))
            or float(sample_age_ms) < 0.0
            or float(sample_age_ms) > TELEMETRY_STALE_AFTER_MS
            or not isinstance(qualification.get("stability_value_c"), (int, float))
            or not math.isfinite(float(qualification["stability_value_c"]))
            or not isinstance(qualification.get("safety_value_c"), (int, float))
            or not math.isfinite(float(qualification["safety_value_c"]))
            or not isinstance(qualification.get("telemetry"), dict)
        ):
            raise RuntimeError("thermal qualification success evidence is malformed")
    elif not isinstance(failure_reason, str) or not failure_reason:
        raise RuntimeError("thermal qualification failure lacks a reason")


def require_successful_thermal_qualification(
    qualification: Mapping[str, Any],
) -> int:
    """Return the release boundary only for complete successful evidence."""

    validate_thermal_qualification_evidence(qualification)
    if qualification["passed"] is not True:
        raise RuntimeError("thermal qualification did not pass")
    boundary_ns = qualification["boundary_monotonic_ns"]
    assert isinstance(boundary_ns, int)
    return boundary_ns


def validate_thermal_qualification_handoff(
    qualification: Mapping[str, Any],
    qualification_result_ns: int,
    measurement_release_ns: int,
    measurement_start_ns: int,
    maximum_ms: float = THERMAL_HANDOFF_MAX_MS,
) -> dict[str, Any]:
    """Validate the strict boundary-to-release and boundary-to-start clocks."""

    if not math.isfinite(maximum_ms) or maximum_ms <= 0.0:
        raise ValueError("thermal handoff bound must be positive and finite")
    boundary_ns = require_successful_thermal_qualification(qualification)
    cleanup_end_ns = int(qualification["cleanup_end_monotonic_ns"])
    qualification_ns = int(qualification["qualification_monotonic_ns"])
    if not (
        boundary_ns
        < cleanup_end_ns
        < qualification_ns
        < qualification_result_ns
        < measurement_release_ns
        <= measurement_start_ns
    ):
        raise RuntimeError("thermal qualification handoff clocks are inconsistent")
    elapsed = {
        "boundary_to_cleanup_end_ms": (cleanup_end_ns - boundary_ns) / 1_000_000.0,
        "boundary_to_qualification_ms": (qualification_ns - boundary_ns)
        / 1_000_000.0,
        "boundary_to_qualification_result_ms": (
            qualification_result_ns - boundary_ns
        )
        / 1_000_000.0,
        "boundary_to_measurement_release_ms": (
            measurement_release_ns - boundary_ns
        )
        / 1_000_000.0,
        "boundary_to_measurement_start_ms": (
            measurement_start_ns - boundary_ns
        )
        / 1_000_000.0,
    }
    if any(value >= maximum_ms for value in elapsed.values()):
        raise RuntimeError(
            "thermal qualification handoff exceeded the strict bound: "
            f"{elapsed}, bound={maximum_ms:.3f} ms"
        )
    return {
        "boundary": THERMAL_HANDOFF_BOUNDARY,
        "boundary_monotonic_ns": boundary_ns,
        "cleanup_end_monotonic_ns": cleanup_end_ns,
        "qualification_monotonic_ns": qualification_ns,
        "qualification_result_monotonic_ns": qualification_result_ns,
        "measurement_release_monotonic_ns": measurement_release_ns,
        "measurement_start_monotonic_ns": measurement_start_ns,
        **elapsed,
        "maximum_ms": maximum_ms,
        "strictly_within_bound": True,
    }


def validate_actual_thermal_start(
    args: argparse.Namespace,
    monitor: TegrastatsMonitor,
    *,
    label: str,
    measurement_start_ns: int,
    window_not_before_ns: int,
) -> dict[str, Any]:
    """Validate the newest raw sample causally visible at CUDA measurement start."""

    if measurement_start_ns <= window_not_before_ns:
        raise ValueError("actual-start clocks are inconsistent")
    query = monitor.sample_window(
        window_not_before_ns + 1,
        measurement_start_ns,
        end_inclusive=True,
        reverse=True,
        limit=1,
    )
    if not query.interval_complete:
        raise RuntimeError("actual-start telemetry was truncated")
    sample = query.samples[0] if query.samples else None
    sample_ns = int(sample.monotonic_ns) if sample is not None else None
    telemetry: dict[str, Any] | None = None
    sample_age_ms: float | None = None
    stability_value: float | None = None
    safety_value: float | None = None
    failure_reason: str | None = None
    if sample is None:
        failure_reason = f"{label} has no causal actual-start telemetry sample"
    else:
        sample_age_ms = (measurement_start_ns - sample_ns) / 1_000_000.0
        telemetry = monitor.aggregate(
            sample_ns - 1,
            sample_ns,
            required_fields=TELEMETRY_REQUIRED_FIELDS,
            minimum_valid_samples=1,
            require_all_samples_valid=True,
            reference_ns=measurement_start_ns,
            end_inclusive=True,
            stale_after_ns=int(TELEMETRY_STALE_AFTER_MS * 1_000_000.0),
            maximum_valid_gap_ns=int(TELEMETRY_STALE_AFTER_MS * 1_000_000.0),
        )
        stability_value = sample.parsed.temperatures_c.get(
            args.thermal_stability_sensor
        )
        safety_value = sample.parsed.temperatures_c.get(args.thermal_safety_sensor)
        if not telemetry["health"]["healthy"]:
            failure_reason = (
                f"{label} actual-start telemetry is unhealthy: "
                f"{telemetry['health']}"
            )
        elif sample_age_ms > TELEMETRY_STALE_AFTER_MS:
            failure_reason = f"{label} actual-start telemetry is stale"
        elif (
            stability_value is None
            or not math.isfinite(float(stability_value))
            or abs(float(stability_value) - args.thermal_target_c)
            > args.thermal_tolerance_c
        ):
            failure_reason = f"{label} actual-start stability is outside the target band"
        elif (
            safety_value is None
            or not math.isfinite(float(safety_value))
            or float(safety_value) >= args.thermal_hard_limit_c
        ):
            failure_reason = f"{label} actual-start thermal hard limit reached"
    passed = failure_reason is None
    result = {
        "passed": passed,
        "measurement_start_monotonic_ns": measurement_start_ns,
        "sample_monotonic_ns": sample_ns,
        "sample_age_ms": sample_age_ms,
        "stability_sensor": args.thermal_stability_sensor,
        "stability_value_c": stability_value,
        "safety_sensor": args.thermal_safety_sensor,
        "safety_value_c": safety_value,
        "target_c": args.thermal_target_c,
        "tolerance_c": args.thermal_tolerance_c,
        "telemetry": telemetry,
        "failure_reason": None if passed else failure_reason,
    }
    return result


def platform_thermal_hard_limit_c() -> float:
    passive: list[float] = []
    critical: list[float] = []
    for type_path in pathlib.Path("/sys/class/thermal").glob(
        "thermal_zone*/trip_point_*_type"
    ):
        kind = type_path.read_text(encoding="utf-8").strip().casefold()
        temperature_path = type_path.with_name(
            type_path.name.replace("_type", "_temp")
        )
        if not temperature_path.is_file():
            continue
        temperature_c = float(
            temperature_path.read_text(encoding="utf-8").strip()
        ) / 1000.0
        if kind == "passive":
            passive.append(temperature_c)
        elif kind == "critical":
            critical.append(temperature_c)
    if passive:
        return min(passive) - 5.0
    if critical:
        return min(critical) - 10.0
    raise RuntimeError("platform exposes no thermal safety trip")


def run_thermal_load(
    args: argparse.Namespace,
    base_env: dict[str, str],
    mig: dict[str, str],
    monitor: TegrastatsMonitor,
    *,
    label: str,
    duration_seconds: float | None = None,
    target_c: float | None = None,
) -> dict[str, Any]:
    if (duration_seconds is None) == (target_c is None):
        raise ValueError("thermal load needs exactly one stop condition")
    stability_sensor = args.thermal_stability_sensor
    safety_sensor = args.thermal_safety_sensor
    plan = plan_for(
        "fixed-full-gate",
        ("audio", "audio", "audio", "audio", "audio", "audio"),
        FeedbackState(),
        args.borrower_quota,
    )
    metadata = {"label": label}
    mark_event(monitor, "thermal_prepare", metadata)
    running: list[RunningWorker] = []
    worker_results: list[dict[str, Any]] = []
    started_ns: int | None = None
    measurement_ended_ns: int | None = None
    last_window: dict[str, float | int] | None = None
    stability_checks: list[dict[str, Any]] = []
    active_stability_checks: list[dict[str, Any]] = []
    checkpoint_index = 0
    consecutive_passes = 0
    active_consecutive_passes = 0
    last_active_sample_ns: int | None = None
    measurement_end_metadata = metadata
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        running = start_workers(plan, args, base_env, mig)
        wait_until_paused(
            [worker.process for worker in running],
            args.readiness_timeout_seconds,
        )
        started_ns = mark_event(monitor, "thermal_start", metadata)
        resume_processes([worker.process for worker in running])
        timeout = time.monotonic() + args.thermal_timeout_seconds
        while True:
            now_ns = time.monotonic_ns()
            elapsed = (now_ns - started_ns) / 1_000_000_000.0
            if elapsed >= 0.3:
                require_live_thermal_telemetry(
                    monitor,
                    started_ns,
                    now_ns,
                    args.thermal_hard_limit_c,
                    label,
                    safety_sensor,
                )
            last_window = thermal_window_summary(
                monitor,
                stability_sensor,
                args.thermal_window_seconds,
                now_ns,
                started_ns,
            )
            if last_window is not None:
                last_window = dict(last_window)
                last_window["maximum_gap_seconds"] = (
                    thermal_window_maximum_gap_seconds(
                        monitor,
                        stability_sensor,
                        args.thermal_window_seconds,
                        now_ns,
                        started_ns,
                    )
                )
            if duration_seconds is not None:
                scheduled_elapsed = (
                    checkpoint_index + 1
                ) * THERMAL_STABILITY_CHECKPOINT_SECONDS
                if elapsed >= scheduled_elapsed:
                    checkpoint_boundary_metadata = {
                        "label": label,
                        "checkpoint_index": checkpoint_index,
                        "scheduled_elapsed_seconds": scheduled_elapsed,
                    }
                    checkpoint_ns = mark_event(
                        monitor,
                        "thermal_stability_boundary",
                        checkpoint_boundary_metadata,
                    )
                    checkpoint_elapsed = (
                        checkpoint_ns - started_ns
                    ) / 1_000_000_000.0
                    if not thermal_stability_checkpoint_is_timely(
                        checkpoint_elapsed, scheduled_elapsed
                    ):
                        raise RuntimeError(
                            f"thermal stability checkpoint {checkpoint_index} "
                            f"was late by "
                            f"{checkpoint_elapsed - scheduled_elapsed:.3f} s"
                        )
                    require_live_thermal_telemetry(
                        monitor,
                        started_ns,
                        checkpoint_ns,
                        args.thermal_hard_limit_c,
                        label,
                        safety_sensor,
                    )
                    last_window = thermal_window_summary(
                        monitor,
                        stability_sensor,
                        args.thermal_window_seconds,
                        checkpoint_ns,
                        started_ns,
                    )
                    if last_window is not None:
                        last_window = dict(last_window)
                        last_window["maximum_gap_seconds"] = (
                            thermal_window_maximum_gap_seconds(
                                monitor,
                                stability_sensor,
                                args.thermal_window_seconds,
                                checkpoint_ns,
                                started_ns,
                            )
                        )
                    checkpoint_start_ns = max(
                        started_ns,
                        checkpoint_ns
                        - int(args.thermal_window_seconds * 1_000_000_000),
                    )
                    checkpoint_telemetry = monitor.aggregate(
                        checkpoint_start_ns,
                        checkpoint_ns,
                        required_fields=TELEMETRY_REQUIRED_FIELDS,
                        minimum_valid_samples=minimum_telemetry_samples(
                            0,
                            int(
                                args.thermal_window_seconds
                                * 1_000_000_000
                            ),
                        ),
                        require_all_samples_valid=True,
                        reference_ns=checkpoint_ns,
                        stale_after_ns=int(
                            TELEMETRY_STALE_AFTER_MS * 1_000_000.0
                        ),
                        maximum_valid_gap_ns=int(
                            TELEMETRY_STALE_AFTER_MS * 1_000_000.0
                        ),
                    )
                    passed = thermal_pilot_checkpoint_is_stable(
                        last_window,
                        checkpoint_telemetry,
                        hard_limit_c=args.thermal_hard_limit_c,
                        window_seconds=args.thermal_window_seconds,
                        maximum_slope_c_per_minute=(
                            args.thermal_max_slope_c_per_minute
                        ),
                        safety_sensor=safety_sensor,
                    )
                    consecutive_passes = (
                        consecutive_passes + 1 if passed else 0
                    )
                    checkpoint_metadata = {
                        "label": label,
                        "checkpoint_index": checkpoint_index,
                        "scheduled_elapsed_seconds": scheduled_elapsed,
                        "actual_elapsed_seconds": checkpoint_elapsed,
                        "checkpoint_monotonic_ns": checkpoint_ns,
                        "passed": passed,
                        "consecutive_passes": consecutive_passes,
                        "window": last_window,
                    }
                    mark_event(
                        monitor,
                        "thermal_stability_check",
                        checkpoint_metadata,
                    )
                    stability_checks.append(checkpoint_metadata)
                    checkpoint_index += 1
                    if (
                        checkpoint_elapsed >= duration_seconds
                        and consecutive_passes
                        >= THERMAL_REQUIRED_STABLE_CHECKPOINTS
                    ):
                        measurement_end_metadata = {
                            key: checkpoint_metadata[key]
                            for key in (
                                "label",
                                "checkpoint_index",
                                "scheduled_elapsed_seconds",
                                "actual_elapsed_seconds",
                                "checkpoint_monotonic_ns",
                                "consecutive_passes",
                                "window",
                            )
                        }
                        break
            if target_c is not None and elapsed >= args.thermal_window_seconds:
                latest = monitor.sample_window(
                    started_ns,
                    now_ns,
                    end_inclusive=True,
                    reverse=True,
                    limit=1,
                )
                if not latest.interval_complete:
                    raise RuntimeError("active thermal telemetry was truncated")
                if latest.samples:
                    sample_ns = int(latest.samples[0].monotonic_ns)
                    spacing_ns = int(
                        THERMAL_ACTIVE_STABLE_SPACING_SECONDS * 1_000_000_000
                    )
                    if (
                        last_active_sample_ns is None
                        or sample_ns - last_active_sample_ns >= spacing_ns
                    ):
                        active_window, _active_telemetry, active_failure = (
                            _thermal_start_evidence(
                                args,
                                monitor,
                                label=label,
                                reference_ns=sample_ns,
                                window_not_before_ns=started_ns,
                            )
                        )
                        active_passed = active_failure is None
                        active_consecutive_passes = (
                            active_consecutive_passes + 1
                            if active_passed
                            else 0
                        )
                        active_metadata = {
                            "label": label,
                            "index": len(active_stability_checks),
                            "sample_monotonic_ns": sample_ns,
                            "passed": active_passed,
                            "consecutive_passes": active_consecutive_passes,
                            "window": active_window,
                        }
                        mark_event(
                            monitor,
                            "thermal_active_stability_check",
                            active_metadata,
                        )
                        active_stability_checks.append(active_metadata)
                        last_active_sample_ns = sample_ns
                        if (
                            active_consecutive_passes
                            >= THERMAL_ACTIVE_STABLE_ENDPOINTS
                        ):
                            last_window = active_window
                            measurement_end_metadata = {
                                "label": label,
                                "boundary_sample_monotonic_ns": sample_ns,
                                "consecutive_passes": active_consecutive_passes,
                                "window": active_window,
                            }
                            break
            if time.monotonic() >= timeout:
                raise TimeoutError(
                    f"thermal load {label} did not converge: {last_window}"
                )
            time.sleep(0.1)
        measurement_ended_ns = mark_event(
            monitor, "thermal_measurement_end", measurement_end_metadata
        )
    except BaseException as error:
        primary_error = error
    finally:
        try:
            worker_results = stop_workers(running, tolerate_failure=False)
        except BaseException as error:
            cleanup_error = error
    ended_ns = time.monotonic_ns()
    try:
        ended_ns = mark_event(
            monitor,
            "thermal_end",
            metadata
            | {"successful": primary_error is None and cleanup_error is None},
        )
    except BaseException as error:
        if primary_error is None and cleanup_error is None:
            cleanup_error = error
    if primary_error is not None:
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error
    if started_ns is None or measurement_ended_ns is None:
        raise RuntimeError("thermal load did not complete its measurement window")
    telemetry = monitor.aggregate(
        started_ns,
        measurement_ended_ns,
        required_fields=TELEMETRY_REQUIRED_FIELDS,
        minimum_valid_samples=minimum_telemetry_samples(
            started_ns, measurement_ended_ns
        ),
        require_all_samples_valid=duration_seconds is not None,
        reference_ns=measurement_ended_ns,
        stale_after_ns=int(TELEMETRY_STALE_AFTER_MS * 1_000_000.0),
        maximum_valid_gap_ns=int(
            TELEMETRY_STALE_AFTER_MS * 1_000_000.0
        ),
    )
    if not telemetry["health"]["healthy"]:
        raise RuntimeError(f"thermal telemetry is unhealthy: {telemetry['health']}")
    safety_summary = telemetry["temperatures_c"][safety_sensor]
    if float(safety_summary["max"]) >= args.thermal_hard_limit_c:
        raise RuntimeError(
            f"thermal hard limit reached during {label}: {safety_summary}"
        )
    result = {
        "label": label,
        "duration_seconds": (
            measurement_ended_ns - started_ns
        ) / 1_000_000_000.0,
        "measurement_start_monotonic_ns": started_ns,
        "measurement_end_monotonic_ns": measurement_ended_ns,
        "cleanup_end_monotonic_ns": ended_ns,
        "target_c": target_c,
        "stability_sensor": stability_sensor,
        "safety_sensor": safety_sensor,
        "last_window": last_window,
        "pressure_rate_per_second": sum(
            worker_rate(result) for result in worker_results
        ),
        "telemetry": telemetry,
    }
    if duration_seconds is not None:
        maximum_gap_seconds = thermal_window_maximum_gap_seconds(
            monitor,
            stability_sensor,
            (measurement_ended_ns - started_ns) / 1_000_000_000.0,
            measurement_ended_ns,
            started_ns,
        )
        if (
            maximum_gap_seconds is None
            or maximum_gap_seconds
            > TELEMETRY_STALE_AFTER_MS / 1000.0
        ):
            raise RuntimeError(
                f"thermal telemetry gap exceeded the limit: "
                f"{maximum_gap_seconds}"
            )
        result.update(
            {
                "stability_checkpoint_seconds": (
                    THERMAL_STABILITY_CHECKPOINT_SECONDS
                ),
                "required_consecutive_stable_checkpoints": (
                    THERMAL_REQUIRED_STABLE_CHECKPOINTS
                ),
                "stability_checks": stability_checks,
                "maximum_gap_seconds": maximum_gap_seconds,
                "termination_reason": "stable-checkpoints",
            }
        )
    else:
        result.update(
            {
                "active_stability_checks": active_stability_checks,
                "active_stable_endpoints": THERMAL_ACTIVE_STABLE_ENDPOINTS,
                "active_stable_spacing_seconds": (
                    THERMAL_ACTIVE_STABLE_SPACING_SECONDS
                ),
                "termination_reason": "active-stability-endpoints",
            }
        )
    return result


def critical_command(
    args: argparse.Namespace,
    trace: pathlib.Path,
    deadline_ms: float | None,
    gate_pids: list[int] | None = None,
    stop_pids: list[int] | None = None,
    guard_ms: float = 0.0,
    start_paused: bool = False,
) -> list[str]:
    command = [
        "taskset",
        "--cpu-list",
        args.critical_cpu,
        str(args.bench),
        "--engine",
        str(engine_path(args.engine_root, "mig-2g", "resnet50-v2")),
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
        "high",
        "--trace",
        str(trace),
    ]
    if start_paused:
        command.extend(("--start-paused", "true"))
    if deadline_ms is not None:
        command.extend(("--deadline-ms", str(deadline_ms)))
    if gate_pids:
        command.extend(("--gate-pids", ",".join(str(pid) for pid in gate_pids)))
        command.extend(("--guard-ms", str(guard_ms)))
        command.extend(("--gate-mode", "cooperative"))
    if stop_pids:
        command.extend(("--stop-pids", ",".join(str(pid) for pid in stop_pids)))
    return command


def worker_command(
    args: argparse.Namespace,
    action: WorkerAction,
    cpu: int,
    dependency_wait_fd: int | None = None,
    dependency_signal_fd: int | None = None,
) -> list[str]:
    model = MODEL_BY_MODALITY[action.modality]
    prefix = "mig-1g" if action.placement == "resident-1g" else "mig-2g"
    priority = "default" if action.placement == "resident-1g" else "low"
    command = [
        "taskset",
        "--cpu-list",
        str(cpu),
        str(args.bench),
        "--engine",
        str(
            engine_path(
                args.engine_root,
                f"{prefix}-q{action.quota_percent}",
                model,
            )
        ),
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
        "--start-paused",
        "true",
    ]
    if args.pressure_rps_per_tenant > 0.0:
        command.extend(("--period-ms", str(1000.0 / args.pressure_rps_per_tenant)))
    if dependency_wait_fd is not None:
        command.extend(("--dependency-wait-fd", str(dependency_wait_fd)))
    if dependency_signal_fd is not None:
        command.extend(("--dependency-signal-fd", str(dependency_signal_fd)))
    return command


def worker_env(
    action: WorkerAction,
    base_env: dict[str, str],
    mig: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, str]:
    if action.placement == "resident-1g":
        device = mig["JDG_MIG_SMALL_UUID"]
        pipe = mig["JDG_MPS_PIPE_DIRECTORY"]
        log = mig["JDG_MPS_LOG_DIRECTORY"]
    else:
        device = mig["JDG_MIG_BIG_UUID"]
        pipe = str(args.big_mps_pipe)
        log = str(args.big_mps_log)
    return base_env | {
        "CUDA_VISIBLE_DEVICES": device,
        "CUDA_MPS_PIPE_DIRECTORY": pipe,
        "CUDA_MPS_LOG_DIRECTORY": log,
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(action.quota_percent),
    }


def start_workers(
    plan: PlacementPlan,
    args: argparse.Namespace,
    base_env: dict[str, str],
    mig: dict[str, str],
    dependency_pipes: tuple[DependencyPipe, ...] = (),
) -> list[RunningWorker]:
    cpus = expand_cpu_list(args.pressure_cpus)
    actions = (*plan.residents, *plan.borrowers)
    running: list[RunningWorker] = []
    wait_fd_by_tenant = {
        pipe.downstream_tenant_id: pipe.read_fd for pipe in dependency_pipes
    }
    signal_fd_by_tenant = {
        pipe.upstream_tenant_id: pipe.write_fd for pipe in dependency_pipes
    }
    try:
        for index, action in enumerate(actions):
            cpu = cpus[index % len(cpus)]
            device = (
                mig["JDG_MIG_SMALL_UUID"]
                if action.placement == "resident-1g"
                else mig["JDG_MIG_BIG_UUID"]
            )
            wait_fd = wait_fd_by_tenant.get(action.tenant_id)
            signal_fd = signal_fd_by_tenant.get(action.tenant_id)
            pass_fds = tuple(
                fd for fd in (wait_fd, signal_fd) if fd is not None
            )
            process = subprocess.Popen(
                worker_command(args, action, cpu, wait_fd, signal_fd),
                env=worker_env(action, base_env, mig, args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                pass_fds=pass_fds,
            )
            running.append(
                RunningWorker(
                    action,
                    process,
                    cpu,
                    device,
                    args.warmup,
                    0.0
                    if args.pressure_rps_per_tenant == 0.0
                    else 1000.0 / args.pressure_rps_per_tenant,
                )
            )
        return running
    except Exception:
        stop_workers(running, tolerate_failure=True)
        raise


def dependency_pipes_for_plan(
    plan: PlacementPlan, scenario: str
) -> tuple[DependencyPipe, ...]:
    if scenario != "dependent":
        return ()
    actions = (*plan.residents, *plan.borrowers)
    by_tenant = {action.tenant_id: action for action in actions}
    pipes: list[DependencyPipe] = []
    try:
        for action in actions:
            if action.modality != "language" or action.tenant_id <= 0:
                continue
            upstream = by_tenant.get(action.tenant_id - 1)
            if upstream is None or upstream.modality != "audio":
                continue
            read_fd, write_fd = os.pipe()
            pipes.append(
                DependencyPipe(
                    upstream_tenant_id=upstream.tenant_id,
                    downstream_tenant_id=action.tenant_id,
                    read_fd=read_fd,
                    write_fd=write_fd,
                )
            )
        return tuple(pipes)
    except BaseException:
        for pipe in pipes:
            os.close(pipe.read_fd)
            os.close(pipe.write_fd)
        raise


def close_dependency_pipes(pipes: tuple[DependencyPipe, ...]) -> None:
    for pipe in pipes:
        for fd in (pipe.read_fd, pipe.write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def process_state(pid: int) -> str:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    command_end = text.rfind(")")
    if command_end < 0 or command_end + 2 >= len(text):
        raise RuntimeError(f"malformed process state for PID {pid}")
    return text[command_end + 2]


def wait_until_paused(
    processes: list[subprocess.Popen[str]], timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        failed = [process for process in processes if process.poll() is not None]
        if failed:
            details = []
            for process in failed:
                stdout, stderr = process.communicate()
                details.append(
                    f"PID {process.pid} exited {process.returncode}: "
                    f"{stderr or stdout}"
                )
            raise RuntimeError("readiness barrier failed: " + "; ".join(details))
        if all(process_state(process.pid) in {"T", "t"} for process in processes):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("processes did not reach the readiness barrier")
        time.sleep(0.001)


def require_processes_paused(
    processes: list[subprocess.Popen[str]],
) -> dict[str, str]:
    """Capture fail-closed evidence that every measured process is stopped."""

    states = {str(process.pid): process_state(process.pid) for process in processes}
    if any(state not in {"T", "t"} for state in states.values()):
        raise RuntimeError(
            "measured processes left the stopped barrier before qualification: "
            f"{states}"
        )
    return states


def process_affinity_snapshot(pid: int, expected_cpu: int) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    task_root = pathlib.Path(f"/proc/{pid}/task")
    for task_directory in sorted(task_root.iterdir(), key=lambda path: int(path.name)):
        status = (task_directory / "status").read_text(encoding="utf-8")
        match = re.search(r"^Cpus_allowed_list:\s*(\S+)\s*$", status, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"PID {pid} task {task_directory.name} lacks affinity")
        cpus = expand_cpu_list(match.group(1))
        if cpus != [expected_cpu]:
            raise RuntimeError(
                f"PID {pid} task {task_directory.name} runs on {cpus}, "
                f"expected CPU {expected_cpu}"
            )
        tasks.append({"tid": int(task_directory.name), "cpus": cpus})
    if not tasks:
        raise RuntimeError(f"PID {pid} has no tasks at the readiness barrier")
    return {"pid": pid, "expected_cpu": expected_cpu, "tasks": tasks}


def resume_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        process.send_signal(signal.SIGCONT)


def wait_until_resumed(
    processes: list[subprocess.Popen[str]], timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        failed = [process for process in processes if process.poll() is not None]
        if failed:
            raise RuntimeError(
                "process exited while leaving the readiness barrier: "
                + ",".join(str(process.pid) for process in failed)
            )
        if all(process_state(process.pid) not in {"T", "t"} for process in processes):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("processes did not leave the readiness barrier")
        time.sleep(0.0001)


def terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGCONT)
    process.terminate()
    try:
        process.communicate(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def validate_worker_result(result: dict[str, Any], worker: RunningWorker) -> None:
    if result.get("schema_version") != 1:
        raise ValueError("worker returned an unsupported schema")
    environment = result.get("execution_environment")
    gpu = result.get("gpu")
    config = result.get("config")
    physical_sms = 8 if worker.action.placement == "resident-1g" else 12
    # Thor exposes SMs to MPS clients in two-SM GPC granules and rounds the
    # percentage down. Empirically, q25 on the 12-SM 2g instance reports two
    # SMs rather than the arithmetic three.
    requested_sms = physical_sms * worker.action.quota_percent // 100
    expected_sms = max(2, requested_sms // 2 * 2)
    expected_gpu_name = (
        "NVIDIA Thor MIG 1g.0gb"
        if worker.action.placement == "resident-1g"
        else "NVIDIA Thor MIG 2g.0gb"
    )
    expected_model = MODEL_BY_MODALITY[worker.action.modality]
    if (
        not isinstance(environment, dict)
        or environment.get("pid") != worker.process.pid
        or environment.get("cuda_visible_devices") != worker.device
        or environment.get("mps_active_thread_percentage")
        != worker.action.quota_percent
        or environment.get("cpu_affinity") != [worker.cpu]
    ):
        raise ValueError("worker reported a different execution environment")
    if (
        not isinstance(gpu, dict)
        or gpu.get("name") != expected_gpu_name
        or gpu.get("multiprocessors") != expected_sms
    ):
        raise ValueError(
            "worker reported a different MIG execution width: "
            f"expected {expected_gpu_name}/{expected_sms} SMs, got {gpu!r}"
        )
    expected_priority = (
        "default" if worker.action.placement == "resident-1g" else "low"
    )
    if (
        result.get("role") != "pressure"
        or result.get("model") != expected_model
        or not isinstance(config, dict)
        or config.get("warmup") != worker.warmup
        or config.get("duration_seconds") != 3600
        or config.get("include_transfers") is not True
        or config.get("priority") != expected_priority
        or config.get("start_paused") is not True
        or config.get("burst_size") != 1
        or not math.isclose(
            float(config.get("period_ms", -1.0)),
            worker.period_ms,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or config.get("deadline_ms") != 0
        or config.get("guard_ms") != 0
        or config.get("gated_processes") != 0
        or config.get("stopped_processes") != 0
        or config.get("gate_mode") != "stop"
        or config.get("stream_priority_value") != 0
    ):
        raise ValueError("worker reported a different benchmark configuration")


def stop_workers(
    workers: list[RunningWorker],
    *,
    tolerate_failure: bool = False,
    dependency_pipes: tuple[DependencyPipe, ...] = (),
) -> list[dict[str, Any]]:
    signal_errors: list[str] = []
    for worker in workers:
        if worker.process.poll() is None:
            try:
                worker.process.send_signal(signal.SIGCONT)
                worker.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            except OSError as error:
                signal_errors.append(
                    f"{worker.action.placement}/{worker.action.modality} "
                    f"signal failed: {error}"
                )
    outputs: list[dict[str, Any]] = []
    errors: list[str] = signal_errors
    for worker in workers:
        try:
            stdout, stderr = worker.process.communicate(timeout=30.0)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            stdout, stderr = worker.process.communicate()
            errors.append(
                f"{worker.action.placement}/{worker.action.modality} timed out"
            )
            continue
        if worker.process.returncode != 0:
            errors.append(
                f"{worker.action.placement}/{worker.action.modality} failed "
                f"({worker.process.returncode}): {stderr}"
            )
            continue
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as error:
            errors.append(
                f"{worker.action.placement}/{worker.action.modality} returned "
                f"invalid JSON: {error}"
            )
            continue
        try:
            validate_worker_result(result, worker)
            expected_wait = any(
                pipe.downstream_tenant_id == worker.action.tenant_id
                for pipe in dependency_pipes
            )
            expected_signal = any(
                pipe.upstream_tenant_id == worker.action.tenant_id
                for pipe in dependency_pipes
            )
            worker_config = result.get("config", {})
            has_dependency_evidence = (
                "dependency_wait_enabled" in worker_config
                or "dependency_signal_enabled" in worker_config
            )
            if (dependency_pipes or has_dependency_evidence) and (
                worker_config.get("dependency_wait_enabled") is not expected_wait
                or worker_config.get("dependency_signal_enabled")
                is not expected_signal
            ):
                raise ValueError(
                    "dependency evidence does not match the workload scenario"
                )
        except (TypeError, ValueError) as error:
            errors.append(
                f"{worker.action.placement}/{worker.action.modality} "
                f"returned invalid provenance: {error}"
            )
            continue
        result.update(dataclasses.asdict(worker.action))
        outputs.append(result)
    if errors and not tolerate_failure:
        raise RuntimeError("; ".join(errors))
    return outputs


def validate_critical_result(
    result: dict[str, Any],
    *,
    process_pid: int | None,
    args: argparse.Namespace,
    critical_uuid: str,
    deadline_ms: float,
    gated_processes: int,
    stopped_processes: int,
    guard_ms: float,
    start_paused: bool,
) -> None:
    environment = result.get("execution_environment")
    gpu = result.get("gpu")
    config = result.get("config")
    expected_engine = str(engine_path(args.engine_root, "mig-2g", "resnet50-v2"))
    expected_gate_mode = "cooperative" if gated_processes else "stop"
    if (
        result.get("schema_version") != 1
        or result.get("role") != "benchmark"
        or result.get("model") != "resnet50-v2"
        or result.get("engine") != expected_engine
        or result.get("completed_requests") != args.samples
    ):
        raise RuntimeError("critical benchmark returned different workload provenance")
    if (
        not isinstance(environment, dict)
        or (
            process_pid is not None
            and environment.get("pid") != process_pid
        )
        or environment.get("cuda_visible_devices") != critical_uuid
        or environment.get("mps_active_thread_percentage") != 100
        or environment.get("cpu_affinity") != expand_cpu_list(args.critical_cpu)
    ):
        raise RuntimeError("critical benchmark returned a different execution environment")
    if (
        not isinstance(gpu, dict)
        or gpu.get("name") != "NVIDIA Thor MIG 2g.0gb"
        or gpu.get("multiprocessors") != 12
    ):
        raise RuntimeError("critical benchmark returned a different MIG execution width")
    expected_config: dict[str, Any] = {
        "warmup": args.warmup,
        "burst_size": args.burst_size,
        "gated_processes": gated_processes,
        "stopped_processes": stopped_processes,
        "gate_mode": expected_gate_mode,
        "start_paused": start_paused,
        "include_transfers": True,
        "priority": "high",
        "stream_priority_value": -5,
    }
    if (
        not isinstance(config, dict)
        or any(
        config.get(key) != value for key, value in expected_config.items()
        )
        or any(
            not math.isclose(
                float(config.get(key, math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for key, value in {
                "period_ms": args.period_ms,
                "deadline_ms": deadline_ms,
                "duration_seconds": 0.0,
                "guard_ms": guard_ms,
            }.items()
        )
    ):
        raise RuntimeError("critical benchmark returned a different run configuration")


def completed_by(
    results: list[dict[str, Any]], key: str, value: str
) -> int:
    return sum(
        int(result["completed_requests"])
        for result in results
        if result[key] == value
    )


def worker_rate(result: dict[str, Any]) -> float:
    elapsed = float(result["elapsed_seconds"])
    if elapsed <= 0.0:
        raise ValueError("worker reported a non-positive measurement window")
    return int(result["completed_requests"]) / elapsed


def rate_by(
    results: list[dict[str, Any]], key: str, value: str
) -> float:
    return sum(worker_rate(result) for result in results if result[key] == value)


def time_weighted_epoch_rate(
    epochs: list[dict[str, Any]], key: str
) -> float:
    total_seconds = sum(float(epoch["measurement_seconds"]) for epoch in epochs)
    if total_seconds <= 0.0:
        raise ValueError("epochs reported a non-positive measurement window")
    return sum(
        float(epoch[key]) * float(epoch["measurement_seconds"])
        for epoch in epochs
    ) / total_seconds


def jain_fairness(values: list[float]) -> float | None:
    if not values or sum(values) == 0:
        return None
    return sum(values) ** 2 / (len(values) * sum(value**2 for value in values))


def run_policy(
    policy: str,
    args: argparse.Namespace,
    base_env: dict[str, str],
    mig: dict[str, str],
    critical_env: dict[str, str],
    deadline_ms: float,
    profile_guard_ms: dict[str, Any],
    telemetry_monitor: TegrastatsMonitor | None = None,
) -> dict[str, Any]:
    state = FeedbackState()
    epochs: list[dict[str, Any]] = []
    timeout = args.samples / args.burst_size * args.period_ms / 1000.0 + 60.0
    thermal_precondition = None
    policy_thermal_start_attempts: list[dict[str, Any]] = []
    policy_thermal_start_qualification: dict[str, Any] | None = None
    policy_thermal_actual_start_qualification: dict[str, Any] | None = None
    mark_event(telemetry_monitor, "policy_start", {"policy": policy})
    for epoch_index in range(args.epochs):
        offered = offered_for_epoch(epoch_index, args.scenario)
        marker_metadata = {
            "policy": policy,
            "epoch": epoch_index,
            "scenario": args.scenario,
        }
        mark_event(telemetry_monitor, "epoch_prepare", marker_metadata)
        state_before = dataclasses.asdict(state) | {
            "resident_quota_percent": state.resident_quota_percent
        }
        plan = plan_for(policy, offered, state, args.borrower_quota)
        guard_ms = guard_for(
            policy,
            plan,
            state,
            profile_guard_ms,
            args.guard_override_ms,
        )
        if guard_ms >= args.period_ms:
            raise RuntimeError(
                f"{policy} epoch {epoch_index} guard {guard_ms:.3f} ms "
                f"does not fit period {args.period_ms:.3f} ms"
            )
        trace = args.output.parent / "raw" / f"{policy}-e{epoch_index}.csv"
        wall_start = time.monotonic()
        running: list[RunningWorker] = []
        critical_process: subprocess.Popen[str] | None = None
        worker_results: list[dict[str, Any]] = []
        critical: dict[str, Any] | None = None
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        measurement_start_ns: int | None = None
        measurement_end_ns: int | None = None
        measurement_release_ns: int | None = None
        result_collected_ns: int | None = None
        cleanup_end_ns: int | None = None
        readiness_affinity: list[dict[str, Any]] = []
        thermal_start_summary: dict[str, float | int] | None = None
        thermal_start_telemetry: dict[str, Any] | None = None
        thermal_handoff: dict[str, Any] | None = None
        thermal_start_attempts: list[dict[str, Any]] = []
        thermal_start_qualification: dict[str, Any] | None = None
        thermal_actual_start_qualification: dict[str, Any] | None = None
        qualification_result_marker_ns: int | None = None
        gated = gated_placements(policy)
        gate_pids: list[int] = []
        dependency_pipes = dependency_pipes_for_plan(plan, args.scenario)
        try:
            running = start_workers(
                plan, args, base_env, mig, dependency_pipes
            )
            gate_pids = [
                worker.process.pid
                for worker in running
                if worker.action.placement in gated
            ]
            all_worker_pids = [worker.process.pid for worker in running]
            critical_process = subprocess.Popen(
                critical_command(
                    args,
                    trace,
                    deadline_ms,
                    gate_pids=gate_pids if guard_ms > 0.0 else None,
                    stop_pids=all_worker_pids,
                    guard_ms=guard_ms,
                    start_paused=True,
                ),
                env=critical_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes = [worker.process for worker in running]
            processes.append(critical_process)
            wait_until_paused(processes, args.readiness_timeout_seconds)
            readiness_affinity = [
                {
                    "role": "pressure",
                    "tenant_id": worker.action.tenant_id,
                    **process_affinity_snapshot(worker.process.pid, worker.cpu),
                }
                for worker in running
            ]
            readiness_affinity.append(
                {
                    "role": "critical",
                    **process_affinity_snapshot(
                        critical_process.pid,
                        expand_cpu_list(args.critical_cpu)[0],
                    ),
                }
            )
            if epoch_index == 0 and args.thermal_target_c is not None:
                if telemetry_monitor is None:
                    raise RuntimeError("thermal preconditioning requires telemetry")
                for attempt in range(1, THERMAL_QUALIFICATION_MAX_ATTEMPTS + 1):
                    wait_until_paused(processes, args.readiness_timeout_seconds)
                    require_processes_paused(processes)
                    thermal_label = (
                        f"pre-{policy}-epoch-{epoch_index:02d}"
                        f"-attempt-{attempt:02d}"
                    )
                    candidate = run_thermal_load(
                        args,
                        base_env,
                        mig,
                        telemetry_monitor,
                        label=thermal_label,
                        target_c=args.thermal_target_c,
                    )
                    wait_until_paused(processes, args.readiness_timeout_seconds)
                    measured_process_states = require_processes_paused(processes)
                    qualification = qualify_thermal_start(
                        args,
                        telemetry_monitor,
                        label=thermal_label,
                        attempt=attempt,
                        boundary_ns=int(candidate["measurement_end_monotonic_ns"]),
                        cleanup_end_ns=int(candidate["cleanup_end_monotonic_ns"]),
                    )
                    validate_thermal_qualification_evidence(
                        qualification, expected_attempt=attempt
                    )
                    measured_process_states = require_processes_paused(processes)
                    qualification_result_marker_ns = mark_event(
                        telemetry_monitor,
                        "thermal_start_qualification_result",
                        marker_metadata
                        | {
                            "label": thermal_label,
                            "attempt": attempt,
                            "qualification_monotonic_ns": int(
                                qualification["qualification_monotonic_ns"]
                            ),
                            "passed": qualification["passed"],
                            "failure_reason": qualification["failure_reason"],
                        },
                    )
                    attempt_record = {
                        "attempt": attempt,
                        "thermal_precondition": candidate,
                        "qualification": qualification,
                        "qualification_result_marker_monotonic_ns": (
                            qualification_result_marker_ns
                        ),
                        "measured_process_states": measured_process_states,
                    }
                    thermal_start_attempts.append(attempt_record)
                    if qualification["passed"]:
                        thermal_precondition = candidate
                        thermal_start_qualification = qualification
                        break
                if thermal_start_qualification is None:
                    raise RuntimeError(
                        "thermal start qualification exhausted the fixed three attempts"
                    )
                require_successful_thermal_qualification(
                    thermal_start_qualification
                )
                assert thermal_precondition is not None
                assert qualification_result_marker_ns is not None
                thermal_start_summary = thermal_precondition["last_window"]
                thermal_start_telemetry = thermal_start_qualification["telemetry"]
                policy_thermal_start_attempts = thermal_start_attempts
                policy_thermal_start_qualification = thermal_start_qualification
            measurement_release_ns = mark_event(
                telemetry_monitor, "measurement_start", marker_metadata
            )
            if thermal_start_qualification is not None and epoch_index == 0:
                thermal_handoff = validate_thermal_qualification_handoff(
                    thermal_start_qualification,
                    qualification_result_marker_ns,
                    measurement_release_ns,
                    measurement_release_ns,
                    args.thermal_handoff_max_ms,
                )
            worker_processes = [worker.process for worker in running]
            resume_processes(worker_processes)
            wait_until_resumed(
                worker_processes, args.readiness_timeout_seconds
            )
            resume_processes([critical_process])
            if args.thermal_target_c is not None:
                if telemetry_monitor is None:
                    raise RuntimeError("thermal validation requires telemetry")
                if epoch_index != 0:
                    thermal_start_summary = thermal_window_summary(
                        telemetry_monitor,
                        args.thermal_stability_sensor,
                        args.thermal_window_seconds,
                        measurement_release_ns,
                    )
                    if thermal_start_summary is not None:
                        thermal_start_summary = dict(thermal_start_summary)
                        thermal_start_summary["maximum_gap_seconds"] = (
                            thermal_window_maximum_gap_seconds(
                                telemetry_monitor,
                                args.thermal_stability_sensor,
                                args.thermal_window_seconds,
                                measurement_release_ns,
                            )
                        )
            critical = collect_json(critical_process, timeout)
            validate_critical_result(
                critical,
                process_pid=critical_process.pid,
                args=args,
                critical_uuid=mig["JDG_MIG_BIG_UUID"],
                deadline_ms=deadline_ms,
                gated_processes=len(gate_pids) if guard_ms > 0.0 else 0,
                stopped_processes=len(all_worker_pids),
                guard_ms=guard_ms,
                start_paused=True,
            )
            measurement_start_ns = int(critical["measurement_start_monotonic_ns"])
            measurement_end_ns = int(critical["measurement_end_monotonic_ns"])
            if not (
                measurement_release_ns <= measurement_start_ns < measurement_end_ns
            ):
                raise RuntimeError("benchmark returned inconsistent measurement clocks")
            if thermal_start_qualification is not None and epoch_index == 0:
                assert telemetry_monitor is not None
                assert qualification_result_marker_ns is not None
                thermal_actual_start_qualification = validate_actual_thermal_start(
                    args,
                    telemetry_monitor,
                    label=f"{policy}-epoch-{epoch_index:02d}",
                    measurement_start_ns=measurement_start_ns,
                    window_not_before_ns=int(
                        thermal_start_qualification[
                            "cleanup_end_monotonic_ns"
                        ]
                    ),
                )
                mark_event(
                    telemetry_monitor,
                    "thermal_actual_start_qualification_result",
                    marker_metadata
                    | {
                        "measurement_start_monotonic_ns": measurement_start_ns,
                        "sample_monotonic_ns": thermal_actual_start_qualification[
                            "sample_monotonic_ns"
                        ],
                        "passed": thermal_actual_start_qualification["passed"],
                        "failure_reason": thermal_actual_start_qualification[
                            "failure_reason"
                        ],
                    },
                )
                if thermal_actual_start_qualification["passed"] is not True:
                    raise RuntimeError(
                        "actual measurement start failed thermal qualification"
                    )
                policy_thermal_actual_start_qualification = (
                    thermal_actual_start_qualification
                )
                thermal_handoff = validate_thermal_qualification_handoff(
                    thermal_start_qualification,
                    qualification_result_marker_ns,
                    measurement_release_ns,
                    measurement_start_ns,
                    args.thermal_handoff_max_ms,
                )
            result_collected_ns = mark_event(
                telemetry_monitor,
                "measurement_result_collected",
                marker_metadata
                | {
                    "measurement_start_monotonic_ns": measurement_start_ns,
                    "measurement_end_monotonic_ns": measurement_end_ns,
                },
            )
            if measurement_end_ns > result_collected_ns:
                raise RuntimeError("benchmark measurement end is in the future")
        except BaseException as error:
            primary_error = error
        finally:
            try:
                terminate_process(critical_process)
            except BaseException as error:
                cleanup_error = error
            try:
                worker_results = stop_workers(
                    running, dependency_pipes=dependency_pipes
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            close_dependency_pipes(dependency_pipes)
        try:
            cleanup_end_ns = mark_event(
                telemetry_monitor, "cleanup_end", marker_metadata
            )
        except BaseException as error:
            if primary_error is None and cleanup_error is None:
                cleanup_error = error
        if primary_error is not None:
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error
        if critical is None:
            raise RuntimeError("critical benchmark produced no result")
        if (
            measurement_start_ns is None
            or measurement_end_ns is None
            or measurement_release_ns is None
            or result_collected_ns is None
            or cleanup_end_ns is None
        ):
            raise RuntimeError("measurement markers are incomplete")
        for result in worker_results:
            worker_start_ns = int(result["measurement_start_monotonic_ns"])
            worker_end_ns = int(result["measurement_end_monotonic_ns"])
            if not (
                measurement_release_ns
                <= worker_start_ns
                <= measurement_start_ns
                < measurement_end_ns
                <= worker_end_ns
                <= cleanup_end_ns
            ):
                raise RuntimeError("worker returned an inconsistent measurement window")

        wall_elapsed = time.monotonic() - wall_start
        measurement_seconds = float(critical["elapsed_seconds"])
        if measurement_seconds <= 0.0:
            raise RuntimeError("critical benchmark reported an invalid window")
        latency = critical["release_to_completion"]
        miss_rate = float(critical["deadline_miss_rate"] or 0.0)
        violated = (
            float(latency["p99_ms"]) > deadline_ms
            or miss_rate > args.dmr_target
        )
        if telemetry_monitor is None:
            telemetry_summary: dict[str, Any] = {
                "health": {
                    "healthy": False,
                    "reasons": ["not_configured"],
                }
            }
        else:
            telemetry_summary = telemetry_monitor.aggregate(
                measurement_start_ns,
                measurement_end_ns,
                required_fields=TELEMETRY_REQUIRED_FIELDS,
                minimum_valid_samples=minimum_telemetry_samples(
                    measurement_start_ns, measurement_end_ns
                ),
                reference_ns=measurement_end_ns,
                stale_after_ns=int(
                    TELEMETRY_STALE_AFTER_MS * 1_000_000.0
                ),
                maximum_valid_gap_ns=int(
                    TELEMETRY_STALE_AFTER_MS * 1_000_000.0
                ),
            )
        telemetry_unhealthy = not bool(telemetry_summary["health"]["healthy"])
        if not telemetry_unhealthy:
            safety_summary = telemetry_summary["temperatures_c"][
                args.thermal_safety_sensor
            ]
            if float(safety_summary["max"]) >= args.thermal_hard_limit_c:
                raise RuntimeError(
                    f"{policy} epoch {epoch_index} reached the thermal hard limit"
                )
        drain_max_ms = float(critical["drain"]["max_ms"])
        guard_utilization = drain_max_ms / guard_ms if guard_ms > 0.0 else 0.0
        drain_near_overrun = guard_ms > 0.0 and guard_utilization >= 0.8
        thermal_high = False
        if args.thermal_target_c is not None and not telemetry_unhealthy:
            stability_summary = telemetry_summary["temperatures_c"].get(
                args.thermal_stability_sensor
            )
            thermal_high = (
                stability_summary is not None
                and float(stability_summary["max"])
                > args.thermal_target_c + args.thermal_tolerance_c
            )
        controller_action = "not-applicable"
        if policy == "mig-governor":
            if telemetry_unhealthy:
                fail_closed_feedback(state)
                controller_action = "telemetry-fail-closed"
            else:
                controller_action = update_feedback(
                    state,
                    violated=violated,
                    critical_p99_ms=float(latency["p99_ms"]),
                    deadline_ms=deadline_ms,
                    drain_near_overrun=drain_near_overrun,
                    residual_pressure=thermal_high,
                )
        state_after = dataclasses.asdict(state) | {
            "resident_quota_percent": state.resident_quota_percent
        }
        resident_completed = completed_by(
            worker_results, "placement", "resident-1g"
        )
        borrower_completed = completed_by(
            worker_results, "placement", "borrower-2g"
        )
        by_modality = {
            modality: completed_by(worker_results, "modality", modality)
            for modality in MODEL_BY_MODALITY
        }
        worker_windows = [
            float(result["elapsed_seconds"]) for result in worker_results
        ]
        if not worker_windows or min(worker_windows) <= 0.0:
            raise RuntimeError("workers reported an invalid measurement window")
        worker_window_seconds = statistics.median(worker_windows)
        window_spread = max(worker_windows) - min(worker_windows)
        resident_goodput = rate_by(
            worker_results, "placement", "resident-1g"
        )
        borrower_goodput = rate_by(
            worker_results, "placement", "borrower-2g"
        )
        pressure_goodput = resident_goodput + borrower_goodput
        goodput_by_modality = {
            modality: rate_by(worker_results, "modality", modality)
            for modality in MODEL_BY_MODALITY
        }
        completed_by_tenant = {
            str(tenant_id): sum(
                int(result["completed_requests"])
                for result in worker_results
                if int(result["tenant_id"]) == tenant_id
            )
            for tenant_id in range(len(plan.residents) + len(plan.borrowers))
        }
        goodput_by_tenant = {
            str(tenant_id): sum(
                worker_rate(result)
                for result in worker_results
                if int(result["tenant_id"]) == tenant_id
            )
            for tenant_id in range(len(plan.residents) + len(plan.borrowers))
        }
        critical_busy_ms = (
            float(critical["gpu_service"]["mean_ms"])
            * int(critical["completed_requests"])
        )
        epochs.append(
            {
                "epoch": epoch_index,
                "scenario": args.scenario,
                "offered_modalities": list(offered),
                "offered_tenants": len(offered),
                "resident_actions": [
                    dataclasses.asdict(action) for action in plan.residents
                ],
                "borrower_actions": [
                    dataclasses.asdict(action) for action in plan.borrowers
                ],
                "resident_workers": len(plan.residents),
                "borrower_workers": len(plan.borrowers),
                "active_workers": len(plan.residents) + len(plan.borrowers),
                "rejected_tenants": len(offered)
                - len(plan.residents)
                - len(plan.borrowers),
                "guard_ms": guard_ms,
                "gate_scope": sorted(gated),
                "gated_workers": len(gate_pids),
                "state_before": state_before,
                "state_after": state_after,
                "critical_p50_ms": latency["p50_ms"],
                "critical_p99_ms": latency["p99_ms"],
                "critical_p999_ms": latency["p999_ms"],
                "critical_max_ms": latency["max_ms"],
                "deadline_misses": critical["deadline_misses"],
                "deadline_miss_rate": miss_rate,
                "queue_delay_p99_ms": critical["queue_delay"]["p99_ms"],
                "gate_overhead_mean_ms": critical["gate_overhead"]["mean_ms"],
                "drain_p99_ms": critical["drain"]["p99_ms"],
                "drain_max_ms": drain_max_ms,
                "resume_p99_ms": critical["resume"]["p99_ms"],
                "guard_utilization": guard_utilization,
                "drain_near_overrun": drain_near_overrun,
                "thermal_high": thermal_high,
                "thermal_start": thermal_start_summary,
                "thermal_start_telemetry": thermal_start_telemetry,
                "thermal_start_attempts": thermal_start_attempts,
                "thermal_start_qualification": thermal_start_qualification,
                "thermal_actual_start_qualification": (
                    thermal_actual_start_qualification
                ),
                "thermal_start_stable": (
                    thermal_window_is_stable(
                        thermal_start_summary,
                        target_c=args.thermal_target_c,
                        tolerance_c=args.thermal_tolerance_c,
                        window_seconds=args.thermal_window_seconds,
                        maximum_slope_c_per_minute=(
                            args.thermal_max_slope_c_per_minute
                        ),
                    )
                    if args.thermal_target_c is not None
                    else None
                ),
                "thermal_handoff": thermal_handoff,
                "controller_action": controller_action,
                "critical_gpu_duty_cycle": critical_busy_ms
                / (measurement_seconds * 1000.0),
                "violated": violated,
                "resident_completed": resident_completed,
                "borrower_completed": borrower_completed,
                "pressure_completed": resident_completed + borrower_completed,
                "resident_goodput_per_second": resident_goodput,
                "borrower_goodput_per_second": borrower_goodput,
                "pressure_goodput_per_second": pressure_goodput,
                "completed_by_modality": by_modality,
                "goodput_by_modality": goodput_by_modality,
                "completed_by_tenant": completed_by_tenant,
                "goodput_by_tenant": goodput_by_tenant,
                "tenant_fairness": jain_fairness(
                    list(goodput_by_tenant.values())
                ),
                "critical": critical,
                "workers": worker_results,
                "readiness_affinity": readiness_affinity,
                "telemetry": telemetry_summary,
                "telemetry_unhealthy": telemetry_unhealthy,
                "measurement_seconds": measurement_seconds,
                "measurement_start_monotonic_ns": measurement_start_ns,
                "measurement_end_monotonic_ns": measurement_end_ns,
                "measurement_release_monotonic_ns": measurement_release_ns,
                "result_collected_monotonic_ns": result_collected_ns,
                "cleanup_end_monotonic_ns": cleanup_end_ns,
                "worker_window_seconds": worker_window_seconds,
                "worker_window_spread_seconds": window_spread,
                "wall_elapsed_seconds": wall_elapsed,
                "dependency_edges": [
                    {
                        "upstream_tenant_id": pipe.upstream_tenant_id,
                        "downstream_tenant_id": pipe.downstream_tenant_id,
                        "semantics": "completion-token-before-next-inference",
                    }
                    for pipe in dependency_pipes
                ],
            }
        )

    mark_event(telemetry_monitor, "policy_end", {"policy": policy})
    measurement_seconds = sum(epoch["measurement_seconds"] for epoch in epochs)
    worker_window_seconds = sum(epoch["worker_window_seconds"] for epoch in epochs)
    wall_elapsed = sum(epoch["wall_elapsed_seconds"] for epoch in epochs)
    total_samples = args.epochs * args.samples
    resident_completed = sum(epoch["resident_completed"] for epoch in epochs)
    borrower_completed = sum(epoch["borrower_completed"] for epoch in epochs)
    return {
        "name": policy,
        "scenario": args.scenario,
        "critical_requests": total_samples,
        "deadline_misses": sum(epoch["deadline_misses"] for epoch in epochs),
        "deadline_miss_rate": sum(epoch["deadline_misses"] for epoch in epochs)
        / total_samples,
        "violation_epoch_rate": sum(epoch["violated"] for epoch in epochs)
        / args.epochs,
        "critical_p99_ms_max": max(epoch["critical_p99_ms"] for epoch in epochs),
        "resident_completed": resident_completed,
        "borrower_completed": borrower_completed,
        "pressure_completed": resident_completed + borrower_completed,
        "resident_goodput_per_second": time_weighted_epoch_rate(
            epochs, "resident_goodput_per_second"
        ),
        "borrower_goodput_per_second": time_weighted_epoch_rate(
            epochs, "borrower_goodput_per_second"
        ),
        "pressure_goodput_per_second": time_weighted_epoch_rate(
            epochs, "pressure_goodput_per_second"
        ),
        "goodput_by_modality": {
            modality: sum(
                float(epoch["goodput_by_modality"][modality])
                * float(epoch["measurement_seconds"])
                for epoch in epochs
            )
            / measurement_seconds
            for modality in MODEL_BY_MODALITY
        },
        "rejected_tenants": sum(epoch["rejected_tenants"] for epoch in epochs),
        "telemetry_unhealthy_epochs": sum(
            epoch["telemetry_unhealthy"] for epoch in epochs
        ),
        "thermal_precondition": thermal_precondition,
        "thermal_start_attempts": policy_thermal_start_attempts,
        "thermal_start_qualification": policy_thermal_start_qualification,
        "thermal_actual_start_qualification": (
            policy_thermal_actual_start_qualification
        ),
        "gate_overhead_mean_ms": statistics.fmean(
            epoch["gate_overhead_mean_ms"] for epoch in epochs
        ),
        "critical_gpu_duty_cycle_mean": statistics.fmean(
            epoch["critical_gpu_duty_cycle"] for epoch in epochs
        ),
        "measurement_seconds": measurement_seconds,
        "worker_window_seconds": worker_window_seconds,
        "wall_elapsed_seconds": wall_elapsed,
        "epochs": epochs,
    }


def maximum_profiled_guard_ms(profile: dict[str, Any]) -> float:
    resident_max = {
        quota: max(
            guard_value_ms(
                profile, WorkerAction(0, modality, "resident-1g", quota)
            )
            for modality in MODEL_BY_MODALITY
        )
        for quota in RESIDENT_QUOTAS
    }
    borrower_max = max(
        guard_value_ms(
            profile, WorkerAction(0, modality, "borrower-2g", 100)
        )
        for modality in MODEL_BY_MODALITY
    )
    candidates = [6.0 * resident_max[100]]
    candidates.extend(
        max(3.0 * resident_max[quota], 3.0 * borrower_max)
        for quota in RESIDENT_QUOTAS
    )
    return max(candidates)


def validate_args(
    args: argparse.Namespace, guard_profile_ms: dict[str, Any] | None = None
) -> list[str]:
    floating_values = {
        "period-ms": args.period_ms,
        "slo-factor": args.slo_factor,
        "dmr-target": args.dmr_target,
        "readiness-timeout-seconds": args.readiness_timeout_seconds,
        "language-guard-ms": args.language_guard_ms,
        "audio-guard-ms": args.audio_guard_ms,
        "thermal-tolerance-c": args.thermal_tolerance_c,
        "thermal-window-seconds": args.thermal_window_seconds,
        "thermal-max-slope-c-per-minute": args.thermal_max_slope_c_per_minute,
        "thermal-timeout-seconds": args.thermal_timeout_seconds,
        "thermal-handoff-max-ms": args.thermal_handoff_max_ms,
        "max-isolated-drift-fraction": args.max_isolated_drift_fraction,
    }
    optional_floating_values = {
        "deadline-ms": args.deadline_ms,
        "guard-override-ms": args.guard_override_ms,
        "thermal-pilot-seconds": args.thermal_pilot_seconds,
        "thermal-target-c": args.thermal_target_c,
        "thermal-hard-limit-c": args.thermal_hard_limit_c,
    }
    if any(not math.isfinite(value) for value in floating_values.values()) or any(
        value is not None and not math.isfinite(value)
        for value in optional_floating_values.values()
    ):
        raise SystemExit("numeric experiment arguments must be finite")
    try:
        critical_cpus = expand_cpu_list(args.critical_cpu)
        pressure_cpus = expand_cpu_list(args.pressure_cpus)
        mps_cpus = expand_cpu_list(args.mps_cpu)
        telemetry_cpus = expand_cpu_list(args.telemetry_cpu)
    except ValueError as error:
        raise SystemExit(f"invalid CPU assignment: {error}") from error
    if (
        len(critical_cpus) != 1
        or len(mps_cpus) != 1
        or len(telemetry_cpus) != 1
        or set(critical_cpus) & set(pressure_cpus)
        or set(critical_cpus) & set(mps_cpus)
        or set(critical_cpus) & set(telemetry_cpus)
        or set(pressure_cpus) & set(mps_cpus)
        or set(pressure_cpus) & set(telemetry_cpus)
        or set(mps_cpus) & set(telemetry_cpus)
    ):
        raise SystemExit(
            "critical, MPS, and telemetry must each use one dedicated CPU, and all "
            "CPU assignments must be disjoint"
        )
    args.critical_cpu = format_cpu_list(critical_cpus)
    args.pressure_cpus = format_cpu_list(pressure_cpus)
    args.mps_cpu = format_cpu_list(mps_cpus)
    args.telemetry_cpu = format_cpu_list(telemetry_cpus)
    if guard_profile_ms is None:
        guard_profile_ms = legacy_guard_profile(
            args.language_guard_ms, args.audio_guard_ms
        )
    profiled_full_gate_bound_ms = maximum_profiled_guard_ms(guard_profile_ms)
    thermal_pilot_protocol_valid = args.thermal_pilot_seconds is None or (
        args.thermal_pilot_seconds < args.thermal_timeout_seconds
        and math.isclose(
            args.thermal_pilot_seconds,
            THERMAL_PILOT_MINIMUM_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            args.thermal_timeout_seconds,
            THERMAL_PILOT_MAXIMUM_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            args.thermal_window_seconds,
            THERMAL_PILOT_WINDOW_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            args.thermal_max_slope_c_per_minute,
            THERMAL_MAXIMUM_SLOPE_C_PER_MINUTE,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    if (
        args.epochs <= 0
        or args.samples <= 0
        or args.warmup < 0
        or args.burst_size <= 0
        or args.samples % args.burst_size != 0
        or args.period_ms <= 0.0
        or args.pressure_rps_per_tenant < 0.0
        or not math.isfinite(args.pressure_rps_per_tenant)
        or args.slo_factor <= 0.0
        or (args.deadline_ms is not None and args.deadline_ms <= 0.0)
        or (args.deadline_source is not None and args.deadline_ms is None)
        or args.dmr_target <= 0.0
        or args.dmr_target >= 1.0
        or args.calibration_repeats <= 0
        or args.readiness_timeout_seconds <= 0.0
        or args.borrower_quota <= 0
        or args.borrower_quota > 100
        or args.language_guard_ms <= 0.0
        or args.audio_guard_ms <= 0.0
        or (
            args.thermal_pilot_seconds is not None
            and args.thermal_pilot_seconds <= 0.0
        )
        or (args.thermal_target_c is not None and args.thermal_target_c <= 0.0)
        or args.thermal_tolerance_c <= 0.0
        or args.thermal_window_seconds <= 0.0
        or args.thermal_max_slope_c_per_minute <= 0.0
        or args.thermal_timeout_seconds <= 0.0
        or not math.isclose(
            args.thermal_handoff_max_ms,
            THERMAL_HANDOFF_MAX_MS,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or args.thermal_stability_sensor != THERMAL_STABILITY_SENSOR
        or args.thermal_safety_sensor != THERMAL_SAFETY_SENSOR
        or (
            args.thermal_hard_limit_c is not None
            and args.thermal_hard_limit_c <= 0.0
        )
        or args.max_isolated_drift_fraction <= 0.0
        or args.max_isolated_drift_fraction >= 1.0
        or not thermal_pilot_protocol_valid
        or (
            args.calibration_only and args.thermal_pilot_seconds is not None
        )
        or (
            args.thermal_target_c is not None
            and args.thermal_pilot_seconds is not None
        )
        or (
            (args.thermal_target_c is not None or args.thermal_pilot_seconds is not None)
            and args.telemetry_log is None
        )
        or (
            args.guard_override_ms is None
            and profiled_full_gate_bound_ms >= args.period_ms
        )
        or (
            args.guard_override_ms is not None
            and (
                args.guard_override_ms <= 0.0
                or args.guard_override_ms >= args.period_ms
            )
        )
        or (
            args.deadline_lock_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", args.deadline_lock_sha256) is None
        )
        or (
            args.thermal_lock_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", args.thermal_lock_sha256) is None
        )
        or (
            args.guard_lock_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", args.guard_lock_sha256) is None
        )
        or ((args.guard_lock is None) != (args.guard_lock_sha256 is None))
        or (
            args.deadline_source == "frozen-isolated-p99-factor"
            and (
                args.deadline_lock_sha256 is None
                or args.thermal_lock_sha256 is None
                or args.guard_lock_sha256 is None
            )
        )
        or (
            args.deadline_lock_sha256 is not None
            and args.deadline_source != "frozen-isolated-p99-factor"
        )
        or (
            args.thermal_target_c is not None
            and args.thermal_lock_sha256 is None
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
    return requested


def run_calibration_set(
    args: argparse.Namespace,
    critical_env: dict[str, str],
    timeout: float,
    monitor: TegrastatsMonitor | None,
    stage: str,
    *,
    base_env: dict[str, str] | None = None,
    mig: dict[str, str] | None = None,
    precondition_each_repeat: bool = False,
) -> tuple[list[dict[str, Any]], list[pathlib.Path], list[dict[str, Any]]]:
    calibrations: list[dict[str, Any]] = []
    traces: list[pathlib.Path] = []
    preconditions: list[dict[str, Any]] = []
    if args.thermal_target_c is not None and not precondition_each_repeat:
        raise ValueError("target-normalized calibration requires every-repeat preheat")
    if precondition_each_repeat and (
        monitor is None
        or base_env is None
        or mig is None
        or args.thermal_target_c is None
    ):
        raise ValueError("per-block calibration preconditioning is incomplete")
    for repeat in range(1, args.calibration_repeats + 1):
        metadata = {"stage": stage, "repeat": repeat}
        mark_event(monitor, "calibration_prepare", metadata)
        trace = args.output.parent / "raw" / f"isolated-{stage}-r{repeat}.csv"
        traces.append(trace)
        process: subprocess.Popen[str] | None = None
        calibration: dict[str, Any] | None = None
        precondition: dict[str, Any] | None = None
        thermal_start: dict[str, float | int] | None = None
        thermal_start_telemetry: dict[str, Any] | None = None
        thermal_handoff: dict[str, Any] | None = None
        thermal_start_attempts: list[dict[str, Any]] = []
        thermal_start_qualification: dict[str, Any] | None = None
        thermal_actual_start_qualification: dict[str, Any] | None = None
        qualification_result_marker_ns: int | None = None
        release_ns: int | None = None
        readiness_affinity: dict[str, Any] | None = None
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            process = subprocess.Popen(
                critical_command(args, trace, None, start_paused=True),
                env=critical_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            wait_until_paused([process], args.readiness_timeout_seconds)
            readiness_affinity = process_affinity_snapshot(
                process.pid, expand_cpu_list(args.critical_cpu)[0]
            )
            if precondition_each_repeat:
                assert monitor is not None
                assert base_env is not None
                assert mig is not None
                assert args.thermal_target_c is not None
                for attempt in range(1, THERMAL_QUALIFICATION_MAX_ATTEMPTS + 1):
                    wait_until_paused([process], args.readiness_timeout_seconds)
                    require_processes_paused([process])
                    thermal_label = (
                        f"pre-{stage}-calibration-r{repeat}"
                        f"-attempt-{attempt:02d}"
                    )
                    candidate = run_thermal_load(
                        args,
                        base_env,
                        mig,
                        monitor,
                        label=thermal_label,
                        target_c=args.thermal_target_c,
                    )
                    wait_until_paused([process], args.readiness_timeout_seconds)
                    measured_process_states = require_processes_paused([process])
                    qualification = qualify_thermal_start(
                        args,
                        monitor,
                        label=thermal_label,
                        attempt=attempt,
                        boundary_ns=int(candidate["measurement_end_monotonic_ns"]),
                        cleanup_end_ns=int(candidate["cleanup_end_monotonic_ns"]),
                    )
                    validate_thermal_qualification_evidence(
                        qualification, expected_attempt=attempt
                    )
                    measured_process_states = require_processes_paused([process])
                    qualification_result_marker_ns = mark_event(
                        monitor,
                        "thermal_start_qualification_result",
                        metadata
                        | {
                            "label": thermal_label,
                            "attempt": attempt,
                            "qualification_monotonic_ns": int(
                                qualification["qualification_monotonic_ns"]
                            ),
                            "passed": qualification["passed"],
                            "failure_reason": qualification["failure_reason"],
                        },
                    )
                    attempt_record = {
                        "attempt": attempt,
                        "thermal_precondition": candidate,
                        "qualification": qualification,
                        "qualification_result_marker_monotonic_ns": (
                            qualification_result_marker_ns
                        ),
                        "measured_process_states": measured_process_states,
                    }
                    thermal_start_attempts.append(attempt_record)
                    if qualification["passed"]:
                        precondition = candidate
                        thermal_start_qualification = qualification
                        break
                if thermal_start_qualification is None:
                    raise RuntimeError(
                        "thermal start qualification exhausted the fixed three attempts"
                    )
                require_successful_thermal_qualification(
                    thermal_start_qualification
                )
                assert precondition is not None
                assert qualification_result_marker_ns is not None
                preconditions.append(precondition)
                thermal_start = precondition["last_window"]
                thermal_start_telemetry = thermal_start_qualification["telemetry"]
            release_ns = mark_event(monitor, "calibration_start", metadata)
            if thermal_start_qualification is not None:
                thermal_handoff = validate_thermal_qualification_handoff(
                    thermal_start_qualification,
                    qualification_result_marker_ns,
                    release_ns,
                    release_ns,
                    args.thermal_handoff_max_ms,
                )
            resume_processes([process])
            calibration = collect_json(process, timeout)
            validate_critical_result(
                calibration,
                process_pid=process.pid,
                args=args,
                critical_uuid=str(critical_env["CUDA_VISIBLE_DEVICES"]),
                deadline_ms=0.0,
                gated_processes=0,
                stopped_processes=0,
                guard_ms=0.0,
                start_paused=True,
            )
            measurement_start_ns = int(
                calibration["measurement_start_monotonic_ns"]
            )
            measurement_end_ns = int(
                calibration["measurement_end_monotonic_ns"]
            )
            if not release_ns <= measurement_start_ns < measurement_end_ns:
                raise RuntimeError(
                    "calibration returned inconsistent measurement clocks"
                )
            if thermal_start_qualification is not None:
                assert monitor is not None
                assert qualification_result_marker_ns is not None
                thermal_actual_start_qualification = validate_actual_thermal_start(
                    args,
                    monitor,
                    label=f"{stage}-calibration-r{repeat}",
                    measurement_start_ns=measurement_start_ns,
                    window_not_before_ns=int(
                        thermal_start_qualification[
                            "cleanup_end_monotonic_ns"
                        ]
                    ),
                )
                mark_event(
                    monitor,
                    "thermal_actual_start_qualification_result",
                    metadata
                    | {
                        "measurement_start_monotonic_ns": measurement_start_ns,
                        "sample_monotonic_ns": thermal_actual_start_qualification[
                            "sample_monotonic_ns"
                        ],
                        "passed": thermal_actual_start_qualification["passed"],
                        "failure_reason": thermal_actual_start_qualification[
                            "failure_reason"
                        ],
                    },
                )
                if thermal_actual_start_qualification["passed"] is not True:
                    raise RuntimeError(
                        "calibration actual start failed thermal qualification"
                    )
                thermal_handoff = validate_thermal_qualification_handoff(
                    thermal_start_qualification,
                    qualification_result_marker_ns,
                    release_ns,
                    measurement_start_ns,
                    args.thermal_handoff_max_ms,
                )
            calibration["thermal_start"] = thermal_start
            calibration["thermal_start_telemetry"] = thermal_start_telemetry
            calibration["thermal_start_attempts"] = thermal_start_attempts
            calibration["thermal_precondition"] = precondition
            calibration["thermal_start_qualification"] = (
                thermal_start_qualification
            )
            calibration["thermal_actual_start_qualification"] = (
                thermal_actual_start_qualification
            )
            calibration["thermal_start_stable"] = (
                thermal_window_is_stable(
                    thermal_start,
                    target_c=args.thermal_target_c,
                    tolerance_c=args.thermal_tolerance_c,
                    window_seconds=args.thermal_window_seconds,
                    maximum_slope_c_per_minute=(
                        args.thermal_max_slope_c_per_minute
                    ),
                )
                if args.thermal_target_c is not None
                else None
            )
            calibration["thermal_handoff"] = thermal_handoff
            calibration["thermal_precondition_label"] = (
                precondition["label"] if precondition is not None else None
            )
            calibration["measurement_release_monotonic_ns"] = release_ns
            calibration["readiness_affinity"] = readiness_affinity
            mark_event(
                monitor,
                "calibration_measurement_window",
                metadata
                | {
                    "measurement_start_monotonic_ns": measurement_start_ns,
                    "measurement_end_monotonic_ns": measurement_end_ns,
                },
            )
            calibrations.append(calibration)
        except BaseException as error:
            primary_error = error
        finally:
            try:
                terminate_process(process)
            except BaseException as error:
                cleanup_error = error
        try:
            mark_event(monitor, "calibration_end", metadata)
        except BaseException as error:
            if primary_error is None and cleanup_error is None:
                cleanup_error = error
        if primary_error is not None:
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error
        if calibration is None:
            raise RuntimeError("calibration benchmark produced no result")
    return calibrations, traces, preconditions


def main() -> int:
    args = parse_args()
    guard_lock: dict[str, Any] | None = None
    if args.guard_lock is not None:
        try:
            guard_lock, profile_guard_ms, actual_guard_lock_sha256 = load_guard_lock(
                args.guard_lock, args.guard_lock_sha256
            )
        except (OSError, TypeError, ValueError) as error:
            raise SystemExit(f"guard lock verification failed: {error}") from error
        args.guard_lock_sha256 = actual_guard_lock_sha256
    else:
        profile_guard_ms = legacy_guard_profile(
            args.language_guard_ms, args.audio_guard_ms
        )
    requested = validate_args(args, profile_guard_ms)
    platform_hard_limit_c = platform_thermal_hard_limit_c()
    if args.thermal_hard_limit_c is None:
        args.thermal_hard_limit_c = platform_hard_limit_c
    elif args.thermal_hard_limit_c > platform_hard_limit_c:
        raise SystemExit(
            f"thermal hard limit {args.thermal_hard_limit_c:.3f} C exceeds "
            f"the platform safety bound {platform_hard_limit_c:.3f} C"
        )
    mig = load_env(args.mig_env)
    base_env = os.environ.copy()
    base_env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.2/lib64:" + base_env.get(
        "LD_LIBRARY_PATH", ""
    )
    critical_env = base_env | {
        "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_BIG_UUID"],
        "CUDA_MPS_PIPE_DIRECTORY": str(args.big_mps_pipe),
        "CUDA_MPS_LOG_DIRECTORY": str(args.big_mps_log),
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "raw").mkdir(parents=True, exist_ok=True)
    initial_artifacts = artifact_hashes(args)
    initial_hardware = hardware_fingerprint(args.output.parent)
    selected_cpus = set(
        expand_cpu_list(args.critical_cpu)
        + expand_cpu_list(args.pressure_cpus)
        + expand_cpu_list(args.mps_cpu)
        + expand_cpu_list(args.telemetry_cpu)
    )
    expected_instances = [
        {
            "gpu": 0,
            "profile_name": "1g.0gb+me",
            "profile_id": 78,
            "instance_id": 2,
            "placement_start": 2,
            "placement_size": 1,
        },
        {
            "gpu": 0,
            "profile_name": "2g.0gb+gfx",
            "profile_id": 83,
            "instance_id": 1,
            "placement_start": 0,
            "placement_size": 2,
        },
    ]
    cpu_clocks = initial_hardware["cpu_clocks_khz"]
    expected_mps_affinity = [
        {"placement": placement, "role": role, "cpus": expand_cpu_list(args.mps_cpu)}
        for placement in ("critical-2g", "resident-1g")
        for role in ("control", "server")
    ]
    if (
        initial_hardware["gpu_product_name"] != "NVIDIA Thor"
        or initial_hardware["mig_mode"] != "Enabled"
        or initial_hardware["active_mig_instances"] != expected_instances
        or initial_hardware["mig_device_uuid_by_profile"]
        != {
            "1g.0gb": mig["JDG_MIG_SMALL_UUID"],
            "2g.0gb": mig["JDG_MIG_BIG_UUID"],
        }
        or initial_hardware["power_mode"] != "MAXN"
        or initial_hardware["mps_affinity"] != expected_mps_affinity
        or selected_cpus != set(initial_hardware["online_cpus"])
        or set(cpu_clocks) != set(initial_hardware["online_cpus"])
        or any(
            clock["min_khz"] != 2_601_000
            or clock["max_khz"] != 2_601_000
            for clock in cpu_clocks.values()
        )
        or initial_hardware["gpu_clock_min_hz"] != 1_575_000_000
        or initial_hardware["gpu_clock_max_hz"] != 1_575_000_000
        or initial_hardware["emc_clock_min_hz"] != 4_266_000_000
        or initial_hardware["emc_clock_max_hz"] != 4_266_000_000
        or initial_hardware["fan_dynamic_control"] != "disabled"
        or initial_hardware["fan_pwm"] != 255
    ):
        raise RuntimeError(f"platform fingerprint is not formal-ready: {initial_hardware}")
    timeout = args.samples / args.burst_size * args.period_ms / 1000.0 + 60.0
    telemetry_session = (
        start_telemetry_session(
            args.telemetry_log,
            args.output.parent / "telemetry.jsonl",
        )
        if args.telemetry_log is not None
        else None
    )
    telemetry_monitor = (
        telemetry_session.monitor if telemetry_session is not None else None
    )
    telemetry_errors: list[str] = []
    try:
        sequence_precondition = None
        post_sequence_precondition = None
        calibration_preconditions: list[dict[str, Any]] = []
        post_calibration_preconditions: list[dict[str, Any]] = []
        post_calibrations: list[dict[str, Any]] = []
        isolated_post_pooled_p99_ms: float | None = None
        isolated_post_pooled_samples = 0
        isolated_reference_p99_ms: float | None = None
        isolated_pre_reference_drift_fraction: float | None = None
        isolated_post_reference_drift_fraction: float | None = None
        isolated_drift_fraction: float | None = None
        isolated_drift_valid: bool | None = None
        (
            calibrations,
            calibration_traces,
            calibration_preconditions,
        ) = run_calibration_set(
            args,
            critical_env,
            timeout,
            telemetry_monitor,
            "pre",
            base_env=base_env,
            mig=mig,
            precondition_each_repeat=args.thermal_target_c is not None,
        )
        isolated_p99_values = [
            float(item["release_to_completion"]["p99_ms"])
            for item in calibrations
        ]
        isolated_pooled_p99_ms, isolated_pooled_samples = pooled_trace_p99(
            calibration_traces
        )
        deadline_ms = (
            args.deadline_ms
            if args.deadline_ms is not None
            else isolated_pooled_p99_ms * args.slo_factor
        )
        if (
            args.deadline_source == "frozen-isolated-p99-factor"
            and args.deadline_ms is not None
        ):
            isolated_reference_p99_ms = args.deadline_ms / args.slo_factor
            isolated_pre_reference_drift_fraction = abs(
                isolated_pooled_p99_ms - isolated_reference_p99_ms
            ) / isolated_reference_p99_ms
        thermal_pilot = None
        if args.thermal_pilot_seconds is not None:
            if telemetry_monitor is None:
                raise RuntimeError("thermal pilot requires telemetry")
            thermal_pilot = run_thermal_load(
                args,
                base_env,
                mig,
                telemetry_monitor,
                label="thermal-pilot",
                duration_seconds=args.thermal_pilot_seconds,
            )
            policies = []
        elif args.calibration_only:
            policies = []
        else:
            policies = [
                run_policy(
                    policy,
                    args,
                    base_env,
                    mig,
                    critical_env,
                    deadline_ms,
                    profile_guard_ms,
                    telemetry_monitor,
                )
                for policy in requested
            ]
            if args.deadline_ms is not None:
                (
                    post_calibrations,
                    post_traces,
                    post_calibration_preconditions,
                ) = run_calibration_set(
                    args,
                    critical_env,
                    timeout,
                    telemetry_monitor,
                    "post",
                    base_env=base_env,
                    mig=mig,
                    precondition_each_repeat=args.thermal_target_c is not None,
                )
                (
                    isolated_post_pooled_p99_ms,
                    isolated_post_pooled_samples,
                ) = pooled_trace_p99(post_traces)
                isolated_drift_fraction = abs(
                    isolated_post_pooled_p99_ms - isolated_pooled_p99_ms
                ) / isolated_pooled_p99_ms
                drift_values = [isolated_drift_fraction]
                if isolated_reference_p99_ms is not None:
                    isolated_post_reference_drift_fraction = abs(
                        isolated_post_pooled_p99_ms - isolated_reference_p99_ms
                    ) / isolated_reference_p99_ms
                    assert isolated_pre_reference_drift_fraction is not None
                    drift_values.extend(
                        (
                            isolated_pre_reference_drift_fraction,
                            isolated_post_reference_drift_fraction,
                        )
                    )
                isolated_drift_valid = (
                    max(drift_values) <= args.max_isolated_drift_fraction
                )
    finally:
        if telemetry_session is not None:
            try:
                mark_event(telemetry_monitor, "collector_end", {})
            except BaseException as error:
                telemetry_errors.append(
                    f"collector_end marker failed: {type(error).__name__}: {error}"
                )
            try:
                telemetry_errors.extend(close_telemetry_session(telemetry_session))
            except BaseException as error:
                telemetry_errors.append(
                    f"collector close failed: {type(error).__name__}: {error}"
                )
    if telemetry_errors:
        raise RuntimeError("telemetry collector failed: " + "; ".join(telemetry_errors))
    final_artifacts = artifact_hashes(args)
    if final_artifacts != initial_artifacts:
        raise RuntimeError("implementation, benchmark, or engine changed during run")
    if hardware_fingerprint(args.output.parent) != initial_hardware:
        raise RuntimeError("platform snapshot changed during run")
    output = {
        "schema_version": THERMAL_PROTOCOL_SCHEMA_VERSION,
        "config": {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples,
            "warmup": args.warmup,
            "burst_size": args.burst_size,
            "period_ms": args.period_ms,
            "pressure_rps_per_tenant": args.pressure_rps_per_tenant,
            "slo_factor": args.slo_factor,
            "fixed_deadline_ms": args.deadline_ms,
            "deadline_source": (
                args.deadline_source or "fixed-explicit"
                if args.deadline_ms is not None
                else "isolated-p99-factor"
            ),
            "deadline_lock_sha256": args.deadline_lock_sha256,
            "thermal_lock_sha256": args.thermal_lock_sha256,
            "guard_lock_sha256": args.guard_lock_sha256,
            "dmr_target": args.dmr_target,
            "calibration_repeats": args.calibration_repeats,
            "scenario": args.scenario,
            "readiness_timeout_seconds": args.readiness_timeout_seconds,
            "cpu_affinity": {
                "critical": expand_cpu_list(args.critical_cpu),
                "pressure": expand_cpu_list(args.pressure_cpus),
                "mps": expand_cpu_list(args.mps_cpu),
                "telemetry": expand_cpu_list(args.telemetry_cpu),
            },
            "policy_order": requested,
            "borrower_quota": args.borrower_quota,
            "guard_override_ms": args.guard_override_ms,
            "profile_guard_ms": profile_guard_ms,
            "guard_profile_source": (
                "frozen-quota-aware-lock"
                if guard_lock is not None
                else "legacy-flat-exploratory"
            ),
            "experiment_label": args.experiment_label,
            "calibration_only": args.calibration_only,
            "thermal_pilot_seconds": args.thermal_pilot_seconds,
            "thermal_pilot_maximum_seconds": (
                args.thermal_timeout_seconds
                if args.thermal_pilot_seconds is not None
                else None
            ),
            "thermal_target_c": args.thermal_target_c,
            "thermal_tolerance_c": args.thermal_tolerance_c,
            "thermal_window_seconds": args.thermal_window_seconds,
            "thermal_max_slope_c_per_minute": (
                args.thermal_max_slope_c_per_minute
            ),
            "thermal_timeout_seconds": args.thermal_timeout_seconds,
            "thermal_stability_checkpoint_seconds": (
                THERMAL_STABILITY_CHECKPOINT_SECONDS
            ),
            "thermal_stability_checkpoint_max_lateness_seconds": (
                THERMAL_STABILITY_CHECKPOINT_MAX_LATENESS_SECONDS
            ),
            "thermal_required_stable_checkpoints": (
                THERMAL_REQUIRED_STABLE_CHECKPOINTS
            ),
            "thermal_hard_limit_c": args.thermal_hard_limit_c,
            "thermal_stability_sensor": args.thermal_stability_sensor,
            "thermal_safety_sensor": args.thermal_safety_sensor,
            "thermal_handoff_max_ms": args.thermal_handoff_max_ms,
            "thermal_handoff_boundary": THERMAL_HANDOFF_BOUNDARY,
            "thermal_qualification_max_attempts": (
                THERMAL_QUALIFICATION_MAX_ATTEMPTS
            ),
            "thermal_active_stable_endpoints": (
                THERMAL_ACTIVE_STABLE_ENDPOINTS
            ),
            "thermal_active_stable_spacing_seconds": (
                THERMAL_ACTIVE_STABLE_SPACING_SECONDS
            ),
            "thermal_calibration_preconditioning": (
                "per-repeat-preloaded-critical"
                if args.thermal_target_c is not None
                else "not-requested"
            ),
            "platform_thermal_hard_limit_c": platform_hard_limit_c,
            "max_isolated_drift_fraction": args.max_isolated_drift_fraction,
            "trace": [
                list(epoch) for epoch in SCENARIO_TRACES[args.scenario]
            ],
            "trace_assignment": "fixed-audio-language-pairs-2-4-6-repeat",
            "dependency_semantics": (
                "none"
                if args.scenario == "independent"
                else "audio-completion-token-precedes-language-inference"
            ),
            "includes_transfers": True,
            "mig_enabled": True,
            "critical_placement": "2g",
            "resident_placement": "1g",
            "borrower_placement": "2g",
            "worker_max_inflight": 1,
            "gate_protocol": "cooperative-drain-ack",
            "start_protocol": (
                "post-warmup-stop-barrier-with-bounded-thermal-handoff"
            ),
            "telemetry_source": (
                "tegrastats-readall-monotonic-jsonl"
                if args.telemetry_log is not None
                else "not-configured"
            ),
            "tegrastats_requested_interval_ms": (
                TEGRASTATS_REQUESTED_INTERVAL_MS
            ),
            "telemetry_interval_ms": TELEMETRY_INTERVAL_MS,
            "telemetry_required_fraction": TELEMETRY_REQUIRED_FRACTION,
            "telemetry_required_fields": list(TELEMETRY_REQUIRED_FIELDS),
            "telemetry_stale_after_ms": TELEMETRY_STALE_AFTER_MS,
            "telemetry_max_gap_ms": TELEMETRY_STALE_AFTER_MS,
            "goodput_denominator": "each-worker-measured-window",
            "goodput_aggregation": (
                "sum-per-worker-rates-time-weighted-by-critical-window"
            ),
            "pressure_load": "saturated-per-tenant-request-loops",
            "worker_semantics": "one-worker-per-admitted-tenant",
        },
        "mig": {
            "critical_uuid": mig["JDG_MIG_BIG_UUID"],
            "resident_uuid": mig["JDG_MIG_SMALL_UUID"],
        },
        "artifacts": initial_artifacts,
        "hardware": initial_hardware,
        "isolated": calibrations,
        "isolated_preconditions": calibration_preconditions,
        "isolated_p99_ms": isolated_p99_values,
        "isolated_pooled_p99_ms": isolated_pooled_p99_ms,
        "isolated_pooled_samples": isolated_pooled_samples,
        "sequence_precondition": sequence_precondition,
        "isolated_post": post_calibrations,
        "isolated_post_preconditions": post_calibration_preconditions,
        "isolated_post_pooled_p99_ms": isolated_post_pooled_p99_ms,
        "isolated_post_pooled_samples": isolated_post_pooled_samples,
        "post_sequence_precondition": post_sequence_precondition,
        "isolated_reference_p99_ms": isolated_reference_p99_ms,
        "isolated_pre_reference_drift_fraction": (
            isolated_pre_reference_drift_fraction
        ),
        "isolated_post_reference_drift_fraction": (
            isolated_post_reference_drift_fraction
        ),
        "isolated_drift_fraction": isolated_drift_fraction,
        "isolated_drift_valid": isolated_drift_valid,
        "deadline_ms": deadline_ms,
        "thermal_pilot": thermal_pilot,
        "policies": policies,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
