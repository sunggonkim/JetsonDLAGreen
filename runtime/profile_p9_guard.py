#!/usr/bin/env python3
"""Collect the fixed P9 quota-aware cooperative-drain calibration campaign.

This producer deliberately records raw evidence only.  Guard estimation and
acceptance of the held-out configurations belong to ``freeze_p9_guard.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.mig_slack_governor import (  # noqa: E402
    TELEMETRY_REQUIRED_FIELDS,
    THERMAL_ACTIVE_STABLE_ENDPOINTS,
    THERMAL_ACTIVE_STABLE_SPACING_SECONDS,
    THERMAL_HANDOFF_BOUNDARY,
    THERMAL_QUALIFICATION_MAX_ATTEMPTS,
    close_telemetry_session,
    hardware_fingerprint,
    load_env,
    mark_event,
    process_affinity_snapshot,
    process_state,
    qualify_thermal_start,
    require_successful_thermal_qualification,
    resume_processes,
    run_thermal_load,
    start_telemetry_session,
    thermal_window_maximum_gap_seconds,
    thermal_window_is_stable,
    thermal_window_summary,
    validate_actual_thermal_start,
    validate_thermal_qualification_evidence,
    validate_thermal_qualification_handoff,
    wait_until_paused,
)


SCHEMA_VERSION = 3
KIND = "p9-quota-aware-guard-profile"
BLOCKS = 10
EVENTS_PER_BLOCK = 1_000
WARMUP_REQUESTS = 100
FORMAL_PERIOD_MS = 20.0
PERIOD_MS = 40.0
PROFILING_GUARD_MS = 20.0
PERCENTILE = 0.999
ESTIMATOR = "empirical-type-7"
MARGIN = 1.20
ROUNDING_MS = 0.1
THERMAL_TIMEOUT_SECONDS = 1_800.0
THERMAL_STABILITY_SENSOR = "soc012"
THERMAL_SAFETY_SENSOR = "tj"
THERMAL_HANDOFF_MAX_MS = 500.0
THERMAL_LOCK_SCHEMA_VERSION = 4
BLOCK_MAX_ATTEMPTS = 3

CPU_AFFINITY = {
    "pressure": list(range(0, 11)),
    "mps": [11],
    "critical": [12],
    "telemetry": [13],
}
HARDWARE_SNAPSHOTS = (
    "nv_tegra_release.txt",
    "nvidia-smi.txt",
    "active-mig-instances.txt",
    "gpu-inventory.txt",
    "jetson-clocks.txt",
    "mps-affinity.tsv",
    "nvpmodel.txt",
)

MODELS = {
    "language": "distilbert-sst2",
    "audio": "whisper-tiny-encoder",
}


@dataclasses.dataclass(frozen=True)
class ClientSpec:
    placement: str
    quota_percent: int
    modality: str
    count: int = 1

    def __post_init__(self) -> None:
        if self.placement not in {"resident-1g", "borrower-2g"}:
            raise ValueError(f"unsupported placement: {self.placement}")
        if self.quota_percent not in {25, 50, 100}:
            raise ValueError(f"unsupported quota: {self.quota_percent}")
        if self.placement == "borrower-2g" and self.quota_percent != 100:
            raise ValueError("borrower-2g is calibrated only at q100")
        if self.modality not in MODELS:
            raise ValueError(f"unsupported modality: {self.modality}")
        if self.count <= 0:
            raise ValueError("client count must be positive")

    @property
    def model(self) -> str:
        return MODELS[self.modality]

    def to_json(self) -> dict[str, Any]:
        return {
            "placement": self.placement,
            "quota_percent": self.quota_percent,
            "modality": self.modality,
            "model": self.model,
            "count": self.count,
        }


@dataclasses.dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    clients: tuple[ClientSpec, ...]
    held_out: bool

    @property
    def expanded_clients(self) -> tuple[ClientSpec, ...]:
        return tuple(
            dataclasses.replace(client, count=1)
            for client in self.clients
            for _ in range(client.count)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "held_out": self.held_out,
            "clients": [client.to_json() for client in self.clients],
        }


def single_client_cases() -> tuple[CalibrationCase, ...]:
    cases: list[CalibrationCase] = []
    for quota in (25, 50, 100):
        for modality in ("language", "audio"):
            cases.append(
                CalibrationCase(
                    f"resident-1g-q{quota}-{modality}",
                    (ClientSpec("resident-1g", quota, modality),),
                    False,
                )
            )
    for modality in ("language", "audio"):
        cases.append(
            CalibrationCase(
                f"borrower-2g-q100-{modality}",
                (ClientSpec("borrower-2g", 100, modality),),
                False,
            )
        )
    return tuple(cases)


def held_out_cases() -> tuple[CalibrationCase, ...]:
    return (
        CalibrationCase(
            "heldout-resident-q100-audio-x6",
            (ClientSpec("resident-1g", 100, "audio", 6),),
            True,
        ),
        CalibrationCase(
            "heldout-split-resident-q50-audio-x3-borrower-q100-audio-x3",
            (
                ClientSpec("resident-1g", 50, "audio", 3),
                ClientSpec("borrower-2g", 100, "audio", 3),
            ),
            True,
        ),
        CalibrationCase(
            "heldout-split-resident-q25-audio-x3-borrower-q100-audio-x3",
            (
                ClientSpec("resident-1g", 25, "audio", 3),
                ClientSpec("borrower-2g", 100, "audio", 3),
            ),
            True,
        ),
    )


def protocol_json(mode: str = "formal") -> dict[str, Any]:
    if mode not in {"formal", "smoke"}:
        raise ValueError("guard protocol mode must be formal or smoke")
    return {
        "mode": mode,
        "formal": mode == "formal",
        "blocks": BLOCKS,
        "events_per_block": EVENTS_PER_BLOCK,
        "warmup_requests": WARMUP_REQUESTS,
        "period_ms": PERIOD_MS,
        "formal_period_ms": FORMAL_PERIOD_MS,
        "profiling_guard_ms": PROFILING_GUARD_MS,
        "gate_protocol": "cooperative-drain-ack",
        "outstanding_depth": 1,
        "percentile": PERCENTILE,
        "estimator": ESTIMATOR,
        "margin": MARGIN,
        "rounding_ms": ROUNDING_MS,
        "cpu_affinity": CPU_AFFINITY,
        "critical_driver": {
            "placement": "mig-2g",
            "quota_percent": 100,
            "model": "resnet50-v2",
            "priority": "high",
        },
        "thermal_precondition": {
            "per_block": True,
            "offered_modalities": ["audio"] * 6,
            "resident_clients": 3,
            "borrower_clients": 3,
            "quota_percent": 100,
            "stability_sensor": THERMAL_STABILITY_SENSOR,
            "safety_sensor": THERMAL_SAFETY_SENSOR,
            "handoff_boundary": THERMAL_HANDOFF_BOUNDARY,
            "handoff_max_ms": THERMAL_HANDOFF_MAX_MS,
            "active_stable_endpoints": THERMAL_ACTIVE_STABLE_ENDPOINTS,
            "active_stable_spacing_seconds": (
                THERMAL_ACTIVE_STABLE_SPACING_SECONDS
            ),
            "qualification_max_attempts": THERMAL_QUALIFICATION_MAX_ATTEMPTS,
            "measured_processes_paused_until_success": True,
            "first_postcleanup_causal_sample": True,
            "actual_start_causal_gate": True,
            "block_max_attempts": BLOCK_MAX_ATTEMPTS,
            "retry_on_performance": False,
        },
        "single_client_cases": [case.to_json() for case in single_client_cases()],
        "held_out_cases": [case.to_json() for case in held_out_cases()],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def engine_path(engine_root: Path, client: ClientSpec) -> Path:
    partition = "mig-1g" if client.placement == "resident-1g" else "mig-2g"
    return engine_root / f"{partition}-q{client.quota_percent}" / f"{client.model}.engine"


def placement_environment(
    client: ClientSpec,
    *,
    base_env: Mapping[str, str],
    small_uuid: str,
    big_uuid: str,
    resident_mps_pipe: Path,
    resident_mps_log: Path,
    big_mps_pipe: Path,
    big_mps_log: Path,
) -> dict[str, str]:
    env = dict(base_env)
    if client.placement == "resident-1g":
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": small_uuid,
                "CUDA_MPS_PIPE_DIRECTORY": str(resident_mps_pipe),
                "CUDA_MPS_LOG_DIRECTORY": str(resident_mps_log),
            }
        )
    else:
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": big_uuid,
                "CUDA_MPS_PIPE_DIRECTORY": str(big_mps_pipe),
                "CUDA_MPS_LOG_DIRECTORY": str(big_mps_log),
            }
        )
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(client.quota_percent)
    return env


def worker_command(
    bench: Path, engine: Path, client: ClientSpec, cpu: int
) -> list[str]:
    priority = "default" if client.placement == "resident-1g" else "low"
    return [
        "taskset",
        "--cpu-list",
        str(cpu),
        str(bench),
        "--engine",
        str(engine),
        "--model-name",
        engine.stem,
        "--role",
        "pressure",
        "--duration-seconds",
        "3600",
        "--warmup",
        str(WARMUP_REQUESTS),
        "--include-transfers",
        "true",
        "--priority",
        priority,
        "--start-paused",
        "true",
    ]


def critical_command(
    bench: Path,
    engine: Path,
    trace_path: Path,
    worker_pids: Sequence[int],
    cpu: int = 12,
) -> list[str]:
    if not worker_pids:
        raise ValueError("critical calibration requires at least one gated worker")
    pid_csv = ",".join(str(pid) for pid in worker_pids)
    return [
        "taskset",
        "--cpu-list",
        str(cpu),
        str(bench),
        "--engine",
        str(engine),
        "--model-name",
        engine.stem,
        "--role",
        "benchmark",
        "--warmup",
        str(WARMUP_REQUESTS),
        "--samples",
        str(EVENTS_PER_BLOCK),
        "--burst-size",
        "1",
        "--period-ms",
        f"{PERIOD_MS:.1f}",
        "--include-transfers",
        "true",
        "--priority",
        "high",
        "--trace",
        str(trace_path),
        "--gate-pids",
        pid_csv,
        "--stop-pids",
        pid_csv,
        "--gate-mode",
        "cooperative",
        "--guard-ms",
        f"{PROFILING_GUARD_MS:.1f}",
        "--start-paused",
        "true",
    ]


def _read_process_json(stdout: str, description: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description} emitted invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} JSON root is not an object")
    return value


def _terminate(processes: Iterable[subprocess.Popen[str]]) -> None:
    alive: list[subprocess.Popen[str]] = []
    errors: list[str] = []
    for process in processes:
        try:
            returncode = process.poll()
        except OSError as error:
            errors.append(f"pid={process.pid}:poll:{type(error).__name__}:{error}")
            returncode = None
        if returncode is None:
            alive.append(process)
    for process in alive:
        try:
            os.kill(process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f"pid={process.pid}:sigcont:{type(error).__name__}:{error}")
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f"pid={process.pid}:sigint:{type(error).__name__}:{error}")
    deadline = time.monotonic() + 5.0
    for process in alive:
        remaining = max(0.0, deadline - time.monotonic())
        escalate = False
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            escalate = True
        except ProcessLookupError:
            continue
        except OSError as error:
            errors.append(f"pid={process.pid}:wait:{type(error).__name__}:{error}")
            escalate = True
        if not escalate:
            continue
        try:
            process.kill()
        except ProcessLookupError:
            continue
        except OSError as error:
            errors.append(f"pid={process.pid}:kill:{type(error).__name__}:{error}")
            continue
        try:
            process.wait(timeout=5.0)
        except ProcessLookupError:
            pass
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(
                f"pid={process.pid}:wait-after-kill:"
                f"{type(error).__name__}:{error}"
            )
    if errors:
        raise RuntimeError("process cleanup failures: " + "; ".join(errors))


def _validate_guard_thermal_lock(thermal_lock: Mapping[str, Any]) -> None:
    if thermal_lock.get("schema_version") != THERMAL_LOCK_SCHEMA_VERSION:
        raise ValueError(
            "guard calibration requires thermal-lock schema version 4"
        )
    if thermal_lock.get("stability_sensor") != THERMAL_STABILITY_SENSOR:
        raise ValueError("thermal lock does not bind the soc012 stability sensor")
    if thermal_lock.get("safety_sensor") != THERMAL_SAFETY_SENSOR:
        raise ValueError("thermal lock does not bind the tj safety sensor")
    handoff_max_ms = thermal_lock.get("thermal_handoff_max_ms")
    if (
        isinstance(handoff_max_ms, bool)
        or not isinstance(handoff_max_ms, (int, float))
        or not math.isfinite(float(handoff_max_ms))
        or float(handoff_max_ms) != THERMAL_HANDOFF_MAX_MS
    ):
        raise ValueError("thermal lock does not bind the 500 ms handoff limit")
    if thermal_lock.get("thermal_handoff_boundary") != THERMAL_HANDOFF_BOUNDARY:
        raise ValueError("thermal lock has a stale handoff boundary")
    if (
        thermal_lock.get("thermal_qualification_max_attempts")
        != THERMAL_QUALIFICATION_MAX_ATTEMPTS
        or thermal_lock.get("thermal_active_stable_endpoints")
        != THERMAL_ACTIVE_STABLE_ENDPOINTS
        or thermal_lock.get("thermal_active_stable_spacing_seconds")
        != THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        or "thermal_qualification_dwell_seconds" in thermal_lock
    ):
        raise ValueError("thermal lock has a different qualification protocol")


def _thermal_start_at_marker(
    monitor: Any,
    thermal_lock: Mapping[str, Any],
    reference_ns: int,
) -> dict[str, Any]:
    """Validate stability and safety at the exact block-start marker."""
    _validate_guard_thermal_lock(thermal_lock)
    stability_sensor = str(thermal_lock["stability_sensor"])
    safety_sensor = str(thermal_lock["safety_sensor"])
    window_seconds = float(thermal_lock["stability_window_seconds"])
    summary = thermal_window_summary(
        monitor,
        stability_sensor,
        window_seconds,
        reference_ns,
    )
    if summary is not None:
        summary = dict(summary)
        summary["maximum_gap_seconds"] = thermal_window_maximum_gap_seconds(
            monitor,
            stability_sensor,
            window_seconds,
            reference_ns,
        )
    if not thermal_window_is_stable(
        summary,
        target_c=float(thermal_lock["target_c"]),
        tolerance_c=float(thermal_lock["tolerance_c"]),
        window_seconds=window_seconds,
        maximum_slope_c_per_minute=float(
            thermal_lock["maximum_slope_c_per_minute"]
        ),
    ):
        raise RuntimeError(f"thermal start precondition failed: {summary}")

    interval_ms = float(thermal_lock["telemetry_interval_ms"])
    required_fraction = float(thermal_lock["telemetry_required_fraction"])
    maximum_gap_ms = float(thermal_lock["telemetry_max_gap_ms"])
    expected_samples = math.floor(window_seconds * 1000.0 / interval_ms)
    minimum_valid_samples = max(
        1, math.floor(expected_samples * required_fraction)
    )
    start_ns = reference_ns - int(window_seconds * 1_000_000_000)
    end_ns = reference_ns
    gap_ns = int(maximum_gap_ms * 1_000_000.0)
    required_fields = tuple(
        dict.fromkeys(
            (*TELEMETRY_REQUIRED_FIELDS, f"temperature:{stability_sensor}")
        )
    )
    telemetry = monitor.aggregate(
        start_ns,
        end_ns,
        required_fields=required_fields,
        minimum_valid_samples=minimum_valid_samples,
        reference_ns=reference_ns,
        end_inclusive=True,
        stale_after_ns=gap_ns,
        maximum_valid_gap_ns=gap_ns,
    )
    health = telemetry.get("health")
    temperatures = telemetry.get("temperatures_c")
    safety = (
        temperatures.get(safety_sensor)
        if isinstance(temperatures, dict)
        else None
    )
    if not isinstance(health, dict) or health.get("healthy") is not True:
        raise RuntimeError(f"thermal start telemetry is unhealthy: {health}")
    if (
        not isinstance(safety, dict)
        or not isinstance(safety.get("max"), (int, float))
        or float(safety["max"]) >= float(thermal_lock["hard_limit_c"])
    ):
        raise RuntimeError(f"thermal safety limit reached at block start: {safety}")
    return summary


def _handoff_elapsed_ms(
    boundary_ns: int,
    observed_ns: int,
    thermal_lock: Mapping[str, Any],
    label: str,
) -> float:
    maximum_ms = float(thermal_lock["thermal_handoff_max_ms"])
    elapsed_ms = (observed_ns - boundary_ns) / 1_000_000.0
    if observed_ns <= boundary_ns or elapsed_ms >= maximum_ms:
        raise RuntimeError(
            f"thermal handoff {label} exceeded {maximum_ms:.1f} ms: "
            f"{elapsed_ms:.6f} ms"
        )
    return elapsed_ms


def _best_effort_process_states(
    processes: Iterable[subprocess.Popen[str]],
) -> dict[str, str]:
    states: dict[str, str] = {}
    for process in processes:
        pid = process.pid
        returncode = process.poll()
        if returncode is not None:
            states[str(pid)] = "exited"
            continue
        try:
            states[str(pid)] = process_state(pid)
        except (FileNotFoundError, ProcessLookupError):
            states[str(pid)] = "exited"
        except (OSError, RuntimeError) as error:
            states[str(pid)] = f"unavailable:{type(error).__name__}"
    return states


def _require_processes_paused(
    processes: Sequence[subprocess.Popen[str]],
) -> dict[str, str]:
    states = _best_effort_process_states(processes)
    expected = {str(process.pid): "T" for process in processes}
    if states != expected:
        raise RuntimeError(
            "measured processes left the stopped barrier before qualification: "
            f"{states}"
        )
    return states


def _cleanup_aborted_block(
    *,
    case: CalibrationCase,
    block: int,
    workers: Sequence[subprocess.Popen[str]],
    critical: subprocess.Popen[str] | None,
    telemetry_writer: Any,
    block_attempt: int | None = None,
) -> None:
    """Best-effort cleanup that must never replace the block's primary error."""
    cleanup_errors: list[str] = []
    try:
        worker_states = _best_effort_process_states(workers)
    except BaseException as error:
        worker_states = {}
        cleanup_errors.append(f"states:{type(error).__name__}:{error}")
    processes = list(workers) + ([critical] if critical is not None else [])
    try:
        _terminate(process for process in processes if process is not None)
    except BaseException as error:
        cleanup_errors.append(f"terminate:{type(error).__name__}:{error}")
    abort_metadata: dict[str, Any] = {
        "case_id": case.case_id,
        "held_out": case.held_out,
        "block": block,
        "worker_states": worker_states,
    }
    if block_attempt is not None:
        abort_metadata["attempt"] = block_attempt
    if cleanup_errors:
        abort_metadata["cleanup_errors"] = cleanup_errors
    try:
        mark_event(
            telemetry_writer,
            "guard_block_abort",
            abort_metadata,
        )
    except BaseException:
        pass


def thermal_load_arguments(
    *,
    bench: Path,
    engine_root: Path,
    big_mps_pipe: Path,
    big_mps_log: Path,
    thermal_lock: Mapping[str, Any],
) -> SimpleNamespace:
    """Build the exact argument surface consumed by governor.run_thermal_load."""
    return SimpleNamespace(
        bench=bench,
        engine_root=engine_root,
        big_mps_pipe=big_mps_pipe,
        big_mps_log=big_mps_log,
        borrower_quota=100,
        warmup=WARMUP_REQUESTS,
        # Guard calibration uses saturated best-effort workers; they are
        # launched without an arrival-rate throttle.
        pressure_rps_per_tenant=0.0,
        pressure_cpus="0-10",
        readiness_timeout_seconds=60.0,
        thermal_timeout_seconds=THERMAL_TIMEOUT_SECONDS,
        thermal_hard_limit_c=float(thermal_lock["hard_limit_c"]),
        thermal_stability_sensor=str(thermal_lock["stability_sensor"]),
        thermal_safety_sensor=str(thermal_lock["safety_sensor"]),
        thermal_handoff_max_ms=float(thermal_lock["thermal_handoff_max_ms"]),
        thermal_target_c=float(thermal_lock["target_c"]),
        thermal_window_seconds=float(thermal_lock["stability_window_seconds"]),
        thermal_tolerance_c=float(thermal_lock["tolerance_c"]),
        thermal_max_slope_c_per_minute=float(
            thermal_lock["maximum_slope_c_per_minute"]
        ),
    )


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _block_paths(
    root: Path,
    case: CalibrationCase,
    block: int,
    attempt: int,
) -> dict[str, Path]:
    stem = f"{case.case_id}-block-{block:02d}-attempt-{attempt:02d}"
    raw = root / "raw"
    return {
        "trace": raw / f"{stem}-critical.csv",
        "critical": raw / f"{stem}-critical.json",
    }


_QUALIFICATION_KEYS = {
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
_ACTUAL_START_KEYS = {
    "passed",
    "measurement_start_monotonic_ns",
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


def _handoff_evidence(
    qualification: Mapping[str, Any],
    *,
    qualification_result_ns: int,
    block_start_ns: int,
    measurement_release_ns: int,
    resume_issued_ns: int,
    measurement_start_ns: int,
    thermal_lock: Mapping[str, Any],
) -> dict[str, Any]:
    boundary_ns = int(qualification["boundary_monotonic_ns"])
    maximum_ms = float(thermal_lock["thermal_handoff_max_ms"])
    clocks = {
        "cleanup_end": int(qualification["cleanup_end_monotonic_ns"]),
        "sample": int(qualification["sample_monotonic_ns"]),
        "qualification": int(qualification["qualification_monotonic_ns"]),
        "qualification_result": qualification_result_ns,
        "block_start": block_start_ns,
        "measurement_release": measurement_release_ns,
        "resume_issued": resume_issued_ns,
        "critical_measurement_start": measurement_start_ns,
    }
    result: dict[str, Any] = {
        "boundary": THERMAL_HANDOFF_BOUNDARY,
        "boundary_monotonic_ns": boundary_ns,
        "maximum_ms": maximum_ms,
    }
    strictly_within_bound = True
    previous_ns = boundary_ns
    for label, observed_ns in clocks.items():
        elapsed_ms = (observed_ns - boundary_ns) / 1_000_000.0
        result[f"{label}_monotonic_ns"] = observed_ns
        result[f"boundary_to_{label}_ms"] = elapsed_ms
        if observed_ns <= boundary_ns or observed_ns <= previous_ns or elapsed_ms >= maximum_ms:
            strictly_within_bound = False
        previous_ns = observed_ns
    result["strictly_within_bound"] = strictly_within_bound
    return result


def _pre_release_is_timely(
    qualification: Mapping[str, Any],
    qualification_result_ns: int,
    block_start_ns: int,
    release_ns: int,
    thermal_lock: Mapping[str, Any],
) -> bool:
    boundary_ns = int(qualification["boundary_monotonic_ns"])
    maximum_ms = float(thermal_lock["thermal_handoff_max_ms"])
    clocks = (
        int(qualification["cleanup_end_monotonic_ns"]),
        int(qualification["sample_monotonic_ns"]),
        int(qualification["qualification_monotonic_ns"]),
        qualification_result_ns,
        block_start_ns,
        release_ns,
    )
    return all(
        boundary_ns < observed_ns
        and (observed_ns - boundary_ns) / 1_000_000.0 < maximum_ms
        for observed_ns in clocks
    ) and all(
        current >= previous
        for previous, current in zip(clocks, clocks[1:], strict=False)
    )


def _validate_qualification_shape(
    value: object,
    *,
    attempt: int,
    boundary_ns: int,
    cleanup_end_ns: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _QUALIFICATION_KEYS:
        raise RuntimeError("thermal qualification returned invalid evidence")
    validate_thermal_qualification_evidence(value, expected_attempt=attempt)
    if (
        value["boundary"] != THERMAL_HANDOFF_BOUNDARY
        or value["boundary_monotonic_ns"] != boundary_ns
        or value["cleanup_end_monotonic_ns"] != cleanup_end_ns
    ):
        raise RuntimeError("thermal qualification changed the active boundary")
    return value


def _validate_actual_start_shape(
    value: object,
    *,
    measurement_start_ns: int,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != _ACTUAL_START_KEYS
        or value.get("measurement_start_monotonic_ns") != measurement_start_ns
        or type(value.get("passed")) is not bool
    ):
        raise RuntimeError("actual thermal-start validation returned invalid evidence")
    return value


def _run_block_attempt(
    *,
    case: CalibrationCase,
    block: int,
    block_attempt: int,
    output_root: Path,
    bench: Path,
    engine_root: Path,
    small_uuid: str,
    big_uuid: str,
    resident_mps_pipe: Path,
    resident_mps_log: Path,
    big_mps_pipe: Path,
    big_mps_log: Path,
    monitor: Any,
    telemetry_writer: Any,
    thermal_lock: Mapping[str, Any],
    popen: Callable[..., subprocess.Popen[str]],
) -> dict[str, Any]:
    paths = _block_paths(output_root, case, block, block_attempt)
    paths["trace"].parent.mkdir(parents=True, exist_ok=True)
    if paths["trace"].exists() or paths["critical"].exists():
        raise FileExistsError(
            "refusing to overwrite block-attempt artifacts: "
            f"{case.case_id}/{block}/{block_attempt}"
        )
    base_metadata = {
        "case_id": case.case_id,
        "held_out": case.held_out,
        "block": block,
        "attempt": block_attempt,
    }
    prepare_marker_ns = mark_event(
        telemetry_writer, "guard_block_prepare", base_metadata
    )
    workers: list[subprocess.Popen[str]] = []
    worker_specs: list[dict[str, Any]] = []
    critical: subprocess.Popen[str] | None = None
    try:
        for index, client in enumerate(case.expanded_clients):
            engine = engine_path(engine_root, client)
            cpu = CPU_AFFINITY["pressure"][index % len(CPU_AFFINITY["pressure"])]
            process = popen(
                worker_command(bench, engine, client, cpu),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=placement_environment(
                    client,
                    base_env=os.environ,
                    small_uuid=small_uuid,
                    big_uuid=big_uuid,
                    resident_mps_pipe=resident_mps_pipe,
                    resident_mps_log=resident_mps_log,
                    big_mps_pipe=big_mps_pipe,
                    big_mps_log=big_mps_log,
                ),
                start_new_session=True,
            )
            workers.append(process)
            worker_specs.append(
                {
                    **client.to_json(),
                    "count": 1,
                    "worker_index": index,
                    "cpu": cpu,
                    "engine": str(engine.resolve()),
                    "engine_sha256": sha256_file(engine),
                    "pid": process.pid,
                }
            )
        wait_until_paused(workers, timeout_seconds=60.0)
        for process, spec in zip(workers, worker_specs, strict=True):
            spec["affinity"] = process_affinity_snapshot(process.pid, int(spec["cpu"]))

        critical_engine = engine_root / "mig-2g" / "resnet50-v2.engine"
        critical = popen(
            critical_command(
                bench,
                critical_engine,
                paths["trace"],
                [process.pid for process in workers],
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=placement_environment(
                ClientSpec("borrower-2g", 100, "language"),
                base_env=os.environ,
                small_uuid=small_uuid,
                big_uuid=big_uuid,
                resident_mps_pipe=resident_mps_pipe,
                resident_mps_log=resident_mps_log,
                big_mps_pipe=big_mps_pipe,
                big_mps_log=big_mps_log,
            ),
            start_new_session=True,
        )
        wait_until_paused([critical], timeout_seconds=60.0)
        critical_affinity = process_affinity_snapshot(
            critical.pid, CPU_AFFINITY["critical"][0]
        )
        measured_processes = [*workers, critical]
        thermal_args = thermal_load_arguments(
            bench=bench,
            engine_root=engine_root,
            big_mps_pipe=big_mps_pipe,
            big_mps_log=big_mps_log,
            thermal_lock=thermal_lock,
        )
        thermal_mig = {
            "JDG_MIG_SMALL_UUID": small_uuid,
            "JDG_MIG_BIG_UUID": big_uuid,
            "JDG_MPS_PIPE_DIRECTORY": str(resident_mps_pipe),
            "JDG_MPS_LOG_DIRECTORY": str(resident_mps_log),
        }
        thermal_attempts: list[dict[str, Any]] = []
        selected_thermal_attempt: dict[str, Any] | None = None
        start_marker_ns: int | None = None
        release_marker_ns: int | None = None
        for thermal_attempt in range(1, THERMAL_QUALIFICATION_MAX_ATTEMPTS + 1):
            wait_until_paused(measured_processes, timeout_seconds=60.0)
            states = _require_processes_paused(measured_processes)
            thermal_label = (
                f"pre-p9-guard-{case.case_id}-block-{block:02d}"
                f"-attempt-{block_attempt:02d}-thermal-{thermal_attempt:02d}"
            )
            precondition = run_thermal_load(
                thermal_args,
                dict(os.environ),
                thermal_mig,
                monitor,
                label=thermal_label,
                target_c=float(thermal_lock["target_c"]),
            )
            states = _require_processes_paused(measured_processes)
            boundary_ns = int(precondition["measurement_end_monotonic_ns"])
            cleanup_end_ns = int(precondition["cleanup_end_monotonic_ns"])
            qualification = _validate_qualification_shape(
                qualify_thermal_start(
                    thermal_args,
                    monitor,
                    label=thermal_label,
                    attempt=thermal_attempt,
                    boundary_ns=boundary_ns,
                    cleanup_end_ns=cleanup_end_ns,
                ),
                attempt=thermal_attempt,
                boundary_ns=boundary_ns,
                cleanup_end_ns=cleanup_end_ns,
            )
            states = _require_processes_paused(measured_processes)
            qualification_result_ns = mark_event(
                telemetry_writer,
                "thermal_start_qualification_result",
                base_metadata
                | {
                    "label": thermal_label,
                    "thermal_attempt": thermal_attempt,
                    "boundary_monotonic_ns": boundary_ns,
                    "cleanup_end_monotonic_ns": cleanup_end_ns,
                    "qualification_monotonic_ns": int(
                        qualification["qualification_monotonic_ns"]
                    ),
                    "sample_monotonic_ns": qualification["sample_monotonic_ns"],
                    "passed": qualification["passed"],
                    "failure_reason": qualification["failure_reason"],
                },
            )
            attempt_record: dict[str, Any] = {
                "attempt": thermal_attempt,
                "label": thermal_label,
                "thermal_precondition": precondition,
                "qualification": qualification,
                "qualification_result_marker_monotonic_ns": qualification_result_ns,
                "measured_process_states": states,
                "start_marker_monotonic_ns": None,
                "measurement_release_marker_monotonic_ns": None,
                "pre_release_passed": False,
                "failure_reason": qualification["failure_reason"],
            }
            thermal_attempts.append(attempt_record)
            if qualification["passed"] is not True:
                continue
            handoff_metadata = {
                "thermal_attempt": thermal_attempt,
                "thermal_boundary": THERMAL_HANDOFF_BOUNDARY,
                "thermal_boundary_monotonic_ns": boundary_ns,
                "thermal_handoff_max_ms": THERMAL_HANDOFF_MAX_MS,
            }
            start_marker_ns = mark_event(
                telemetry_writer,
                "guard_block_start",
                base_metadata
                | {
                    "worker_pids": [process.pid for process in workers],
                    "critical_pid": critical.pid,
                }
                | handoff_metadata,
            )
            release_marker_ns = mark_event(
                telemetry_writer,
                "guard_block_measurement_release",
                base_metadata | {"critical_pid": critical.pid} | handoff_metadata,
            )
            attempt_record["start_marker_monotonic_ns"] = start_marker_ns
            attempt_record["measurement_release_marker_monotonic_ns"] = (
                release_marker_ns
            )
            timely = _pre_release_is_timely(
                qualification,
                qualification_result_ns,
                start_marker_ns,
                release_marker_ns,
                thermal_lock,
            )
            attempt_record["pre_release_passed"] = timely
            if not timely:
                attempt_record["failure_reason"] = (
                    "active-boundary pre-release handoff exceeded the strict limit"
                )
                continue
            attempt_record["failure_reason"] = None
            selected_thermal_attempt = attempt_record
            break
        if selected_thermal_attempt is None:
            raise RuntimeError(
                "thermal start qualification exhausted the fixed three attempts"
            )
        if start_marker_ns is None or release_marker_ns is None:
            raise RuntimeError("successful thermal attempt lacks release markers")

        qualification = selected_thermal_attempt["qualification"]
        require_successful_thermal_qualification(qualification)
        resume_marker_ns = mark_event(
            telemetry_writer,
            "guard_block_resume",
            base_metadata
            | {
                "critical_pid": critical.pid,
                "worker_pids": [process.pid for process in workers],
                "thermal_attempt": selected_thermal_attempt["attempt"],
                "thermal_boundary": THERMAL_HANDOFF_BOUNDARY,
                "thermal_boundary_monotonic_ns": qualification[
                    "boundary_monotonic_ns"
                ],
                "thermal_handoff_max_ms": THERMAL_HANDOFF_MAX_MS,
                "resume_semantics": "issued-before-sigcont",
            },
        )
        resume_processes(measured_processes)
        timeout_seconds = EVENTS_PER_BLOCK * PERIOD_MS / 1000.0 + 180.0
        critical_stdout, critical_stderr = critical.communicate(timeout=timeout_seconds)
        if critical.returncode != 0:
            raise RuntimeError(
                f"critical process failed ({critical.returncode}): "
                f"{critical_stderr.strip()}"
            )
        critical_result = _read_process_json(critical_stdout, "critical benchmark")
        measurement_start_ns = int(critical_result["measurement_start_monotonic_ns"])
        measurement_end_ns = int(critical_result["measurement_end_monotonic_ns"])
        thermal_handoff = _handoff_evidence(
            qualification,
            qualification_result_ns=int(
                selected_thermal_attempt[
                    "qualification_result_marker_monotonic_ns"
                ]
            ),
            block_start_ns=start_marker_ns,
            measurement_release_ns=release_marker_ns,
            resume_issued_ns=resume_marker_ns,
            measurement_start_ns=measurement_start_ns,
            thermal_lock=thermal_lock,
        )
        if thermal_handoff["strictly_within_bound"]:
            validate_thermal_qualification_handoff(
                qualification,
                int(
                    selected_thermal_attempt[
                        "qualification_result_marker_monotonic_ns"
                    ]
                ),
                release_marker_ns,
                measurement_start_ns,
                float(thermal_lock["thermal_handoff_max_ms"]),
            )
        actual_start = _validate_actual_start_shape(
            validate_actual_thermal_start(
                thermal_args,
                monitor,
                label=str(selected_thermal_attempt["label"]),
                measurement_start_ns=measurement_start_ns,
                window_not_before_ns=int(
                    qualification["cleanup_end_monotonic_ns"]
                ),
            ),
            measurement_start_ns=measurement_start_ns,
        )
        thermally_valid = bool(
            thermal_handoff["strictly_within_bound"]
            and actual_start["passed"] is True
        )
        actual_marker_ns = mark_event(
            telemetry_writer,
            "guard_actual_start_qualification",
            base_metadata | actual_start,
        )
        atomic_json(paths["critical"], critical_result)

        worker_paths: list[Path] = []
        for index, (process, spec) in enumerate(zip(workers, worker_specs, strict=True)):
            stdout, stderr = process.communicate(timeout=30.0)
            if process.returncode != 0:
                raise RuntimeError(
                    f"worker {index} failed ({process.returncode}): {stderr.strip()}"
                )
            record = {
                "schema_version": SCHEMA_VERSION,
                "kind": "p9-guard-worker-evidence",
                "client": spec,
                "result": _read_process_json(stdout, f"worker {index}"),
            }
            worker_path = (
                paths["trace"].parent
                / (
                    f"{case.case_id}-block-{block:02d}"
                    f"-attempt-{block_attempt:02d}-worker-{index:02d}.json"
                )
            )
            atomic_json(worker_path, record)
            worker_paths.append(worker_path)

        result_marker_ns = mark_event(
            telemetry_writer,
            "guard_block_result",
            base_metadata
            | {
                "measurement_start_monotonic_ns": measurement_start_ns,
                "measurement_end_monotonic_ns": measurement_end_ns,
                "actual_start_qualification_marker_monotonic_ns": actual_marker_ns,
                "thermal_handoff": thermal_handoff,
                "thermally_valid": thermally_valid,
            },
        )
        end_marker_ns = mark_event(
            telemetry_writer,
            "guard_block_end",
            base_metadata | {"thermally_valid": thermally_valid},
        )
        return {
            "attempt": block_attempt,
            "thermally_valid": thermally_valid,
            "prepare_marker_monotonic_ns": prepare_marker_ns,
            "start_marker_monotonic_ns": start_marker_ns,
            "measurement_release_marker_monotonic_ns": release_marker_ns,
            "resume_marker_monotonic_ns": resume_marker_ns,
            "actual_start_qualification_marker_monotonic_ns": actual_marker_ns,
            "result_marker_monotonic_ns": result_marker_ns,
            "end_marker_monotonic_ns": end_marker_ns,
            "measurement_start_monotonic_ns": measurement_start_ns,
            "measurement_end_monotonic_ns": measurement_end_ns,
            "thermal_attempts": thermal_attempts,
            "selected_thermal_attempt": selected_thermal_attempt["attempt"],
            "thermal_handoff": thermal_handoff,
            "actual_start_qualification": actual_start,
            "critical_affinity": critical_affinity,
            "critical_trace": _relative(paths["trace"], output_root),
            "critical_json": _relative(paths["critical"], output_root),
            "worker_json": [_relative(path, output_root) for path in worker_paths],
        }
    except BaseException:
        _cleanup_aborted_block(
            case=case,
            block=block,
            workers=workers,
            critical=critical,
            telemetry_writer=telemetry_writer,
            block_attempt=block_attempt,
        )
        raise


def run_block(
    *,
    case: CalibrationCase,
    block: int,
    output_root: Path,
    bench: Path,
    engine_root: Path,
    small_uuid: str,
    big_uuid: str,
    resident_mps_pipe: Path,
    resident_mps_log: Path,
    big_mps_pipe: Path,
    big_mps_log: Path,
    monitor: Any,
    telemetry_writer: Any,
    thermal_lock: Mapping[str, Any],
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
) -> dict[str, Any]:
    if block < 1 or block > BLOCKS:
        raise ValueError(f"block out of range: {block}")
    _validate_guard_thermal_lock(thermal_lock)
    attempts: list[dict[str, Any]] = []
    for block_attempt in range(1, BLOCK_MAX_ATTEMPTS + 1):
        attempt = _run_block_attempt(
            case=case,
            block=block,
            block_attempt=block_attempt,
            output_root=output_root,
            bench=bench,
            engine_root=engine_root,
            small_uuid=small_uuid,
            big_uuid=big_uuid,
            resident_mps_pipe=resident_mps_pipe,
            resident_mps_log=resident_mps_log,
            big_mps_pipe=big_mps_pipe,
            big_mps_log=big_mps_log,
            monitor=monitor,
            telemetry_writer=telemetry_writer,
            thermal_lock=thermal_lock,
            popen=popen,
        )
        attempts.append(attempt)
        if attempt["thermally_valid"]:
            return {
                "block": block,
                "selected_attempt": block_attempt,
                "attempts": attempts,
            }
    raise RuntimeError(
        "actual thermal-start qualification exhausted the fixed three block attempts"
    )


def _required_artifacts(bench: Path, engine_root: Path) -> dict[str, dict[str, str]]:
    paths: dict[str, Path] = {
        "benchmark": bench,
        "producer": Path(__file__).resolve(),
        "freezer": ROOT / "analysis" / "freeze_p9_guard.py",
        "telemetry_runtime": ROOT / "runtime" / "tegrastats_telemetry.py",
        "governor_runtime": ROOT / "runtime" / "mig_slack_governor.py",
        "guard_runner": ROOT / "scripts" / "run_p9_guard_calibration.sh",
        "formal_runner": ROOT / "scripts" / "run_p9_mig_slack_governor.sh",
        "mig_configurator": ROOT / "scripts" / "configure_thor_mig.sh",
        "benchmark_source": ROOT / "benchmarks" / "trt_inference.cpp",
    }
    for case in single_client_cases() + held_out_cases():
        for client in case.expanded_clients:
            path = engine_path(engine_root, client)
            paths[f"engine:{client.placement}:q{client.quota_percent}:{client.modality}"] = path
    paths["engine:critical:2g:resnet50-v2"] = (
        engine_root / "mig-2g" / "resnet50-v2.engine"
    )
    result: dict[str, dict[str, str]] = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"required artifact missing: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    return result


def _read_json_once(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    mode = getattr(args, "mode", "formal")
    if mode not in {"formal", "smoke"}:
        raise ValueError("guard campaign mode must be formal or smoke")
    # Smoke is an explicitly non-promotable execution path.  The freezer
    # compares the emitted protocol against the fixed formal protocol, so a
    # shortened run cannot accidentally become a calibration lock.
    global BLOCKS, EVENTS_PER_BLOCK
    original_blocks, original_events = BLOCKS, EVENTS_PER_BLOCK
    if mode == "smoke":
        BLOCKS, EVENTS_PER_BLOCK = 1, 20
    producer_affinity = sorted(os.sched_getaffinity(0))
    if producer_affinity != CPU_AFFINITY["telemetry"]:
        raise RuntimeError(
            "guard producer must be launched on the dedicated telemetry CPU"
        )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output_root = output.parent
    telemetry_jsonl = output_root / "telemetry.jsonl"
    if telemetry_jsonl.exists() or (output_root / "raw").exists():
        raise FileExistsError("guard campaign output directory contains prior raw evidence")

    thermal_lock, thermal_lock_sha256 = _read_json_once(args.thermal_lock.resolve())
    _validate_guard_thermal_lock(thermal_lock)
    env_values = load_env(args.mig_env)
    small_uuid = env_values["JDG_MIG_SMALL_UUID"]
    big_uuid = env_values["JDG_MIG_BIG_UUID"]
    if (
        args.resident_mps_pipe.resolve()
        != Path(env_values["JDG_MPS_PIPE_DIRECTORY"]).resolve()
        or args.resident_mps_log.resolve()
        != Path(env_values["JDG_MPS_LOG_DIRECTORY"]).resolve()
    ):
        raise ValueError("resident MPS paths differ from the frozen MIG environment")
    artifacts = _required_artifacts(args.bench.resolve(), args.engine_root.resolve())
    hardware = hardware_fingerprint(output_root)
    hardware_snapshot_sha256 = {
        name: sha256_file(output_root / name) for name in HARDWARE_SNAPSHOTS
    }
    telemetry_session = start_telemetry_session(
        args.telemetry_log.resolve(), telemetry_jsonl
    )
    try:
        os.sched_setaffinity(
            telemetry_session.tail_process.pid, set(CPU_AFFINITY["telemetry"])
        )
        campaign_start_ns = mark_event(
            telemetry_session.monitor,
            "guard_campaign_start",
            {"protocol": protocol_json(mode)},
        )
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "protocol": protocol_json(mode),
            "campaign_start_monotonic_ns": campaign_start_ns,
            "thermal_lock": {
                "path": str(args.thermal_lock.resolve()),
                "sha256": thermal_lock_sha256,
            },
            "hardware": hardware,
            "hardware_snapshot_sha256": hardware_snapshot_sha256,
            "mig": {
                "small_uuid": small_uuid,
                "big_uuid": big_uuid,
                "env_path": str(args.mig_env.resolve()),
                "env_sha256": sha256_file(args.mig_env.resolve()),
            },
            "mps": {
                "resident_pipe": str(args.resident_mps_pipe.resolve()),
                "resident_log": str(args.resident_mps_log.resolve()),
                "big_pipe": str(args.big_mps_pipe.resolve()),
                "big_log": str(args.big_mps_log.resolve()),
            },
            "cpu_affinity": CPU_AFFINITY,
            "producer_cpu_affinity": producer_affinity,
            "artifacts": artifacts,
            "telemetry_jsonl": _relative(telemetry_jsonl, output_root),
            "single_client": [],
            "held_out": [],
        }
        cases = single_client_cases() + held_out_cases()
        if mode == "smoke":
            cases = cases[:1]
        for case in cases:
            case_result = {**case.to_json(), "blocks": []}
            for block in range(1, BLOCKS + 1):
                case_result["blocks"].append(
                    run_block(
                        case=case,
                        block=block,
                        output_root=output_root,
                        bench=args.bench.resolve(),
                        engine_root=args.engine_root.resolve(),
                        small_uuid=small_uuid,
                        big_uuid=big_uuid,
                        resident_mps_pipe=args.resident_mps_pipe.resolve(),
                        resident_mps_log=args.resident_mps_log.resolve(),
                        big_mps_pipe=args.big_mps_pipe.resolve(),
                        big_mps_log=args.big_mps_log.resolve(),
                        monitor=telemetry_session.monitor,
                        telemetry_writer=telemetry_session.monitor,
                        thermal_lock=thermal_lock,
                    )
                )
            destination = "held_out" if case.held_out else "single_client"
            result[destination].append(case_result)
        result["campaign_end_monotonic_ns"] = mark_event(
            telemetry_session.monitor, "guard_campaign_end", {}
        )
    except BaseException as primary_error:
        try:
            telemetry_errors = close_telemetry_session(telemetry_session)
        except BaseException as cleanup_error:
            primary_error.add_note(
                "telemetry cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        else:
            if telemetry_errors:
                primary_error.add_note(
                    "telemetry cleanup also reported: "
                    + "; ".join(telemetry_errors)
                )
        raise
    telemetry_errors = close_telemetry_session(telemetry_session)
    if telemetry_errors:
        raise RuntimeError("telemetry collector failed: " + "; ".join(telemetry_errors))
    result["telemetry_jsonl_sha256"] = sha256_file(telemetry_jsonl)
    atomic_json(output, result)
    BLOCKS, EVENTS_PER_BLOCK = original_blocks, original_events
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--mig-env", type=Path, required=True)
    parser.add_argument("--thermal-lock", type=Path, required=True)
    parser.add_argument("--telemetry-log", type=Path, required=True)
    parser.add_argument("--resident-mps-pipe", type=Path, required=True)
    parser.add_argument("--resident-mps-log", type=Path, required=True)
    parser.add_argument("--big-mps-pipe", type=Path, required=True)
    parser.add_argument("--big-mps-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("formal", "smoke"),
        default="formal",
        help="formal fixed campaign, or explicitly non-promotable one-case smoke",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(PROFILING_GUARD_MS) or PROFILING_GUARD_MS >= PERIOD_MS:
        parser.error("fixed profiling guard must be finite and below the request period")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run_campaign(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
