#!/usr/bin/env python3
"""Replay and freeze the fixed P9 quota-aware cooperative-drain profile."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import statistics
from decimal import Decimal, ROUND_CEILING
from typing import Any, Iterable, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from analysis.freeze_p9_thermal import (  # noqa: E402
    LOCK_SCHEMA_VERSION as THERMAL_LOCK_SCHEMA_VERSION,
    MARKER_RECORD_KEYS,
    SAMPLE_RECORD_KEYS,
    THERMAL_ACTIVE_STABLE_ENDPOINTS,
    THERMAL_ACTIVE_STABLE_SPACING_SECONDS,
    THERMAL_HANDOFF_BOUNDARY,
    THERMAL_QUALIFICATION_MAX_ATTEMPTS,
    verify_lock as verify_thermal_lock,
)
from runtime.tegrastats_telemetry import (  # noqa: E402
    TelemetrySample,
    aggregate_samples,
    parse_tegrastats_line,
)


SCHEMA_VERSION = 3
PROFILE_KIND = "p9-quota-aware-guard-profile"
LOCK_KIND = "p9-quota-aware-guard-lock"
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
THERMAL_STABILITY_SENSOR = "soc012"
THERMAL_SAFETY_SENSOR = "tj"
THERMAL_HANDOFF_MAX_MS = 500.0
BLOCK_MAX_ATTEMPTS = 3
CPU_AFFINITY = {
    "pressure": list(range(0, 11)),
    "mps": [11],
    "critical": [12],
    "telemetry": [13],
}
TELEMETRY_REQUIRED_FIELDS = (
    "ram",
    "mem_available",
    "cpu",
    f"temperature:{THERMAL_STABILITY_SENSOR}",
    f"temperature:{THERMAL_SAFETY_SENSOR}",
    "power:VIN",
)
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
TRACE_FIELDS = (
    "request",
    "release_to_completion_ms",
    "gpu_service_ms",
    "queue_delay_ms",
    "gate_overhead_ms",
    "drain_ms",
    "resume_ms",
)
LATENCY_SUMMARY_KEYS = frozenset(
    {
        "count",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "p999_ms",
        "max_ms",
    }
)
WORKER_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "model",
        "role",
        "engine",
        "execution_environment",
        "gpu",
        "config",
        "release_to_completion",
        "gpu_service",
        "queue_delay",
        "gate_overhead",
        "drain",
        "resume",
        "completed_requests",
        "throughput_per_second",
        "measurement_start_monotonic_ns",
        "measurement_end_monotonic_ns",
        "elapsed_seconds",
        "deadline_misses",
        "deadline_miss_rate",
    }
)
CURRENT_FILES = {
    "producer": ROOT / "runtime" / "profile_p9_guard.py",
    "freezer": pathlib.Path(__file__).resolve(),
    "telemetry_runtime": ROOT / "runtime" / "tegrastats_telemetry.py",
    "governor_runtime": ROOT / "runtime" / "mig_slack_governor.py",
    "guard_runner": ROOT / "scripts" / "run_p9_guard_calibration.sh",
    "formal_runner": ROOT / "scripts" / "run_p9_mig_slack_governor.sh",
    "mig_configurator": ROOT / "scripts" / "configure_thor_mig.sh",
    "benchmark_source": ROOT / "benchmarks" / "trt_inference.cpp",
}


def _client(
    placement: str,
    quota_percent: int,
    modality: str,
    count: int = 1,
) -> dict[str, Any]:
    return {
        "placement": placement,
        "quota_percent": quota_percent,
        "modality": modality,
        "model": MODELS[modality],
        "count": count,
    }


def expected_single_cases() -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    for quota in (25, 50, 100):
        for modality in ("language", "audio"):
            cases.append(
                {
                    "case_id": f"resident-1g-q{quota}-{modality}",
                    "held_out": False,
                    "clients": [_client("resident-1g", quota, modality)],
                }
            )
    for modality in ("language", "audio"):
        cases.append(
            {
                "case_id": f"borrower-2g-q100-{modality}",
                "held_out": False,
                "clients": [_client("borrower-2g", 100, modality)],
            }
        )
    return tuple(cases)


def expected_held_out_cases() -> tuple[dict[str, Any], ...]:
    return (
        {
            "case_id": "heldout-resident-q100-audio-x6",
            "held_out": True,
            "clients": [_client("resident-1g", 100, "audio", 6)],
        },
        {
            "case_id": (
                "heldout-split-resident-q50-audio-x3-"
                "borrower-q100-audio-x3"
            ),
            "held_out": True,
            "clients": [
                _client("resident-1g", 50, "audio", 3),
                _client("borrower-2g", 100, "audio", 3),
            ],
        },
        {
            "case_id": (
                "heldout-split-resident-q25-audio-x3-"
                "borrower-q100-audio-x3"
            ),
            "held_out": True,
            "clients": [
                _client("resident-1g", 25, "audio", 3),
                _client("borrower-2g", 100, "audio", 3),
            ],
        },
    )


def expected_protocol() -> dict[str, Any]:
    return {
        "mode": "formal",
        "formal": True,
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
        "single_client_cases": list(expected_single_cases()),
        "held_out_cases": list(expected_held_out_cases()),
    }


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_type7(values: Sequence[float], quantile: float) -> float:
    if not values or not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("Type-7 percentile requires samples and q in [0, 1]")
    ordered = sorted(_finite_nonnegative(value, "percentile sample") for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def round_up_ms(value: float, quantum_ms: float = ROUNDING_MS) -> float:
    value = _finite_nonnegative(value, "guard value")
    quantum_ms = _finite_positive(quantum_ms, "rounding quantum")
    decimal_value = Decimal(str(value))
    quantum = Decimal(str(quantum_ms))
    units = (decimal_value / quantum).to_integral_value(rounding=ROUND_CEILING)
    return float(units * quantum)


def estimate_guard_ms(values: Sequence[float]) -> tuple[float, float]:
    raw_p999 = percentile_type7(values, PERCENTILE)
    guard_ms = round_up_ms(raw_p999 * MARGIN)
    if guard_ms >= FORMAL_PERIOD_MS:
        raise ValueError("single-client guard does not fit the formal 20 ms period")
    return raw_p999, guard_ms


def require_held_out_coverage(
    case_id: str, *, envelope_ms: float, observed_max_ms: float
) -> None:
    envelope = _finite_nonnegative(envelope_ms, "held-out envelope")
    observed = _finite_nonnegative(observed_max_ms, "held-out observed maximum")
    if envelope >= FORMAL_PERIOD_MS:
        raise ValueError(f"held-out case {case_id} does not fit the formal 20 ms period")
    if envelope + 1e-12 < observed:
        raise ValueError(f"held-out case {case_id} exceeds the additive guard envelope")


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} is not finite and nonnegative")
    return result


def _finite_positive(value: object, label: str) -> float:
    result = _finite_nonnegative(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _close(left: object, right: object, label: str, tolerance: float = 1e-8) -> None:
    lhs = _finite_nonnegative(left, label)
    rhs = _finite_nonnegative(right, label)
    if not math.isclose(lhs, rhs, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label} differs from raw evidence: {lhs} != {rhs}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class EvidenceRegistry:
    """Claims independent raw files and records immutable file identities."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root.resolve()
        self._paths: set[pathlib.Path] = set()
        self._inodes: set[tuple[int, int]] = set()
        self._hashes: set[str] = set()
        self.manifest: list[dict[str, Any]] = []

    def claim(self, raw_path: object, *, kind: str, case_id: str, block: int) -> bytes:
        if not isinstance(raw_path, str) or not raw_path or pathlib.Path(raw_path).is_absolute():
            raise ValueError("raw evidence paths must be non-empty and relative")
        candidate = self.root / raw_path
        if candidate.is_symlink():
            raise ValueError("raw evidence symlinks are forbidden")
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"raw evidence is missing: {raw_path}") from exc
        if self.root != resolved and self.root not in resolved.parents:
            raise ValueError("raw evidence path escapes the campaign directory")
        if not resolved.is_file():
            raise ValueError(f"raw evidence is not a regular file: {raw_path}")
        stat = resolved.stat()
        inode = (int(stat.st_dev), int(stat.st_ino))
        payload = resolved.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if resolved in self._paths:
            raise ValueError("raw evidence path was reused")
        if inode in self._inodes:
            raise ValueError("raw evidence inode was reused (hardlink detected)")
        if digest in self._hashes:
            raise ValueError("byte-identical raw evidence was reused")
        self._paths.add(resolved)
        self._inodes.add(inode)
        self._hashes.add(digest)
        self.manifest.append(
            {
                "path": str(resolved),
                "relative_path": raw_path,
                "kind": kind,
                "case_id": case_id,
                "block": block,
                "st_dev": inode[0],
                "st_ino": inode[1],
                "size_bytes": len(payload),
                "sha256": digest,
            }
        )
        return payload


def replay_critical_trace(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("critical trace is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != TRACE_FIELDS:
        raise ValueError("critical trace has an unexpected header")
    drains: list[float] = []
    resumes: list[float] = []
    for expected_request, row in enumerate(reader):
        if set(row) != set(TRACE_FIELDS) or int(row["request"]) != expected_request:
            raise ValueError("critical trace request sequence is invalid")
        numeric = {
            name: _finite_nonnegative(float(row[name]), f"trace {name}")
            for name in TRACE_FIELDS[1:]
        }
        _close(
            numeric["gate_overhead_ms"],
            numeric["drain_ms"] + numeric["resume_ms"],
            "trace gate overhead",
            tolerance=1e-7,
        )
        drains.append(numeric["drain_ms"])
        resumes.append(numeric["resume_ms"])
    if len(drains) != EVENTS_PER_BLOCK:
        raise ValueError("critical trace does not contain exactly 1000 drain events")
    if any(value <= 0.0 for value in drains):
        raise ValueError("every cooperative drain event must have a positive acknowledgement")
    return {
        "drains_ms": drains,
        "resumes_ms": resumes,
        "samples": len(drains),
        "drain_p999_ms": percentile_type7(drains, PERCENTILE),
        "drain_max_ms": max(drains),
    }


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile_type7(values, 0.50),
        "p95_ms": percentile_type7(values, 0.95),
        "p99_ms": percentile_type7(values, 0.99),
        "p999_ms": percentile_type7(values, 0.999),
        "max_ms": max(values),
    }


def _load_json_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root is not an object")
    return value


def _validate_critical_json(
    result: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    expected_workers: int,
    expected_engine: pathlib.Path,
    big_uuid: str,
) -> None:
    if result.get("schema_version") != 1 or result.get("role") != "benchmark":
        raise ValueError("critical JSON has invalid schema or role")
    if result.get("model") != "resnet50-v2":
        raise ValueError("critical JSON model identity is invalid")
    if pathlib.Path(str(result.get("engine", ""))).resolve() != expected_engine.resolve():
        raise ValueError("critical JSON engine identity is invalid")
    environment = result.get("execution_environment")
    if not isinstance(environment, dict) or set(environment) != {
        "pid",
        "cuda_visible_devices",
        "mps_active_thread_percentage",
        "cpu_affinity",
    } or (
        int(environment.get("pid", -1)) <= 0
        or environment.get("cuda_visible_devices") != big_uuid
        or environment.get("mps_active_thread_percentage") != 100
        or environment.get("cpu_affinity") != CPU_AFFINITY["critical"]
    ):
        raise ValueError("critical JSON execution environment is invalid")
    if result.get("gpu") != {
        "name": "NVIDIA Thor MIG 2g.0gb",
        "multiprocessors": 12,
    }:
        raise ValueError("critical JSON differs from the formal 2g MIG width")
    config = result.get("config")
    if not isinstance(config, dict):
        raise ValueError("critical JSON lacks configuration")
    exact_config = {
        "warmup": WARMUP_REQUESTS,
        "burst_size": 1,
        "period_ms": PERIOD_MS,
        "guard_ms": PROFILING_GUARD_MS,
        "gated_processes": expected_workers,
        "stopped_processes": expected_workers,
        "gate_mode": "cooperative",
        "start_paused": True,
        "include_transfers": True,
        "priority": "high",
        "deadline_ms": 0,
        "duration_seconds": 0,
        "stream_priority_value": -5,
    }
    if set(config) != {
        "warmup",
        "burst_size",
        "period_ms",
        "deadline_ms",
        "duration_seconds",
        "guard_ms",
        "gated_processes",
        "stopped_processes",
        "gate_mode",
        "start_paused",
        "include_transfers",
        "priority",
        "stream_priority_value",
    } or any(config.get(key) != value for key, value in exact_config.items()):
        raise ValueError("critical JSON differs from the fixed gate protocol")
    if int(result.get("completed_requests", -1)) != EVENTS_PER_BLOCK:
        raise ValueError("critical JSON completed request count is invalid")
    drains = trace["drains_ms"]
    raw_summary = _summary(drains)
    stored_summary = result.get("drain")
    if not isinstance(stored_summary, dict):
        raise ValueError("critical JSON lacks a drain summary")
    for key, value in raw_summary.items():
        if key == "count":
            if stored_summary.get(key) != value:
                raise ValueError("critical drain count differs from trace")
        else:
            _close(stored_summary.get(key), value, f"critical drain {key}")
    start_ns = int(result.get("measurement_start_monotonic_ns", -1))
    end_ns = int(result.get("measurement_end_monotonic_ns", -1))
    if start_ns < 0 or end_ns <= start_ns:
        raise ValueError("critical measurement clock interval is invalid")
    _close(
        result.get("elapsed_seconds"),
        (end_ns - start_ns) / 1_000_000_000.0,
        "critical elapsed_seconds",
    )


def _expanded_clients(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for client in case["clients"]:
        for _ in range(int(client["count"])):
            expanded.append({**client, "count": 1})
    return expanded


def _engine_artifact_key(client: Mapping[str, Any]) -> str:
    return (
        f"engine:{client['placement']}:q{client['quota_percent']}:"
        f"{client['modality']}"
    )


def _validate_affinity_snapshot(value: object, pid: int, cpu: int) -> None:
    if not isinstance(value, dict) or set(value) != {
        "pid",
        "expected_cpu",
        "tasks",
    }:
        raise ValueError("process affinity snapshot is malformed")
    tasks = value.get("tasks")
    if value.get("pid") != pid or value.get("expected_cpu") != cpu:
        raise ValueError("process affinity snapshot identity is invalid")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("process affinity snapshot contains no tasks")
    tids: set[int] = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"tid", "cpus"}:
            raise ValueError("process affinity task entry is malformed")
        tid = int(task.get("tid", -1))
        if tid <= 0 or tid in tids or task.get("cpus") != [cpu]:
            raise ValueError("not every process task has the exact CPU affinity")
        tids.add(tid)
    if pid not in tids:
        raise ValueError("process affinity snapshot omits the main task")


def _validate_worker_latency_summary(
    value: object,
    *,
    completed_requests: int,
    label: str,
    require_zero: bool = False,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != LATENCY_SUMMARY_KEYS:
        raise ValueError(f"worker {label} summary has an invalid schema")
    if type(value.get("count")) is not int or value.get("count") != (
        completed_requests
    ):
        raise ValueError(f"worker {label} count differs from completed requests")
    metrics = {
        key: _finite_nonnegative(value.get(key), f"worker {label} {key}")
        for key in LATENCY_SUMMARY_KEYS
        if key != "count"
    }
    ordered = [
        metrics["p50_ms"],
        metrics["p95_ms"],
        metrics["p99_ms"],
        metrics["p999_ms"],
        metrics["max_ms"],
    ]
    if any(
        current + 1e-12 < previous
        for previous, current in zip(ordered, ordered[1:])
    ) or metrics["mean_ms"] > metrics["max_ms"] + 1e-12:
        raise ValueError(f"worker {label} summary statistics are inconsistent")
    if require_zero and any(metric != 0.0 for metric in metrics.values()):
        raise ValueError(f"worker {label} must be zero without a process gate")
    return metrics


def _validate_worker_json(
    record: Mapping[str, Any],
    *,
    expected_client: Mapping[str, Any],
    expected_index: int,
    expected_cpu: int,
    expected_engine: pathlib.Path,
    expected_engine_sha256: str,
    expected_uuid: str,
    release_marker_ns: int,
    critical_end_ns: int,
    result_marker_ns: int,
) -> int:
    if set(record) != {"schema_version", "kind", "client", "result"} or (
        type(record.get("schema_version")) is not int
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind") != "p9-guard-worker-evidence"
    ):
        raise ValueError("worker evidence wrapper has invalid schema")
    client = record.get("client")
    if not isinstance(client, dict):
        raise ValueError("worker evidence lacks a client identity")
    identity = {
        **expected_client,
        "worker_index": expected_index,
        "cpu": expected_cpu,
        "engine": str(expected_engine.resolve()),
        "engine_sha256": expected_engine_sha256,
    }
    if set(client) != set(identity) | {"pid", "affinity"} or any(
        client.get(key) != value for key, value in identity.items()
    ):
        raise ValueError("worker evidence client identity is invalid")
    pid_value = client.get("pid")
    if type(pid_value) is not int or pid_value <= 0:
        raise ValueError("worker evidence PID is invalid")
    pid = pid_value
    _validate_affinity_snapshot(client.get("affinity"), pid, expected_cpu)
    result = record.get("result")
    if (
        not isinstance(result, dict)
        or set(result) != WORKER_RESULT_KEYS
        or type(result.get("schema_version")) is not int
        or result.get("schema_version") != 1
    ):
        raise ValueError("worker benchmark JSON has invalid schema")
    if result.get("role") != "pressure" or result.get("model") != expected_client["model"]:
        raise ValueError("worker benchmark role or model is invalid")
    if pathlib.Path(str(result.get("engine", ""))).resolve() != expected_engine.resolve():
        raise ValueError("worker benchmark engine is invalid")
    environment = result.get("execution_environment")
    if not isinstance(environment, dict):
        raise ValueError("worker benchmark lacks execution environment")
    if set(environment) != {
        "pid",
        "cuda_visible_devices",
        "mps_active_thread_percentage",
        "cpu_affinity",
    } or (
        int(environment.get("pid", -1)) != pid
        or environment.get("cuda_visible_devices") != expected_uuid
        or environment.get("mps_active_thread_percentage")
        != expected_client["quota_percent"]
        or environment.get("cpu_affinity") != [expected_cpu]
    ):
        raise ValueError("worker benchmark execution provenance is invalid")
    expected_multiprocessors = {
        ("resident-1g", 25): 2,
        ("resident-1g", 50): 4,
        ("resident-1g", 100): 8,
        ("borrower-2g", 100): 12,
    }[(str(expected_client["placement"]), int(expected_client["quota_percent"]))]
    expected_gpu_name = (
        "NVIDIA Thor MIG 1g.0gb"
        if expected_client["placement"] == "resident-1g"
        else "NVIDIA Thor MIG 2g.0gb"
    )
    if result.get("gpu") != {
        "name": expected_gpu_name,
        "multiprocessors": expected_multiprocessors,
    }:
        raise ValueError("worker benchmark differs from the formal MIG width")
    config = result.get("config")
    if not isinstance(config, dict) or any(
        config.get(key) != value
        for key, value in {
            "warmup": WARMUP_REQUESTS,
            "burst_size": 1,
            "period_ms": 0,
            "duration_seconds": 3600,
            "guard_ms": 0,
            "gated_processes": 0,
            "stopped_processes": 0,
            "gate_mode": "stop",
            "start_paused": True,
            "include_transfers": True,
            "priority": (
                "default"
                if expected_client["placement"] == "resident-1g"
                else "low"
            ),
            "deadline_ms": 0,
            "stream_priority_value": 0,
        }.items()
    ) or set(config) != {
        "warmup",
        "burst_size",
        "period_ms",
        "deadline_ms",
        "duration_seconds",
        "guard_ms",
        "gated_processes",
        "stopped_processes",
        "gate_mode",
        "start_paused",
        "include_transfers",
        "priority",
        "stream_priority_value",
    }:
        raise ValueError("worker benchmark configuration is invalid")
    completed_requests = result.get("completed_requests")
    if type(completed_requests) is not int or completed_requests <= 0:
        raise ValueError("worker benchmark contains no completed work")
    for label in ("release_to_completion", "gpu_service", "queue_delay"):
        _validate_worker_latency_summary(
            result.get(label),
            completed_requests=completed_requests,
            label=label,
        )
    for label in ("gate_overhead", "drain", "resume"):
        _validate_worker_latency_summary(
            result.get(label),
            completed_requests=completed_requests,
            label=label,
            require_zero=True,
        )
    start_ns = result.get("measurement_start_monotonic_ns")
    end_ns = result.get("measurement_end_monotonic_ns")
    if (
        type(start_ns) is not int
        or type(end_ns) is not int
        or start_ns < 0
        or end_ns <= start_ns
    ):
        raise ValueError("worker measurement clock interval is invalid")
    elapsed_seconds = (end_ns - start_ns) / 1_000_000_000.0
    _close(result.get("elapsed_seconds"), elapsed_seconds, "worker elapsed_seconds")
    _close(
        result.get("throughput_per_second"),
        completed_requests / elapsed_seconds,
        "worker throughput_per_second",
    )
    if (
        type(result.get("deadline_misses")) is not int
        or result.get("deadline_misses") != 0
        or result.get("deadline_miss_rate") is not None
    ):
        raise ValueError("worker deadline summary differs from deadline-disabled mode")
    if not (
        release_marker_ns
        <= start_ns
        < critical_end_ns
        <= end_ns
        <= result_marker_ns
    ):
        raise ValueError("worker measurement window is outside the block evidence chain")
    return pid


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _canonical_json(value: object) -> str | None:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return None


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
        not _finite_number(handoff_max_ms)
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


def _structurally_valid_sample(
    record: Mapping[str, Any],
    *,
    stability_sensor: str,
    safety_sensor: str,
) -> bool:
    parsed = record.get("parsed", {})
    if not isinstance(parsed, dict):
        return False
    cpu = parsed.get("cpu", [])
    power = parsed.get("power", {}).get("VIN", {})
    return (
        isinstance(parsed.get("ram"), dict)
        and _finite_number(record.get("mem_available_mb"))
        and isinstance(cpu, list)
        and any(
            isinstance(core, dict) and _finite_number(core.get("utilization_pct"))
            for core in cpu
        )
        and _finite_number(
            parsed.get("temperatures_c", {}).get(stability_sensor)
        )
        and _finite_number(parsed.get("temperatures_c", {}).get(safety_sensor))
        and isinstance(power, dict)
        and _finite_number(power.get("current_mw"))
        and not record.get("collection_errors")
    )


class TelemetryReplay:
    def __init__(self, payload: bytes) -> None:
        self.samples: list[dict[str, Any]] = []
        self.typed_samples: list[TelemetrySample] = []
        self.markers: list[dict[str, Any]] = []
        previous_timestamp: int | None = None
        for line_number, raw_line in enumerate(
            payload.splitlines(keepends=True), start=1
        ):
            if not raw_line.endswith(b"\n"):
                raise ValueError(
                    f"unterminated telemetry JSONL at line {line_number}"
                )
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid telemetry JSONL at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"telemetry JSONL record is not an object at line {line_number}"
                )
            if (
                type(record.get("schema_version")) is not int
                or record.get("schema_version") != 1
            ):
                raise ValueError(
                    f"telemetry JSONL schema version is invalid at line {line_number}"
                )
            timestamp = record.get("monotonic_ns")
            if type(timestamp) is not int or timestamp < 0:
                raise ValueError(
                    f"telemetry timestamp is invalid at line {line_number}"
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    "telemetry JSONL timestamps must be globally strictly increasing"
                )
            previous_timestamp = timestamp
            if record.get("record_type") == "sample":
                if set(record) != SAMPLE_RECORD_KEYS:
                    raise ValueError(
                        f"telemetry sample schema is invalid at line {line_number}"
                    )
                raw = record.get("raw")
                if not isinstance(raw, str):
                    raise ValueError(
                        f"telemetry sample raw line is invalid at line {line_number}"
                    )
                reparsed = parse_tegrastats_line(raw).to_dict()
                if _canonical_json(record.get("parsed")) != _canonical_json(reparsed):
                    raise ValueError(
                        "telemetry sample parsed data differs from raw line "
                        f"at line {line_number}"
                    )
                mem_available = record.get("mem_available_mb")
                if not _finite_number(mem_available) or float(mem_available) < 0.0:
                    raise ValueError(
                        f"telemetry MemAvailable is invalid at line {line_number}"
                    )
                if record.get("collection_errors") != []:
                    raise ValueError(
                        f"telemetry collection errors at line {line_number}"
                    )
                self.samples.append(record)
                self.typed_samples.append(
                    TelemetrySample(
                        monotonic_ns=int(record["monotonic_ns"]),
                        raw=raw,
                        parsed=parse_tegrastats_line(raw),
                        mem_available_mb=float(mem_available),
                        collection_errors=(),
                    )
                )
            elif record.get("record_type") == "marker":
                if set(record) != MARKER_RECORD_KEYS:
                    raise ValueError(
                        f"telemetry marker schema is invalid at line {line_number}"
                    )
                name = record.get("name")
                if not isinstance(name, str) or not name or name.isspace():
                    raise ValueError(
                        f"telemetry marker name is invalid at line {line_number}"
                    )
                if not isinstance(record.get("metadata"), dict):
                    raise ValueError(
                        f"telemetry marker metadata is invalid at line {line_number}"
                    )
                self.markers.append(record)
            else:
                raise ValueError(
                    f"telemetry record type is invalid at line {line_number}"
                )
        if not self.samples:
            raise ValueError("telemetry JSONL contains no samples")
        self.sample_timestamps = [
            int(sample["monotonic_ns"]) for sample in self.samples
        ]

    def marker(
        self,
        name: str,
        *,
        case_id: str | None = None,
        block: int | None = None,
        label: str | None = None,
        attempt: int | None = None,
        thermal_attempt: int | None = None,
    ) -> dict[str, Any]:
        matches = []
        for marker in self.markers:
            if marker["name"] != name:
                continue
            metadata = marker["metadata"]
            if case_id is not None and metadata.get("case_id") != case_id:
                continue
            if block is not None and metadata.get("block") != block:
                continue
            if label is not None and metadata.get("label") != label:
                continue
            if attempt is not None and metadata.get("attempt") != attempt:
                continue
            if (
                thermal_attempt is not None
                and metadata.get("thermal_attempt") != thermal_attempt
            ):
                continue
            matches.append(marker)
        if len(matches) != 1:
            raise ValueError(
                "expected exactly one telemetry marker "
                f"{name}/{case_id}/{block}/{label}/{attempt}/{thermal_attempt}"
            )
        return matches[0]

    def _healthy_interval(
        self,
        start_ns: int,
        end_ns: int,
        *,
        interval_ms: float,
        required_fraction: float,
        hard_limit_c: float,
        maximum_gap_ms: float,
        stability_sensor: str,
        safety_sensor: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if end_ns <= start_ns:
            raise ValueError("telemetry interval is not positive")
        selected = [
            sample
            for sample in self.samples
            if start_ns <= int(sample["monotonic_ns"]) < end_ns
        ]
        duration_seconds = (end_ns - start_ns) / 1_000_000_000.0
        expected = math.floor(duration_seconds * 1000.0 / interval_ms)
        required = max(1, math.floor(expected * required_fraction))
        valid = [
            sample
            for sample in selected
            if _structurally_valid_sample(
                sample,
                stability_sensor=stability_sensor,
                safety_sensor=safety_sensor,
            )
        ]
        if len(valid) < required:
            raise ValueError("telemetry interval lacks structurally valid coverage")
        timestamps = [int(sample["monotonic_ns"]) for sample in valid]
        if (
            timestamps[0] - start_ns > maximum_gap_ms * 1_000_000.0
            or end_ns - timestamps[-1] > maximum_gap_ms * 1_000_000.0
            or any(
                current - previous > maximum_gap_ms * 1_000_000.0
                for previous, current in zip(timestamps, timestamps[1:])
            )
        ):
            raise ValueError("telemetry interval has a gap larger than 300 ms")
        if any(
            float(sample["parsed"]["temperatures_c"][safety_sensor])
            >= hard_limit_c
            for sample in selected
            if _finite_number(
                sample.get("parsed", {})
                .get("temperatures_c", {})
                .get(safety_sensor)
            )
        ):
            raise ValueError("telemetry reached the frozen thermal hard limit")
        return selected, valid

    def _thermal_window_evidence(
        self,
        reference_ns: int,
        *,
        not_before_ns: int,
        thermal_lock: Mapping[str, Any],
        label: str = "thermal qualification",
    ) -> tuple[dict[str, Any] | None, dict[str, Any], bool, str | None]:
        _validate_guard_thermal_lock(thermal_lock)
        stability_sensor = str(thermal_lock["stability_sensor"])
        safety_sensor = str(thermal_lock["safety_sensor"])
        window_seconds = _finite_positive(
            thermal_lock.get("stability_window_seconds"), "thermal window"
        )
        interval_ms = _finite_positive(
            thermal_lock.get("telemetry_interval_ms"), "telemetry cadence"
        )
        required_fraction = _finite_positive(
            thermal_lock.get("telemetry_required_fraction"),
            "telemetry required fraction",
        )
        if required_fraction > 1.0:
            raise ValueError("telemetry required fraction exceeds one")
        hard_limit = _finite_positive(thermal_lock.get("hard_limit_c"), "hard limit")
        maximum_gap_ms = _finite_positive(
            thermal_lock.get("telemetry_max_gap_ms"), "telemetry maximum gap"
        )
        if (
            type(reference_ns) is not int
            or type(not_before_ns) is not int
            or reference_ns <= not_before_ns
        ):
            raise ValueError("thermal qualification clocks are invalid")
        start_ns = max(
            not_before_ns,
            reference_ns - int(window_seconds * 1_000_000_000),
        )
        left = bisect.bisect_left(self.sample_timestamps, start_ns)
        right = bisect.bisect_right(self.sample_timestamps, reference_ns)
        selected = self.samples[left:right]
        points = [
            (
                int(sample["monotonic_ns"]),
                float(sample["parsed"]["temperatures_c"][stability_sensor]),
            )
            for sample in selected
            if _finite_number(
                sample.get("parsed", {})
                .get("temperatures_c", {})
                .get(stability_sensor)
            )
        ]
        replayed: dict[str, Any] | None = None
        if len(points) >= 2:
            times = [(timestamp - points[0][0]) / 1e9 for timestamp, _ in points]
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
            timestamps = [timestamp for timestamp, _ in points]
            maximum_gap_seconds = max(
                [
                    timestamps[0] - start_ns,
                    reference_ns - timestamps[-1],
                    *(
                        current - previous
                        for previous, current in zip(
                            timestamps, timestamps[1:], strict=False
                        )
                    ),
                ]
            ) / 1_000_000_000.0
            replayed = {
                "samples": len(values),
                "window_seconds": window_seconds,
                "observed_span_seconds": times[-1] - times[0],
                "mean_c": mean_value,
                "min_c": min(values),
                "max_c": max(values),
                "latest_c": values[-1],
                "slope_c_per_minute": slope,
                "maximum_gap_seconds": maximum_gap_seconds,
            }
        duration_ns = reference_ns - start_ns
        expected_samples = max(
            1, math.floor(duration_ns / (interval_ms * 1_000_000.0))
        )
        minimum_samples = max(1, math.floor(expected_samples * required_fraction))
        maximum_gap_ns = int(maximum_gap_ms * 1_000_000.0)
        aggregate_inputs = list(self.typed_samples[left:right])
        if right:
            latest = self.typed_samples[right - 1]
            if latest not in aggregate_inputs:
                aggregate_inputs.append(latest)
        for index in range(right - 1, -1, -1):
            if _structurally_valid_sample(
                self.samples[index],
                stability_sensor=stability_sensor,
                safety_sensor=safety_sensor,
            ):
                latest_valid = self.typed_samples[index]
                if latest_valid not in aggregate_inputs:
                    aggregate_inputs.append(latest_valid)
                break
        telemetry = aggregate_samples(
            aggregate_inputs,
            start_ns,
            reference_ns,
            required_fields=TELEMETRY_REQUIRED_FIELDS,
            minimum_valid_samples=minimum_samples,
            reference_ns=reference_ns,
            stale_after_ns=maximum_gap_ns,
            maximum_valid_gap_ns=maximum_gap_ns,
            end_inclusive=True,
        )
        telemetry["retention"] = {
            "bounded": False,
            "max_samples": None,
            "dropped_samples": 0,
            "last_dropped_sample_ns": None,
            "earliest_retained_sample_ns": self.sample_timestamps[0],
            "interval_complete": True,
        }
        target = _finite_positive(thermal_lock.get("target_c"), "thermal target")
        tolerance = _finite_nonnegative(
            thermal_lock.get("tolerance_c"), "thermal tolerance"
        )
        maximum_slope = _finite_nonnegative(
            thermal_lock.get("maximum_slope_c_per_minute"),
            "thermal maximum slope",
        )
        safety = telemetry.get("temperatures_c", {}).get(safety_sensor)
        failure_reason: str | None = None
        if replayed is None:
            failure_reason = f"{label} thermal start has no stability window"
        elif telemetry.get("health", {}).get("healthy") is not True:
            failure_reason = (
                f"{label} thermal start telemetry is unhealthy: "
                f"{telemetry['health']}"
            )
        elif not isinstance(safety, dict):
            failure_reason = f"{label} thermal start telemetry lacks {safety_sensor}"
        elif float(safety["max"]) >= hard_limit:
            failure_reason = (
                f"thermal hard limit reached during {label}: {safety}"
            )
        elif not (
            replayed["samples"] >= math.floor(window_seconds * 1000.0 / interval_ms * required_fraction)
            and replayed["observed_span_seconds"] >= window_seconds * 0.99
            and abs(float(replayed["mean_c"]) - target) <= tolerance
            and abs(float(replayed["latest_c"]) - target) <= tolerance
            and abs(float(replayed["slope_c_per_minute"])) <= maximum_slope
            and float(replayed["maximum_gap_seconds"])
            <= maximum_gap_ms / 1000.0
        ):
            failure_reason = f"{label} thermal start is unstable: {replayed}"
        return replayed, telemetry, failure_reason is None, failure_reason

    @staticmethod
    def _compare_thermal_window(stored: object, replayed: object) -> None:
        if replayed is None:
            if stored is not None:
                raise ValueError("thermal window differs from raw telemetry")
            return
        if not isinstance(stored, dict) or set(stored) != set(replayed):
            raise ValueError("thermal window has an invalid schema")
        for name, value in replayed.items():
            if name == "samples":
                if stored.get(name) != value:
                    raise ValueError("thermal window sample count differs from telemetry")
            else:
                _close(stored.get(name), value, f"thermal window {name}")

    def thermal_start(
        self,
        reference_ns: int,
        *,
        thermal_lock: Mapping[str, Any],
        stored: object,
        stored_stable: object,
        not_before_ns: int | None = None,
    ) -> dict[str, Any]:
        if stored_stable is not True:
            raise ValueError("block lacks a stable thermal-start summary")
        replayed, _telemetry, passed, _failure = self._thermal_window_evidence(
            reference_ns,
            not_before_ns=(
                reference_ns
                - int(
                    _finite_positive(
                        thermal_lock.get("stability_window_seconds"),
                        "thermal window",
                    )
                    * 1_000_000_000
                )
                if not_before_ns is None
                else not_before_ns
            ),
            thermal_lock=thermal_lock,
        )
        self._compare_thermal_window(stored, replayed)
        if not passed or replayed is None:
            if (
                replayed is not None
                and float(replayed["maximum_gap_seconds"])
                > _finite_positive(
                    thermal_lock.get("telemetry_max_gap_ms"),
                    "telemetry maximum gap",
                )
                / 1000.0
            ):
                raise ValueError("telemetry interval has a gap larger than 300 ms")
            if _failure is not None and "thermal hard limit" in _failure:
                raise ValueError(_failure)
            raise ValueError("thermal-start window violates the frozen precondition")
        return replayed

    def _causal_sample(
        self,
        *,
        reference_ns: int,
        not_before_ns: int,
        first_after_boundary: bool,
    ) -> dict[str, Any] | None:
        if reference_ns <= not_before_ns:
            raise ValueError("causal thermal qualification clocks are invalid")
        if first_after_boundary:
            index = bisect.bisect_right(self.sample_timestamps, not_before_ns)
            if (
                index >= len(self.samples)
                or int(self.samples[index]["monotonic_ns"]) > reference_ns
            ):
                return None
            return self.samples[index]
        index = bisect.bisect_right(self.sample_timestamps, reference_ns) - 1
        if index < 0 or int(self.samples[index]["monotonic_ns"]) <= not_before_ns:
            return None
        return self.samples[index]

    def causal_qualification(
        self,
        stored: object,
        *,
        label: str,
        reference_ns: int,
        not_before_ns: int,
        first_after_boundary: bool,
        thermal_lock: Mapping[str, Any],
        expected_prefix: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(label, str) or not label:
            raise ValueError("causal thermal qualification label is invalid")
        expected_keys = set(expected_prefix) | {
            "passed",
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
        if not isinstance(stored, dict) or set(stored) != expected_keys:
            raise ValueError("causal thermal qualification has an invalid schema")
        if any(stored.get(key) != value for key, value in expected_prefix.items()):
            raise ValueError("causal thermal qualification identity is invalid")
        sample = self._causal_sample(
            reference_ns=reference_ns,
            not_before_ns=not_before_ns,
            first_after_boundary=first_after_boundary,
        )
        stability_sensor = str(thermal_lock["stability_sensor"])
        safety_sensor = str(thermal_lock["safety_sensor"])
        target_c = _finite_positive(thermal_lock.get("target_c"), "thermal target")
        tolerance_c = _finite_nonnegative(
            thermal_lock.get("tolerance_c"), "thermal tolerance"
        )
        hard_limit_c = _finite_positive(
            thermal_lock.get("hard_limit_c"), "thermal hard limit"
        )
        maximum_age_ms = _finite_positive(
            thermal_lock.get("telemetry_max_gap_ms"), "telemetry maximum gap"
        )
        sample_ns: int | None = None
        stability_value: float | None = None
        safety_value: float | None = None
        sample_age_ms: float | None = None
        if sample is not None:
            sample_ns = int(sample["monotonic_ns"])
            sample_age_ms = (reference_ns - sample_ns) / 1_000_000.0
            temperatures = sample.get("parsed", {}).get("temperatures_c", {})
            if _finite_number(temperatures.get(stability_sensor)):
                stability_value = float(temperatures[stability_sensor])
            if _finite_number(temperatures.get(safety_sensor)):
                safety_value = float(temperatures[safety_sensor])
        if stored.get("stability_sensor") != stability_sensor or stored.get(
            "safety_sensor"
        ) != safety_sensor:
            raise ValueError("causal qualification sensor binding is invalid")
        _close(stored.get("target_c"), target_c, "causal qualification target")
        _close(
            stored.get("tolerance_c"),
            tolerance_c,
            "causal qualification tolerance",
        )
        if stored.get("sample_monotonic_ns") != sample_ns:
            raise ValueError("causal qualification did not select the raw sample")
        if sample_age_ms is None:
            if stored.get("sample_age_ms") is not None:
                raise ValueError("causal qualification sample age is invalid")
        else:
            _close(
                stored.get("sample_age_ms"), sample_age_ms, "causal sample age"
            )
        for key, value in (
            ("stability_value_c", stability_value),
            ("safety_value_c", safety_value),
        ):
            if value is None:
                if stored.get(key) is not None:
                    raise ValueError(f"causal qualification {key} is invalid")
            else:
                _close(stored.get(key), value, f"causal qualification {key}")
        telemetry = stored.get("telemetry")
        if telemetry is not None and not isinstance(telemetry, dict):
            raise ValueError("causal qualification telemetry is invalid")
        if sample_ns is None:
            replayed_telemetry = None
            raw_health = False
        else:
            sample_index = bisect.bisect_left(self.sample_timestamps, sample_ns)
            maximum_gap_ns = int(maximum_age_ms * 1_000_000.0)
            replayed_telemetry = aggregate_samples(
                [self.typed_samples[sample_index]],
                sample_ns - 1,
                sample_ns,
                required_fields=TELEMETRY_REQUIRED_FIELDS,
                minimum_valid_samples=1,
                require_all_samples_valid=True,
                reference_ns=reference_ns,
                stale_after_ns=maximum_gap_ns,
                maximum_valid_gap_ns=maximum_gap_ns,
                end_inclusive=True,
            )
            replayed_telemetry["retention"] = {
                "bounded": False,
                "max_samples": None,
                "dropped_samples": 0,
                "last_dropped_sample_ns": None,
                "earliest_retained_sample_ns": self.sample_timestamps[0],
                "interval_complete": True,
            }
            raw_health = replayed_telemetry["health"].get("healthy") is True
        if _canonical_json(telemetry) != _canonical_json(replayed_telemetry):
            raise ValueError("causal qualification telemetry differs from raw JSONL")
        if sample is None:
            expected_failure = (
                f"{label} observed no causal post-cleanup telemetry sample"
                if first_after_boundary
                else f"{label} has no causal actual-start telemetry sample"
            )
        elif not raw_health:
            phase = "causal" if first_after_boundary else "actual-start"
            assert replayed_telemetry is not None
            expected_failure = (
                f"{label} {phase} telemetry is unhealthy: "
                f"{replayed_telemetry['health']}"
            )
        elif (
            first_after_boundary
            and sample_ns is not None
            and (sample_ns - not_before_ns) / 1_000_000.0 > maximum_age_ms
        ):
            expected_failure = f"{label} causal telemetry arrived too late"
        elif sample_age_ms is None or sample_age_ms > maximum_age_ms:
            expected_failure = (
                f"{label} causal telemetry is stale"
                if first_after_boundary
                else f"{label} actual-start telemetry is stale"
            )
        elif (
            stability_value is None
            or abs(stability_value - target_c) > tolerance_c
        ):
            expected_failure = (
                f"{label} stability sensor is outside the target band"
                if first_after_boundary
                else f"{label} actual-start stability is outside the target band"
            )
        elif safety_value is None or safety_value >= hard_limit_c:
            expected_failure = (
                f"{label} thermal hard limit reached"
                if first_after_boundary
                else f"{label} actual-start thermal hard limit reached"
            )
        else:
            expected_failure = None
        passed = expected_failure is None
        if type(stored.get("passed")) is not bool or stored["passed"] != passed:
            raise ValueError("causal qualification outcome differs from raw telemetry")
        if stored.get("failure_reason") != expected_failure:
            raise ValueError(
                "causal qualification failure reason differs from raw telemetry"
            )
        return dict(stored)

    def measurement_health(
        self,
        start_ns: int,
        end_ns: int,
        thermal_lock: Mapping[str, Any],
    ) -> dict[str, Any]:
        _validate_guard_thermal_lock(thermal_lock)
        stability_sensor = str(thermal_lock["stability_sensor"])
        safety_sensor = str(thermal_lock["safety_sensor"])
        selected, valid = self._healthy_interval(
            start_ns,
            end_ns,
            interval_ms=_finite_positive(
                thermal_lock.get("telemetry_interval_ms"), "telemetry cadence"
            ),
            required_fraction=_finite_positive(
                thermal_lock.get("telemetry_required_fraction"),
                "telemetry required fraction",
            ),
            hard_limit_c=_finite_positive(
                thermal_lock.get("hard_limit_c"), "thermal hard limit"
            ),
            maximum_gap_ms=_finite_positive(
                thermal_lock.get("telemetry_max_gap_ms"),
                "telemetry maximum gap",
            ),
            stability_sensor=stability_sensor,
            safety_sensor=safety_sensor,
        )
        safety_values = [
            float(sample["parsed"]["temperatures_c"][safety_sensor])
            for sample in selected
            if _finite_number(
                sample.get("parsed", {})
                .get("temperatures_c", {})
                .get(safety_sensor)
            )
        ]
        return {
            "healthy": True,
            "total_samples": len(selected),
            "valid_samples": len(valid),
            "safety_sensor": safety_sensor,
            "safety_max_c": max(safety_values),
        }


def _validate_thermal_precondition(
    telemetry: TelemetryReplay,
    value: object,
    *,
    label: str,
    thermal_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    expected_keys = {
        "label",
        "duration_seconds",
        "measurement_start_monotonic_ns",
        "measurement_end_monotonic_ns",
        "cleanup_end_monotonic_ns",
        "target_c",
        "stability_sensor",
        "safety_sensor",
        "last_window",
        "pressure_rate_per_second",
        "telemetry",
        "active_stability_checks",
        "active_stable_endpoints",
        "active_stable_spacing_seconds",
        "termination_reason",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("guard attempt lacks active thermal-precondition evidence")
    if value.get("label") != label:
        raise ValueError("thermal-precondition label is invalid")
    base_markers: dict[str, dict[str, Any]] = {}
    for name in ("thermal_prepare", "thermal_start", "thermal_measurement_end", "thermal_end"):
        matches = [
            marker
            for marker in telemetry.markers
            if marker["name"] == name and marker["metadata"].get("label") == label
        ]
        if len(matches) != 1:
            raise ValueError(f"thermal preheater marker is missing or duplicated: {name}")
        base_markers[name] = matches[0]
    prepare = base_markers["thermal_prepare"]
    start = base_markers["thermal_start"]
    measurement_end = base_markers["thermal_measurement_end"]
    end = base_markers["thermal_end"]
    if prepare["metadata"] != {"label": label}:
        raise ValueError("thermal_prepare metadata is invalid")
    if start["metadata"] != {"label": label}:
        raise ValueError("thermal_start metadata is invalid")
    if end["metadata"] != {"label": label, "successful": True}:
        raise ValueError("thermal_end does not prove a successful preheater")
    check_markers = sorted(
        (
            marker
            for marker in telemetry.markers
            if marker["name"] == "thermal_active_stability_check"
            and marker["metadata"].get("label") == label
        ),
        key=_marker_timestamp,
    )
    if len(check_markers) < THERMAL_ACTIVE_STABLE_ENDPOINTS:
        raise ValueError("thermal preheater lacks three active stable endpoints")
    start_ns = _marker_timestamp(start)
    replayed_checks: list[dict[str, Any]] = []
    consecutive = 0
    last_success_sample_ns: int | None = None
    for index, marker in enumerate(check_markers):
        metadata = marker.get("metadata")
        if not isinstance(metadata, dict) or set(metadata) != {
            "label",
            "index",
            "sample_monotonic_ns",
            "passed",
            "consecutive_passes",
            "window",
        }:
            raise ValueError("active thermal stability marker has an invalid schema")
        if metadata.get("label") != label or metadata.get("index") != index:
            raise ValueError("active thermal stability markers are not consecutive")
        sample_ns = metadata.get("sample_monotonic_ns")
        if (
            type(sample_ns) is not int
            or sample_ns <= start_ns
            or sample_ns > _marker_timestamp(marker)
        ):
            raise ValueError("active thermal endpoint has an invalid causal sample")
        replayed_window, _aggregate, passed, _failure = (
            telemetry._thermal_window_evidence(
                sample_ns,
                not_before_ns=start_ns,
                thermal_lock=thermal_lock,
                label=label,
            )
        )
        telemetry._compare_thermal_window(metadata.get("window"), replayed_window)
        if type(metadata.get("passed")) is not bool or metadata["passed"] != passed:
            raise ValueError("active thermal endpoint outcome differs from raw telemetry")
        consecutive = consecutive + 1 if passed else 0
        if metadata.get("consecutive_passes") != consecutive:
            raise ValueError("active thermal endpoint consecutive count is invalid")
        if passed and last_success_sample_ns is not None and (
            sample_ns - last_success_sample_ns
            < int(THERMAL_ACTIVE_STABLE_SPACING_SECONDS * 1_000_000_000)
        ):
            raise ValueError("active stable endpoints are not independently spaced")
        last_success_sample_ns = sample_ns if passed else None
        replayed_checks.append(dict(metadata))
    if not all(
        check["passed"] is True
        for check in replayed_checks[-THERMAL_ACTIVE_STABLE_ENDPOINTS:]
    ) or replayed_checks[-1]["consecutive_passes"] != THERMAL_ACTIVE_STABLE_ENDPOINTS:
        raise ValueError("thermal preheater did not end at three active stable endpoints")
    boundary_metadata = measurement_end.get("metadata")
    final_check = replayed_checks[-1]
    expected_boundary_metadata = {
        "label": label,
        "boundary_sample_monotonic_ns": final_check["sample_monotonic_ns"],
        "consecutive_passes": THERMAL_ACTIVE_STABLE_ENDPOINTS,
        "window": final_check["window"],
    }
    if boundary_metadata != expected_boundary_metadata:
        raise ValueError("thermal_measurement_end does not bind the final active endpoint")
    times = [
        _marker_timestamp(prepare),
        start_ns,
        *(_marker_timestamp(marker) for marker in check_markers),
        _marker_timestamp(measurement_end),
        _marker_timestamp(end),
    ]
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError("thermal preheater marker chain is not strictly ordered")
    measurement_end_ns = _marker_timestamp(measurement_end)
    if (
        value.get("measurement_start_monotonic_ns") != start_ns
        or value.get("measurement_end_monotonic_ns") != measurement_end_ns
        or value.get("cleanup_end_monotonic_ns") != _marker_timestamp(end)
    ):
        raise ValueError("thermal-precondition clocks differ from raw markers")
    if (
        value.get("stability_sensor") != thermal_lock.get("stability_sensor")
        or value.get("safety_sensor") != thermal_lock.get("safety_sensor")
    ):
        raise ValueError("thermal-precondition sensor binding is invalid")
    _close(
        value.get("duration_seconds"),
        (measurement_end_ns - start_ns) / 1e9,
        "thermal-precondition duration",
    )
    _close(
        value.get("target_c"),
        thermal_lock.get("target_c"),
        "thermal-precondition target",
    )
    telemetry._compare_thermal_window(value.get("last_window"), final_check["window"])
    stored_checks = value.get("active_stability_checks")
    if stored_checks != replayed_checks:
        raise ValueError("thermal precondition active checks differ from raw markers")
    if (
        value.get("active_stable_endpoints") != THERMAL_ACTIVE_STABLE_ENDPOINTS
        or value.get("active_stable_spacing_seconds")
        != THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        or value.get("termination_reason") != "active-stability-endpoints"
    ):
        raise ValueError("thermal precondition active endpoint protocol is invalid")
    replayed_health = telemetry.measurement_health(
        start_ns, measurement_end_ns, thermal_lock
    )
    stored_health = value.get("telemetry")
    if not isinstance(stored_health, dict):
        raise ValueError("thermal preheater lacks a telemetry aggregate")
    temperatures = stored_health.get("temperatures_c", {})
    safety_sensor = str(thermal_lock["safety_sensor"])
    safety = (
        temperatures.get(safety_sensor, {})
        if isinstance(temperatures, dict)
        else {}
    )
    if (
        stored_health.get("total_samples") != replayed_health["total_samples"]
        or stored_health.get("valid_samples") != replayed_health["valid_samples"]
        or not isinstance(stored_health.get("health"), dict)
        or stored_health["health"].get("healthy") is not True
        or stored_health["health"].get("required_fields")
        != list(TELEMETRY_REQUIRED_FIELDS)
    ):
        raise ValueError("thermal preheater telemetry aggregate differs from raw JSONL")
    _close(
        safety.get("max"),
        replayed_health["safety_max_c"],
        "thermal preheater safety maximum",
    )
    _finite_positive(value.get("pressure_rate_per_second"), "preheater pressure rate")
    return {
        "label": label,
        "measurement_start_monotonic_ns": start_ns,
        "measurement_end_monotonic_ns": measurement_end_ns,
        "cleanup_end_monotonic_ns": _marker_timestamp(end),
        "stability_sensor": thermal_lock["stability_sensor"],
        "safety_sensor": thermal_lock["safety_sensor"],
        "last_window": final_check["window"],
        "active_stability_checks": replayed_checks,
        "active_stable_endpoints": THERMAL_ACTIVE_STABLE_ENDPOINTS,
        "active_stable_spacing_seconds": THERMAL_ACTIVE_STABLE_SPACING_SECONDS,
        "measurement_telemetry": replayed_health,
    }, times


def _artifact_keys() -> set[str]:
    result = {"benchmark", *CURRENT_FILES}
    for case in expected_single_cases():
        result.add(_engine_artifact_key(case["clients"][0]))
    result.add("engine:critical:2g:resnet50-v2")
    return result


def validate_artifacts(artifacts: object) -> dict[str, dict[str, str]]:
    if not isinstance(artifacts, dict) or set(artifacts) != _artifact_keys():
        raise ValueError("guard profile artifact manifest is incomplete")
    validated: dict[str, dict[str, str]] = {}
    for name, raw_record in sorted(artifacts.items()):
        if not isinstance(raw_record, dict) or set(raw_record) != {"path", "sha256"}:
            raise ValueError(f"artifact record is malformed: {name}")
        path = pathlib.Path(str(raw_record["path"]))
        digest = raw_record["sha256"]
        if not path.is_absolute() or not path.is_file() or not _is_sha256(digest):
            raise ValueError(f"artifact is missing or invalid: {name}")
        if file_sha256(path) != digest:
            raise ValueError(f"artifact changed after profiling: {name}")
        if name in CURRENT_FILES and path.resolve() != CURRENT_FILES[name].resolve():
            raise ValueError(f"implementation artifact path is invalid: {name}")
        validated[name] = {"path": str(path.resolve()), "sha256": str(digest)}
    critical_path = pathlib.Path(
        validated["engine:critical:2g:resnet50-v2"]["path"]
    )
    engine_root = critical_path.parent.parent
    if critical_path != engine_root / "mig-2g" / "resnet50-v2.engine":
        raise ValueError("critical engine does not use the formal mig-2g tag")
    for case in expected_single_cases():
        client = case["clients"][0]
        key = _engine_artifact_key(client)
        prefix = "mig-1g" if client["placement"] == "resident-1g" else "mig-2g"
        expected = (
            engine_root
            / f"{prefix}-q{client['quota_percent']}"
            / f"{client['model']}.engine"
        )
        if pathlib.Path(validated[key]["path"]) != expected:
            raise ValueError(f"engine artifact does not use the formal tag: {key}")
    return validated


def _validate_profile_provenance(
    summary: Mapping[str, Any],
    thermal_lock: Mapping[str, Any],
    root: pathlib.Path,
) -> tuple[dict[str, dict[str, str]], str, str]:
    _validate_guard_thermal_lock(thermal_lock)
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("kind") != PROFILE_KIND:
        raise ValueError("guard profile has an invalid schema or kind")
    if summary.get("protocol") != expected_protocol():
        raise ValueError("guard profile differs from the fixed P9 protocol")
    if summary.get("cpu_affinity") != CPU_AFFINITY:
        raise ValueError("guard profile CPU affinity is invalid")
    if summary.get("producer_cpu_affinity") != CPU_AFFINITY["telemetry"]:
        raise ValueError("guard producer was not pinned to the telemetry CPU")
    hardware = summary.get("hardware")
    if not isinstance(hardware, dict) or hardware != thermal_lock.get("pilot_hardware"):
        raise ValueError("guard profile hardware differs from the thermal lock")
    snapshot_hashes = summary.get("hardware_snapshot_sha256")
    if not isinstance(snapshot_hashes, dict) or set(snapshot_hashes) != set(
        HARDWARE_SNAPSHOTS
    ):
        raise ValueError("guard profile hardware snapshot hashes are incomplete")
    for name in HARDWARE_SNAPSHOTS:
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or not _is_sha256(snapshot_hashes.get(name))
            or file_sha256(path) != snapshot_hashes[name]
        ):
            raise ValueError(f"hardware snapshot changed or is missing: {name}")
    mig = summary.get("mig")
    if not isinstance(mig, dict):
        raise ValueError("guard profile lacks MIG provenance")
    small_uuid = str(mig.get("small_uuid", ""))
    big_uuid = str(mig.get("big_uuid", ""))
    if thermal_lock.get("pilot_mig") != {
        "critical_uuid": big_uuid,
        "resident_uuid": small_uuid,
    }:
        raise ValueError("guard profile MIG mapping differs from the thermal lock")
    env_path = pathlib.Path(str(mig.get("env_path", "")))
    if (
        not env_path.is_absolute()
        or not env_path.is_file()
        or file_sha256(env_path) != mig.get("env_sha256")
    ):
        raise ValueError("guard profile MIG environment changed or is missing")
    env_values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError("MIG environment contains a malformed assignment")
            env_values[key] = value
    mps = summary.get("mps")
    if not isinstance(mps, dict) or set(mps) != {
        "resident_pipe",
        "resident_log",
        "big_pipe",
        "big_log",
    }:
        raise ValueError("guard profile lacks exact MPS directory provenance")
    expected_resident = {
        "resident_pipe": str(
            pathlib.Path(env_values.get("JDG_MPS_PIPE_DIRECTORY", "")).resolve()
        ),
        "resident_log": str(
            pathlib.Path(env_values.get("JDG_MPS_LOG_DIRECTORY", "")).resolve()
        ),
    }
    if any(mps.get(key) != value for key, value in expected_resident.items()):
        raise ValueError("resident MPS directories differ from the MIG environment")
    directory_values = [pathlib.Path(str(value)) for value in mps.values()]
    if any(not path.is_absolute() for path in directory_values) or len(
        {str(path) for path in directory_values}
    ) != len(directory_values):
        raise ValueError("MPS directories must be absolute and placement-distinct")
    expected_mps_affinity = [
        {"placement": placement, "role": role, "cpus": CPU_AFFINITY["mps"]}
        for placement in ("critical-2g", "resident-1g")
        for role in ("control", "server")
    ]
    if hardware.get("mps_affinity") != expected_mps_affinity:
        raise ValueError("MPS control/server tasks are not pinned to the formal CPU")
    if thermal_lock.get("pilot_cpu_affinity") != CPU_AFFINITY:
        raise ValueError("thermal lock CPU affinity differs from guard calibration")
    return validate_artifacts(summary.get("artifacts")), small_uuid, big_uuid


def _marker_timestamp(marker: Mapping[str, Any]) -> int:
    return int(marker["monotonic_ns"])


def _validate_marker_metadata(
    marker: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    metadata = marker.get("metadata")
    if not isinstance(metadata, dict) or metadata != expected:
        raise ValueError(f"telemetry marker metadata is invalid: {marker.get('name')}")


def _replay_thermal_handoff(
    stored: object,
    *,
    qualification: Mapping[str, Any],
    qualification_result_ns: int,
    start_ns: int,
    release_ns: int,
    resume_issued_ns: int,
    measurement_start_ns: int,
    thermal_lock: Mapping[str, Any],
) -> dict[str, Any]:
    clock_values = {
        "cleanup_end": int(qualification["cleanup_end_monotonic_ns"]),
        "sample": int(qualification["sample_monotonic_ns"]),
        "qualification": int(qualification["qualification_monotonic_ns"]),
        "qualification_result": qualification_result_ns,
        "block_start": start_ns,
        "measurement_release": release_ns,
        "resume_issued": resume_issued_ns,
        "critical_measurement_start": measurement_start_ns,
    }
    expected_keys = {
        "boundary",
        "boundary_monotonic_ns",
        "maximum_ms",
        "strictly_within_bound",
    } | {
        key
        for label in clock_values
        for key in (f"{label}_monotonic_ns", f"boundary_to_{label}_ms")
    }
    if not isinstance(stored, dict) or set(stored) != expected_keys:
        raise ValueError("thermal handoff evidence has an invalid schema")
    boundary_ns = int(qualification["boundary_monotonic_ns"])
    maximum_ms = _finite_positive(
        thermal_lock.get("thermal_handoff_max_ms"), "thermal handoff maximum"
    )
    if (
        stored.get("boundary") != THERMAL_HANDOFF_BOUNDARY
        or stored.get("boundary_monotonic_ns") != boundary_ns
    ):
        raise ValueError("thermal handoff boundary differs from raw telemetry")
    _close(stored.get("maximum_ms"), maximum_ms, "thermal handoff maximum")
    replayed: dict[str, Any] = {
        "boundary": THERMAL_HANDOFF_BOUNDARY,
        "boundary_monotonic_ns": boundary_ns,
        "maximum_ms": maximum_ms,
    }
    strictly_within_bound = True
    previous_ns = boundary_ns
    for label, observed_ns in clock_values.items():
        elapsed_ms = (observed_ns - boundary_ns) / 1_000_000.0
        if observed_ns <= boundary_ns or observed_ns <= previous_ns or elapsed_ms >= maximum_ms:
            strictly_within_bound = False
        clock_field = f"{label}_monotonic_ns"
        elapsed_field = f"boundary_to_{label}_ms"
        if stored.get(clock_field) != observed_ns:
            raise ValueError(f"thermal handoff {clock_field} differs from raw evidence")
        _close(stored.get(elapsed_field), elapsed_ms, f"thermal handoff {elapsed_field}")
        replayed[clock_field] = observed_ns
        replayed[elapsed_field] = elapsed_ms
        previous_ns = observed_ns
    if stored.get("strictly_within_bound") is not strictly_within_bound:
        raise ValueError("thermal handoff validity differs from raw clocks")
    replayed["strictly_within_bound"] = strictly_within_bound
    return replayed


def _replay_qualification_attempts(
    telemetry: TelemetryReplay,
    block_attempt_record: Mapping[str, Any],
    *,
    base_metadata: Mapping[str, Any],
    worker_pids: Sequence[int],
    critical_pid: int,
    thermal_lock: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[int]]:
    block_attempt = int(block_attempt_record["attempt"])
    raw_attempts = block_attempt_record.get("thermal_attempts")
    if (
        not isinstance(raw_attempts, list)
        or not 1 <= len(raw_attempts) <= THERMAL_QUALIFICATION_MAX_ATTEMPTS
    ):
        raise ValueError("guard block-attempt has an invalid thermal qualification count")
    expected_states = {
        str(pid): "T" for pid in [*worker_pids, critical_pid]
    }
    replayed_attempts: list[dict[str, Any]] = []
    all_times: list[int] = []
    for attempt, raw_attempt in enumerate(raw_attempts, start=1):
        if not isinstance(raw_attempt, dict) or set(raw_attempt) != {
            "attempt",
            "label",
            "thermal_precondition",
            "qualification",
            "qualification_result_marker_monotonic_ns",
            "measured_process_states",
            "start_marker_monotonic_ns",
            "measurement_release_marker_monotonic_ns",
            "pre_release_passed",
            "failure_reason",
        }:
            raise ValueError("thermal qualification attempt has an invalid schema")
        if raw_attempt.get("attempt") != attempt:
            raise ValueError("thermal qualification attempts are not consecutive")
        label = (
            f"pre-p9-guard-{base_metadata['case_id']}-"
            f"block-{int(base_metadata['block']):02d}"
            f"-attempt-{block_attempt:02d}-thermal-{attempt:02d}"
        )
        if raw_attempt.get("label") != label:
            raise ValueError("thermal qualification label is invalid")
        precondition, precondition_times = _validate_thermal_precondition(
            telemetry,
            raw_attempt.get("thermal_precondition"),
            label=label,
            thermal_lock=thermal_lock,
        )
        qualification_marker = telemetry.marker(
            "thermal_start_qualification",
            label=label,
            attempt=attempt,
        )
        result_marker = telemetry.marker(
            "thermal_start_qualification_result",
            case_id=str(base_metadata["case_id"]),
            block=int(base_metadata["block"]),
            attempt=block_attempt,
            label=label,
            thermal_attempt=attempt,
        )
        boundary_ns = int(precondition["measurement_end_monotonic_ns"])
        cleanup_end_ns = int(precondition["cleanup_end_monotonic_ns"])
        qualification_ns = _marker_timestamp(qualification_marker)
        marker_metadata = {
            "label": label,
            "attempt": attempt,
            "boundary_monotonic_ns": boundary_ns,
            "cleanup_end_monotonic_ns": cleanup_end_ns,
            "sample_monotonic_ns": raw_attempt.get("qualification", {}).get(
                "sample_monotonic_ns"
            )
            if isinstance(raw_attempt.get("qualification"), dict)
            else None,
        }
        _validate_marker_metadata(qualification_marker, marker_metadata)
        qualification = telemetry.causal_qualification(
            raw_attempt.get("qualification"),
            label=label,
            reference_ns=qualification_ns,
            not_before_ns=cleanup_end_ns,
            first_after_boundary=True,
            thermal_lock=thermal_lock,
            expected_prefix={
                "attempt": attempt,
                "boundary": THERMAL_HANDOFF_BOUNDARY,
                "boundary_monotonic_ns": boundary_ns,
                "cleanup_end_monotonic_ns": cleanup_end_ns,
                "qualification_monotonic_ns": qualification_ns,
            },
        )
        result_ns = _marker_timestamp(result_marker)
        if raw_attempt.get("qualification_result_marker_monotonic_ns") != result_ns:
            raise ValueError("qualification result clock differs from raw telemetry")
        _validate_marker_metadata(
            result_marker,
            dict(base_metadata)
            | {
                "label": label,
                "thermal_attempt": attempt,
                "boundary_monotonic_ns": boundary_ns,
                "cleanup_end_monotonic_ns": cleanup_end_ns,
                "qualification_monotonic_ns": qualification_ns,
                "sample_monotonic_ns": qualification["sample_monotonic_ns"],
                "passed": qualification["passed"],
                "failure_reason": qualification["failure_reason"],
            },
        )
        if raw_attempt.get("measured_process_states") != expected_states:
            raise ValueError("measured processes were not all paused at qualification")
        start_marker: dict[str, Any] | None = None
        release_marker: dict[str, Any] | None = None
        if qualification["passed"]:
            start_marker = telemetry.marker(
                "guard_block_start",
                case_id=str(base_metadata["case_id"]),
                block=int(base_metadata["block"]),
                attempt=block_attempt,
                thermal_attempt=attempt,
            )
            release_marker = telemetry.marker(
                "guard_block_measurement_release",
                case_id=str(base_metadata["case_id"]),
                block=int(base_metadata["block"]),
                attempt=block_attempt,
                thermal_attempt=attempt,
            )
            handoff_identity = {
                "thermal_attempt": attempt,
                "thermal_boundary": THERMAL_HANDOFF_BOUNDARY,
                "thermal_boundary_monotonic_ns": boundary_ns,
                "thermal_handoff_max_ms": THERMAL_HANDOFF_MAX_MS,
            }
            _validate_marker_metadata(
                start_marker,
                dict(base_metadata)
                | {"worker_pids": list(worker_pids), "critical_pid": critical_pid}
                | handoff_identity,
            )
            _validate_marker_metadata(
                release_marker,
                dict(base_metadata) | {"critical_pid": critical_pid} | handoff_identity,
            )
        stored_start_ns = raw_attempt.get("start_marker_monotonic_ns")
        stored_release_ns = raw_attempt.get("measurement_release_marker_monotonic_ns")
        if qualification["passed"]:
            assert start_marker is not None and release_marker is not None
            start_ns = _marker_timestamp(start_marker)
            release_ns = _marker_timestamp(release_marker)
            if stored_start_ns != start_ns or stored_release_ns != release_ns:
                raise ValueError("pre-release marker clocks differ from raw telemetry")
            boundary_clocks = (
                cleanup_end_ns,
                int(qualification["sample_monotonic_ns"]),
                qualification_ns,
                result_ns,
                start_ns,
                release_ns,
            )
            maximum_ns = int(THERMAL_HANDOFF_MAX_MS * 1_000_000.0)
            pre_release_passed = all(
                boundary_ns < clock and clock - boundary_ns < maximum_ns
                for clock in boundary_clocks
            ) and all(
                current >= previous
                for previous, current in zip(
                    boundary_clocks, boundary_clocks[1:], strict=False
                )
            )
        else:
            if stored_start_ns is not None or stored_release_ns is not None:
                raise ValueError("failed qualification released a measured process")
            pre_release_passed = False
        if raw_attempt.get("pre_release_passed") is not pre_release_passed:
            raise ValueError("pre-release qualification outcome differs from raw clocks")
        expected_failure = (
            None
            if pre_release_passed
            else qualification["failure_reason"]
            if not qualification["passed"]
            else "active-boundary pre-release handoff exceeded the strict limit"
        )
        if raw_attempt.get("failure_reason") != expected_failure:
            raise ValueError("pre-release failure reason is invalid")
        attempt_times = [*precondition_times, qualification_ns, result_ns]
        if start_marker is not None and release_marker is not None:
            attempt_times.extend(
                [_marker_timestamp(start_marker), _marker_timestamp(release_marker)]
            )
        if any(
            current <= previous
            for previous, current in zip(
                attempt_times, attempt_times[1:], strict=False
            )
        ):
            raise ValueError("thermal qualification attempt markers are not ordered")
        if all_times and all_times[-1] >= attempt_times[0]:
            raise ValueError("thermal qualification attempts overlap")
        all_times.extend(attempt_times)
        is_final = attempt == len(raw_attempts)
        if pre_release_passed is not is_final:
            raise ValueError("measurement did not use the first valid pre-release attempt")
        replayed_attempts.append(
            {
                "attempt": attempt,
                "label": label,
                "thermal_precondition": precondition,
                "qualification": qualification,
                "qualification_result_marker_monotonic_ns": result_ns,
                "measured_process_states": expected_states,
                "start_marker_monotonic_ns": stored_start_ns,
                "measurement_release_marker_monotonic_ns": stored_release_ns,
                "pre_release_passed": pre_release_passed,
                "failure_reason": expected_failure,
            }
        )
    successful = replayed_attempts[-1]
    if (
        block_attempt_record.get("selected_thermal_attempt")
        != len(replayed_attempts)
        or successful["pre_release_passed"] is not True
    ):
        raise ValueError("guard block does not bind the successful qualification")
    return (
        replayed_attempts,
        successful,
        all_times,
    )


_BLOCK_ATTEMPT_KEYS = {
    "attempt",
    "thermally_valid",
    "prepare_marker_monotonic_ns",
    "start_marker_monotonic_ns",
    "measurement_release_marker_monotonic_ns",
    "resume_marker_monotonic_ns",
    "actual_start_qualification_marker_monotonic_ns",
    "result_marker_monotonic_ns",
    "end_marker_monotonic_ns",
    "measurement_start_monotonic_ns",
    "measurement_end_monotonic_ns",
    "thermal_attempts",
    "selected_thermal_attempt",
    "thermal_handoff",
    "actual_start_qualification",
    "critical_affinity",
    "critical_trace",
    "critical_json",
    "worker_json",
}


def _require_first_thermally_valid_attempt(
    attempts: Sequence[Mapping[str, Any]], selected_attempt: object
) -> Mapping[str, Any]:
    if (
        not attempts
        or selected_attempt != len(attempts)
        or any(
            attempt.get("thermally_valid") is not (index == len(attempts))
            for index, attempt in enumerate(attempts, start=1)
        )
    ):
        raise ValueError(
            "guard retry did not select the first thermally valid block-attempt"
        )
    return attempts[-1]


def _replay_block_attempt(
    value: object,
    *,
    expected_attempt: int,
    base_metadata: Mapping[str, Any],
    expanded: Sequence[Mapping[str, Any]],
    registry: EvidenceRegistry,
    telemetry: TelemetryReplay,
    thermal_lock: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
    small_uuid: str,
    big_uuid: str,
    prior_end_ns: int,
) -> tuple[dict[str, Any], int]:
    if (
        not isinstance(value, dict)
        or set(value) != _BLOCK_ATTEMPT_KEYS
        or value.get("attempt") != expected_attempt
    ):
        raise ValueError("guard block-attempt has an invalid schema or order")
    case_id = str(base_metadata["case_id"])
    block = int(base_metadata["block"])
    identity = dict(base_metadata) | {"attempt": expected_attempt}
    selected_thermal_attempt = value.get("selected_thermal_attempt")
    if type(selected_thermal_attempt) is not int:
        raise ValueError("guard block-attempt lacks a selected thermal attempt")
    prepare = telemetry.marker(
        "guard_block_prepare", case_id=case_id, block=block, attempt=expected_attempt
    )
    start = telemetry.marker(
        "guard_block_start",
        case_id=case_id,
        block=block,
        attempt=expected_attempt,
        thermal_attempt=selected_thermal_attempt,
    )
    release = telemetry.marker(
        "guard_block_measurement_release",
        case_id=case_id,
        block=block,
        attempt=expected_attempt,
        thermal_attempt=selected_thermal_attempt,
    )
    resume = telemetry.marker(
        "guard_block_resume", case_id=case_id, block=block, attempt=expected_attempt
    )
    actual_marker = telemetry.marker(
        "guard_actual_start_qualification",
        case_id=case_id,
        block=block,
        attempt=expected_attempt,
    )
    result_marker = telemetry.marker(
        "guard_block_result", case_id=case_id, block=block, attempt=expected_attempt
    )
    end = telemetry.marker(
        "guard_block_end", case_id=case_id, block=block, attempt=expected_attempt
    )
    _validate_marker_metadata(prepare, identity)
    marker_fields = {
        "prepare_marker_monotonic_ns": prepare,
        "start_marker_monotonic_ns": start,
        "measurement_release_marker_monotonic_ns": release,
        "resume_marker_monotonic_ns": resume,
        "actual_start_qualification_marker_monotonic_ns": actual_marker,
        "result_marker_monotonic_ns": result_marker,
        "end_marker_monotonic_ns": end,
    }
    for field, marker in marker_fields.items():
        if value.get(field) != _marker_timestamp(marker):
            raise ValueError(f"block-attempt {field} differs from raw telemetry")

    trace = replay_critical_trace(
        registry.claim(
            value.get("critical_trace"),
            kind="critical-csv",
            case_id=case_id,
            block=block,
        )
    )
    critical = _load_json_payload(
        registry.claim(
            value.get("critical_json"),
            kind="critical-json",
            case_id=case_id,
            block=block,
        ),
        "critical JSON",
    )
    critical_engine = artifacts["engine:critical:2g:resnet50-v2"]
    _validate_critical_json(
        critical,
        trace,
        expected_workers=len(expanded),
        expected_engine=pathlib.Path(critical_engine["path"]),
        big_uuid=big_uuid,
    )
    measurement_start_ns = int(critical["measurement_start_monotonic_ns"])
    measurement_end_ns = int(critical["measurement_end_monotonic_ns"])
    if (
        value.get("measurement_start_monotonic_ns") != measurement_start_ns
        or value.get("measurement_end_monotonic_ns") != measurement_end_ns
    ):
        raise ValueError("block-attempt clocks differ from critical JSON")
    worker_paths = value.get("worker_json")
    if not isinstance(worker_paths, list) or len(worker_paths) != len(expanded):
        raise ValueError("block-attempt worker evidence count is invalid")
    worker_pids: list[int] = []
    for index, (raw_path, client) in enumerate(
        zip(worker_paths, expanded, strict=True)
    ):
        worker = _load_json_payload(
            registry.claim(
                raw_path,
                kind="worker-json",
                case_id=case_id,
                block=block,
            ),
            "worker JSON",
        )
        artifact = artifacts[_engine_artifact_key(client)]
        expected_cpu = CPU_AFFINITY["pressure"][index % len(CPU_AFFINITY["pressure"])]
        worker_pids.append(
            _validate_worker_json(
                worker,
                expected_client=client,
                expected_index=index,
                expected_cpu=expected_cpu,
                expected_engine=pathlib.Path(artifact["path"]),
                expected_engine_sha256=artifact["sha256"],
                expected_uuid=(
                    small_uuid if client["placement"] == "resident-1g" else big_uuid
                ),
                release_marker_ns=_marker_timestamp(release),
                critical_end_ns=measurement_end_ns,
                result_marker_ns=_marker_timestamp(result_marker),
            )
        )
    if len(set(worker_pids)) != len(worker_pids):
        raise ValueError("worker PIDs are not unique within a block-attempt")
    critical_pid = int(critical["execution_environment"]["pid"])
    _validate_affinity_snapshot(
        value.get("critical_affinity"), critical_pid, CPU_AFFINITY["critical"][0]
    )
    thermal_attempts, selected, thermal_times = _replay_qualification_attempts(
        telemetry,
        value,
        base_metadata=identity,
        worker_pids=worker_pids,
        critical_pid=critical_pid,
        thermal_lock=thermal_lock,
    )
    qualification = selected["qualification"]
    if (
        selected["start_marker_monotonic_ns"] != _marker_timestamp(start)
        or selected["measurement_release_marker_monotonic_ns"]
        != _marker_timestamp(release)
    ):
        raise ValueError("selected pre-release attempt differs from block markers")
    handoff_identity = {
        "thermal_attempt": selected_thermal_attempt,
        "thermal_boundary": THERMAL_HANDOFF_BOUNDARY,
        "thermal_boundary_monotonic_ns": qualification["boundary_monotonic_ns"],
        "thermal_handoff_max_ms": THERMAL_HANDOFF_MAX_MS,
    }
    _validate_marker_metadata(
        resume,
        identity
        | {"critical_pid": critical_pid, "worker_pids": worker_pids}
        | handoff_identity
        | {"resume_semantics": "issued-before-sigcont"},
    )
    thermal_handoff = _replay_thermal_handoff(
        value.get("thermal_handoff"),
        qualification=qualification,
        qualification_result_ns=int(
            selected["qualification_result_marker_monotonic_ns"]
        ),
        start_ns=_marker_timestamp(start),
        release_ns=_marker_timestamp(release),
        resume_issued_ns=_marker_timestamp(resume),
        measurement_start_ns=measurement_start_ns,
        thermal_lock=thermal_lock,
    )
    actual = telemetry.causal_qualification(
        value.get("actual_start_qualification"),
        label=str(selected["label"]),
        reference_ns=measurement_start_ns,
        not_before_ns=int(qualification["cleanup_end_monotonic_ns"]),
        first_after_boundary=False,
        thermal_lock=thermal_lock,
        expected_prefix={"measurement_start_monotonic_ns": measurement_start_ns},
    )
    _validate_marker_metadata(actual_marker, identity | actual)
    thermally_valid = bool(
        thermal_handoff["strictly_within_bound"] and actual["passed"] is True
    )
    if value.get("thermally_valid") is not thermally_valid:
        raise ValueError("block-attempt thermal validity differs from raw evidence")
    _validate_marker_metadata(
        result_marker,
        identity
        | {
            "measurement_start_monotonic_ns": measurement_start_ns,
            "measurement_end_monotonic_ns": measurement_end_ns,
            "actual_start_qualification_marker_monotonic_ns": _marker_timestamp(
                actual_marker
            ),
            "thermal_handoff": thermal_handoff,
            "thermally_valid": thermally_valid,
        },
    )
    _validate_marker_metadata(end, identity | {"thermally_valid": thermally_valid})
    prepare_ns = _marker_timestamp(prepare)
    start_ns = _marker_timestamp(start)
    release_ns = _marker_timestamp(release)
    resume_ns = _marker_timestamp(resume)
    actual_ns = _marker_timestamp(actual_marker)
    result_ns = _marker_timestamp(result_marker)
    end_ns = _marker_timestamp(end)
    if not (
        prior_end_ns < prepare_ns < thermal_times[0]
        and thermal_times[-1] == release_ns
        and release_ns < resume_ns < measurement_start_ns < measurement_end_ns
        and measurement_end_ns < actual_ns < result_ns < end_ns
        and start_ns < release_ns
    ):
        raise ValueError("guard block-attempt evidence chain is not strictly ordered")
    measurement_health = telemetry.measurement_health(
        measurement_start_ns, measurement_end_ns, thermal_lock
    )
    return (
        {
            "attempt": expected_attempt,
            "thermally_valid": thermally_valid,
            "samples": trace["samples"],
            "drain_p999_ms": trace["drain_p999_ms"],
            "drain_max_ms": trace["drain_max_ms"],
            "thermal_attempts": thermal_attempts,
            "thermal_handoff": thermal_handoff,
            "actual_start_qualification": actual,
            "measurement_telemetry": measurement_health,
            "_pooled": trace["drains_ms"],
        },
        end_ns,
    )


def _replay_case(
    entry: Mapping[str, Any],
    expected_case: Mapping[str, Any],
    *,
    root: pathlib.Path,
    registry: EvidenceRegistry,
    telemetry: TelemetryReplay,
    thermal_lock: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
    small_uuid: str,
    big_uuid: str,
    prior_end_ns: int,
) -> tuple[dict[str, Any], int]:
    del root
    identity = {
        "case_id": expected_case["case_id"],
        "held_out": expected_case["held_out"],
        "clients": expected_case["clients"],
    }
    if any(entry.get(key) != value for key, value in identity.items()):
        raise ValueError("guard case identity differs from the fixed protocol")
    blocks = entry.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != BLOCKS:
        raise ValueError("guard case must contain exactly ten blocks")
    expanded = _expanded_clients(expected_case)
    pooled: list[float] = []
    replayed_blocks: list[dict[str, Any]] = []
    for expected_block, block_record in enumerate(blocks, start=1):
        if not isinstance(block_record, dict) or set(block_record) != {
            "block",
            "selected_attempt",
            "attempts",
        } or block_record.get("block") != expected_block:
            raise ValueError("guard logical block has an invalid schema or order")
        attempts = block_record.get("attempts")
        if (
            not isinstance(attempts, list)
            or not 1 <= len(attempts) <= BLOCK_MAX_ATTEMPTS
            or block_record.get("selected_attempt") != len(attempts)
        ):
            raise ValueError("guard logical block has an invalid attempt count")
        base_metadata = {
            "case_id": str(expected_case["case_id"]),
            "held_out": bool(expected_case["held_out"]),
            "block": expected_block,
        }
        replayed_attempts: list[dict[str, Any]] = []
        for expected_attempt, attempt in enumerate(attempts, start=1):
            replayed, prior_end_ns = _replay_block_attempt(
                attempt,
                expected_attempt=expected_attempt,
                base_metadata=base_metadata,
                expanded=expanded,
                registry=registry,
                telemetry=telemetry,
                thermal_lock=thermal_lock,
                artifacts=artifacts,
                small_uuid=small_uuid,
                big_uuid=big_uuid,
                prior_end_ns=prior_end_ns,
            )
            replayed_attempts.append(replayed)
        selected = _require_first_thermally_valid_attempt(
            replayed_attempts, block_record.get("selected_attempt")
        )
        pooled.extend(selected["_pooled"])
        replayed_blocks.append(
            {
                "block": expected_block,
                "selected_attempt": len(replayed_attempts),
                "attempts": [
                    {key: item for key, item in attempt.items() if key != "_pooled"}
                    for attempt in replayed_attempts
                ],
                "samples": selected["samples"],
                "drain_p999_ms": selected["drain_p999_ms"],
                "drain_max_ms": selected["drain_max_ms"],
            }
        )
    if len(pooled) != BLOCKS * EVENTS_PER_BLOCK:
        raise ValueError("case does not contain exactly 10,000 selected drain events")
    return (
        {
            **identity,
            "samples": len(pooled),
            "pooled_drain_p999_ms": percentile_type7(pooled, PERCENTILE),
            "observed_max_ms": max(pooled),
            "blocks": replayed_blocks,
            "_pooled": pooled,
        },
        prior_end_ns,
    )


def _guard_key(case: Mapping[str, Any]) -> tuple[str, str, str]:
    client = case["clients"][0]
    return (
        str(client["placement"]),
        str(client["quota_percent"]),
        str(client["modality"]),
    )


def additive_envelope_ms(
    clients: Sequence[Mapping[str, Any]],
    guard_values: Mapping[tuple[str, str, str], float],
) -> tuple[float, dict[str, float]]:
    per_instance: dict[str, float] = {}
    for client in clients:
        key = (
            str(client["placement"]),
            str(client["quota_percent"]),
            str(client["modality"]),
        )
        if key not in guard_values:
            raise ValueError(f"held-out case refers to an uncalibrated guard: {key}")
        per_instance[key[0]] = per_instance.get(key[0], 0.0) + (
            int(client["count"]) * guard_values[key]
        )
    return max(per_instance.values()), dict(sorted(per_instance.items()))


def _load_json_once(path: pathlib.Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root is not an object")
    return value, payload


def build_lock(
    summary: dict[str, Any],
    summary_path: pathlib.Path,
    *,
    summary_payload: bytes | None = None,
    thermal_lock_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    if summary_payload is None:
        summary_payload = summary_path.read_bytes()
    if json.loads(summary_payload) != summary:
        raise ValueError("summary object differs from the source bytes")
    thermal_source = summary.get("thermal_lock")
    if not isinstance(thermal_source, dict):
        raise ValueError("guard profile lacks thermal-lock provenance")
    recorded_thermal_path = pathlib.Path(str(thermal_source.get("path", ""))).resolve()
    if thermal_lock_path is not None and thermal_lock_path.resolve() != recorded_thermal_path:
        raise ValueError("requested thermal lock differs from the producer provenance")
    thermal_lock, thermal_payload = _load_json_once(recorded_thermal_path, "thermal lock")
    thermal_sha256 = hashlib.sha256(thermal_payload).hexdigest()
    if thermal_source.get("sha256") != thermal_sha256:
        raise ValueError("thermal lock changed after profiling")
    verify_thermal_lock(thermal_lock)
    root = summary_path.parent
    artifacts, small_uuid, big_uuid = _validate_profile_provenance(
        summary, thermal_lock, root
    )

    telemetry_raw = summary.get("telemetry_jsonl")
    if not isinstance(telemetry_raw, str) or pathlib.Path(telemetry_raw).is_absolute():
        raise ValueError("telemetry JSONL path must be relative")
    telemetry_candidate = root / telemetry_raw
    if telemetry_candidate.is_symlink():
        raise ValueError("telemetry JSONL symlinks are forbidden")
    telemetry_path = telemetry_candidate.resolve(strict=True)
    if root != telemetry_path and root not in telemetry_path.parents:
        raise ValueError("telemetry JSONL escapes the campaign directory")
    if not telemetry_path.is_file():
        raise ValueError("telemetry JSONL is not a regular file")
    telemetry_payload = telemetry_path.read_bytes()
    telemetry_sha256 = hashlib.sha256(telemetry_payload).hexdigest()
    if summary.get("telemetry_jsonl_sha256") != telemetry_sha256:
        raise ValueError("telemetry JSONL changed after profiling")
    telemetry = TelemetryReplay(telemetry_payload)
    collector_ready = telemetry.marker("collector_ready")
    campaign_start = telemetry.marker("guard_campaign_start")
    campaign_end = telemetry.marker("guard_campaign_end")
    if campaign_start["metadata"] != {"protocol": expected_protocol()}:
        raise ValueError("campaign-start marker contains a different protocol")
    if campaign_end["metadata"] != {}:
        raise ValueError("campaign-end marker metadata is invalid")
    if (
        summary.get("campaign_start_monotonic_ns")
        != _marker_timestamp(campaign_start)
        or summary.get("campaign_end_monotonic_ns") != _marker_timestamp(campaign_end)
        or not (
            _marker_timestamp(collector_ready)
            < _marker_timestamp(campaign_start)
            < _marker_timestamp(campaign_end)
        )
    ):
        raise ValueError("campaign marker clocks are invalid")
    allowed_markers = {
        "collector_ready",
        "guard_campaign_start",
        "thermal_prepare",
        "thermal_start",
        "thermal_active_stability_check",
        "thermal_measurement_end",
        "thermal_end",
        "thermal_start_qualification",
        "thermal_start_qualification_result",
        "guard_block_prepare",
        "guard_block_start",
        "guard_block_measurement_release",
        "guard_block_resume",
        "guard_actual_start_qualification",
        "guard_block_result",
        "guard_block_end",
        "guard_campaign_end",
    }
    if any(marker["name"] not in allowed_markers for marker in telemetry.markers):
        raise ValueError("telemetry contains an unexpected or aborted campaign marker")
    single_entries = summary.get("single_client")
    held_entries = summary.get("held_out")
    if not isinstance(single_entries, list) or not isinstance(held_entries, list):
        raise ValueError("guard profile case collections are missing")
    if [entry.get("case_id") for entry in single_entries] != [
        case["case_id"] for case in expected_single_cases()
    ] or [entry.get("case_id") for entry in held_entries] != [
        case["case_id"] for case in expected_held_out_cases()
    ]:
        raise ValueError("guard profile case order differs from the fixed protocol")
    all_entries = [*single_entries, *held_entries]
    expected_marker_count = 3
    for entry in all_entries:
        blocks = entry.get("blocks") if isinstance(entry, dict) else None
        if not isinstance(blocks, list):
            raise ValueError("guard profile case lacks block evidence")
        for block in blocks:
            block_attempts = block.get("attempts") if isinstance(block, dict) else None
            if not isinstance(block_attempts, list):
                raise ValueError("guard block lacks block-attempt evidence")
            for block_attempt in block_attempts:
                if not isinstance(block_attempt, dict):
                    raise ValueError("guard block-attempt evidence is invalid")
                thermal_attempts = block_attempt.get("thermal_attempts")
                if not isinstance(thermal_attempts, list):
                    raise ValueError("guard block-attempt lacks thermal attempts")
                expected_marker_count += 5
                for thermal_attempt in thermal_attempts:
                    if not isinstance(thermal_attempt, dict):
                        raise ValueError("guard thermal attempt is invalid")
                    precondition = thermal_attempt.get("thermal_precondition")
                    checks = (
                        precondition.get("active_stability_checks")
                        if isinstance(precondition, dict)
                        else None
                    )
                    if not isinstance(checks, list):
                        raise ValueError("guard precondition lacks active checks")
                    expected_marker_count += 6 + len(checks)
                    qualification = thermal_attempt.get("qualification")
                    if isinstance(qualification, dict) and qualification.get("passed") is True:
                        expected_marker_count += 2
    if len(telemetry.markers) != expected_marker_count:
        raise ValueError("telemetry marker count differs from the fixed campaign")

    registry = EvidenceRegistry(root)
    prior_end_ns = _marker_timestamp(campaign_start)
    replayed_single: list[dict[str, Any]] = []
    for entry, expected_case in zip(
        single_entries, expected_single_cases(), strict=True
    ):
        replayed, prior_end_ns = _replay_case(
            entry,
            expected_case,
            root=root,
            registry=registry,
            telemetry=telemetry,
            thermal_lock=thermal_lock,
            artifacts=artifacts,
            small_uuid=small_uuid,
            big_uuid=big_uuid,
            prior_end_ns=prior_end_ns,
        )
        replayed_single.append(replayed)
    replayed_held: list[dict[str, Any]] = []
    for entry, expected_case in zip(
        held_entries, expected_held_out_cases(), strict=True
    ):
        replayed, prior_end_ns = _replay_case(
            entry,
            expected_case,
            root=root,
            registry=registry,
            telemetry=telemetry,
            thermal_lock=thermal_lock,
            artifacts=artifacts,
            small_uuid=small_uuid,
            big_uuid=big_uuid,
            prior_end_ns=prior_end_ns,
        )
        replayed_held.append(replayed)
    if prior_end_ns >= _marker_timestamp(campaign_end):
        raise ValueError("campaign end precedes the final guard block")

    guards: dict[str, dict[str, dict[str, Any]]] = {}
    guard_values: dict[tuple[str, str, str], float] = {}
    single_evidence: list[dict[str, Any]] = []
    for replayed in replayed_single:
        raw_p999, guard_ms = estimate_guard_ms(replayed["_pooled"])
        placement, quota, modality = _guard_key(replayed)
        guard_values[(placement, quota, modality)] = guard_ms
        guards.setdefault(placement, {}).setdefault(quota, {})[modality] = {
            "guard_ms": guard_ms,
            "raw_pooled_p999_ms": raw_p999,
            "samples": replayed["samples"],
            "observed_max_ms": replayed["observed_max_ms"],
        }
        single_evidence.append(
            {key: value for key, value in replayed.items() if key != "_pooled"}
        )

    held_validation: list[dict[str, Any]] = []
    for replayed in replayed_held:
        envelope, components = additive_envelope_ms(
            replayed["clients"], guard_values
        )
        observed = float(replayed["observed_max_ms"])
        require_held_out_coverage(
            str(replayed["case_id"]),
            envelope_ms=envelope,
            observed_max_ms=observed,
        )
        held_validation.append(
            {
                **{
                    key: value
                    for key, value in replayed.items()
                    if key != "_pooled"
                },
                "additive_per_instance_ms": components,
                "derived_envelope_ms": envelope,
                "formal_period_ms": FORMAL_PERIOD_MS,
                "headroom_ms": envelope - observed,
                "covered": True,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": LOCK_KIND,
        "protocol": expected_protocol(),
        "estimator": {
            "quantile": PERCENTILE,
            "method": "pooled-empirical-Hyndman-Fan-Type-7",
            "margin": MARGIN,
            "rounding": {"mode": "upward", "quantum_ms": ROUNDING_MS},
        },
        "guards": guards,
        "single_client_evidence": single_evidence,
        "held_out_validation": held_validation,
        "source": {
            "profile_summary": str(summary_path),
            "profile_summary_sha256": hashlib.sha256(summary_payload).hexdigest(),
            "telemetry_jsonl": str(telemetry_path),
            "telemetry_jsonl_sha256": telemetry_sha256,
            "raw_files": registry.manifest,
        },
        "thermal_lock": {
            "path": str(recorded_thermal_path),
            "sha256": thermal_sha256,
        },
        "hardware": summary["hardware"],
        "hardware_snapshot_sha256": summary["hardware_snapshot_sha256"],
        "mig": summary["mig"],
        "mps": summary["mps"],
        "cpu_affinity": CPU_AFFINITY,
        "producer_cpu_affinity": summary["producer_cpu_affinity"],
        "artifacts": artifacts,
    }


def verify_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != SCHEMA_VERSION or lock.get("kind") != LOCK_KIND:
        raise ValueError("guard lock has an invalid schema or kind")
    source = lock.get("source")
    thermal = lock.get("thermal_lock")
    if not isinstance(source, dict) or not isinstance(thermal, dict):
        raise ValueError("guard lock lacks source provenance")
    summary_path = pathlib.Path(str(source.get("profile_summary", "")))
    summary, summary_payload = _load_json_once(summary_path, "guard profile")
    if hashlib.sha256(summary_payload).hexdigest() != source.get(
        "profile_summary_sha256"
    ):
        raise ValueError("guard profile changed after freezing")
    rebuilt = build_lock(
        summary,
        summary_path,
        summary_payload=summary_payload,
        thermal_lock_path=pathlib.Path(str(thermal.get("path", ""))),
    )
    if rebuilt != lock:
        raise ValueError("guard lock fields do not match replayed raw evidence")


def _atomic_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=pathlib.Path)
    parser.add_argument("--thermal-lock", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verify", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        if any(value is not None for value in (args.summary, args.thermal_lock, args.output)):
            parser.error("--verify cannot be combined with lock creation")
    elif args.summary is None or args.thermal_lock is None or args.output is None:
        parser.error("summary, --thermal-lock, and --output are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify is not None:
        lock, _ = _load_json_once(args.verify.resolve(), "guard lock")
        verify_lock(lock)
        return 0
    summary, payload = _load_json_once(args.summary.resolve(), "guard profile")
    lock = build_lock(
        summary,
        args.summary,
        summary_payload=payload,
        thermal_lock_path=args.thermal_lock,
    )
    _atomic_json(args.output.resolve(), lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
