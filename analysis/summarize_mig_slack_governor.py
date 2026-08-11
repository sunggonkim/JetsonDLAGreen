#!/usr/bin/env python3
"""Aggregate repeated MIG slack-borrowing experiments."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any

try:
    from freeze_p9_deadline import verify_lock as verify_deadline_lock
    from freeze_p9_guard import (
        expected_protocol as expected_guard_protocol,
        verify_lock as verify_guard_lock,
    )
    from freeze_p9_thermal import (
        LOCK_SCHEMA_VERSION as THERMAL_LOCK_SCHEMA_VERSION,
        THERMAL_ACTIVE_STABLE_ENDPOINTS,
        THERMAL_ACTIVE_STABLE_SPACING_SECONDS,
        THERMAL_HANDOFF_BOUNDARY,
        THERMAL_QUALIFICATION_MAX_ATTEMPTS,
        verify_lock as verify_thermal_lock,
    )
except ModuleNotFoundError:
    from analysis.freeze_p9_deadline import verify_lock as verify_deadline_lock
    from analysis.freeze_p9_guard import (
        expected_protocol as expected_guard_protocol,
        verify_lock as verify_guard_lock,
    )
    from analysis.freeze_p9_thermal import (
        LOCK_SCHEMA_VERSION as THERMAL_LOCK_SCHEMA_VERSION,
        THERMAL_ACTIVE_STABLE_ENDPOINTS,
        THERMAL_ACTIVE_STABLE_SPACING_SECONDS,
        THERMAL_HANDOFF_BOUNDARY,
        THERMAL_QUALIFICATION_MAX_ATTEMPTS,
        verify_lock as verify_thermal_lock,
    )

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.tegrastats_telemetry import (  # noqa: E402
    TelemetryMarker,
    TelemetrySample,
    aggregate_samples,
    parse_tegrastats_line,
)


T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
}
POLICIES = {
    "static-mig",
    "resident-full-gate",
    "same-mig",
    "uncoordinated-borrow",
    "fixed-borrow",
    "fixed-full-gate",
    "mig-governor",
}
PROPOSED_SYSTEM = "QUIET"
PROPOSED_POLICY_ID = "mig-governor"
PRIMARY_BASELINE = "resident-full-gate"
ADAPTIVE_BASELINE = "fixed-full-gate"
POLICY_PRESENTATION = {
    "static-mig": ("NVIDIA MIG isolation", "baseline"),
    "same-mig": ("NVIDIA MIG+MPS spatial sharing", "baseline"),
    "resident-full-gate": ("Resident-only quiescence", "ablation"),
    "uncoordinated-borrow": ("Uncoordinated co-location", "ablation"),
    "fixed-borrow": ("Borrower-only static gating", "ablation"),
    "fixed-full-gate": ("Static full gating", "ablation"),
    PROPOSED_POLICY_ID: (PROPOSED_SYSTEM, "proposed"),
}
MODALITIES = ("language", "audio")
PLACEMENTS = ("resident-1g", "borrower-2g")
MODEL_BY_MODALITY = {
    "language": "distilbert-sst2",
    "audio": "whisper-tiny-encoder",
}
RESIDENT_QUOTAS = (25, 50, 100)
SCHEDULED_WILLIAMS_ORDERS = (
    ("fixed-borrow", "fixed-full-gate", "uncoordinated-borrow", "mig-governor", "same-mig", "static-mig", "resident-full-gate"),
    ("fixed-borrow", "uncoordinated-borrow", "fixed-full-gate", "same-mig", "mig-governor", "resident-full-gate", "static-mig"),
    ("same-mig", "resident-full-gate", "uncoordinated-borrow", "static-mig", "fixed-borrow", "mig-governor", "fixed-full-gate"),
    ("mig-governor", "static-mig", "fixed-full-gate", "resident-full-gate", "fixed-borrow", "same-mig", "uncoordinated-borrow"),
    ("uncoordinated-borrow", "fixed-borrow", "same-mig", "fixed-full-gate", "resident-full-gate", "mig-governor", "static-mig"),
    ("uncoordinated-borrow", "same-mig", "fixed-borrow", "resident-full-gate", "fixed-full-gate", "static-mig", "mig-governor"),
    ("resident-full-gate", "same-mig", "static-mig", "uncoordinated-borrow", "mig-governor", "fixed-borrow", "fixed-full-gate"),
    ("resident-full-gate", "static-mig", "same-mig", "mig-governor", "uncoordinated-borrow", "fixed-full-gate", "fixed-borrow"),
    ("same-mig", "uncoordinated-borrow", "resident-full-gate", "fixed-borrow", "static-mig", "fixed-full-gate", "mig-governor"),
    ("mig-governor", "fixed-full-gate", "static-mig", "fixed-borrow", "resident-full-gate", "uncoordinated-borrow", "same-mig"),
    ("fixed-full-gate", "mig-governor", "fixed-borrow", "static-mig", "uncoordinated-borrow", "resident-full-gate", "same-mig"),
    ("static-mig", "resident-full-gate", "mig-governor", "same-mig", "fixed-full-gate", "uncoordinated-borrow", "fixed-borrow"),
    ("fixed-full-gate", "fixed-borrow", "mig-governor", "uncoordinated-borrow", "static-mig", "same-mig", "resident-full-gate"),
    ("static-mig", "mig-governor", "resident-full-gate", "fixed-full-gate", "same-mig", "fixed-borrow", "uncoordinated-borrow"),
)


def confidence(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.fmean(values)
    if count == 1:
        return {"n": 1, "mean": mean, "stdev": 0.0, "ci95": 0.0}
    stdev = statistics.stdev(values)
    return {
        "n": count,
        "mean": mean,
        "stdev": stdev,
        "ci95": T95.get(count - 1, 1.96) * stdev / math.sqrt(count),
    }


def binomial_log_cdf(successes: int, trials: int, probability: float) -> float:
    if successes < 0 or trials <= 0 or successes > trials:
        raise ValueError("invalid binomial dimensions")
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 0.0 if successes == trials else -math.inf
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * math.log(probability)
        + (trials - index) * math.log1p(-probability)
        for index in range(successes + 1)
    ]
    maximum = max(terms)
    return maximum + math.log(sum(math.exp(term - maximum) for term in terms))


def clopper_pearson_upper(
    successes: int, trials: int, confidence_level: float = 0.95
) -> float:
    if successes < 0 or trials <= 0 or successes > trials:
        raise ValueError("invalid binomial dimensions")
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence level must be in (0, 1)")
    if successes == trials:
        return 1.0
    alpha_log = math.log1p(-confidence_level)
    if successes == 0:
        return -math.expm1(alpha_log / trials)
    lower = successes / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if binomial_log_cdf(successes, trials, midpoint) > alpha_log:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def strict_float(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be a finite number")
    return result


def strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def require_close(
    actual: object,
    expected: float,
    label: str,
    *,
    absolute_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-10,
) -> None:
    value = strict_float(actual, label)
    if not math.isclose(
        value,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(f"{label} differs from replay ({value} != {expected})")


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def guard_profile_from_lock(
    guard_lock: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    expected_quotas = {
        "resident-1g": (25, 50, 100),
        "borrower-2g": (100,),
    }
    raw_guards = guard_lock.get("guards")
    if not isinstance(raw_guards, dict) or set(raw_guards) != set(expected_quotas):
        raise ValueError("guard lock has invalid placement guards")
    profile: dict[str, dict[str, dict[str, float]]] = {}
    for placement, quotas in expected_quotas.items():
        raw_placement = raw_guards.get(placement)
        expected_quota_keys = {str(quota) for quota in quotas}
        if not isinstance(raw_placement, dict) or set(raw_placement) != expected_quota_keys:
            raise ValueError(f"guard lock has invalid quotas for {placement}")
        profile[placement] = {}
        for quota in quotas:
            raw_modalities = raw_placement[str(quota)]
            if not isinstance(raw_modalities, dict) or set(raw_modalities) != set(
                MODALITIES
            ):
                raise ValueError(
                    f"guard lock has invalid modalities for {placement}/q{quota}"
                )
            profile[placement][str(quota)] = {}
            for modality in MODALITIES:
                evidence = raw_modalities[modality]
                if not isinstance(evidence, dict):
                    raise ValueError("guard lock profile evidence must be an object")
                guard_ms = strict_float(
                    evidence.get("guard_ms"),
                    f"guard lock {placement}/q{quota}/{modality}",
                    minimum=0.0,
                )
                if guard_ms <= 0.0 or guard_ms >= 20.0:
                    raise ValueError("guard lock contains an invalid formal guard")
                profile[placement][str(quota)][modality] = guard_ms
    return profile


def guard_artifact_sha256(artifacts: object, key: str) -> str:
    if not isinstance(artifacts, dict):
        raise ValueError("guard lock lacks artifact provenance")
    record = artifacts.get(key)
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256"}
        or not is_sha256(record.get("sha256"))
    ):
        raise ValueError(f"guard lock artifact is invalid: {key}")
    return str(record["sha256"])


def validate_guard_lock_binding(
    guard_lock: dict[str, Any],
    guard_lock_sha256: str,
    deadline_lock: dict[str, Any],
    thermal_lock: dict[str, Any],
    thermal_lock_sha256: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if not is_sha256(guard_lock_sha256):
        raise ValueError("guard lock SHA-256 is invalid")
    if guard_lock.get("schema_version") != 3:
        raise ValueError("formal aggregation requires guard-lock schema 3")
    if guard_lock.get("protocol") != expected_guard_protocol():
        raise ValueError("guard lock differs from the exact formal protocol")
    if (
        guard_lock.get("cpu_affinity") != FORMAL_CPU_AFFINITY
        or guard_lock.get("producer_cpu_affinity")
        != FORMAL_CPU_AFFINITY["telemetry"]
    ):
        raise ValueError("guard lock differs from formal CPU affinity")
    guard_thermal = guard_lock.get("thermal_lock")
    if (
        not isinstance(guard_thermal, dict)
        or guard_thermal.get("sha256") != thermal_lock_sha256
        or deadline_lock.get("thermal_lock_sha256") != thermal_lock_sha256
    ):
        raise ValueError("guard, deadline, and thermal locks are not cross-bound")
    if deadline_lock.get("guard_lock_sha256") != guard_lock_sha256:
        raise ValueError("deadline lock is not bound to the supplied guard lock")
    if (
        guard_lock.get("hardware") != thermal_lock.get("pilot_hardware")
        or guard_lock.get("hardware") != deadline_lock.get("calibration_hardware")
    ):
        raise ValueError("guard lock hardware differs from frozen calibration")
    guard_mig = guard_lock.get("mig")
    deadline_mig = deadline_lock.get("calibration_mig")
    if (
        not isinstance(guard_mig, dict)
        or not isinstance(deadline_mig, dict)
        or guard_mig.get("big_uuid") != deadline_mig.get("critical_uuid")
        or guard_mig.get("small_uuid") != deadline_mig.get("resident_uuid")
    ):
        raise ValueError("guard lock MIG mapping differs from deadline calibration")

    calibration_artifacts = deadline_lock.get("calibration_artifacts")
    if not isinstance(calibration_artifacts, dict):
        raise ValueError("deadline lock lacks calibration artifact provenance")
    benchmark_sha256 = calibration_artifacts.get("benchmark_sha256")
    engine_sha256 = calibration_artifacts.get("engines_sha256")
    implementation_sha256 = calibration_artifacts.get("implementation_sha256")
    if (
        not is_sha256(benchmark_sha256)
        or not isinstance(engine_sha256, dict)
        or not isinstance(implementation_sha256, dict)
        or guard_artifact_sha256(guard_lock.get("artifacts"), "benchmark")
        != benchmark_sha256
    ):
        raise ValueError("guard benchmark differs from deadline calibration")
    guard_to_implementation = {
        "producer": "runtime/profile_p9_guard.py",
        "freezer": "analysis/freeze_p9_guard.py",
        "telemetry_runtime": "runtime/tegrastats_telemetry.py",
        "governor_runtime": "runtime/mig_slack_governor.py",
        "guard_runner": "scripts/run_p9_guard_calibration.sh",
        "formal_runner": "scripts/run_p9_mig_slack_governor.sh",
        "mig_configurator": "scripts/configure_thor_mig.sh",
        "benchmark_source": "benchmarks/trt_inference.cpp",
    }
    for guard_key, implementation_key in guard_to_implementation.items():
        if guard_artifact_sha256(
            guard_lock.get("artifacts"), guard_key
        ) != implementation_sha256.get(implementation_key):
            raise ValueError(
                f"guard implementation differs from deadline calibration: {guard_key}"
            )
    expected_engines = {
        "engine:critical:2g:resnet50-v2": "critical-2g-resnet50-v2"
    }
    for placement, quotas in (("resident-1g", RESIDENT_QUOTAS), ("borrower-2g", (100,))):
        prefix = "resident-1g" if placement == "resident-1g" else "borrower-2g"
        for quota in quotas:
            for modality, model in MODEL_BY_MODALITY.items():
                expected_engines[
                    f"engine:{placement}:q{quota}:{modality}"
                ] = f"{prefix}-q{quota}-{model}"
    for guard_key, calibration_key in expected_engines.items():
        if guard_artifact_sha256(
            guard_lock.get("artifacts"), guard_key
        ) != engine_sha256.get(calibration_key):
            raise ValueError(
                f"guard engine differs from deadline calibration: {guard_key}"
            )
    return guard_profile_from_lock(guard_lock)


@dataclasses.dataclass(frozen=True)
class CriticalTrace:
    release_to_completion_ms: list[float]
    gpu_service_ms: list[float]
    queue_delay_ms: list[float]
    gate_overhead_ms: list[float]
    drain_ms: list[float]
    resume_ms: list[float]


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized["policy_order"] = None
    normalized["experiment_label"] = None
    return normalized


def relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return candidate / baseline - 1.0


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile requires samples and a quantile in [0, 1]")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def read_critical_trace(path: pathlib.Path, expected_samples: int) -> CriticalTrace:
    if not path.is_file():
        raise ValueError(f"missing raw trace: {path}")
    columns = {
        "release_to_completion_ms": [],
        "gpu_service_ms": [],
        "queue_delay_ms": [],
        "gate_overhead_ms": [],
        "drain_ms": [],
        "resume_ms": [],
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"request", *columns}
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"raw trace lacks required fields: {path}")
        for expected_request, row in enumerate(reader):
            if int(row["request"]) != expected_request:
                raise ValueError(f"raw trace request sequence is invalid: {path}")
            for name, values in columns.items():
                value = float(row[name])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"raw trace contains invalid {name}: {path}")
                values.append(value)
    if len(columns["release_to_completion_ms"]) != expected_samples:
        raise ValueError(
            f"raw trace {path} has "
            f"{len(columns['release_to_completion_ms'])} samples, "
            f"expected {expected_samples}"
        )
    return CriticalTrace(**columns)


def read_latency_trace(path: pathlib.Path, expected_samples: int) -> list[float]:
    return read_critical_trace(path, expected_samples).release_to_completion_ms


@dataclasses.dataclass
class RawTraceClaims:
    paths: set[pathlib.Path] = dataclasses.field(default_factory=set)
    inodes: set[tuple[int, int]] = dataclasses.field(default_factory=set)
    sha256: set[str] = dataclasses.field(default_factory=set)
    provenance: dict[str, dict[str, int | str]] = dataclasses.field(
        default_factory=dict
    )


def claim_raw_trace(
    path: pathlib.Path, claimed: RawTraceClaims
) -> tuple[tuple[int, int], str]:
    if not path.is_file():
        raise ValueError(f"missing raw trace: {path}")
    resolved = path.resolve()
    stat = path.stat()
    inode = (stat.st_dev, stat.st_ino)
    digest = file_sha256(path)
    if resolved in claimed.paths:
        raise ValueError(f"raw trace was reused: {path}")
    if inode in claimed.inodes:
        raise ValueError(f"raw trace hardlink was reused: {path}")
    if digest in claimed.sha256:
        raise ValueError(f"byte-identical raw trace was reused: {path}")
    claimed.paths.add(resolved)
    claimed.inodes.add(inode)
    claimed.sha256.add(digest)
    claimed.provenance[str(resolved)] = {
        "st_dev": stat.st_dev,
        "st_ino": stat.st_ino,
        "sha256": digest,
    }
    return inode, digest


def verify_raw_trace_snapshot(
    path: pathlib.Path, snapshot: tuple[tuple[int, int], str]
) -> None:
    stat = path.stat()
    inode = (stat.st_dev, stat.st_ino)
    if inode != snapshot[0] or file_sha256(path) != snapshot[1]:
        raise ValueError(f"raw trace changed while being replayed: {path}")


def action_identity(action: dict[str, Any], label: str) -> tuple[int, str, str, int]:
    tenant_id = strict_int(action.get("tenant_id"), f"{label}.tenant_id")
    modality = action.get("modality")
    placement = action.get("placement")
    quota = strict_int(action.get("quota_percent"), f"{label}.quota_percent", minimum=1)
    if modality not in MODALITIES:
        raise ValueError(f"{label}.modality is invalid")
    if placement not in PLACEMENTS:
        raise ValueError(f"{label}.placement is invalid")
    return tenant_id, modality, placement, quota


def expected_worker_sm_count(placement: str, quota: int) -> int:
    if placement == "resident-1g":
        resident_widths = {25: 2, 50: 4, 100: 8}
        if quota in resident_widths:
            return resident_widths[quota]
    elif placement == "borrower-2g" and quota == 100:
        return 12
    raise ValueError(f"unsupported formal MIG quota: {placement}/q{quota}")


def validate_engine_artifact(
    actual: object,
    expected: pathlib.Path,
    artifact_key: str,
    artifacts: dict[str, Any],
    label: str,
) -> None:
    if not isinstance(actual, str) or pathlib.Path(actual) != expected:
        raise ValueError(f"{label} differs from the formal engine path")
    engine_hashes = artifacts.get("engines_sha256")
    if not isinstance(engine_hashes, dict):
        raise ValueError("formal run lacks engine artifact hashes")
    expected_digest = engine_hashes.get(artifact_key)
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or not expected.is_file()
        or file_sha256(expected) != expected_digest
    ):
        raise ValueError(f"{label} differs from the frozen engine artifact")


def validate_execution_environment(
    value: object,
    *,
    label: str,
    expected_device: str,
    expected_quota: int,
    expected_cpus: list[int],
) -> int:
    expected_fields = {
        "pid",
        "cuda_visible_devices",
        "mps_active_thread_percentage",
        "cpu_affinity",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{label} has invalid execution-environment fields")
    pid = strict_int(value.get("pid"), f"{label}.pid", minimum=1)
    cpu_affinity = value.get("cpu_affinity")
    if not isinstance(cpu_affinity, list) or len(cpu_affinity) != len(expected_cpus):
        raise ValueError(f"{label}.cpu_affinity has invalid dimensions")
    actual_cpus = [
        strict_int(cpu, f"{label}.cpu_affinity[{index}]", minimum=0)
        for index, cpu in enumerate(cpu_affinity)
    ]
    if (
        value.get("cuda_visible_devices") != expected_device
        or strict_int(
            value.get("mps_active_thread_percentage"),
            f"{label}.mps_active_thread_percentage",
            minimum=1,
        )
        != expected_quota
        or actual_cpus != expected_cpus
    ):
        raise ValueError(f"{label} differs from the formal execution environment")
    return pid


def validate_trace_summary(
    value: object, values: list[float], label: str
) -> None:
    expected_fields = {
        "count",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "p999_ms",
        "max_ms",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{label} has invalid latency-summary fields")
    if strict_int(value.get("count"), f"{label}.count") != len(values):
        raise ValueError(f"{label}.count differs from the raw trace")
    expected = {
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "p999_ms": percentile(values, 0.999),
        "max_ms": max(values),
    }
    for field, expected_value in expected.items():
        require_close(
            value.get(field),
            expected_value,
            f"{label}.{field}",
            absolute_tolerance=1e-6,
        )


def validate_readiness_entry(
    value: object,
    *,
    label: str,
    role: str,
    pid: int,
    expected_cpu: int,
    tenant_id: int | None = None,
) -> None:
    expected_fields = {"role", "pid", "expected_cpu", "tasks"}
    if tenant_id is not None:
        expected_fields.add("tenant_id")
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{label} has invalid readiness-affinity fields")
    if (
        value.get("role") != role
        or strict_int(value.get("pid"), f"{label}.pid", minimum=1) != pid
        or strict_int(
            value.get("expected_cpu"), f"{label}.expected_cpu", minimum=0
        )
        != expected_cpu
        or (
            tenant_id is not None
            and strict_int(value.get("tenant_id"), f"{label}.tenant_id", minimum=0)
            != tenant_id
        )
    ):
        raise ValueError(f"{label} differs from the benchmark execution environment")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{label}.tasks must be non-empty")
    tids: set[int] = set()
    for index, task in enumerate(tasks):
        task_label = f"{label}.tasks[{index}]"
        if not isinstance(task, dict) or set(task) != {"tid", "cpus"}:
            raise ValueError(f"{task_label} has invalid fields")
        tid = strict_int(task.get("tid"), f"{task_label}.tid", minimum=1)
        cpus = task.get("cpus")
        if (
            tid in tids
            or not isinstance(cpus, list)
            or len(cpus) != 1
            or strict_int(cpus[0], f"{task_label}.cpus[0]", minimum=0)
            != expected_cpu
        ):
            raise ValueError(f"{task_label} has invalid task affinity")
        tids.add(tid)
    if pid not in tids:
        raise ValueError(f"{label}.tasks omits the process leader")


def replay_critical_result(
    epoch: dict[str, Any],
    label: str,
    trace: CriticalTrace,
    config: dict[str, Any],
    deadline_ms: float,
    mig: dict[str, str],
    artifacts: dict[str, Any],
    *,
    expected_samples: int,
    expected_workers: int,
    gated_workers: int,
    guard_ms: float,
    release_ns: int,
    start_ns: int,
    end_ns: int,
    collected_ns: int,
    cleanup_ns: int,
) -> tuple[pathlib.Path, int]:
    critical = epoch.get("critical")
    expected_fields = {
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
    if not isinstance(critical, dict) or set(critical) != expected_fields:
        raise ValueError(f"{label}.critical has invalid benchmark-result fields")
    if (
        strict_int(
            critical.get("schema_version"), f"{label}.critical.schema_version"
        )
        != 1
        or critical.get("role") != "benchmark"
        or critical.get("model") != "resnet50-v2"
    ):
        raise ValueError(f"{label}.critical is not the formal ResNet50 benchmark")

    engine_value = critical.get("engine")
    if not isinstance(engine_value, str):
        raise ValueError(f"{label}.critical.engine is invalid")
    engine_path = pathlib.Path(engine_value)
    if not engine_path.is_absolute() or engine_path.parts[-2:] != (
        "mig-2g",
        "resnet50-v2.engine",
    ):
        raise ValueError(f"{label}.critical.engine is not the formal 2g engine")
    engine_root = engine_path.parent.parent
    validate_engine_artifact(
        engine_value,
        engine_root / "mig-2g" / "resnet50-v2.engine",
        "critical-2g-resnet50-v2",
        artifacts,
        f"{label}.critical.engine",
    )

    cpu_affinity = config.get("cpu_affinity")
    if not isinstance(cpu_affinity, dict) or not isinstance(
        cpu_affinity.get("critical"), list
    ):
        raise ValueError("formal config lacks critical CPU affinity")
    critical_cpus = [
        strict_int(cpu, f"config.cpu_affinity.critical[{index}]", minimum=0)
        for index, cpu in enumerate(cpu_affinity["critical"])
    ]
    if len(critical_cpus) != 1:
        raise ValueError("formal config requires one critical CPU")
    critical_uuid = mig.get("critical_uuid")
    if not isinstance(critical_uuid, str) or not critical_uuid:
        raise ValueError("formal run lacks the critical MIG UUID")
    critical_pid = validate_execution_environment(
        critical.get("execution_environment"),
        label=f"{label}.critical.execution_environment",
        expected_device=critical_uuid,
        expected_quota=100,
        expected_cpus=critical_cpus,
    )
    if critical.get("gpu") != {
        "name": "NVIDIA Thor MIG 2g.0gb",
        "multiprocessors": 12,
    }:
        raise ValueError(f"{label}.critical differs from the formal 2g MIG width")

    critical_config = critical.get("config")
    expected_config_fields = {
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
    }
    if not isinstance(critical_config, dict) or set(critical_config) != expected_config_fields:
        raise ValueError(f"{label}.critical.config has invalid fields")
    expected_gated = gated_workers if guard_ms > 0.0 else 0
    expected_integer = {
        "warmup": strict_int(config.get("warmup"), "config.warmup"),
        "burst_size": strict_int(config.get("burst_size"), "config.burst_size"),
        "gated_processes": expected_gated,
        "stopped_processes": expected_workers,
        "stream_priority_value": -5,
    }
    expected_boolean = {
        "start_paused": True,
        "include_transfers": True,
    }
    expected_string = {
        "gate_mode": "cooperative" if expected_gated else "stop",
        "priority": "high",
    }
    for key, expected_value in expected_integer.items():
        if strict_int(
            critical_config.get(key),
            f"{label}.critical.config.{key}",
            minimum=-5 if key == "stream_priority_value" else 0,
        ) != expected_value:
            raise ValueError(f"{label}.critical.config differs from the formal command")
    for key, expected_value in expected_boolean.items():
        if strict_bool(
            critical_config.get(key), f"{label}.critical.config.{key}"
        ) is not expected_value:
            raise ValueError(f"{label}.critical.config differs from the formal command")
    if any(
        critical_config.get(key) != expected_value
        for key, expected_value in expected_string.items()
    ):
        raise ValueError(f"{label}.critical.config differs from the formal command")
    for key, expected_value in {
        "period_ms": strict_float(config.get("period_ms"), "config.period_ms"),
        "deadline_ms": deadline_ms,
        "duration_seconds": 0.0,
        "guard_ms": guard_ms,
    }.items():
        require_close(
            critical_config.get(key),
            expected_value,
            f"{label}.critical.config.{key}",
            absolute_tolerance=1e-9,
            relative_tolerance=0.0,
        )

    critical_start = strict_int(
        critical.get("measurement_start_monotonic_ns"),
        f"{label}.critical.measurement_start_monotonic_ns",
    )
    critical_end = strict_int(
        critical.get("measurement_end_monotonic_ns"),
        f"{label}.critical.measurement_end_monotonic_ns",
    )
    if (
        critical_start != start_ns
        or critical_end != end_ns
        or not release_ns <= critical_start < critical_end <= collected_ns <= cleanup_ns
    ):
        raise ValueError(f"{label}.critical has inconsistent measurement clocks")
    elapsed = (critical_end - critical_start) / 1_000_000_000.0
    require_close(
        critical.get("elapsed_seconds"),
        elapsed,
        f"{label}.critical.elapsed_seconds",
        absolute_tolerance=5e-10,
    )
    if strict_int(
        critical.get("completed_requests"), f"{label}.critical.completed_requests"
    ) != expected_samples:
        raise ValueError(f"{label}.critical completed-request count differs")
    require_close(
        critical.get("throughput_per_second"),
        expected_samples / elapsed,
        f"{label}.critical.throughput_per_second",
        absolute_tolerance=1e-6,
    )

    for field, values in (
        ("release_to_completion", trace.release_to_completion_ms),
        ("gpu_service", trace.gpu_service_ms),
        ("queue_delay", trace.queue_delay_ms),
        ("gate_overhead", trace.gate_overhead_ms),
        ("drain", trace.drain_ms),
        ("resume", trace.resume_ms),
    ):
        validate_trace_summary(critical.get(field), values, f"{label}.critical.{field}")
    misses = sum(value > deadline_ms for value in trace.release_to_completion_ms)
    if strict_int(
        critical.get("deadline_misses"), f"{label}.critical.deadline_misses"
    ) != misses:
        raise ValueError(f"{label}.critical deadline misses differ from the raw trace")
    require_close(
        critical.get("deadline_miss_rate"),
        misses / expected_samples,
        f"{label}.critical.deadline_miss_rate",
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    return engine_root, critical_pid


def replay_workers(
    epoch: dict[str, Any],
    label: str,
    expected_identities: list[tuple[int, str, str, int]],
    expected_dependency_edges: list[dict[str, Any]],
    config: dict[str, Any],
    mig: dict[str, str],
    artifacts: dict[str, Any],
    engine_root: pathlib.Path,
    critical_pid: int,
    *,
    release_ns: int,
    critical_start_ns: int,
    critical_end_ns: int,
    cleanup_ns: int,
) -> dict[str, Any]:
    workers = epoch.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError(f"{label}.workers must be a non-empty list")
    identities = [
        action_identity(worker, f"{label}.workers[{index}]")
        for index, worker in enumerate(workers)
    ]
    tenant_ids = [identity[0] for identity in identities]
    if len(tenant_ids) != len(set(tenant_ids)):
        raise ValueError(f"{label} contains duplicate worker tenant IDs")
    if sorted(tenant_ids) != list(range(len(workers))):
        raise ValueError(f"{label} worker tenant IDs are not contiguous")
    if identities != expected_identities:
        raise ValueError(f"{label} workers differ from the frozen policy plan")

    action_lists: list[tuple[int, str, str, int]] = []
    for key, placement in (
        ("resident_actions", "resident-1g"),
        ("borrower_actions", "borrower-2g"),
    ):
        actions = epoch.get(key)
        if not isinstance(actions, list):
            raise ValueError(f"{label}.{key} must be a list")
        for index, action in enumerate(actions):
            identity = action_identity(action, f"{label}.{key}[{index}]")
            if identity[2] != placement:
                raise ValueError(f"{label}.{key} has the wrong placement")
            action_lists.append(identity)
    if action_lists != identities:
        raise ValueError(f"{label} worker results differ from placement actions")

    completed_by_placement = dict.fromkeys(PLACEMENTS, 0)
    rate_by_placement = dict.fromkeys(PLACEMENTS, 0.0)
    completed_by_modality = dict.fromkeys(MODALITIES, 0)
    rate_by_modality = dict.fromkeys(MODALITIES, 0.0)
    completed_by_tenant: dict[str, int] = {}
    rate_by_tenant: dict[str, float] = {}
    windows: list[float] = []
    pressure_affinity = config.get("cpu_affinity", {}).get("pressure")
    if not isinstance(pressure_affinity, list) or not pressure_affinity:
        raise ValueError("formal config lacks pressure CPU affinity")
    worker_pids: set[int] = set()
    worker_processes: list[tuple[int, int, int]] = []
    for index, (worker, identity) in enumerate(zip(workers, identities, strict=True)):
        worker_label = f"{label}.workers[{index}]"
        expected_worker_fields = {
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
            "tenant_id",
            "modality",
            "placement",
            "quota_percent",
        }
        if set(worker) != expected_worker_fields:
            raise ValueError(f"{worker_label} has invalid benchmark-result fields")
        if (
            strict_int(worker.get("schema_version"), f"{worker_label}.schema_version")
            != 1
            or worker.get("role") != "pressure"
        ):
            raise ValueError(f"{worker_label} is not a pressure benchmark result")
        completed = strict_int(
            worker.get("completed_requests"),
            f"{worker_label}.completed_requests",
            minimum=1,
        )
        start_ns = strict_int(
            worker.get("measurement_start_monotonic_ns"),
            f"{worker_label}.measurement_start_monotonic_ns",
        )
        end_ns = strict_int(
            worker.get("measurement_end_monotonic_ns"),
            f"{worker_label}.measurement_end_monotonic_ns",
        )
        if not (
            release_ns
            <= start_ns
            <= critical_start_ns
            < critical_end_ns
            <= end_ns
            <= cleanup_ns
        ):
            raise ValueError(f"{worker_label} has an invalid measurement window")
        clock_elapsed = (end_ns - start_ns) / 1_000_000_000.0
        elapsed = strict_float(
            worker.get("elapsed_seconds"),
            f"{worker_label}.elapsed_seconds",
            minimum=0.0,
        )
        if elapsed == 0.0:
            raise ValueError(f"{worker_label}.elapsed_seconds must be positive")
        require_close(
            elapsed,
            clock_elapsed,
            f"{worker_label}.elapsed_seconds",
            absolute_tolerance=5e-10,
        )
        release_summary = worker.get("release_to_completion")
        if not isinstance(release_summary, dict) or strict_int(
            release_summary.get("count"), f"{worker_label}.release_to_completion.count"
        ) != completed:
            raise ValueError(f"{worker_label} completion counts differ")
        rate = completed / elapsed
        require_close(
            worker.get("throughput_per_second"),
            rate,
            f"{worker_label}.throughput_per_second",
            absolute_tolerance=1e-6,
        )
        tenant_id, modality, placement, quota = identity
        model = MODEL_BY_MODALITY[modality]
        if worker.get("model") != model:
            raise ValueError(f"{worker_label}.model differs from modality")
        engine_tag = (
            f"mig-1g-q{quota}"
            if placement == "resident-1g"
            else f"mig-2g-q{quota}"
        )
        artifact_key = (
            f"resident-1g-q{quota}-{model}"
            if placement == "resident-1g"
            else f"borrower-2g-q{quota}-{model}"
        )
        validate_engine_artifact(
            worker.get("engine"),
            engine_root / engine_tag / f"{model}.engine",
            artifact_key,
            artifacts,
            f"{worker_label}.engine",
        )
        expected_device = (
            mig.get("resident_uuid")
            if placement == "resident-1g"
            else mig.get("critical_uuid")
        )
        if not isinstance(expected_device, str) or not expected_device:
            raise ValueError(f"{worker_label} lacks an expected MIG UUID")
        expected_cpu = pressure_affinity[index % len(pressure_affinity)]
        if isinstance(expected_cpu, bool) or not isinstance(expected_cpu, int):
            raise ValueError("formal pressure CPU affinity is invalid")
        worker_pid = validate_execution_environment(
            worker.get("execution_environment"),
            label=f"{worker_label}.execution_environment",
            expected_device=expected_device,
            expected_quota=quota,
            expected_cpus=[expected_cpu],
        )
        if worker_pid == critical_pid or worker_pid in worker_pids:
            raise ValueError(f"{worker_label} reuses an epoch process ID")
        worker_pids.add(worker_pid)
        worker_processes.append((tenant_id, worker_pid, expected_cpu))
        expected_gpu_name = (
            "NVIDIA Thor MIG 1g.0gb"
            if placement == "resident-1g"
            else "NVIDIA Thor MIG 2g.0gb"
        )
        if worker.get("gpu") != {
            "name": expected_gpu_name,
            "multiprocessors": expected_worker_sm_count(placement, quota),
        }:
            raise ValueError(f"{worker_label} differs from the formal MIG width")
        worker_config = worker.get("config")
        expected_config_fields = {
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
        }
        optional_dependency_fields = {
            "dependency_wait_enabled",
            "dependency_signal_enabled",
        }
        if (
            not isinstance(worker_config, dict)
            or not expected_config_fields.issubset(worker_config)
            or not set(worker_config).issubset(
                expected_config_fields | optional_dependency_fields
            )
        ):
            raise ValueError(f"{worker_label}.config has invalid fields")
        expected_wait = any(
            edge["downstream_tenant_id"] == tenant_id
            for edge in expected_dependency_edges
        )
        expected_signal = any(
            edge["upstream_tenant_id"] == tenant_id
            for edge in expected_dependency_edges
        )
        if "dependency_wait_enabled" in worker_config:
            if strict_bool(
                worker_config["dependency_wait_enabled"],
                f"{worker_label}.config.dependency_wait_enabled",
            ) != expected_wait:
                raise ValueError(f"{worker_label} dependency wait differs")
        elif expected_wait:
            raise ValueError(f"{worker_label} lacks dependency wait evidence")
        if "dependency_signal_enabled" in worker_config:
            if strict_bool(
                worker_config["dependency_signal_enabled"],
                f"{worker_label}.config.dependency_signal_enabled",
            ) != expected_signal:
                raise ValueError(f"{worker_label} dependency signal differs")
        elif expected_signal:
            raise ValueError(f"{worker_label} lacks dependency signal evidence")
        expected_integer = {
            "warmup": strict_int(config.get("warmup"), "config.warmup"),
            "burst_size": 1,
            "gated_processes": 0,
            "stopped_processes": 0,
            "stream_priority_value": 0,
        }
        if any(
            strict_int(worker_config.get(key), f"{worker_label}.config.{key}")
            != expected_value
            for key, expected_value in expected_integer.items()
        ):
            raise ValueError(f"{worker_label}.config differs from the formal command")
        for key, expected_value in {
            "period_ms": 0.0,
            "deadline_ms": 0.0,
            "duration_seconds": 3600.0,
            "guard_ms": 0.0,
        }.items():
            require_close(
                worker_config.get(key),
                expected_value,
                f"{worker_label}.config.{key}",
                absolute_tolerance=1e-9,
                relative_tolerance=0.0,
            )
        if (
            strict_bool(
                worker_config.get("start_paused"),
                f"{worker_label}.config.start_paused",
            )
            is not True
            or strict_bool(
                worker_config.get("include_transfers"),
                f"{worker_label}.config.include_transfers",
            )
            is not True
            or worker_config.get("gate_mode") != "stop"
            or worker_config.get("priority")
            != ("default" if placement == "resident-1g" else "low")
        ):
            raise ValueError(f"{worker_label}.config differs from the formal command")
        if strict_int(
            worker.get("deadline_misses"), f"{worker_label}.deadline_misses"
        ) != 0 or worker.get("deadline_miss_rate") is not None:
            raise ValueError(f"{worker_label} has invalid pressure deadline metrics")
        for summary_name in (
            "release_to_completion",
            "gpu_service",
            "queue_delay",
            "gate_overhead",
            "drain",
            "resume",
        ):
            summary = worker.get(summary_name)
            if not isinstance(summary, dict) or strict_int(
                summary.get("count"), f"{worker_label}.{summary_name}.count"
            ) != completed:
                raise ValueError(f"{worker_label}.{summary_name} count differs")
        completed_by_placement[placement] += completed
        rate_by_placement[placement] += rate
        completed_by_modality[modality] += completed
        rate_by_modality[modality] += rate
        completed_by_tenant[str(tenant_id)] = completed
        rate_by_tenant[str(tenant_id)] = rate
        windows.append(elapsed)

    readiness = epoch.get("readiness_affinity")
    if not isinstance(readiness, list) or len(readiness) != len(workers) + 1:
        raise ValueError(f"{label}.readiness_affinity has invalid dimensions")
    for index, (tenant_id, pid, expected_cpu) in enumerate(worker_processes):
        validate_readiness_entry(
            readiness[index],
            label=f"{label}.readiness_affinity[{index}]",
            role="pressure",
            tenant_id=tenant_id,
            pid=pid,
            expected_cpu=expected_cpu,
        )
    critical_affinity = config.get("cpu_affinity", {}).get("critical")
    if not isinstance(critical_affinity, list) or len(critical_affinity) != 1:
        raise ValueError("formal config requires one critical CPU")
    critical_cpu = strict_int(
        critical_affinity[0], "config.cpu_affinity.critical[0]", minimum=0
    )
    validate_readiness_entry(
        readiness[-1],
        label=f"{label}.readiness_affinity[{len(readiness) - 1}]",
        role="critical",
        pid=critical_pid,
        expected_cpu=critical_cpu,
    )

    offered = epoch.get("offered_modalities")
    if not isinstance(offered, list) or any(value not in MODALITIES for value in offered):
        raise ValueError(f"{label}.offered_modalities is invalid")
    offered_tenants = strict_int(epoch.get("offered_tenants"), f"{label}.offered_tenants")
    if offered_tenants != len(offered):
        raise ValueError(f"{label}.offered_tenants differs from offered modalities")
    active = len(workers)
    resident_workers = sum(identity[2] == "resident-1g" for identity in identities)
    borrower_workers = active - resident_workers
    expected_integers = {
        "active_workers": active,
        "resident_workers": resident_workers,
        "borrower_workers": borrower_workers,
        "rejected_tenants": offered_tenants - active,
        "resident_completed": completed_by_placement["resident-1g"],
        "borrower_completed": completed_by_placement["borrower-2g"],
        "pressure_completed": sum(completed_by_placement.values()),
    }
    if expected_integers["rejected_tenants"] < 0:
        raise ValueError(f"{label} admitted more workers than offered tenants")
    for key, expected in expected_integers.items():
        if strict_int(epoch.get(key), f"{label}.{key}") != expected:
            raise ValueError(f"{label}.{key} differs from worker replay")

    expected_rates = {
        "resident_goodput_per_second": rate_by_placement["resident-1g"],
        "borrower_goodput_per_second": rate_by_placement["borrower-2g"],
        "pressure_goodput_per_second": sum(rate_by_placement.values()),
    }
    for key, expected in expected_rates.items():
        require_close(epoch.get(key), expected, f"{label}.{key}", absolute_tolerance=1e-6)
    for key, expected in (
        ("completed_by_modality", completed_by_modality),
        ("completed_by_tenant", completed_by_tenant),
    ):
        if epoch.get(key) != expected:
            raise ValueError(f"{label}.{key} differs from worker replay")
    for key, expected in (
        ("goodput_by_modality", rate_by_modality),
        ("goodput_by_tenant", rate_by_tenant),
    ):
        actual = epoch.get(key)
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label}.{key} has invalid dimensions")
        for name, value in expected.items():
            require_close(
                actual[name], value, f"{label}.{key}.{name}", absolute_tolerance=1e-6
            )

    worker_window = statistics.median(windows)
    window_spread = max(windows) - min(windows)
    require_close(
        epoch.get("worker_window_seconds"),
        worker_window,
        f"{label}.worker_window_seconds",
    )
    require_close(
        epoch.get("worker_window_spread_seconds"),
        window_spread,
        f"{label}.worker_window_spread_seconds",
    )
    return {
        **expected_integers,
        **expected_rates,
        "completed_by_modality": completed_by_modality,
        "goodput_by_modality": rate_by_modality,
        "worker_window_seconds": worker_window,
    }


def formal_offered_modalities(
    epoch_index: int, scenario: str = "legacy"
) -> tuple[str, ...]:
    try:
        trace = FORMAL_SCENARIO_TRACES[scenario]
    except KeyError as error:
        raise ValueError(f"unknown formal scenario: {scenario}") from error
    offered = trace[epoch_index % len(trace)]
    cycle = epoch_index // len(trace)
    if scenario == "legacy" and cycle % 2 == 1 and len(offered) > 1:
        return offered[1:] + offered[:1]
    return offered


def validated_feedback_state(value: object, label: str) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a feedback state")
    expected_fields = {
        "resident_admission_limit",
        "resident_quota_index",
        "borrower_limit",
        "guard_adjustment_ms",
        "safe_epochs",
        "resident_quota_percent",
    }
    if set(value) != expected_fields:
        raise ValueError(f"{label} has invalid feedback state fields")
    admission = strict_int(
        value["resident_admission_limit"],
        f"{label}.resident_admission_limit",
        minimum=1,
    )
    quota_index = strict_int(
        value["resident_quota_index"], f"{label}.resident_quota_index"
    )
    borrower_limit = strict_int(
        value["borrower_limit"], f"{label}.borrower_limit"
    )
    safe_epochs = strict_int(value["safe_epochs"], f"{label}.safe_epochs")
    guard = strict_float(
        value["guard_adjustment_ms"],
        f"{label}.guard_adjustment_ms",
        minimum=0.0,
    )
    if quota_index >= len(RESIDENT_QUOTAS):
        raise ValueError(f"{label}.resident_quota_index is invalid")
    quota = strict_int(
        value["resident_quota_percent"], f"{label}.resident_quota_percent"
    )
    if quota != RESIDENT_QUOTAS[quota_index]:
        raise ValueError(f"{label}.resident_quota_percent differs from index")
    if admission > 6 or borrower_limit > 6 or guard != 0.0:
        raise ValueError(f"{label} exceeds formal controller bounds")
    return {
        "resident_admission_limit": admission,
        "resident_quota_index": quota_index,
        "borrower_limit": borrower_limit,
        "guard_adjustment_ms": guard,
        "safe_epochs": safe_epochs,
        "resident_quota_percent": quota,
    }


def default_feedback_state() -> dict[str, int | float]:
    return {
        "resident_admission_limit": 6,
        "resident_quota_index": len(RESIDENT_QUOTAS) - 1,
        "borrower_limit": 6,
        "guard_adjustment_ms": 0.0,
        "safe_epochs": 0,
        "resident_quota_percent": RESIDENT_QUOTAS[-1],
    }


def expected_policy_identities(
    policy_name: str,
    offered: tuple[str, ...],
    state: dict[str, int | float],
    borrower_quota: int,
) -> list[tuple[int, str, str, int]]:
    admitted = min(len(offered), int(state["resident_admission_limit"]))
    resident_quota = 100
    borrower_count = 0
    if policy_name == "same-mig":
        borrower_count = admitted
    elif policy_name in {
        "uncoordinated-borrow",
        "fixed-borrow",
        "fixed-full-gate",
    }:
        borrower_count = admitted // 2
    elif policy_name == "mig-governor":
        resident_quota = int(state["resident_quota_percent"])
        borrower_count = min(admitted // 2, int(state["borrower_limit"]))
    elif policy_name not in {"static-mig", "resident-full-gate"}:
        raise ValueError(f"unknown policy: {policy_name}")
    resident_count = admitted - borrower_count
    identities = [
        (index, offered[index], "resident-1g", resident_quota)
        for index in range(resident_count)
    ]
    identities.extend(
        (index, offered[index], "borrower-2g", borrower_quota)
        for index in range(resident_count, admitted)
    )
    return identities


def expected_gate_scope(policy_name: str) -> list[str]:
    if policy_name == "fixed-borrow":
        return ["borrower-2g"]
    if policy_name == "resident-full-gate":
        return ["resident-1g"]
    if policy_name in {"fixed-full-gate", "mig-governor"}:
        return ["borrower-2g", "resident-1g"]
    return []


def expected_guard_ms(
    policy_name: str,
    identities: list[tuple[int, str, str, int]],
    guard_profile: dict[str, dict[str, dict[str, float]]],
) -> float:
    scope = set(expected_gate_scope(policy_name))
    gated = [identity for identity in identities if identity[2] in scope]
    if not gated:
        return 0.0
    per_placement: dict[str, float] = {}
    for _tenant, modality, placement, quota in gated:
        try:
            guard_ms = strict_float(
                guard_profile[placement][str(quota)][modality],
                f"guard profile {placement}/q{quota}/{modality}",
                minimum=0.0,
            )
        except KeyError as error:
            raise ValueError(
                f"guard profile lacks {placement}/q{quota}/{modality}"
            ) from error
        if guard_ms <= 0.0:
            raise ValueError("guard profile contains a non-positive guard")
        per_placement[placement] = per_placement.get(placement, 0.0) + float(
            guard_ms
        )
    return max(per_placement.values())


def adaptive_action_differs_from_fixed_full(
    offered: tuple[str, ...],
    governor_state: dict[str, int | float],
    borrower_quota: int,
    guard_profile: dict[str, dict[str, dict[str, float]]],
) -> bool:
    governor_identities = expected_policy_identities(
        "mig-governor", offered, governor_state, borrower_quota
    )
    fixed_identities = expected_policy_identities(
        ADAPTIVE_BASELINE, offered, default_feedback_state(), borrower_quota
    )
    if governor_identities != fixed_identities:
        return True
    governor_guard = expected_guard_ms(
        "mig-governor", governor_identities, guard_profile
    )
    fixed_guard = expected_guard_ms(
        ADAPTIVE_BASELINE, fixed_identities, guard_profile
    )
    return not math.isclose(
        governor_guard, fixed_guard, rel_tol=0.0, abs_tol=1e-12
    )


def adaptive_claim_status(
    *,
    adaptive_action_epochs: int,
    adaptive_action_runs: int,
    total_runs: int,
    drift_valid: bool,
    governor_feasible: bool,
    baseline_admission_valid: bool,
    baseline_telemetry_valid: bool,
    baseline_slo_feasible: bool,
    paired_gain_supported: bool,
) -> str:
    if adaptive_action_epochs == 0:
        return "not-exercised"
    if adaptive_action_runs < total_runs:
        return "partially-exercised"
    if not drift_valid or not baseline_admission_valid or not baseline_telemetry_valid:
        return "not-evaluable"
    if not governor_feasible:
        return "not-supported"
    if not baseline_slo_feasible:
        return "protection-supported"
    if paired_gain_supported:
        return "goodput-gain-supported"
    return "no-incremental-benefit"


def replay_governor_transition(
    state_before: dict[str, int | float],
    *,
    telemetry_unhealthy: bool,
    violated: bool,
    critical_p99_ms: float,
    deadline_ms: float,
    drain_near_overrun: bool,
    thermal_high: bool,
) -> tuple[dict[str, int | float], str]:
    state = dict(state_before)
    if telemetry_unhealthy:
        state["resident_admission_limit"] = 1
        state["resident_quota_index"] = 0
        state["borrower_limit"] = 0
        state["safe_epochs"] = 0
        action = "telemetry-fail-closed"
    elif drain_near_overrun:
        state["borrower_limit"] = max(0, int(state["borrower_limit"]) - 1)
        state["resident_admission_limit"] = max(
            1, int(state["resident_admission_limit"]) - 1
        )
        state["safe_epochs"] = 0
        action = "drain-reclaim"
    elif violated or thermal_high:
        state["borrower_limit"] = max(0, int(state["borrower_limit"]) - 1)
        if int(state["resident_quota_index"]) > 0:
            state["resident_quota_index"] = int(state["resident_quota_index"]) - 1
        else:
            state["resident_admission_limit"] = max(
                1, int(state["resident_admission_limit"]) - 1
            )
        state["safe_epochs"] = 0
        action = "residual-reclaim"
    elif critical_p99_ms >= deadline_ms * 0.90:
        state["safe_epochs"] = 0
        action = "hold-near-deadline"
    else:
        state["safe_epochs"] = int(state["safe_epochs"]) + 1
        if int(state["safe_epochs"]) < 3:
            action = "hold-hysteresis"
        else:
            if int(state["resident_quota_index"]) < len(RESIDENT_QUOTAS) - 1:
                state["resident_quota_index"] = int(state["resident_quota_index"]) + 1
                action = "recover-resident-quota"
            elif int(state["resident_admission_limit"]) < 6:
                state["resident_admission_limit"] = int(
                    state["resident_admission_limit"]
                ) + 1
                action = "recover-admission"
            elif int(state["borrower_limit"]) < 6:
                state["borrower_limit"] = int(state["borrower_limit"]) + 1
                action = "recover-borrower"
            else:
                action = "hold-full-capacity"
            state["safe_epochs"] = 0
    state["resident_quota_percent"] = RESIDENT_QUOTAS[
        int(state["resident_quota_index"])
    ]
    return state, action


def recompute_policy_metrics(
    policy: dict[str, Any],
    input_path: pathlib.Path,
    config: dict[str, Any],
    deadline_ms: float,
    claimed_traces: RawTraceClaims,
    mig: dict[str, str],
    artifacts: dict[str, Any],
    guard_profile: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict[str, float], list[float]]:
    if not isinstance(mig, dict) or not isinstance(artifacts, dict):
        raise ValueError("formal run lacks MIG or artifact provenance")
    name = policy.get("name")
    if name not in POLICIES:
        raise ValueError(f"invalid policy name: {name}")
    expected_epochs = strict_int(config.get("epochs"), "config.epochs", minimum=1)
    expected_samples = strict_int(
        config.get("samples_per_epoch"), "config.samples_per_epoch", minimum=1
    )
    burst_size = strict_int(config.get("burst_size"), "config.burst_size", minimum=1)
    dmr_target = strict_float(config.get("dmr_target"), "config.dmr_target", minimum=0.0)
    epochs = policy.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != expected_epochs:
        raise ValueError(f"{name} has an invalid epoch count")
    if [strict_int(epoch.get("epoch"), f"{name}.epoch") for epoch in epochs] != list(
        range(expected_epochs)
    ):
        raise ValueError(f"epoch sequence is invalid for {name}")

    all_latencies: list[float] = []
    epoch_metrics: list[dict[str, Any]] = []
    default_state = default_feedback_state()
    previous_state_after: dict[str, int | float] | None = None
    borrower_quota = strict_int(
        config.get("borrower_quota", 100), "config.borrower_quota", minimum=1
    )
    scenario = str(config.get("scenario", "legacy"))
    for epoch_index, epoch in enumerate(epochs):
        label = f"{name}.epochs[{epoch_index}]"
        offered = formal_offered_modalities(
            epoch_index, str(config.get("scenario", "legacy"))
        )
        if epoch.get("offered_modalities") != list(offered):
            raise ValueError(f"{label}.offered_modalities differs from frozen trace")
        state_before = validated_feedback_state(
            epoch.get("state_before"), f"{label}.state_before"
        )
        state_after = validated_feedback_state(
            epoch.get("state_after"), f"{label}.state_after"
        )
        expected_before = default_state if previous_state_after is None else previous_state_after
        if state_before != expected_before:
            raise ValueError(f"{label}.state_before breaks controller continuity")
        if name != "mig-governor":
            if state_after != default_state or epoch.get("controller_action") != "not-applicable":
                raise ValueError(f"{label} changed state outside the governor")
        expected_identities = expected_policy_identities(
            str(name), offered, state_before, borrower_quota
        )
        admitted_tenants = {identity[0] for identity in expected_identities}
        expected_dependency_edges = (
            [
                {
                    "upstream_tenant_id": tenant_id - 1,
                    "downstream_tenant_id": tenant_id,
                    "semantics": "completion-token-before-next-inference",
                }
                for tenant_id, modality in enumerate(offered)
                if scenario == "dependent"
                and modality == "language"
                and tenant_id > 0
                and offered[tenant_id - 1] == "audio"
                and tenant_id - 1 in admitted_tenants
                and tenant_id in admitted_tenants
            ]
            if scenario == "dependent"
            else []
        )
        if epoch.get("dependency_edges", []) != expected_dependency_edges:
            raise ValueError(f"{label}.dependency_edges differs from scenario")
        trace_path = input_path.parent / "raw" / f"{name}-e{epoch_index}.csv"
        trace_snapshot = claim_raw_trace(trace_path, claimed_traces)
        trace = read_critical_trace(trace_path, expected_samples)
        verify_raw_trace_snapshot(trace_path, trace_snapshot)
        latencies = trace.release_to_completion_ms
        misses = sum(value > deadline_ms for value in latencies)
        miss_rate = misses / expected_samples
        p99 = percentile(latencies, 0.99)
        gate_mean = statistics.fmean(trace.gate_overhead_ms)
        drain_p99 = percentile(trace.drain_ms, 0.99)
        drain_max = max(trace.drain_ms)
        resume_p99 = percentile(trace.resume_ms, 0.99)
        scope = expected_gate_scope(str(name))
        if epoch.get("gate_scope") != scope:
            raise ValueError(f"{label}.gate_scope differs from policy")
        gated_workers = sum(identity[2] in scope for identity in expected_identities)
        if strict_int(epoch.get("gated_workers"), f"{label}.gated_workers") != gated_workers:
            raise ValueError(f"{label}.gated_workers differs from policy plan")
        guard_ms = expected_guard_ms(
            str(name), expected_identities, guard_profile
        )
        adaptive_action_diff = name == "mig-governor" and (
            adaptive_action_differs_from_fixed_full(
                offered, state_before, borrower_quota, guard_profile
            )
        )
        require_close(epoch.get("guard_ms"), guard_ms, f"{label}.guard_ms")
        guard_utilization = drain_max / guard_ms if guard_ms > 0.0 else 0.0
        drain_near_overrun = guard_ms > 0.0 and guard_utilization >= 0.8
        require_close(
            epoch.get("guard_utilization"),
            guard_utilization,
            f"{label}.guard_utilization",
        )
        if strict_bool(
            epoch.get("drain_near_overrun"), f"{label}.drain_near_overrun"
        ) != drain_near_overrun:
            raise ValueError(f"{label}.drain_near_overrun differs from raw trace")
        require_close(epoch.get("drain_p99_ms"), drain_p99, f"{label}.drain_p99_ms")
        require_close(epoch.get("drain_max_ms"), drain_max, f"{label}.drain_max_ms")
        require_close(epoch.get("resume_p99_ms"), resume_p99, f"{label}.resume_p99_ms")

        start_ns = strict_int(
            epoch.get("measurement_start_monotonic_ns"),
            f"{label}.measurement_start_monotonic_ns",
        )
        end_ns = strict_int(
            epoch.get("measurement_end_monotonic_ns"),
            f"{label}.measurement_end_monotonic_ns",
        )
        release_ns = strict_int(
            epoch.get("measurement_release_monotonic_ns"),
            f"{label}.measurement_release_monotonic_ns",
        )
        collected_ns = strict_int(
            epoch.get("result_collected_monotonic_ns"),
            f"{label}.result_collected_monotonic_ns",
        )
        cleanup_ns = strict_int(
            epoch.get("cleanup_end_monotonic_ns"),
            f"{label}.cleanup_end_monotonic_ns",
        )
        if not release_ns <= start_ns < end_ns <= collected_ns <= cleanup_ns:
            raise ValueError(f"{label} has inconsistent measurement clocks")
        engine_root, critical_pid = replay_critical_result(
            epoch,
            label,
            trace,
            config,
            deadline_ms,
            mig,
            artifacts,
            expected_samples=expected_samples,
            expected_workers=len(expected_identities),
            gated_workers=gated_workers,
            guard_ms=guard_ms,
            release_ns=release_ns,
            start_ns=start_ns,
            end_ns=end_ns,
            collected_ns=collected_ns,
            cleanup_ns=cleanup_ns,
        )
        measurement_seconds = (end_ns - start_ns) / 1_000_000_000.0
        require_close(
            epoch.get("measurement_seconds"),
            measurement_seconds,
            f"{label}.measurement_seconds",
            absolute_tolerance=5e-10,
        )
        duty = sum(trace.gpu_service_ms) / (measurement_seconds * 1000.0)
        violated = p99 > deadline_ms or miss_rate > dmr_target
        if strict_int(epoch.get("deadline_misses"), f"{label}.deadline_misses") != misses:
            raise ValueError(f"{label}.deadline_misses differs from raw trace")
        require_close(
            epoch.get("deadline_miss_rate"),
            miss_rate,
            f"{label}.deadline_miss_rate",
            absolute_tolerance=1e-12,
            relative_tolerance=1e-12,
        )
        require_close(epoch.get("critical_p99_ms"), p99, f"{label}.critical_p99_ms")
        require_close(
            epoch.get("critical_p50_ms"),
            percentile(latencies, 0.50),
            f"{label}.critical_p50_ms",
        )
        require_close(
            epoch.get("critical_p999_ms"),
            percentile(latencies, 0.999),
            f"{label}.critical_p999_ms",
        )
        require_close(
            epoch.get("critical_max_ms"), max(latencies), f"{label}.critical_max_ms"
        )
        require_close(
            epoch.get("queue_delay_p99_ms"),
            percentile(trace.queue_delay_ms, 0.99),
            f"{label}.queue_delay_p99_ms",
        )
        require_close(
            epoch.get("gate_overhead_mean_ms"),
            gate_mean,
            f"{label}.gate_overhead_mean_ms",
        )
        require_close(
            epoch.get("critical_gpu_duty_cycle"),
            duty,
            f"{label}.critical_gpu_duty_cycle",
            absolute_tolerance=1e-6,
        )
        if strict_bool(epoch.get("violated"), f"{label}.violated") != violated:
            raise ValueError(f"{label}.violated differs from raw trace")

        telemetry = epoch.get("telemetry")
        if not isinstance(telemetry, dict) or not isinstance(
            telemetry.get("health"), dict
        ):
            raise ValueError(f"{label}.telemetry lacks health provenance")
        healthy = strict_bool(
            telemetry["health"].get("healthy"), f"{label}.telemetry.health.healthy"
        )
        telemetry_unhealthy = strict_bool(
            epoch.get("telemetry_unhealthy"), f"{label}.telemetry_unhealthy"
        )
        if telemetry_unhealthy == healthy:
            raise ValueError(f"{label}.telemetry_unhealthy contradicts telemetry health")
        thermal_high = strict_bool(epoch.get("thermal_high"), f"{label}.thermal_high")

        if name == "mig-governor":
            expected_after, expected_action = replay_governor_transition(
                state_before,
                telemetry_unhealthy=telemetry_unhealthy,
                violated=violated,
                critical_p99_ms=p99,
                deadline_ms=deadline_ms,
                drain_near_overrun=drain_near_overrun,
                thermal_high=thermal_high,
            )
            if state_after != expected_after:
                raise ValueError(f"{label}.state_after differs from controller replay")
            if epoch.get("controller_action") != expected_action:
                raise ValueError(f"{label}.controller_action differs from replay")

        workers = replay_workers(
            epoch,
            label,
            expected_identities,
            expected_dependency_edges,
            config,
            mig,
            artifacts,
            engine_root,
            critical_pid,
            release_ns=release_ns,
            critical_start_ns=start_ns,
            critical_end_ns=end_ns,
            cleanup_ns=cleanup_ns,
        )
        wall_elapsed = strict_float(
            epoch.get("wall_elapsed_seconds"),
            f"{label}.wall_elapsed_seconds",
            minimum=measurement_seconds,
        )
        epoch_metrics.append(
            {
                **workers,
                "deadline_misses": misses,
                "deadline_miss_rate": miss_rate,
                "critical_p99_ms": p99,
                "violated": violated,
                "telemetry_unhealthy": telemetry_unhealthy,
                "gate_overhead_mean_ms": gate_mean,
                "critical_gpu_duty_cycle": duty,
                "measurement_seconds": measurement_seconds,
                "wall_elapsed_seconds": wall_elapsed,
                "adaptive_action_diff": adaptive_action_diff,
            }
        )
        all_latencies.extend(latencies)
        previous_state_after = state_after

    total_measurement_seconds = sum(
        epoch["measurement_seconds"] for epoch in epoch_metrics
    )
    if total_measurement_seconds <= 0.0:
        raise ValueError(f"{name} has a non-positive aggregate measurement window")

    def time_weighted(key: str) -> float:
        return sum(
            float(epoch[key]) * float(epoch["measurement_seconds"])
            for epoch in epoch_metrics
        ) / total_measurement_seconds

    deadline_misses = sum(epoch["deadline_misses"] for epoch in epoch_metrics)
    critical_requests = expected_epochs * expected_samples
    rejected_tenants = sum(epoch["rejected_tenants"] for epoch in epoch_metrics)
    telemetry_unhealthy_epochs = sum(
        epoch["telemetry_unhealthy"] for epoch in epoch_metrics
    )
    adaptive_action_epochs = sum(
        epoch["adaptive_action_diff"] for epoch in epoch_metrics
    )
    completed = {
        key: sum(epoch[key] for epoch in epoch_metrics)
        for key in ("resident_completed", "borrower_completed", "pressure_completed")
    }
    rates = {
        key: time_weighted(key)
        for key in (
            "resident_goodput_per_second",
            "borrower_goodput_per_second",
            "pressure_goodput_per_second",
        )
    }
    modality_rates = {
        modality: sum(
            epoch["goodput_by_modality"][modality]
            * epoch["measurement_seconds"]
            for epoch in epoch_metrics
        )
        / total_measurement_seconds
        for modality in MODALITIES
    }
    replay = {
        "critical_requests": critical_requests,
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": deadline_misses / critical_requests,
        "violation_epoch_rate": sum(epoch["violated"] for epoch in epoch_metrics)
        / expected_epochs,
        "critical_p99_ms_max": max(
            epoch["critical_p99_ms"] for epoch in epoch_metrics
        ),
        **completed,
        **rates,
        "goodput_by_modality": modality_rates,
        "rejected_tenants": rejected_tenants,
        "telemetry_unhealthy_epochs": telemetry_unhealthy_epochs,
        "gate_overhead_mean_ms": statistics.fmean(
            epoch["gate_overhead_mean_ms"] for epoch in epoch_metrics
        ),
        "critical_gpu_duty_cycle_mean": statistics.fmean(
            epoch["critical_gpu_duty_cycle"] for epoch in epoch_metrics
        ),
        "measurement_seconds": total_measurement_seconds,
        "worker_window_seconds": sum(
            epoch["worker_window_seconds"] for epoch in epoch_metrics
        ),
        "wall_elapsed_seconds": sum(
            epoch["wall_elapsed_seconds"] for epoch in epoch_metrics
        ),
    }
    integer_fields = (
        "critical_requests",
        "deadline_misses",
        "resident_completed",
        "borrower_completed",
        "pressure_completed",
        "rejected_tenants",
        "telemetry_unhealthy_epochs",
    )
    for key in integer_fields:
        if strict_int(policy.get(key), f"{name}.{key}") != replay[key]:
            raise ValueError(f"{name}.{key} differs from epoch/raw replay")
    float_tolerances = {
        "deadline_miss_rate": 1e-12,
        "violation_epoch_rate": 1e-12,
        "critical_p99_ms_max": 1e-7,
        "resident_goodput_per_second": 1e-6,
        "borrower_goodput_per_second": 1e-6,
        "pressure_goodput_per_second": 1e-6,
        "gate_overhead_mean_ms": 1e-7,
        "critical_gpu_duty_cycle_mean": 1e-6,
        "measurement_seconds": 1e-8,
        "worker_window_seconds": 1e-8,
        "wall_elapsed_seconds": 1e-8,
    }
    for key, tolerance in float_tolerances.items():
        require_close(
            policy.get(key),
            float(replay[key]),
            f"{name}.{key}",
            absolute_tolerance=tolerance,
        )
    actual_modalities = policy.get("goodput_by_modality")
    if not isinstance(actual_modalities, dict) or set(actual_modalities) != set(
        modality_rates
    ):
        raise ValueError(f"{name}.goodput_by_modality has invalid dimensions")
    for modality, value in modality_rates.items():
        require_close(
            actual_modalities[modality],
            value,
            f"{name}.goodput_by_modality.{modality}",
            absolute_tolerance=1e-6,
        )

    gate_mean = float(replay["gate_overhead_mean_ms"])
    metrics = {
        "deadline_miss_rate": float(replay["deadline_miss_rate"]),
        "violation_epoch_rate": float(replay["violation_epoch_rate"]),
        "critical_p99_ms_max": float(replay["critical_p99_ms_max"]),
        "critical_requests": float(critical_requests),
        "deadline_misses": float(deadline_misses),
        "resident_goodput_per_second": rates["resident_goodput_per_second"],
        "borrower_goodput_per_second": rates["borrower_goodput_per_second"],
        "pressure_goodput_per_second": rates["pressure_goodput_per_second"],
        "language_goodput_per_second": modality_rates["language"],
        "audio_goodput_per_second": modality_rates["audio"],
        "rejected_tenants": float(rejected_tenants),
        "telemetry_unhealthy_epochs": float(telemetry_unhealthy_epochs),
        "adaptive_action_epochs": float(adaptive_action_epochs),
        "slo_feasible": float(float(replay["deadline_miss_rate"]) <= dmr_target),
        "admission_feasible": float(rejected_tenants == 0),
        "telemetry_healthy": float(telemetry_unhealthy_epochs == 0),
        "critical_gpu_duty_cycle_mean": float(
            replay["critical_gpu_duty_cycle_mean"]
        ),
        "gate_overhead_amortized_per_request_ms": gate_mean,
        "gate_overhead_per_burst_ms": gate_mean * burst_size,
    }
    return metrics, all_latencies


FORMAL_TRACE = (
    ("language",),
    ("audio",),
    ("language", "audio"),
    ("language", "audio", "language", "audio"),
    ("audio", "audio", "audio", "audio", "audio", "audio"),
    ("language", "language", "language", "language", "language", "language"),
)
FORMAL_MULTIMODAL_SCENARIO_TRACE = (
    ("audio", "language"),
    ("audio", "language", "audio", "language"),
    ("audio", "language", "audio", "language", "audio", "language"),
    ("audio", "language"),
    ("audio", "language", "audio", "language"),
    ("audio", "language", "audio", "language", "audio", "language"),
)
FORMAL_SCENARIO_TRACES = {
    "legacy": FORMAL_TRACE,
    "independent": FORMAL_MULTIMODAL_SCENARIO_TRACE,
    "dependent": FORMAL_MULTIMODAL_SCENARIO_TRACE,
}
FORMAL_CPU_AFFINITY = {
    "pressure": list(range(11)),
    "mps": [11],
    "critical": [12],
    "telemetry": [13],
}
FORMAL_MAX_ISOLATED_DRIFT_FRACTION = 0.05
FORMAL_TELEMETRY_REQUIRED_FIELDS = (
    "ram",
    "mem_available",
    "cpu",
    "temperature:soc012",
    "temperature:tj",
    "power:VIN",
)
FORMAL_THERMAL_STABILITY_SENSOR = "soc012"
FORMAL_THERMAL_SAFETY_SENSOR = "tj"
FORMAL_THERMAL_HANDOFF_MAX_MS = 500.0
FORMAL_THERMAL_CALIBRATION_PRECONDITIONING = (
    "per-repeat-preloaded-critical"
)


def validate_formal_protocol(
    config: dict[str, Any],
    guard_profile: dict[str, dict[str, dict[str, float]]],
) -> None:
    expected_integers = {
        "epochs": 36,
        "samples_per_epoch": 800,
        "warmup": 100,
        "burst_size": 8,
        "borrower_quota": 100,
        "calibration_repeats": 3,
    }
    for key, expected in expected_integers.items():
        if strict_int(config.get(key), f"config.{key}") != expected:
            raise ValueError(f"formal protocol requires {key}={expected}")
    expected_floats = {
        "period_ms": 20.0,
        "dmr_target": 0.0005,
        "max_isolated_drift_fraction": FORMAL_MAX_ISOLATED_DRIFT_FRACTION,
        "thermal_window_seconds": 60.0,
        "thermal_timeout_seconds": 900.0,
        "thermal_stability_checkpoint_seconds": 30.0,
        "thermal_stability_checkpoint_max_lateness_seconds": 1.0,
        "thermal_required_stable_checkpoints": 3.0,
        "tegrastats_requested_interval_ms": 75.0,
        "telemetry_interval_ms": 100.0,
        "telemetry_required_fraction": 0.8,
        "telemetry_stale_after_ms": 300.0,
        "telemetry_max_gap_ms": 300.0,
        "thermal_handoff_max_ms": FORMAL_THERMAL_HANDOFF_MAX_MS,
        "thermal_active_stable_spacing_seconds": (
            THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        ),
    }
    for key, expected in expected_floats.items():
        require_close(
            config.get(key),
            expected,
            f"config.{key}",
            absolute_tolerance=1e-12,
            relative_tolerance=0.0,
        )
    require_close(
        config.get("pressure_rps_per_tenant", 0.0),
        0.0,
        "config.pressure_rps_per_tenant",
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    scenario = config.get("scenario", "legacy")
    if scenario not in FORMAL_SCENARIO_TRACES:
        raise ValueError("formal protocol requires a supported workload scenario")
    expected_trace = [
        list(epoch) for epoch in FORMAL_SCENARIO_TRACES[scenario]
    ]
    if config.get("trace") != expected_trace:
        raise ValueError("formal protocol requires the frozen six-epoch trace")
    expected_assignment = (
        "rotate-left-one-on-odd-six-epoch-cycle"
        if scenario == "legacy"
        else "fixed-audio-language-pairs-2-4-6-repeat"
    )
    if config.get("trace_assignment") != expected_assignment:
        raise ValueError("formal protocol requires the frozen trace assignment")
    if config.get("cpu_affinity") != FORMAL_CPU_AFFINITY:
        raise ValueError("formal protocol requires the frozen CPU affinity mapping")
    if config.get("telemetry_source") != "tegrastats-readall-monotonic-jsonl":
        raise ValueError("formal protocol requires monotonic tegrastats JSONL")
    if config.get("telemetry_required_fields") != list(
        FORMAL_TELEMETRY_REQUIRED_FIELDS
    ):
        raise ValueError("formal protocol requires the frozen telemetry fields")
    if (
        config.get("thermal_stability_sensor")
        != FORMAL_THERMAL_STABILITY_SENSOR
        or config.get("thermal_safety_sensor") != FORMAL_THERMAL_SAFETY_SENSOR
        or config.get("thermal_handoff_boundary") != THERMAL_HANDOFF_BOUNDARY
        or config.get("thermal_qualification_max_attempts")
        != THERMAL_QUALIFICATION_MAX_ATTEMPTS
        or config.get("thermal_active_stable_endpoints")
        != THERMAL_ACTIVE_STABLE_ENDPOINTS
        or "thermal_qualification_dwell_seconds" in config
    ):
        raise ValueError("formal protocol requires the frozen thermal semantics")
    if (
        config.get("thermal_calibration_preconditioning")
        != FORMAL_THERMAL_CALIBRATION_PRECONDITIONING
    ):
        raise ValueError(
            "formal protocol requires per-repeat preloaded-critical calibration"
        )
    if config.get("guard_override_ms") is not None:
        raise ValueError("formal protocol forbids a guard override")
    if config.get("guard_profile_source") != "frozen-quota-aware-lock":
        raise ValueError("formal protocol requires a frozen quota-aware guard profile")
    require_json_equivalent(
        config.get("profile_guard_ms"),
        guard_profile,
        "config.profile_guard_ms",
    )


@dataclasses.dataclass(frozen=True)
class TelemetryEvidence:
    path: pathlib.Path
    sha256: str
    samples: tuple[TelemetrySample, ...]
    markers: tuple[TelemetryMarker, ...]


def json_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_normalized(item) for item in value]
    return value


def require_json_equivalent(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} has different fields from telemetry replay")
        for key, value in expected.items():
            require_json_equivalent(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} has different dimensions from telemetry replay")
        for index, value in enumerate(expected):
            require_json_equivalent(actual[index], value, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"{label} differs from telemetry replay")
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or actual != expected:
            raise ValueError(f"{label} differs from telemetry replay")
        return
    if isinstance(expected, float):
        require_close(
            actual,
            expected,
            label,
            absolute_tolerance=1e-9,
            relative_tolerance=1e-12,
        )
        return
    raise ValueError(f"{label} contains an unsupported telemetry value")


def load_telemetry_evidence(path: pathlib.Path) -> TelemetryEvidence:
    if not path.is_file():
        raise ValueError(f"missing telemetry JSONL: {path}")
    digest = hashlib.sha256()
    samples: list[TelemetrySample] = []
    markers: list[TelemetryMarker] = []
    last_timestamp: int | None = None
    last_sample_timestamp: int | None = None
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"unterminated telemetry JSONL line {line_number}")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid telemetry JSONL line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict) or record.get("schema_version") != 1:
                raise ValueError(f"invalid telemetry record at line {line_number}")
            timestamp = strict_int(
                record.get("monotonic_ns"),
                f"telemetry line {line_number}.monotonic_ns",
            )
            if last_timestamp is not None and timestamp <= last_timestamp:
                raise ValueError(
                    "telemetry JSONL timestamps must be strictly increasing"
                )
            last_timestamp = timestamp
            record_type = record.get("record_type")
            if record_type == "sample":
                expected_fields = {
                    "schema_version",
                    "record_type",
                    "monotonic_ns",
                    "raw",
                    "parsed",
                    "mem_available_mb",
                    "collection_errors",
                }
                if set(record) != expected_fields or not isinstance(record["raw"], str):
                    raise ValueError(f"invalid sample record at line {line_number}")
                parsed = parse_tegrastats_line(record["raw"])
                if record["parsed"] != json_normalized(parsed.to_dict()):
                    raise ValueError(
                        f"sample parsed payload differs from raw line {line_number}"
                    )
                errors = record["collection_errors"]
                if not isinstance(errors, list) or any(
                    not isinstance(error, str) for error in errors
                ):
                    raise ValueError(
                        f"invalid collection errors at telemetry line {line_number}"
                    )
                samples.append(
                    TelemetrySample(
                        timestamp,
                        record["raw"],
                        parsed,
                        record["mem_available_mb"],
                        tuple(errors),
                    )
                )
                if (
                    last_sample_timestamp is not None
                    and timestamp <= last_sample_timestamp
                ):
                    raise ValueError(
                        "telemetry sample timestamps must be strictly increasing"
                    )
                last_sample_timestamp = timestamp
            elif record_type == "marker":
                expected_fields = {
                    "schema_version",
                    "record_type",
                    "monotonic_ns",
                    "name",
                    "metadata",
                }
                if (
                    set(record) != expected_fields
                    or not isinstance(record["name"], str)
                    or not isinstance(record["metadata"], dict)
                ):
                    raise ValueError(f"invalid marker record at line {line_number}")
                markers.append(
                    TelemetryMarker(timestamp, record["name"], record["metadata"])
                )
            else:
                raise ValueError(f"unknown telemetry record type at line {line_number}")
    if not samples:
        raise ValueError("telemetry JSONL contains no samples")
    evidence = TelemetryEvidence(
        path.resolve(), digest.hexdigest(), tuple(samples), tuple(markers)
    )
    collector_ready = [marker for marker in markers if marker.name == "collector_ready"]
    collector_end = [marker for marker in markers if marker.name == "collector_end"]
    if len(collector_ready) != 1 or len(collector_end) != 1:
        raise ValueError("telemetry JSONL lacks unique collector boundaries")
    if collector_ready[0].monotonic_ns >= collector_end[0].monotonic_ns:
        raise ValueError("telemetry collector boundaries are inverted")
    return evidence


def unique_marker(
    evidence: TelemetryEvidence,
    name: str,
    metadata: dict[str, Any],
) -> TelemetryMarker:
    matches = [
        marker
        for marker in evidence.markers
        if marker.name == name and dict(marker.metadata) == metadata
    ]
    if len(matches) != 1:
        raise ValueError(f"telemetry requires one {name} marker with {metadata}")
    return matches[0]


def validate_thermal_marker_labels(
    evidence: TelemetryEvidence, expected_labels: list[str]
) -> None:
    for marker_name in ("thermal_prepare", "thermal_start"):
        actual_metadata = [
            dict(marker.metadata)
            for marker in evidence.markers
            if marker.name == marker_name
        ]
        if actual_metadata != [{"label": label} for label in expected_labels]:
            raise ValueError(
                f"telemetry {marker_name} labels differ from the v4 formal run"
            )
    thermal_end_metadata = [
        dict(marker.metadata)
        for marker in evidence.markers
        if marker.name == "thermal_end"
    ]
    if thermal_end_metadata != [
        {"label": label, "successful": True} for label in expected_labels
    ]:
        raise ValueError("telemetry thermal_end labels differ from the v4 formal run")
    for marker_name in (
        "thermal_measurement_end",
        "thermal_start_qualification",
        "thermal_start_qualification_result",
    ):
        actual_labels = [
            marker.metadata.get("label")
            for marker in evidence.markers
            if marker.name == marker_name
        ]
        if actual_labels != expected_labels:
            raise ValueError(
                f"telemetry {marker_name} labels differ from the v4 formal run"
            )
    expected_label_set = set(expected_labels)
    active_labels = [
        marker.metadata.get("label")
        for marker in evidence.markers
        if marker.name == "thermal_active_stability_check"
    ]
    if set(active_labels) != expected_label_set or any(
        label not in expected_label_set for label in active_labels
    ):
        raise ValueError(
            "telemetry active-stability labels differ from the v4 formal run"
        )


def validate_calibration_markers(
    evidence: TelemetryEvidence,
    run: dict[str, Any],
    stage: str,
    repeats: int,
    thermal_lock: dict[str, Any],
    config: dict[str, Any],
    mig: dict[str, Any],
) -> tuple[int, int]:
    result_key = "isolated" if stage == "pre" else "isolated_post"
    precondition_key = (
        "isolated_preconditions"
        if stage == "pre"
        else "isolated_post_preconditions"
    )
    results = run.get(result_key)
    preconditions = run.get(precondition_key)
    if not isinstance(results, list) or len(results) != repeats:
        raise ValueError(f"{result_key} calibration results are incomplete")
    if not isinstance(preconditions, list) or len(preconditions) != repeats:
        raise ValueError(
            f"{precondition_key} must contain one thermal precondition per repeat"
        )
    critical_affinity = config.get("cpu_affinity", {}).get("critical")
    if critical_affinity != FORMAL_CPU_AFFINITY["critical"]:
        raise ValueError("formal calibration requires the frozen critical CPU")
    critical_uuid = mig.get("critical_uuid")
    if not isinstance(critical_uuid, str) or not critical_uuid:
        raise ValueError("formal calibration lacks the critical MIG UUID")
    boundaries: list[tuple[int, int]] = []
    for repeat, (result, final_precondition) in enumerate(
        zip(results, preconditions, strict=True), start=1
    ):
        if not isinstance(result, dict) or not isinstance(final_precondition, dict):
            raise ValueError(f"{result_key}[{repeat - 1}] is invalid")
        metadata = {"stage": stage, "repeat": repeat}
        prepare_marker = unique_marker(evidence, "calibration_prepare", metadata)
        start_marker = unique_marker(evidence, "calibration_start", metadata)
        end_marker = unique_marker(evidence, "calibration_end", metadata)
        critical_config = result.get("config")
        if not isinstance(critical_config, dict) or strict_bool(
            critical_config.get("start_paused"),
            f"{result_key}[{repeat - 1}].config.start_paused",
        ) is not True:
            raise ValueError(
                f"{result_key}[{repeat - 1}] was not a preloaded paused critical"
            )
        critical_pid = validate_execution_environment(
            result.get("execution_environment"),
            label=f"{result_key}[{repeat - 1}].execution_environment",
            expected_device=critical_uuid,
            expected_quota=100,
            expected_cpus=FORMAL_CPU_AFFINITY["critical"],
        )
        readiness = result.get("readiness_affinity")
        if not isinstance(readiness, dict):
            raise ValueError(
                f"{result_key}[{repeat - 1}] lacks readiness-affinity evidence"
            )
        validate_readiness_entry(
            {"role": "critical", **readiness},
            label=f"{result_key}[{repeat - 1}].readiness_affinity",
            role="critical",
            pid=critical_pid,
            expected_cpu=FORMAL_CPU_AFFINITY["critical"][0],
        )
        base_label = f"pre-{stage}-calibration-r{repeat}"
        successful = validate_thermal_start_attempts(
            result.get("thermal_start_attempts"),
            base_label=base_label,
            expected_pids=[critical_pid],
            evidence=evidence,
            thermal_lock=thermal_lock,
            result_marker_metadata=metadata,
        )
        attempts = result["thermal_start_attempts"]
        expected_precondition_label = (
            f"{base_label}-attempt-{len(attempts):02d}"
        )
        if (
            final_precondition != successful["precondition"]
            or result.get("thermal_precondition") != successful["precondition"]
            or result.get("thermal_start_qualification")
            != successful["qualification"]
            or result.get("thermal_precondition_label")
            != expected_precondition_label
        ):
            raise ValueError(
                f"{result_key}[{repeat - 1}] final qualification aliases differ"
            )
        release_ns = strict_int(
            result.get("measurement_release_monotonic_ns"),
            f"{result_key}[{repeat - 1}].measurement_release_monotonic_ns",
        )
        if start_marker.monotonic_ns != release_ns:
            raise ValueError(
                f"{result_key}[{repeat - 1}] calibration release marker differs"
            )
        measurement_start = strict_int(
            result.get("measurement_start_monotonic_ns"),
            f"{result_key}[{repeat - 1}].measurement_start_monotonic_ns",
        )
        measurement_end = strict_int(
            result.get("measurement_end_monotonic_ns"),
            f"{result_key}[{repeat - 1}].measurement_end_monotonic_ns",
        )
        actual_start, actual_start_marker_ns = replay_actual_start_qualification(
            result.get("thermal_actual_start_qualification"),
            measurement_start_ns=measurement_start,
            not_before_ns=int(successful["raw"]["cleanup_end_monotonic_ns"]),
            qualification_sample_ns=int(successful["raw"]["sample_monotonic_ns"]),
            evidence=evidence,
            thermal_lock=thermal_lock,
            result_marker_metadata=metadata,
        )
        if actual_start["passed"] is not True:
            raise ValueError(
                f"{result_key}[{repeat - 1}] failed actual-start qualification"
            )
        window_marker = unique_marker(
            evidence,
            "calibration_measurement_window",
            metadata
            | {
                "measurement_start_monotonic_ns": measurement_start,
                "measurement_end_monotonic_ns": measurement_end,
            },
        )
        if not (
            prepare_marker.monotonic_ns
            < int(successful["first_prepare_monotonic_ns"])
            < int(successful["result_marker_monotonic_ns"])
            < start_marker.monotonic_ns
            <= measurement_start
            < measurement_end
            < actual_start_marker_ns
            <= window_marker.monotonic_ns
            <= end_marker.monotonic_ns
        ):
            raise ValueError(f"{result_key}[{repeat - 1}] marker order is invalid")
        validate_thermal_handoff(
            result.get("thermal_handoff"),
            boundary_ns=int(successful["raw"]["boundary_monotonic_ns"]),
            cleanup_end_ns=int(successful["raw"]["cleanup_end_monotonic_ns"]),
            qualification_ns=int(successful["raw"]["qualification_monotonic_ns"]),
            qualification_result_ns=int(
                successful["result_marker_monotonic_ns"]
            ),
            release_ns=release_ns,
            measurement_start_ns=measurement_start,
            thermal_lock=thermal_lock,
            label=f"{result_key}[{repeat - 1}].thermal_handoff",
        )
        result_handoff_ms = (
            int(successful["result_marker_monotonic_ns"])
            - int(successful["raw"]["boundary_monotonic_ns"])
        ) / 1_000_000.0
        if result_handoff_ms >= float(thermal_lock["thermal_handoff_max_ms"]):
            raise ValueError(
                f"{result_key}[{repeat - 1}] qualification result exceeds the handoff bound"
            )
        require_json_equivalent(
            result.get("thermal_start"),
            successful["precondition"]["last_window"],
            f"{result_key}[{repeat - 1}].thermal_start",
        )
        if strict_bool(
            result.get("thermal_start_stable"),
            f"{result_key}[{repeat - 1}].thermal_start_stable",
        ) is not True or not thermal_summary_is_stable(
            successful["precondition"]["last_window"], thermal_lock
        ):
            raise ValueError(f"{result_key}[{repeat - 1}] thermal start is unstable")
        require_json_equivalent(
            result.get("thermal_start_telemetry"),
            successful["qualification"]["telemetry"],
            f"{result_key}[{repeat - 1}].thermal_start_telemetry",
        )
        validate_raw_safety_interval(
            evidence,
            int(successful["raw"]["boundary_monotonic_ns"]),
            release_ns,
            thermal_lock,
            f"{result_key}[{repeat - 1}].thermal_start",
        )
        validate_raw_safety_interval(
            evidence,
            measurement_start,
            measurement_end,
            thermal_lock,
            f"{result_key}[{repeat - 1}].measurement",
        )
        if boundaries and boundaries[-1][1] >= prepare_marker.monotonic_ns:
            raise ValueError(f"{stage} calibration repeats overlap or reorder")
        boundaries.append((prepare_marker.monotonic_ns, end_marker.monotonic_ns))
    return boundaries[0][0], boundaries[-1][1]


def telemetry_minimum_samples(start_ns: int, end_ns: int) -> int:
    duration_ns = end_ns - start_ns
    if duration_ns <= 0:
        raise ValueError("telemetry interval must be positive")
    expected = max(1, math.floor(duration_ns / 100_000_000.0))
    return max(1, math.floor(expected * 0.8))


def telemetry_sample_is_structurally_valid(sample: TelemetrySample) -> bool:
    return (
        sample.parsed.ram is not None
        and sample.mem_available_mb is not None
        and any(core.utilization_pct is not None for core in sample.parsed.cpu)
        and FORMAL_THERMAL_STABILITY_SENSOR in sample.parsed.temperatures_c
        and FORMAL_THERMAL_SAFETY_SENSOR in sample.parsed.temperatures_c
        and "VIN" in sample.parsed.power
    )


def validate_telemetry_coverage(
    evidence: TelemetryEvidence,
    start_ns: int,
    end_ns: int,
    label: str,
) -> None:
    valid_timestamps = [
        sample.monotonic_ns
        for sample in evidence.samples
        if start_ns <= sample.monotonic_ns <= end_ns
        and telemetry_sample_is_structurally_valid(sample)
    ]
    if not valid_timestamps:
        raise ValueError(f"{label} has no structurally valid telemetry samples")
    maximum_gap_ns = 300_000_000
    gaps = [
        valid_timestamps[0] - start_ns,
        *(right - left for left, right in zip(valid_timestamps, valid_timestamps[1:])),
        end_ns - valid_timestamps[-1],
    ]
    if any(gap < 0 or gap > maximum_gap_ns for gap in gaps):
        raise ValueError(f"{label} has a structurally valid telemetry gap over 300 ms")


def thermal_summary_from_samples(
    samples: tuple[TelemetrySample, ...],
    reference_ns: int,
    *,
    sensor: str = FORMAL_THERMAL_STABILITY_SENSOR,
    window_seconds: float = 60.0,
    not_before_ns: int | None = None,
) -> dict[str, float | int] | None:
    start_ns = reference_ns - int(window_seconds * 1_000_000_000)
    if not_before_ns is not None:
        start_ns = max(start_ns, not_before_ns)
    points = [
        (sample.monotonic_ns, sample.parsed.temperatures_c[sensor])
        for sample in samples
        if start_ns <= sample.monotonic_ns <= reference_ns
        and sensor in sample.parsed.temperatures_c
    ]
    if len(points) < 2:
        return None
    times = [
        (timestamp - points[0][0]) / 1_000_000_000.0
        for timestamp, _value in points
    ]
    values = [value for _timestamp, value in points]
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
    timestamps = [timestamp for timestamp, _value in points]
    maximum_gap_seconds = max(
        timestamps[0] - start_ns,
        reference_ns - timestamps[-1],
        *(right - left for left, right in zip(timestamps, timestamps[1:])),
    ) / 1_000_000_000.0
    return {
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


def thermal_summary_is_stable(
    summary: dict[str, float | int] | None, thermal_lock: dict[str, Any]
) -> bool:
    if summary is None:
        return False
    window_seconds = float(thermal_lock["stability_window_seconds"])
    required_samples = telemetry_minimum_samples(
        0, int(window_seconds * 1_000_000_000)
    )
    return (
        int(summary["samples"]) >= required_samples
        and float(summary["observed_span_seconds"]) >= window_seconds * 0.99
        and abs(float(summary["mean_c"]) - float(thermal_lock["target_c"]))
        <= float(thermal_lock["tolerance_c"])
        and abs(float(summary["latest_c"]) - float(thermal_lock["target_c"]))
        <= float(thermal_lock["tolerance_c"])
        and abs(float(summary["slope_c_per_minute"]))
        <= float(thermal_lock["maximum_slope_c_per_minute"])
        and float(summary.get("maximum_gap_seconds", math.inf))
        <= float(thermal_lock["telemetry_max_gap_ms"]) / 1000.0
    )


def replay_telemetry_aggregate(
    evidence: TelemetryEvidence,
    start_ns: int,
    end_ns: int,
    *,
    end_inclusive: bool = False,
) -> dict[str, Any]:
    validate_telemetry_coverage(evidence, start_ns, end_ns, "telemetry aggregate")
    aggregate = aggregate_samples(
        evidence.samples,
        start_ns,
        end_ns,
        required_fields=FORMAL_TELEMETRY_REQUIRED_FIELDS,
        minimum_valid_samples=telemetry_minimum_samples(start_ns, end_ns),
        reference_ns=end_ns,
        stale_after_ns=300_000_000,
        maximum_valid_gap_ns=300_000_000,
        end_inclusive=end_inclusive,
    )
    aggregate["retention"] = {
        "bounded": False,
        "max_samples": None,
        "dropped_samples": 0,
        "last_dropped_sample_ns": None,
        "earliest_retained_sample_ns": evidence.samples[0].monotonic_ns,
        "interval_complete": True,
    }
    return aggregate


def replay_point_telemetry_aggregate(
    evidence: TelemetryEvidence,
    *,
    sample_ns: int,
    reference_ns: int,
) -> dict[str, Any]:
    """Replay the frozen one-sample qualification aggregate exactly."""

    if sample_ns <= 0 or reference_ns < sample_ns:
        raise ValueError("point telemetry aggregate has invalid clocks")
    aggregate = aggregate_samples(
        evidence.samples,
        sample_ns - 1,
        sample_ns,
        required_fields=FORMAL_TELEMETRY_REQUIRED_FIELDS,
        minimum_valid_samples=1,
        require_all_samples_valid=True,
        reference_ns=reference_ns,
        stale_after_ns=300_000_000,
        maximum_valid_gap_ns=300_000_000,
        end_inclusive=True,
    )
    aggregate["retention"] = {
        "bounded": False,
        "max_samples": None,
        "dropped_samples": 0,
        "last_dropped_sample_ns": None,
        "earliest_retained_sample_ns": evidence.samples[0].monotonic_ns,
        "interval_complete": True,
    }
    return aggregate


def validate_telemetry_aggregate(
    actual: object,
    expected: dict[str, Any],
    label: str,
    *,
    require_healthy: bool,
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} telemetry summary is missing")
    require_json_equivalent(actual, expected, label)
    if require_healthy and expected["health"]["healthy"] is not True:
        raise ValueError(f"{label} raw telemetry is unhealthy")


def validate_raw_safety_interval(
    evidence: TelemetryEvidence,
    start_ns: int,
    end_ns: int,
    thermal_lock: dict[str, Any],
    label: str,
) -> float:
    """Enforce structural gaps and TJ safety independently of stability."""

    validate_telemetry_coverage(evidence, start_ns, end_ns, label)
    safety_sensor = str(thermal_lock["safety_sensor"])
    values = [
        float(sample.parsed.temperatures_c[safety_sensor])
        for sample in evidence.samples
        if start_ns <= sample.monotonic_ns <= end_ns
        and safety_sensor in sample.parsed.temperatures_c
    ]
    if not values:
        raise ValueError(f"{label} lacks raw {safety_sensor} telemetry")
    maximum = max(values)
    if maximum >= float(thermal_lock["hard_limit_c"]):
        raise ValueError(f"{label} reached the {safety_sensor} thermal hard limit")
    return maximum


def validate_thermal_handoff(
    stored: object,
    *,
    boundary_ns: int,
    cleanup_end_ns: int,
    qualification_ns: int,
    qualification_result_ns: int,
    release_ns: int,
    measurement_start_ns: int,
    thermal_lock: dict[str, Any],
    label: str,
) -> dict[str, float | int | bool | str]:
    expected_fields = {
        "boundary",
        "boundary_monotonic_ns",
        "cleanup_end_monotonic_ns",
        "qualification_monotonic_ns",
        "qualification_result_monotonic_ns",
        "measurement_release_monotonic_ns",
        "measurement_start_monotonic_ns",
        "boundary_to_cleanup_end_ms",
        "boundary_to_qualification_ms",
        "boundary_to_qualification_result_ms",
        "boundary_to_measurement_release_ms",
        "boundary_to_measurement_start_ms",
        "maximum_ms",
        "strictly_within_bound",
    }
    if not isinstance(stored, dict) or set(stored) != expected_fields:
        raise ValueError(f"{label} has invalid thermal handoff fields")
    if not (
        boundary_ns
        < cleanup_end_ns
        < qualification_ns
        < qualification_result_ns
        < release_ns
        <= measurement_start_ns
    ):
        raise ValueError(f"{label} has inconsistent thermal handoff clocks")
    if stored.get("boundary") != THERMAL_HANDOFF_BOUNDARY:
        raise ValueError(f"{label} uses a stale thermal handoff boundary")
    maximum_ms = strict_float(
        thermal_lock.get("thermal_handoff_max_ms"),
        "thermal lock handoff maximum",
        minimum=0.0,
    )
    if maximum_ms <= 0.0:
        raise ValueError("thermal lock handoff maximum must be positive")
    elapsed = {
        "boundary_to_cleanup_end_ms": (cleanup_end_ns - boundary_ns) / 1_000_000.0,
        "boundary_to_qualification_ms": (qualification_ns - boundary_ns)
        / 1_000_000.0,
        "boundary_to_qualification_result_ms": (
            qualification_result_ns - boundary_ns
        )
        / 1_000_000.0,
        "boundary_to_measurement_release_ms": (release_ns - boundary_ns)
        / 1_000_000.0,
        "boundary_to_measurement_start_ms": (measurement_start_ns - boundary_ns)
        / 1_000_000.0,
    }
    if any(value >= maximum_ms for value in elapsed.values()):
        raise ValueError(f"{label} is not strictly within the thermal handoff bound")
    expected_integers = {
        "boundary_monotonic_ns": boundary_ns,
        "cleanup_end_monotonic_ns": cleanup_end_ns,
        "qualification_monotonic_ns": qualification_ns,
        "qualification_result_monotonic_ns": qualification_result_ns,
        "measurement_release_monotonic_ns": release_ns,
        "measurement_start_monotonic_ns": measurement_start_ns,
    }
    for field, expected in expected_integers.items():
        if strict_int(stored.get(field), f"{label}.{field}") != expected:
            raise ValueError(f"{label}.{field} differs from raw clocks")
    for field, expected in {**elapsed, "maximum_ms": maximum_ms}.items():
        require_close(
            stored.get(field),
            expected,
            f"{label}.{field}",
            absolute_tolerance=1e-9,
            relative_tolerance=0.0,
        )
    if strict_bool(
        stored.get("strictly_within_bound"),
        f"{label}.strictly_within_bound",
    ) is not True:
        raise ValueError(f"{label} does not attest a strict handoff")
    return {
        "boundary": THERMAL_HANDOFF_BOUNDARY,
        **expected_integers,
        **elapsed,
        "maximum_ms": maximum_ms,
        "strictly_within_bound": True,
    }


def replay_active_stability_checks(
    stored: object,
    *,
    expected_label: str,
    evidence: TelemetryEvidence,
    thermal_lock: dict[str, Any],
    not_before_ns: int,
    boundary_ns: int,
) -> list[dict[str, Any]]:
    if not isinstance(stored, list) or not stored:
        raise ValueError("thermal precondition lacks active stability checks")
    markers = [
        marker
        for marker in evidence.markers
        if marker.name == "thermal_active_stability_check"
        and marker.metadata.get("label") == expected_label
    ]
    if len(markers) != len(stored):
        raise ValueError("active stability marker count differs from stored evidence")
    sample_timestamps = {sample.monotonic_ns for sample in evidence.samples}
    spacing_ns = int(THERMAL_ACTIVE_STABLE_SPACING_SECONDS * 1_000_000_000.0)
    consecutive = 0
    previous_sample_ns: int | None = None
    replayed: list[dict[str, Any]] = []
    expected_fields = {
        "label",
        "index",
        "sample_monotonic_ns",
        "passed",
        "consecutive_passes",
        "window",
    }
    for index, (reported, marker) in enumerate(zip(stored, markers, strict=True)):
        metadata = dict(marker.metadata)
        if (
            not isinstance(reported, dict)
            or set(reported) != expected_fields
            or set(metadata) != expected_fields
            or reported != metadata
            or metadata.get("label") != expected_label
            or strict_int(metadata.get("index"), "active check index") != index
        ):
            raise ValueError("active stability check has invalid fields")
        sample_ns = strict_int(
            metadata.get("sample_monotonic_ns"),
            "active check sample_monotonic_ns",
        )
        if (
            sample_ns not in sample_timestamps
            or not_before_ns > sample_ns
            or sample_ns >= boundary_ns
            or not sample_ns < marker.monotonic_ns < boundary_ns
            or (
                previous_sample_ns is not None
                and sample_ns - previous_sample_ns < spacing_ns
            )
        ):
            raise ValueError("active stability check selected a non-causal endpoint")
        raw_window = thermal_summary_from_samples(
            evidence.samples,
            sample_ns,
            sensor=str(thermal_lock["stability_sensor"]),
            window_seconds=float(thermal_lock["stability_window_seconds"]),
            not_before_ns=not_before_ns,
        )
        if raw_window is None:
            raise ValueError("active stability check lacks a raw thermal window")
        window_start_ns = max(
            not_before_ns,
            sample_ns
            - int(float(thermal_lock["stability_window_seconds"]) * 1e9),
        )
        validate_telemetry_coverage(
            evidence, window_start_ns, sample_ns, "active stability check"
        )
        validate_raw_safety_interval(
            evidence,
            window_start_ns,
            sample_ns,
            thermal_lock,
            "active stability check",
        )
        passed = thermal_summary_is_stable(raw_window, thermal_lock)
        consecutive = consecutive + 1 if passed else 0
        if (
            metadata.get("passed") is not passed
            or strict_int(
                metadata.get("consecutive_passes"),
                "active check consecutive_passes",
            )
            != consecutive
        ):
            raise ValueError("active stability result differs from raw telemetry")
        require_json_equivalent(
            metadata.get("window"), raw_window, "active stability check window"
        )
        replayed.append(metadata)
        previous_sample_ns = sample_ns
    if consecutive != THERMAL_ACTIVE_STABLE_ENDPOINTS:
        raise ValueError("active precondition lacks three stable endpoints")
    return replayed


def validate_thermal_precondition(
    summary: object,
    expected_label: str,
    evidence: TelemetryEvidence,
    thermal_lock: dict[str, Any],
) -> tuple[int, int]:
    expected_fields = {
        "label",
        "duration_seconds",
        "measurement_start_monotonic_ns",
        "measurement_end_monotonic_ns",
        "cleanup_end_monotonic_ns",
        "target_c",
        "stability_sensor",
        "safety_sensor",
        "last_window",
        "active_stability_checks",
        "active_stable_endpoints",
        "active_stable_spacing_seconds",
        "termination_reason",
        "pressure_rate_per_second",
        "telemetry",
    }
    if (
        not isinstance(summary, dict)
        or set(summary) != expected_fields
        or summary.get("label") != expected_label
    ):
        raise ValueError(f"missing thermal precondition summary {expected_label}")
    if (
        summary.get("stability_sensor") != thermal_lock.get("stability_sensor")
        or summary.get("safety_sensor") != thermal_lock.get("safety_sensor")
        or summary.get("active_stable_endpoints")
        != THERMAL_ACTIVE_STABLE_ENDPOINTS
        or summary.get("active_stable_spacing_seconds")
        != THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        or summary.get("termination_reason") != "active-stability-endpoints"
    ):
        raise ValueError(f"thermal precondition {expected_label} sensor binding differs")
    metadata = {"label": expected_label}
    prepare = unique_marker(evidence, "thermal_prepare", metadata)
    start = unique_marker(evidence, "thermal_start", metadata)
    boundaries = [
        marker
        for marker in evidence.markers
        if marker.name == "thermal_measurement_end"
        and marker.metadata.get("label") == expected_label
    ]
    if len(boundaries) != 1:
        raise ValueError(f"thermal boundary is missing for {expected_label}")
    measurement_end = boundaries[0]
    cleanup_end = unique_marker(
        evidence, "thermal_end", metadata | {"successful": True}
    )
    if not (
        prepare.monotonic_ns
        < start.monotonic_ns
        < measurement_end.monotonic_ns
        <= cleanup_end.monotonic_ns
    ):
        raise ValueError(f"thermal marker order is invalid for {expected_label}")
    if strict_int(
        summary.get("measurement_start_monotonic_ns"),
        f"{expected_label}.measurement_start_monotonic_ns",
    ) != start.monotonic_ns or strict_int(
        summary.get("measurement_end_monotonic_ns"),
        f"{expected_label}.measurement_end_monotonic_ns",
    ) != measurement_end.monotonic_ns or strict_int(
        summary.get("cleanup_end_monotonic_ns"),
        f"{expected_label}.cleanup_end_monotonic_ns",
    ) != cleanup_end.monotonic_ns:
        raise ValueError(f"thermal summary timestamps differ for {expected_label}")
    duration = (measurement_end.monotonic_ns - start.monotonic_ns) / 1e9
    require_close(
        summary.get("duration_seconds"),
        duration,
        f"{expected_label}.duration_seconds",
        absolute_tolerance=5e-10,
    )
    require_close(
        summary.get("target_c"),
        float(thermal_lock["target_c"]),
        f"{expected_label}.target_c",
        absolute_tolerance=1e-12,
        relative_tolerance=0.0,
    )
    active_checks = replay_active_stability_checks(
        summary.get("active_stability_checks"),
        expected_label=expected_label,
        evidence=evidence,
        thermal_lock=thermal_lock,
        not_before_ns=start.monotonic_ns,
        boundary_ns=measurement_end.monotonic_ns,
    )
    final_check = active_checks[-1]
    if dict(measurement_end.metadata) != {
        "label": expected_label,
        "boundary_sample_monotonic_ns": final_check["sample_monotonic_ns"],
        "consecutive_passes": THERMAL_ACTIVE_STABLE_ENDPOINTS,
        "window": final_check["window"],
    }:
        raise ValueError("thermal boundary does not bind the final active endpoint")
    boundary_sample_ns = int(final_check["sample_monotonic_ns"])
    if not 0 <= measurement_end.monotonic_ns - boundary_sample_ns < int(
        float(thermal_lock["telemetry_max_gap_ms"]) * 1_000_000.0
    ):
        raise ValueError("thermal boundary sample is stale")
    raw_window = thermal_summary_from_samples(
        evidence.samples,
        boundary_sample_ns,
        sensor=str(thermal_lock["stability_sensor"]),
        window_seconds=float(thermal_lock["stability_window_seconds"]),
        not_before_ns=start.monotonic_ns,
    )
    if raw_window is None:
        raise ValueError(
            f"thermal precondition {expected_label} lacks a stability-sensor window"
        )
    validate_telemetry_coverage(
        evidence,
        max(
            start.monotonic_ns,
            boundary_sample_ns
            - int(float(thermal_lock["stability_window_seconds"]) * 1e9),
        ),
        boundary_sample_ns,
        f"{expected_label}.last_window",
    )
    require_json_equivalent(
        summary.get("last_window"), raw_window, f"{expected_label}.last_window"
    )
    if not thermal_summary_is_stable(raw_window, thermal_lock):
        raise ValueError(f"thermal precondition {expected_label} is not stable")
    expected_aggregate = replay_telemetry_aggregate(
        evidence, start.monotonic_ns, measurement_end.monotonic_ns
    )
    validate_telemetry_aggregate(
        summary.get("telemetry"),
        expected_aggregate,
        f"{expected_label}.telemetry",
        require_healthy=True,
    )
    validate_raw_safety_interval(
        evidence,
        start.monotonic_ns,
        measurement_end.monotonic_ns,
        thermal_lock,
        f"thermal precondition {expected_label}",
    )
    strict_float(
        summary.get("pressure_rate_per_second"),
        f"{expected_label}.pressure_rate_per_second",
        minimum=0.0,
    )
    return prepare.monotonic_ns, cleanup_end.monotonic_ns


def validate_paused_process_states(
    stored: object, *, expected_pids: list[int], label: str
) -> dict[str, str]:
    expected = {str(pid): "T" for pid in sorted(expected_pids)}
    if not isinstance(stored, dict) or stored != expected:
        raise ValueError(f"{label} does not prove every measured PID remained paused")
    return expected


def replay_thermal_qualification(
    qualification: object,
    *,
    attempt: int,
    attempt_label: str,
    precondition: dict[str, Any],
    evidence: TelemetryEvidence,
    thermal_lock: dict[str, Any],
    result_marker_metadata: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    expected_fields = {
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
    if not isinstance(qualification, dict) or set(qualification) != expected_fields:
        raise ValueError("thermal qualification has invalid fields")
    if (
        strict_int(qualification.get("attempt"), "qualification.attempt", minimum=1)
        != attempt
        or qualification.get("boundary") != THERMAL_HANDOFF_BOUNDARY
    ):
        raise ValueError("thermal qualification identity differs")
    boundary_ns = strict_int(
        qualification.get("boundary_monotonic_ns"),
        "qualification.boundary_monotonic_ns",
    )
    cleanup_ns = strict_int(
        qualification.get("cleanup_end_monotonic_ns"),
        "qualification.cleanup_end_monotonic_ns",
    )
    qualification_ns = strict_int(
        qualification.get("qualification_monotonic_ns"),
        "qualification.qualification_monotonic_ns",
    )
    reported_passed = qualification.get("passed")
    if not isinstance(reported_passed, bool):
        raise ValueError("thermal qualification passed flag is not boolean")
    sample_value = qualification.get("sample_monotonic_ns")
    sample_ns = (
        None
        if sample_value is None
        else strict_int(
            sample_value,
            "qualification.sample_monotonic_ns",
        )
    )
    if (
        boundary_ns != strict_int(
            precondition.get("measurement_end_monotonic_ns"),
            f"{attempt_label}.measurement_end_monotonic_ns",
        )
        or cleanup_ns
        != strict_int(
            precondition.get("cleanup_end_monotonic_ns"),
            f"{attempt_label}.cleanup_end_monotonic_ns",
        )
        or not boundary_ns < cleanup_ns < qualification_ns
        or (
            sample_ns is not None
            and not cleanup_ns < sample_ns <= qualification_ns
        )
    ):
        raise ValueError("thermal qualification clocks are not causal")
    qualification_marker = unique_marker(
        evidence,
        "thermal_start_qualification",
        {
            "label": attempt_label,
            "attempt": attempt,
            "boundary_monotonic_ns": boundary_ns,
            "cleanup_end_monotonic_ns": cleanup_ns,
            "sample_monotonic_ns": sample_ns,
        },
    )
    if qualification_marker.monotonic_ns != qualification_ns:
        raise ValueError("thermal qualification marker clock differs")
    causal_samples = [
        sample
        for sample in evidence.samples
        if cleanup_ns < sample.monotonic_ns <= qualification_ns
    ]
    if sample_ns is None:
        if causal_samples:
            raise ValueError(
                "sample-free thermal qualification hides a causal raw sample"
            )
        failure_reason = qualification.get("failure_reason")
        target_c = float(thermal_lock["target_c"])
        tolerance_c = float(thermal_lock["tolerance_c"])
        if (
            reported_passed is not False
            or qualification.get("sample_age_ms") is not None
            or qualification.get("stability_value_c") is not None
            or qualification.get("safety_value_c") is not None
            or qualification.get("telemetry") is not None
            or qualification.get("stability_sensor")
            != str(thermal_lock["stability_sensor"])
            or qualification.get("safety_sensor")
            != str(thermal_lock["safety_sensor"])
            or not math.isclose(
                strict_float(
                    qualification.get("target_c"), "qualification.target_c"
                ),
                target_c,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                strict_float(
                    qualification.get("tolerance_c"),
                    "qualification.tolerance_c",
                ),
                tolerance_c,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not isinstance(failure_reason, str)
            or not failure_reason
        ):
            raise ValueError("sample-free thermal qualification is malformed")
        result_marker = unique_marker(
            evidence,
            "thermal_start_qualification_result",
            result_marker_metadata
            | {
                "label": attempt_label,
                "attempt": attempt,
                "qualification_monotonic_ns": qualification_ns,
                "passed": False,
                "failure_reason": failure_reason,
            },
        )
        if result_marker.monotonic_ns <= qualification_ns:
            raise ValueError("thermal qualification result marker is reordered")
        return {
            "boundary_monotonic_ns": boundary_ns,
            "cleanup_end_monotonic_ns": cleanup_ns,
            "qualification_monotonic_ns": qualification_ns,
            "sample_monotonic_ns": None,
            "sample_age_ms": None,
            "stability_value_c": None,
            "safety_value_c": None,
            "passed": False,
        }, result_marker.monotonic_ns

    assert sample_ns is not None
    if not causal_samples or causal_samples[0].monotonic_ns != sample_ns:
        raise ValueError("thermal qualification did not select the first causal sample")
    sample = causal_samples[0]
    structurally_valid = telemetry_sample_is_structurally_valid(sample)
    stability_sensor = str(thermal_lock["stability_sensor"])
    safety_sensor = str(thermal_lock["safety_sensor"])
    raw_stability = (
        float(sample.parsed.temperatures_c[stability_sensor])
        if structurally_valid
        else math.nan
    )
    raw_safety = (
        float(sample.parsed.temperatures_c[safety_sensor])
        if structurally_valid
        else math.nan
    )
    sample_age_ms = (qualification_ns - sample_ns) / 1_000_000.0
    sample_gap_ms = (sample_ns - cleanup_ns) / 1_000_000.0
    expected_telemetry = replay_point_telemetry_aggregate(
        evidence,
        sample_ns=sample_ns,
        reference_ns=qualification_ns,
    )
    validate_telemetry_aggregate(
        qualification.get("telemetry"),
        expected_telemetry,
        "thermal qualification.telemetry",
        require_healthy=True,
    )
    target_c = float(thermal_lock["target_c"])
    tolerance_c = float(thermal_lock["tolerance_c"])
    maximum_gap_ms = float(thermal_lock["telemetry_max_gap_ms"])
    raw_passed = (
        structurally_valid
        and sample_age_ms <= maximum_gap_ms
        and sample_gap_ms <= maximum_gap_ms
        and abs(raw_stability - target_c) <= tolerance_c
        and raw_safety < float(thermal_lock["hard_limit_c"])
        and expected_telemetry["health"]["healthy"] is True
    )
    if (
        qualification.get("stability_sensor") != stability_sensor
        or qualification.get("safety_sensor") != safety_sensor
        or not math.isclose(
            strict_float(qualification.get("sample_age_ms"), "sample_age_ms"),
            sample_age_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(qualification.get("stability_value_c"), "stability_value_c"),
            raw_stability,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(qualification.get("safety_value_c"), "safety_value_c"),
            raw_safety,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(qualification.get("target_c"), "qualification.target_c"),
            target_c,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(
                qualification.get("tolerance_c"), "qualification.tolerance_c"
            ),
            tolerance_c,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or qualification.get("passed") is not raw_passed
        or (raw_passed and qualification.get("failure_reason") is not None)
        or (
            not raw_passed
            and (
                not isinstance(qualification.get("failure_reason"), str)
                or not qualification["failure_reason"]
            )
        )
    ):
        raise ValueError("thermal qualification differs from raw telemetry")
    result_marker = unique_marker(
        evidence,
        "thermal_start_qualification_result",
        result_marker_metadata
        | {
            "label": attempt_label,
            "attempt": attempt,
            "qualification_monotonic_ns": qualification_ns,
            "passed": raw_passed,
            "failure_reason": qualification.get("failure_reason"),
        },
    )
    if result_marker.monotonic_ns <= qualification_ns:
        raise ValueError("thermal qualification result marker is reordered")
    return {
        "boundary_monotonic_ns": boundary_ns,
        "cleanup_end_monotonic_ns": cleanup_ns,
        "qualification_monotonic_ns": qualification_ns,
        "sample_monotonic_ns": sample_ns,
        "sample_age_ms": sample_age_ms,
        "stability_value_c": raw_stability,
        "safety_value_c": raw_safety,
        "passed": raw_passed,
    }, result_marker.monotonic_ns


def replay_actual_start_qualification(
    reported: object,
    *,
    measurement_start_ns: int,
    not_before_ns: int,
    qualification_sample_ns: int,
    evidence: TelemetryEvidence,
    thermal_lock: dict[str, Any],
    result_marker_metadata: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    expected_fields = {
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
    if not isinstance(reported, dict) or set(reported) != expected_fields:
        raise ValueError("actual-start qualification has invalid fields")
    stored_start_ns = strict_int(
        reported.get("measurement_start_monotonic_ns"),
        "actual-start measurement_start_monotonic_ns",
    )
    sample_ns = strict_int(
        reported.get("sample_monotonic_ns"),
        "actual-start sample_monotonic_ns",
    )
    if stored_start_ns != measurement_start_ns or not (
        not_before_ns < qualification_sample_ns <= sample_ns <= measurement_start_ns
    ):
        raise ValueError("actual-start qualification clocks are not causal")
    causal_samples = [
        sample
        for sample in evidence.samples
        if not_before_ns < sample.monotonic_ns <= measurement_start_ns
    ]
    if not causal_samples or causal_samples[-1].monotonic_ns != sample_ns:
        raise ValueError("actual start did not select the latest causal sample")
    sample = causal_samples[-1]
    structurally_valid = telemetry_sample_is_structurally_valid(sample)
    stability_sensor = str(thermal_lock["stability_sensor"])
    safety_sensor = str(thermal_lock["safety_sensor"])
    raw_stability = (
        float(sample.parsed.temperatures_c[stability_sensor])
        if structurally_valid
        else math.nan
    )
    raw_safety = (
        float(sample.parsed.temperatures_c[safety_sensor])
        if structurally_valid
        else math.nan
    )
    sample_age_ms = (measurement_start_ns - sample_ns) / 1_000_000.0
    expected_telemetry = replay_point_telemetry_aggregate(
        evidence,
        sample_ns=sample_ns,
        reference_ns=measurement_start_ns,
    )
    validate_telemetry_aggregate(
        reported.get("telemetry"),
        expected_telemetry,
        "actual-start qualification.telemetry",
        require_healthy=True,
    )
    target_c = float(thermal_lock["target_c"])
    tolerance_c = float(thermal_lock["tolerance_c"])
    raw_passed = (
        structurally_valid
        and sample_age_ms <= float(thermal_lock["telemetry_max_gap_ms"])
        and abs(raw_stability - target_c) <= tolerance_c
        and raw_safety < float(thermal_lock["hard_limit_c"])
        and expected_telemetry["health"]["healthy"] is True
    )
    if (
        reported.get("stability_sensor") != stability_sensor
        or reported.get("safety_sensor") != safety_sensor
        or not math.isclose(
            strict_float(reported.get("sample_age_ms"), "actual-start sample_age_ms"),
            sample_age_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(
                reported.get("stability_value_c"),
                "actual-start stability_value_c",
            ),
            raw_stability,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(
                reported.get("safety_value_c"), "actual-start safety_value_c"
            ),
            raw_safety,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(reported.get("target_c"), "actual-start target_c"),
            target_c,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(reported.get("tolerance_c"), "actual-start tolerance_c"),
            tolerance_c,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or reported.get("passed") is not raw_passed
        or (raw_passed and reported.get("failure_reason") is not None)
        or (
            not raw_passed
            and (
                not isinstance(reported.get("failure_reason"), str)
                or not reported["failure_reason"]
            )
        )
    ):
        raise ValueError("actual-start qualification differs from raw telemetry")
    result_marker = unique_marker(
        evidence,
        "thermal_actual_start_qualification_result",
        result_marker_metadata
        | {
            "measurement_start_monotonic_ns": measurement_start_ns,
            "sample_monotonic_ns": sample_ns,
            "passed": raw_passed,
            "failure_reason": reported.get("failure_reason"),
        },
    )
    if result_marker.monotonic_ns <= measurement_start_ns:
        raise ValueError("actual-start qualification marker is reordered")
    return {
        "measurement_start_monotonic_ns": measurement_start_ns,
        "sample_monotonic_ns": sample_ns,
        "sample_age_ms": sample_age_ms,
        "stability_value_c": raw_stability,
        "safety_value_c": raw_safety,
        "passed": raw_passed,
    }, result_marker.monotonic_ns


def validate_thermal_start_attempts(
    attempts: object,
    *,
    base_label: str,
    expected_pids: list[int],
    evidence: TelemetryEvidence,
    thermal_lock: dict[str, Any],
    result_marker_metadata: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(attempts, list)
        or not 1 <= len(attempts) <= THERMAL_QUALIFICATION_MAX_ATTEMPTS
    ):
        raise ValueError("thermal start has an invalid qualification attempt count")
    prior_end: int | None = None
    first_prepare_ns: int | None = None
    successful: dict[str, Any] | None = None
    for attempt, record in enumerate(attempts, start=1):
        if not isinstance(record, dict) or set(record) != {
            "attempt",
            "thermal_precondition",
            "qualification",
            "qualification_result_marker_monotonic_ns",
            "measured_process_states",
        }:
            raise ValueError("thermal qualification attempt has invalid fields")
        if strict_int(record.get("attempt"), "thermal attempt", minimum=1) != attempt:
            raise ValueError("thermal qualification attempts are not consecutive")
        attempt_label = f"{base_label}-attempt-{attempt:02d}"
        precondition = record.get("thermal_precondition")
        thermal_prepare_ns, thermal_cleanup_ns = validate_thermal_precondition(
            precondition, attempt_label, evidence, thermal_lock
        )
        if first_prepare_ns is None:
            first_prepare_ns = thermal_prepare_ns
        assert isinstance(precondition, dict)
        paused = validate_paused_process_states(
            record.get("measured_process_states"),
            expected_pids=expected_pids,
            label=f"{attempt_label}.measured_process_states",
        )
        qualification = record.get("qualification")
        raw, result_marker_ns = replay_thermal_qualification(
            qualification,
            attempt=attempt,
            attempt_label=attempt_label,
            precondition=precondition,
            evidence=evidence,
            thermal_lock=thermal_lock,
            result_marker_metadata=result_marker_metadata,
        )
        if record.get("qualification_result_marker_monotonic_ns") != result_marker_ns:
            raise ValueError("qualification result marker timestamp differs")
        sample_ns = raw["sample_monotonic_ns"]
        sample_is_ordered = sample_ns is None or (
            thermal_cleanup_ns
            < int(sample_ns)
            <= int(raw["qualification_monotonic_ns"])
        )
        if not (
            (prior_end is None or prior_end < thermal_prepare_ns)
            and int(raw["boundary_monotonic_ns"])
            < thermal_cleanup_ns
            < int(raw["qualification_monotonic_ns"])
            < result_marker_ns
            and sample_is_ordered
        ):
            raise ValueError("thermal qualification attempt markers overlap or reorder")
        passed = raw["passed"] is True
        if passed != (attempt == len(attempts)):
            raise ValueError("runtime did not stop at the first successful qualification")
        prior_end = result_marker_ns
        if passed:
            assert isinstance(qualification, dict)
            successful = {
                "precondition": precondition,
                "qualification": qualification,
                "raw": raw,
                "first_prepare_monotonic_ns": first_prepare_ns,
                "result_marker_monotonic_ns": result_marker_ns,
                "measured_process_states": paused,
            }
    if successful is None:
        raise ValueError("thermal qualification exhausted without a success")
    return successful


def validate_epoch_telemetry(
    epoch: dict[str, Any],
    policy_name: str,
    epoch_index: int,
    evidence: TelemetryEvidence,
    thermal_lock: dict[str, Any],
    policy_precondition: object,
) -> tuple[int, int]:
    label = f"{policy_name}.epochs[{epoch_index}]"
    metadata = {"policy": policy_name, "epoch": epoch_index}
    prepare_marker = unique_marker(evidence, "epoch_prepare", metadata)
    release_marker = unique_marker(evidence, "measurement_start", metadata)
    cleanup_marker = unique_marker(evidence, "cleanup_end", metadata)
    start_ns = strict_int(
        epoch.get("measurement_start_monotonic_ns"),
        f"{label}.measurement_start_monotonic_ns",
    )
    end_ns = strict_int(
        epoch.get("measurement_end_monotonic_ns"),
        f"{label}.measurement_end_monotonic_ns",
    )
    release_ns = strict_int(
        epoch.get("measurement_release_monotonic_ns"),
        f"{label}.measurement_release_monotonic_ns",
    )
    collected_ns = strict_int(
        epoch.get("result_collected_monotonic_ns"),
        f"{label}.result_collected_monotonic_ns",
    )
    if release_marker.monotonic_ns != release_ns:
        raise ValueError(f"{label} measurement release marker differs")
    result_marker = unique_marker(
        evidence,
        "measurement_result_collected",
        metadata
        | {
            "measurement_start_monotonic_ns": start_ns,
            "measurement_end_monotonic_ns": end_ns,
        },
    )
    if result_marker.monotonic_ns != collected_ns:
        raise ValueError(f"{label} result marker differs")
    if not (
        prepare_marker.monotonic_ns
        < release_ns
        <= start_ns
        < end_ns
        <= collected_ns
        < cleanup_marker.monotonic_ns
    ):
        raise ValueError(f"{label} marker chain is invalid")
    qualified_start: dict[str, Any] | None = None
    actual_start: dict[str, Any] | None = None
    actual_start_marker_ns: int | None = None
    if epoch_index == 0:
        readiness = epoch.get("readiness_affinity")
        if not isinstance(readiness, list) or not readiness:
            raise ValueError(f"{label} lacks measured-process readiness evidence")
        measured_pids = [
            strict_int(entry.get("pid") if isinstance(entry, dict) else None,
                       f"{label}.readiness_affinity[{index}].pid", minimum=1)
            for index, entry in enumerate(readiness)
        ]
        qualified_start = validate_thermal_start_attempts(
            epoch.get("thermal_start_attempts"),
            base_label=f"pre-{policy_name}-epoch-{epoch_index:02d}",
            expected_pids=measured_pids,
            evidence=evidence,
            thermal_lock=thermal_lock,
            result_marker_metadata=metadata,
        )
        if (
            policy_precondition != qualified_start["precondition"]
            or epoch.get("thermal_start_qualification")
            != qualified_start["qualification"]
        ):
            raise ValueError(f"{label} final thermal qualification aliases differ")
        assert isinstance(policy_precondition, dict)
        if not (
            prepare_marker.monotonic_ns
            < int(qualified_start["first_prepare_monotonic_ns"])
            < int(qualified_start["result_marker_monotonic_ns"])
            < release_ns
        ):
            raise ValueError(
                f"{label} preheater is not inside epoch 0 before measurement"
            )
        actual_start, actual_start_marker_ns = replay_actual_start_qualification(
            epoch.get("thermal_actual_start_qualification"),
            measurement_start_ns=start_ns,
            not_before_ns=int(
                qualified_start["raw"]["cleanup_end_monotonic_ns"]
            ),
            qualification_sample_ns=int(
                qualified_start["raw"]["sample_monotonic_ns"]
            ),
            evidence=evidence,
            thermal_lock=thermal_lock,
            result_marker_metadata=metadata,
        )
        if actual_start["passed"] is not True or not (
            end_ns < actual_start_marker_ns <= collected_ns
        ):
            raise ValueError(f"{label} failed actual-start qualification")
        validate_thermal_handoff(
            epoch.get("thermal_handoff"),
            boundary_ns=int(
                qualified_start["raw"]["boundary_monotonic_ns"]
            ),
            cleanup_end_ns=int(
                qualified_start["raw"]["cleanup_end_monotonic_ns"]
            ),
            qualification_ns=int(
                qualified_start["raw"]["qualification_monotonic_ns"]
            ),
            qualification_result_ns=int(
                qualified_start["result_marker_monotonic_ns"]
            ),
            release_ns=release_ns,
            measurement_start_ns=start_ns,
            thermal_lock=thermal_lock,
            label=f"{label}.thermal_handoff",
        )
        result_handoff_ms = (
            int(qualified_start["result_marker_monotonic_ns"])
            - int(qualified_start["raw"]["boundary_monotonic_ns"])
        ) / 1_000_000.0
        if result_handoff_ms >= float(thermal_lock["thermal_handoff_max_ms"]):
            raise ValueError(f"{label} qualification result exceeds the handoff bound")
    else:
        if (
            epoch.get("thermal_handoff") is not None
            or epoch.get("thermal_start_attempts") != []
            or epoch.get("thermal_start_qualification") is not None
            or epoch.get("thermal_actual_start_qualification") is not None
        ):
            raise ValueError(f"{label} has unexpected thermal qualification evidence")
    expected = replay_telemetry_aggregate(evidence, start_ns, end_ns)
    validate_telemetry_aggregate(
        epoch.get("telemetry"), expected, f"{label}.telemetry", require_healthy=False
    )
    raw_unhealthy = not bool(expected["health"]["healthy"])
    if strict_bool(epoch.get("telemetry_unhealthy"), f"{label}.telemetry_unhealthy") != raw_unhealthy:
        raise ValueError(f"{label}.telemetry_unhealthy differs from raw telemetry")
    if qualified_start is not None:
        expected_start_window = qualified_start["precondition"]["last_window"]
        start_window_ns = int(qualified_start["raw"]["boundary_monotonic_ns"])
    else:
        raw_start_window = thermal_summary_from_samples(
            evidence.samples,
            release_ns,
            sensor=str(thermal_lock["stability_sensor"]),
            window_seconds=float(thermal_lock["stability_window_seconds"]),
        )
        if raw_start_window is None:
            raise ValueError(f"{label} lacks a thermal start window")
        expected_start_window = raw_start_window
        start_window_ns = release_ns - int(
            float(thermal_lock["stability_window_seconds"]) * 1e9
        )
        validate_telemetry_coverage(
            evidence,
            start_window_ns,
            release_ns,
            f"{label}.thermal_start",
        )
    require_json_equivalent(
        epoch.get("thermal_start"), expected_start_window, f"{label}.thermal_start"
    )
    stable = thermal_summary_is_stable(expected_start_window, thermal_lock)
    if strict_bool(epoch.get("thermal_start_stable"), f"{label}.thermal_start_stable") != stable:
        raise ValueError(f"{label}.thermal_start_stable differs from raw telemetry")
    if epoch_index == 0 and not stable:
        raise ValueError(f"{label} did not begin from a stable thermal window")
    if epoch_index == 0:
        assert qualified_start is not None
        require_json_equivalent(
            epoch.get("thermal_start_telemetry"),
            qualified_start["qualification"]["telemetry"],
            f"{label}.thermal_start_telemetry",
        )
    elif epoch.get("thermal_start_telemetry") is not None:
        raise ValueError(f"{label} has unexpected thermal-start telemetry")
    validate_raw_safety_interval(
        evidence,
        start_window_ns,
        release_ns,
        thermal_lock,
        f"{label}.thermal_start",
    )
    validate_raw_safety_interval(
        evidence,
        start_ns,
        end_ns,
        thermal_lock,
        f"{label}.measurement",
    )
    stability = expected.get("temperatures_c", {}).get(
        str(thermal_lock["stability_sensor"])
    )
    if not isinstance(stability, dict):
        raise ValueError(f"{label} lacks raw stability-sensor telemetry")
    thermal_high = float(stability["max"]) > float(
        thermal_lock["target_c"]
    ) + float(thermal_lock["tolerance_c"])
    if strict_bool(epoch.get("thermal_high"), f"{label}.thermal_high") != thermal_high:
        raise ValueError(f"{label}.thermal_high differs from raw telemetry")
    return prepare_marker.monotonic_ns, cleanup_marker.monotonic_ns


def replay_run_telemetry(
    input_path: pathlib.Path,
    run: dict[str, Any],
    thermal_lock: dict[str, Any],
) -> dict[str, Any]:
    telemetry_path = input_path.parent / "telemetry.jsonl"
    evidence = load_telemetry_evidence(telemetry_path)
    config = run.get("config")
    mig = run.get("mig")
    if not isinstance(config, dict) or not isinstance(mig, dict):
        raise ValueError("formal run lacks config or MIG provenance")
    collector_ready = unique_marker(
        evidence,
        "collector_ready",
        dict(next(marker.metadata for marker in evidence.markers if marker.name == "collector_ready")),
    )
    collector_end = unique_marker(evidence, "collector_end", {})
    policy_order = config.get("policy_order")
    if not isinstance(policy_order, list):
        raise ValueError("formal run lacks policy_order")
    for marker_name in ("policy_start", "policy_end"):
        marker_metadata = [
            dict(marker.metadata)
            for marker in evidence.markers
            if marker.name == marker_name
        ]
        if marker_metadata != [{"policy": policy} for policy in policy_order]:
            raise ValueError(f"telemetry {marker_name} order differs from Williams order")
    repeats = strict_int(
        config.get("calibration_repeats"),
        "config.calibration_repeats",
        minimum=1,
    )
    expected_calibration_metadata = [
        {"stage": stage, "repeat": repeat}
        for stage in ("pre", "post")
        for repeat in range(1, repeats + 1)
    ]
    for marker_name in (
        "calibration_prepare",
        "calibration_start",
        "calibration_end",
    ):
        actual_metadata = [
            dict(marker.metadata)
            for marker in evidence.markers
            if marker.name == marker_name
        ]
        if actual_metadata != expected_calibration_metadata:
            raise ValueError(f"telemetry {marker_name} order differs from protocol")
    measurement_windows = [
        (marker.metadata.get("stage"), marker.metadata.get("repeat"))
        for marker in evidence.markers
        if marker.name == "calibration_measurement_window"
    ]
    expected_windows = [
        (metadata["stage"], metadata["repeat"])
        for metadata in expected_calibration_metadata
    ]
    if measurement_windows != expected_windows:
        raise ValueError("telemetry calibration measurement windows differ from protocol")
    expected_thermal_labels: list[str] = []
    for stage, results in (
        ("pre", run.get("isolated")),
        ("post", run.get("isolated_post")),
    ):
        if not isinstance(results, list) or len(results) != repeats:
            raise ValueError(f"{stage} calibration qualification attempts are missing")
        labels = [
            f"pre-{stage}-calibration-r{repeat}-attempt-{attempt:02d}"
            for repeat, result in enumerate(results, start=1)
            for attempt in range(
                1,
                len(result.get("thermal_start_attempts", [])) + 1
                if isinstance(result, dict)
                else 1,
            )
        ]
        if stage == "pre":
            expected_thermal_labels.extend(labels)
        else:
            post_thermal_labels = labels
    policies_by_name = {
        policy.get("name"): policy
        for policy in run.get("policies", [])
        if isinstance(policy, dict)
    }
    for policy_name in policy_order:
        policy = policies_by_name.get(policy_name)
        if not isinstance(policy, dict):
            raise ValueError(f"policy {policy_name} is missing")
        attempts = policy.get("thermal_start_attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"policy {policy_name} qualification attempts are missing")
        expected_thermal_labels.extend(
            f"pre-{policy_name}-epoch-00-attempt-{attempt:02d}"
            for attempt in range(1, len(attempts) + 1)
        )
    expected_thermal_labels.extend(post_thermal_labels)
    validate_thermal_marker_labels(evidence, expected_thermal_labels)
    actual_start_markers = [
        marker
        for marker in evidence.markers
        if marker.name == "thermal_actual_start_qualification_result"
    ]
    if len(actual_start_markers) != repeats * 2 + len(policy_order):
        raise ValueError("telemetry actual-start marker count differs from protocol")
    if run.get("sequence_precondition") is not None or run.get(
        "post_sequence_precondition"
    ) is not None:
        raise ValueError("v4 formal runs forbid sequence-level thermal preconditions")
    pre_calibration_start, prior_end = validate_calibration_markers(
        evidence, run, "pre", repeats, thermal_lock, config, mig
    )
    if collector_ready.monotonic_ns >= pre_calibration_start:
        raise ValueError("collector did not start before calibration evidence")
    for policy in run["policies"]:
        name = str(policy["name"])
        policy_start = unique_marker(evidence, "policy_start", {"policy": name})
        policy_end = unique_marker(evidence, "policy_end", {"policy": name})
        if prior_end >= policy_start.monotonic_ns:
            raise ValueError(f"policy start order is invalid for {name}")
        epochs = policy.get("epochs")
        if not isinstance(epochs, list) or not epochs:
            raise ValueError(f"{name} epochs are missing")
        previous_epoch_end = policy_start.monotonic_ns
        for epoch_index, epoch in enumerate(epochs):
            if not isinstance(epoch, dict):
                raise ValueError(f"{name} epoch {epoch_index} is invalid")
            epoch_prepare, epoch_cleanup = validate_epoch_telemetry(
                epoch,
                name,
                epoch_index,
                evidence,
                thermal_lock,
                policy.get("thermal_precondition") if epoch_index == 0 else None,
            )
            if not previous_epoch_end < epoch_prepare < epoch_cleanup:
                raise ValueError(f"epoch marker order is invalid for {name}")
            previous_epoch_end = epoch_cleanup
        epoch_zero = epochs[0]
        if (
            policy.get("thermal_start_attempts")
            != epoch_zero.get("thermal_start_attempts")
            or policy.get("thermal_start_qualification")
            != epoch_zero.get("thermal_start_qualification")
            or policy.get("thermal_actual_start_qualification")
            != epoch_zero.get("thermal_actual_start_qualification")
            or policy.get("thermal_precondition")
            != epoch_zero.get("thermal_start_attempts")[-1].get(
                "thermal_precondition"
            )
        ):
            raise ValueError(f"policy {name} thermal qualification aliases differ")
        if previous_epoch_end >= policy_end.monotonic_ns:
            raise ValueError(f"policy_end precedes epoch cleanup for {name}")
        prior_end = policy_end.monotonic_ns
    post_calibration_start, post_calibration_end = validate_calibration_markers(
        evidence, run, "post", repeats, thermal_lock, config, mig
    )
    if not (
        prior_end
        < post_calibration_start
        < post_calibration_end
        < collector_end.monotonic_ns
    ):
        raise ValueError("post-calibration marker order is invalid")
    return {
        "path": str(evidence.path),
        "sha256": evidence.sha256,
        "samples": len(evidence.samples),
        "markers": len(evidence.markers),
    }


def validate_calibration_block(
    result: dict[str, Any],
    trace: CriticalTrace,
    expected_samples: int,
    label: str,
) -> float:
    if result.get("schema_version") != 1 or result.get("role") != "benchmark":
        raise ValueError(f"{label} is not a benchmark result")
    completed = strict_int(result.get("completed_requests"), f"{label}.completed_requests")
    summary = result.get("release_to_completion")
    if not isinstance(summary, dict):
        raise ValueError(f"{label}.release_to_completion is missing")
    count = strict_int(summary.get("count"), f"{label}.release_to_completion.count")
    if completed != expected_samples or count != expected_samples:
        raise ValueError(f"{label} has an invalid sample count")
    p99 = percentile(trace.release_to_completion_ms, 0.99)
    require_close(summary.get("p99_ms"), p99, f"{label}.release_to_completion.p99_ms")
    return p99


def replay_isolated_calibrations(
    input_path: pathlib.Path,
    run: dict[str, Any],
    claimed_traces: RawTraceClaims,
) -> dict[str, float | int | bool]:
    config = run.get("config")
    if not isinstance(config, dict):
        raise ValueError("run config is missing")
    repeats = strict_int(
        config.get("calibration_repeats"), "config.calibration_repeats", minimum=1
    )
    samples = strict_int(
        config.get("samples_per_epoch"), "config.samples_per_epoch", minimum=1
    )
    raw_directory = input_path.parent / "raw"

    def replay_stage(stage: str, key: str) -> tuple[float, int, list[float]]:
        blocks = run.get(key)
        if not isinstance(blocks, list) or len(blocks) != repeats:
            raise ValueError(f"{key} must contain {repeats} calibration blocks")
        pooled_values: list[float] = []
        block_p99s: list[float] = []
        for repeat, result in enumerate(blocks, start=1):
            if not isinstance(result, dict):
                raise ValueError(f"{key}[{repeat - 1}] must be an object")
            path = raw_directory / f"isolated-{stage}-r{repeat}.csv"
            trace_snapshot = claim_raw_trace(path, claimed_traces)
            trace = read_critical_trace(path, samples)
            verify_raw_trace_snapshot(path, trace_snapshot)
            block_p99s.append(
                validate_calibration_block(
                    result, trace, samples, f"{key}[{repeat - 1}]"
                )
            )
            pooled_values.extend(trace.release_to_completion_ms)
        return percentile(pooled_values, 0.99), len(pooled_values), block_p99s

    pre_p99, pre_samples, pre_block_p99s = replay_stage("pre", "isolated")
    post_p99, post_samples, _post_block_p99s = replay_stage(
        "post", "isolated_post"
    )
    expected_total = repeats * samples
    if pre_samples != expected_total or post_samples != expected_total:
        raise ValueError("isolated calibration replay has incomplete dimensions")
    reported_pre_blocks = run.get("isolated_p99_ms")
    if not isinstance(reported_pre_blocks, list) or len(reported_pre_blocks) != repeats:
        raise ValueError("isolated_p99_ms has invalid dimensions")
    for index, expected in enumerate(pre_block_p99s):
        require_close(
            reported_pre_blocks[index], expected, f"isolated_p99_ms[{index}]"
        )
    if strict_int(run.get("isolated_pooled_samples"), "isolated_pooled_samples") != pre_samples:
        raise ValueError("isolated_pooled_samples differs from raw replay")
    if strict_int(
        run.get("isolated_post_pooled_samples"), "isolated_post_pooled_samples"
    ) != post_samples:
        raise ValueError("isolated_post_pooled_samples differs from raw replay")
    require_close(run.get("isolated_pooled_p99_ms"), pre_p99, "isolated_pooled_p99_ms")
    require_close(
        run.get("isolated_post_pooled_p99_ms"),
        post_p99,
        "isolated_post_pooled_p99_ms",
    )

    deadline = strict_float(run.get("deadline_ms"), "deadline_ms", minimum=0.0)
    factor = strict_float(config.get("slo_factor"), "config.slo_factor", minimum=0.0)
    if deadline == 0.0 or factor == 0.0:
        raise ValueError("frozen deadline and SLO factor must be positive")
    reference = deadline / factor
    require_close(
        run.get("isolated_reference_p99_ms"),
        reference,
        "isolated_reference_p99_ms",
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    pre_reference_drift = abs(pre_p99 - reference) / reference
    post_reference_drift = abs(post_p99 - reference) / reference
    pre_post_drift = abs(post_p99 - pre_p99) / pre_p99
    for key, expected in (
        ("isolated_pre_reference_drift_fraction", pre_reference_drift),
        ("isolated_post_reference_drift_fraction", post_reference_drift),
        ("isolated_drift_fraction", pre_post_drift),
    ):
        require_close(
            run.get(key),
            expected,
            key,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-12,
        )
    drift_valid = max(
        pre_reference_drift, post_reference_drift, pre_post_drift
    ) <= FORMAL_MAX_ISOLATED_DRIFT_FRACTION
    if strict_bool(run.get("isolated_drift_valid"), "isolated_drift_valid") != drift_valid:
        raise ValueError("isolated_drift_valid differs from raw replay")
    return {
        "pre_pooled_p99_ms": pre_p99,
        "post_pooled_p99_ms": post_p99,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "pre_reference_drift_fraction": pre_reference_drift,
        "post_reference_drift_fraction": post_reference_drift,
        "pre_post_drift_fraction": pre_post_drift,
        "drift_valid": drift_valid,
    }


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_with_sha256(path: pathlib.Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        decoded = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return decoded, hashlib.sha256(payload).hexdigest()


def load_verified_v4_thermal_lock(
    path: pathlib.Path,
) -> tuple[dict[str, Any], str]:
    """Hash, decode, and semantically verify one immutable lock buffer."""

    lock, digest = load_json_with_sha256(path, "thermal lock")
    verify_thermal_lock(lock)
    if (
        lock.get("schema_version") != THERMAL_LOCK_SCHEMA_VERSION
        or lock.get("thermal_handoff_boundary") != THERMAL_HANDOFF_BOUNDARY
        or lock.get("thermal_qualification_max_attempts")
        != THERMAL_QUALIFICATION_MAX_ATTEMPTS
        or lock.get("thermal_active_stable_endpoints")
        != THERMAL_ACTIVE_STABLE_ENDPOINTS
        or lock.get("thermal_active_stable_spacing_seconds")
        != THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        or "thermal_qualification_dwell_seconds" in lock
    ):
        raise ValueError(
            "thermal lock must use the active-boundary schema 4 contract"
        )
    return lock, digest


def validate_williams_blocks(runs: list[dict[str, Any]]) -> None:
    if len(runs) != len(SCHEDULED_WILLIAMS_ORDERS):
        raise ValueError("formal inputs must contain exactly 14 Williams runs")
    actual = tuple(
        tuple(run.get("config", {}).get("policy_order", ())) for run in runs
    )
    if actual != SCHEDULED_WILLIAMS_ORDERS:
        raise ValueError("formal inputs do not match the frozen Williams schedule")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--deadline-lock", type=pathlib.Path, required=True)
    parser.add_argument("--thermal-lock", type=pathlib.Path, required=True)
    parser.add_argument("--guard-lock", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not all(
        path.is_file()
        for path in (args.deadline_lock, args.thermal_lock, args.guard_lock)
    ):
        raise SystemExit("formal aggregation requires all three frozen lock files")
    try:
        deadline_lock, deadline_lock_sha256 = load_json_with_sha256(
            args.deadline_lock, "deadline lock"
        )
        thermal_lock, thermal_lock_sha256 = load_verified_v4_thermal_lock(
            args.thermal_lock
        )
        guard_lock, guard_lock_sha256 = load_json_with_sha256(
            args.guard_lock, "guard lock"
        )
        verify_deadline_lock(deadline_lock)
        verify_guard_lock(guard_lock)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"frozen lock verification failed: {error}") from error
    if (
        int(deadline_lock.get("calibration_blocks", -1)) != 10
        or int(deadline_lock.get("samples_per_block", -1)) != 9600
        or int(deadline_lock.get("isolated_samples", -1)) != 96_000
        or not math.isclose(
            float(deadline_lock.get("slo_factor", math.nan)),
            1.10,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SystemExit("deadline lock does not implement the frozen 10x9600 protocol")
    if (
        deadline_lock.get("calibration_cpu_affinity") != FORMAL_CPU_AFFINITY
        or thermal_lock.get("pilot_cpu_affinity") != FORMAL_CPU_AFFINITY
    ):
        raise SystemExit("frozen locks do not use the formal CPU affinity mapping")
    if deadline_lock.get("thermal_lock_sha256") != thermal_lock_sha256:
        raise SystemExit("deadline and thermal locks were not calibrated together")
    try:
        guard_profile = validate_guard_lock_binding(
            guard_lock,
            guard_lock_sha256,
            deadline_lock,
            thermal_lock,
            thermal_lock_sha256,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"frozen guard lock binding failed: {error}") from error
    resolved_inputs = [path.resolve() for path in args.inputs]
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise SystemExit("formal inputs must not reuse a run summary")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if any(run.get("schema_version") != 4 for run in runs):
        raise SystemExit("all inputs must use schema version 4")
    try:
        validate_williams_blocks(runs)
        for run in runs:
            config = run.get("config")
            if not isinstance(config, dict):
                raise ValueError("each input must contain a config object")
            validate_formal_protocol(config, guard_profile)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if any(
        run["config"].get("deadline_source")
        != "frozen-isolated-p99-factor"
        for run in runs
    ):
        raise SystemExit("formal inputs must use a frozen relative deadline")
    reference = comparable_config(runs[0]["config"])
    reference_artifacts = runs[0].get("artifacts")
    for run in runs:
        if comparable_config(run["config"]) != reference:
            raise SystemExit("all inputs must have the same workload configuration")
        if run.get("artifacts") != reference_artifacts:
            raise SystemExit("benchmark or TensorRT engine hashes changed across runs")
        if (
            run["config"].get("deadline_lock_sha256") != deadline_lock_sha256
            or run["config"].get("thermal_lock_sha256") != thermal_lock_sha256
            or run["config"].get("guard_lock_sha256") != guard_lock_sha256
        ):
            raise SystemExit("a formal run is not bound to the supplied locks")
        if run["config"].get("cpu_affinity") != deadline_lock.get(
            "calibration_cpu_affinity"
        ):
            raise SystemExit("a formal run differs from calibration CPU affinity")
        thermal_scalar_fields = {
            "thermal_target_c": "target_c",
            "thermal_tolerance_c": "tolerance_c",
            "thermal_window_seconds": "stability_window_seconds",
            "thermal_max_slope_c_per_minute": "maximum_slope_c_per_minute",
            "thermal_hard_limit_c": "hard_limit_c",
            "thermal_handoff_max_ms": "thermal_handoff_max_ms",
            "thermal_active_stable_spacing_seconds": (
                "thermal_active_stable_spacing_seconds"
            ),
            "tegrastats_requested_interval_ms": (
                "tegrastats_requested_interval_ms"
            ),
            "telemetry_interval_ms": "telemetry_interval_ms",
            "telemetry_required_fraction": "telemetry_required_fraction",
            "telemetry_max_gap_ms": "telemetry_max_gap_ms",
        }
        if not math.isclose(
            float(run["deadline_ms"]),
            float(deadline_lock["deadline_ms"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(run["config"]["slo_factor"]),
            float(deadline_lock["slo_factor"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit("a formal run differs from frozen deadline values")
        if any(
            not math.isclose(
                float(run["config"].get(config_name, math.nan)),
                float(thermal_lock.get(lock_name, math.nan)),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for config_name, lock_name in thermal_scalar_fields.items()
        ):
            raise SystemExit("a formal run differs from frozen thermal values")
        thermal_string_fields = {
            "thermal_stability_sensor": "stability_sensor",
            "thermal_safety_sensor": "safety_sensor",
            "thermal_handoff_boundary": "thermal_handoff_boundary",
        }
        if any(
            run["config"].get(config_name) != thermal_lock.get(lock_name)
            for config_name, lock_name in thermal_string_fields.items()
        ) or run["config"].get("thermal_qualification_max_attempts") != (
            thermal_lock.get("thermal_qualification_max_attempts")
        ) or run["config"].get("thermal_active_stable_endpoints") != (
            thermal_lock.get("thermal_active_stable_endpoints")
        ) or run["config"].get("telemetry_required_fields") != thermal_lock.get(
            "telemetry_required_fields"
        ):
            raise SystemExit("a formal run differs from frozen thermal semantics")
        if (
            run.get("artifacts") != deadline_lock.get("calibration_artifacts")
            or run.get("hardware") != deadline_lock.get("calibration_hardware")
            or run.get("mig") != deadline_lock.get("calibration_mig")
        ):
            raise SystemExit("a formal run differs from calibration provenance")
        policies = run.get("policies")
        if not isinstance(policies, list) or any(
            not isinstance(policy, dict) for policy in policies
        ):
            raise SystemExit("each input must contain a policy list")
        names = [policy.get("name") for policy in policies]
        if set(names) != POLICIES or len(names) != len(POLICIES):
            raise SystemExit("each input must contain every policy exactly once")
        if names != run["config"].get("policy_order"):
            raise SystemExit("policy summaries must follow config.policy_order")
    deadlines = [float(run["deadline_ms"]) for run in runs]
    if max(deadlines) - min(deadlines) > 1e-9:
        raise SystemExit("formal inputs must use one frozen deadline")

    raw_values: dict[str, list[float]] = {name: [] for name in POLICIES}
    raw_trace_claims = RawTraceClaims()
    validated_runs: list[dict[str, dict[str, float]]] = []
    isolated_replays: list[dict[str, float | int | bool]] = []
    telemetry_inputs: list[dict[str, Any]] = []
    try:
        for input_path, run in zip(args.inputs, runs, strict=True):
            telemetry_inputs.append(
                replay_run_telemetry(input_path, run, thermal_lock)
            )
            isolated_replays.append(
                replay_isolated_calibrations(input_path, run, raw_trace_claims)
            )
            deadline = strict_float(run.get("deadline_ms"), "deadline_ms", minimum=0.0)
            validated_policies: dict[str, dict[str, float]] = {}
            for policy in run["policies"]:
                metrics, policy_values = recompute_policy_metrics(
                    policy,
                    input_path,
                    run["config"],
                    deadline,
                    raw_trace_claims,
                    run["mig"],
                    run["artifacts"],
                    guard_profile,
                )
                name = str(policy["name"])
                validated_policies[name] = metrics
                raw_values[name].extend(policy_values)
            validated_runs.append(validated_policies)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    telemetry_hashes = [item["sha256"] for item in telemetry_inputs]
    if len(telemetry_hashes) != len(set(telemetry_hashes)):
        raise SystemExit("formal runs must not reuse byte-identical telemetry JSONL")

    aggregate: dict[str, Any] = {}
    for name in sorted(POLICIES):
        per_run = [run[name] for run in validated_runs]
        aggregate[name] = {
            metric: confidence([sample[metric] for sample in per_run])
            for metric in per_run[0]
        }

    governor = aggregate[PROPOSED_POLICY_ID]
    comparisons: dict[str, Any] = {}
    for baseline_name in sorted(POLICIES - {PROPOSED_POLICY_ID}):
        baseline = aggregate[baseline_name]
        comparisons[baseline_name] = {
            "deadline_miss_rate_change": relative_change(
                governor["deadline_miss_rate"]["mean"],
                baseline["deadline_miss_rate"]["mean"],
            ),
            "goodput_change": relative_change(
                governor["pressure_goodput_per_second"]["mean"],
                baseline["pressure_goodput_per_second"]["mean"],
            ),
        }

    target = float(runs[0]["config"]["dmr_target"])
    drift_valid = all(bool(replay["drift_valid"]) for replay in isolated_replays)
    pooled: dict[str, dict[str, float | int | bool]] = {}
    feasible: dict[str, bool] = {}
    for name in sorted(POLICIES):
        per_run = [run[name] for run in validated_runs]
        requests = len(raw_values[name])
        misses = sum(value > deadlines[0] for value in raw_values[name])
        upper = clopper_pearson_upper(misses, requests)
        rejected = sum(int(policy["rejected_tenants"]) for policy in per_run)
        unhealthy = sum(
            int(policy["telemetry_unhealthy_epochs"]) for policy in per_run
        )
        feasible[name] = (
            drift_valid and upper <= target and rejected == 0 and unhealthy == 0
        )
        pooled[name] = {
            "critical_requests": requests,
            "deadline_misses": misses,
            "observed_deadline_miss_rate": misses / requests,
            "deadline_miss_rate_cp95_upper": upper,
            "release_to_completion_p99_ms": percentile(raw_values[name], 0.99),
            "release_to_completion_p999_ms": percentile(raw_values[name], 0.999),
            "release_to_completion_p9995_ms": percentile(
                raw_values[name], 0.9995
            ),
            "rejected_tenants": rejected,
            "telemetry_unhealthy_epochs": unhealthy,
            "feasible": feasible[name],
        }
    baseline_names = sorted(
        name
        for name in POLICIES - {PROPOSED_POLICY_ID}
        if feasible[name]
    )
    best_baseline = (
        max(
            baseline_names,
            key=lambda name: aggregate[name]["pressure_goodput_per_second"][
                "mean"
            ],
        )
        if baseline_names
        else None
    )
    governor_goodput = governor["pressure_goodput_per_second"]["mean"]
    # The primary contrast isolates protected 2g slack borrowing from a
    # resident-only, fully quiesced MIG baseline. fixed-full-gate remains the
    # action-matched secondary contrast for incremental feedback behavior.
    primary_baseline = PRIMARY_BASELINE
    adaptive_baseline = ADAPTIVE_BASELINE
    paired_gain: dict[str, float | int] | None = None
    if feasible[PROPOSED_POLICY_ID] and feasible[primary_baseline]:
        gains = []
        for policies in validated_runs:
            baseline_goodput = policies[primary_baseline][
                "pressure_goodput_per_second"
            ]
            if baseline_goodput <= 0.0:
                raise SystemExit("a feasible baseline reported non-positive goodput")
            gains.append(
                policies[PROPOSED_POLICY_ID]["pressure_goodput_per_second"]
                / baseline_goodput
                - 1.0
            )
        paired_gain = confidence(gains)
    adaptive_paired_gain: dict[str, float | int] | None = None
    if feasible[PROPOSED_POLICY_ID] and feasible[adaptive_baseline]:
        adaptive_gains = []
        for policies in validated_runs:
            baseline_goodput = policies[adaptive_baseline][
                "pressure_goodput_per_second"
            ]
            if baseline_goodput <= 0.0:
                raise SystemExit(
                    "the feasible action-matched baseline reported non-positive goodput"
                )
            adaptive_gains.append(
                policies[PROPOSED_POLICY_ID]["pressure_goodput_per_second"]
                / baseline_goodput
                - 1.0
            )
        adaptive_paired_gain = confidence(adaptive_gains)
    if feasible[PROPOSED_POLICY_ID] and not feasible[adaptive_baseline]:
        adaptivity_feasibility_status = "quiet-feasible-no-feedback-infeasible"
    elif feasible[PROPOSED_POLICY_ID] and feasible[adaptive_baseline]:
        adaptivity_feasibility_status = "both-feasible"
    elif not feasible[PROPOSED_POLICY_ID] and feasible[adaptive_baseline]:
        adaptivity_feasibility_status = "quiet-infeasible-no-feedback-feasible"
    else:
        adaptivity_feasibility_status = "neither-feasible"
    adaptive_action_epochs = sum(
        int(run[PROPOSED_POLICY_ID]["adaptive_action_epochs"])
        for run in validated_runs
    )
    adaptive_action_runs = sum(
        int(run[PROPOSED_POLICY_ID]["adaptive_action_epochs"] > 0.0)
        for run in validated_runs
    )
    adaptive_gain_supported = (
        adaptive_paired_gain is not None
        and float(adaptive_paired_gain["mean"]) > 0.0
        and float(adaptive_paired_gain["mean"])
        - float(adaptive_paired_gain["ci95"])
        > 0.0
    )
    adaptivity_status = adaptive_claim_status(
        adaptive_action_epochs=adaptive_action_epochs,
        adaptive_action_runs=adaptive_action_runs,
        total_runs=len(validated_runs),
        drift_valid=drift_valid,
        governor_feasible=feasible[PROPOSED_POLICY_ID],
        baseline_admission_valid=(
            int(pooled[adaptive_baseline]["rejected_tenants"]) == 0
        ),
        baseline_telemetry_valid=(
            int(pooled[adaptive_baseline]["telemetry_unhealthy_epochs"]) == 0
        ),
        baseline_slo_feasible=(
            float(pooled[adaptive_baseline]["deadline_miss_rate_cp95_upper"])
            <= target
        ),
        paired_gain_supported=adaptive_gain_supported,
    )
    supported = (
        paired_gain is not None
        and float(paired_gain["mean"]) > 0.0
        and float(paired_gain["mean"]) - float(paired_gain["ci95"]) > 0.0
    )
    output = {
        "schema_version": 4,
        "config": runs[0]["config"],
        "input_files": [str(path) for path in args.inputs],
        "deadline_lock": {
            "path": str(args.deadline_lock),
            "sha256": deadline_lock_sha256,
        },
        "thermal_lock": {
            "path": str(args.thermal_lock),
            "sha256": thermal_lock_sha256,
        },
        "guard_lock": {
            "path": str(args.guard_lock),
            "sha256": guard_lock_sha256,
        },
        "hardware": runs[0]["hardware"],
        "mig": runs[0]["mig"],
        "policy_orders": [run["config"]["policy_order"] for run in runs],
        "telemetry_inputs": telemetry_inputs,
        "raw_trace_files_verified": len(raw_trace_claims.paths),
        "raw_trace_inputs": raw_trace_claims.provenance,
        "isolated_p99_ms": confidence(
            [
                float(replay["pre_pooled_p99_ms"])
                for replay in isolated_replays
            ]
        ),
        "isolated_post_p99_ms": confidence(
            [
                float(replay["post_pooled_p99_ms"])
                for replay in isolated_replays
            ]
        ),
        "isolated_pre_post_drift_fraction": confidence(
            [
                float(replay["pre_post_drift_fraction"])
                for replay in isolated_replays
            ]
        ),
        "deadline_ms": confidence([float(run["deadline_ms"]) for run in runs]),
        "presentation": {
            "proposed_system": PROPOSED_SYSTEM,
            "proposed_policy_id": PROPOSED_POLICY_ID,
            "measured_rows": [
                {
                    "policy_id": policy_id,
                    "label": POLICY_PRESENTATION[policy_id][0],
                    "role": POLICY_PRESENTATION[policy_id][1],
                }
                for policy_id in sorted(POLICIES)
            ],
            "comparison_contract": (
                "Only QUIET is the proposed system. Baselines and ablations are "
                "not relabeled as published SOTA systems."
            ),
        },
        "policies": aggregate,
        "pooled_slo": pooled,
        "quiet_vs_measured_controls": comparisons,
        "headline": {
            "claim_status": "supported" if supported else "not-supported",
            "all_runs_isolated_drift_valid": drift_valid,
            "pooled_slo_admission_telemetry_feasible": feasible,
            "primary_baseline_policy_id": primary_baseline,
            "no_feedback_ablation_policy_id": adaptive_baseline,
            "adaptivity_status": adaptivity_status,
            "adaptivity_feasibility_status": adaptivity_feasibility_status,
            "adaptive_action_epochs": adaptive_action_epochs,
            "adaptive_action_runs": adaptive_action_runs,
            "adaptive_action_runs_required": len(validated_runs),
            "best_feasible_baseline_descriptive": best_baseline,
            "quiet_goodput_gain_over_primary_baseline": (
                relative_change(
                    governor_goodput,
                    aggregate[primary_baseline]["pressure_goodput_per_second"][
                        "mean"
                    ],
                )
                if feasible[PROPOSED_POLICY_ID] and feasible[primary_baseline]
                else None
            ),
            "paired_quiet_goodput_gain": paired_gain,
            "paired_quiet_goodput_gain_over_no_feedback_ablation": (
                adaptive_paired_gain
            ),
            "quiet_deadline_miss_rate": governor["deadline_miss_rate"][
                "mean"
            ],
        },
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
