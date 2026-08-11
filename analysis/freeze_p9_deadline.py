#!/usr/bin/env python3
"""Create or verify a frozen P9 deadline calibration lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any

ANALYSIS_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(ANALYSIS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIRECTORY))
from freeze_p9_thermal import (  # noqa: E402
    LOCK_SCHEMA_VERSION as THERMAL_LOCK_SCHEMA_VERSION,
    SAFETY_SENSOR as THERMAL_SAFETY_SENSOR,
    STABILITY_SENSOR as THERMAL_STABILITY_SENSOR,
    TARGET_SOURCE as THERMAL_TARGET_SOURCE,
    THERMAL_HANDOFF_BOUNDARY,
    THERMAL_HANDOFF_MAX_MS,
    THERMAL_HANDOFF_RATIONALE,
    THERMAL_ACTIVE_STABLE_ENDPOINTS,
    THERMAL_ACTIVE_STABLE_SPACING_SECONDS,
    THERMAL_QUALIFICATION_MAX_ATTEMPTS,
    THERMAL_REQUIRED_FIELDS,
    TelemetryRecords,
    load_telemetry_jsonl,
    verify_lock as verify_thermal_lock,
)
from freeze_p9_guard import verify_lock as verify_guard_lock  # noqa: E402
from runtime.tegrastats_telemetry import (  # noqa: E402
    TelemetrySample,
    aggregate_samples,
    parse_tegrastats_line,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
HASHED_FILES = (
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

TRACE_COLUMNS = (
    "request",
    "release_to_completion_ms",
    "gpu_service_ms",
    "queue_delay_ms",
    "gate_overhead_ms",
    "drain_ms",
    "resume_ms",
)
TRACE_SUMMARIES = {
    "release_to_completion": "release_to_completion_ms",
    "gpu_service": "gpu_service_ms",
    "queue_delay": "queue_delay_ms",
    "gate_overhead": "gate_overhead_ms",
    "drain": "drain_ms",
    "resume": "resume_ms",
}
GUARD_IMPLEMENTATION_ARTIFACTS = {
    "producer": "runtime/profile_p9_guard.py",
    "freezer": "analysis/freeze_p9_guard.py",
    "telemetry_runtime": "runtime/tegrastats_telemetry.py",
    "governor_runtime": "runtime/mig_slack_governor.py",
    "guard_runner": "scripts/run_p9_guard_calibration.sh",
    "formal_runner": "scripts/run_p9_mig_slack_governor.sh",
    "mig_configurator": "scripts/configure_thor_mig.sh",
    "benchmark_source": "benchmarks/trt_inference.cpp",
}
GUARD_MODELS = {
    "language": "distilbert-sst2",
    "audio": "whisper-tiny-encoder",
}
GUARD_QUOTAS = {
    "resident-1g": (25, 50, 100),
    "borrower-2g": (100,),
}
THERMAL_WINDOW_FIELDS = {
    "samples",
    "window_seconds",
    "observed_span_seconds",
    "mean_c",
    "min_c",
    "max_c",
    "latest_c",
    "slope_c_per_minute",
    "maximum_gap_seconds",
}
THERMAL_PRECONDITION_WINDOW_FIELDS = THERMAL_WINDOW_FIELDS


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hashes() -> dict[str, str]:
    return {name: file_sha256(ROOT / name) for name in HASHED_FILES}


def load_json_buffer(
    path: pathlib.Path, label: str
) -> tuple[dict[str, Any], bytes, str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value, payload, hashlib.sha256(payload).hexdigest()


def guard_profile_from_lock(lock: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    raw_guards = lock.get("guards")
    if not isinstance(raw_guards, dict) or set(raw_guards) != set(GUARD_QUOTAS):
        raise ValueError("guard lock has invalid placement coverage")
    profile: dict[str, dict[str, dict[str, float]]] = {}
    for placement, quotas in GUARD_QUOTAS.items():
        raw_placement = raw_guards.get(placement)
        expected_quotas = {str(quota) for quota in quotas}
        if not isinstance(raw_placement, dict) or set(raw_placement) != expected_quotas:
            raise ValueError(f"guard lock has invalid quotas for {placement}")
        profile[placement] = {}
        for quota in quotas:
            raw_modalities = raw_placement[str(quota)]
            if not isinstance(raw_modalities, dict) or set(raw_modalities) != set(
                GUARD_MODELS
            ):
                raise ValueError(
                    f"guard lock has invalid modalities for {placement}/q{quota}"
                )
            profile[placement][str(quota)] = {}
            for modality in GUARD_MODELS:
                evidence = raw_modalities[modality]
                if not isinstance(evidence, dict):
                    raise ValueError("guard lock profile evidence must be an object")
                value = strict_float(
                    evidence.get("guard_ms"),
                    f"guard.{placement}.q{quota}.{modality}",
                )
                if value <= 0.0 or value >= 20.0:
                    raise ValueError("guard lock contains an unusable formal guard")
                profile[placement][str(quota)][modality] = value
    return profile


def _guard_artifact(
    artifacts: dict[str, Any], name: str
) -> tuple[pathlib.Path, str]:
    record = artifacts.get(name)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"guard lock artifact is malformed: {name}")
    path = pathlib.Path(str(record.get("path", "")))
    digest = record.get("sha256")
    if (
        not path.is_absolute()
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"guard lock artifact provenance is invalid: {name}")
    return path.resolve(), digest


def validate_guard_platform_and_artifacts(
    guard_lock: dict[str, Any],
    *,
    hardware: dict[str, Any],
    mig: dict[str, Any],
    cpu_affinity: object,
    calibration_artifacts: dict[str, Any],
) -> None:
    if (
        guard_lock.get("schema_version") != 3
        or guard_lock.get("kind") != "p9-quota-aware-guard-lock"
    ):
        raise ValueError("supplied guard lock has an invalid schema or kind")
    if guard_lock.get("hardware") != hardware:
        raise ValueError("deadline calibration hardware differs from the guard lock")
    guard_mig = guard_lock.get("mig")
    if not isinstance(guard_mig, dict) or (
        guard_mig.get("big_uuid") != mig.get("critical_uuid")
        or guard_mig.get("small_uuid") != mig.get("resident_uuid")
    ):
        raise ValueError("deadline calibration MIG mapping differs from the guard lock")
    if guard_lock.get("cpu_affinity") != cpu_affinity:
        raise ValueError("deadline calibration CPU affinity differs from the guard lock")

    guard_artifacts = guard_lock.get("artifacts")
    implementation = calibration_artifacts.get("implementation_sha256")
    engines = calibration_artifacts.get("engines_sha256")
    if (
        not isinstance(guard_artifacts, dict)
        or not isinstance(implementation, dict)
        or not isinstance(engines, dict)
    ):
        raise ValueError("guard/deadline artifact provenance is incomplete")
    _benchmark_path, benchmark_digest = _guard_artifact(
        guard_artifacts, "benchmark"
    )
    if benchmark_digest != calibration_artifacts.get("benchmark_sha256"):
        raise ValueError("deadline benchmark differs from the guard calibration")
    for guard_name, relative_path in GUARD_IMPLEMENTATION_ARTIFACTS.items():
        path, digest = _guard_artifact(guard_artifacts, guard_name)
        if (
            path != (ROOT / relative_path).resolve()
            or implementation.get(relative_path) != digest
        ):
            raise ValueError(
                f"deadline implementation differs from guard artifact {guard_name}"
            )
    for placement, quotas in GUARD_QUOTAS.items():
        for quota in quotas:
            for modality, model in GUARD_MODELS.items():
                guard_name = f"engine:{placement}:q{quota}:{modality}"
                _path, digest = _guard_artifact(guard_artifacts, guard_name)
                prefix = "resident-1g" if placement == "resident-1g" else "borrower-2g"
                deadline_name = f"{prefix}-q{quota}-{model}"
                if engines.get(deadline_name) != digest:
                    raise ValueError(
                        f"deadline engine differs from guard artifact {guard_name}"
                    )
    _critical_path, critical_digest = _guard_artifact(
        guard_artifacts, "engine:critical:2g:resnet50-v2"
    )
    if engines.get("critical-2g-resnet50-v2") != critical_digest:
        raise ValueError("deadline critical engine differs from the guard calibration")


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile requires samples and a quantile in [0, 1]")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def strict_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def read_calibration_trace(
    path: pathlib.Path, expected_samples: int
) -> dict[str, list[float]]:
    values = {column: [] for column in TRACE_COLUMNS[1:]}
    if not path.is_file():
        raise ValueError(f"missing calibration trace: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
            raise ValueError(f"invalid calibration trace header: {path}")
        for expected_request, row in enumerate(reader):
            if set(row) != set(TRACE_COLUMNS) or any(
                row[column] is None for column in TRACE_COLUMNS
            ):
                raise ValueError(f"invalid calibration trace row: {path}")
            try:
                request = int(row["request"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid calibration request sequence: {path}"
                ) from error
            if str(request) != row["request"] or request != expected_request:
                raise ValueError(f"invalid calibration request sequence: {path}")
            for column in TRACE_COLUMNS[1:]:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid calibration latency: {path}") from error
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"invalid calibration latency: {path}")
                values[column].append(value)
    if any(len(column_values) != expected_samples for column_values in values.values()):
        raise ValueError(f"incomplete calibration trace: {path}")
    return values


def validate_latency_summary(
    reported: object, values: list[float], label: str
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
    if not isinstance(reported, dict) or set(reported) != expected_fields:
        raise ValueError(f"{label} has invalid latency-summary fields")
    if strict_int(reported.get("count"), f"{label}.count") != len(values):
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
        actual = strict_float(reported.get(field), f"{label}.{field}")
        if not math.isclose(
            actual, expected_value, rel_tol=1e-9, abs_tol=1e-7
        ):
            raise ValueError(f"{label}.{field} differs from the raw trace")


def validate_calibration_result(
    result: object,
    trace: dict[str, list[float]],
    *,
    label: str,
    expected_samples: int,
    config: dict[str, Any],
    critical_uuid: str,
    critical_engine: pathlib.Path,
) -> None:
    expected_fields = {
        "schema_version",
        "model",
        "role",
        "engine",
        "execution_environment",
        "gpu",
        "config",
        *TRACE_SUMMARIES,
        "completed_requests",
        "throughput_per_second",
        "measurement_start_monotonic_ns",
        "measurement_end_monotonic_ns",
        "elapsed_seconds",
        "deadline_misses",
        "deadline_miss_rate",
        "thermal_start",
        "thermal_start_telemetry",
        "thermal_start_stable",
        "thermal_start_attempts",
        "thermal_precondition",
        "thermal_start_qualification",
        "thermal_actual_start_qualification",
        "thermal_handoff",
        "thermal_precondition_label",
        "measurement_release_monotonic_ns",
        "readiness_affinity",
    }
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise ValueError(f"{label} has invalid benchmark-result fields")
    if (
        strict_int(result.get("schema_version"), f"{label}.schema_version") != 1
        or result.get("model") != "resnet50-v2"
        or result.get("role") != "benchmark"
        or result.get("engine") != str(critical_engine)
    ):
        raise ValueError(f"{label} is not the frozen ResNet50 calibration")

    environment = result.get("execution_environment")
    if not isinstance(environment, dict) or set(environment) != {
        "pid",
        "cuda_visible_devices",
        "mps_active_thread_percentage",
        "cpu_affinity",
    }:
        raise ValueError(f"{label} has invalid execution-environment fields")
    strict_int(
        environment.get("pid"), f"{label}.execution_environment.pid", minimum=1
    )
    cpu_affinity = environment.get("cpu_affinity")
    if (
        environment.get("cuda_visible_devices") != critical_uuid
        or strict_int(
            environment.get("mps_active_thread_percentage"),
            f"{label}.execution_environment.mps_active_thread_percentage",
            minimum=1,
        )
        != 100
        or not isinstance(cpu_affinity, list)
        or len(cpu_affinity) != 1
        or strict_int(
            cpu_affinity[0], f"{label}.execution_environment.cpu_affinity[0]"
        )
        != 12
    ):
        raise ValueError(f"{label} differs from the frozen execution environment")
    gpu = result.get("gpu")
    if (
        not isinstance(gpu, dict)
        or set(gpu) != {"name", "multiprocessors"}
        or gpu.get("name") != "NVIDIA Thor MIG 2g.0gb"
        or strict_int(gpu.get("multiprocessors"), f"{label}.gpu.multiprocessors")
        != 12
    ):
        raise ValueError(f"{label} differs from the frozen 2g MIG width")

    benchmark_config = result.get("config")
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
    if not isinstance(benchmark_config, dict) or set(benchmark_config) != expected_config_fields:
        raise ValueError(f"{label}.config has invalid fields")
    expected_integers = {
        "warmup": strict_int(config.get("warmup"), "config.warmup"),
        "burst_size": strict_int(config.get("burst_size"), "config.burst_size"),
        "gated_processes": 0,
        "stopped_processes": 0,
    }
    for field, expected_value in expected_integers.items():
        if strict_int(
            benchmark_config.get(field), f"{label}.config.{field}"
        ) != expected_value:
            raise ValueError(f"{label}.config differs from the frozen protocol")
    if strict_int(
        benchmark_config.get("stream_priority_value"),
        f"{label}.config.stream_priority_value",
        minimum=-5,
    ) != -5:
        raise ValueError(f"{label}.config differs from the frozen protocol")
    for field, expected_value in {
        "period_ms": strict_float(config.get("period_ms"), "config.period_ms"),
        "deadline_ms": 0.0,
        "duration_seconds": 0.0,
        "guard_ms": 0.0,
    }.items():
        actual = strict_float(benchmark_config.get(field), f"{label}.config.{field}")
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{label}.config differs from the frozen protocol")
    if (
        benchmark_config.get("gate_mode") != "stop"
        or benchmark_config.get("start_paused") is not True
        or benchmark_config.get("include_transfers") is not True
        or benchmark_config.get("priority") != "high"
    ):
        raise ValueError(f"{label}.config differs from the frozen protocol")

    completed = strict_int(
        result.get("completed_requests"), f"{label}.completed_requests", minimum=1
    )
    if completed != expected_samples:
        raise ValueError(f"{label}.completed_requests differs from the protocol")
    start_ns = strict_int(
        result.get("measurement_start_monotonic_ns"),
        f"{label}.measurement_start_monotonic_ns",
        minimum=1,
    )
    end_ns = strict_int(
        result.get("measurement_end_monotonic_ns"),
        f"{label}.measurement_end_monotonic_ns",
        minimum=1,
    )
    if end_ns <= start_ns:
        raise ValueError(f"{label} has invalid measurement clocks")
    elapsed = (end_ns - start_ns) / 1_000_000_000.0
    if not math.isclose(
        strict_float(result.get("elapsed_seconds"), f"{label}.elapsed_seconds"),
        elapsed,
        rel_tol=5e-10,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{label}.elapsed_seconds differs from its clocks")
    if not math.isclose(
        strict_float(
            result.get("throughput_per_second"), f"{label}.throughput_per_second"
        ),
        completed / elapsed,
        rel_tol=5e-10,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{label}.throughput_per_second differs from its clocks")
    for summary_name, column in TRACE_SUMMARIES.items():
        validate_latency_summary(
            result.get(summary_name), trace[column], f"{label}.{summary_name}"
        )
    if (
        strict_int(result.get("deadline_misses"), f"{label}.deadline_misses") != 0
        or result.get("deadline_miss_rate") is not None
    ):
        raise ValueError(f"{label} has invalid deadline-disabled metrics")


def _structurally_valid_thermal_sample(
    record: dict[str, Any],
    *,
    stability_sensor: str,
    safety_sensor: str,
) -> bool:
    parsed = record.get("parsed", {})
    cpu = parsed.get("cpu", [])
    temperatures = parsed.get("temperatures_c", {})
    power = parsed.get("power", {}).get("VIN", {})
    return (
        isinstance(parsed.get("ram"), dict)
        and isinstance(cpu, list)
        and any(
            isinstance(core, dict)
            and isinstance(core.get("utilization_pct"), (int, float))
            and not isinstance(core.get("utilization_pct"), bool)
            and math.isfinite(float(core["utilization_pct"]))
            for core in cpu
        )
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (
                record.get("mem_available_mb"),
                temperatures.get(stability_sensor),
                temperatures.get(safety_sensor),
                power.get("current_mw"),
            )
        )
    )


def _maximum_gap_seconds(
    timestamps: list[int], start_ns: int, end_ns: int
) -> float | None:
    if not timestamps:
        return None
    gaps = [timestamps[0] - start_ns, end_ns - timestamps[-1]]
    gaps.extend(
        current - previous
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
    )
    return max(gaps) / 1_000_000_000.0


def _temperature_statistics(
    points: list[tuple[int, float]],
    *,
    window_seconds: float,
    window_start_ns: int,
    reference_ns: int,
) -> dict[str, float | int]:
    timestamps = [timestamp for timestamp, _value in points]
    values = [value for _timestamp, value in points]
    times = [
        (timestamp - timestamps[0]) / 1_000_000_000.0
        for timestamp in timestamps
    ]
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
        "maximum_gap_seconds": _maximum_gap_seconds(
            timestamps, window_start_ns, reference_ns
        ),
    }


def replay_raw_thermal_window(
    telemetry: TelemetryRecords,
    *,
    reference_ns: int,
    not_before_ns: int,
    window_seconds: float,
    interval_ms: float,
    required_fraction: float,
    maximum_gap_ms: float,
    stability_sensor: str,
    safety_sensor: str,
    hard_limit_c: float,
    end_inclusive: bool = False,
) -> dict[str, Any]:
    if reference_ns <= not_before_ns:
        raise ValueError("calibration thermal window has invalid clocks")
    window_start_ns = max(
        not_before_ns,
        reference_ns - int(window_seconds * 1_000_000_000),
    )
    samples, _markers = telemetry
    selected = [
        record
        for record in samples
        if window_start_ns <= int(record["monotonic_ns"])
        and (
            int(record["monotonic_ns"]) < reference_ns
            or (
                end_inclusive
                and int(record["monotonic_ns"]) == reference_ns
            )
        )
    ]
    valid = [
        record
        for record in selected
        if _structurally_valid_thermal_sample(
            record,
            stability_sensor=stability_sensor,
            safety_sensor=safety_sensor,
        )
    ]
    expected_samples = math.floor(window_seconds * 1000.0 / interval_ms)
    required_samples = max(1, math.floor(expected_samples * required_fraction))
    if len(valid) != len(selected) or len(valid) < required_samples:
        raise ValueError(
            "calibration thermal window lacks complete required-field coverage"
        )
    stability_points = [
        (
            int(record["monotonic_ns"]),
            float(record["parsed"]["temperatures_c"][stability_sensor]),
        )
        for record in valid
    ]
    safety_points = [
        (
            int(record["monotonic_ns"]),
            float(record["parsed"]["temperatures_c"][safety_sensor]),
        )
        for record in valid
    ]
    if len(stability_points) < 2 or len(safety_points) < 2:
        raise ValueError("calibration thermal window has too few sensor samples")
    stability = _temperature_statistics(
        stability_points,
        window_seconds=window_seconds,
        window_start_ns=window_start_ns,
        reference_ns=reference_ns,
    )
    safety = _temperature_statistics(
        safety_points,
        window_seconds=window_seconds,
        window_start_ns=window_start_ns,
        reference_ns=reference_ns,
    )
    for sensor, evidence in (
        (stability_sensor, stability),
        (safety_sensor, safety),
    ):
        if (
            int(evidence["samples"]) < required_samples
            or float(evidence["observed_span_seconds"])
            < window_seconds * 0.99
            or not math.isfinite(float(evidence["maximum_gap_seconds"]))
            or float(evidence["maximum_gap_seconds"])
            > maximum_gap_ms / 1000.0
        ):
            raise ValueError(
                f"calibration {sensor} thermal window is incomplete or stale"
            )
    if float(safety["max_c"]) >= hard_limit_c:
        raise ValueError("calibration TJ safety window reached the hard limit")
    return {
        "interval": {
            "start_ns": window_start_ns,
            "end_ns": reference_ns,
        },
        "total_samples": len(selected),
        "valid_samples": len(valid),
        "required_fields": list(THERMAL_REQUIRED_FIELDS),
        "stability_sensor": stability_sensor,
        "safety_sensor": safety_sensor,
        "stability_window": stability,
        "safety_window": safety,
    }


def _compare_thermal_window(
    reported: object,
    replayed: dict[str, float | int],
    *,
    expected_fields: set[str],
    label: str,
) -> None:
    if not isinstance(reported, dict) or set(reported) != expected_fields:
        raise ValueError(f"{label} has invalid fields")
    for field in expected_fields:
        expected = replayed[field]
        if field == "samples":
            if type(reported.get(field)) is not int or reported[field] != expected:
                raise ValueError(f"{label}.{field} differs from raw telemetry")
        elif not math.isclose(
            strict_float(reported.get(field), f"{label}.{field}"),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{label}.{field} differs from raw telemetry")


def _numeric_temperature_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def validate_stored_thermal_aggregate(
    stored: object,
    evidence: dict[str, Any],
    telemetry: TelemetryRecords,
    *,
    maximum_gap_ms: float,
    label: str,
    end_inclusive: bool = False,
) -> None:
    if not isinstance(stored, dict):
        raise ValueError(f"{label} lacks its telemetry aggregate")
    interval = evidence["interval"]
    if stored.get("interval") != {
        "start_ns": interval["start_ns"],
        "end_ns": interval["end_ns"],
        "duration_ns": interval["end_ns"] - interval["start_ns"],
        "end_inclusive": end_inclusive,
    }:
        raise ValueError(f"{label} telemetry interval differs from raw clocks")
    if (
        stored.get("total_samples") != evidence["total_samples"]
        or stored.get("valid_samples") != evidence["valid_samples"]
        or stored.get("invalid_samples") != 0
    ):
        raise ValueError(f"{label} telemetry counts differ from raw telemetry")
    health = stored.get("health")
    observed_gap_ns = int(
        round(
            float(evidence["stability_window"]["maximum_gap_seconds"])
            * 1_000_000_000.0
        )
    )
    if (
        not isinstance(health, dict)
        or health.get("healthy") is not True
        or health.get("reasons") != []
        or health.get("required_fields") != list(THERMAL_REQUIRED_FIELDS)
        or health.get("missing_counts")
        != {field: 0 for field in THERMAL_REQUIRED_FIELDS}
        or health.get("incomplete_samples") != 0
        or health.get("maximum_valid_gap_ns")
        != int(maximum_gap_ms * 1_000_000.0)
        or health.get("observed_maximum_valid_gap_ns") != observed_gap_ns
        or health.get("valid_gap_exceeded") is not False
    ):
        raise ValueError(f"{label} telemetry health differs from raw telemetry")
    samples, _markers = telemetry
    selected = [
        record
        for record in samples
        if interval["start_ns"]
        <= int(record["monotonic_ns"])
        and (
            int(record["monotonic_ns"]) < interval["end_ns"]
            or (
                end_inclusive
                and int(record["monotonic_ns"]) == interval["end_ns"]
            )
        )
    ]
    temperatures = stored.get("temperatures_c")
    if not isinstance(temperatures, dict):
        raise ValueError(f"{label} lacks sensor aggregates")
    for sensor in (THERMAL_STABILITY_SENSOR, THERMAL_SAFETY_SENSOR):
        expected = _numeric_temperature_summary(
            [
                float(record["parsed"]["temperatures_c"][sensor])
                for record in selected
            ]
        )
        reported = temperatures.get(sensor)
        if not isinstance(reported, dict) or set(reported) != set(expected):
            raise ValueError(f"{label} {sensor} aggregate has invalid fields")
        for field, expected_value in expected.items():
            if field == "count":
                if reported.get(field) != expected_value:
                    raise ValueError(f"{label} {sensor} count differs from raw telemetry")
            elif not math.isclose(
                strict_float(reported.get(field), f"{label}.{sensor}.{field}"),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{label} {sensor} aggregate differs from raw telemetry"
                )


def replay_point_telemetry_aggregate(
    telemetry: TelemetryRecords,
    *,
    sample_ns: int,
    reference_ns: int,
) -> dict[str, Any]:
    """Replay the schema-4 one-sample qualification aggregate."""

    if sample_ns <= 0 or reference_ns < sample_ns:
        raise ValueError("point telemetry aggregate has invalid clocks")
    raw_samples, _markers = telemetry
    samples = tuple(
        TelemetrySample(
            monotonic_ns=int(record["monotonic_ns"]),
            raw=str(record["raw"]),
            parsed=parse_tegrastats_line(str(record["raw"])),
            mem_available_mb=(
                None
                if record.get("mem_available_mb") is None
                else float(record["mem_available_mb"])
            ),
            collection_errors=tuple(record.get("collection_errors", ())),
        )
        for record in raw_samples
    )
    aggregate = aggregate_samples(
        samples,
        sample_ns - 1,
        sample_ns,
        required_fields=THERMAL_REQUIRED_FIELDS,
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
        "earliest_retained_sample_ns": int(raw_samples[0]["monotonic_ns"]),
        "interval_complete": True,
    }
    return aggregate


def marker_timestamp(
    markers: list[dict[str, Any]], name: str, metadata: dict[str, Any]
) -> int:
    matches = [
        marker["monotonic_ns"]
        for marker in markers
        if marker["name"] == name and marker["metadata"] == metadata
    ]
    if len(matches) != 1:
        raise ValueError(
            f"deadline marker chain requires one {name} marker with {metadata}"
        )
    return matches[0]


def validate_readiness_affinity(
    reported: object,
    *,
    process_pid: int,
    expected_cpu: int,
) -> dict[str, Any]:
    if not isinstance(reported, dict) or set(reported) != {
        "pid",
        "expected_cpu",
        "tasks",
    }:
        raise ValueError("calibration readiness affinity has invalid fields")
    if (
        strict_int(reported.get("pid"), "readiness.pid", minimum=1)
        != process_pid
        or strict_int(reported.get("expected_cpu"), "readiness.expected_cpu")
        != expected_cpu
    ):
        raise ValueError("calibration readiness affinity differs from the process")
    tasks = reported.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("calibration readiness affinity has no tasks")
    replayed_tasks: list[dict[str, Any]] = []
    tids: set[int] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != {"tid", "cpus"}:
            raise ValueError("calibration readiness task has invalid fields")
        tid = strict_int(task.get("tid"), f"readiness.tasks[{index}].tid", minimum=1)
        if tid in tids or task.get("cpus") != [expected_cpu]:
            raise ValueError("calibration readiness task affinity is invalid")
        tids.add(tid)
        replayed_tasks.append({"tid": tid, "cpus": [expected_cpu]})
    if process_pid not in tids:
        raise ValueError("calibration readiness does not include the process leader")
    return {
        "pid": process_pid,
        "expected_cpu": expected_cpu,
        "tasks": replayed_tasks,
    }


def replay_thermal_handoff(
    reported: object,
    *,
    boundary_ns: int,
    cleanup_end_ns: int,
    qualification_ns: int,
    qualification_result_ns: int,
    release_ns: int,
    measurement_start_ns: int,
    maximum_ms: float,
) -> dict[str, Any]:
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
    if not isinstance(reported, dict) or set(reported) != expected_fields:
        raise ValueError("calibration thermal handoff has invalid fields")
    if not (
        boundary_ns
        < cleanup_end_ns
        < qualification_ns
        < qualification_result_ns
        < release_ns
        <= measurement_start_ns
        and reported.get("boundary") == THERMAL_HANDOFF_BOUNDARY
        and reported.get("boundary_monotonic_ns") == boundary_ns
        and reported.get("cleanup_end_monotonic_ns") == cleanup_end_ns
        and reported.get("qualification_monotonic_ns") == qualification_ns
        and reported.get("qualification_result_monotonic_ns")
        == qualification_result_ns
        and reported.get("measurement_release_monotonic_ns") == release_ns
        and reported.get("measurement_start_monotonic_ns")
        == measurement_start_ns
        and reported.get("strictly_within_bound") is True
    ):
        raise ValueError("calibration thermal handoff has invalid clocks")
    replayed = {
        "boundary": THERMAL_HANDOFF_BOUNDARY,
        "boundary_monotonic_ns": boundary_ns,
        "cleanup_end_monotonic_ns": cleanup_end_ns,
        "qualification_monotonic_ns": qualification_ns,
        "qualification_result_monotonic_ns": qualification_result_ns,
        "measurement_release_monotonic_ns": release_ns,
        "measurement_start_monotonic_ns": measurement_start_ns,
        "boundary_to_cleanup_end_ms": (cleanup_end_ns - boundary_ns)
        / 1_000_000.0,
        "boundary_to_qualification_ms": (qualification_ns - boundary_ns)
        / 1_000_000.0,
        "boundary_to_qualification_result_ms": (
            qualification_result_ns - boundary_ns
        )
        / 1_000_000.0,
        "boundary_to_measurement_release_ms": (release_ns - boundary_ns)
        / 1_000_000.0,
        "boundary_to_measurement_start_ms": (
            measurement_start_ns - boundary_ns
        )
        / 1_000_000.0,
        "maximum_ms": maximum_ms,
        "strictly_within_bound": True,
    }
    if (
        any(
            float(replayed[field]) >= maximum_ms
            for field in (
                "boundary_to_cleanup_end_ms",
                "boundary_to_qualification_ms",
                "boundary_to_qualification_result_ms",
                "boundary_to_measurement_release_ms",
                "boundary_to_measurement_start_ms",
            )
        )
    ):
        raise ValueError("calibration thermal handoff exceeds the strict bound")
    for field in (
        "boundary_to_cleanup_end_ms",
        "boundary_to_qualification_ms",
        "boundary_to_qualification_result_ms",
        "boundary_to_measurement_release_ms",
        "boundary_to_measurement_start_ms",
        "maximum_ms",
    ):
        if not math.isclose(
            strict_float(reported.get(field), f"thermal_handoff.{field}"),
            float(replayed[field]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("calibration thermal handoff differs from raw clocks")
    return replayed


def _thermal_envelope_is_stable(
    evidence: dict[str, Any],
    *,
    target_c: float,
    tolerance_c: float,
    maximum_slope_c_per_minute: float,
) -> bool:
    window = evidence["stability_window"]
    return (
        abs(float(window["mean_c"]) - target_c) <= tolerance_c
        and abs(float(window["latest_c"]) - target_c) <= tolerance_c
        and abs(float(window["slope_c_per_minute"]))
        <= maximum_slope_c_per_minute
    )


def replay_active_stability_checks(
    reported: object,
    *,
    expected_label: str,
    telemetry: TelemetryRecords,
    markers: list[dict[str, Any]],
    not_before_ns: int,
    boundary_ns: int,
    interval_ms: float,
    required_fraction: float,
    window_seconds: float,
    target_c: float,
    tolerance_c: float,
    maximum_slope_c_per_minute: float,
    hard_limit_c: float,
    maximum_gap_ms: float,
    stability_sensor: str,
    safety_sensor: str,
) -> list[dict[str, Any]]:
    """Replay every active-load endpoint and bind the selected boundary sample."""

    if not isinstance(reported, list) or not reported:
        raise ValueError("thermal precondition lacks active stability checks")
    active_markers = [
        marker
        for marker in markers
        if marker.get("name") == "thermal_active_stability_check"
        and marker.get("metadata", {}).get("label") == expected_label
    ]
    if len(active_markers) != len(reported):
        raise ValueError("active stability marker count differs from stored evidence")
    replayed: list[dict[str, Any]] = []
    consecutive = 0
    previous_sample_ns: int | None = None
    required_spacing_ns = int(
        THERMAL_ACTIVE_STABLE_SPACING_SECONDS * 1_000_000_000.0
    )
    samples, _ = telemetry
    raw_sample_timestamps = {int(sample["monotonic_ns"]) for sample in samples}
    for index, (stored, marker) in enumerate(
        zip(reported, active_markers, strict=True)
    ):
        metadata = marker.get("metadata")
        expected_keys = {
            "label",
            "index",
            "sample_monotonic_ns",
            "passed",
            "consecutive_passes",
            "window",
        }
        if (
            not isinstance(stored, dict)
            or not isinstance(metadata, dict)
            or set(stored) != expected_keys
            or set(metadata) != expected_keys
            or stored != metadata
            or metadata.get("label") != expected_label
            or strict_int(metadata.get("index"), "active check index") != index
        ):
            raise ValueError("active stability check has invalid fields")
        sample_ns = strict_int(
            metadata.get("sample_monotonic_ns"),
            "active check sample_monotonic_ns",
            minimum=1,
        )
        marker_ns = strict_int(
            marker.get("monotonic_ns"), "active check marker timestamp", minimum=1
        )
        if (
            sample_ns not in raw_sample_timestamps
            or sample_ns < not_before_ns
            or sample_ns >= boundary_ns
            or marker_ns <= sample_ns
            or marker_ns >= boundary_ns
            or (
                previous_sample_ns is not None
                and sample_ns - previous_sample_ns < required_spacing_ns
            )
        ):
            raise ValueError("active stability check selected a non-causal endpoint")
        raw = replay_raw_thermal_window(
            telemetry,
            reference_ns=sample_ns,
            not_before_ns=not_before_ns,
            window_seconds=window_seconds,
            interval_ms=interval_ms,
            required_fraction=required_fraction,
            maximum_gap_ms=maximum_gap_ms,
            stability_sensor=stability_sensor,
            safety_sensor=safety_sensor,
            hard_limit_c=hard_limit_c,
            end_inclusive=True,
        )
        passed = _thermal_envelope_is_stable(
            raw,
            target_c=target_c,
            tolerance_c=tolerance_c,
            maximum_slope_c_per_minute=maximum_slope_c_per_minute,
        )
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
        _compare_thermal_window(
            metadata.get("window"),
            raw["stability_window"],
            expected_fields=THERMAL_WINDOW_FIELDS,
            label="active stability check window",
        )
        replayed.append(dict(metadata))
        previous_sample_ns = sample_ns
    if consecutive != THERMAL_ACTIVE_STABLE_ENDPOINTS:
        raise ValueError("active thermal precondition lacks three stable endpoints")
    if any(
        check["passed"] is not True
        for check in replayed[-THERMAL_ACTIVE_STABLE_ENDPOINTS:]
    ):
        raise ValueError("active thermal endpoint sequence is not stable")
    return replayed


def replay_deadline_precondition(
    precondition: object,
    *,
    expected_label: str,
    telemetry: TelemetryRecords,
    markers: list[dict[str, Any]],
    interval_ms: float,
    required_fraction: float,
    window_seconds: float,
    target_c: float,
    tolerance_c: float,
    maximum_slope_c_per_minute: float,
    hard_limit_c: float,
    maximum_gap_ms: float,
    stability_sensor: str,
    safety_sensor: str,
) -> dict[str, Any]:
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
    if not isinstance(precondition, dict) or set(precondition) != expected_fields:
        raise ValueError("a calibration thermal precondition has invalid fields")
    if precondition.get("label") != expected_label:
        raise ValueError("a calibration thermal precondition label is invalid")
    start_ns = strict_int(
        precondition.get("measurement_start_monotonic_ns"),
        f"{expected_label}.measurement_start_monotonic_ns",
        minimum=1,
    )
    end_ns = strict_int(
        precondition.get("measurement_end_monotonic_ns"),
        f"{expected_label}.measurement_end_monotonic_ns",
        minimum=1,
    )
    cleanup_end_ns = strict_int(
        precondition.get("cleanup_end_monotonic_ns"),
        f"{expected_label}.cleanup_end_monotonic_ns",
        minimum=1,
    )
    duration_seconds = strict_float(
        precondition.get("duration_seconds"), f"{expected_label}.duration_seconds"
    )
    if (
        precondition.get("stability_sensor") != stability_sensor
        or precondition.get("safety_sensor") != safety_sensor
        or precondition.get("active_stable_endpoints")
        != THERMAL_ACTIVE_STABLE_ENDPOINTS
        or not math.isclose(
            strict_float(
                precondition.get("active_stable_spacing_seconds"),
                f"{expected_label}.active_stable_spacing_seconds",
            ),
            THERMAL_ACTIVE_STABLE_SPACING_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or precondition.get("termination_reason")
        != "active-stability-endpoints"
    ):
        raise ValueError("a calibration thermal precondition has invalid sensors")
    full_evidence = replay_raw_thermal_window(
        telemetry,
        reference_ns=end_ns,
        not_before_ns=start_ns,
        window_seconds=duration_seconds,
        interval_ms=interval_ms,
        required_fraction=required_fraction,
        maximum_gap_ms=maximum_gap_ms,
        stability_sensor=stability_sensor,
        safety_sensor=safety_sensor,
        hard_limit_c=hard_limit_c,
    )
    if not (
        duration_seconds >= window_seconds * 0.99
        and math.isclose(
            duration_seconds,
            (end_ns - start_ns) / 1_000_000_000.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            strict_float(precondition.get("target_c"), f"{expected_label}.target_c"),
            target_c,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and strict_float(
            precondition.get("pressure_rate_per_second"),
            f"{expected_label}.pressure_rate_per_second",
        )
        > 0.0
    ):
        raise ValueError("a calibration thermal precondition is invalid")
    validate_stored_thermal_aggregate(
        precondition.get("telemetry"),
        full_evidence,
        telemetry,
        maximum_gap_ms=maximum_gap_ms,
        label=expected_label,
    )
    metadata = {"label": expected_label}
    prepare_marker_ns = marker_timestamp(markers, "thermal_prepare", metadata)
    start_marker_ns = marker_timestamp(markers, "thermal_start", metadata)
    matching_boundaries = [
        marker
        for marker in markers
        if marker.get("name") == THERMAL_HANDOFF_BOUNDARY
        and marker.get("metadata", {}).get("label") == expected_label
    ]
    if len(matching_boundaries) != 1:
        raise ValueError("thermal precondition lacks one active boundary")
    measurement_end_marker = matching_boundaries[0]
    measurement_end_marker_ns = strict_int(
        measurement_end_marker.get("monotonic_ns"),
        "thermal boundary timestamp",
        minimum=1,
    )
    cleanup_marker_ns = marker_timestamp(
        markers, "thermal_end", metadata | {"successful": True}
    )
    if not (
        prepare_marker_ns
        < start_marker_ns
        == start_ns
        < measurement_end_marker_ns
        == end_ns
        <= cleanup_marker_ns
        == cleanup_end_ns
    ):
        raise ValueError(f"{expected_label} thermal marker chain is invalid")
    active_checks = replay_active_stability_checks(
        precondition.get("active_stability_checks"),
        expected_label=expected_label,
        telemetry=telemetry,
        markers=markers,
        not_before_ns=start_ns,
        boundary_ns=end_ns,
        interval_ms=interval_ms,
        required_fraction=required_fraction,
        window_seconds=window_seconds,
        target_c=target_c,
        tolerance_c=tolerance_c,
        maximum_slope_c_per_minute=maximum_slope_c_per_minute,
        hard_limit_c=hard_limit_c,
        maximum_gap_ms=maximum_gap_ms,
        stability_sensor=stability_sensor,
        safety_sensor=safety_sensor,
    )
    final_check = active_checks[-1]
    expected_boundary_metadata = {
        "label": expected_label,
        "boundary_sample_monotonic_ns": final_check["sample_monotonic_ns"],
        "consecutive_passes": THERMAL_ACTIVE_STABLE_ENDPOINTS,
        "window": final_check["window"],
    }
    if measurement_end_marker.get("metadata") != expected_boundary_metadata:
        raise ValueError("thermal boundary does not bind the final active endpoint")
    boundary_sample_ns = int(final_check["sample_monotonic_ns"])
    if (
        end_ns - boundary_sample_ns < 0
        or end_ns - boundary_sample_ns >= int(maximum_gap_ms * 1_000_000.0)
    ):
        raise ValueError("thermal boundary sample is stale")
    last_window_evidence = replay_raw_thermal_window(
        telemetry,
        reference_ns=boundary_sample_ns,
        not_before_ns=start_ns,
        window_seconds=window_seconds,
        interval_ms=interval_ms,
        required_fraction=required_fraction,
        maximum_gap_ms=maximum_gap_ms,
        stability_sensor=stability_sensor,
        safety_sensor=safety_sensor,
        hard_limit_c=hard_limit_c,
        end_inclusive=True,
    )
    _compare_thermal_window(
        precondition.get("last_window"),
        last_window_evidence["stability_window"],
        expected_fields=THERMAL_PRECONDITION_WINDOW_FIELDS,
        label=f"{expected_label}.last_window",
    )
    require_thermal_envelope(
        last_window_evidence,
        target_c=target_c,
        tolerance_c=tolerance_c,
        maximum_slope_c_per_minute=maximum_slope_c_per_minute,
        label=expected_label,
    )
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "cleanup_end_ns": cleanup_end_ns,
        "prepare_marker_ns": prepare_marker_ns,
        "full_evidence": full_evidence,
        "last_window_evidence": last_window_evidence,
        "active_stability_checks": active_checks,
        "boundary_sample_ns": boundary_sample_ns,
    }


def validate_paused_process_states(
    reported: object, *, expected_pids: list[int], label: str
) -> dict[str, str]:
    expected = {str(pid): "T" for pid in sorted(expected_pids)}
    if not isinstance(reported, dict) or reported != expected:
        raise ValueError(f"{label} does not prove every measured PID remained paused")
    return expected


def replay_deadline_qualification(
    qualification: object,
    *,
    attempt: int,
    base_label: str,
    precondition_evidence: dict[str, Any],
    telemetry: TelemetryRecords,
    markers: list[dict[str, Any]],
    result_marker_metadata: dict[str, Any],
    interval_ms: float,
    required_fraction: float,
    window_seconds: float,
    target_c: float,
    tolerance_c: float,
    maximum_slope_c_per_minute: float,
    hard_limit_c: float,
    maximum_gap_ms: float,
    stability_sensor: str,
    safety_sensor: str,
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
        raise ValueError("thermal qualification differs from the frozen contract")
    boundary_ns = strict_int(
        qualification.get("boundary_monotonic_ns"),
        "qualification.boundary_monotonic_ns",
        minimum=1,
    )
    cleanup_ns = strict_int(
        qualification.get("cleanup_end_monotonic_ns"),
        "qualification.cleanup_end_monotonic_ns",
        minimum=1,
    )
    qualification_ns = strict_int(
        qualification.get("qualification_monotonic_ns"),
        "qualification.qualification_monotonic_ns",
        minimum=1,
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
            minimum=1,
        )
    )
    if (
        boundary_ns != int(precondition_evidence["end_ns"])
        or cleanup_ns != int(precondition_evidence["cleanup_end_ns"])
        or not boundary_ns < cleanup_ns < qualification_ns
        or (
            sample_ns is not None
            and not cleanup_ns < sample_ns <= qualification_ns
        )
    ):
        raise ValueError("thermal qualification clocks are not causal")
    qualification_metadata = {
        "label": base_label,
        "attempt": attempt,
        "boundary_monotonic_ns": boundary_ns,
        "cleanup_end_monotonic_ns": cleanup_ns,
        "sample_monotonic_ns": sample_ns,
    }
    marker_ns = marker_timestamp(
        markers, "thermal_start_qualification", qualification_metadata
    )
    if marker_ns != qualification_ns:
        raise ValueError("thermal qualification marker clock differs")
    samples, _ = telemetry
    causal_samples = [
        sample
        for sample in samples
        if cleanup_ns < int(sample["monotonic_ns"]) <= qualification_ns
    ]
    if sample_ns is None:
        if causal_samples:
            raise ValueError(
                "sample-free thermal qualification hides a causal raw sample"
            )
        failure_reason = qualification.get("failure_reason")
        if (
            reported_passed is not False
            or qualification.get("sample_age_ms") is not None
            or qualification.get("stability_value_c") is not None
            or qualification.get("safety_value_c") is not None
            or qualification.get("telemetry") is not None
            or qualification.get("stability_sensor") != stability_sensor
            or qualification.get("safety_sensor") != safety_sensor
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
        result_marker_ns = marker_timestamp(
            markers,
            "thermal_start_qualification_result",
            result_marker_metadata
            | {
                "label": base_label,
                "attempt": attempt,
                "qualification_monotonic_ns": qualification_ns,
                "passed": False,
                "failure_reason": failure_reason,
            },
        )
        if result_marker_ns <= qualification_ns:
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
        }, result_marker_ns

    assert sample_ns is not None
    if not causal_samples or int(causal_samples[0]["monotonic_ns"]) != sample_ns:
        raise ValueError("thermal qualification did not select the first causal sample")
    sample = causal_samples[0]
    structurally_valid = _structurally_valid_thermal_sample(
        sample,
        stability_sensor=stability_sensor,
        safety_sensor=safety_sensor,
    )
    temperatures = sample.get("parsed", {}).get("temperatures_c", {})
    raw_stability = (
        float(temperatures[stability_sensor]) if structurally_valid else math.nan
    )
    raw_safety = float(temperatures[safety_sensor]) if structurally_valid else math.nan
    raw_age_ms = (qualification_ns - sample_ns) / 1_000_000.0
    sample_gap_ms = (sample_ns - cleanup_ns) / 1_000_000.0
    expected_telemetry = replay_point_telemetry_aggregate(
        telemetry,
        sample_ns=sample_ns,
        reference_ns=qualification_ns,
    )
    if qualification.get("telemetry") != expected_telemetry:
        raise ValueError("thermal qualification telemetry differs from raw telemetry")
    raw_passed = (
        structurally_valid
        and raw_age_ms <= maximum_gap_ms
        and sample_gap_ms <= maximum_gap_ms
        and abs(raw_stability - target_c) <= tolerance_c
        and raw_safety < hard_limit_c
        and expected_telemetry["health"]["healthy"] is True
    )
    if (
        qualification.get("stability_sensor") != stability_sensor
        or qualification.get("safety_sensor") != safety_sensor
        or not math.isclose(
            strict_float(qualification.get("sample_age_ms"), "sample_age_ms"),
            raw_age_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            strict_float(
                qualification.get("stability_value_c"), "stability_value_c"
            ),
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
        raise ValueError("thermal qualification result differs from raw telemetry")
    result_marker_ns = marker_timestamp(
        markers,
        "thermal_start_qualification_result",
        result_marker_metadata
        | {
            "label": base_label,
            "attempt": attempt,
            "qualification_monotonic_ns": qualification_ns,
            "passed": raw_passed,
            "failure_reason": qualification.get("failure_reason"),
        },
    )
    if result_marker_ns <= qualification_ns:
        raise ValueError("thermal qualification result marker is reordered")
    return {
        "boundary_monotonic_ns": boundary_ns,
        "cleanup_end_monotonic_ns": cleanup_ns,
        "qualification_monotonic_ns": qualification_ns,
        "sample_monotonic_ns": sample_ns,
        "sample_age_ms": raw_age_ms,
        "stability_value_c": raw_stability,
        "safety_value_c": raw_safety,
        "passed": raw_passed,
    }, result_marker_ns


def replay_actual_start_qualification(
    reported: object,
    *,
    measurement_start_ns: int,
    not_before_ns: int,
    qualification_sample_ns: int,
    telemetry: TelemetryRecords,
    markers: list[dict[str, Any]],
    result_marker_metadata: dict[str, Any],
    target_c: float,
    tolerance_c: float,
    hard_limit_c: float,
    maximum_gap_ms: float,
    stability_sensor: str,
    safety_sensor: str,
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
        minimum=1,
    )
    sample_ns = strict_int(
        reported.get("sample_monotonic_ns"),
        "actual-start sample_monotonic_ns",
        minimum=1,
    )
    if stored_start_ns != measurement_start_ns or not (
        not_before_ns < qualification_sample_ns <= sample_ns <= measurement_start_ns
    ):
        raise ValueError("actual-start qualification clocks are not causal")
    samples, _ = telemetry
    causal_samples = [
        sample
        for sample in samples
        if not_before_ns < int(sample["monotonic_ns"]) <= measurement_start_ns
    ]
    if not causal_samples or int(causal_samples[-1]["monotonic_ns"]) != sample_ns:
        raise ValueError("actual start did not select the latest causal sample")
    sample = causal_samples[-1]
    structurally_valid = _structurally_valid_thermal_sample(
        sample,
        stability_sensor=stability_sensor,
        safety_sensor=safety_sensor,
    )
    temperatures = sample.get("parsed", {}).get("temperatures_c", {})
    raw_stability = (
        float(temperatures[stability_sensor]) if structurally_valid else math.nan
    )
    raw_safety = float(temperatures[safety_sensor]) if structurally_valid else math.nan
    raw_age_ms = (measurement_start_ns - sample_ns) / 1_000_000.0
    expected_telemetry = replay_point_telemetry_aggregate(
        telemetry,
        sample_ns=sample_ns,
        reference_ns=measurement_start_ns,
    )
    if reported.get("telemetry") != expected_telemetry:
        raise ValueError("actual-start telemetry differs from raw telemetry")
    raw_passed = (
        structurally_valid
        and raw_age_ms <= maximum_gap_ms
        and abs(raw_stability - target_c) <= tolerance_c
        and raw_safety < hard_limit_c
        and expected_telemetry["health"]["healthy"] is True
    )
    if (
        reported.get("stability_sensor") != stability_sensor
        or reported.get("safety_sensor") != safety_sensor
        or not math.isclose(
            strict_float(reported.get("sample_age_ms"), "actual-start sample_age_ms"),
            raw_age_ms,
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
            strict_float(
                reported.get("tolerance_c"), "actual-start tolerance_c"
            ),
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
    result_marker_ns = marker_timestamp(
        markers,
        "thermal_actual_start_qualification_result",
        result_marker_metadata
        | {
            "measurement_start_monotonic_ns": measurement_start_ns,
            "sample_monotonic_ns": sample_ns,
            "passed": raw_passed,
            "failure_reason": reported.get("failure_reason"),
        },
    )
    return {
        "measurement_start_monotonic_ns": measurement_start_ns,
        "sample_monotonic_ns": sample_ns,
        "sample_age_ms": raw_age_ms,
        "stability_value_c": raw_stability,
        "safety_value_c": raw_safety,
        "passed": raw_passed,
    }, result_marker_ns


def require_thermal_envelope(
    evidence: dict[str, Any],
    *,
    target_c: float,
    tolerance_c: float,
    maximum_slope_c_per_minute: float,
    label: str,
) -> None:
    window = evidence["stability_window"]
    if (
        abs(float(window["mean_c"]) - target_c) > tolerance_c
        or abs(float(window["latest_c"]) - target_c) > tolerance_c
        or abs(float(window["slope_c_per_minute"]))
        > maximum_slope_c_per_minute
    ):
        raise ValueError(f"{label} violates the frozen soc012 envelope")


def build_lock(
    summary: dict[str, Any],
    summary_path: pathlib.Path,
    *,
    guard_lock_path: pathlib.Path,
    expected_blocks: int,
    expected_samples_per_block: int,
) -> dict[str, Any]:
    current_code = code_hashes()
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("deadline calibration lacks artifact provenance")
    implementation = artifacts.get("implementation_sha256")
    if not isinstance(implementation, dict) or any(
        implementation.get(name) != digest
        for name, digest in current_code.items()
    ):
        raise ValueError(
            "deadline calibration was not produced by the current implementation"
        )
    hardware = summary.get("hardware")
    mig = summary.get("mig")
    if not isinstance(hardware, dict) or not isinstance(mig, dict):
        raise ValueError("deadline calibration lacks hardware or MIG provenance")
    if summary.get("schema_version") != 4:
        raise ValueError("calibration summary must use schema version 4")
    config = summary.get("config", {})
    calibration_protocol = {
        "warmup": config.get("warmup"),
        "burst_size": config.get("burst_size"),
        "period_ms": config.get("period_ms"),
        "slo_factor": config.get("slo_factor"),
        "cpu_affinity": config.get("cpu_affinity"),
        "thermal_stability_sensor": config.get("thermal_stability_sensor"),
        "thermal_safety_sensor": config.get("thermal_safety_sensor"),
        "thermal_handoff_max_ms": config.get("thermal_handoff_max_ms"),
        "thermal_handoff_boundary": config.get("thermal_handoff_boundary"),
        "thermal_qualification_max_attempts": config.get(
            "thermal_qualification_max_attempts"
        ),
        "thermal_active_stable_endpoints": config.get(
            "thermal_active_stable_endpoints"
        ),
        "thermal_active_stable_spacing_seconds": config.get(
            "thermal_active_stable_spacing_seconds"
        ),
        "thermal_calibration_preconditioning": config.get(
            "thermal_calibration_preconditioning"
        ),
        "start_protocol": config.get("start_protocol"),
        "telemetry_interval_ms": config.get("telemetry_interval_ms"),
        "telemetry_required_fraction": config.get(
            "telemetry_required_fraction"
        ),
        "telemetry_required_fields": config.get("telemetry_required_fields"),
        "telemetry_stale_after_ms": config.get("telemetry_stale_after_ms"),
        "telemetry_max_gap_ms": config.get("telemetry_max_gap_ms"),
    }
    if calibration_protocol != {
        "warmup": 100,
        "burst_size": 8,
        "period_ms": 20.0,
        "slo_factor": 1.10,
        "cpu_affinity": {
            "critical": [12],
            "pressure": list(range(11)),
            "mps": [11],
            "telemetry": [13],
        },
        "thermal_stability_sensor": THERMAL_STABILITY_SENSOR,
        "thermal_safety_sensor": THERMAL_SAFETY_SENSOR,
        "thermal_handoff_max_ms": THERMAL_HANDOFF_MAX_MS,
        "thermal_handoff_boundary": THERMAL_HANDOFF_BOUNDARY,
        "thermal_qualification_max_attempts": (
            THERMAL_QUALIFICATION_MAX_ATTEMPTS
        ),
        "thermal_active_stable_endpoints": THERMAL_ACTIVE_STABLE_ENDPOINTS,
        "thermal_active_stable_spacing_seconds": (
            THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        ),
        "thermal_calibration_preconditioning": "per-repeat-preloaded-critical",
        "start_protocol": (
            "post-warmup-stop-barrier-with-bounded-thermal-handoff"
        ),
        "telemetry_interval_ms": 100.0,
        "telemetry_required_fraction": 0.8,
        "telemetry_required_fields": list(THERMAL_REQUIRED_FIELDS),
        "telemetry_stale_after_ms": 300.0,
        "telemetry_max_gap_ms": 300.0,
    } or "thermal_qualification_dwell_seconds" in config:
        raise ValueError("deadline calibration protocol differs from the frozen design")
    if not config.get("calibration_only") or summary.get("policies") != []:
        raise ValueError("summary is not a calibration-only run")
    guard_lock_path = guard_lock_path.resolve()
    guard_lock, _guard_lock_bytes, guard_lock_sha256 = load_json_buffer(
        guard_lock_path, "guard lock"
    )
    verify_guard_lock(guard_lock)
    profile_guard_ms = guard_profile_from_lock(guard_lock)
    guard_protocol = guard_lock.get("protocol")
    guard_estimator = guard_lock.get("estimator")
    guard_source = guard_lock.get("source")
    if (
        not isinstance(guard_protocol, dict)
        or not isinstance(guard_estimator, dict)
        or not isinstance(guard_source, dict)
    ):
        raise ValueError("guard lock lacks protocol or source provenance")
    guard_profile_summary_sha256 = guard_source.get("profile_summary_sha256")
    guard_telemetry_jsonl_sha256 = guard_source.get("telemetry_jsonl_sha256")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (
            guard_profile_summary_sha256,
            guard_telemetry_jsonl_sha256,
        )
    ):
        raise ValueError("guard lock source hashes are invalid")
    if config.get("guard_lock_sha256") != guard_lock_sha256:
        raise ValueError("deadline calibration did not record its guard lock hash")
    if config.get("guard_profile_source") != "frozen-quota-aware-lock":
        raise ValueError("deadline calibration did not use the frozen guard profile")
    if config.get("profile_guard_ms") != profile_guard_ms:
        raise ValueError("deadline calibration guard profile differs from its lock")
    if config.get("guard_override_ms") is not None:
        raise ValueError("deadline calibration cannot override the frozen guard profile")
    validate_guard_platform_and_artifacts(
        guard_lock,
        hardware=hardware,
        mig=mig,
        cpu_affinity=config.get("cpu_affinity"),
        calibration_artifacts=artifacts,
    )
    if int(config.get("calibration_repeats", -1)) != expected_blocks:
        raise ValueError("unexpected calibration block count")
    if int(config.get("samples_per_epoch", -1)) != expected_samples_per_block:
        raise ValueError("unexpected samples per calibration block")
    expected_total = expected_blocks * expected_samples_per_block
    if int(summary.get("isolated_pooled_samples", -1)) != expected_total:
        raise ValueError("pooled isolated sample count is incomplete")
    isolated_results = summary.get("isolated")
    if not isinstance(isolated_results, list) or len(isolated_results) != expected_blocks:
        raise ValueError("isolated calibration result count is incomplete")
    preconditions = summary.get("isolated_preconditions", [])
    if not isinstance(preconditions, list) or len(preconditions) != expected_blocks:
        raise ValueError("each calibration block requires thermal preconditioning")
    first_result = isolated_results[0] if isolated_results else None
    engine_value = first_result.get("engine") if isinstance(first_result, dict) else None
    if not isinstance(engine_value, str):
        raise ValueError("deadline calibration lacks its critical engine path")
    critical_engine = pathlib.Path(engine_value)
    if not critical_engine.is_absolute() or critical_engine.parts[-2:] != (
        "mig-2g",
        "resnet50-v2.engine",
    ):
        raise ValueError("deadline calibration has an invalid critical engine path")
    engine_hashes = artifacts.get("engines_sha256")
    expected_engine_hash = (
        engine_hashes.get("critical-2g-resnet50-v2")
        if isinstance(engine_hashes, dict)
        else None
    )
    if (
        not isinstance(expected_engine_hash, str)
        or len(expected_engine_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_engine_hash)
        or not critical_engine.is_file()
        or file_sha256(critical_engine) != expected_engine_hash
    ):
        raise ValueError("deadline calibration critical engine differs from its artifact")
    critical_uuid = mig.get("critical_uuid")
    if not isinstance(critical_uuid, str) or not critical_uuid:
        raise ValueError("deadline calibration lacks its critical MIG UUID")
    raw_directory = summary_path.resolve().parent / "raw"
    pooled_latencies: list[float] = []
    trace_hashes: dict[str, str] = {}
    trace_file_identities: set[tuple[int, int]] = set()
    trace_content_hashes: set[str] = set()
    for repeat, isolated in enumerate(isolated_results, start=1):
        trace = raw_directory / f"isolated-pre-r{repeat}.csv"
        trace_values = read_calibration_trace(trace, expected_samples_per_block)
        stat = trace.stat()
        identity = (stat.st_dev, stat.st_ino)
        digest = file_sha256(trace)
        if identity in trace_file_identities or digest in trace_content_hashes:
            raise ValueError("calibration traces must be independent files")
        trace_file_identities.add(identity)
        trace_content_hashes.add(digest)
        trace_hashes[trace.name] = digest
        validate_calibration_result(
            isolated,
            trace_values,
            label=f"isolated[{repeat - 1}]",
            expected_samples=expected_samples_per_block,
            config=config,
            critical_uuid=critical_uuid,
            critical_engine=critical_engine,
        )
        pooled_latencies.extend(trace_values["release_to_completion_ms"])
    raw_p99 = percentile(pooled_latencies, 0.99)
    p99 = float(summary["isolated_pooled_p99_ms"])
    if not math.isclose(p99, raw_p99, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("pooled calibration p99 differs from raw traces")
    factor = float(config["slo_factor"])
    deadline = float(summary["deadline_ms"])
    if not all(math.isfinite(value) and value > 0.0 for value in (p99, factor, deadline)):
        raise ValueError("calibration contains non-positive values")
    if not math.isclose(deadline, p99 * factor, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("deadline does not match pooled p99 and SLO factor")
    if not math.isclose(factor, 1.10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("formal deadline calibration requires a 10% tail budget")
    directory = summary_path.resolve().parent
    thermal_lock_path = directory / "thermal-lock.json"
    if not thermal_lock_path.is_file():
        raise ValueError("deadline calibration lacks its frozen thermal lock")
    thermal_lock, thermal_lock_bytes, thermal_lock_sha256 = load_json_buffer(
        thermal_lock_path, "thermal lock"
    )
    verify_thermal_lock(thermal_lock)
    if (
        thermal_lock.get("schema_version") != THERMAL_LOCK_SCHEMA_VERSION
        or thermal_lock.get("target_source") != THERMAL_TARGET_SOURCE
        or thermal_lock.get("stability_sensor") != THERMAL_STABILITY_SENSOR
        or thermal_lock.get("safety_sensor") != THERMAL_SAFETY_SENSOR
        or thermal_lock.get("thermal_handoff_max_ms")
        != THERMAL_HANDOFF_MAX_MS
        or thermal_lock.get("thermal_handoff_boundary")
        != THERMAL_HANDOFF_BOUNDARY
        or thermal_lock.get("thermal_qualification_max_attempts")
        != THERMAL_QUALIFICATION_MAX_ATTEMPTS
        or thermal_lock.get("thermal_active_stable_endpoints")
        != THERMAL_ACTIVE_STABLE_ENDPOINTS
        or thermal_lock.get("thermal_active_stable_spacing_seconds")
        != THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        or "thermal_qualification_dwell_seconds" in thermal_lock
        or thermal_lock.get("thermal_handoff_rationale")
        != THERMAL_HANDOFF_RATIONALE
        or thermal_lock.get("telemetry_required_fields")
        != list(THERMAL_REQUIRED_FIELDS)
    ):
        raise ValueError("deadline calibration thermal lock is invalid")
    if (
        thermal_lock.get("pilot_artifacts") != artifacts
        or thermal_lock.get("pilot_hardware") != hardware
        or thermal_lock.get("pilot_mig") != mig
        or thermal_lock.get("pilot_cpu_affinity") != config.get("cpu_affinity")
    ):
        raise ValueError(
            "deadline calibration platform differs from the thermal pilot"
        )
    if config.get("thermal_lock_sha256") != thermal_lock_sha256:
        raise ValueError("deadline calibration did not record its thermal lock hash")
    guard_thermal = guard_lock.get("thermal_lock")
    if (
        not isinstance(guard_thermal, dict)
        or guard_thermal.get("sha256") != thermal_lock_sha256
    ):
        raise ValueError("guard and deadline calibrations use different thermal locks")
    thermal_fields = {
        "thermal_target_c": "target_c",
        "thermal_tolerance_c": "tolerance_c",
        "thermal_window_seconds": "stability_window_seconds",
        "thermal_max_slope_c_per_minute": "maximum_slope_c_per_minute",
        "thermal_hard_limit_c": "hard_limit_c",
        "thermal_handoff_max_ms": "thermal_handoff_max_ms",
        "thermal_qualification_max_attempts": (
            "thermal_qualification_max_attempts"
        ),
        "thermal_active_stable_endpoints": "thermal_active_stable_endpoints",
        "thermal_active_stable_spacing_seconds": (
            "thermal_active_stable_spacing_seconds"
        ),
        "telemetry_interval_ms": "telemetry_interval_ms",
        "telemetry_required_fraction": "telemetry_required_fraction",
        "telemetry_max_gap_ms": "telemetry_max_gap_ms",
    }
    for config_name, lock_name in thermal_fields.items():
        if not math.isclose(
            float(config.get(config_name, math.nan)),
            float(thermal_lock.get(lock_name, math.nan)),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"deadline calibration {config_name} differs from its thermal lock"
            )
    for config_name, lock_name in {
        "thermal_stability_sensor": "stability_sensor",
        "thermal_safety_sensor": "safety_sensor",
        "thermal_handoff_boundary": "thermal_handoff_boundary",
    }.items():
        if config.get(config_name) != thermal_lock.get(lock_name):
            raise ValueError(
                f"deadline calibration {config_name} differs from its thermal lock"
            )
    if config.get("telemetry_required_fields") != thermal_lock.get(
        "telemetry_required_fields"
    ):
        raise ValueError(
            "deadline calibration telemetry required fields differ from its thermal lock"
        )
    telemetry_path = directory / "telemetry.jsonl"
    telemetry = load_telemetry_jsonl(telemetry_path)
    _telemetry_samples, telemetry_markers = telemetry
    chain_marker_names = {
        "calibration_prepare",
        "thermal_prepare",
        "thermal_start",
        "thermal_active_stability_check",
        "thermal_measurement_end",
        "thermal_end",
        "thermal_start_qualification",
        "thermal_start_qualification_result",
        "thermal_actual_start_qualification_result",
        "calibration_start",
        "calibration_measurement_window",
        "calibration_end",
    }
    total_attempts = 0
    total_active_checks = 0
    for index, calibration in enumerate(isolated_results):
        attempts = calibration.get("thermal_start_attempts")
        if (
            not isinstance(attempts, list)
            or not 1 <= len(attempts) <= THERMAL_QUALIFICATION_MAX_ATTEMPTS
        ):
            raise ValueError(
                f"isolated[{index}] has an invalid thermal qualification attempt count"
            )
        total_attempts += len(attempts)
        for attempt in attempts:
            precondition = (
                attempt.get("thermal_precondition")
                if isinstance(attempt, dict)
                else None
            )
            checks = (
                precondition.get("active_stability_checks")
                if isinstance(precondition, dict)
                else None
            )
            if not isinstance(checks, list):
                raise ValueError("deadline thermal attempt lacks active checks")
            total_active_checks += len(checks)
    expected_chain_markers = (
        expected_blocks * 5 + total_attempts * 6 + total_active_checks
    )
    if sum(
        marker["name"] in chain_marker_names for marker in telemetry_markers
    ) != expected_chain_markers:
        raise ValueError("deadline telemetry has an incomplete or extra marker chain")
    telemetry_interval_ms = float(thermal_lock["telemetry_interval_ms"])
    telemetry_required_fraction = float(
        thermal_lock["telemetry_required_fraction"]
    )
    window_seconds = float(thermal_lock["stability_window_seconds"])
    target_c = float(thermal_lock["target_c"])
    tolerance_c = float(thermal_lock["tolerance_c"])
    maximum_slope = float(thermal_lock["maximum_slope_c_per_minute"])
    hard_limit_c = float(thermal_lock["hard_limit_c"])
    maximum_gap_ms = float(thermal_lock["telemetry_max_gap_ms"])
    handoff_max_ms = float(thermal_lock["thermal_handoff_max_ms"])
    stability_sensor = str(thermal_lock["stability_sensor"])
    safety_sensor = str(thermal_lock["safety_sensor"])
    raw_preconditions: list[dict[str, Any]] = []
    previous_chain_end: int | None = None
    for repeat, final_precondition in enumerate(preconditions, start=1):
        calibration = isolated_results[repeat - 1]
        calibration_start_ns = strict_int(
            calibration.get("measurement_start_monotonic_ns"),
            f"isolated[{repeat - 1}].measurement_start_monotonic_ns",
            minimum=1,
        )
        calibration_end_ns = strict_int(
            calibration.get("measurement_end_monotonic_ns"),
            f"isolated[{repeat - 1}].measurement_end_monotonic_ns",
            minimum=1,
        )
        base_label = f"pre-pre-calibration-r{repeat}"
        calibration_metadata = {"stage": "pre", "repeat": repeat}
        calibration_prepare_marker_ns = marker_timestamp(
            telemetry_markers, "calibration_prepare", calibration_metadata
        )
        attempts = calibration["thermal_start_attempts"]
        attempt_evidence: list[dict[str, Any]] = []
        prior_attempt_end = calibration_prepare_marker_ns
        final_qualification: dict[str, Any] | None = None
        final_qualification_raw: dict[str, Any] | None = None
        final_qualification_marker_ns: int | None = None
        final_precondition_evidence: dict[str, Any] | None = None
        for attempt, record in enumerate(attempts, start=1):
            if not isinstance(record, dict) or set(record) != {
                "attempt",
                "thermal_precondition",
                "qualification",
                "qualification_result_marker_monotonic_ns",
                "measured_process_states",
            }:
                raise ValueError("a deadline thermal attempt has invalid fields")
            if strict_int(record.get("attempt"), "thermal attempt", minimum=1) != attempt:
                raise ValueError("deadline thermal attempts are not consecutive")
            attempt_label = f"{base_label}-attempt-{attempt:02d}"
            precondition_evidence = replay_deadline_precondition(
                record.get("thermal_precondition"),
                expected_label=attempt_label,
                telemetry=telemetry,
                markers=telemetry_markers,
                interval_ms=telemetry_interval_ms,
                required_fraction=telemetry_required_fraction,
                window_seconds=window_seconds,
                target_c=target_c,
                tolerance_c=tolerance_c,
                maximum_slope_c_per_minute=maximum_slope,
                hard_limit_c=hard_limit_c,
                maximum_gap_ms=maximum_gap_ms,
                stability_sensor=stability_sensor,
                safety_sensor=safety_sensor,
            )
            environment = calibration["execution_environment"]
            paused_states = validate_paused_process_states(
                record.get("measured_process_states"),
                expected_pids=[int(environment["pid"])],
                label=f"isolated[{repeat - 1}].thermal_start_attempts[{attempt - 1}]",
            )
            qualification = record.get("qualification")
            qualification_raw, result_marker_ns = replay_deadline_qualification(
                qualification,
                attempt=attempt,
                base_label=attempt_label,
                precondition_evidence=precondition_evidence,
                telemetry=telemetry,
                markers=telemetry_markers,
                result_marker_metadata=calibration_metadata,
                interval_ms=telemetry_interval_ms,
                required_fraction=telemetry_required_fraction,
                window_seconds=window_seconds,
                target_c=target_c,
                tolerance_c=tolerance_c,
                maximum_slope_c_per_minute=maximum_slope,
                hard_limit_c=hard_limit_c,
                maximum_gap_ms=maximum_gap_ms,
                stability_sensor=stability_sensor,
                safety_sensor=safety_sensor,
            )
            if record.get("qualification_result_marker_monotonic_ns") != result_marker_ns:
                raise ValueError("thermal attempt result marker timestamp differs")
            sample_ns = qualification_raw["sample_monotonic_ns"]
            sample_is_ordered = sample_ns is None or (
                int(precondition_evidence["cleanup_end_ns"])
                < int(sample_ns)
                <= int(qualification["qualification_monotonic_ns"])
            )
            if not (
                prior_attempt_end < int(precondition_evidence["prepare_marker_ns"])
                and int(qualification["boundary_monotonic_ns"])
                < int(precondition_evidence["cleanup_end_ns"])
                < int(qualification["qualification_monotonic_ns"])
                < result_marker_ns
                and sample_is_ordered
            ):
                raise ValueError("deadline thermal attempts overlap or reorder")
            passed = qualification.get("passed") is True
            if passed != (attempt == len(attempts)):
                raise ValueError(
                    "deadline must use the first successful thermal qualification"
                )
            prior_attempt_end = result_marker_ns
            attempt_evidence.append(
                {
                    "attempt": attempt,
                    "label": attempt_label,
                    "precondition": precondition_evidence,
                    "qualification": qualification_raw,
                    "qualification_result_marker_monotonic_ns": result_marker_ns,
                    "measured_process_states": paused_states,
                }
            )
            if passed:
                assert isinstance(qualification, dict)
                final_qualification = qualification
                final_qualification_raw = qualification_raw
                final_qualification_marker_ns = result_marker_ns
                final_precondition_evidence = precondition_evidence
        if (
            final_qualification is None
            or final_qualification_raw is None
            or final_qualification_marker_ns is None
            or final_precondition_evidence is None
        ):
            raise ValueError("deadline calibration lacks a successful qualification")
        successful_record = attempts[-1]
        successful_precondition = successful_record["thermal_precondition"]
        if (
            final_precondition != successful_precondition
            or calibration.get("thermal_precondition") != successful_precondition
            or calibration.get("thermal_start_qualification")
            != final_qualification
        ):
            raise ValueError("deadline final qualification aliases are inconsistent")
        calibration_start_marker_ns = marker_timestamp(
            telemetry_markers, "calibration_start", calibration_metadata
        )
        release_ns = strict_int(
            calibration.get("measurement_release_monotonic_ns"),
            f"isolated[{repeat - 1}].measurement_release_monotonic_ns",
            minimum=1,
        )
        calibration_window_marker_ns = marker_timestamp(
            telemetry_markers,
            "calibration_measurement_window",
            calibration_metadata
            | {
                "measurement_start_monotonic_ns": calibration_start_ns,
                "measurement_end_monotonic_ns": calibration_end_ns,
            },
        )
        calibration_end_marker_ns = marker_timestamp(
            telemetry_markers, "calibration_end", calibration_metadata
        )
        if not (
            calibration_prepare_marker_ns
            < prior_attempt_end
            < calibration_start_marker_ns
            == release_ns
            <= calibration_start_ns
            < calibration_end_ns
            <= calibration_window_marker_ns
            <= calibration_end_marker_ns
        ):
            raise ValueError(f"deadline marker chain is invalid for repeat {repeat}")
        if previous_chain_end is not None and (
            previous_chain_end >= calibration_prepare_marker_ns
        ):
            raise ValueError("deadline calibration repeats overlap or reorder")
        previous_chain_end = calibration_end_marker_ns
        reported_start_window = calibration.get("thermal_start")
        if (
            not isinstance(reported_start_window, dict)
            or calibration.get("thermal_start_stable") is not True
        ):
            raise ValueError("a calibration block lacks a stable thermal start")
        _compare_thermal_window(
            reported_start_window,
            final_precondition_evidence["last_window_evidence"][
                "stability_window"
            ],
            expected_fields=THERMAL_WINDOW_FIELDS,
            label="calibration active-boundary thermal window",
        )
        if calibration.get("thermal_start_telemetry") != final_qualification.get(
            "telemetry"
        ):
            raise ValueError("calibration thermal-start telemetry is not qualified")
        expected_label = f"{base_label}-attempt-{len(attempts):02d}"
        if calibration.get("thermal_precondition_label") != expected_label:
            raise ValueError("calibration result has an invalid precondition label")
        environment = calibration["execution_environment"]
        readiness = validate_readiness_affinity(
            calibration.get("readiness_affinity"),
            process_pid=int(environment["pid"]),
            expected_cpu=12,
        )
        actual_start_raw, actual_start_marker_ns = replay_actual_start_qualification(
            calibration.get("thermal_actual_start_qualification"),
            measurement_start_ns=calibration_start_ns,
            not_before_ns=int(final_precondition_evidence["cleanup_end_ns"]),
            qualification_sample_ns=int(final_qualification_raw["sample_monotonic_ns"]),
            telemetry=telemetry,
            markers=telemetry_markers,
            result_marker_metadata=calibration_metadata,
            target_c=target_c,
            tolerance_c=tolerance_c,
            hard_limit_c=hard_limit_c,
            maximum_gap_ms=maximum_gap_ms,
            stability_sensor=stability_sensor,
            safety_sensor=safety_sensor,
        )
        if actual_start_raw["passed"] is not True:
            raise ValueError("calibration actual-start thermal gate did not pass")
        if not (
            calibration_end_ns < actual_start_marker_ns <= calibration_window_marker_ns
        ):
            raise ValueError("calibration actual-start marker is reordered")
        handoff = replay_thermal_handoff(
            calibration.get("thermal_handoff"),
            boundary_ns=int(final_qualification["boundary_monotonic_ns"]),
            cleanup_end_ns=int(final_qualification["cleanup_end_monotonic_ns"]),
            qualification_ns=int(final_qualification["qualification_monotonic_ns"]),
            qualification_result_ns=final_qualification_marker_ns,
            release_ns=release_ns,
            measurement_start_ns=calibration_start_ns,
            maximum_ms=handoff_max_ms,
        )
        qualification_result_ms = (
            final_qualification_marker_ns
            - int(final_qualification["boundary_monotonic_ns"])
        ) / 1_000_000.0
        if qualification_result_ms >= handoff_max_ms:
            raise ValueError(
                "calibration qualification result exceeds the strict handoff bound"
            )
        raw_preconditions.append(
            {
                "label": base_label,
                "stability_sensor": stability_sensor,
                "safety_sensor": safety_sensor,
                "attempts": attempt_evidence,
                "actual_start_qualification": actual_start_raw,
                "thermal_handoff": handoff,
                "readiness_affinity": readiness,
            }
        )
    return {
        "schema_version": 1,
        "deadline_source": "frozen-isolated-p99-factor",
        "deadline_ms": deadline,
        "slo_factor": factor,
        "isolated_pooled_p99_ms": p99,
        "isolated_samples": expected_total,
        "calibration_blocks": expected_blocks,
        "samples_per_block": expected_samples_per_block,
        "percentile_estimator": "pooled-Hyndman-Fan-Type-7",
        "calibration_protocol": calibration_protocol,
        "calibration_trace_sha256": trace_hashes,
        "calibration_telemetry_jsonl_sha256": file_sha256(telemetry_path),
        "calibration_thermal_preconditions": raw_preconditions,
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": file_sha256(summary_path),
        "thermal_lock_sha256": thermal_lock_sha256,
        "thermal_lock_schema_version": THERMAL_LOCK_SCHEMA_VERSION,
        "thermal_target_source": THERMAL_TARGET_SOURCE,
        "thermal_stability_sensor": stability_sensor,
        "thermal_safety_sensor": safety_sensor,
        "thermal_handoff_max_ms": handoff_max_ms,
        "thermal_handoff_boundary": THERMAL_HANDOFF_BOUNDARY,
        "thermal_qualification_max_attempts": (
            THERMAL_QUALIFICATION_MAX_ATTEMPTS
        ),
        "thermal_active_stable_endpoints": THERMAL_ACTIVE_STABLE_ENDPOINTS,
        "thermal_active_stable_spacing_seconds": (
            THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        ),
        "thermal_handoff_rationale": THERMAL_HANDOFF_RATIONALE,
        "thermal_required_fields": list(THERMAL_REQUIRED_FIELDS),
        "guard_lock_path": str(guard_lock_path),
        "guard_lock_sha256": guard_lock_sha256,
        "guard_profile_source": "frozen-quota-aware-lock",
        "profile_guard_ms": profile_guard_ms,
        "guard_lock_protocol": guard_protocol,
        "guard_lock_estimator": guard_estimator,
        "guard_profile_summary_sha256": guard_profile_summary_sha256,
        "guard_telemetry_jsonl_sha256": guard_telemetry_jsonl_sha256,
        "calibration_artifacts": artifacts,
        "calibration_hardware": hardware,
        "calibration_mig": mig,
        "calibration_cpu_affinity": config.get("cpu_affinity"),
        "code_sha256": current_code,
    }


def verify_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != 1:
        raise ValueError("deadline lock must use schema version 1")
    if lock.get("deadline_source") != "frozen-isolated-p99-factor":
        raise ValueError("deadline lock has invalid provenance")
    source = pathlib.Path(str(lock.get("source_summary", "")))
    if not source.is_file() or file_sha256(source) != lock.get(
        "source_summary_sha256"
    ):
        raise ValueError("deadline calibration summary changed or is missing")
    summary = json.loads(source.read_text(encoding="utf-8"))
    rebuilt = build_lock(
        summary,
        source,
        guard_lock_path=pathlib.Path(str(lock.get("guard_lock_path", ""))),
        expected_blocks=int(lock.get("calibration_blocks", -1)),
        expected_samples_per_block=int(lock.get("samples_per_block", -1)),
    )
    if lock != rebuilt:
        raise ValueError("deadline lock fields do not match the source calibration")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--guard-lock", type=pathlib.Path)
    parser.add_argument("--verify", type=pathlib.Path)
    parser.add_argument("--expected-blocks", type=int, default=10)
    parser.add_argument("--expected-samples-per-block", type=int, default=9600)
    args = parser.parse_args()
    if args.verify is not None:
        if (
            args.summary is not None
            or args.output is not None
            or args.guard_lock is not None
        ):
            raise SystemExit("--verify cannot be combined with summary creation")
        verify_lock(json.loads(args.verify.read_text(encoding="utf-8")))
        return 0
    if args.summary is None or args.output is None or args.guard_lock is None:
        raise SystemExit("summary, --output, and --guard-lock are required")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    lock = build_lock(
        summary,
        args.summary,
        guard_lock_path=args.guard_lock,
        expected_blocks=args.expected_blocks,
        expected_samples_per_block=args.expected_samples_per_block,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
