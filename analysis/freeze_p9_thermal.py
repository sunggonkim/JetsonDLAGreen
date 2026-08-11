#!/usr/bin/env python3
"""Create or verify a frozen P9 thermal-preconditioning lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.tegrastats_telemetry import parse_tegrastats_line  # noqa: E402


HASHED_FILES = (
    "analysis/freeze_p9_guard.py",
    "analysis/freeze_p9_thermal.py",
    "benchmarks/trt_inference.cpp",
    "runtime/mig_slack_governor.py",
    "runtime/profile_p9_guard.py",
    "runtime/tegrastats_telemetry.py",
    "scripts/configure_thor_mig.sh",
    "scripts/run_p9_guard_calibration.sh",
    "scripts/run_p9_mig_slack_governor.sh",
    "scripts/run_p9_thermal_pilot.sh",
)

SAMPLE_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "monotonic_ns",
        "raw",
        "parsed",
        "mem_available_mb",
        "collection_errors",
    }
)
MARKER_RECORD_KEYS = frozenset(
    {"schema_version", "record_type", "monotonic_ns", "name", "metadata"}
)
TelemetryRecords = tuple[list[dict[str, Any]], list[dict[str, Any]]]
LOCK_SCHEMA_VERSION = 4
STABILITY_SENSOR = "soc012"
SAFETY_SENSOR = "tj"
THERMAL_HANDOFF_MAX_MS = 500.0
THERMAL_HANDOFF_BOUNDARY = "thermal_measurement_end"
THERMAL_QUALIFICATION_MAX_ATTEMPTS = 3
THERMAL_ACTIVE_STABLE_ENDPOINTS = 3
THERMAL_ACTIVE_STABLE_SPACING_SECONDS = 1.0
THERMAL_HANDOFF_RATIONALE = (
    "active-precondition-boundary-to-critical-measurement-start"
)
TARGET_SOURCE = (
    "minimum-ten-minute-full-soc-soc012-until-three-minute-stable-"
    "with-tj-safety"
)
PILOT_MINIMUM_SECONDS = 600.0
PILOT_MAXIMUM_SECONDS = 900.0
PILOT_WINDOW_SECONDS = 180.0
PILOT_CHECK_INTERVAL_SECONDS = 30.0
PILOT_CONSECUTIVE_PASSES = 3
TEGRASTATS_REQUESTED_INTERVAL_MS = 75.0
TELEMETRY_EVALUATION_INTERVAL_MS = 100.0
THERMAL_CHECK_METADATA_KEYS = frozenset(
    {
        "label",
        "checkpoint_index",
        "scheduled_elapsed_seconds",
        "actual_elapsed_seconds",
        "checkpoint_monotonic_ns",
        "passed",
        "consecutive_passes",
        "window",
    }
)
THERMAL_MEASUREMENT_END_METADATA_KEYS = THERMAL_CHECK_METADATA_KEYS - {"passed"}
THERMAL_BOUNDARY_METADATA_KEYS = frozenset(
    {"label", "checkpoint_index", "scheduled_elapsed_seconds"}
)
THERMAL_WINDOW_KEYS = frozenset(
    {
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
)
THERMAL_MAXIMUM_GAP_NS = 300_000_000
THERMAL_REQUIRED_FIELDS = (
    "ram",
    "mem_available",
    "cpu",
    f"temperature:{STABILITY_SENSOR}",
    f"temperature:{SAFETY_SENSOR}",
    "power:VIN",
)
THERMAL_STATE_DEPENDENCIES = {
    "thermal_target_c": STABILITY_SENSOR,
    "thermal_stability_windows": STABILITY_SENSOR,
    "thermal_hard_limit_c": SAFETY_SENSOR,
    "thermal_handoff_boundary": THERMAL_HANDOFF_BOUNDARY,
    "thermal_qualification_max_attempts": THERMAL_QUALIFICATION_MAX_ATTEMPTS,
    "thermal_active_stable_endpoints": THERMAL_ACTIVE_STABLE_ENDPOINTS,
    "thermal_active_stable_spacing_seconds": (
        THERMAL_ACTIVE_STABLE_SPACING_SECONDS
    ),
    "thermal_handoff_rationale": THERMAL_HANDOFF_RATIONALE,
}


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hashes() -> dict[str, str]:
    return {name: file_sha256(ROOT / name) for name in HASHED_FILES}


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


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _structurally_valid_sample(
    record: dict[str, Any],
    *,
    temperature_sensors: tuple[str, ...] = (SAFETY_SENSOR,),
) -> bool:
    parsed = record.get("parsed", {})
    cpu = parsed.get("cpu", [])
    power = parsed.get("power", {}).get("VIN", {})
    temperatures = parsed.get("temperatures_c", {})
    return (
        isinstance(parsed.get("ram"), dict)
        and finite_number(record.get("mem_available_mb"))
        and isinstance(cpu, list)
        and any(
            isinstance(core, dict)
            and finite_number(core.get("utilization_pct"))
            for core in cpu
        )
        and all(
            finite_number(temperatures.get(sensor))
            for sensor in temperature_sensors
        )
        and finite_number(power.get("current_mw"))
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


def _thermal_window_from_raw(
    samples: list[dict[str, Any]],
    *,
    pilot_start_ns: int,
    reference_ns: int,
    window_seconds: float,
    sensor: str = STABILITY_SENSOR,
) -> dict[str, float | int] | None:
    window_start_ns = max(
        pilot_start_ns,
        reference_ns - int(window_seconds * 1_000_000_000),
    )
    points = [
        (
            int(record["monotonic_ns"]),
            float(record["parsed"]["temperatures_c"][sensor]),
        )
        for record in samples
        if window_start_ns <= int(record["monotonic_ns"]) <= reference_ns
        and finite_number(
            record.get("parsed", {}).get("temperatures_c", {}).get(sensor)
        )
    ]
    if len(points) < 2:
        return None
    times = [
        (timestamp - points[0][0]) / 1_000_000_000.0
        for timestamp, _ in points
    ]
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
        "maximum_gap_seconds": _maximum_gap_seconds(
            [timestamp for timestamp, _ in points], window_start_ns, reference_ns
        ),
    }


def _window_matches_raw(
    reported: object, expected: dict[str, float | int] | None
) -> bool:
    if expected is None:
        return reported is None
    if not isinstance(reported, dict) or set(reported) != THERMAL_WINDOW_KEYS:
        return False
    if type(reported.get("samples")) is not int:
        return False
    for name, expected_value in expected.items():
        reported_value = reported.get(name)
        if name == "samples":
            if reported_value != expected_value:
                return False
        elif (
            not finite_number(reported_value)
            or not math.isclose(
                float(reported_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return False
    return True


def _checkpoint_passes(
    samples: list[dict[str, Any]],
    *,
    pilot_start_ns: int,
    checkpoint_ns: int,
    stability_window: dict[str, float | int] | None,
    safety_window: dict[str, float | int] | None,
    window_seconds: float,
    evaluation_interval_ms: float,
    required_fraction: float,
    hard_limit_c: float,
    maximum_slope_c_per_minute: float,
) -> bool:
    if stability_window is None or safety_window is None:
        return False
    window_start_ns = max(
        pilot_start_ns,
        checkpoint_ns - int(window_seconds * 1_000_000_000),
    )
    selected = [
        record
        for record in samples
        if window_start_ns <= int(record["monotonic_ns"]) < checkpoint_ns
    ]
    expected_samples = math.floor(
        window_seconds * 1000.0 / evaluation_interval_ms
    )
    minimum_samples = max(1, math.floor(expected_samples * required_fraction))
    valid = [
        record
        for record in selected
        if _structurally_valid_sample(
            record,
            temperature_sensors=(STABILITY_SENSOR, SAFETY_SENSOR),
        )
    ]
    valid_timestamps = [int(record["monotonic_ns"]) for record in valid]
    maximum_valid_gap = _maximum_gap_seconds(
        valid_timestamps, window_start_ns, checkpoint_ns
    )
    return (
        len(valid) == len(selected)
        and len(valid) >= minimum_samples
        and int(stability_window["samples"]) >= minimum_samples
        and int(safety_window["samples"]) >= minimum_samples
        and float(stability_window["observed_span_seconds"])
        >= window_seconds * 0.99
        and float(safety_window["observed_span_seconds"])
        >= window_seconds * 0.99
        and maximum_valid_gap is not None
        and maximum_valid_gap <= THERMAL_MAXIMUM_GAP_NS / 1_000_000_000.0
        and finite_number(stability_window.get("maximum_gap_seconds"))
        and float(stability_window["maximum_gap_seconds"])
        <= THERMAL_MAXIMUM_GAP_NS / 1_000_000_000.0
        and finite_number(safety_window.get("maximum_gap_seconds"))
        and float(safety_window["maximum_gap_seconds"])
        <= THERMAL_MAXIMUM_GAP_NS / 1_000_000_000.0
        and abs(float(stability_window["slope_c_per_minute"]))
        <= maximum_slope_c_per_minute
        and float(safety_window["max_c"]) < hard_limit_c
    )


def load_telemetry_jsonl(path: pathlib.Path) -> TelemetryRecords:
    """Load a complete writer trace and validate its provenance-bearing fields."""

    if not path.is_file():
        raise ValueError("telemetry JSONL is missing")
    samples: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    previous_timestamp: int | None = None
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"unterminated telemetry JSONL at line {line_number}"
                )
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid telemetry JSONL at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"telemetry JSONL record is not an object at line {line_number}"
                )
            if type(record.get("schema_version")) is not int or record.get(
                "schema_version"
            ) != 1:
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

            record_type = record.get("record_type")
            if record_type == "sample":
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
                if not finite_number(mem_available) or float(mem_available) < 0.0:
                    raise ValueError(
                        f"telemetry MemAvailable is invalid at line {line_number}"
                    )
                if record.get("collection_errors") != []:
                    raise ValueError(
                        f"telemetry collection errors at line {line_number}"
                    )
                samples.append(record)
            elif record_type == "marker":
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
                markers.append(record)
            else:
                raise ValueError(
                    f"telemetry record type is invalid at line {line_number}"
                )
    return samples, markers


def replay_thermal_stability_protocol(
    telemetry: TelemetryRecords,
    *,
    label: str,
    pilot_start_ns: int,
    measurement_end_ns: int,
    cleanup_end_ns: int,
    reported_checks: object,
    minimum_soak_seconds: float,
    maximum_soak_seconds: float,
    window_seconds: float,
    checkpoint_seconds: float,
    checkpoint_max_lateness_seconds: float,
    required_consecutive_passes: int,
    evaluation_interval_ms: float,
    required_fraction: float,
    hard_limit_c: float,
    maximum_slope_c_per_minute: float,
) -> dict[str, Any]:
    samples, markers = telemetry
    if not label:
        raise ValueError("thermal pilot label is missing")
    protocol_names = {
        "thermal_prepare",
        "thermal_start",
        "thermal_stability_boundary",
        "thermal_stability_check",
        "thermal_measurement_end",
        "thermal_end",
    }
    protocol_markers = [
        marker
        for marker in markers
        if marker["name"].startswith("thermal_")
        and marker["metadata"].get("label") == label
    ]
    if any(marker["name"] not in protocol_names for marker in protocol_markers):
        raise ValueError("thermal pilot contains an unknown protocol marker")
    names = [marker["name"] for marker in protocol_markers]
    if len(protocol_markers) < 6:
        raise ValueError("thermal pilot marker chain is incomplete")
    if names[0:2] != ["thermal_prepare", "thermal_start"]:
        raise ValueError("thermal pilot does not start with prepare/start markers")
    if names[-2:] != ["thermal_measurement_end", "thermal_end"]:
        raise ValueError("thermal pilot does not end with measurement/end markers")
    middle = names[2:-2]
    if len(middle) % 2 != 0 or middle != [
        name
        for _ in range(len(middle) // 2)
        for name in ("thermal_stability_boundary", "thermal_stability_check")
    ]:
        raise ValueError("thermal pilot stability markers are not paired and ordered")
    prepare_marker, start_marker = protocol_markers[:2]
    measurement_marker, end_marker = protocol_markers[-2:]
    if prepare_marker["metadata"] != {"label": label}:
        raise ValueError("thermal prepare marker metadata is invalid")
    if start_marker["metadata"] != {"label": label}:
        raise ValueError("thermal start marker metadata is invalid")
    if int(start_marker["monotonic_ns"]) != pilot_start_ns:
        raise ValueError("thermal start marker differs from the pilot summary")
    if int(measurement_marker["monotonic_ns"]) != measurement_end_ns:
        raise ValueError("thermal measurement marker differs from the pilot summary")
    if int(end_marker["monotonic_ns"]) != cleanup_end_ns:
        raise ValueError("thermal cleanup marker differs from the pilot summary")
    if end_marker["metadata"] != {"label": label, "successful": True}:
        raise ValueError("thermal pilot did not record a successful thermal_end")

    check_pairs = list(
        zip(protocol_markers[2:-2:2], protocol_markers[3:-2:2], strict=True)
    )
    if not isinstance(reported_checks, list) or len(reported_checks) != len(
        check_pairs
    ):
        raise ValueError("thermal stability-check summary count is invalid")
    consecutive_passes = 0
    first_eligible_index: int | None = None
    checkpoint_evidence: list[dict[str, Any]] = []
    for expected_index, (boundary, check) in enumerate(check_pairs):
        boundary_metadata = boundary["metadata"]
        check_metadata = check["metadata"]
        if set(boundary_metadata) != THERMAL_BOUNDARY_METADATA_KEYS:
            raise ValueError("thermal stability boundary metadata is invalid")
        if set(check_metadata) != THERMAL_CHECK_METADATA_KEYS:
            raise ValueError("thermal stability check metadata is invalid")
        scheduled = (expected_index + 1) * checkpoint_seconds
        if (
            type(boundary_metadata.get("checkpoint_index")) is not int
            or boundary_metadata["checkpoint_index"] != expected_index
            or not finite_number(
                boundary_metadata.get("scheduled_elapsed_seconds")
            )
            or not math.isclose(
                float(boundary_metadata["scheduled_elapsed_seconds"]),
                scheduled,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("thermal stability boundary schedule is invalid")
        for name in ("label", "checkpoint_index", "scheduled_elapsed_seconds"):
            if check_metadata.get(name) != boundary_metadata.get(name):
                raise ValueError("thermal stability check does not bind its boundary")
        checkpoint_ns = int(boundary["monotonic_ns"])
        if (
            type(check_metadata.get("checkpoint_monotonic_ns")) is not int
            or check_metadata["checkpoint_monotonic_ns"] != checkpoint_ns
        ):
            raise ValueError("thermal stability check has an invalid boundary clock")
        actual_elapsed = (checkpoint_ns - pilot_start_ns) / 1_000_000_000.0
        if (
            not finite_number(check_metadata.get("actual_elapsed_seconds"))
            or not math.isclose(
                float(check_metadata["actual_elapsed_seconds"]),
                actual_elapsed,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or actual_elapsed < scheduled
            or actual_elapsed > scheduled + checkpoint_max_lateness_seconds
        ):
            raise ValueError("thermal stability checkpoint missed its schedule")
        if int(check["monotonic_ns"]) <= checkpoint_ns:
            raise ValueError("thermal stability result does not follow its boundary")

        raw_window = _thermal_window_from_raw(
            samples,
            pilot_start_ns=pilot_start_ns,
            reference_ns=checkpoint_ns,
            window_seconds=window_seconds,
            sensor=STABILITY_SENSOR,
        )
        safety_window = _thermal_window_from_raw(
            samples,
            pilot_start_ns=pilot_start_ns,
            reference_ns=checkpoint_ns,
            window_seconds=window_seconds,
            sensor=SAFETY_SENSOR,
        )
        if not _window_matches_raw(check_metadata.get("window"), raw_window):
            raise ValueError("thermal stability window differs from raw telemetry")
        passed = _checkpoint_passes(
            samples,
            pilot_start_ns=pilot_start_ns,
            checkpoint_ns=checkpoint_ns,
            stability_window=raw_window,
            safety_window=safety_window,
            window_seconds=window_seconds,
            evaluation_interval_ms=evaluation_interval_ms,
            required_fraction=required_fraction,
            hard_limit_c=hard_limit_c,
            maximum_slope_c_per_minute=maximum_slope_c_per_minute,
        )
        if type(check_metadata.get("passed")) is not bool or check_metadata[
            "passed"
        ] is not passed:
            raise ValueError("thermal stability pass result differs from raw telemetry")
        consecutive_passes = consecutive_passes + 1 if passed else 0
        if (
            type(check_metadata.get("consecutive_passes")) is not int
            or check_metadata["consecutive_passes"] != consecutive_passes
        ):
            raise ValueError("thermal stability consecutive-pass count is invalid")
        if _canonical_json(reported_checks[expected_index]) != _canonical_json(
            check_metadata
        ):
            raise ValueError("thermal stability marker differs from summary metadata")
        if (
            actual_elapsed >= minimum_soak_seconds
            and consecutive_passes >= required_consecutive_passes
            and first_eligible_index is None
        ):
            first_eligible_index = expected_index
        checkpoint_evidence.append(
            {
                "boundary_monotonic_ns": checkpoint_ns,
                "result_monotonic_ns": int(check["monotonic_ns"]),
                "metadata": check_metadata,
                "stability_sensor": STABILITY_SENSOR,
                "safety_sensor": SAFETY_SENSOR,
                "raw_stability_window": raw_window,
                "raw_safety_window": safety_window,
            }
        )

    if not checkpoint_evidence or first_eligible_index is None:
        raise ValueError("thermal pilot never reached the frozen stability condition")
    if first_eligible_index != len(checkpoint_evidence) - 1:
        raise ValueError("thermal pilot continued after its first eligible checkpoint")
    final_evidence = checkpoint_evidence[-1]
    final_metadata = final_evidence["metadata"]
    if float(final_metadata["scheduled_elapsed_seconds"]) > maximum_soak_seconds:
        raise ValueError("thermal pilot exceeded the maximum checkpoint schedule")
    expected_measurement_metadata = {
        name: final_metadata[name]
        for name in THERMAL_MEASUREMENT_END_METADATA_KEYS
    }
    if (
        set(measurement_marker["metadata"])
        != THERMAL_MEASUREMENT_END_METADATA_KEYS
        or _canonical_json(measurement_marker["metadata"])
        != _canonical_json(expected_measurement_metadata)
    ):
        raise ValueError("thermal measurement end does not bind the chosen checkpoint")
    if not (
        int(final_evidence["result_monotonic_ns"])
        < measurement_end_ns
        < cleanup_end_ns
    ):
        raise ValueError("thermal pilot terminal markers are out of order")
    if len(checkpoint_evidence) < required_consecutive_passes or not all(
        evidence["metadata"]["passed"]
        for evidence in checkpoint_evidence[-required_consecutive_passes:]
    ):
        raise ValueError("thermal pilot lacks its final stability confirmation")

    pilot_samples = [
        record
        for record in samples
        if pilot_start_ns <= int(record["monotonic_ns"]) < measurement_end_ns
    ]
    expected_total = math.floor(
        (measurement_end_ns - pilot_start_ns)
        / (evaluation_interval_ms * 1_000_000.0)
    )
    minimum_total = max(1, math.floor(expected_total * required_fraction))
    if len(pilot_samples) < minimum_total:
        raise ValueError("thermal pilot has insufficient raw telemetry coverage")
    pilot_timestamps = [int(record["monotonic_ns"]) for record in pilot_samples]
    pilot_maximum_gap = _maximum_gap_seconds(
        pilot_timestamps, pilot_start_ns, measurement_end_ns
    )
    if (
        pilot_maximum_gap is None
        or pilot_maximum_gap
        > THERMAL_MAXIMUM_GAP_NS / 1_000_000_000.0
    ):
        raise ValueError("thermal pilot raw telemetry has a gap larger than 300 ms")
    pilot_duration_seconds = (
        measurement_end_ns - pilot_start_ns
    ) / 1_000_000_000.0
    pilot_stability_window = _thermal_window_from_raw(
        pilot_samples,
        pilot_start_ns=pilot_start_ns,
        reference_ns=measurement_end_ns,
        window_seconds=pilot_duration_seconds,
        sensor=STABILITY_SENSOR,
    )
    pilot_safety_window = _thermal_window_from_raw(
        pilot_samples,
        pilot_start_ns=pilot_start_ns,
        reference_ns=measurement_end_ns,
        window_seconds=pilot_duration_seconds,
        sensor=SAFETY_SENSOR,
    )
    if pilot_stability_window is None or pilot_safety_window is None:
        raise ValueError("thermal pilot lacks sensor-separated raw telemetry")
    for sensor, evidence in (
        (STABILITY_SENSOR, pilot_stability_window),
        (SAFETY_SENSOR, pilot_safety_window),
    ):
        if (
            int(evidence["samples"]) < minimum_total
            or float(evidence["observed_span_seconds"])
            < pilot_duration_seconds * 0.99
            or not finite_number(evidence.get("maximum_gap_seconds"))
            or float(evidence["maximum_gap_seconds"])
            > THERMAL_MAXIMUM_GAP_NS / 1_000_000_000.0
        ):
            raise ValueError(
                f"thermal pilot {sensor} telemetry has insufficient coverage or a gap"
            )
    if not all(
        _structurally_valid_sample(
            record,
            temperature_sensors=(STABILITY_SENSOR, SAFETY_SENSOR),
        )
        for record in pilot_samples
    ):
        raise ValueError("thermal pilot contains an incomplete telemetry sample")
    if float(pilot_safety_window["max_c"]) >= hard_limit_c:
        raise ValueError("thermal pilot reached the raw hard safety limit")
    return {
        "checks": checkpoint_evidence,
        "final_confirmation": checkpoint_evidence[
            -required_consecutive_passes:
        ],
        "final_window": final_metadata["window"],
        "pilot_total_samples": len(pilot_samples),
        "pilot_valid_samples": len(pilot_samples),
        "pilot_maximum_gap_seconds": pilot_maximum_gap,
        "pilot_stability_sensor_evidence": pilot_stability_window,
        "pilot_safety_sensor_evidence": pilot_safety_window,
        "pilot_max_c": pilot_safety_window["max_c"],
        "thermal_end_monotonic_ns": cleanup_end_ns,
    }


def raw_thermal_evidence(
    path: pathlib.Path,
    *,
    label: str,
    start_ns: int,
    end_ns: int,
    window_seconds: float,
    interval_ms: float,
    required_fraction: float,
    telemetry: TelemetryRecords | None = None,
) -> dict[str, Any]:
    if not path.is_file() or end_ns <= start_ns:
        raise ValueError("thermal telemetry evidence is missing or invalid")
    loaded_samples, loaded_markers = (
        load_telemetry_jsonl(path) if telemetry is None else telemetry
    )
    samples = [
        record
        for record in loaded_samples
        if start_ns <= record["monotonic_ns"] < end_ns
    ]
    markers: dict[tuple[str, str], list[int]] = {}
    for record in loaded_markers:
        marker_label = str(record["metadata"].get("label", ""))
        markers.setdefault((record["name"], marker_label), []).append(
            record["monotonic_ns"]
        )
    if markers.get(("thermal_start", label)) != [start_ns] or markers.get(
        ("thermal_measurement_end", label)
    ) != [end_ns]:
        raise ValueError("thermal telemetry markers do not match the pilot summary")

    duration_seconds = (end_ns - start_ns) / 1_000_000_000.0
    sample_timestamps = [int(record["monotonic_ns"]) for record in samples]
    if any(
        current <= previous
        for previous, current in zip(sample_timestamps, sample_timestamps[1:])
    ):
        raise ValueError("thermal telemetry sample timestamps are not strictly increasing")
    expected_total = math.floor(duration_seconds * 1000.0 / interval_ms)
    required_total = max(1, math.floor(expected_total * required_fraction))
    valid_samples = [
        record for record in samples if _structurally_valid_sample(record)
    ]
    if len(valid_samples) < required_total:
        raise ValueError("thermal telemetry has insufficient structurally valid samples")
    if not valid_samples or end_ns - int(valid_samples[-1]["monotonic_ns"]) > 300_000_000:
        raise ValueError("thermal telemetry is stale at the pilot boundary")
    valid_timestamps = [int(record["monotonic_ns"]) for record in valid_samples]
    if (
        valid_timestamps[0] - start_ns > 300_000_000
        or any(
            current - previous > 300_000_000
            for previous, current in zip(valid_timestamps, valid_timestamps[1:])
        )
    ):
        raise ValueError("thermal telemetry has a gap larger than 300 ms")

    window_start_ns = end_ns - int(window_seconds * 1_000_000_000)
    points = [
        (
            int(record["monotonic_ns"]),
            float(record["parsed"]["temperatures_c"][STABILITY_SENSOR]),
        )
        for record in samples
        if window_start_ns <= int(record["monotonic_ns"]) < end_ns
        and finite_number(
            record.get("parsed", {})
            .get("temperatures_c", {})
            .get(STABILITY_SENSOR)
        )
    ]
    expected_window = math.floor(window_seconds * 1000.0 / interval_ms)
    required_window = max(1, math.floor(expected_window * required_fraction))
    if len(points) < required_window or len(points) < 2:
        raise ValueError("thermal stability window has insufficient raw coverage")
    observed_span = (points[-1][0] - points[0][0]) / 1_000_000_000.0
    if observed_span < window_seconds * 0.99:
        raise ValueError("thermal stability window does not span the required interval")
    if (
        points[0][0] - window_start_ns > 300_000_000
        or end_ns - points[-1][0] > 300_000_000
        or any(
            current[0] - previous[0] > 300_000_000
            for previous, current in zip(points, points[1:])
        )
    ):
        raise ValueError("thermal stability window has a gap larger than 300 ms")
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
    all_safety_temperatures = [
        float(record["parsed"]["temperatures_c"][SAFETY_SENSOR])
        for record in samples
        if finite_number(
            record.get("parsed", {})
            .get("temperatures_c", {})
            .get(SAFETY_SENSOR)
        )
    ]
    return {
        "total_samples": len(samples),
        "valid_samples": len(valid_samples),
        "window_sensor": STABILITY_SENSOR,
        "window_samples": len(values),
        "window_observed_span_seconds": observed_span,
        "window_mean_c": mean_value,
        "window_min_c": min(values),
        "window_max_c": max(values),
        "window_latest_c": values[-1],
        "window_slope_c_per_minute": slope,
        "pilot_max_sensor": SAFETY_SENSOR,
        "pilot_max_c": max(all_safety_temperatures),
    }


def build_lock(
    summary: dict[str, Any],
    summary_path: pathlib.Path,
    *,
    minimum_soak_seconds: float = PILOT_MINIMUM_SECONDS,
    maximum_soak_seconds: float = PILOT_MAXIMUM_SECONDS,
    minimum_window_seconds: float = PILOT_WINDOW_SECONDS,
    maximum_slope_c_per_minute: float = 0.2,
) -> dict[str, Any]:
    current_code = code_hashes()
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("thermal pilot lacks artifact provenance")
    implementation = artifacts.get("implementation_sha256")
    if not isinstance(implementation, dict) or any(
        implementation.get(name) != digest
        for name, digest in current_code.items()
    ):
        raise ValueError("thermal pilot was not produced by the current implementation")
    hardware = summary.get("hardware")
    mig = summary.get("mig")
    if not isinstance(hardware, dict) or not isinstance(mig, dict):
        raise ValueError("thermal pilot lacks hardware or MIG provenance")
    if summary.get("schema_version") != 4:
        raise ValueError("thermal pilot must use schema version 4")
    pilot = summary.get("thermal_pilot")
    if not isinstance(pilot, dict) or summary.get("policies") != []:
        raise ValueError("summary is not a thermal-pilot run")
    if (
        pilot.get("stability_sensor") != STABILITY_SENSOR
        or pilot.get("safety_sensor") != SAFETY_SENSOR
    ):
        raise ValueError("thermal pilot reports invalid sensor roles")
    duration_seconds = float(pilot["duration_seconds"])
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds < minimum_soak_seconds
        or duration_seconds > maximum_soak_seconds + 1.0
    ):
        raise ValueError("thermal soak is outside the frozen stability protocol")
    window = pilot.get("last_window")
    if not isinstance(window, dict):
        raise ValueError("thermal pilot lacks a stable-window summary")
    requested_window_seconds = float(window["window_seconds"])
    observed_span_seconds = float(window.get("observed_span_seconds", 0.0))
    if (
        not math.isfinite(requested_window_seconds)
        or requested_window_seconds < minimum_window_seconds
    ):
        raise ValueError("thermal stability window is too short")
    if (
        not math.isfinite(observed_span_seconds)
        or observed_span_seconds < minimum_window_seconds * 0.99
    ):
        raise ValueError("thermal pilot did not observe the full stability window")
    config = summary.get("config", {})
    if {
        "thermal_pilot_seconds": config.get("thermal_pilot_seconds"),
        "thermal_pilot_maximum_seconds": config.get(
            "thermal_pilot_maximum_seconds"
        ),
        "thermal_window_seconds": config.get("thermal_window_seconds"),
        "thermal_max_slope_c_per_minute": config.get(
            "thermal_max_slope_c_per_minute"
        ),
        "calibration_repeats": config.get("calibration_repeats"),
        "samples_per_epoch": config.get("samples_per_epoch"),
        "warmup": config.get("warmup"),
        "burst_size": config.get("burst_size"),
        "period_ms": config.get("period_ms"),
        "thermal_timeout_seconds": config.get("thermal_timeout_seconds"),
        "thermal_stability_checkpoint_seconds": config.get(
            "thermal_stability_checkpoint_seconds"
        ),
        "thermal_stability_checkpoint_max_lateness_seconds": config.get(
            "thermal_stability_checkpoint_max_lateness_seconds"
        ),
        "thermal_required_stable_checkpoints": config.get(
            "thermal_required_stable_checkpoints"
        ),
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
        "tegrastats_requested_interval_ms": config.get(
            "tegrastats_requested_interval_ms"
        ),
        "telemetry_required_fields": config.get("telemetry_required_fields"),
        "telemetry_stale_after_ms": config.get("telemetry_stale_after_ms"),
        "telemetry_max_gap_ms": config.get("telemetry_max_gap_ms"),
    } != {
        "thermal_pilot_seconds": 600.0,
        "thermal_pilot_maximum_seconds": PILOT_MAXIMUM_SECONDS,
        "thermal_window_seconds": 180.0,
        "thermal_max_slope_c_per_minute": maximum_slope_c_per_minute,
        "calibration_repeats": 1,
        "samples_per_epoch": 160,
        "warmup": 100,
        "burst_size": 8,
        "period_ms": 20.0,
        "thermal_timeout_seconds": PILOT_MAXIMUM_SECONDS,
        "thermal_stability_checkpoint_seconds": PILOT_CHECK_INTERVAL_SECONDS,
        "thermal_stability_checkpoint_max_lateness_seconds": 1.0,
        "thermal_required_stable_checkpoints": PILOT_CONSECUTIVE_PASSES,
        "thermal_stability_sensor": STABILITY_SENSOR,
        "thermal_safety_sensor": SAFETY_SENSOR,
        "thermal_handoff_max_ms": THERMAL_HANDOFF_MAX_MS,
        "thermal_handoff_boundary": THERMAL_HANDOFF_BOUNDARY,
        "thermal_qualification_max_attempts": THERMAL_QUALIFICATION_MAX_ATTEMPTS,
        "thermal_active_stable_endpoints": THERMAL_ACTIVE_STABLE_ENDPOINTS,
        "thermal_active_stable_spacing_seconds": (
            THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        ),
        "tegrastats_requested_interval_ms": TEGRASTATS_REQUESTED_INTERVAL_MS,
        "telemetry_required_fields": list(THERMAL_REQUIRED_FIELDS),
        "telemetry_stale_after_ms": 300.0,
        "telemetry_max_gap_ms": 300.0,
    } or "thermal_qualification_dwell_seconds" in config:
        raise ValueError("thermal pilot protocol differs from the frozen design")
    cpu_affinity = config.get("cpu_affinity")
    if cpu_affinity != {
        "critical": [12],
        "pressure": list(range(11)),
        "mps": [11],
        "telemetry": [13],
    }:
        raise ValueError("thermal pilot CPU affinity does not match the formal protocol")
    interval_ms = float(config.get("telemetry_interval_ms", 0.0))
    required_fraction = float(config.get("telemetry_required_fraction", 0.0))
    if (
        not math.isfinite(interval_ms)
        or not math.isfinite(required_fraction)
        or interval_ms <= 0.0
        or not 0.0 < required_fraction <= 1.0
    ):
        raise ValueError("thermal pilot lacks telemetry cadence provenance")
    if not math.isclose(
        interval_ms,
        TELEMETRY_EVALUATION_INTERVAL_MS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("thermal pilot uses an unfrozen evaluation cadence")
    expected_samples = math.floor(minimum_window_seconds * 1000.0 / interval_ms)
    minimum_samples = max(1, math.floor(expected_samples * required_fraction))
    if int(window.get("samples", 0)) < minimum_samples:
        raise ValueError("thermal pilot stability window has insufficient coverage")
    slope = float(window["slope_c_per_minute"])
    if not math.isfinite(slope) or abs(slope) > maximum_slope_c_per_minute:
        raise ValueError("thermal pilot did not reach a stable slope")
    target = float(window["mean_c"])
    hard_limit = float(summary["config"]["thermal_hard_limit_c"])
    platform_hard_limit = float(
        summary["config"].get("platform_thermal_hard_limit_c", math.nan)
    )
    observed_max = float(
        pilot["telemetry"]["temperatures_c"][SAFETY_SENSOR]["max"]
    )
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (target, hard_limit, platform_hard_limit, observed_max)
    ):
        raise ValueError("thermal pilot contains invalid temperatures")
    if hard_limit > platform_hard_limit:
        raise ValueError("thermal hard limit exceeds the platform safety bound")
    if observed_max >= hard_limit:
        raise ValueError("thermal pilot reached the hard safety limit")
    if pilot["telemetry"]["health"]["healthy"] is not True:
        raise ValueError("thermal pilot telemetry is unhealthy")
    directory = summary_path.resolve().parent
    telemetry_path = directory / "telemetry.jsonl"
    measurement_start_ns = int(pilot.get("measurement_start_monotonic_ns", -1))
    measurement_end_ns = int(pilot.get("measurement_end_monotonic_ns", -1))
    cleanup_end_ns = int(pilot.get("cleanup_end_monotonic_ns", -1))
    telemetry_records = load_telemetry_jsonl(telemetry_path)
    raw_evidence = raw_thermal_evidence(
        telemetry_path,
        label=str(pilot.get("label", "")),
        start_ns=measurement_start_ns,
        end_ns=measurement_end_ns,
        window_seconds=minimum_window_seconds,
        interval_ms=interval_ms,
        required_fraction=required_fraction,
        telemetry=telemetry_records,
    )
    if not math.isclose(
        duration_seconds,
        (measurement_end_ns - measurement_start_ns) / 1_000_000_000.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("thermal pilot duration differs from raw markers")
    if pilot.get("termination_reason") != "stable-checkpoints":
        raise ValueError("thermal pilot has an invalid termination reason")
    if pilot.get("stability_checkpoint_seconds") != PILOT_CHECK_INTERVAL_SECONDS:
        raise ValueError("thermal pilot has an invalid checkpoint interval")
    if (
        pilot.get("required_consecutive_stable_checkpoints")
        != PILOT_CONSECUTIVE_PASSES
    ):
        raise ValueError("thermal pilot has an invalid stability confirmation count")
    protocol_evidence = replay_thermal_stability_protocol(
        telemetry_records,
        label=str(pilot.get("label", "")),
        pilot_start_ns=measurement_start_ns,
        measurement_end_ns=measurement_end_ns,
        cleanup_end_ns=cleanup_end_ns,
        reported_checks=pilot.get("stability_checks"),
        minimum_soak_seconds=minimum_soak_seconds,
        maximum_soak_seconds=maximum_soak_seconds,
        window_seconds=minimum_window_seconds,
        checkpoint_seconds=PILOT_CHECK_INTERVAL_SECONDS,
        checkpoint_max_lateness_seconds=1.0,
        required_consecutive_passes=PILOT_CONSECUTIVE_PASSES,
        evaluation_interval_ms=interval_ms,
        required_fraction=required_fraction,
        hard_limit_c=hard_limit,
        maximum_slope_c_per_minute=maximum_slope_c_per_minute,
    )
    if _canonical_json(window) != _canonical_json(
        protocol_evidence["final_window"]
    ):
        raise ValueError("thermal pilot last window differs from its final checkpoint")
    target = float(protocol_evidence["final_window"]["mean_c"])
    reported_maximum_gap = pilot.get("maximum_gap_seconds")
    if (
        not finite_number(reported_maximum_gap)
        or not math.isclose(
            float(reported_maximum_gap),
            float(protocol_evidence["pilot_maximum_gap_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("thermal pilot maximum gap differs from raw telemetry")
    if not math.isclose(
        observed_max,
        float(protocol_evidence["pilot_max_c"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("thermal pilot maximum differs from raw telemetry")
    clock_snapshot = directory / "jetson-clocks.txt"
    power_mode_snapshot = directory / "nvpmodel.txt"
    clock_text = clock_snapshot.read_text(encoding="utf-8")
    if (
        "FAN Dynamic Speed Control=disabled" not in clock_text
        or "pwm1=255" not in clock_text
    ):
        raise ValueError("thermal pilot did not record maximum fan PWM")
    if (
        "gpu-gpc-0 MinFreq=1575000000 MaxFreq=1575000000" not in clock_text
        or "EMC MinFreq=4266000000 MaxFreq=4266000000" not in clock_text
    ):
        raise ValueError("thermal pilot did not record locked GPU and EMC clocks")
    if "NV Power Mode: MAXN" not in power_mode_snapshot.read_text(
        encoding="utf-8"
    ):
        raise ValueError("thermal pilot did not run in MAXN")
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "target_source": TARGET_SOURCE,
        "stability_sensor": STABILITY_SENSOR,
        "safety_sensor": SAFETY_SENSOR,
        "thermal_handoff_max_ms": THERMAL_HANDOFF_MAX_MS,
        "thermal_handoff_boundary": THERMAL_HANDOFF_BOUNDARY,
        "thermal_qualification_max_attempts": THERMAL_QUALIFICATION_MAX_ATTEMPTS,
        "thermal_active_stable_endpoints": THERMAL_ACTIVE_STABLE_ENDPOINTS,
        "thermal_active_stable_spacing_seconds": (
            THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        ),
        "thermal_handoff_rationale": THERMAL_HANDOFF_RATIONALE,
        "state_dependencies": dict(THERMAL_STATE_DEPENDENCIES),
        "minimum_soak_seconds": minimum_soak_seconds,
        "maximum_soak_seconds": maximum_soak_seconds,
        "pilot_stability_window_seconds": minimum_window_seconds,
        "pilot_minimum_window_samples": minimum_samples,
        "target_c": target,
        "tolerance_c": 1.0,
        "stability_window_seconds": 60.0,
        "maximum_slope_c_per_minute": maximum_slope_c_per_minute,
        "hard_limit_c": hard_limit,
        "telemetry_interval_ms": interval_ms,
        "tegrastats_requested_interval_ms": TEGRASTATS_REQUESTED_INTERVAL_MS,
        "telemetry_required_fraction": required_fraction,
        "telemetry_required_fields": list(THERMAL_REQUIRED_FIELDS),
        "telemetry_max_gap_ms": THERMAL_MAXIMUM_GAP_NS / 1_000_000.0,
        "stability_checkpoint_seconds": PILOT_CHECK_INTERVAL_SECONDS,
        "stability_checkpoint_max_lateness_seconds": 1.0,
        "required_consecutive_stable_checkpoints": PILOT_CONSECUTIVE_PASSES,
        "pilot_observed_max_c": observed_max,
        "pilot_observed_safety_max_c": observed_max,
        "pilot_maximum_gap_seconds": protocol_evidence[
            "pilot_maximum_gap_seconds"
        ],
        "pilot_stability_check_count": len(protocol_evidence["checks"]),
        "pilot_termination_reason": "stable-checkpoints",
        "pilot_window": protocol_evidence["final_window"],
        "pilot_stability_sensor_evidence": protocol_evidence[
            "pilot_stability_sensor_evidence"
        ],
        "pilot_safety_sensor_evidence": protocol_evidence[
            "pilot_safety_sensor_evidence"
        ],
        "pilot_final_stability_confirmation": protocol_evidence[
            "final_confirmation"
        ],
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": file_sha256(summary_path),
        "jetson_clocks_sha256": file_sha256(clock_snapshot),
        "nvpmodel_sha256": file_sha256(power_mode_snapshot),
        "telemetry_jsonl_sha256": file_sha256(telemetry_path),
        "raw_thermal_evidence": raw_evidence,
        "pilot_artifacts": artifacts,
        "pilot_hardware": hardware,
        "pilot_mig": mig,
        "pilot_cpu_affinity": cpu_affinity,
        "code_sha256": current_code,
    }


def verify_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError(
            f"thermal lock must use schema version {LOCK_SCHEMA_VERSION}"
        )
    if lock.get("target_source") != TARGET_SOURCE:
        raise ValueError("thermal lock has invalid provenance")
    if (
        lock.get("stability_sensor") != STABILITY_SENSOR
        or lock.get("safety_sensor") != SAFETY_SENSOR
        or lock.get("thermal_handoff_max_ms") != THERMAL_HANDOFF_MAX_MS
        or lock.get("thermal_handoff_boundary") != THERMAL_HANDOFF_BOUNDARY
        or lock.get("thermal_qualification_max_attempts")
        != THERMAL_QUALIFICATION_MAX_ATTEMPTS
        or lock.get("thermal_active_stable_endpoints")
        != THERMAL_ACTIVE_STABLE_ENDPOINTS
        or lock.get("thermal_active_stable_spacing_seconds")
        != THERMAL_ACTIVE_STABLE_SPACING_SECONDS
        or "thermal_qualification_dwell_seconds" in lock
        or lock.get("thermal_handoff_rationale")
        != THERMAL_HANDOFF_RATIONALE
        or lock.get("telemetry_required_fields")
        != list(THERMAL_REQUIRED_FIELDS)
        or lock.get("state_dependencies") != THERMAL_STATE_DEPENDENCIES
    ):
        raise ValueError("thermal lock has invalid sensor dependencies")
    source = pathlib.Path(str(lock.get("source_summary", "")))
    if not source.is_file() or file_sha256(source) != lock.get(
        "source_summary_sha256"
    ):
        raise ValueError("thermal pilot summary changed or is missing")
    directory = source.parent
    if file_sha256(directory / "jetson-clocks.txt") != lock.get(
        "jetson_clocks_sha256"
    ):
        raise ValueError("jetson_clocks snapshot changed or is missing")
    if file_sha256(directory / "nvpmodel.txt") != lock.get("nvpmodel_sha256"):
        raise ValueError("nvpmodel snapshot changed or is missing")
    summary = json.loads(source.read_text(encoding="utf-8"))
    rebuilt = build_lock(summary, source)
    if lock != rebuilt:
        raise ValueError("thermal lock fields do not match the source pilot")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verify", type=pathlib.Path)
    args = parser.parse_args()
    if args.verify is not None:
        if args.summary is not None or args.output is not None:
            raise SystemExit("--verify cannot be combined with summary creation")
        verify_lock(json.loads(args.verify.read_text(encoding="utf-8")))
        return 0
    if args.summary is None or args.output is None:
        raise SystemExit("summary and --output are required")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    lock = build_lock(summary, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
