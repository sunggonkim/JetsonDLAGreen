#!/usr/bin/env python3
import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.tegrastats_telemetry import parse_tegrastats_line


SPEC = importlib.util.spec_from_file_location(
    "freeze_p9_deadline", ROOT / "analysis" / "freeze_p9_deadline.py"
)
assert SPEC is not None and SPEC.loader is not None
FREEZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FREEZE
SPEC.loader.exec_module(FREEZE)


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": FREEZE.percentile(values, 0.50),
        "p95_ms": FREEZE.percentile(values, 0.95),
        "p99_ms": FREEZE.percentile(values, 0.99),
        "p999_ms": FREEZE.percentile(values, 0.999),
        "max_ms": max(values),
    }


class FreezeP9DeadlineTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(FREEZE, "verify_guard_lock")
        self.verify_guard_lock = patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def guard_profile() -> dict[str, dict[str, dict[str, float]]]:
        return {
            "resident-1g": {
                "25": {"language": 0.5, "audio": 0.8},
                "50": {"language": 0.4, "audio": 0.7},
                "100": {"language": 0.3, "audio": 0.6},
            },
            "borrower-2g": {
                "100": {"language": 0.3, "audio": 0.6},
            },
        }

    @staticmethod
    def engine_hashes() -> dict[str, str]:
        result: dict[str, str] = {}
        for placement, quotas in FREEZE.GUARD_QUOTAS.items():
            prefix = "resident-1g" if placement == "resident-1g" else "borrower-2g"
            for quota in quotas:
                for model in FREEZE.GUARD_MODELS.values():
                    name = f"{prefix}-q{quota}-{model}"
                    result[name] = hashlib.sha256(name.encode()).hexdigest()
        result["critical-2g-resnet50-v2"] = "c" * 64
        return result

    def summary(self) -> dict:
        return {
            "schema_version": 4,
            "hardware": {"gpu_product_name": "NVIDIA Thor"},
            "mig": {"critical_uuid": "big", "resident_uuid": "small"},
            "artifacts": {
                "benchmark_sha256": "b" * 64,
                "engines_sha256": self.engine_hashes(),
                "implementation_sha256": FREEZE.code_hashes(),
            },
            "config": {
                "calibration_only": True,
                "calibration_repeats": 2,
                "samples_per_epoch": 5,
                "warmup": 100,
                "burst_size": 8,
                "period_ms": 20.0,
                "slo_factor": 1.1,
                "cpu_affinity": {
                    "critical": [12],
                    "pressure": list(range(11)),
                    "mps": [11],
                    "telemetry": [13],
                },
                "thermal_target_c": 74.8,
                "thermal_tolerance_c": 1.0,
                "thermal_window_seconds": 60.0,
                "thermal_max_slope_c_per_minute": 0.2,
                "thermal_hard_limit_c": 104.0,
                "thermal_stability_sensor": "soc012",
                "thermal_safety_sensor": "tj",
                "thermal_handoff_max_ms": 500.0,
                "thermal_handoff_boundary": "thermal_measurement_end",
                "thermal_qualification_max_attempts": 3,
                "thermal_active_stable_endpoints": 3,
                "thermal_active_stable_spacing_seconds": 1.0,
                "thermal_calibration_preconditioning": (
                    "per-repeat-preloaded-critical"
                ),
                "start_protocol": (
                    "post-warmup-stop-barrier-with-bounded-thermal-handoff"
                ),
                "telemetry_interval_ms": 100.0,
                "telemetry_required_fraction": 0.8,
                "telemetry_required_fields": list(
                    FREEZE.THERMAL_REQUIRED_FIELDS
                ),
                "telemetry_stale_after_ms": 300.0,
                "telemetry_max_gap_ms": 300.0,
                "guard_lock_sha256": None,
                "guard_profile_source": "frozen-quota-aware-lock",
                "profile_guard_ms": self.guard_profile(),
                "guard_override_ms": None,
            },
            "isolated": [
                {
                    "completed_requests": 5,
                    "release_to_completion": {"p99_ms": 5.0},
                    "measurement_start_monotonic_ns": 63_300_000_000,
                    "measurement_end_monotonic_ns": 63_400_000_000,
                    "thermal_start": {
                        "samples": 598,
                        "observed_span_seconds": 59.7,
                        "window_seconds": 60.0,
                        "mean_c": 74.8,
                        "min_c": 74.8,
                        "max_c": 74.8,
                        "latest_c": 74.8,
                        "slope_c_per_minute": 0.0,
                        "maximum_gap_seconds": 0.1,
                    },
                    "thermal_start_stable": True,
                },
                {
                    "completed_requests": 5,
                    "release_to_completion": {"p99_ms": 5.0},
                    "measurement_start_monotonic_ns": 126_300_000_000,
                    "measurement_end_monotonic_ns": 126_400_000_000,
                    "thermal_start": {
                        "samples": 598,
                        "observed_span_seconds": 59.7,
                        "window_seconds": 60.0,
                        "mean_c": 74.8,
                        "min_c": 74.8,
                        "max_c": 74.8,
                        "latest_c": 74.8,
                        "slope_c_per_minute": 0.0,
                        "maximum_gap_seconds": 0.1,
                    },
                    "thermal_start_stable": True,
                },
            ],
            "isolated_preconditions": [
                {
                    "label": "pre-pre-calibration-r1-attempt-01",
                    "duration_seconds": 62.0,
                    "measurement_start_monotonic_ns": 1_000_000_000,
                    "measurement_end_monotonic_ns": 63_000_000_000,
                    "cleanup_end_monotonic_ns": 63_025_000_000,
                    "target_c": 74.8,
                    "stability_sensor": "soc012",
                    "safety_sensor": "tj",
                    "last_window": {
                        "samples": 600,
                        "observed_span_seconds": 59.9,
                        "window_seconds": 60.0,
                        "mean_c": 74.8,
                        "min_c": 74.8,
                        "max_c": 74.8,
                        "latest_c": 74.8,
                        "slope_c_per_minute": 0.0,
                        "maximum_gap_seconds": 0.1,
                    },
                    "active_stability_checks": [],
                    "active_stable_endpoints": 3,
                    "active_stable_spacing_seconds": 1.0,
                    "termination_reason": "active-stability-endpoints",
                    "pressure_rate_per_second": 100.0,
                    "telemetry": {
                        "health": {"healthy": True},
                        "temperatures_c": {"tj": {"max": 89.5}},
                    },
                },
                {
                    "label": "pre-pre-calibration-r2-attempt-01",
                    "duration_seconds": 62.0,
                    "measurement_start_monotonic_ns": 64_000_000_000,
                    "measurement_end_monotonic_ns": 126_000_000_000,
                    "cleanup_end_monotonic_ns": 126_025_000_000,
                    "target_c": 74.8,
                    "stability_sensor": "soc012",
                    "safety_sensor": "tj",
                    "last_window": {
                        "samples": 600,
                        "observed_span_seconds": 59.9,
                        "window_seconds": 60.0,
                        "mean_c": 74.8,
                        "min_c": 74.8,
                        "max_c": 74.8,
                        "latest_c": 74.8,
                        "slope_c_per_minute": 0.0,
                        "maximum_gap_seconds": 0.1,
                    },
                    "active_stability_checks": [],
                    "active_stable_endpoints": 3,
                    "active_stable_spacing_seconds": 1.0,
                    "termination_reason": "active-stability-endpoints",
                    "pressure_rate_per_second": 100.0,
                    "telemetry": {
                        "health": {"healthy": True},
                        "temperatures_c": {"tj": {"max": 89.5}},
                    },
                },
            ],
            "isolated_pooled_samples": 10,
            "isolated_pooled_p99_ms": 5.0,
            "deadline_ms": 5.5,
            "policies": [],
        }

    @staticmethod
    def write_thermal_lock(directory: pathlib.Path, summary: dict) -> None:
        (directory / "thermal-lock.json").write_text(
            json.dumps(
                {
                    "schema_version": FREEZE.THERMAL_LOCK_SCHEMA_VERSION,
                    "target_source": FREEZE.THERMAL_TARGET_SOURCE,
                    "stability_sensor": FREEZE.THERMAL_STABILITY_SENSOR,
                    "safety_sensor": FREEZE.THERMAL_SAFETY_SENSOR,
                    "thermal_handoff_max_ms": FREEZE.THERMAL_HANDOFF_MAX_MS,
                    "thermal_handoff_boundary": (
                        FREEZE.THERMAL_HANDOFF_BOUNDARY
                    ),
                    "thermal_qualification_max_attempts": (
                        FREEZE.THERMAL_QUALIFICATION_MAX_ATTEMPTS
                    ),
                    "thermal_active_stable_endpoints": (
                        FREEZE.THERMAL_ACTIVE_STABLE_ENDPOINTS
                    ),
                    "thermal_active_stable_spacing_seconds": (
                        FREEZE.THERMAL_ACTIVE_STABLE_SPACING_SECONDS
                    ),
                    "thermal_handoff_rationale": (
                        FREEZE.THERMAL_HANDOFF_RATIONALE
                    ),
                    "target_c": 74.8,
                    "tolerance_c": 1.0,
                    "stability_window_seconds": 60.0,
                    "maximum_slope_c_per_minute": 0.2,
                    "hard_limit_c": 104.0,
                    "telemetry_interval_ms": 100.0,
                    "telemetry_required_fraction": 0.8,
                    "telemetry_required_fields": list(
                        FREEZE.THERMAL_REQUIRED_FIELDS
                    ),
                    "telemetry_max_gap_ms": 300.0,
                    "pilot_artifacts": summary["artifacts"],
                    "pilot_hardware": summary["hardware"],
                    "pilot_mig": summary["mig"],
                    "pilot_cpu_affinity": summary["config"]["cpu_affinity"],
                }
            ),
            encoding="utf-8",
        )

    def write_guard_lock(self, directory: pathlib.Path, summary: dict) -> None:
        implementation = summary["artifacts"]["implementation_sha256"]
        engine_hashes = summary["artifacts"]["engines_sha256"]
        artifacts = {
            "benchmark": {
                "path": str((directory / "jdg-trt-bench").resolve()),
                "sha256": summary["artifacts"]["benchmark_sha256"],
            }
        }
        for name, relative_path in FREEZE.GUARD_IMPLEMENTATION_ARTIFACTS.items():
            artifacts[name] = {
                "path": str((ROOT / relative_path).resolve()),
                "sha256": implementation[relative_path],
            }
        for placement, quotas in FREEZE.GUARD_QUOTAS.items():
            for quota in quotas:
                for modality, model in FREEZE.GUARD_MODELS.items():
                    prefix = (
                        "resident-1g"
                        if placement == "resident-1g"
                        else "borrower-2g"
                    )
                    deadline_name = f"{prefix}-q{quota}-{model}"
                    guard_name = f"engine:{placement}:q{quota}:{modality}"
                    artifacts[guard_name] = {
                        "path": str(
                            (
                                directory
                                / "models"
                                / "engines"
                                / f"{'mig-1g' if placement == 'resident-1g' else 'mig-2g'}-q{quota}"
                                / f"{model}.engine"
                            ).resolve()
                        ),
                        "sha256": engine_hashes[deadline_name],
                    }
        artifacts["engine:critical:2g:resnet50-v2"] = {
            "path": str(
                (
                    directory
                    / "models"
                    / "engines"
                    / "mig-2g"
                    / "resnet50-v2.engine"
                ).resolve()
            ),
            "sha256": engine_hashes["critical-2g-resnet50-v2"],
        }
        guards = {
            placement: {
                quota: {
                    modality: {
                        "guard_ms": value,
                        "raw_pooled_p999_ms": value / 1.2,
                        "samples": 10_000,
                        "observed_max_ms": value,
                    }
                    for modality, value in modalities.items()
                }
                for quota, modalities in quotas.items()
            }
            for placement, quotas in self.guard_profile().items()
        }
        thermal_path = directory / "thermal-lock.json"
        lock = {
            "schema_version": 3,
            "kind": "p9-quota-aware-guard-lock",
            "protocol": {
                "blocks": 10,
                "events_per_block": 1_000,
                "formal_period_ms": 20.0,
            },
            "estimator": {
                "quantile": 0.999,
                "method": "pooled-empirical-Hyndman-Fan-Type-7",
                "margin": 1.2,
                "rounding": {"mode": "upward", "quantum_ms": 0.1},
            },
            "guards": guards,
            "source": {
                "profile_summary_sha256": "1" * 64,
                "telemetry_jsonl_sha256": "2" * 64,
            },
            "thermal_lock": {
                "path": str(thermal_path.resolve()),
                "sha256": FREEZE.file_sha256(thermal_path),
            },
            "hardware": summary["hardware"],
            "mig": {
                "big_uuid": summary["mig"]["critical_uuid"],
                "small_uuid": summary["mig"]["resident_uuid"],
            },
            "cpu_affinity": summary["config"]["cpu_affinity"],
            "artifacts": artifacts,
        }
        guard_path = directory / "guard-lock.json"
        guard_path.write_text(json.dumps(lock), encoding="utf-8")
        summary["config"]["guard_lock_sha256"] = FREEZE.file_sha256(guard_path)

    @staticmethod
    def stored_thermal_aggregate(
        evidence: dict,
        telemetry: FREEZE.TelemetryRecords,
        *,
        end_inclusive: bool = False,
    ) -> dict:
        interval = evidence["interval"]
        samples, _markers = telemetry
        selected = [
            record
            for record in samples
            if interval["start_ns"] <= record["monotonic_ns"]
            and (
                record["monotonic_ns"] < interval["end_ns"]
                or (
                    end_inclusive
                    and record["monotonic_ns"] == interval["end_ns"]
                )
            )
        ]
        observed_gap_ns = int(
            round(
                evidence["stability_window"]["maximum_gap_seconds"]
                * 1_000_000_000.0
            )
        )
        temperatures = {
            sensor: FREEZE._numeric_temperature_summary(
                [
                    record["parsed"]["temperatures_c"][sensor]
                    for record in selected
                ]
            )
            for sensor in (
                FREEZE.THERMAL_STABILITY_SENSOR,
                FREEZE.THERMAL_SAFETY_SENSOR,
            )
        }
        return {
            "schema_version": 1,
            "interval": {
                **interval,
                "duration_ns": interval["end_ns"] - interval["start_ns"],
                "end_inclusive": end_inclusive,
            },
            "total_samples": evidence["total_samples"],
            "valid_samples": evidence["valid_samples"],
            "invalid_samples": 0,
            "health": {
                "healthy": True,
                "reasons": [],
                "required_fields": list(FREEZE.THERMAL_REQUIRED_FIELDS),
                "missing_counts": {
                    field: 0 for field in FREEZE.THERMAL_REQUIRED_FIELDS
                },
                "incomplete_samples": 0,
                "maximum_valid_gap_ns": 300_000_000,
                "observed_maximum_valid_gap_ns": observed_gap_ns,
                "valid_gap_exceeded": False,
            },
            "temperatures_c": temperatures,
        }

    def prepared_summary(self, directory: pathlib.Path) -> dict:
        summary = self.summary()
        engine = directory / "models" / "engines" / "mig-2g" / "resnet50-v2.engine"
        engine.parent.mkdir(parents=True)
        engine.write_bytes(b"critical-resnet50-engine")
        summary["artifacts"]["engines_sha256"][
            "critical-2g-resnet50-v2"
        ] = FREEZE.file_sha256(engine)
        self.write_thermal_lock(directory, summary)
        self.write_guard_lock(directory, summary)
        raw = directory / "raw"
        raw.mkdir()
        for repeat in (1, 2):
            latency = 5.0 + (repeat - 1) * 0.001
            gpu = 0.5 + (repeat - 1) * 0.001
            queue = latency - gpu
            columns = {
                "release_to_completion": [latency] * 5,
                "gpu_service": [gpu] * 5,
                "queue_delay": [queue] * 5,
                "gate_overhead": [0.0] * 5,
                "drain": [0.0] * 5,
                "resume": [0.0] * 5,
            }
            (raw / f"isolated-pre-r{repeat}.csv").write_text(
                ",".join(FREEZE.TRACE_COLUMNS)
                + "\n"
                + "".join(
                    f"{request},{latency},{gpu},{queue},0,0,0\n"
                    for request in range(5)
                ),
                encoding="utf-8",
            )
            previous = summary["isolated"][repeat - 1]
            start_ns = previous["measurement_start_monotonic_ns"]
            end_ns = previous["measurement_end_monotonic_ns"]
            elapsed = (end_ns - start_ns) / 1_000_000_000.0
            summary["isolated"][repeat - 1] = {
                "schema_version": 1,
                "model": "resnet50-v2",
                "role": "benchmark",
                "engine": str(engine.resolve()),
                "execution_environment": {
                    "pid": 1000 + repeat,
                    "cuda_visible_devices": summary["mig"]["critical_uuid"],
                    "mps_active_thread_percentage": 100,
                    "cpu_affinity": [12],
                },
                "gpu": {
                    "name": "NVIDIA Thor MIG 2g.0gb",
                    "multiprocessors": 12,
                },
                "config": {
                    "warmup": summary["config"]["warmup"],
                    "burst_size": summary["config"]["burst_size"],
                    "period_ms": summary["config"]["period_ms"],
                    "deadline_ms": 0.0,
                    "duration_seconds": 0.0,
                    "guard_ms": 0.0,
                    "gated_processes": 0,
                    "stopped_processes": 0,
                    "gate_mode": "stop",
                    "start_paused": True,
                    "include_transfers": True,
                    "priority": "high",
                    "stream_priority_value": -5,
                },
                **{
                    name: latency_summary(values)
                    for name, values in columns.items()
                },
                "completed_requests": 5,
                "throughput_per_second": 5 / elapsed,
                "measurement_start_monotonic_ns": start_ns,
                "measurement_end_monotonic_ns": end_ns,
                "elapsed_seconds": elapsed,
                "deadline_misses": 0,
                "deadline_miss_rate": None,
                "thermal_start": previous["thermal_start"],
                "thermal_start_telemetry": None,
                "thermal_start_stable": True,
                "thermal_start_attempts": [],
                "thermal_precondition": None,
                "thermal_start_qualification": None,
                "thermal_actual_start_qualification": None,
                "thermal_handoff": None,
                "thermal_precondition_label": (
                    f"pre-pre-calibration-r{repeat}-attempt-01"
                ),
                "measurement_release_monotonic_ns": start_ns - 100_000_000,
                "readiness_affinity": {
                    "pid": 1000 + repeat,
                    "expected_cpu": 12,
                    "tasks": [
                        {"tid": 1000 + repeat, "cpus": [12]},
                        {"tid": 2000 + repeat, "cpus": [12]},
                    ],
                },
            }
        pooled = [5.0] * 5 + [5.001] * 5
        summary["isolated_pooled_p99_ms"] = FREEZE.percentile(pooled, 0.99)
        summary["deadline_ms"] = summary["isolated_pooled_p99_ms"] * 1.1
        records = []
        raw_line = (
            "RAM 1/2MB CPU [50%@1000] "
            "soc012@74.8C tj@89.5C VIN 100000mW"
        )
        parsed = json.loads(json.dumps(parse_tegrastats_line(raw_line).to_dict()))
        for repeat, start_ns in ((1, 1_000_000_000), (2, 64_000_000_000)):
            base_label = f"pre-pre-calibration-r{repeat}"
            label = f"{base_label}-attempt-01"
            end_ns = start_ns + 62_000_000_000
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": start_ns - 200_000_000,
                    "name": "calibration_prepare",
                    "metadata": {"stage": "pre", "repeat": repeat},
                }
            )
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": start_ns - 100_000_000,
                    "name": "thermal_prepare",
                    "metadata": {"label": label},
                }
            )
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": start_ns,
                    "name": "thermal_start",
                    "metadata": {"label": label},
                }
            )
            block_samples = []
            for index in range(624):
                sample = {
                    "schema_version": 1,
                    "record_type": "sample",
                    "monotonic_ns": start_ns
                    + 50_000_000
                    + index * 100_000_000,
                    "raw": raw_line,
                    "mem_available_mb": 1000.0,
                    "parsed": parsed,
                    "collection_errors": [],
                }
                block_samples.append(sample)
                records.append(
                    sample
                )
            active_checks = []
            block_telemetry = (block_samples, [])
            for index, sample_ns in enumerate(
                (end_ns - 2_150_000_000, end_ns - 1_150_000_000, end_ns - 150_000_000)
            ):
                raw_endpoint = FREEZE.replay_raw_thermal_window(
                    block_telemetry,
                    reference_ns=sample_ns,
                    not_before_ns=start_ns,
                    window_seconds=60.0,
                    interval_ms=100.0,
                    required_fraction=0.8,
                    maximum_gap_ms=300.0,
                    stability_sensor="soc012",
                    safety_sensor="tj",
                    hard_limit_c=104.0,
                    end_inclusive=True,
                )
                check = {
                    "label": label,
                    "index": index,
                    "sample_monotonic_ns": sample_ns,
                    "passed": True,
                    "consecutive_passes": index + 1,
                    "window": raw_endpoint["stability_window"],
                }
                active_checks.append(check)
                records.append(
                    {
                        "schema_version": 1,
                        "record_type": "marker",
                        "monotonic_ns": sample_ns + 10_000_000,
                        "name": "thermal_active_stability_check",
                        "metadata": check,
                    }
                )
            summary["isolated_preconditions"][repeat - 1][
                "active_stability_checks"
            ] = active_checks
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": end_ns,
                    "name": "thermal_measurement_end",
                    "metadata": {
                        "label": label,
                        "boundary_sample_monotonic_ns": active_checks[-1][
                            "sample_monotonic_ns"
                        ],
                        "consecutive_passes": 3,
                        "window": active_checks[-1]["window"],
                    },
                }
            )
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": end_ns + 25_000_000,
                    "name": "thermal_end",
                    "metadata": {"label": label, "successful": True},
                }
            )
            qualification_sample_ns = end_ns + 50_000_000
            qualification_ns = end_ns + 75_000_000
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": qualification_ns,
                    "name": "thermal_start_qualification",
                    "metadata": {
                        "label": label,
                        "attempt": 1,
                        "boundary_monotonic_ns": end_ns,
                        "cleanup_end_monotonic_ns": end_ns + 25_000_000,
                        "sample_monotonic_ns": qualification_sample_ns,
                    },
                }
            )
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": qualification_ns + 10_000_000,
                    "name": "thermal_start_qualification_result",
                    "metadata": {
                        "stage": "pre",
                        "repeat": repeat,
                        "label": label,
                        "attempt": 1,
                        "qualification_monotonic_ns": qualification_ns,
                        "passed": True,
                        "failure_reason": None,
                    },
                }
            )
            calibration_start_ns = end_ns + 300_000_000
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": end_ns + 200_000_000,
                    "name": "calibration_start",
                    "metadata": {"stage": "pre", "repeat": repeat},
                }
            )
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": calibration_start_ns + 110_000_000,
                    "name": "thermal_actual_start_qualification_result",
                    "metadata": {
                        "stage": "pre",
                        "repeat": repeat,
                        "measurement_start_monotonic_ns": calibration_start_ns,
                        "sample_monotonic_ns": end_ns + 250_000_000,
                        "passed": True,
                        "failure_reason": None,
                    },
                }
            )
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": calibration_start_ns + 200_000_000,
                    "name": "calibration_measurement_window",
                    "metadata": {
                        "stage": "pre",
                        "repeat": repeat,
                        "measurement_start_monotonic_ns": calibration_start_ns,
                        "measurement_end_monotonic_ns": calibration_start_ns
                        + 100_000_000,
                    },
                }
            )
            records.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": calibration_start_ns + 300_000_000,
                    "name": "calibration_end",
                    "metadata": {"stage": "pre", "repeat": repeat},
                }
            )
        records.sort(key=lambda record: record["monotonic_ns"])
        (directory / "telemetry.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        telemetry = FREEZE.load_telemetry_jsonl(directory / "telemetry.jsonl")
        for repeat, precondition in enumerate(
            summary["isolated_preconditions"], start=1
        ):
            result = summary["isolated"][repeat - 1]
            precondition_start_ns = precondition[
                "measurement_start_monotonic_ns"
            ]
            precondition_end_ns = precondition[
                "measurement_end_monotonic_ns"
            ]
            duration_seconds = precondition["duration_seconds"]
            full_evidence = FREEZE.replay_raw_thermal_window(
                telemetry,
                reference_ns=precondition_end_ns,
                not_before_ns=precondition_start_ns,
                window_seconds=duration_seconds,
                interval_ms=100.0,
                required_fraction=0.8,
                maximum_gap_ms=300.0,
                stability_sensor="soc012",
                safety_sensor="tj",
                hard_limit_c=104.0,
            )
            last_evidence = FREEZE.replay_raw_thermal_window(
                telemetry,
                reference_ns=precondition["active_stability_checks"][-1][
                    "sample_monotonic_ns"
                ],
                not_before_ns=precondition_start_ns,
                window_seconds=60.0,
                interval_ms=100.0,
                required_fraction=0.8,
                maximum_gap_ms=300.0,
                stability_sensor="soc012",
                safety_sensor="tj",
                hard_limit_c=104.0,
                end_inclusive=True,
            )
            precondition["last_window"] = {
                field: last_evidence["stability_window"][field]
                for field in FREEZE.THERMAL_PRECONDITION_WINDOW_FIELDS
            }
            precondition["telemetry"] = self.stored_thermal_aggregate(
                full_evidence, telemetry
            )
            release_ns = result["measurement_release_monotonic_ns"]
            qualification_ns = precondition_end_ns + 75_000_000
            qualification_sample_ns = precondition_end_ns + 50_000_000
            release_evidence = FREEZE.replay_raw_thermal_window(
                telemetry,
                reference_ns=release_ns,
                not_before_ns=precondition_start_ns,
                window_seconds=60.0,
                interval_ms=100.0,
                required_fraction=0.8,
                maximum_gap_ms=300.0,
                stability_sensor="soc012",
                safety_sensor="tj",
                hard_limit_c=104.0,
            )
            qualification_telemetry = FREEZE.replay_point_telemetry_aggregate(
                telemetry,
                sample_ns=qualification_sample_ns,
                reference_ns=qualification_ns,
            )
            qualification = {
                "attempt": 1,
                "passed": True,
                "boundary": "thermal_measurement_end",
                "boundary_monotonic_ns": precondition_end_ns,
                "cleanup_end_monotonic_ns": precondition_end_ns + 25_000_000,
                "qualification_monotonic_ns": qualification_ns,
                "sample_monotonic_ns": qualification_sample_ns,
                "sample_age_ms": 25.0,
                "stability_sensor": "soc012",
                "stability_value_c": 74.8,
                "safety_sensor": "tj",
                "safety_value_c": 89.5,
                "target_c": 74.8,
                "tolerance_c": 1.0,
                "telemetry": qualification_telemetry,
                "failure_reason": None,
            }
            result["thermal_start"] = precondition["last_window"]
            result["thermal_start_telemetry"] = qualification["telemetry"]
            result["thermal_start_qualification"] = qualification
            actual_sample_ns = result["measurement_start_monotonic_ns"] - 50_000_000
            result["thermal_actual_start_qualification"] = {
                "passed": True,
                "measurement_start_monotonic_ns": result[
                    "measurement_start_monotonic_ns"
                ],
                "sample_monotonic_ns": actual_sample_ns,
                "sample_age_ms": 50.0,
                "stability_sensor": "soc012",
                "stability_value_c": 74.8,
                "safety_sensor": "tj",
                "safety_value_c": 89.5,
                "target_c": 74.8,
                "tolerance_c": 1.0,
                "telemetry": FREEZE.replay_point_telemetry_aggregate(
                    telemetry,
                    sample_ns=actual_sample_ns,
                    reference_ns=result["measurement_start_monotonic_ns"],
                ),
                "failure_reason": None,
            }
            result["thermal_precondition"] = precondition
            result["thermal_start_attempts"] = [
                {
                    "attempt": 1,
                    "thermal_precondition": precondition,
                    "qualification": qualification,
                    "qualification_result_marker_monotonic_ns": (
                        qualification_ns + 10_000_000
                    ),
                    "measured_process_states": {
                        str(result["execution_environment"]["pid"]): "T"
                    },
                }
            ]
            result["thermal_handoff"] = {
                "boundary": "thermal_measurement_end",
                "boundary_monotonic_ns": precondition_end_ns,
                "cleanup_end_monotonic_ns": precondition_end_ns + 25_000_000,
                "qualification_monotonic_ns": qualification_ns,
                "qualification_result_monotonic_ns": qualification_ns
                + 10_000_000,
                "measurement_release_monotonic_ns": release_ns,
                "measurement_start_monotonic_ns": result[
                    "measurement_start_monotonic_ns"
                ],
                "boundary_to_cleanup_end_ms": 25.0,
                "boundary_to_qualification_ms": 75.0,
                "boundary_to_qualification_result_ms": 85.0,
                "boundary_to_measurement_release_ms": (
                    release_ns - precondition_end_ns
                )
                / 1_000_000.0,
                "boundary_to_measurement_start_ms": (
                    result["measurement_start_monotonic_ns"]
                    - precondition_end_ns
                )
                / 1_000_000.0,
                "maximum_ms": FREEZE.THERMAL_HANDOFF_MAX_MS,
                "strictly_within_bound": True,
            }
        summary["config"]["thermal_lock_sha256"] = FREEZE.file_sha256(
            directory / "thermal-lock.json"
        )
        return summary

    def test_build_and_verify_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(
                FREEZE, "verify_thermal_lock"
            ) as verify_thermal:
                lock = FREEZE.build_lock(
                    summary,
                    summary_path,
                    guard_lock_path=directory_path / "guard-lock.json",
                    expected_blocks=2,
                    expected_samples_per_block=5,
                )
                FREEZE.verify_lock(lock)
                self.assertGreaterEqual(verify_thermal.call_count, 2)
                self.assertGreaterEqual(self.verify_guard_lock.call_count, 2)
        self.assertEqual(lock["isolated_samples"], 10)
        self.assertAlmostEqual(lock["deadline_ms"], 5.5011)
        self.assertEqual(
            lock["guard_lock_sha256"],
            summary["config"]["guard_lock_sha256"],
        )
        self.assertEqual(lock["profile_guard_ms"], self.guard_profile())
        self.assertEqual(lock["guard_profile_source"], "frozen-quota-aware-lock")
        self.assertEqual(lock["guard_lock_protocol"]["formal_period_ms"], 20.0)
        self.assertEqual(lock["guard_lock_estimator"]["quantile"], 0.999)
        self.assertEqual(lock["guard_profile_summary_sha256"], "1" * 64)
        self.assertEqual(lock["guard_telemetry_jsonl_sha256"], "2" * 64)
        self.assertEqual(lock["thermal_lock_schema_version"], 4)
        self.assertEqual(lock["thermal_target_source"], FREEZE.THERMAL_TARGET_SOURCE)
        self.assertEqual(lock["thermal_stability_sensor"], "soc012")
        self.assertEqual(lock["thermal_safety_sensor"], "tj")
        self.assertEqual(lock["thermal_handoff_max_ms"], 500.0)
        self.assertEqual(
            lock["thermal_handoff_boundary"], "thermal_measurement_end"
        )
        self.assertEqual(lock["thermal_qualification_max_attempts"], 3)
        self.assertEqual(lock["thermal_active_stable_endpoints"], 3)
        self.assertEqual(lock["thermal_active_stable_spacing_seconds"], 1.0)
        self.assertNotIn("thermal_qualification_dwell_seconds", lock)
        self.assertEqual(
            lock["thermal_handoff_rationale"],
            FREEZE.THERMAL_HANDOFF_RATIONALE,
        )
        self.assertEqual(
            lock["thermal_required_fields"],
            list(FREEZE.THERMAL_REQUIRED_FIELDS),
        )

    def test_old_or_tampered_thermal_v2_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            thermal_path = directory_path / "thermal-lock.json"
            original = json.loads(thermal_path.read_text(encoding="utf-8"))
            cases = {
                "old-schema": ("schema_version", 3),
                "target-source": ("target_source", "old-source"),
                "stability-sensor": ("stability_sensor", "tj"),
                "safety-sensor": ("safety_sensor", "soc012"),
                "handoff": ("thermal_handoff_max_ms", 300.0),
                "stale-boundary": (
                    "thermal_handoff_boundary",
                    "thermal_start_qualification",
                ),
                "rationale": ("thermal_handoff_rationale", "unbound"),
            }
            for name, (field, value) in cases.items():
                with self.subTest(name=name):
                    tampered = copy.deepcopy(original)
                    tampered[field] = value
                    thermal_path.write_text(json.dumps(tampered), encoding="utf-8")
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(
                            ValueError, "thermal lock is invalid"
                        ):
                            FREEZE.build_lock(
                                summary,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )
            thermal_path.write_text(json.dumps(original), encoding="utf-8")

    def test_thermal_config_sensor_scalar_and_fields_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            mutations = {
                "stability-sensor": ("thermal_stability_sensor", "tj"),
                "safety-sensor": ("thermal_safety_sensor", "soc012"),
                "handoff": ("thermal_handoff_max_ms", 300.0),
                "boundary": (
                    "thermal_handoff_boundary",
                    "thermal_start_qualification",
                ),
                "attempts": ("thermal_qualification_max_attempts", 2),
                "active-endpoints": ("thermal_active_stable_endpoints", 2),
                "active-spacing": (
                    "thermal_active_stable_spacing_seconds",
                    0.5,
                ),
                "old-dwell": ("thermal_qualification_dwell_seconds", 1.0),
                "preconditioning": (
                    "thermal_calibration_preconditioning",
                    "per-repeat-after-launch",
                ),
                "start-protocol": ("start_protocol", "unbounded"),
                "required-fields": (
                    "telemetry_required_fields",
                    [
                        field
                        for field in FREEZE.THERMAL_REQUIRED_FIELDS
                        if field != "temperature:soc012"
                    ],
                ),
            }
            for name, (field, value) in mutations.items():
                with self.subTest(name=name):
                    tampered = copy.deepcopy(summary)
                    tampered["config"][field] = value
                    with self.assertRaisesRegex(ValueError, "frozen design"):
                        FREEZE.build_lock(
                            tampered,
                            summary_path,
                            guard_lock_path=directory_path / "guard-lock.json",
                            expected_blocks=2,
                            expected_samples_per_block=5,
                        )

    def test_calibration_prepare_marker_is_required_and_ordered(self) -> None:
        for mutation, message in (
            ("missing", "incomplete or extra marker chain"),
            ("reordered", "overlap or reorder"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    directory_path = pathlib.Path(directory)
                    summary_path = directory_path / "summary.json"
                    summary = self.prepared_summary(directory_path)
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    telemetry_path = directory_path / "telemetry.jsonl"
                    records = [
                        json.loads(line)
                        for line in telemetry_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    prepare = next(
                        record
                        for record in records
                        if record.get("name") == "calibration_prepare"
                        and record["metadata"] == {"stage": "pre", "repeat": 1}
                    )
                    if mutation == "missing":
                        records.remove(prepare)
                    else:
                        prepare["monotonic_ns"] = 950_000_001
                    records.sort(key=lambda record: record["monotonic_ns"])
                    telemetry_path.write_text(
                        "".join(json.dumps(record) + "\n" for record in records),
                        encoding="utf-8",
                    )
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(ValueError, message):
                            FREEZE.build_lock(
                                summary,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )

    def test_readiness_and_nested_handoff_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            mutations = (
                (
                    ("readiness_affinity", "expected_cpu"),
                    11,
                    "readiness affinity differs",
                ),
                (
                    ("readiness_affinity", "tasks"),
                    [],
                    "has no tasks",
                ),
                (
                    ("thermal_handoff", "boundary_to_measurement_release_ms"),
                    101.0,
                    "handoff differs",
                ),
                (
                    ("thermal_handoff", "strictly_within_bound"),
                    False,
                    "invalid clocks",
                ),
            )
            for path, value, message in mutations:
                with self.subTest(path=path):
                    tampered = copy.deepcopy(summary)
                    tampered["isolated"][0][path[0]][path[1]] = value
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(ValueError, message):
                            FREEZE.build_lock(
                                tampered,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )

    def test_handoff_at_500_ms_boundary_is_rejected(self) -> None:
        end_ns = 1_000_000_000
        release_ns = end_ns + 100_000_000
        start_ns = end_ns + 500_000_000
        stored = {
            "boundary": "thermal_measurement_end",
            "boundary_monotonic_ns": end_ns,
            "cleanup_end_monotonic_ns": end_ns + 25_000_000,
            "qualification_monotonic_ns": end_ns + 50_000_000,
            "qualification_result_monotonic_ns": end_ns + 75_000_000,
            "measurement_release_monotonic_ns": release_ns,
            "measurement_start_monotonic_ns": start_ns,
            "boundary_to_cleanup_end_ms": 25.0,
            "boundary_to_qualification_ms": 50.0,
            "boundary_to_qualification_result_ms": 75.0,
            "boundary_to_measurement_release_ms": 100.0,
            "boundary_to_measurement_start_ms": 500.0,
            "maximum_ms": 500.0,
            "strictly_within_bound": True,
        }
        with self.assertRaisesRegex(ValueError, "strict bound"):
            FREEZE.replay_thermal_handoff(
                stored,
                boundary_ns=end_ns,
                cleanup_end_ns=end_ns + 25_000_000,
                qualification_ns=end_ns + 50_000_000,
                qualification_result_ns=end_ns + 75_000_000,
                release_ns=release_ns,
                measurement_start_ns=start_ns,
                maximum_ms=500.0,
            )

    def test_qualification_attempt_contract_tamper_is_rejected(self) -> None:
        mutations = (
            (
                lambda result: result["thermal_start_attempts"][0].__setitem__(
                    "measured_process_states", {}
                ),
                "remained paused",
            ),
            (
                lambda result: result["thermal_start_attempts"][0][
                    "qualification"
                ].__setitem__("dwell_seconds", 1.0),
                "invalid fields",
            ),
            (
                lambda result: result["thermal_start_attempts"][0].__setitem__(
                    "measurement_release_monotonic_ns", 1
                ),
                "attempt has invalid fields",
            ),
            (
                lambda result: result.__setitem__(
                    "thermal_start_attempts",
                    result["thermal_start_attempts"] * 4,
                ),
                "attempt count",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    directory_path = pathlib.Path(directory)
                    summary = self.prepared_summary(directory_path)
                    mutate(summary["isolated"][0])
                    summary_path = directory_path / "summary.json"
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(ValueError, message):
                            FREEZE.build_lock(
                                summary,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )

    def test_qualification_rejects_wrong_first_causal_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary = self.prepared_summary(directory_path)
            telemetry_path = directory_path / "telemetry.jsonl"
            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            ]
            qualification_marker = next(
                record
                for record in records
                if record.get("name") == "thermal_start_qualification"
                and record["metadata"].get("label")
                == "pre-pre-calibration-r1-attempt-01"
            )
            result_marker = next(
                record
                for record in records
                if record.get("name") == "thermal_start_qualification_result"
                and record["metadata"].get("repeat") == 1
            )
            qualification_marker["monotonic_ns"] = 63_175_000_000
            qualification_marker["metadata"]["sample_monotonic_ns"] = 63_150_000_000
            result_marker["monotonic_ns"] = 63_185_000_000
            result_marker["metadata"]["qualification_monotonic_ns"] = 63_175_000_000
            records.sort(key=lambda record: record["monotonic_ns"])
            telemetry_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = summary["isolated"][0]
            qualification = result["thermal_start_qualification"]
            qualification["sample_monotonic_ns"] = 63_150_000_000
            qualification["qualification_monotonic_ns"] = 63_175_000_000
            qualification["sample_age_ms"] = 25.0
            result["thermal_start_attempts"][0][
                "qualification_result_marker_monotonic_ns"
            ] = 63_185_000_000
            handoff = result["thermal_handoff"]
            handoff["qualification_monotonic_ns"] = 63_175_000_000
            handoff["qualification_result_monotonic_ns"] = 63_185_000_000
            handoff["boundary_to_qualification_ms"] = 175.0
            handoff["boundary_to_qualification_result_ms"] = 185.0
            summary_path = directory_path / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(ValueError, "first causal sample"):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=directory_path / "guard-lock.json",
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )

    def test_active_endpoint_sample_omission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary = self.prepared_summary(directory_path)
            telemetry_path = directory_path / "telemetry.jsonl"
            omitted_ns = summary["isolated_preconditions"][0][
                "active_stability_checks"
            ][0]["sample_monotonic_ns"]
            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            ]
            records = [
                record
                for record in records
                if not (
                    record.get("record_type") == "sample"
                    and record["monotonic_ns"] == omitted_ns
                )
            ]
            telemetry_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary_path = directory_path / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(
                    ValueError, "counts differ|non-causal endpoint"
                ):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=directory_path / "guard-lock.json",
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )

    def test_postcleanup_cooling_outside_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary = self.prepared_summary(directory_path)
            telemetry_path = directory_path / "telemetry.jsonl"
            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            ]
            sample = next(
                record
                for record in records
                if record.get("record_type") == "sample"
                and record["monotonic_ns"] == 63_050_000_000
            )
            sample["raw"] = sample["raw"].replace("soc012@74.8C", "soc012@72.0C")
            sample["parsed"] = parse_tegrastats_line(sample["raw"]).to_dict()
            telemetry_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary["isolated"][0]["thermal_start_qualification"][
                "stability_value_c"
            ] = 72.0
            summary_path = directory_path / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(ValueError, "differs from raw telemetry"):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=directory_path / "guard-lock.json",
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )

    def test_guard_hash_and_nested_profile_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            wrong_hash = copy.deepcopy(summary)
            wrong_hash["config"]["guard_lock_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "guard lock hash"):
                FREEZE.build_lock(
                    wrong_hash,
                    summary_path,
                    guard_lock_path=directory_path / "guard-lock.json",
                    expected_blocks=2,
                    expected_samples_per_block=5,
                )

            wrong_profile = copy.deepcopy(summary)
            wrong_profile["config"]["profile_guard_ms"]["resident-1g"]["25"][
                "audio"
            ] += 0.1
            with self.assertRaisesRegex(ValueError, "guard profile differs"):
                FREEZE.build_lock(
                    wrong_profile,
                    summary_path,
                    guard_lock_path=directory_path / "guard-lock.json",
                    expected_blocks=2,
                    expected_samples_per_block=5,
                )

    def test_guard_thermal_lock_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            guard_path = directory_path / "guard-lock.json"
            guard = json.loads(guard_path.read_text(encoding="utf-8"))
            guard["thermal_lock"]["sha256"] = "0" * 64
            guard_path.write_text(json.dumps(guard), encoding="utf-8")
            summary["config"]["guard_lock_sha256"] = FREEZE.file_sha256(guard_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(ValueError, "different thermal locks"):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=guard_path,
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )

    def test_guard_platform_and_artifact_mismatches_are_rejected(self) -> None:
        mutations = (
            (
                lambda lock: lock.__setitem__("schema_version", 1),
                "invalid schema or kind",
            ),
            (
                lambda lock: lock.__setitem__("hardware", {"gpu": "other"}),
                "hardware differs",
            ),
            (
                lambda lock: lock["mig"].__setitem__("big_uuid", "other"),
                "MIG mapping differs",
            ),
            (
                lambda lock: lock.__setitem__(
                    "cpu_affinity",
                    {**lock["cpu_affinity"], "critical": [11]},
                ),
                "CPU affinity differs",
            ),
            (
                lambda lock: lock["artifacts"]["benchmark"].__setitem__(
                    "sha256", "0" * 64
                ),
                "benchmark differs",
            ),
            (
                lambda lock: lock["artifacts"]["producer"].__setitem__(
                    "sha256", "0" * 64
                ),
                "implementation differs",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    directory_path = pathlib.Path(directory)
                    summary_path = directory_path / "summary.json"
                    summary = self.prepared_summary(directory_path)
                    guard_path = directory_path / "guard-lock.json"
                    guard = json.loads(guard_path.read_text(encoding="utf-8"))
                    mutate(guard)
                    guard_path.write_text(json.dumps(guard), encoding="utf-8")
                    summary["config"]["guard_lock_sha256"] = FREEZE.file_sha256(
                        guard_path
                    )
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        FREEZE.build_lock(
                            summary,
                            summary_path,
                            guard_lock_path=guard_path,
                            expected_blocks=2,
                            expected_samples_per_block=5,
                        )

    def test_incomplete_calibration_is_rejected(self) -> None:
        summary = self.summary()
        summary["isolated_pooled_samples"] = 9
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            self.write_thermal_lock(directory_path, summary)
            self.write_guard_lock(directory_path, summary)
            raw = directory_path / "raw"
            raw.mkdir()
            for repeat in (1, 2):
                (raw / f"isolated-pre-r{repeat}.csv").write_text(
                    "request,release_to_completion_ms\n"
                    + "".join(f"{request},5.0\n" for request in range(5)),
                    encoding="utf-8",
                )
            summary["config"]["thermal_lock_sha256"] = FREEZE.file_sha256(
                directory_path / "thermal-lock.json"
            )
            path = directory_path / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(ValueError):
                FREEZE.build_lock(
                    summary,
                    path,
                    guard_lock_path=directory_path / "guard-lock.json",
                    expected_blocks=2,
                    expected_samples_per_block=5,
                )

    def test_modified_deadline_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                lock = FREEZE.build_lock(
                    summary,
                    summary_path,
                    guard_lock_path=directory_path / "guard-lock.json",
                    expected_blocks=2,
                    expected_samples_per_block=5,
                )
                lock["deadline_ms"] = 999.0
                with self.assertRaises(ValueError):
                    FREEZE.verify_lock(lock)

    def test_modified_guard_provenance_in_deadline_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                lock = FREEZE.build_lock(
                    summary,
                    summary_path,
                    guard_lock_path=directory_path / "guard-lock.json",
                    expected_blocks=2,
                    expected_samples_per_block=5,
                )
                lock["profile_guard_ms"]["resident-1g"]["25"]["audio"] += 0.1
                with self.assertRaisesRegex(ValueError, "do not match"):
                    FREEZE.verify_lock(lock)

    def test_calibration_result_provenance_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            def replace_path(root, path, value) -> None:
                current = root
                for component in path[:-1]:
                    current = current[component]
                current[path[-1]] = value

            mutations = (
                (("isolated", 0, "schema_version"), 2, "ResNet50 calibration"),
                (("isolated", 0, "model"), "resnet18", "ResNet50 calibration"),
                (("isolated", 0, "role"), "pressure", "ResNet50 calibration"),
                (
                    ("isolated", 0, "engine"),
                    str(directory_path / "wrong.engine"),
                    "critical engine path",
                ),
                (
                    (
                        "isolated",
                        0,
                        "execution_environment",
                        "cuda_visible_devices",
                    ),
                    "wrong-MIG",
                    "frozen execution environment",
                ),
                (
                    (
                        "isolated",
                        0,
                        "execution_environment",
                        "mps_active_thread_percentage",
                    ),
                    50,
                    "frozen execution environment",
                ),
                (
                    ("isolated", 0, "execution_environment", "cpu_affinity"),
                    [11],
                    "frozen execution environment",
                ),
                (
                    ("isolated", 0, "execution_environment", "cpu_affinity"),
                    [12.0],
                    "must be an integer",
                ),
                (
                    ("isolated", 0, "gpu", "multiprocessors"),
                    11,
                    "frozen 2g MIG width",
                ),
                (
                    ("isolated", 0, "gpu", "multiprocessors"),
                    12.0,
                    "must be an integer",
                ),
                (
                    ("isolated", 0, "config", "warmup"),
                    99,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "burst_size"),
                    4,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "period_ms"),
                    21.0,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "deadline_ms"),
                    1.0,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "duration_seconds"),
                    1.0,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "guard_ms"),
                    1.0,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "gated_processes"),
                    1,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "stopped_processes"),
                    1,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "gate_mode"),
                    "cooperative",
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "start_paused"),
                    False,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "include_transfers"),
                    False,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "priority"),
                    "default",
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "config", "stream_priority_value"),
                    0,
                    "frozen protocol",
                ),
                (
                    ("isolated", 0, "completed_requests"),
                    4,
                    "completed_requests",
                ),
                (
                    ("isolated", 0, "elapsed_seconds"),
                    0.2,
                    "differs from its clocks",
                ),
                (
                    ("isolated", 0, "throughput_per_second"),
                    49.0,
                    "differs from its clocks",
                ),
                (
                    ("isolated", 0, "deadline_misses"),
                    1,
                    "deadline-disabled metrics",
                ),
                (
                    ("isolated", 0, "deadline_miss_rate"),
                    0.0,
                    "deadline-disabled metrics",
                ),
            )
            for path, value, message in mutations:
                with self.subTest(path=path):
                    tampered = copy.deepcopy(summary)
                    replace_path(tampered, path, value)
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(ValueError, message):
                            FREEZE.build_lock(
                                tampered,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )

    def test_all_latency_summaries_are_replayed_from_raw_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            for summary_name in FREEZE.TRACE_SUMMARIES:
                with self.subTest(summary_name=summary_name):
                    tampered = copy.deepcopy(summary)
                    tampered["isolated"][0][summary_name]["p99_ms"] += 1.0
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(ValueError, "differs from the raw trace"):
                            FREEZE.build_lock(
                                tampered,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )

    def test_engine_artifact_bytes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            pathlib.Path(summary["isolated"][0]["engine"]).write_bytes(
                b"tampered-engine"
            )
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(ValueError, "differs from its artifact"):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=directory_path / "guard-lock.json",
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )

    def test_trace_schema_and_all_latency_columns_are_required(self) -> None:
        for mutation, message in (
            ("header", "trace header"),
            ("extra", "trace row"),
            ("value", "raw trace"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    directory_path = pathlib.Path(directory)
                    summary_path = directory_path / "summary.json"
                    summary = self.prepared_summary(directory_path)
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    trace_path = directory_path / "raw" / "isolated-pre-r1.csv"
                    lines = trace_path.read_text(encoding="utf-8").splitlines()
                    if mutation == "header":
                        lines[0] = lines[0].replace(",resume_ms", "")
                        lines = [line.rsplit(",", 1)[0] for line in lines]
                    elif mutation == "extra":
                        lines[1] += ",123"
                    else:
                        row = lines[1].split(",")
                        row[2] = "0.9"
                        lines[1] = ",".join(row)
                    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(ValueError, message):
                            FREEZE.build_lock(
                                summary,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )

    def test_measurement_clocks_are_bound_to_raw_marker_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            result = summary["isolated"][0]
            result["measurement_start_monotonic_ns"] += 1
            result["measurement_end_monotonic_ns"] += 1
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(
                    ValueError, "calibration_measurement_window marker"
                ):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=directory_path / "guard-lock.json",
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )

    def test_raw_parsed_temperature_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            telemetry_path = directory_path / "telemetry.jsonl"
            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            ]
            sample = next(
                record for record in records if record["record_type"] == "sample"
            )
            sample["raw"] = sample["raw"].replace("tj@89.5C", "tj@99C")
            sample["parsed"]["temperatures_c"]["tj"] = 50.0
            telemetry_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(ValueError, "differs from raw"):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=directory_path / "guard-lock.json",
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )

    def test_raw_soc012_precondition_slope_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            telemetry_path = directory_path / "telemetry.jsonl"
            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            ]
            for record in records:
                timestamp = record.get("monotonic_ns", -1)
                if record.get("record_type") != "sample" or not (
                    1_000_000_000 <= timestamp < 63_000_000_000
                ):
                    continue
                elapsed_seconds = (timestamp - 1_000_000_000) / 1e9
                temperature = 74.8 + 0.3 * elapsed_seconds / 60.0
                record["raw"] = record["raw"].replace(
                    "soc012@74.8C", f"soc012@{temperature:.6f}C"
                )
                record["parsed"] = parse_tegrastats_line(record["raw"]).to_dict()
            telemetry_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            telemetry = FREEZE.load_telemetry_jsonl(telemetry_path)
            precondition = summary["isolated_preconditions"][0]
            full = FREEZE.replay_raw_thermal_window(
                telemetry,
                reference_ns=63_000_000_000,
                not_before_ns=1_000_000_000,
                window_seconds=62.0,
                interval_ms=100.0,
                required_fraction=0.8,
                maximum_gap_ms=300.0,
                stability_sensor="soc012",
                safety_sensor="tj",
                hard_limit_c=104.0,
            )
            precondition["last_window"] = {
                field: full["stability_window"][field]
                for field in FREEZE.THERMAL_PRECONDITION_WINDOW_FIELDS
            }
            precondition["telemetry"] = self.stored_thermal_aggregate(
                full, telemetry
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(
                    ValueError, "active stability|soc012 envelope|telemetry interval"
                ):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=directory_path / "guard-lock.json",
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )

    def test_exact_measurement_start_soc012_and_tj_are_replayed(self) -> None:
        for sensor, value, message in (
            ("soc012", 80.0, "measurement start violates"),
            ("tj", 104.0, "TJ safety window reached"),
        ):
            with self.subTest(sensor=sensor):
                with tempfile.TemporaryDirectory() as directory:
                    directory_path = pathlib.Path(directory)
                    summary_path = directory_path / "summary.json"
                    summary = self.prepared_summary(directory_path)
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    telemetry_path = directory_path / "telemetry.jsonl"
                    records = [
                        json.loads(line)
                        for line in telemetry_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    raw = (
                        "RAM 1/2MB CPU [50%@1000] "
                        f"soc012@{80.0 if sensor == 'soc012' else 74.8}C "
                        f"tj@{104.0 if sensor == 'tj' else 89.5}C "
                        "VIN 100000mW"
                    )
                    selected = next(
                        record
                        for record in records
                        if record.get("record_type") == "sample"
                        and record["monotonic_ns"] == 63_250_000_000
                    )
                    selected["raw"] = raw
                    selected["parsed"] = parse_tegrastats_line(raw).to_dict()
                    records.sort(key=lambda record: record["monotonic_ns"])
                    telemetry_path.write_text(
                        "".join(json.dumps(record) + "\n" for record in records),
                        encoding="utf-8",
                    )
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(
                            ValueError, "actual-start|" + message
                        ):
                            FREEZE.build_lock(
                                summary,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )

    def test_tj_missing_and_measurement_start_gap_are_rejected(self) -> None:
        for mutation, message in (
            ("missing-tj", "required-field coverage"),
            ("start-gap", "incomplete or stale"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    directory_path = pathlib.Path(directory)
                    summary_path = directory_path / "summary.json"
                    summary = self.prepared_summary(directory_path)
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    telemetry_path = directory_path / "telemetry.jsonl"
                    records = [
                        json.loads(line)
                        for line in telemetry_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    sample = next(
                        record
                        for record in records
                        if record.get("record_type") == "sample"
                        and record["monotonic_ns"] == 60_950_000_000
                    )
                    if mutation == "missing-tj":
                        sample["raw"] = sample["raw"].replace("tj@89.5C ", "")
                        sample["parsed"] = parse_tegrastats_line(
                            sample["raw"]
                        ).to_dict()
                    else:
                        records[:] = [
                            record
                            for record in records
                            if record.get("monotonic_ns")
                            not in {
                                61_850_000_000,
                                61_950_000_000,
                                62_050_000_000,
                            }
                        ]
                    telemetry_path.write_text(
                        "".join(json.dumps(record) + "\n" for record in records),
                        encoding="utf-8",
                    )
                    with mock.patch.object(FREEZE, "verify_thermal_lock"):
                        with self.assertRaisesRegex(ValueError, message):
                            FREEZE.build_lock(
                                summary,
                                summary_path,
                                guard_lock_path=directory_path / "guard-lock.json",
                                expected_blocks=2,
                                expected_samples_per_block=5,
                            )

    def test_repeat_marker_chains_must_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            summary_path = directory_path / "summary.json"
            summary = self.prepared_summary(directory_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            telemetry_path = directory_path / "telemetry.jsonl"
            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            ]
            marker = next(
                record
                for record in records
                if record["record_type"] == "marker"
                and record["name"] == "calibration_prepare"
                and record["metadata"] == {"stage": "pre", "repeat": 2}
            )
            marker["monotonic_ns"] = 63_550_000_000
            records.sort(key=lambda record: record["monotonic_ns"])
            telemetry_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with mock.patch.object(FREEZE, "verify_thermal_lock"):
                with self.assertRaisesRegex(ValueError, "overlap or reorder"):
                    FREEZE.build_lock(
                        summary,
                        summary_path,
                        guard_lock_path=directory_path / "guard-lock.json",
                        expected_blocks=2,
                        expected_samples_per_block=5,
                    )


if __name__ == "__main__":
    unittest.main()
