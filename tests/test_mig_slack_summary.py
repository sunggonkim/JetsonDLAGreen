#!/usr/bin/env python3
import copy
import csv
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "summarize_mig_slack_governor",
    ROOT / "analysis" / "summarize_mig_slack_governor.py",
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)

MIG_FIXTURE = {
    "critical_uuid": "MIG-GPU-big",
    "resident_uuid": "MIG-GPU-small",
}

GUARD_PROFILE_FIXTURE = {
    "resident-1g": {
        "25": {"language": 1.0, "audio": 1.2},
        "50": {"language": 1.2, "audio": 1.5},
        "100": {"language": 1.5, "audio": 2.0},
    },
    "borrower-2g": {
        "100": {"language": 1.4, "audio": 1.8},
    },
}


def guard_profile_fixture() -> dict[str, dict[str, dict[str, float]]]:
    return copy.deepcopy(GUARD_PROFILE_FIXTURE)


def guard_binding_fixture() -> tuple[
    dict[str, object],
    str,
    dict[str, object],
    dict[str, object],
    str,
]:
    hashes = iter(f"{index:064x}" for index in range(1, 100))
    thermal_sha256 = next(hashes)
    guard_sha256 = next(hashes)
    benchmark_sha256 = next(hashes)
    hardware = {"platform_sha256": next(hashes)}
    implementation_keys = {
        "producer": "runtime/profile_p9_guard.py",
        "freezer": "analysis/freeze_p9_guard.py",
        "telemetry_runtime": "runtime/tegrastats_telemetry.py",
        "governor_runtime": "runtime/mig_slack_governor.py",
        "guard_runner": "scripts/run_p9_guard_calibration.sh",
        "formal_runner": "scripts/run_p9_mig_slack_governor.sh",
        "mig_configurator": "scripts/configure_thor_mig.sh",
        "benchmark_source": "benchmarks/trt_inference.cpp",
    }
    implementation_sha256 = {
        path: next(hashes) for path in implementation_keys.values()
    }
    engine_keys = {
        "engine:critical:2g:resnet50-v2": "critical-2g-resnet50-v2"
    }
    for placement, quotas in (
        ("resident-1g", (25, 50, 100)),
        ("borrower-2g", (100,)),
    ):
        for quota in quotas:
            for modality, model in SUMMARY.MODEL_BY_MODALITY.items():
                engine_keys[f"engine:{placement}:q{quota}:{modality}"] = (
                    f"{placement}-q{quota}-{model}"
                )
    engines_sha256 = {key: next(hashes) for key in engine_keys.values()}
    artifacts = {
        "benchmark": {"path": "/benchmark", "sha256": benchmark_sha256},
    }
    artifacts.update(
        {
            guard_key: {
                "path": f"/{guard_key}",
                "sha256": implementation_sha256[implementation_key],
            }
            for guard_key, implementation_key in implementation_keys.items()
        }
    )
    artifacts.update(
        {
            guard_key: {
                "path": f"/{guard_key}",
                "sha256": engines_sha256[calibration_key],
            }
            for guard_key, calibration_key in engine_keys.items()
        }
    )
    guard_lock = {
        "schema_version": 3,
        "protocol": SUMMARY.expected_guard_protocol(),
        "guards": {
            placement: {
                quota: {
                    modality: {"guard_ms": guard_ms}
                    for modality, guard_ms in modalities.items()
                }
                for quota, modalities in quotas.items()
            }
            for placement, quotas in GUARD_PROFILE_FIXTURE.items()
        },
        "thermal_lock": {"path": "/thermal-lock.json", "sha256": thermal_sha256},
        "hardware": hardware,
        "mig": {
            "big_uuid": MIG_FIXTURE["critical_uuid"],
            "small_uuid": MIG_FIXTURE["resident_uuid"],
        },
        "cpu_affinity": copy.deepcopy(SUMMARY.FORMAL_CPU_AFFINITY),
        "producer_cpu_affinity": [13],
        "artifacts": artifacts,
    }
    deadline_lock = {
        "thermal_lock_sha256": thermal_sha256,
        "guard_lock_sha256": guard_sha256,
        "calibration_hardware": hardware,
        "calibration_mig": copy.deepcopy(MIG_FIXTURE),
        "calibration_artifacts": {
            "benchmark_sha256": benchmark_sha256,
            "engines_sha256": engines_sha256,
            "implementation_sha256": implementation_sha256,
        },
    }
    thermal_lock = {"pilot_hardware": hardware}
    return guard_lock, guard_sha256, deadline_lock, thermal_lock, thermal_sha256


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": SUMMARY.percentile(values, 0.50),
        "p95_ms": SUMMARY.percentile(values, 0.95),
        "p99_ms": SUMMARY.percentile(values, 0.99),
        "p999_ms": SUMMARY.percentile(values, 0.999),
        "max_ms": max(values),
    }


def write_trace(path: pathlib.Path, rows: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "request",
                "release_to_completion_ms",
                "gpu_service_ms",
                "queue_delay_ms",
                "gate_overhead_ms",
                "drain_ms",
                "resume_ms",
            )
        )
        for request, (latency, gpu, gate) in enumerate(rows):
            writer.writerow((request, latency, gpu, 0.0, gate, 0.0, 0.0))


def calibration_result(values: list[float]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "benchmark",
        "completed_requests": len(values),
        "release_to_completion": {
            "count": len(values),
            "p99_ms": SUMMARY.percentile(values, 0.99),
        },
    }


def policy_fixture(
    directory: pathlib.Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, str],
    dict[str, object],
]:
    rows = [(1.0, 0.1, 0.2), (2.0, 0.1, 0.2), (3.0, 0.1, 0.2), (4.0, 0.1, 0.2)]
    write_trace(directory / "raw" / "static-mig-e0.csv", rows)
    engine_root = directory / "engines"
    critical_engine = engine_root / "mig-2g" / "resnet50-v2.engine"
    resident_engine = (
        engine_root / "mig-1g-q100" / "distilbert-sst2.engine"
    )
    critical_engine.parent.mkdir(parents=True, exist_ok=True)
    resident_engine.parent.mkdir(parents=True, exist_ok=True)
    critical_engine.write_bytes(b"critical-engine")
    resident_engine.write_bytes(b"resident-engine")
    artifacts: dict[str, object] = {
        "engines_sha256": {
            "critical-2g-resnet50-v2": SUMMARY.file_sha256(critical_engine),
            "resident-1g-q100-distilbert-sst2": SUMMARY.file_sha256(
                resident_engine
            ),
        }
    }
    worker_values = [1.0] * 10
    worker = {
        "schema_version": 1,
        "role": "pressure",
        "model": "distilbert-sst2",
        "engine": str(resident_engine),
        "execution_environment": {
            "pid": 1000,
            "cuda_visible_devices": MIG_FIXTURE["resident_uuid"],
            "mps_active_thread_percentage": 100,
            "cpu_affinity": [0],
        },
        "gpu": {"name": "NVIDIA Thor MIG 1g.0gb", "multiprocessors": 8},
        "config": {
            "warmup": 100,
            "burst_size": 1,
            "period_ms": 0,
            "deadline_ms": 0,
            "duration_seconds": 3600,
            "guard_ms": 0,
            "gated_processes": 0,
            "stopped_processes": 0,
            "gate_mode": "stop",
            "start_paused": True,
            "include_transfers": True,
            "priority": "default",
            "stream_priority_value": 0,
        },
        "tenant_id": 0,
        "modality": "language",
        "placement": "resident-1g",
        "quota_percent": 100,
        "completed_requests": 10,
        "measurement_start_monotonic_ns": 1_000_000_000,
        "measurement_end_monotonic_ns": 3_000_000_000,
        "elapsed_seconds": 2.0,
        "throughput_per_second": 5.0,
        "release_to_completion": latency_summary(worker_values),
        "gpu_service": latency_summary(worker_values),
        "queue_delay": latency_summary(worker_values),
        "gate_overhead": latency_summary(worker_values),
        "drain": latency_summary(worker_values),
        "resume": latency_summary(worker_values),
        "deadline_misses": 0,
        "deadline_miss_rate": None,
    }
    action = {
        "tenant_id": 0,
        "modality": "language",
        "placement": "resident-1g",
        "quota_percent": 100,
    }
    epoch = {
        "epoch": 0,
        "offered_modalities": ["language"],
        "offered_tenants": 1,
        "state_before": {
            "resident_admission_limit": 6,
            "resident_quota_index": 2,
            "borrower_limit": 6,
            "guard_adjustment_ms": 0.0,
            "safe_epochs": 0,
            "resident_quota_percent": 100,
        },
        "state_after": {
            "resident_admission_limit": 6,
            "resident_quota_index": 2,
            "borrower_limit": 6,
            "guard_adjustment_ms": 0.0,
            "safe_epochs": 0,
            "resident_quota_percent": 100,
        },
        "controller_action": "not-applicable",
        "guard_ms": 0.0,
        "gate_scope": [],
        "gated_workers": 0,
        "resident_actions": [action],
        "borrower_actions": [],
        "resident_workers": 1,
        "borrower_workers": 0,
        "active_workers": 1,
        "rejected_tenants": 0,
        "critical_p99_ms": 3.97,
        "deadline_misses": 1,
        "deadline_miss_rate": 0.25,
        "queue_delay_p99_ms": 0.0,
        "gate_overhead_mean_ms": 0.2,
        "drain_p99_ms": 0.0,
        "drain_max_ms": 0.0,
        "resume_p99_ms": 0.0,
        "guard_utilization": 0.0,
        "drain_near_overrun": False,
        "thermal_high": False,
        "critical_p50_ms": 2.5,
        "critical_gpu_duty_cycle": 0.0002,
        "critical_p999_ms": 3.997,
        "critical_max_ms": 4.0,
        "violated": True,
        "resident_completed": 10,
        "borrower_completed": 0,
        "pressure_completed": 10,
        "resident_goodput_per_second": 5.0,
        "borrower_goodput_per_second": 0.0,
        "pressure_goodput_per_second": 5.0,
        "completed_by_modality": {"language": 10, "audio": 0},
        "goodput_by_modality": {"language": 5.0, "audio": 0.0},
        "completed_by_tenant": {"0": 10},
        "goodput_by_tenant": {"0": 5.0},
        "workers": [worker],
        "readiness_affinity": [
            {
                "role": "pressure",
                "tenant_id": 0,
                "pid": 1000,
                "expected_cpu": 0,
                "tasks": [{"tid": 1000, "cpus": [0]}],
            },
            {
                "role": "critical",
                "pid": 999,
                "expected_cpu": 12,
                "tasks": [{"tid": 999, "cpus": [12]}],
            },
        ],
        "telemetry": {"health": {"healthy": True}},
        "telemetry_unhealthy": False,
        "measurement_seconds": 2.0,
        "measurement_release_monotonic_ns": 900_000_000,
        "measurement_start_monotonic_ns": 1_000_000_000,
        "measurement_end_monotonic_ns": 3_000_000_000,
        "result_collected_monotonic_ns": 3_100_000_000,
        "cleanup_end_monotonic_ns": 3_200_000_000,
        "worker_window_seconds": 2.0,
        "worker_window_spread_seconds": 0.0,
        "wall_elapsed_seconds": 3.0,
    }
    latencies = [row[0] for row in rows]
    gpu_values = [row[1] for row in rows]
    queue_values = [0.0] * len(rows)
    gate_values = [row[2] for row in rows]
    zero_values = [0.0] * len(rows)
    epoch["critical"] = {
        "schema_version": 1,
        "model": "resnet50-v2",
        "role": "benchmark",
        "engine": str(critical_engine),
        "execution_environment": {
            "pid": 999,
            "cuda_visible_devices": MIG_FIXTURE["critical_uuid"],
            "mps_active_thread_percentage": 100,
            "cpu_affinity": [12],
        },
        "gpu": {"name": "NVIDIA Thor MIG 2g.0gb", "multiprocessors": 12},
        "config": {
            "warmup": 100,
            "burst_size": 2,
            "period_ms": 20.0,
            "deadline_ms": 3.5,
            "duration_seconds": 0.0,
            "guard_ms": 0.0,
            "gated_processes": 0,
            "stopped_processes": 1,
            "gate_mode": "stop",
            "start_paused": True,
            "include_transfers": True,
            "priority": "high",
            "stream_priority_value": -5,
        },
        "release_to_completion": latency_summary(latencies),
        "gpu_service": latency_summary(gpu_values),
        "queue_delay": latency_summary(queue_values),
        "gate_overhead": latency_summary(gate_values),
        "drain": latency_summary(zero_values),
        "resume": latency_summary(zero_values),
        "completed_requests": 4,
        "throughput_per_second": 2.0,
        "measurement_start_monotonic_ns": 1_000_000_000,
        "measurement_end_monotonic_ns": 3_000_000_000,
        "elapsed_seconds": 2.0,
        "deadline_misses": 1,
        "deadline_miss_rate": 0.25,
    }
    policy = {
        "name": "static-mig",
        "critical_requests": 4,
        "deadline_misses": 1,
        "deadline_miss_rate": 0.25,
        "violation_epoch_rate": 1.0,
        "critical_p99_ms_max": 3.97,
        "resident_completed": 10,
        "borrower_completed": 0,
        "pressure_completed": 10,
        "resident_goodput_per_second": 5.0,
        "borrower_goodput_per_second": 0.0,
        "pressure_goodput_per_second": 5.0,
        "goodput_by_modality": {"language": 5.0, "audio": 0.0},
        "rejected_tenants": 0,
        "telemetry_unhealthy_epochs": 0,
        "gate_overhead_mean_ms": 0.2,
        "critical_gpu_duty_cycle_mean": 0.0002,
        "measurement_seconds": 2.0,
        "worker_window_seconds": 2.0,
        "wall_elapsed_seconds": 3.0,
        "epochs": [epoch],
    }
    config = {
        "epochs": 1,
        "samples_per_epoch": 4,
        "warmup": 100,
        "burst_size": 2,
        "period_ms": 20.0,
        "dmr_target": 0.3,
        "borrower_quota": 100,
        "cpu_affinity": {"pressure": [0], "critical": [12]},
        "profile_guard_ms": guard_profile_fixture(),
        "guard_profile_source": "frozen-quota-aware-lock",
    }
    return policy, config, copy.deepcopy(MIG_FIXTURE), artifacts


def isolated_fixture(directory: pathlib.Path) -> dict[str, object]:
    pre_blocks = ([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1])
    post_blocks = ([1.0, 2.0, 3.0, 4.02], [1.1, 2.1, 3.1, 4.08])
    for stage, blocks in (("pre", pre_blocks), ("post", post_blocks)):
        for repeat, values in enumerate(blocks, start=1):
            write_trace(
                directory / "raw" / f"isolated-{stage}-r{repeat}.csv",
                [(value, 0.1, 0.0) for value in values],
            )
    pre_values = [value for block in pre_blocks for value in block]
    post_values = [value for block in post_blocks for value in block]
    pre_p99 = SUMMARY.percentile(pre_values, 0.99)
    post_p99 = SUMMARY.percentile(post_values, 0.99)
    reference = pre_p99
    pre_reference = abs(pre_p99 - reference) / reference
    post_reference = abs(post_p99 - reference) / reference
    pre_post = abs(post_p99 - pre_p99) / pre_p99
    return {
        "config": {
            "calibration_repeats": 2,
            "samples_per_epoch": 4,
            "slo_factor": 1.1,
            "max_isolated_drift_fraction": 0.05,
            "thermal_window_seconds": 60.0,
            "thermal_timeout_seconds": 900.0,
            "thermal_stability_checkpoint_seconds": 30.0,
            "thermal_stability_checkpoint_max_lateness_seconds": 1.0,
            "thermal_required_stable_checkpoints": 3,
            "tegrastats_requested_interval_ms": 75.0,
            "telemetry_interval_ms": 100,
            "telemetry_required_fraction": 0.8,
            "telemetry_stale_after_ms": 300,
        },
        "deadline_ms": reference * 1.1,
        "isolated": [calibration_result(list(values)) for values in pre_blocks],
        "isolated_post": [
            calibration_result(list(values)) for values in post_blocks
        ],
        "isolated_p99_ms": [
            SUMMARY.percentile(list(values), 0.99) for values in pre_blocks
        ],
        "isolated_pooled_p99_ms": pre_p99,
        "isolated_pooled_samples": 8,
        "isolated_post_pooled_p99_ms": post_p99,
        "isolated_post_pooled_samples": 8,
        "isolated_reference_p99_ms": reference,
        "isolated_pre_reference_drift_fraction": pre_reference,
        "isolated_post_reference_drift_fraction": post_reference,
        "isolated_drift_fraction": pre_post,
        "isolated_drift_valid": True,
    }


def thermal_lock_fixture() -> dict[str, object]:
    return {
        "schema_version": 4,
        "stability_sensor": "soc012",
        "safety_sensor": "tj",
        "thermal_handoff_max_ms": 500.0,
        "thermal_handoff_boundary": "thermal_measurement_end",
        "thermal_qualification_max_attempts": 3,
        "thermal_active_stable_endpoints": 3,
        "thermal_active_stable_spacing_seconds": 1.0,
        "target_c": 90.0,
        "tolerance_c": 1.0,
        "stability_window_seconds": 60.0,
        "maximum_slope_c_per_minute": 0.2,
        "hard_limit_c": 104.0,
        "telemetry_interval_ms": 100.0,
        "tegrastats_requested_interval_ms": 75.0,
        "telemetry_required_fraction": 0.8,
        "telemetry_max_gap_ms": 300.0,
        "telemetry_required_fields": list(
            SUMMARY.FORMAL_TELEMETRY_REQUIRED_FIELDS
        ),
    }


def telemetry_sample(
    timestamp: int,
    temperature: float = 90.0,
    *,
    safety_temperature: float = 90.0,
):
    raw = (
        "RAM 100/1000MB (lfb 1x4MB) CPU [10%@1000] "
        f"soc012@{temperature}C tj@{safety_temperature}C VIN 1000mW"
    )
    return SUMMARY.TelemetrySample(
        timestamp,
        raw,
        SUMMARY.parse_tegrastats_line(raw),
        900.0,
        (),
    )


def stable_telemetry_evidence(markers):
    samples = tuple(
        telemetry_sample(timestamp)
        for timestamp in range(1_000_000_000, 65_000_000_001, 100_000_000)
    )
    return SUMMARY.TelemetryEvidence(
        pathlib.Path("telemetry.jsonl"),
        "0" * 64,
        samples,
        tuple(markers),
    )


def thermal_precondition_fixture(
    label: str = "pre-static-mig",
    start: int = 1_000_000_000,
    end: int = 62_500_000_000,
    cleanup: int = 62_525_000_000,
):
    evidence_without_checks = stable_telemetry_evidence(())
    active_checks = []
    active_markers = []
    for index, sample_ns in enumerate(
        (end - 2_100_000_000, end - 1_100_000_000, end - 100_000_000)
    ):
        window = SUMMARY.thermal_summary_from_samples(
            evidence_without_checks.samples,
            sample_ns,
            window_seconds=60.0,
            not_before_ns=start,
        )
        check = {
            "label": label,
            "index": index,
            "sample_monotonic_ns": sample_ns,
            "passed": True,
            "consecutive_passes": index + 1,
            "window": window,
        }
        active_checks.append(check)
        active_markers.append(
            SUMMARY.TelemetryMarker(
                sample_ns + 10_000_000,
                "thermal_active_stability_check",
                check,
            )
        )
    markers = (
        SUMMARY.TelemetryMarker(900_000_000, "thermal_prepare", {"label": label}),
        SUMMARY.TelemetryMarker(start, "thermal_start", {"label": label}),
        *active_markers,
        SUMMARY.TelemetryMarker(
            end,
            "thermal_measurement_end",
            {
                "label": label,
                "boundary_sample_monotonic_ns": active_checks[-1][
                    "sample_monotonic_ns"
                ],
                "consecutive_passes": 3,
                "window": active_checks[-1]["window"],
            },
        ),
        SUMMARY.TelemetryMarker(
            cleanup, "thermal_end", {"label": label, "successful": True}
        ),
    )
    evidence = stable_telemetry_evidence(markers)
    last_window = SUMMARY.thermal_summary_from_samples(
        evidence.samples,
        active_checks[-1]["sample_monotonic_ns"],
        window_seconds=60.0,
        not_before_ns=start,
    )
    summary = {
        "label": label,
        "duration_seconds": (end - start) / 1_000_000_000.0,
        "measurement_start_monotonic_ns": start,
        "measurement_end_monotonic_ns": end,
        "cleanup_end_monotonic_ns": cleanup,
        "target_c": 90.0,
        "stability_sensor": "soc012",
        "safety_sensor": "tj",
        "last_window": last_window,
        "active_stability_checks": active_checks,
        "active_stable_endpoints": 3,
        "active_stable_spacing_seconds": 1.0,
        "termination_reason": "active-stability-endpoints",
        "pressure_rate_per_second": 100.0,
        "telemetry": SUMMARY.replay_telemetry_aggregate(evidence, start, end),
    }
    return summary, evidence, label


def thermal_attempt_fixture(
    base_label: str,
    phase_metadata: dict[str, object],
    measured_pids: list[int],
    *,
    start: int = 1_000_000_000,
    end: int = 62_500_000_000,
    cleanup: int = 62_525_000_000,
    qualification_ns: int = 62_625_000_000,
):
    attempt_label = f"{base_label}-attempt-01"
    precondition, precondition_evidence, _ = thermal_precondition_fixture(
        attempt_label, start, end, cleanup
    )
    qualification_sample_ns = cleanup + 75_000_000
    qualification_marker = SUMMARY.TelemetryMarker(
        qualification_ns,
        "thermal_start_qualification",
        {
            "label": attempt_label,
            "attempt": 1,
            "boundary_monotonic_ns": end,
            "cleanup_end_monotonic_ns": cleanup,
            "sample_monotonic_ns": qualification_sample_ns,
        },
    )
    result_marker_ns = qualification_ns + 20_000_000
    result_marker = SUMMARY.TelemetryMarker(
        result_marker_ns,
        "thermal_start_qualification_result",
        phase_metadata
        | {
            "label": attempt_label,
            "attempt": 1,
            "qualification_monotonic_ns": qualification_ns,
            "passed": True,
            "failure_reason": None,
        },
    )
    evidence = stable_telemetry_evidence(
        precondition_evidence.markers + (qualification_marker, result_marker)
    )
    qualification = {
        "attempt": 1,
        "passed": True,
        "boundary": "thermal_measurement_end",
        "boundary_monotonic_ns": end,
        "cleanup_end_monotonic_ns": cleanup,
        "qualification_monotonic_ns": qualification_ns,
        "sample_monotonic_ns": qualification_sample_ns,
        "sample_age_ms": (qualification_ns - qualification_sample_ns)
        / 1_000_000.0,
        "stability_sensor": "soc012",
        "stability_value_c": 90.0,
        "safety_sensor": "tj",
        "safety_value_c": 90.0,
        "target_c": 90.0,
        "tolerance_c": 1.0,
        "telemetry": SUMMARY.replay_point_telemetry_aggregate(
            evidence,
            sample_ns=qualification_sample_ns,
            reference_ns=qualification_ns,
        ),
        "failure_reason": None,
    }
    attempt = {
        "attempt": 1,
        "thermal_precondition": precondition,
        "qualification": qualification,
        "qualification_result_marker_monotonic_ns": result_marker_ns,
        "measured_process_states": {str(pid): "T" for pid in sorted(measured_pids)},
    }
    return attempt, precondition, qualification, evidence


def thermal_handoff_fixture(
    boundary_ns: int,
    cleanup_end_ns: int,
    qualification_ns: int,
    qualification_result_ns: int,
    release_ns: int,
    measurement_start_ns: int,
) -> dict[str, object]:
    return {
        "boundary": "thermal_measurement_end",
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
        "maximum_ms": 500.0,
        "strictly_within_bound": True,
    }


class MigSlackSummaryTest(unittest.TestCase):
    def test_presentation_has_one_proposed_system_name(self) -> None:
        self.assertEqual(SUMMARY.PROPOSED_SYSTEM, "QUIET")
        self.assertEqual(SUMMARY.PROPOSED_POLICY_ID, "mig-governor")
        self.assertEqual(set(SUMMARY.POLICY_PRESENTATION), SUMMARY.POLICIES)
        proposed = [
            (policy_id, label)
            for policy_id, (label, role) in SUMMARY.POLICY_PRESENTATION.items()
            if role == "proposed"
        ]
        self.assertEqual(proposed, [("mig-governor", "QUIET")])
        self.assertTrue(
            all(
                role == "proposed" or "QUIET" not in label
                for label, role in SUMMARY.POLICY_PRESENTATION.values()
            )
        )
        self.assertTrue(
            all(
                label not in {"BOER", "ParvaGPU", "BLESS", "REEF", "XSched"}
                for label, _ in SUMMARY.POLICY_PRESENTATION.values()
            )
        )

    def test_clopper_pearson_zero_miss_bound(self) -> None:
        self.assertAlmostEqual(
            SUMMARY.clopper_pearson_upper(0, 960),
            0.003115690582214472,
        )

    def test_formal_miss_threshold(self) -> None:
        self.assertLessEqual(
            SUMMARY.clopper_pearson_upper(178, 403_200), 0.0005
        )
        self.assertGreater(
            SUMMARY.clopper_pearson_upper(179, 403_200), 0.0005
        )

    def test_invalid_binomial_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SUMMARY.clopper_pearson_upper(2, 1)

    def test_formal_schedule_is_enforced(self) -> None:
        runs = [
            {"config": {"policy_order": list(order)}}
            for order in SUMMARY.SCHEDULED_WILLIAMS_ORDERS
        ]
        SUMMARY.validate_williams_blocks(runs)
        runs[-1] = runs[0]
        with self.assertRaises(ValueError):
            SUMMARY.validate_williams_blocks(runs)
        doubled = [
            {"config": {"policy_order": list(order)}}
            for order in SUMMARY.SCHEDULED_WILLIAMS_ORDERS * 2
        ]
        with self.assertRaises(ValueError):
            SUMMARY.validate_williams_blocks(doubled)

    def test_formal_protocol_rejects_relaxed_values(self) -> None:
        config = {
            "epochs": 36,
            "samples_per_epoch": 800,
            "warmup": 100,
            "burst_size": 8,
            "period_ms": 20.0,
            "dmr_target": 0.0005,
            "borrower_quota": 100,
            "calibration_repeats": 3,
            "max_isolated_drift_fraction": 0.05,
            "thermal_window_seconds": 60.0,
            "thermal_timeout_seconds": 900.0,
            "thermal_stability_checkpoint_seconds": 30.0,
            "thermal_stability_checkpoint_max_lateness_seconds": 1.0,
            "thermal_required_stable_checkpoints": 3,
            "tegrastats_requested_interval_ms": 75.0,
            "telemetry_interval_ms": 100,
            "telemetry_required_fraction": 0.8,
            "telemetry_stale_after_ms": 300,
            "telemetry_max_gap_ms": 300,
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
            "trace": [list(epoch) for epoch in SUMMARY.FORMAL_TRACE],
            "trace_assignment": "rotate-left-one-on-odd-six-epoch-cycle",
            "guard_override_ms": None,
            "profile_guard_ms": guard_profile_fixture(),
            "guard_profile_source": "frozen-quota-aware-lock",
            "cpu_affinity": copy.deepcopy(SUMMARY.FORMAL_CPU_AFFINITY),
            "telemetry_source": "tegrastats-readall-monotonic-jsonl",
            "telemetry_required_fields": list(
                SUMMARY.FORMAL_TELEMETRY_REQUIRED_FIELDS
            ),
        }
        guard_profile = guard_profile_fixture()
        SUMMARY.validate_formal_protocol(config, guard_profile)
        relaxed = {
            "epochs": 6,
            "samples_per_epoch": 400,
            "warmup": 20,
            "burst_size": 4,
            "period_ms": 25.0,
            "dmr_target": 0.001,
            "borrower_quota": 25,
            "calibration_repeats": 1,
            "max_isolated_drift_fraction": 0.10,
            "thermal_window_seconds": 30.0,
            "thermal_timeout_seconds": 600.0,
            "thermal_stability_checkpoint_seconds": 60.0,
            "thermal_stability_checkpoint_max_lateness_seconds": 2.0,
            "thermal_required_stable_checkpoints": 1,
            "tegrastats_requested_interval_ms": 100.0,
            "telemetry_interval_ms": 200,
            "telemetry_required_fraction": 0.5,
            "telemetry_stale_after_ms": 500,
            "telemetry_max_gap_ms": 500,
            "thermal_stability_sensor": "tj",
            "thermal_safety_sensor": "soc012",
            "thermal_handoff_max_ms": 501.0,
            "thermal_handoff_boundary": "thermal_start_qualification",
            "thermal_qualification_max_attempts": 4,
            "thermal_active_stable_endpoints": 2,
            "thermal_active_stable_spacing_seconds": 0.5,
            "thermal_calibration_preconditioning": "sequence-level",
        }
        for key, value in relaxed.items():
            with self.subTest(key=key):
                tampered = copy.deepcopy(config)
                tampered[key] = value
                with self.assertRaises(ValueError):
                    SUMMARY.validate_formal_protocol(tampered, guard_profile)
        tampered = copy.deepcopy(config)
        tampered["cpu_affinity"]["critical"] = [13]
        with self.assertRaises(ValueError):
            SUMMARY.validate_formal_protocol(tampered, guard_profile)
        tampered = copy.deepcopy(config)
        tampered["profile_guard_ms"]["resident-1g"]["25"]["audio"] += 0.1
        with self.assertRaisesRegex(ValueError, "profile_guard_ms"):
            SUMMARY.validate_formal_protocol(tampered, guard_profile)
        tampered = copy.deepcopy(config)
        tampered["guard_profile_source"] = "runtime-default"
        with self.assertRaisesRegex(ValueError, "frozen quota-aware"):
            SUMMARY.validate_formal_protocol(tampered, guard_profile)
        tampered = copy.deepcopy(config)
        tampered["thermal_qualification_dwell_seconds"] = 1.0
        with self.assertRaisesRegex(ValueError, "thermal semantics"):
            SUMMARY.validate_formal_protocol(tampered, guard_profile)

    def test_guard_lock_is_cross_bound_to_all_frozen_inputs(self) -> None:
        (
            guard_lock,
            guard_sha256,
            deadline_lock,
            thermal_lock,
            thermal_sha256,
        ) = guard_binding_fixture()
        profile = SUMMARY.validate_guard_lock_binding(
            guard_lock,
            guard_sha256,
            deadline_lock,
            thermal_lock,
            thermal_sha256,
        )
        self.assertEqual(profile, guard_profile_fixture())

        old_guard = copy.deepcopy(guard_lock)
        old_guard["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema 3"):
            SUMMARY.validate_guard_lock_binding(
                old_guard,
                guard_sha256,
                deadline_lock,
                thermal_lock,
                thermal_sha256,
            )

        mutations = (
            ("deadline guard SHA", deadline_lock, "guard_lock_sha256"),
            (
                "deadline thermal SHA",
                deadline_lock,
                "thermal_lock_sha256",
            ),
            ("guard hardware", guard_lock, "hardware"),
        )
        for label, target, key in mutations:
            with self.subTest(label=label):
                tampered_guard = copy.deepcopy(guard_lock)
                tampered_deadline = copy.deepcopy(deadline_lock)
                if target is guard_lock:
                    tampered_guard[key] = {"platform_sha256": "f" * 64}
                else:
                    tampered_deadline[key] = "f" * 64
                with self.assertRaises(ValueError):
                    SUMMARY.validate_guard_lock_binding(
                        tampered_guard,
                        guard_sha256,
                        tampered_deadline,
                        thermal_lock,
                        thermal_sha256,
                    )

        tampered = copy.deepcopy(guard_lock)
        tampered["artifacts"]["engine:resident-1g:q25:language"][
            "sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(ValueError, "guard engine"):
            SUMMARY.validate_guard_lock_binding(
                tampered,
                guard_sha256,
                deadline_lock,
                thermal_lock,
                thermal_sha256,
            )

    def test_guard_profile_is_quota_and_placement_aware(self) -> None:
        identities = [
            (0, "audio", "resident-1g", 25),
            (1, "audio", "resident-1g", 25),
            (2, "audio", "resident-1g", 25),
            (3, "language", "borrower-2g", 100),
            (4, "language", "borrower-2g", 100),
            (5, "language", "borrower-2g", 100),
        ]
        profile = guard_profile_fixture()
        self.assertAlmostEqual(
            SUMMARY.expected_guard_ms("fixed-full-gate", identities, profile),
            4.2,
        )
        self.assertAlmostEqual(
            SUMMARY.expected_guard_ms("resident-full-gate", identities, profile),
            3.6,
        )
        self.assertAlmostEqual(
            SUMMARY.expected_guard_ms("fixed-borrow", identities, profile),
            4.2,
        )
        self.assertEqual(
            SUMMARY.expected_guard_ms("static-mig", identities, profile),
            0.0,
        )

    def test_adaptive_action_difference_requires_an_effective_plan_change(self) -> None:
        offered = ("audio",) * 6
        profile = guard_profile_fixture()
        full = SUMMARY.default_feedback_state()
        self.assertFalse(
            SUMMARY.adaptive_action_differs_from_fixed_full(
                offered, full, 100, profile
            )
        )

        ineffective_cap = copy.deepcopy(full)
        ineffective_cap["borrower_limit"] = 5
        self.assertFalse(
            SUMMARY.adaptive_action_differs_from_fixed_full(
                offered, ineffective_cap, 100, profile
            )
        )

        lower_quota = copy.deepcopy(full)
        lower_quota["resident_quota_index"] = 1
        lower_quota["resident_quota_percent"] = 50
        self.assertTrue(
            SUMMARY.adaptive_action_differs_from_fixed_full(
                offered, lower_quota, 100, profile
            )
        )

        lower_admission = copy.deepcopy(full)
        lower_admission["resident_admission_limit"] = 5
        self.assertTrue(
            SUMMARY.adaptive_action_differs_from_fixed_full(
                offered, lower_admission, 100, profile
            )
        )

    def test_adaptive_claim_requires_clean_evidence_in_every_run(self) -> None:
        common = {
            "adaptive_action_epochs": 14,
            "adaptive_action_runs": 14,
            "total_runs": 14,
            "drift_valid": True,
            "governor_feasible": True,
            "baseline_admission_valid": True,
            "baseline_telemetry_valid": True,
            "baseline_slo_feasible": True,
            "paired_gain_supported": True,
        }
        self.assertEqual(
            SUMMARY.adaptive_claim_status(**common),
            "goodput-gain-supported",
        )

        cases = (
            ({"adaptive_action_epochs": 0, "adaptive_action_runs": 0}, "not-exercised"),
            ({"adaptive_action_runs": 13}, "partially-exercised"),
            ({"baseline_telemetry_valid": False}, "not-evaluable"),
            ({"baseline_admission_valid": False}, "not-evaluable"),
            ({"governor_feasible": False}, "not-supported"),
            (
                {
                    "baseline_slo_feasible": False,
                    "paired_gain_supported": False,
                },
                "protection-supported",
            ),
            ({"paired_gain_supported": False}, "no-incremental-benefit"),
        )
        for changes, expected in cases:
            with self.subTest(expected=expected):
                arguments = common | changes
                self.assertEqual(
                    SUMMARY.adaptive_claim_status(**arguments), expected
                )

    def test_policy_metrics_are_replayed_and_summary_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            policy, config, mig, artifacts = policy_fixture(directory)
            metrics, values = SUMMARY.recompute_policy_metrics(
                policy,
                directory / "summary.json",
                config,
                3.5,
                SUMMARY.RawTraceClaims(),
                mig,
                artifacts,
                guard_profile_fixture(),
            )
            self.assertEqual(values, [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(metrics["deadline_miss_rate"], 0.25)
            self.assertEqual(metrics["pressure_goodput_per_second"], 5.0)
            tampered_fields = {
                "deadline_miss_rate": 0.0,
                "violation_epoch_rate": 0.0,
                "critical_p99_ms_max": 1.0,
                "resident_goodput_per_second": 6.0,
                "borrower_goodput_per_second": 1.0,
                "pressure_goodput_per_second": 5000.0,
                "rejected_tenants": 1,
                "telemetry_unhealthy_epochs": 1,
                "gate_overhead_mean_ms": 0.0,
                "critical_gpu_duty_cycle_mean": 0.0,
            }
            for key, value in tampered_fields.items():
                with self.subTest(key=key):
                    tampered = copy.deepcopy(policy)
                    tampered[key] = value
                    with self.assertRaises(ValueError):
                        SUMMARY.recompute_policy_metrics(
                            tampered,
                            directory / "summary.json",
                            config,
                            3.5,
                            SUMMARY.RawTraceClaims(),
                            mig,
                            artifacts,
                            guard_profile_fixture(),
                        )
            tampered = copy.deepcopy(policy)
            tampered["goodput_by_modality"]["language"] = 6.0
            with self.assertRaisesRegex(ValueError, "goodput_by_modality"):
                SUMMARY.recompute_policy_metrics(
                    tampered,
                    directory / "summary.json",
                    config,
                    3.5,
                    SUMMARY.RawTraceClaims(),
                    mig,
                    artifacts,
                    guard_profile_fixture(),
                )

    def test_worker_and_critical_execution_provenance_is_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            policy, config, mig, artifacts = policy_fixture(directory)

            def replace_path(root, path, value) -> None:
                current = root
                for component in path[:-1]:
                    current = current[component]
                current[path[-1]] = value

            mutations = (
                (
                    ("epochs", 0, "workers", 0, "config", "include_transfers"),
                    False,
                    "formal command",
                ),
                (
                    (
                        "epochs",
                        0,
                        "workers",
                        0,
                        "execution_environment",
                        "mps_active_thread_percentage",
                    ),
                    50,
                    "formal execution environment",
                ),
                (
                    (
                        "epochs",
                        0,
                        "workers",
                        0,
                        "execution_environment",
                        "cuda_visible_devices",
                    ),
                    "MIG-GPU-wrong",
                    "formal execution environment",
                ),
                (
                    (
                        "epochs",
                        0,
                        "workers",
                        0,
                        "execution_environment",
                        "cpu_affinity",
                    ),
                    [1],
                    "formal execution environment",
                ),
                (
                    ("epochs", 0, "workers", 0, "gpu", "multiprocessors"),
                    4,
                    "formal MIG width",
                ),
                (
                    ("epochs", 0, "workers", 0, "engine"),
                    str(directory / "wrong.engine"),
                    "formal engine path",
                ),
                (
                    (
                        "epochs",
                        0,
                        "workers",
                        0,
                        "measurement_start_monotonic_ns",
                    ),
                    1_100_000_000,
                    "invalid measurement window",
                ),
                (
                    ("epochs", 0, "readiness_affinity", 0, "pid"),
                    1001,
                    "benchmark execution environment",
                ),
                (
                    ("epochs", 0, "readiness_affinity", 0, "tasks"),
                    [],
                    "must be non-empty",
                ),
                (
                    ("epochs", 0, "readiness_affinity", 0, "tasks", 0, "cpus"),
                    [1],
                    "invalid task affinity",
                ),
                (
                    ("epochs", 0, "critical", "model"),
                    "resnet18",
                    "formal ResNet50",
                ),
                (
                    ("epochs", 0, "critical", "gpu", "multiprocessors"),
                    11,
                    "formal 2g MIG width",
                ),
                (
                    (
                        "epochs",
                        0,
                        "critical",
                        "execution_environment",
                        "mps_active_thread_percentage",
                    ),
                    50,
                    "formal execution environment",
                ),
                (
                    ("epochs", 0, "critical", "config", "priority"),
                    "default",
                    "formal command",
                ),
                (
                    ("epochs", 0, "critical", "config", "start_paused"),
                    False,
                    "formal command",
                ),
                (
                    ("epochs", 0, "critical", "completed_requests"),
                    3,
                    "completed-request count",
                ),
                (
                    ("epochs", 0, "critical", "release_to_completion", "count"),
                    3,
                    "raw trace",
                ),
                (
                    (
                        "epochs",
                        0,
                        "critical",
                        "measurement_start_monotonic_ns",
                    ),
                    1_100_000_000,
                    "inconsistent measurement clocks",
                ),
                (
                    ("epochs", 0, "readiness_affinity", 1, "pid"),
                    998,
                    "benchmark execution environment",
                ),
            )
            for path, value, message in mutations:
                with self.subTest(path=path):
                    tampered = copy.deepcopy(policy)
                    replace_path(tampered, path, value)
                    with self.assertRaisesRegex(ValueError, message):
                        SUMMARY.recompute_policy_metrics(
                            tampered,
                            directory / "summary.json",
                            config,
                            3.5,
                            SUMMARY.RawTraceClaims(),
                            mig,
                            artifacts,
                            guard_profile_fixture(),
                        )

    def test_formal_worker_sm_widths_are_exact(self) -> None:
        self.assertEqual(SUMMARY.expected_worker_sm_count("resident-1g", 25), 2)
        self.assertEqual(SUMMARY.expected_worker_sm_count("resident-1g", 50), 4)
        self.assertEqual(SUMMARY.expected_worker_sm_count("resident-1g", 100), 8)
        self.assertEqual(SUMMARY.expected_worker_sm_count("borrower-2g", 100), 12)
        for placement, quota in (("resident-1g", 75), ("borrower-2g", 50)):
            with self.subTest(placement=placement, quota=quota):
                with self.assertRaisesRegex(ValueError, "unsupported formal MIG quota"):
                    SUMMARY.expected_worker_sm_count(placement, quota)

    def test_duplicate_worker_tenant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            policy, config, mig, artifacts = policy_fixture(directory)
            epoch = policy["epochs"][0]
            epoch["workers"].append(copy.deepcopy(epoch["workers"][0]))
            epoch["resident_actions"].append(
                copy.deepcopy(epoch["resident_actions"][0])
            )
            with self.assertRaisesRegex(ValueError, "duplicate worker tenant"):
                SUMMARY.recompute_policy_metrics(
                    policy,
                    directory / "summary.json",
                    config,
                    3.5,
                    SUMMARY.RawTraceClaims(),
                    mig,
                    artifacts,
                    guard_profile_fixture(),
                )

    def test_isolated_raw_replay_rejects_pooled_and_drift_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            run = isolated_fixture(directory)
            replay = SUMMARY.replay_isolated_calibrations(
                directory / "summary.json", run, SUMMARY.RawTraceClaims()
            )
            self.assertEqual(replay["pre_samples"], 8)
            self.assertTrue(replay["drift_valid"])

            tampered_p99 = copy.deepcopy(run)
            tampered_p99["isolated_pooled_p99_ms"] += 1.0
            with self.assertRaisesRegex(ValueError, "isolated_pooled_p99_ms"):
                SUMMARY.replay_isolated_calibrations(
                    directory / "summary.json",
                    tampered_p99,
                    SUMMARY.RawTraceClaims(),
                )

            tampered_boolean = copy.deepcopy(run)
            tampered_boolean["isolated_drift_valid"] = False
            with self.assertRaisesRegex(ValueError, "isolated_drift_valid"):
                SUMMARY.replay_isolated_calibrations(
                    directory / "summary.json",
                    tampered_boolean,
                    SUMMARY.RawTraceClaims(),
                )

            tampered_count = copy.deepcopy(run)
            tampered_count["isolated_post_pooled_samples"] = 7
            with self.assertRaisesRegex(ValueError, "isolated_post_pooled_samples"):
                SUMMARY.replay_isolated_calibrations(
                    directory / "summary.json",
                    tampered_count,
                    SUMMARY.RawTraceClaims(),
                )

            tampered_drift = copy.deepcopy(run)
            tampered_drift["isolated_post_reference_drift_fraction"] = 0.0
            with self.assertRaisesRegex(ValueError, "post_reference_drift"):
                SUMMARY.replay_isolated_calibrations(
                    directory / "summary.json",
                    tampered_drift,
                    SUMMARY.RawTraceClaims(),
                )

    def test_duplicate_isolated_trace_reuse_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            run = isolated_fixture(directory)
            claimed = SUMMARY.RawTraceClaims()
            SUMMARY.replay_isolated_calibrations(
                directory / "summary.json", run, claimed
            )
            with self.assertRaisesRegex(ValueError, "raw trace was reused"):
                SUMMARY.replay_isolated_calibrations(
                    directory / "summary.json", run, claimed
                )

    def test_raw_trace_hardlinks_and_identical_copies_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            rows = [(1.0, 0.1, 0.0), (2.0, 0.1, 0.0)]
            original = directory / "original.csv"
            duplicate = directory / "duplicate.csv"
            hardlink = directory / "hardlink.csv"
            write_trace(original, rows)
            write_trace(duplicate, rows)
            os.link(original, hardlink)

            claims = SUMMARY.RawTraceClaims()
            SUMMARY.claim_raw_trace(original, claims)
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                SUMMARY.claim_raw_trace(duplicate, claims)

            hardlink_claims = SUMMARY.RawTraceClaims()
            SUMMARY.claim_raw_trace(original, hardlink_claims)
            with self.assertRaisesRegex(ValueError, "hardlink"):
                SUMMARY.claim_raw_trace(hardlink, hardlink_claims)

    def test_coherent_modality_relabel_is_rejected_by_frozen_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            policy, config, mig, artifacts = policy_fixture(directory)
            epoch = policy["epochs"][0]
            epoch["offered_modalities"] = ["audio"]
            epoch["resident_actions"][0]["modality"] = "audio"
            epoch["workers"][0]["modality"] = "audio"
            epoch["workers"][0]["model"] = "whisper-tiny-encoder"
            epoch["workers"][0]["engine"] = (
                "/engines/mig-1g-q100/whisper-tiny-encoder.engine"
            )
            with self.assertRaisesRegex(ValueError, "frozen trace"):
                SUMMARY.recompute_policy_metrics(
                    policy,
                    directory / "summary.json",
                    config,
                    3.5,
                    SUMMARY.RawTraceClaims(),
                    mig,
                    artifacts,
                    guard_profile_fixture(),
                )

    def test_telemetry_jsonl_rejects_tampered_parsed_payload_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            path = directory / "telemetry.jsonl"
            sample = telemetry_sample(2_000_000_000)
            records = [
                SUMMARY.TelemetryMarker(
                    1_000_000_000, "collector_ready", {"source": "test"}
                ).to_record(),
                sample.to_record(),
                SUMMARY.TelemetryMarker(
                    3_000_000_000, "collector_end", {}
                ).to_record(),
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            evidence = SUMMARY.load_telemetry_evidence(path)
            self.assertEqual(len(evidence.samples), 1)
            self.assertEqual(evidence.sha256, SUMMARY.file_sha256(path))

            tampered = copy.deepcopy(records)
            tampered[1]["parsed"]["temperatures_c"]["tj"] = 10.0
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in tampered),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "differs from raw"):
                SUMMARY.load_telemetry_evidence(path)

            duplicate = [records[0], records[1], records[1], records[2]]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in duplicate),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                SUMMARY.load_telemetry_evidence(path)

    def test_thermal_precondition_replays_raw_window_and_aggregate(self) -> None:
        summary, evidence, label = thermal_precondition_fixture()
        SUMMARY.validate_thermal_precondition(
            summary, label, evidence, thermal_lock_fixture()
        )

        tampered = copy.deepcopy(summary)
        tampered["last_window"]["mean_c"] = 80.0
        with self.assertRaisesRegex(ValueError, "last_window"):
            SUMMARY.validate_thermal_precondition(
                tampered, label, evidence, thermal_lock_fixture()
            )

        boundary_sample_ns = summary["active_stability_checks"][-1][
            "sample_monotonic_ns"
        ]
        omitted = SUMMARY.TelemetryEvidence(
            evidence.path,
            evidence.sha256,
            tuple(
                sample
                for sample in evidence.samples
                if sample.monotonic_ns != boundary_sample_ns
            ),
            evidence.markers,
        )
        with self.assertRaisesRegex(ValueError, "non-causal endpoint"):
            SUMMARY.validate_thermal_precondition(
                summary, label, omitted, thermal_lock_fixture()
            )

    def test_v4_thermal_labels_reject_stale_sequence_precondition(self) -> None:
        attempt, summary, _qualification, evidence = thermal_attempt_fixture(
            "pre-static-mig-epoch-00",
            {"policy": "static-mig", "epoch": 0},
            [999],
        )
        label = attempt["thermal_precondition"]["label"]
        SUMMARY.validate_thermal_marker_labels(evidence, [label])
        stale_label = "pre-sequence-calibration"
        stale_markers = evidence.markers + (
            SUMMARY.TelemetryMarker(
                61_200_000_000, "thermal_prepare", {"label": stale_label}
            ),
            SUMMARY.TelemetryMarker(
                61_300_000_000, "thermal_start", {"label": stale_label}
            ),
            SUMMARY.TelemetryMarker(
                61_400_000_000,
                "thermal_measurement_end",
                {"label": stale_label},
            ),
            SUMMARY.TelemetryMarker(
                61_500_000_000,
                "thermal_end",
                {"label": stale_label, "successful": True},
            ),
        )
        stale = SUMMARY.TelemetryEvidence(
            evidence.path, evidence.sha256, evidence.samples, stale_markers
        )
        with self.assertRaisesRegex(ValueError, "v4 formal run"):
            SUMMARY.validate_thermal_marker_labels(stale, [label])

        tampered = copy.deepcopy(summary)
        tampered["telemetry"]["total_samples"] -= 1
        with self.assertRaisesRegex(ValueError, "total_samples"):
            SUMMARY.validate_thermal_precondition(
                tampered, label, evidence, thermal_lock_fixture()
            )

        tampered = copy.deepcopy(summary)
        tampered["stability_sensor"] = "tj"
        with self.assertRaisesRegex(ValueError, "sensor binding"):
            SUMMARY.validate_thermal_precondition(
                tampered, label, evidence, thermal_lock_fixture()
            )

    def test_epoch_telemetry_replay_rejects_tampered_tj_and_large_gap(self) -> None:
        policy = "static-mig"
        epoch_index = 0
        metadata = {"policy": policy, "epoch": epoch_index}
        attempt, precondition, qualification, precondition_evidence = (
            thermal_attempt_fixture(
                "pre-static-mig-epoch-00", metadata, [999, 1000]
            )
        )
        release = 62_700_000_000
        start = 62_800_000_000
        end = 63_200_000_000
        actual_marker_ns = 63_210_000_000
        collected = 63_300_000_000
        actual_sample_ns = start
        markers = (
            SUMMARY.TelemetryMarker(800_000_000, "epoch_prepare", metadata),
        ) + precondition_evidence.markers + (
            SUMMARY.TelemetryMarker(release, "measurement_start", metadata),
            SUMMARY.TelemetryMarker(
                actual_marker_ns,
                "thermal_actual_start_qualification_result",
                metadata
                | {
                    "measurement_start_monotonic_ns": start,
                    "sample_monotonic_ns": actual_sample_ns,
                    "passed": True,
                    "failure_reason": None,
                },
            ),
            SUMMARY.TelemetryMarker(
                collected,
                "measurement_result_collected",
                metadata
                | {
                    "measurement_start_monotonic_ns": start,
                    "measurement_end_monotonic_ns": end,
                },
            ),
            SUMMARY.TelemetryMarker(63_400_000_000, "cleanup_end", metadata),
        )
        evidence = stable_telemetry_evidence(markers)
        epoch = {
            "measurement_release_monotonic_ns": release,
            "measurement_start_monotonic_ns": start,
            "measurement_end_monotonic_ns": end,
            "result_collected_monotonic_ns": collected,
            "telemetry": SUMMARY.replay_telemetry_aggregate(evidence, start, end),
            "telemetry_unhealthy": False,
            "thermal_start": precondition["last_window"],
            "thermal_start_telemetry": qualification["telemetry"],
            "thermal_start_attempts": [attempt],
            "thermal_start_qualification": qualification,
            "thermal_actual_start_qualification": {
                "passed": True,
                "measurement_start_monotonic_ns": start,
                "sample_monotonic_ns": actual_sample_ns,
                "sample_age_ms": 0.0,
                "stability_sensor": "soc012",
                "stability_value_c": 90.0,
                "safety_sensor": "tj",
                "safety_value_c": 90.0,
                "target_c": 90.0,
                "tolerance_c": 1.0,
                "telemetry": SUMMARY.replay_point_telemetry_aggregate(
                    evidence,
                    sample_ns=actual_sample_ns,
                    reference_ns=start,
                ),
                "failure_reason": None,
            },
            "thermal_start_stable": True,
            "thermal_handoff": thermal_handoff_fixture(
                qualification["boundary_monotonic_ns"],
                qualification["cleanup_end_monotonic_ns"],
                qualification["qualification_monotonic_ns"],
                attempt["qualification_result_marker_monotonic_ns"],
                release,
                start,
            ),
            "thermal_high": False,
            "readiness_affinity": [
                {"role": "pressure", "pid": 1000},
                {"role": "critical", "pid": 999},
            ],
        }
        SUMMARY.validate_epoch_telemetry(
            epoch,
            policy,
            epoch_index,
            evidence,
            thermal_lock_fixture(),
            precondition,
        )

        pressure_samples = tuple(
            telemetry_sample(sample.monotonic_ns, 92.0)
            if sample.monotonic_ns == 62_900_000_000
            else sample
            for sample in evidence.samples
        )
        pressure_evidence = SUMMARY.TelemetryEvidence(
            evidence.path,
            evidence.sha256,
            pressure_samples,
            evidence.markers,
        )
        pressure_epoch = copy.deepcopy(epoch)
        pressure_epoch["telemetry"] = SUMMARY.replay_telemetry_aggregate(
            pressure_evidence, start, end
        )
        pressure_epoch["thermal_high"] = True
        SUMMARY.validate_epoch_telemetry(
            pressure_epoch,
            policy,
            epoch_index,
            pressure_evidence,
            thermal_lock_fixture(),
            precondition,
        )

        tampered = copy.deepcopy(epoch)
        tampered["telemetry"]["temperatures_c"]["tj"]["max"] = 91.0
        with self.assertRaisesRegex(ValueError, "temperatures_c.tj.max"):
            SUMMARY.validate_epoch_telemetry(
                tampered,
                policy,
                epoch_index,
                evidence,
                thermal_lock_fixture(),
                precondition,
            )

        hot_samples = tuple(
            telemetry_sample(
                sample.monotonic_ns, safety_temperature=104.0
            )
            if sample.monotonic_ns == 62_900_000_000
            else sample
            for sample in evidence.samples
        )
        hot_evidence = SUMMARY.TelemetryEvidence(
            evidence.path,
            evidence.sha256,
            hot_samples,
            evidence.markers,
        )
        hot_epoch = copy.deepcopy(epoch)
        hot_epoch["telemetry"] = SUMMARY.replay_telemetry_aggregate(
            hot_evidence, start, end
        )
        with self.assertRaisesRegex(ValueError, "tj thermal hard limit"):
            SUMMARY.validate_epoch_telemetry(
                hot_epoch,
                policy,
                epoch_index,
                hot_evidence,
                thermal_lock_fixture(),
                precondition,
            )

        unstable_samples = tuple(
            telemetry_sample(sample.monotonic_ns, 80.0)
            if sample.monotonic_ns == release
            else sample
            for sample in evidence.samples
        )
        unstable_evidence = SUMMARY.TelemetryEvidence(
            evidence.path,
            evidence.sha256,
            unstable_samples,
            evidence.markers,
        )
        unstable_epoch = copy.deepcopy(epoch)
        # Schema 4 gates the active boundary, first post-cleanup sample, and
        # actual measurement-start sample. It intentionally does not restore
        # the retired post-cleanup 60-second OLS gate at release.
        SUMMARY.validate_epoch_telemetry(
            unstable_epoch,
            policy,
            epoch_index,
            unstable_evidence,
            thermal_lock_fixture(),
            precondition,
        )

        sparse_samples = tuple(
            sample
            for sample in evidence.samples
            if not start < sample.monotonic_ns < end
        )
        sparse = SUMMARY.TelemetryEvidence(
            evidence.path,
            evidence.sha256,
            sparse_samples,
            evidence.markers,
        )
        with self.assertRaisesRegex(ValueError, "gap over 300 ms"):
            SUMMARY.validate_epoch_telemetry(
                epoch,
                policy,
                epoch_index,
                sparse,
                thermal_lock_fixture(),
                precondition,
            )

        reordered_markers = tuple(
            SUMMARY.TelemetryMarker(
                63_250_000_000, marker.name, marker.metadata
            )
            if marker.name == "cleanup_end"
            else marker
            for marker in evidence.markers
        )
        reordered = SUMMARY.TelemetryEvidence(
            evidence.path,
            evidence.sha256,
            evidence.samples,
            reordered_markers,
        )
        with self.assertRaisesRegex(ValueError, "marker chain"):
            SUMMARY.validate_epoch_telemetry(
                epoch,
                policy,
                epoch_index,
                reordered,
                thermal_lock_fixture(),
                precondition,
            )

    def test_governor_transition_is_replayed(self) -> None:
        state = {
            "resident_admission_limit": 6,
            "resident_quota_index": 2,
            "borrower_limit": 6,
            "guard_adjustment_ms": 0.0,
            "safe_epochs": 0,
            "resident_quota_percent": 100,
        }
        after, action = SUMMARY.replay_governor_transition(
            state,
            telemetry_unhealthy=False,
            violated=False,
            critical_p99_ms=4.0,
            deadline_ms=5.0,
            drain_near_overrun=True,
            thermal_high=False,
        )
        self.assertEqual(action, "drain-reclaim")
        self.assertEqual(after["borrower_limit"], 5)
        self.assertEqual(after["resident_admission_limit"], 5)
        self.assertEqual(after["guard_adjustment_ms"], 0.0)

        after, action = SUMMARY.replay_governor_transition(
            state,
            telemetry_unhealthy=True,
            violated=False,
            critical_p99_ms=4.0,
            deadline_ms=5.0,
            drain_near_overrun=False,
            thermal_high=False,
        )
        self.assertEqual(action, "telemetry-fail-closed")
        self.assertEqual(after["resident_admission_limit"], 1)
        self.assertEqual(after["resident_quota_percent"], 25)

        recovering = {
            "resident_admission_limit": 1,
            "resident_quota_index": 0,
            "borrower_limit": 0,
            "guard_adjustment_ms": 0.0,
            "safe_epochs": 2,
            "resident_quota_percent": 25,
        }
        after, action = SUMMARY.replay_governor_transition(
            recovering,
            telemetry_unhealthy=False,
            violated=False,
            critical_p99_ms=1.0,
            deadline_ms=5.0,
            drain_near_overrun=False,
            thermal_high=False,
        )
        self.assertEqual(action, "recover-resident-quota")
        self.assertEqual(after["resident_quota_percent"], 50)
        self.assertEqual(after["resident_admission_limit"], 1)
        self.assertEqual(after["borrower_limit"], 0)

        recovery_cases = (
            (
                {
                    "resident_admission_limit": 1,
                    "resident_quota_index": 2,
                    "borrower_limit": 0,
                    "guard_adjustment_ms": 0.0,
                    "safe_epochs": 2,
                    "resident_quota_percent": 100,
                },
                "recover-admission",
            ),
            (
                {
                    "resident_admission_limit": 6,
                    "resident_quota_index": 2,
                    "borrower_limit": 0,
                    "guard_adjustment_ms": 0.0,
                    "safe_epochs": 2,
                    "resident_quota_percent": 100,
                },
                "recover-borrower",
            ),
            (
                {
                    "resident_admission_limit": 6,
                    "resident_quota_index": 2,
                    "borrower_limit": 6,
                    "guard_adjustment_ms": 0.0,
                    "safe_epochs": 2,
                    "resident_quota_percent": 100,
                },
                "hold-full-capacity",
            ),
        )
        for recovery_state, expected_action in recovery_cases:
            with self.subTest(action=expected_action):
                _after, action = SUMMARY.replay_governor_transition(
                    recovery_state,
                    telemetry_unhealthy=False,
                    violated=False,
                    critical_p99_ms=1.0,
                    deadline_ms=5.0,
                    drain_near_overrun=False,
                    thermal_high=False,
                )
                self.assertEqual(action, expected_action)

    def test_policy_replay_rejects_coherent_governor_state_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            policy, config, mig, artifacts = policy_fixture(directory)
            rows = [
                (1.0, 0.1, 0.2),
                (2.0, 0.1, 0.2),
                (3.0, 0.1, 0.2),
                (4.0, 0.1, 0.2),
            ]
            write_trace(directory / "raw" / "mig-governor-e0.csv", rows)
            policy["name"] = "mig-governor"
            epoch = policy["epochs"][0]
            epoch["gate_scope"] = ["borrower-2g", "resident-1g"]
            epoch["gated_workers"] = 1
            epoch["guard_ms"] = 1.5
            epoch["critical"]["config"].update(
                {
                    "guard_ms": 1.5,
                    "gated_processes": 1,
                    "gate_mode": "cooperative",
                }
            )
            epoch["controller_action"] = "residual-reclaim"
            epoch["state_after"] = {
                "resident_admission_limit": 6,
                "resident_quota_index": 1,
                "borrower_limit": 5,
                "guard_adjustment_ms": 0.0,
                "safe_epochs": 0,
                "resident_quota_percent": 50,
            }
            SUMMARY.recompute_policy_metrics(
                policy,
                directory / "summary.json",
                config,
                3.5,
                SUMMARY.RawTraceClaims(),
                mig,
                artifacts,
                guard_profile_fixture(),
            )

            tampered = copy.deepcopy(policy)
            tampered["epochs"][0]["state_after"]["borrower_limit"] = 6
            with self.assertRaisesRegex(ValueError, "controller replay"):
                SUMMARY.recompute_policy_metrics(
                    tampered,
                    directory / "summary.json",
                    config,
                    3.5,
                    SUMMARY.RawTraceClaims(),
                    mig,
                    artifacts,
                    guard_profile_fixture(),
                )

            tampered = copy.deepcopy(policy)
            tampered["epochs"][0]["critical"]["config"]["gated_processes"] = 0
            with self.assertRaisesRegex(ValueError, "formal command"):
                SUMMARY.recompute_policy_metrics(
                    tampered,
                    directory / "summary.json",
                    config,
                    3.5,
                    SUMMARY.RawTraceClaims(),
                    mig,
                    artifacts,
                    guard_profile_fixture(),
                )

            tampered = copy.deepcopy(policy)
            tampered["epochs"][0]["state_before"]["guard_adjustment_ms"] = 0.1
            with self.assertRaisesRegex(ValueError, "controller bounds"):
                SUMMARY.recompute_policy_metrics(
                    tampered,
                    directory / "summary.json",
                    config,
                    3.5,
                    SUMMARY.RawTraceClaims(),
                    mig,
                    artifacts,
                    guard_profile_fixture(),
                )

    def test_calibration_markers_bind_thermal_window(self) -> None:
        stage = "pre"
        repeat = 1
        metadata = {"stage": stage, "repeat": repeat}
        attempt, precondition, qualification, precondition_evidence = (
            thermal_attempt_fixture(
                "pre-pre-calibration-r1", metadata, [999]
            )
        )
        release = 62_700_000_000
        measurement_start = 62_800_000_000
        measurement_end = 63_100_000_000
        actual_marker_ns = 63_110_000_000
        actual_sample_ns = measurement_start
        markers = (
            SUMMARY.TelemetryMarker(800_000_000, "calibration_prepare", metadata),
        ) + precondition_evidence.markers + (
            SUMMARY.TelemetryMarker(release, "calibration_start", metadata),
            SUMMARY.TelemetryMarker(
                actual_marker_ns,
                "thermal_actual_start_qualification_result",
                metadata
                | {
                    "measurement_start_monotonic_ns": measurement_start,
                    "sample_monotonic_ns": actual_sample_ns,
                    "passed": True,
                    "failure_reason": None,
                },
            ),
            SUMMARY.TelemetryMarker(
                63_200_000_000,
                "calibration_measurement_window",
                metadata
                | {
                    "measurement_start_monotonic_ns": measurement_start,
                    "measurement_end_monotonic_ns": measurement_end,
                },
            ),
            SUMMARY.TelemetryMarker(63_300_000_000, "calibration_end", metadata),
        )
        evidence = stable_telemetry_evidence(markers)
        result = {
            "config": {"start_paused": True},
            "execution_environment": {
                "pid": 999,
                "cuda_visible_devices": MIG_FIXTURE["critical_uuid"],
                "mps_active_thread_percentage": 100,
                "cpu_affinity": [12],
            },
            "readiness_affinity": {
                "pid": 999,
                "expected_cpu": 12,
                "tasks": [{"tid": 999, "cpus": [12]}],
            },
            "measurement_release_monotonic_ns": release,
            "measurement_start_monotonic_ns": measurement_start,
            "measurement_end_monotonic_ns": measurement_end,
            "thermal_start": precondition["last_window"],
            "thermal_start_telemetry": qualification["telemetry"],
            "thermal_start_attempts": [attempt],
            "thermal_precondition": precondition,
            "thermal_start_qualification": qualification,
            "thermal_actual_start_qualification": {
                "passed": True,
                "measurement_start_monotonic_ns": measurement_start,
                "sample_monotonic_ns": actual_sample_ns,
                "sample_age_ms": 0.0,
                "stability_sensor": "soc012",
                "stability_value_c": 90.0,
                "safety_sensor": "tj",
                "safety_value_c": 90.0,
                "target_c": 90.0,
                "tolerance_c": 1.0,
                "telemetry": SUMMARY.replay_point_telemetry_aggregate(
                    evidence,
                    sample_ns=actual_sample_ns,
                    reference_ns=measurement_start,
                ),
                "failure_reason": None,
            },
            "thermal_start_stable": True,
            "thermal_handoff": thermal_handoff_fixture(
                qualification["boundary_monotonic_ns"],
                qualification["cleanup_end_monotonic_ns"],
                qualification["qualification_monotonic_ns"],
                attempt["qualification_result_marker_monotonic_ns"],
                release,
                measurement_start,
            ),
            "thermal_precondition_label": "pre-pre-calibration-r1-attempt-01",
        }
        run = {
            "isolated": [result],
            "isolated_preconditions": [precondition],
        }
        config = {"cpu_affinity": copy.deepcopy(SUMMARY.FORMAL_CPU_AFFINITY)}
        SUMMARY.validate_calibration_markers(
            evidence,
            run,
            stage,
            1,
            thermal_lock_fixture(),
            config,
            MIG_FIXTURE,
        )

        tampered = copy.deepcopy(run)
        tampered["isolated"][0]["thermal_start"]["latest_c"] = 80.0
        with self.assertRaisesRegex(ValueError, "last_window"):
            SUMMARY.validate_calibration_markers(
                evidence,
                tampered,
                stage,
                1,
                thermal_lock_fixture(),
                config,
                MIG_FIXTURE,
            )

        missing = copy.deepcopy(run)
        missing["isolated_preconditions"] = []
        with self.assertRaisesRegex(ValueError, "one thermal precondition per repeat"):
            SUMMARY.validate_calibration_markers(
                evidence,
                missing,
                stage,
                1,
                thermal_lock_fixture(),
                config,
                MIG_FIXTURE,
            )

        reordered_markers = tuple(
            SUMMARY.TelemetryMarker(
                62_490_000_000, marker.name, marker.metadata
            )
            if marker.name == "thermal_end"
            else marker
            for marker in evidence.markers
        )
        reordered = SUMMARY.TelemetryEvidence(
            evidence.path,
            evidence.sha256,
            evidence.samples,
            reordered_markers,
        )
        with self.assertRaisesRegex(ValueError, "timestamps|marker order"):
            SUMMARY.validate_calibration_markers(
                reordered,
                run,
                stage,
                1,
                thermal_lock_fixture(),
                config,
                MIG_FIXTURE,
            )

    def test_thermal_handoff_rejects_500_ms_boundary(self) -> None:
        boundary = 61_000_000_000
        cleanup = boundary + 25_000_000
        qualification = boundary + 75_000_000
        qualification_result = boundary + 85_000_000
        release = 61_200_000_000
        measurement_start = 61_500_000_000
        stored = thermal_handoff_fixture(
            boundary,
            cleanup,
            qualification,
            qualification_result,
            release,
            measurement_start,
        )
        with self.assertRaisesRegex(ValueError, "strictly within"):
            SUMMARY.validate_thermal_handoff(
                stored,
                boundary_ns=boundary,
                cleanup_end_ns=cleanup,
                qualification_ns=qualification,
                qualification_result_ns=qualification_result,
                release_ns=release,
                measurement_start_ns=measurement_start,
                thermal_lock=thermal_lock_fixture(),
                label="calibration handoff",
            )

    def test_qualification_attempt_contract_tamper_is_rejected(self) -> None:
        metadata = {"stage": "pre", "repeat": 1}
        attempt, _precondition, _qualification, evidence = thermal_attempt_fixture(
            "pre-pre-calibration-r1", metadata, [999]
        )
        SUMMARY.validate_thermal_start_attempts(
            [attempt],
            base_label="pre-pre-calibration-r1",
            expected_pids=[999],
            evidence=evidence,
            thermal_lock=thermal_lock_fixture(),
            result_marker_metadata=metadata,
        )
        cases = (
            (
                lambda value: value.__setitem__("measured_process_states", {}),
                "remained paused",
            ),
            (
                lambda value: value["qualification"].__setitem__(
                    "dwell_seconds", 0.5
                ),
                "invalid fields",
            ),
            (
                lambda value: value.__setitem__(
                    "measurement_release_monotonic_ns", 1
                ),
                "invalid fields",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                tampered = copy.deepcopy(attempt)
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, message):
                    SUMMARY.validate_thermal_start_attempts(
                        [tampered],
                        base_label="pre-pre-calibration-r1",
                        expected_pids=[999],
                        evidence=evidence,
                        thermal_lock=thermal_lock_fixture(),
                        result_marker_metadata=metadata,
                    )
        with self.assertRaisesRegex(ValueError, "attempt count"):
            SUMMARY.validate_thermal_start_attempts(
                [copy.deepcopy(attempt) for _ in range(4)],
                base_label="pre-pre-calibration-r1",
                expected_pids=[999],
                evidence=evidence,
                thermal_lock=thermal_lock_fixture(),
                result_marker_metadata=metadata,
            )
        with self.assertRaisesRegex(ValueError, "first successful"):
            SUMMARY.validate_thermal_start_attempts(
                [copy.deepcopy(attempt), copy.deepcopy(attempt)],
                base_label="pre-pre-calibration-r1",
                expected_pids=[999],
                evidence=evidence,
                thermal_lock=thermal_lock_fixture(),
                result_marker_metadata=metadata,
            )

    def test_schema3_lock_and_stale_handoff_boundary_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "thermal-lock.json"
            stale = thermal_lock_fixture()
            stale["schema_version"] = 3
            path.write_text(json.dumps(stale), encoding="utf-8")
            with mock.patch.object(SUMMARY, "verify_thermal_lock"):
                with self.assertRaisesRegex(ValueError, "schema 4"):
                    SUMMARY.load_verified_v4_thermal_lock(path)
            stale = thermal_lock_fixture()
            stale["thermal_qualification_dwell_seconds"] = 1.0
            path.write_text(json.dumps(stale), encoding="utf-8")
            with mock.patch.object(SUMMARY, "verify_thermal_lock"):
                with self.assertRaisesRegex(ValueError, "schema 4"):
                    SUMMARY.load_verified_v4_thermal_lock(path)
        boundary = 1_000_000_000
        cleanup = boundary + 25_000_000
        qualification = boundary + 75_000_000
        qualification_result = boundary + 85_000_000
        release = boundary + 100_000_000
        measurement_start = boundary + 200_000_000
        stored = thermal_handoff_fixture(
            boundary,
            cleanup,
            qualification,
            qualification_result,
            release,
            measurement_start,
        )
        stored["boundary"] = "thermal_start_qualification"
        with self.assertRaisesRegex(ValueError, "stale"):
            SUMMARY.validate_thermal_handoff(
                stored,
                boundary_ns=boundary,
                cleanup_end_ns=cleanup,
                qualification_ns=qualification,
                qualification_result_ns=qualification_result,
                release_ns=release,
                measurement_start_ns=measurement_start,
                thermal_lock=thermal_lock_fixture(),
                label="stale handoff",
            )

    def test_qualification_rejects_wrong_first_causal_sample(self) -> None:
        metadata = {"stage": "pre", "repeat": 1}
        attempt, _precondition, _qualification, evidence = thermal_attempt_fixture(
            "pre-pre-calibration-r1",
            metadata,
            [999],
            qualification_ns=62_725_000_000,
        )
        first_sample_ns = 62_600_000_000
        forged_sample_ns = 62_700_000_000
        samples = tuple(
            telemetry_sample(sample.monotonic_ns, 92.0)
            if sample.monotonic_ns == first_sample_ns
            else sample
            for sample in evidence.samples
        )
        markers = tuple(
            SUMMARY.TelemetryMarker(
                marker.monotonic_ns,
                marker.name,
                dict(marker.metadata) | {"sample_monotonic_ns": forged_sample_ns},
            )
            if marker.name == "thermal_start_qualification"
            else marker
            for marker in evidence.markers
        )
        forged_evidence = SUMMARY.TelemetryEvidence(
            evidence.path, evidence.sha256, samples, markers
        )
        forged = copy.deepcopy(attempt)
        forged_qualification = forged["qualification"]
        forged_qualification["sample_monotonic_ns"] = forged_sample_ns
        forged_qualification["sample_age_ms"] = 25.0
        forged_qualification["telemetry"] = SUMMARY.replay_point_telemetry_aggregate(
            forged_evidence,
            sample_ns=forged_sample_ns,
            reference_ns=forged_qualification["qualification_monotonic_ns"],
        )
        with self.assertRaisesRegex(ValueError, "first causal sample"):
            SUMMARY.validate_thermal_start_attempts(
                [forged],
                base_label="pre-pre-calibration-r1",
                expected_pids=[999],
                evidence=forged_evidence,
                thermal_lock=thermal_lock_fixture(),
                result_marker_metadata=metadata,
            )

    def test_qualification_rejects_postcleanup_cooling_outside_band(self) -> None:
        metadata = {"stage": "pre", "repeat": 1}
        attempt, _precondition, _qualification, evidence = thermal_attempt_fixture(
            "pre-pre-calibration-r1", metadata, [999]
        )
        sample_ns = attempt["qualification"]["sample_monotonic_ns"]
        cooled = SUMMARY.TelemetryEvidence(
            evidence.path,
            evidence.sha256,
            tuple(
                telemetry_sample(sample.monotonic_ns, 92.0)
                if sample.monotonic_ns == sample_ns
                else sample
                for sample in evidence.samples
            ),
            evidence.markers,
        )
        forged = copy.deepcopy(attempt)
        forged["qualification"]["stability_value_c"] = 92.0
        forged["qualification"]["telemetry"] = (
            SUMMARY.replay_point_telemetry_aggregate(
                cooled,
                sample_ns=sample_ns,
                reference_ns=forged["qualification"][
                    "qualification_monotonic_ns"
                ],
            )
        )
        with self.assertRaisesRegex(ValueError, "differs from raw telemetry"):
            SUMMARY.validate_thermal_start_attempts(
                [forged],
                base_label="pre-pre-calibration-r1",
                expected_pids=[999],
                evidence=cooled,
                thermal_lock=thermal_lock_fixture(),
                result_marker_metadata=metadata,
            )


if __name__ == "__main__":
    unittest.main()
