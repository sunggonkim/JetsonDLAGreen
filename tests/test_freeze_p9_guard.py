#!/usr/bin/env python3
"""Focused replay and fail-closed tests for the P9 guard freezer."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import freeze_p9_guard as freeze


TELEMETRY_RAW = "RAM 1/2MB CPU [50%@1000] tj@89.5C VIN 100000mW"


def thermal_lock() -> dict[str, object]:
    return {
        "schema_version": 4,
        "stability_sensor": "soc012",
        "safety_sensor": "tj",
        "thermal_handoff_max_ms": 500.0,
        "thermal_handoff_boundary": "thermal_measurement_end",
        "thermal_qualification_max_attempts": 3,
        "thermal_active_stable_endpoints": 3,
        "thermal_active_stable_spacing_seconds": 1.0,
        "target_c": 75.0,
        "tolerance_c": 1.0,
        "stability_window_seconds": 60.0,
        "maximum_slope_c_per_minute": 0.2,
        "hard_limit_c": 104.0,
        "telemetry_interval_ms": 100.0,
        "telemetry_required_fraction": 0.8,
        "telemetry_max_gap_ms": 300.0,
    }


def trace_bytes(*, tampered_request: bool = False, offset: float = 0.0) -> bytes:
    rows = [",".join(freeze.TRACE_FIELDS)]
    for request in range(freeze.EVENTS_PER_BLOCK):
        recorded_request = 0 if tampered_request and request == 1 else request
        drain = 0.2 + offset + request * 0.000001
        resume = 0.01
        rows.append(
            f"{recorded_request},1.0,0.5,0.1,{drain + resume:.9f},"
            f"{drain:.9f},{resume:.9f}"
        )
    return ("\n".join(rows) + "\n").encode()


def telemetry_sample(timestamp: int = 100) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "sample",
        "monotonic_ns": timestamp,
        "raw": TELEMETRY_RAW,
        "parsed": freeze.parse_tegrastats_line(TELEMETRY_RAW).to_dict(),
        "mem_available_mb": 1.0,
        "collection_errors": [],
    }


def telemetry_marker(timestamp: int = 200) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "marker",
        "monotonic_ns": timestamp,
        "name": "test_marker",
        "metadata": {},
    }


def thermal_sample(
    timestamp: int, *, soc012_c: float = 75.0, tj_c: float = 89.5
) -> dict[str, object]:
    raw = (
        "RAM 1/2MB CPU [50%@1000] "
        f"soc012@{soc012_c}C tj@{tj_c}C VIN 100000mW"
    )
    return {
        "schema_version": 1,
        "record_type": "sample",
        "monotonic_ns": timestamp,
        "raw": raw,
        "parsed": freeze.parse_tegrastats_line(raw).to_dict(),
        "mem_available_mb": 1.0,
        "collection_errors": [],
    }


def telemetry_bytes(*records: dict[str, object], terminated: bool = True) -> bytes:
    payload = "\n".join(json.dumps(record) for record in records)
    if terminated:
        payload += "\n"
    return payload.encode()


def latency_summary(count: int, value: float) -> dict[str, float | int]:
    return {
        "count": count,
        "mean_ms": value,
        "p50_ms": value,
        "p95_ms": value * 1.1,
        "p99_ms": value * 1.2,
        "p999_ms": value * 1.3,
        "max_ms": value * 1.4,
    }


def worker_evidence(engine: pathlib.Path) -> tuple[dict[str, object], dict[str, object]]:
    completed = 20
    pid = 321
    expected_client = {
        "placement": "resident-1g",
        "quota_percent": 25,
        "modality": "language",
        "model": "distilbert-sst2",
        "count": 1,
    }
    zero_summary = latency_summary(completed, 0.0)
    record: dict[str, object] = {
        "schema_version": freeze.SCHEMA_VERSION,
        "kind": "p9-guard-worker-evidence",
        "client": {
            **expected_client,
            "worker_index": 0,
            "cpu": 0,
            "engine": str(engine.resolve()),
            "engine_sha256": "a" * 64,
            "pid": pid,
            "affinity": {
                "pid": pid,
                "expected_cpu": 0,
                "tasks": [{"tid": pid, "cpus": [0]}],
            },
        },
        "result": {
            "schema_version": 1,
            "model": "distilbert-sst2",
            "role": "pressure",
            "engine": str(engine.resolve()),
            "execution_environment": {
                "pid": pid,
                "cuda_visible_devices": "MIG-small",
                "mps_active_thread_percentage": 25,
                "cpu_affinity": [0],
            },
            "gpu": {
                "name": "NVIDIA Thor MIG 1g.0gb",
                "multiprocessors": 2,
            },
            "config": {
                "warmup": freeze.WARMUP_REQUESTS,
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
            "release_to_completion": latency_summary(completed, 1.0),
            "gpu_service": latency_summary(completed, 0.5),
            "queue_delay": latency_summary(completed, 0.1),
            "gate_overhead": copy.deepcopy(zero_summary),
            "drain": copy.deepcopy(zero_summary),
            "resume": copy.deepcopy(zero_summary),
            "completed_requests": completed,
            "throughput_per_second": 10.0,
            "measurement_start_monotonic_ns": 1_000_000_000,
            "measurement_end_monotonic_ns": 3_000_000_000,
            "elapsed_seconds": 2.0,
            "deadline_misses": 0,
            "deadline_miss_rate": None,
        },
    }
    return record, expected_client


class FreezeP9GuardTests(unittest.TestCase):
    CAUSAL_LABEL = "test-thermal"

    def causal_fixture(
        self, *, soc012_c: float = 75.0
    ) -> tuple[freeze.TelemetryReplay, dict[str, object]]:
        samples = [
            thermal_sample(1_000_000_000),
            thermal_sample(1_100_000_000, soc012_c=soc012_c),
        ]
        replay = freeze.TelemetryReplay(telemetry_bytes(*samples))
        sample_ns = 1_100_000_000
        reference_ns = 1_110_000_000
        point_telemetry = freeze.aggregate_samples(
            [replay.typed_samples[1]],
            sample_ns - 1,
            sample_ns,
            required_fields=freeze.TELEMETRY_REQUIRED_FIELDS,
            minimum_valid_samples=1,
            require_all_samples_valid=True,
            reference_ns=reference_ns,
            stale_after_ns=300_000_000,
            maximum_valid_gap_ns=300_000_000,
            end_inclusive=True,
        )
        point_telemetry["retention"] = {
            "bounded": False,
            "max_samples": None,
            "dropped_samples": 0,
            "last_dropped_sample_ns": None,
            "earliest_retained_sample_ns": 1_000_000_000,
            "interval_complete": True,
        }
        passed = soc012_c == 75.0
        stored: dict[str, object] = {
            "attempt": 1,
            "passed": passed,
            "boundary": "thermal_measurement_end",
            "boundary_monotonic_ns": 1_000_000_000,
            "cleanup_end_monotonic_ns": 1_020_000_000,
            "qualification_monotonic_ns": reference_ns,
            "sample_monotonic_ns": sample_ns,
            "sample_age_ms": 10.0,
            "stability_sensor": "soc012",
            "stability_value_c": soc012_c,
            "safety_sensor": "tj",
            "safety_value_c": 89.5,
            "target_c": 75.0,
            "tolerance_c": 1.0,
            "telemetry": point_telemetry,
            "failure_reason": (
                None
                if passed
                else f"{self.CAUSAL_LABEL} stability sensor is outside the target band"
            ),
        }
        return replay, stored

    def test_type7_margin_and_upward_rounding(self) -> None:
        values = [float(value) for value in range(10_000)]
        raw, guard = freeze.estimate_guard_ms([value / 10_000 for value in values])
        self.assertAlmostEqual(raw, 0.9989001)
        self.assertEqual(guard, 1.2)
        self.assertEqual(freeze.round_up_ms(1.2000001), 1.3)

    def test_formal_period_and_held_out_coverage_fail_closed(self) -> None:
        freeze.require_held_out_coverage(
            "covered", envelope_ms=3.0, observed_max_ms=2.999
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            freeze.require_held_out_coverage(
                "uncovered", envelope_ms=3.0, observed_max_ms=3.001
            )
        with self.assertRaisesRegex(ValueError, "formal 20 ms"):
            freeze.require_held_out_coverage(
                "too-wide", envelope_ms=20.0, observed_max_ms=1.0
            )

    def test_additive_envelope_sums_per_instance_then_takes_max(self) -> None:
        clients = [
            {
                "placement": "resident-1g",
                "quota_percent": 50,
                "modality": "audio",
                "count": 3,
            },
            {
                "placement": "borrower-2g",
                "quota_percent": 100,
                "modality": "audio",
                "count": 3,
            },
        ]
        guard_values = {
            ("resident-1g", "50", "audio"): 0.7,
            ("borrower-2g", "100", "audio"): 0.9,
        }
        envelope, components = freeze.additive_envelope_ms(clients, guard_values)
        self.assertAlmostEqual(envelope, 2.7)
        self.assertAlmostEqual(components["borrower-2g"], 2.7)
        self.assertAlmostEqual(components["resident-1g"], 2.1)

    def test_trace_replay_and_request_tamper(self) -> None:
        replayed = freeze.replay_critical_trace(trace_bytes())
        self.assertEqual(replayed["samples"], 1_000)
        self.assertGreater(replayed["drain_p999_ms"], 0.2)
        with self.assertRaisesRegex(ValueError, "request sequence"):
            freeze.replay_critical_trace(trace_bytes(tampered_request=True))

    def test_registry_rejects_hardlink_and_identical_copy(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = pathlib.Path(raw_directory)
            original = root / "a.csv"
            hardlink = root / "b.csv"
            copied = root / "c.csv"
            original.write_bytes(trace_bytes())
            os.link(original, hardlink)
            copied.write_bytes(original.read_bytes())

            registry = freeze.EvidenceRegistry(root)
            registry.claim("a.csv", kind="critical-csv", case_id="case", block=1)
            with self.assertRaisesRegex(ValueError, "inode"):
                registry.claim("b.csv", kind="critical-csv", case_id="case", block=2)
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                registry.claim("c.csv", kind="critical-csv", case_id="case", block=3)

    def test_telemetry_duplicate_sample_timestamp_is_rejected(self) -> None:
        sample = telemetry_sample()
        with self.assertRaisesRegex(ValueError, "globally strictly increasing"):
            freeze.TelemetryReplay(telemetry_bytes(sample, sample))

    def test_telemetry_raw_schema_and_health_tamper_are_rejected(self) -> None:
        freeze.TelemetryReplay(
            telemetry_bytes(telemetry_sample(), telemetry_marker())
        )
        cases: list[tuple[str, object, str]] = []

        parsed = telemetry_sample()
        parsed["parsed"] = {}
        cases.append(("parsed", parsed, "differs from raw"))

        raw = telemetry_sample()
        raw["raw"] = "RAM 2/2MB CPU [50%@1000] tj@89.5C VIN 100000mW"
        cases.append(("raw", raw, "differs from raw"))

        extra = telemetry_sample()
        extra["unexpected"] = True
        cases.append(("sample schema", extra, "sample schema"))

        negative_memory = telemetry_sample()
        negative_memory["mem_available_mb"] = -1.0
        cases.append(("negative memory", negative_memory, "MemAvailable"))

        errors = telemetry_sample()
        errors["collection_errors"] = ["mem_available:OSError"]
        cases.append(("collection error", errors, "collection errors"))

        marker = telemetry_marker()
        marker["unexpected"] = True
        cases.append(("marker schema", marker, "marker schema"))

        for label, changed, message in cases:
            with self.subTest(label=label):
                records = (
                    (changed, telemetry_marker())
                    if isinstance(changed, dict)
                    and changed.get("record_type") == "sample"
                    else (telemetry_sample(), changed)
                )
                with self.assertRaisesRegex(ValueError, message):
                    freeze.TelemetryReplay(telemetry_bytes(*records))

    def test_telemetry_requires_newline_and_global_record_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "unterminated"):
            freeze.TelemetryReplay(
                telemetry_bytes(telemetry_sample(), terminated=False)
            )
        with self.assertRaisesRegex(ValueError, "globally strictly increasing"):
            freeze.TelemetryReplay(
                telemetry_bytes(telemetry_sample(200), telemetry_marker(100))
            )

    def test_soc012_start_survives_fast_tj_collapse_but_tj_remains_safety(self) -> None:
        samples = [
            thermal_sample(
                100_000_000 * (index + 1),
                tj_c=80.0 if index == 599 else 89.5,
            )
            for index in range(600)
        ]
        replay = freeze.TelemetryReplay(telemetry_bytes(*samples))
        stored = {
            "samples": 600,
            "window_seconds": 60.0,
            "observed_span_seconds": 59.9,
            "mean_c": 75.0,
            "min_c": 75.0,
            "max_c": 75.0,
            "latest_c": 75.0,
            "slope_c_per_minute": 0.0,
            "maximum_gap_seconds": 0.1,
        }
        self.assertEqual(
            replay.thermal_start(
                60_000_000_000,
                thermal_lock=thermal_lock(),
                stored=stored,
                stored_stable=True,
            ),
            stored,
        )

        unsafe = list(samples)
        unsafe[300] = thermal_sample(30_100_000_000, tj_c=104.0)
        with self.assertRaisesRegex(ValueError, "hard limit"):
            freeze.TelemetryReplay(telemetry_bytes(*unsafe)).thermal_start(
                60_000_000_000,
                thermal_lock=thermal_lock(),
                stored=stored,
                stored_stable=True,
            )

    def test_soc012_start_replay_rejects_telemetry_gap(self) -> None:
        samples = [
            thermal_sample(100_000_000 * (index + 1))
            for index in range(600)
            if index not in {300, 301, 302, 303}
        ]
        stored = {
            "samples": len(samples),
            "window_seconds": 60.0,
            "observed_span_seconds": 59.9,
            "mean_c": 75.0,
            "min_c": 75.0,
            "max_c": 75.0,
            "latest_c": 75.0,
            "slope_c_per_minute": 0.0,
            "maximum_gap_seconds": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "gap larger than 300 ms"):
            freeze.TelemetryReplay(telemetry_bytes(*samples)).thermal_start(
                60_000_000_000,
                thermal_lock=thermal_lock(),
                stored=stored,
                stored_stable=True,
            )

    def test_handoff_replay_is_exact_and_strictly_below_500_ms(self) -> None:
        lock = thermal_lock()
        qualification = {
            "boundary_monotonic_ns": 1_000_000_000,
            "cleanup_end_monotonic_ns": 1_020_000_000,
            "sample_monotonic_ns": 1_100_000_000,
            "qualification_monotonic_ns": 1_110_000_000,
        }
        clocks = {
            "cleanup_end": 1_020_000_000,
            "sample": 1_100_000_000,
            "qualification": 1_110_000_000,
            "qualification_result": 1_120_000_000,
            "block_start": 1_130_000_000,
            "measurement_release": 1_140_000_000,
            "resume_issued": 1_150_000_000,
            "critical_measurement_start": 1_499_999_999,
        }
        stored: dict[str, object] = {
            "boundary": "thermal_measurement_end",
            "boundary_monotonic_ns": 1_000_000_000,
            "maximum_ms": 500.0,
            "strictly_within_bound": True,
        }
        for label, clock in clocks.items():
            stored[f"{label}_monotonic_ns"] = clock
            stored[f"boundary_to_{label}_ms"] = (clock - 1_000_000_000) / 1e6
        self.assertEqual(
            freeze._replay_thermal_handoff(
                stored,
                qualification=qualification,
                qualification_result_ns=1_120_000_000,
                start_ns=1_130_000_000,
                release_ns=1_140_000_000,
                resume_issued_ns=1_150_000_000,
                measurement_start_ns=1_499_999_999,
                thermal_lock=lock,
            ),
            stored,
        )
        invalid = dict(stored)
        invalid["critical_measurement_start_monotonic_ns"] = 1_500_000_000
        invalid["boundary_to_critical_measurement_start_ms"] = 500.0
        invalid["strictly_within_bound"] = False
        self.assertFalse(
            freeze._replay_thermal_handoff(
                invalid,
                qualification=qualification,
                qualification_result_ns=1_120_000_000,
                start_ns=1_130_000_000,
                release_ns=1_140_000_000,
                resume_issued_ns=1_150_000_000,
                measurement_start_ns=1_500_000_000,
                thermal_lock=lock,
            )["strictly_within_bound"]
        )

    def test_qualification_replays_first_postcleanup_causal_sample(self) -> None:
        replay, stored = self.causal_fixture()
        validated = replay.causal_qualification(
            stored,
            label=self.CAUSAL_LABEL,
            reference_ns=1_110_000_000,
            not_before_ns=1_020_000_000,
            first_after_boundary=True,
            thermal_lock=thermal_lock(),
            expected_prefix={
                "attempt": 1,
                "boundary": "thermal_measurement_end",
                "boundary_monotonic_ns": 1_000_000_000,
                "cleanup_end_monotonic_ns": 1_020_000_000,
                "qualification_monotonic_ns": 1_110_000_000,
            },
        )
        self.assertTrue(validated["passed"])
        self.assertEqual(validated["sample_monotonic_ns"], 1_100_000_000)

        failed_replay, failed = self.causal_fixture(soc012_c=80.0)
        validated_failure = failed_replay.causal_qualification(
            failed,
            label=self.CAUSAL_LABEL,
            reference_ns=1_110_000_000,
            not_before_ns=1_020_000_000,
            first_after_boundary=True,
            thermal_lock=thermal_lock(),
            expected_prefix={
                "attempt": 1,
                "boundary": "thermal_measurement_end",
                "boundary_monotonic_ns": 1_000_000_000,
                "cleanup_end_monotonic_ns": 1_020_000_000,
                "qualification_monotonic_ns": 1_110_000_000,
            },
        )
        self.assertFalse(validated_failure["passed"])
        self.assertIn("outside", str(validated_failure["failure_reason"]))
        changed_failure = copy.deepcopy(failed)
        changed_failure["failure_reason"] = "arbitrary failure"
        with self.assertRaisesRegex(ValueError, "failure reason differs"):
            failed_replay.causal_qualification(
                changed_failure,
                label=self.CAUSAL_LABEL,
                reference_ns=1_110_000_000,
                not_before_ns=1_020_000_000,
                first_after_boundary=True,
                thermal_lock=thermal_lock(),
                expected_prefix={
                    "attempt": 1,
                    "boundary": "thermal_measurement_end",
                    "boundary_monotonic_ns": 1_000_000_000,
                    "cleanup_end_monotonic_ns": 1_020_000_000,
                    "qualification_monotonic_ns": 1_110_000_000,
                },
            )

    def test_qualification_replays_no_sample_failure_before_later_pass(self) -> None:
        replay, successful = self.causal_fixture()
        first_label = "test-thermal-attempt-01"
        failed = {
            "attempt": 1,
            "passed": False,
            "boundary": "thermal_measurement_end",
            "boundary_monotonic_ns": 1_000_000_000,
            "cleanup_end_monotonic_ns": 1_020_000_000,
            "qualification_monotonic_ns": 1_050_000_000,
            "sample_monotonic_ns": None,
            "sample_age_ms": None,
            "stability_sensor": "soc012",
            "stability_value_c": None,
            "safety_sensor": "tj",
            "safety_value_c": None,
            "target_c": 75.0,
            "tolerance_c": 1.0,
            "telemetry": None,
            "failure_reason": (
                f"{first_label} observed no causal post-cleanup telemetry sample"
            ),
        }
        replayed_failure = replay.causal_qualification(
            failed,
            label=first_label,
            reference_ns=1_050_000_000,
            not_before_ns=1_020_000_000,
            first_after_boundary=True,
            thermal_lock=thermal_lock(),
            expected_prefix={
                "attempt": 1,
                "boundary": "thermal_measurement_end",
                "boundary_monotonic_ns": 1_000_000_000,
                "cleanup_end_monotonic_ns": 1_020_000_000,
                "qualification_monotonic_ns": 1_050_000_000,
            },
        )
        self.assertFalse(replayed_failure["passed"])
        self.assertIsNone(replayed_failure["sample_monotonic_ns"])

        second_label = "test-thermal-attempt-02"
        successful.update(
            {
                "attempt": 2,
                "boundary_monotonic_ns": 1_060_000_000,
                "cleanup_end_monotonic_ns": 1_070_000_000,
            }
        )
        replayed_success = replay.causal_qualification(
            successful,
            label=second_label,
            reference_ns=1_110_000_000,
            not_before_ns=1_070_000_000,
            first_after_boundary=True,
            thermal_lock=thermal_lock(),
            expected_prefix={
                "attempt": 2,
                "boundary": "thermal_measurement_end",
                "boundary_monotonic_ns": 1_060_000_000,
                "cleanup_end_monotonic_ns": 1_070_000_000,
                "qualification_monotonic_ns": 1_110_000_000,
            },
        )
        self.assertTrue(replayed_success["passed"])
        self.assertEqual(replayed_success["sample_monotonic_ns"], 1_100_000_000)

    def test_qualification_tamper_and_old_lock_are_rejected(self) -> None:
        replay, stored = self.causal_fixture()
        changed = copy.deepcopy(stored)
        changed["sample_monotonic_ns"] = 1_000_000_000
        with self.assertRaisesRegex(ValueError, "select the raw sample"):
            replay.causal_qualification(
                changed,
                label=self.CAUSAL_LABEL,
                reference_ns=1_110_000_000,
                not_before_ns=1_020_000_000,
                first_after_boundary=True,
                thermal_lock=thermal_lock(),
                expected_prefix={
                    "attempt": 1,
                    "boundary": "thermal_measurement_end",
                    "boundary_monotonic_ns": 1_000_000_000,
                    "cleanup_end_monotonic_ns": 1_020_000_000,
                    "qualification_monotonic_ns": 1_110_000_000,
                },
            )
        stale = thermal_lock()
        stale["schema_version"] = 3
        with self.assertRaisesRegex(ValueError, "schema version 4"):
            freeze._validate_guard_thermal_lock(stale)

    def test_first_and_latest_causal_sample_selection_are_distinct(self) -> None:
        replay = freeze.TelemetryReplay(
            telemetry_bytes(
                thermal_sample(1_100_000_000),
                thermal_sample(1_150_000_000),
            )
        )
        first = replay._causal_sample(
            reference_ns=1_160_000_000,
            not_before_ns=1_020_000_000,
            first_after_boundary=True,
        )
        latest = replay._causal_sample(
            reference_ns=1_160_000_000,
            not_before_ns=1_020_000_000,
            first_after_boundary=False,
        )
        assert first is not None and latest is not None
        self.assertEqual(first["monotonic_ns"], 1_100_000_000)
        self.assertEqual(latest["monotonic_ns"], 1_150_000_000)

    def test_retry_accepts_only_first_thermal_success_not_performance(self) -> None:
        failed = {"thermally_valid": False, "drain_max_ms": 0.1}
        valid = {"thermally_valid": True, "drain_max_ms": 19.0}
        self.assertIs(
            freeze._require_first_thermally_valid_attempt([failed, valid], 2),
            valid,
        )
        with self.assertRaisesRegex(ValueError, "first thermally valid"):
            freeze._require_first_thermally_valid_attempt([valid, failed], 2)

    def test_old_guard_lock_schema_v2_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid schema"):
            freeze.verify_lock(
                {"schema_version": 2, "kind": freeze.LOCK_KIND}
            )
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "invalid schema"
        ):
            freeze._validate_profile_provenance(
                {"schema_version": 2, "kind": freeze.PROFILE_KIND},
                thermal_lock(),
                pathlib.Path(directory),
            )

    def test_active_precondition_replays_three_spaced_raw_endpoints(self) -> None:
        label = "pre-p9-guard-case-block-01-attempt-01-thermal-01"
        samples = [
            thermal_sample(100_000_000 * index)
            for index in range(1, 621)
        ]
        sample_replay = freeze.TelemetryReplay(telemetry_bytes(*samples))
        checks: list[dict[str, object]] = []
        check_markers: list[dict[str, object]] = []
        for index, sample_ns in enumerate(
            (60_000_000_000, 61_000_000_000, 62_000_000_000)
        ):
            window, _telemetry, passed, _failure = (
                sample_replay._thermal_window_evidence(
                    sample_ns,
                    not_before_ns=2,
                    thermal_lock=thermal_lock(),
                    label=label,
                )
            )
            self.assertTrue(passed)
            metadata: dict[str, object] = {
                "label": label,
                "index": index,
                "sample_monotonic_ns": sample_ns,
                "passed": True,
                "consecutive_passes": index + 1,
                "window": window,
            }
            checks.append(metadata)
            check_markers.append(
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": sample_ns + 1,
                    "name": "thermal_active_stability_check",
                    "metadata": metadata,
                }
            )
        records: list[dict[str, object]] = [
            {
                "schema_version": 1,
                "record_type": "marker",
                "monotonic_ns": 1,
                "name": "thermal_prepare",
                "metadata": {"label": label},
            },
            {
                "schema_version": 1,
                "record_type": "marker",
                "monotonic_ns": 2,
                "name": "thermal_start",
                "metadata": {"label": label},
            },
            *samples,
            *check_markers,
            {
                "schema_version": 1,
                "record_type": "marker",
                "monotonic_ns": 62_000_000_002,
                "name": "thermal_measurement_end",
                "metadata": {
                    "label": label,
                    "boundary_sample_monotonic_ns": 62_000_000_000,
                    "consecutive_passes": 3,
                    "window": checks[-1]["window"],
                },
            },
            {
                "schema_version": 1,
                "record_type": "marker",
                "monotonic_ns": 62_050_000_000,
                "name": "thermal_end",
                "metadata": {"label": label, "successful": True},
            },
        ]
        records.sort(key=lambda record: int(record["monotonic_ns"]))
        replay = freeze.TelemetryReplay(telemetry_bytes(*records))
        stored = {
            "label": label,
            "duration_seconds": 62.0,
            "measurement_start_monotonic_ns": 2,
            "measurement_end_monotonic_ns": 62_000_000_002,
            "cleanup_end_monotonic_ns": 62_050_000_000,
            "target_c": 75.0,
            "stability_sensor": "soc012",
            "safety_sensor": "tj",
            "last_window": checks[-1]["window"],
            "pressure_rate_per_second": 1.0,
            "telemetry": {
                "total_samples": 620,
                "valid_samples": 620,
                "health": {
                    "healthy": True,
                    "required_fields": list(freeze.TELEMETRY_REQUIRED_FIELDS),
                },
                "temperatures_c": {"tj": {"max": 89.5}},
            },
            "active_stability_checks": checks,
            "active_stable_endpoints": 3,
            "active_stable_spacing_seconds": 1.0,
            "termination_reason": "active-stability-endpoints",
        }
        validated, _times = freeze._validate_thermal_precondition(
            replay,
            stored,
            label=label,
            thermal_lock=thermal_lock(),
        )
        self.assertEqual(len(validated["active_stability_checks"]), 3)
        tampered = copy.deepcopy(stored)
        tampered["active_stability_checks"][1]["consecutive_passes"] = 1
        with self.assertRaisesRegex(ValueError, "differ from raw markers"):
            freeze._validate_thermal_precondition(
                replay,
                tampered,
                label=label,
                thermal_lock=thermal_lock(),
            )
        stale = copy.deepcopy(stored)
        stale["stability_checks"] = []
        with self.assertRaisesRegex(ValueError, "active thermal-precondition"):
            freeze._validate_thermal_precondition(
                replay,
                stale,
                label=label,
                thermal_lock=thermal_lock(),
            )

    def test_worker_measurement_and_summary_evidence_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            engine = pathlib.Path(raw_directory) / "distilbert-sst2.engine"
            record, expected_client = worker_evidence(engine)

            def validate(value: dict[str, object]) -> int:
                return freeze._validate_worker_json(
                    value,
                    expected_client=expected_client,
                    expected_index=0,
                    expected_cpu=0,
                    expected_engine=engine,
                    expected_engine_sha256="a" * 64,
                    expected_uuid="MIG-small",
                    release_marker_ns=900_000_000,
                    critical_end_ns=2_000_000_000,
                    result_marker_ns=4_000_000_000,
                )

            self.assertEqual(validate(record), 321)

            cases: list[tuple[str, dict[str, object], str]] = []
            elapsed = copy.deepcopy(record)
            elapsed["result"]["elapsed_seconds"] = 2.1  # type: ignore[index]
            cases.append(("elapsed", elapsed, "elapsed_seconds"))

            throughput = copy.deepcopy(record)
            throughput["result"]["throughput_per_second"] = 9.0  # type: ignore[index]
            cases.append(("throughput", throughput, "throughput_per_second"))

            count = copy.deepcopy(record)
            count["result"]["gpu_service"]["count"] = 19  # type: ignore[index]
            cases.append(("summary count", count, "count differs"))

            percentile = copy.deepcopy(record)
            percentile["result"]["queue_delay"]["p99_ms"] = 0.01  # type: ignore[index]
            cases.append(("summary ordering", percentile, "inconsistent"))

            gate = copy.deepcopy(record)
            gate["result"]["drain"]["max_ms"] = 0.01  # type: ignore[index]
            cases.append(("gate summary", gate, "must be zero"))

            for label, changed, message in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, message):
                        validate(changed)

    def test_worker_measurement_window_is_bound_to_block_markers(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            engine = pathlib.Path(raw_directory) / "distilbert-sst2.engine"
            record, expected_client = worker_evidence(engine)

            def validate(value: dict[str, object]) -> None:
                freeze._validate_worker_json(
                    value,
                    expected_client=expected_client,
                    expected_index=0,
                    expected_cpu=0,
                    expected_engine=engine,
                    expected_engine_sha256="a" * 64,
                    expected_uuid="MIG-small",
                    release_marker_ns=900_000_000,
                    critical_end_ns=2_000_000_000,
                    result_marker_ns=4_000_000_000,
                )

            def clocks(start_ns: int, end_ns: int) -> dict[str, object]:
                changed = copy.deepcopy(record)
                result = changed["result"]
                assert isinstance(result, dict)
                elapsed_seconds = (end_ns - start_ns) / 1_000_000_000.0
                result["measurement_start_monotonic_ns"] = start_ns
                result["measurement_end_monotonic_ns"] = end_ns
                result["elapsed_seconds"] = elapsed_seconds
                result["throughput_per_second"] = 20 / elapsed_seconds
                return changed

            for label, changed in (
                ("before release", clocks(800_000_000, 3_000_000_000)),
                ("after critical", clocks(2_000_000_000, 3_000_000_000)),
                ("ends before critical", clocks(1_000_000_000, 1_900_000_000)),
                ("ends after result", clocks(1_000_000_000, 4_100_000_000)),
            ):
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, "outside"):
                        validate(changed)

    def test_artifact_hash_and_engine_tag_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = pathlib.Path(raw_directory)
            benchmark = root / "jdg-trt-bench"
            benchmark.write_bytes(b"benchmark")
            artifacts = {
                "benchmark": {
                    "path": str(benchmark.resolve()),
                    "sha256": freeze.file_sha256(benchmark),
                }
            }
            for name, path in freeze.CURRENT_FILES.items():
                artifacts[name] = {
                    "path": str(path.resolve()),
                    "sha256": freeze.file_sha256(path),
                }
            engine_root = root / "engines"
            for case in freeze.expected_single_cases():
                client = case["clients"][0]
                prefix = (
                    "mig-1g"
                    if client["placement"] == "resident-1g"
                    else "mig-2g"
                )
                path = (
                    engine_root
                    / f"{prefix}-q{client['quota_percent']}"
                    / f"{client['model']}.engine"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(freeze._engine_artifact_key(client).encode())
                artifacts[freeze._engine_artifact_key(client)] = {
                    "path": str(path.resolve()),
                    "sha256": freeze.file_sha256(path),
                }
            critical = engine_root / "mig-2g" / "resnet50-v2.engine"
            critical.parent.mkdir(parents=True, exist_ok=True)
            critical.write_bytes(b"critical")
            artifacts["engine:critical:2g:resnet50-v2"] = {
                "path": str(critical.resolve()),
                "sha256": freeze.file_sha256(critical),
            }
            freeze.validate_artifacts(artifacts)
            critical.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                freeze.validate_artifacts(artifacts)

    def test_freezer_protocol_matches_producer_protocol(self) -> None:
        from runtime import profile_p9_guard as producer

        self.assertEqual(freeze.expected_protocol(), producer.protocol_json())


if __name__ == "__main__":
    unittest.main()
