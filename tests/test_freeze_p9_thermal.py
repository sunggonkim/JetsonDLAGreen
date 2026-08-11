#!/usr/bin/env python3
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_p9_thermal", ROOT / "analysis" / "freeze_p9_thermal.py"
)
assert SPEC is not None and SPEC.loader is not None
THERMAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = THERMAL
SPEC.loader.exec_module(THERMAL)


class FreezeP9ThermalTest(unittest.TestCase):
    @staticmethod
    def write_snapshots(directory: pathlib.Path, fan_mode: str = "disabled") -> None:
        (directory / "jetson-clocks.txt").write_text(
            f"FAN Dynamic Speed Control={fan_mode} hwmon0_pwm1=255\n"
            "gpu-gpc-0 MinFreq=1575000000 MaxFreq=1575000000\n"
            "EMC MinFreq=4266000000 MaxFreq=4266000000\n",
            encoding="utf-8",
        )
        (directory / "nvpmodel.txt").write_text(
            "NV Power Mode: MAXN\n", encoding="utf-8"
        )

    @staticmethod
    def _sample(
        timestamp_ns: int,
        stability_temperature_c: float,
        safety_temperature_c: float = 89.5,
    ) -> dict:
        raw = (
            "RAM 1/2MB CPU [50%@1000] "
            f"soc012@{stability_temperature_c:.6f}C "
            f"tj@{safety_temperature_c:.6f}C VIN 100000mW"
        )
        return {
            "schema_version": 1,
            "record_type": "sample",
            "monotonic_ns": timestamp_ns,
            "raw": raw,
            "mem_available_mb": 1000.0,
            "parsed": THERMAL.parse_tegrastats_line(raw).to_dict(),
            "collection_errors": [],
        }

    @staticmethod
    def _checkpoint_passed(window: dict | None) -> bool:
        return bool(
            window is not None
            and window["samples"] >= 1440
            and window["observed_span_seconds"] >= 178.2
            and window["maximum_gap_seconds"] <= 0.3
            and abs(window["slope_c_per_minute"]) <= 0.2
        )

    def write_telemetry(
        self,
        directory: pathlib.Path,
        *,
        extended: bool = False,
        continued: bool = False,
        unstable: bool = False,
    ) -> dict:
        start_ns = 1_000_000_000
        final_schedule = 750 if extended else 630 if continued else 600

        def stability_temperature(elapsed_seconds: float) -> float:
            if unstable:
                return 74.0 + 0.3 * elapsed_seconds / 60.0
            if not extended:
                return 74.8
            return 74.0 + 0.3 * min(elapsed_seconds, 600.0) / 60.0

        samples = [
            self._sample(
                start_ns + index * 100_000_000,
                stability_temperature(index / 10.0),
            )
            for index in range(1, final_schedule * 10 + 1)
        ]
        records = [
            {
                "schema_version": 1,
                "record_type": "marker",
                "monotonic_ns": start_ns - 100_000_000,
                "name": "thermal_prepare",
                "metadata": {"label": "thermal-pilot"},
            },
            {
                "schema_version": 1,
                "record_type": "marker",
                "monotonic_ns": start_ns,
                "name": "thermal_start",
                "metadata": {"label": "thermal-pilot"},
            },
            *samples,
        ]
        stability_checks = []
        consecutive_passes = 0
        final_boundary_ns = -1
        final_result_ns = -1
        for checkpoint_index, scheduled in enumerate(
            range(30, final_schedule + 1, 30)
        ):
            boundary_ns = start_ns + scheduled * 1_000_000_000 + 50_000_000
            window = THERMAL._thermal_window_from_raw(
                samples,
                pilot_start_ns=start_ns,
                reference_ns=boundary_ns,
                window_seconds=180.0,
                sensor=THERMAL.STABILITY_SENSOR,
            )
            passed = self._checkpoint_passed(window)
            consecutive_passes = consecutive_passes + 1 if passed else 0
            boundary_metadata = {
                "label": "thermal-pilot",
                "checkpoint_index": checkpoint_index,
                "scheduled_elapsed_seconds": float(scheduled),
            }
            check_metadata = {
                **boundary_metadata,
                "actual_elapsed_seconds": scheduled + 0.05,
                "checkpoint_monotonic_ns": boundary_ns,
                "passed": passed,
                "consecutive_passes": consecutive_passes,
                "window": window,
            }
            result_ns = boundary_ns + 1_000_000
            records.extend(
                [
                    {
                        "schema_version": 1,
                        "record_type": "marker",
                        "monotonic_ns": boundary_ns,
                        "name": "thermal_stability_boundary",
                        "metadata": boundary_metadata,
                    },
                    {
                        "schema_version": 1,
                        "record_type": "marker",
                        "monotonic_ns": result_ns,
                        "name": "thermal_stability_check",
                        "metadata": check_metadata,
                    },
                ]
            )
            stability_checks.append(check_metadata)
            final_boundary_ns = boundary_ns
            final_result_ns = result_ns
        if not unstable:
            self.assertGreaterEqual(consecutive_passes, 3)
            self.assertTrue(stability_checks[-1]["passed"])
        measurement_end_ns = final_result_ns + 1_000_000
        cleanup_end_ns = final_result_ns + 2_000_000
        final_metadata = stability_checks[-1]
        measurement_metadata = {
            name: final_metadata[name]
            for name in THERMAL.THERMAL_MEASUREMENT_END_METADATA_KEYS
        }
        records.extend(
            [
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": measurement_end_ns,
                    "name": "thermal_measurement_end",
                    "metadata": measurement_metadata,
                },
                {
                    "schema_version": 1,
                    "record_type": "marker",
                    "monotonic_ns": cleanup_end_ns,
                    "name": "thermal_end",
                    "metadata": {"label": "thermal-pilot", "successful": True},
                },
            ]
        )
        records.sort(key=lambda record: record["monotonic_ns"])
        (directory / "telemetry.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        pilot_temperatures = [
            record["parsed"]["temperatures_c"]["tj"] for record in samples
        ]
        pilot_stability_temperatures = [
            record["parsed"]["temperatures_c"]["soc012"]
            for record in samples
        ]
        return {
            "duration_seconds": (measurement_end_ns - start_ns) / 1e9,
            "label": "thermal-pilot",
            "measurement_start_monotonic_ns": start_ns,
            "measurement_end_monotonic_ns": measurement_end_ns,
            "cleanup_end_monotonic_ns": cleanup_end_ns,
            "stability_sensor": "soc012",
            "safety_sensor": "tj",
            "last_window": stability_checks[-1]["window"],
            "stability_checkpoint_seconds": 30.0,
            "required_consecutive_stable_checkpoints": 3,
            "stability_checks": stability_checks,
            "maximum_gap_seconds": 0.1,
            "termination_reason": "stable-checkpoints",
            "telemetry": {
                "health": {"healthy": True},
                "temperatures_c": {
                    "soc012": {"max": max(pilot_stability_temperatures)},
                    "tj": {"max": max(pilot_temperatures)},
                },
            },
        }

    @staticmethod
    def summary(pilot: dict) -> dict:
        return {
            "schema_version": 4,
            "hardware": {"gpu_product_name": "NVIDIA Thor"},
            "mig": {"critical_uuid": "big", "resident_uuid": "small"},
            "artifacts": {
                "benchmark_sha256": "bench",
                "engines_sha256": {},
                "implementation_sha256": {},
            },
            "config": {
                "cpu_affinity": {
                    "critical": [12],
                    "pressure": list(range(11)),
                    "mps": [11],
                    "telemetry": [13],
                },
                "thermal_pilot_seconds": 600.0,
                "thermal_pilot_maximum_seconds": 900.0,
                "thermal_window_seconds": 180.0,
                "thermal_max_slope_c_per_minute": 0.2,
                "calibration_repeats": 1,
                "samples_per_epoch": 160,
                "warmup": 100,
                "burst_size": 8,
                "period_ms": 20.0,
                "thermal_timeout_seconds": 900.0,
                "thermal_stability_checkpoint_seconds": 30.0,
                "thermal_stability_checkpoint_max_lateness_seconds": 1.0,
                "thermal_required_stable_checkpoints": 3,
                "thermal_stability_sensor": "soc012",
                "thermal_safety_sensor": "tj",
                "thermal_handoff_max_ms": 500.0,
                "thermal_handoff_boundary": "thermal_measurement_end",
                "thermal_qualification_max_attempts": 3,
                "thermal_active_stable_endpoints": 3,
                "thermal_active_stable_spacing_seconds": 1.0,
                "thermal_hard_limit_c": 104.0,
                "platform_thermal_hard_limit_c": 104.0,
                "tegrastats_requested_interval_ms": 75.0,
                "telemetry_interval_ms": 100.0,
                "telemetry_required_fraction": 0.8,
                "telemetry_required_fields": list(THERMAL.THERMAL_REQUIRED_FIELDS),
                "telemetry_stale_after_ms": 300.0,
                "telemetry_max_gap_ms": 300.0,
            },
            "thermal_pilot": pilot,
            "policies": [],
        }

    def fixture(
        self,
        directory: pathlib.Path,
        *,
        extended: bool = False,
        continued: bool = False,
        unstable: bool = False,
        fan_mode: str = "disabled",
    ) -> tuple[dict, pathlib.Path]:
        self.write_snapshots(directory, fan_mode=fan_mode)
        summary = self.summary(
            self.write_telemetry(
                directory,
                extended=extended,
                continued=continued,
                unstable=unstable,
            )
        )
        path = directory / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        return summary, path

    @staticmethod
    def telemetry_records(path: pathlib.Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def write_records(path: pathlib.Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    @staticmethod
    def resynchronize_checkpoints(summary: dict, records: list[dict]) -> None:
        pilot = summary["thermal_pilot"]
        start_ns = pilot["measurement_start_monotonic_ns"]
        samples = [
            record for record in records if record["record_type"] == "sample"
        ]
        boundaries = [
            record
            for record in records
            if record.get("name") == "thermal_stability_boundary"
        ]
        checks = [
            record
            for record in records
            if record.get("name") == "thermal_stability_check"
        ]
        consecutive = 0
        reported = []
        for boundary, check in zip(boundaries, checks, strict=True):
            checkpoint_ns = boundary["monotonic_ns"]
            stability_window = THERMAL._thermal_window_from_raw(
                samples,
                pilot_start_ns=start_ns,
                reference_ns=checkpoint_ns,
                window_seconds=180.0,
                sensor=THERMAL.STABILITY_SENSOR,
            )
            safety_window = THERMAL._thermal_window_from_raw(
                samples,
                pilot_start_ns=start_ns,
                reference_ns=checkpoint_ns,
                window_seconds=180.0,
                sensor=THERMAL.SAFETY_SENSOR,
            )
            passed = THERMAL._checkpoint_passes(
                samples,
                pilot_start_ns=start_ns,
                checkpoint_ns=checkpoint_ns,
                stability_window=stability_window,
                safety_window=safety_window,
                window_seconds=180.0,
                evaluation_interval_ms=100.0,
                required_fraction=0.8,
                hard_limit_c=104.0,
                maximum_slope_c_per_minute=0.2,
            )
            consecutive = consecutive + 1 if passed else 0
            check["metadata"]["window"] = stability_window
            check["metadata"]["passed"] = passed
            check["metadata"]["consecutive_passes"] = consecutive
            reported.append(json.loads(json.dumps(check["metadata"])))
        pilot["stability_checks"] = reported
        pilot["last_window"] = reported[-1]["window"]
        measurement_end = next(
            record
            for record in records
            if record.get("name") == "thermal_measurement_end"
        )
        measurement_end["metadata"] = {
            name: reported[-1][name]
            for name in THERMAL.THERMAL_MEASUREMENT_END_METADATA_KEYS
        }

    def test_stable_pilot_creates_target_and_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, summary_path = self.fixture(directory)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                lock = THERMAL.build_lock(summary, summary_path)
                THERMAL.verify_lock(lock)
        self.assertEqual(lock["schema_version"], 4)
        self.assertEqual(lock["stability_sensor"], "soc012")
        self.assertEqual(lock["safety_sensor"], "tj")
        self.assertEqual(lock["thermal_handoff_max_ms"], 500.0)
        self.assertEqual(
            lock["thermal_handoff_boundary"], "thermal_measurement_end"
        )
        self.assertEqual(lock["thermal_qualification_max_attempts"], 3)
        self.assertEqual(lock["thermal_active_stable_endpoints"], 3)
        self.assertEqual(
            lock["thermal_active_stable_spacing_seconds"], 1.0
        )
        self.assertNotIn("thermal_qualification_dwell_seconds", lock)
        self.assertEqual(
            lock["thermal_handoff_rationale"],
            THERMAL.THERMAL_HANDOFF_RATIONALE,
        )
        self.assertEqual(lock["target_c"], 74.8)
        self.assertEqual(
            lock["raw_thermal_evidence"]["window_sensor"], "soc012"
        )
        self.assertAlmostEqual(
            lock["raw_thermal_evidence"]["window_mean_c"], 74.8
        )
        self.assertEqual(lock["raw_thermal_evidence"]["pilot_max_sensor"], "tj")
        self.assertAlmostEqual(
            lock["raw_thermal_evidence"]["pilot_max_c"], 89.5
        )
        self.assertEqual(lock["stability_checkpoint_seconds"], 30.0)
        self.assertEqual(len(lock["pilot_final_stability_confirmation"]), 3)
        for confirmation in lock["pilot_final_stability_confirmation"]:
            self.assertEqual(confirmation["stability_sensor"], "soc012")
            self.assertEqual(confirmation["safety_sensor"], "tj")
            self.assertEqual(
                confirmation["raw_stability_window"],
                confirmation["metadata"]["window"],
            )
            self.assertLess(
                confirmation["raw_safety_window"]["max_c"],
                lock["hard_limit_c"],
            )

    def test_extended_pilot_stops_at_first_three_pass_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory, extended=True)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                lock = THERMAL.build_lock(summary, path)
        schedules = [
            item["metadata"]["scheduled_elapsed_seconds"]
            for item in lock["pilot_final_stability_confirmation"]
        ]
        self.assertEqual(schedules, [690.0, 720.0, 750.0])
        self.assertGreater(summary["thermal_pilot"]["duration_seconds"], 750.0)

    def test_old_and_sensor_tampered_locks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                lock = THERMAL.build_lock(summary, path)
                cases = {
                    "old-schema": ("schema_version", 3),
                    "stability-sensor": ("stability_sensor", "tj"),
                    "safety-sensor": ("safety_sensor", "soc012"),
                    "handoff": ("thermal_handoff_max_ms", 501.0),
                    "handoff-boundary": (
                        "thermal_handoff_boundary",
                        "thermal_start_qualification",
                    ),
                    "qualification-attempts": (
                        "thermal_qualification_max_attempts",
                        4,
                    ),
                    "active-endpoints": (
                        "thermal_active_stable_endpoints",
                        2,
                    ),
                    "active-spacing": (
                        "thermal_active_stable_spacing_seconds",
                        0.5,
                    ),
                }
                for name, (field, value) in cases.items():
                    with self.subTest(name=name):
                        tampered = json.loads(json.dumps(lock))
                        tampered[field] = value
                        with self.assertRaises(ValueError):
                            THERMAL.verify_lock(tampered)
                tampered = json.loads(json.dumps(lock))
                tampered["state_dependencies"]["thermal_target_c"] = "tj"
                with self.assertRaisesRegex(ValueError, "sensor dependencies"):
                    THERMAL.verify_lock(tampered)
                tampered = json.loads(json.dumps(lock))
                tampered["thermal_qualification_dwell_seconds"] = 1.0
                with self.assertRaises(ValueError):
                    THERMAL.verify_lock(tampered)
                tampered = json.loads(json.dumps(lock))
                tampered["telemetry_required_fields"].remove(
                    "temperature:soc012"
                )
                with self.assertRaisesRegex(ValueError, "sensor dependencies"):
                    THERMAL.verify_lock(tampered)

    def test_sensor_and_required_field_config_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            cases = {
                "stability-sensor": ("thermal_stability_sensor", "tj"),
                "safety-sensor": ("thermal_safety_sensor", "soc012"),
                "handoff": ("thermal_handoff_max_ms", 501.0),
                "handoff-boundary": (
                    "thermal_handoff_boundary",
                    "thermal_start_qualification",
                ),
                "qualification-attempts": (
                    "thermal_qualification_max_attempts",
                    4,
                ),
                "active-endpoints": (
                    "thermal_active_stable_endpoints",
                    2,
                ),
                "active-spacing": (
                    "thermal_active_stable_spacing_seconds",
                    0.5,
                ),
                "required-fields": (
                    "telemetry_required_fields",
                    [
                        field
                        for field in THERMAL.THERMAL_REQUIRED_FIELDS
                        if field != "temperature:soc012"
                    ],
                ),
            }
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                for name, (field, value) in cases.items():
                    with self.subTest(name=name):
                        tampered = json.loads(json.dumps(summary))
                        tampered["config"][field] = value
                        with self.assertRaisesRegex(ValueError, "frozen design"):
                            THERMAL.build_lock(tampered, path)
                tampered = json.loads(json.dumps(summary))
                tampered["config"]["thermal_qualification_dwell_seconds"] = 1.0
                with self.assertRaisesRegex(ValueError, "frozen design"):
                    THERMAL.build_lock(tampered, path)

    def test_pilot_sensor_role_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            summary["thermal_pilot"]["stability_sensor"] = "tj"
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "sensor roles"):
                    THERMAL.build_lock(summary, path)

    def test_raw_soc012_instability_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory, unstable=True)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "stable slope"):
                    THERMAL.build_lock(summary, path)

    def test_raw_tj_hard_limit_spike_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            sample = next(
                record for record in records if record["record_type"] == "sample"
            )
            sample["raw"] = sample["raw"].replace(
                "tj@89.500000C", "tj@104.000000C"
            )
            sample["parsed"] = THERMAL.parse_tegrastats_line(sample["raw"]).to_dict()
            self.resynchronize_checkpoints(summary, records)
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "hard safety limit"):
                    THERMAL.build_lock(summary, path)

    def test_raw_tj_missing_sample_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            sample = next(
                record for record in records if record["record_type"] == "sample"
            )
            sample["raw"] = sample["raw"].replace("tj@89.500000C ", "")
            sample["parsed"] = THERMAL.parse_tegrastats_line(sample["raw"]).to_dict()
            self.resynchronize_checkpoints(summary, records)
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "incomplete telemetry sample"):
                    THERMAL.build_lock(summary, path)

    def test_raw_tj_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            samples = [
                record
                for record in records
                if record["record_type"] == "sample"
            ]
            for sample in samples[:4]:
                sample["raw"] = sample["raw"].replace("tj@89.500000C ", "")
                sample["parsed"] = THERMAL.parse_tegrastats_line(
                    sample["raw"]
                ).to_dict()
            self.resynchronize_checkpoints(summary, records)
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "gap larger than 300 ms"):
                    THERMAL.build_lock(summary, path)

    def test_continuing_after_first_eligible_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory, continued=True)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "continued after"):
                    THERMAL.build_lock(summary, path)

    def test_modified_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                lock = THERMAL.build_lock(summary, path)
                lock["target_c"] = 1.0
                with self.assertRaises(ValueError):
                    THERMAL.verify_lock(lock)

    def test_summary_window_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            summary["thermal_pilot"]["last_window"]["slope_c_per_minute"] = 0.3
            path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaises(ValueError):
                    THERMAL.build_lock(summary, path)

    def test_nan_duration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            summary["thermal_pilot"]["duration_seconds"] = math.nan
            path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaises(ValueError):
                    THERMAL.build_lock(summary, path)

    def test_dynamic_fan_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory, fan_mode="enabled")
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaises(ValueError):
                    THERMAL.build_lock(summary, path)

    def test_duplicate_telemetry_timestamps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            first_two = [
                record for record in records if record["record_type"] == "sample"
            ][:2]
            first_two[1]["monotonic_ns"] = first_two[0]["monotonic_ns"]
            records.sort(key=lambda record: record["monotonic_ns"])
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    THERMAL.build_lock(summary, path)

    def test_raw_parsed_temperature_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            sample = next(
                record for record in records if record["record_type"] == "sample"
            )
            sample["raw"] = sample["raw"].replace("tj@89.500000C", "tj@99C")
            sample["parsed"]["temperatures_c"]["tj"] = 50.0
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "differs from raw"):
                    THERMAL.build_lock(summary, path)

    def test_late_checkpoint_is_rejected_even_when_summary_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            boundary = next(
                record
                for record in records
                if record["record_type"] == "marker"
                and record["name"] == "thermal_stability_boundary"
            )
            boundary["metadata"]["scheduled_elapsed_seconds"] = 1.0
            check = next(
                record
                for record in records
                if record["record_type"] == "marker"
                and record["name"] == "thermal_stability_check"
            )
            check["metadata"]["scheduled_elapsed_seconds"] = 1.0
            summary["thermal_pilot"]["stability_checks"][0][
                "scheduled_elapsed_seconds"
            ] = 1.0
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "schedule"):
                    THERMAL.build_lock(summary, path)

    def test_consecutive_pass_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            checks = [
                record
                for record in records
                if record.get("name") == "thermal_stability_check"
            ]
            checks[-2]["metadata"]["consecutive_passes"] += 1
            summary["thermal_pilot"]["stability_checks"][-2][
                "consecutive_passes"
            ] += 1
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "consecutive-pass"):
                    THERMAL.build_lock(summary, path)

    def test_checkpoint_window_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            check = next(
                record
                for record in records
                if record.get("name") == "thermal_stability_check"
            )
            check["metadata"]["window"]["mean_c"] += 1.0
            summary["thermal_pilot"]["stability_checks"][0]["window"][
                "mean_c"
            ] += 1.0
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "differs from raw"):
                    THERMAL.build_lock(summary, path)

    def test_boundary_clock_binding_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            check = next(
                record
                for record in records
                if record.get("name") == "thermal_stability_check"
            )
            check["metadata"]["checkpoint_monotonic_ns"] += 1
            summary["thermal_pilot"]["stability_checks"][0][
                "checkpoint_monotonic_ns"
            ] += 1
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "boundary clock"):
                    THERMAL.build_lock(summary, path)

    def test_measurement_end_binding_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            measurement_end = next(
                record
                for record in records
                if record.get("name") == "thermal_measurement_end"
            )
            measurement_end["metadata"]["checkpoint_index"] -= 1
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "chosen checkpoint"):
                    THERMAL.build_lock(summary, path)

    def test_summary_stability_metadata_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            summary["thermal_pilot"]["stability_checks"][0][
                "actual_elapsed_seconds"
            ] += 0.001
            path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "summary metadata"):
                    THERMAL.build_lock(summary, path)

    def test_unsuccessful_thermal_end_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            summary, path = self.fixture(directory)
            telemetry_path = directory / "telemetry.jsonl"
            records = self.telemetry_records(telemetry_path)
            end = next(
                record for record in records if record.get("name") == "thermal_end"
            )
            end["metadata"]["successful"] = False
            self.write_records(telemetry_path, records)
            with mock.patch.object(THERMAL, "code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "successful"):
                    THERMAL.build_lock(summary, path)

    def test_strict_jsonl_loader_rejects_invalid_provenance_fields(self) -> None:
        raw = "RAM 1/2MB CPU [50%@1000] tj@89.5C VIN 100000mW"
        base_records = [
            {
                "schema_version": 1,
                "record_type": "sample",
                "monotonic_ns": 1,
                "raw": raw,
                "parsed": THERMAL.parse_tegrastats_line(raw).to_dict(),
                "mem_available_mb": 1000.0,
                "collection_errors": [],
            },
            {
                "schema_version": 1,
                "record_type": "marker",
                "monotonic_ns": 2,
                "name": "done",
                "metadata": {},
            },
        ]
        cases = (
            "schema_version",
            "sample_key",
            "marker_key",
            "mem_available",
            "collection_errors",
            "unterminated",
        )
        with tempfile.TemporaryDirectory() as directory_name:
            path = pathlib.Path(directory_name) / "telemetry.jsonl"
            for case in cases:
                with self.subTest(case=case):
                    records = json.loads(json.dumps(base_records))
                    if case == "schema_version":
                        records[0]["schema_version"] = 2
                    elif case == "sample_key":
                        records[0]["unexpected"] = True
                    elif case == "marker_key":
                        records[1]["unexpected"] = True
                    elif case == "mem_available":
                        records[0]["mem_available_mb"] = math.nan
                    elif case == "collection_errors":
                        records[0]["collection_errors"] = ["read failed"]
                    contents = "".join(
                        json.dumps(record) + "\n" for record in records
                    )
                    if case == "unterminated":
                        contents = contents.rstrip("\n")
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        THERMAL.load_telemetry_jsonl(path)


if __name__ == "__main__":
    unittest.main()
