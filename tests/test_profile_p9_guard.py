#!/usr/bin/env python3
"""Focused unit tests for the hardware-independent P9 guard producer API."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import profile_p9_guard as profile


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


class ProfileP9GuardTests(unittest.TestCase):
    class FakeProcess:
        def __init__(self, pid: int, returncode: int | None) -> None:
            self.pid = pid
            self._returncode = returncode

        def poll(self) -> int | None:
            return self._returncode

        @property
        def returncode(self) -> int | None:
            return self._returncode

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            del timeout
            if self._returncode is None:
                self._returncode = 0
            return "{}", ""

    @staticmethod
    def campaign_args(root: pathlib.Path) -> SimpleNamespace:
        return SimpleNamespace(
            output=root / "summary.json",
            thermal_lock=root / "thermal-lock.json",
            mig_env=root / "mig.env",
            resident_mps_pipe=root / "resident-pipe",
            resident_mps_log=root / "resident-log",
            big_mps_pipe=root / "big-pipe",
            big_mps_log=root / "big-log",
            bench=root / "bench",
            engine_root=root / "engines",
            telemetry_log=root / "tegrastats.log",
        )

    def test_fixed_case_matrix(self) -> None:
        singles = profile.single_client_cases()
        held_out = profile.held_out_cases()
        self.assertEqual(len(singles), 8)
        self.assertEqual(len(held_out), 3)
        self.assertEqual(
            [case.case_id for case in singles],
            [
                "resident-1g-q25-language",
                "resident-1g-q25-audio",
                "resident-1g-q50-language",
                "resident-1g-q50-audio",
                "resident-1g-q100-language",
                "resident-1g-q100-audio",
                "borrower-2g-q100-language",
                "borrower-2g-q100-audio",
            ],
        )
        self.assertEqual([len(case.expanded_clients) for case in held_out], [6, 6, 6])

    def test_protocol_is_not_runtime_relaxable(self) -> None:
        protocol = profile.protocol_json()
        self.assertEqual(protocol["mode"], "formal")
        self.assertTrue(protocol["formal"])
        self.assertEqual(protocol["blocks"], 10)
        self.assertEqual(protocol["events_per_block"], 1_000)
        self.assertEqual(protocol["period_ms"], 40.0)
        self.assertEqual(protocol["profiling_guard_ms"], 20.0)
        self.assertEqual(protocol["formal_period_ms"], 20.0)
        self.assertEqual(protocol["percentile"], 0.999)
        self.assertEqual(protocol["margin"], 1.20)
        self.assertEqual(protocol["rounding_ms"], 0.1)
        self.assertEqual(
            protocol["thermal_precondition"],
            {
                "per_block": True,
                "offered_modalities": ["audio"] * 6,
                "resident_clients": 3,
                "borrower_clients": 3,
                "quota_percent": 100,
                "stability_sensor": "soc012",
                "safety_sensor": "tj",
                "handoff_boundary": "thermal_measurement_end",
                "handoff_max_ms": 500.0,
                "active_stable_endpoints": 3,
                "active_stable_spacing_seconds": 1.0,
                "qualification_max_attempts": 3,
                "measured_processes_paused_until_success": True,
                "first_postcleanup_causal_sample": True,
                "actual_start_causal_gate": True,
                "block_max_attempts": 3,
                "retry_on_performance": False,
            },
        )

    def test_smoke_mode_is_explicitly_nonformal(self) -> None:
        protocol = profile.protocol_json("smoke")
        self.assertEqual(protocol["mode"], "smoke")
        self.assertFalse(protocol["formal"])
        with self.assertRaises(ValueError):
            profile.protocol_json("unexpected")

    def test_worker_command_matches_formal_priority_and_affinity(self) -> None:
        bench = pathlib.Path("/opt/jdg-trt-bench")
        engine = pathlib.Path("/engines/mig-2g-q100/whisper-tiny-encoder.engine")
        client = profile.ClientSpec("borrower-2g", 100, "audio")
        command = profile.worker_command(bench, engine, client, 7)
        self.assertEqual(command[:4], ["taskset", "--cpu-list", "7", str(bench)])
        self.assertEqual(command[command.index("--model-name") + 1], client.model)
        self.assertEqual(command[command.index("--warmup") + 1], "100")
        self.assertEqual(command[command.index("--priority") + 1], "low")
        self.assertEqual(command[command.index("--include-transfers") + 1], "true")
        self.assertEqual(command[command.index("--start-paused") + 1], "true")

    def test_critical_command_uses_resnet_and_wide_calibration_period(self) -> None:
        engine = pathlib.Path("/engines/mig-2g/resnet50-v2.engine")
        command = profile.critical_command(
            pathlib.Path("/opt/jdg-trt-bench"),
            engine,
            pathlib.Path("/out/trace.csv"),
            [101, 102],
        )
        self.assertEqual(command[:3], ["taskset", "--cpu-list", "12"])
        self.assertEqual(command[command.index("--model-name") + 1], "resnet50-v2")
        self.assertEqual(command[command.index("--period-ms") + 1], "40.0")
        self.assertEqual(command[command.index("--guard-ms") + 1], "20.0")
        self.assertEqual(command[command.index("--gate-mode") + 1], "cooperative")
        self.assertEqual(command[command.index("--priority") + 1], "high")

    def test_environment_and_engine_tag_are_placement_specific(self) -> None:
        root = pathlib.Path("/engines")
        resident = profile.ClientSpec("resident-1g", 25, "language")
        borrower = profile.ClientSpec("borrower-2g", 100, "audio")
        self.assertEqual(
            profile.engine_path(root, resident),
            root / "mig-1g-q25" / "distilbert-sst2.engine",
        )
        env = profile.placement_environment(
            borrower,
            base_env={"BASE": "1"},
            small_uuid="MIG-small",
            big_uuid="MIG-big",
            resident_mps_pipe=pathlib.Path("/small/pipe"),
            resident_mps_log=pathlib.Path("/small/log"),
            big_mps_pipe=pathlib.Path("/big/pipe"),
            big_mps_log=pathlib.Path("/big/log"),
        )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "MIG-big")
        self.assertEqual(env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"], "100")
        self.assertEqual(env["CUDA_MPS_PIPE_DIRECTORY"], "/big/pipe")

    def test_thermal_lock_binds_slow_state_sensor_and_tj_safety(self) -> None:
        lock = thermal_lock()
        profile._validate_guard_thermal_lock(lock)
        args = profile.thermal_load_arguments(
            bench=pathlib.Path("/bench"),
            engine_root=pathlib.Path("/engines"),
            big_mps_pipe=pathlib.Path("/mps/pipe"),
            big_mps_log=pathlib.Path("/mps/log"),
            thermal_lock=lock,
        )
        self.assertEqual(args.thermal_stability_sensor, "soc012")
        self.assertEqual(args.thermal_safety_sensor, "tj")
        self.assertEqual(args.thermal_handoff_max_ms, 500.0)
        self.assertEqual(args.pressure_rps_per_tenant, 0.0)
        for key, value in (
            ("schema_version", 3),
            ("stability_sensor", "tj"),
            ("safety_sensor", "gpu"),
            ("thermal_handoff_max_ms", 501.0),
            ("thermal_handoff_boundary", "thermal_start_qualification"),
            ("thermal_qualification_max_attempts", 4),
            ("thermal_active_stable_endpoints", 4),
        ):
            changed = dict(lock)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                profile._validate_guard_thermal_lock(changed)
        stale_dwell = dict(lock)
        stale_dwell["thermal_qualification_dwell_seconds"] = 1.0
        with self.assertRaises(ValueError):
            profile._validate_guard_thermal_lock(stale_dwell)

    def test_thermal_start_computes_marker_bounded_maximum_gap(self) -> None:
        class FakeMonitor:
            def __init__(self) -> None:
                self._samples = [
                    SimpleNamespace(
                        monotonic_ns=100_000_000 * (index + 1),
                        parsed=SimpleNamespace(temperatures_c={"soc012": 75.0}),
                    )
                    for index in range(600)
                ]
                self.aggregate_arguments: dict[str, object] | None = None

            def samples(self) -> list[SimpleNamespace]:
                return self._samples

            def aggregate(
                self, start_ns: int, end_ns: int, **kwargs: object
            ) -> dict[str, object]:
                self.aggregate_arguments = {
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    **kwargs,
                }
                return {
                    "health": {"healthy": True},
                    "temperatures_c": {"tj": {"max": 89.5}},
                }

        monitor = FakeMonitor()
        summary = profile._thermal_start_at_marker(
            monitor, thermal_lock(), 60_000_000_000
        )

        self.assertAlmostEqual(summary["maximum_gap_seconds"], 0.1)
        self.assertEqual(summary["samples"], 600)
        assert monitor.aggregate_arguments is not None
        self.assertEqual(
            monitor.aggregate_arguments["maximum_valid_gap_ns"], 300_000_000
        )

    def test_block_attempt_repreheats_after_no_sample_without_measurement(self) -> None:
        events: list[str] = []
        processes = iter(
            [self.FakeProcess(301, None), self.FakeProcess(302, None)]
        )
        marker_times = {
            "guard_block_prepare": 800_000_000,
            "guard_block_start": 1_330_000_000,
            "guard_block_measurement_release": 1_340_000_000,
            "guard_block_resume": 1_350_000_000,
            "guard_actual_start_qualification": 2_100_000_000,
            "guard_block_result": 2_200_000_000,
            "guard_block_end": 2_300_000_000,
        }
        marker_metadata: dict[str, dict[str, object]] = {}
        qualification_result_metadata: list[dict[str, object]] = []
        qualification_result_times = iter((1_120_000_000, 1_320_000_000))

        def popen(*_args: object, **_kwargs: object) -> ProfileP9GuardTests.FakeProcess:
            process = next(processes)
            events.append(f"spawn:{process.pid}")
            return process

        def paused(items: list[ProfileP9GuardTests.FakeProcess], **_kwargs: object) -> None:
            events.append("paused:" + ",".join(str(item.pid) for item in items))

        def require_paused(
            items: list[ProfileP9GuardTests.FakeProcess],
        ) -> dict[str, str]:
            events.append("verify-paused")
            return {str(item.pid): "T" for item in items}

        def marker(
            _writer: object, name: str, metadata: dict[str, object]
        ) -> int:
            events.append(f"marker:{name}")
            marker_metadata[name] = metadata
            if name == "thermal_start_qualification_result":
                qualification_result_metadata.append(metadata)
                return next(qualification_result_times)
            return marker_times[name]

        def preheat(*_args: object, **kwargs: object) -> dict[str, object]:
            events.append("preheat")
            attempt = events.count("preheat")
            boundary_ns = 1_000_000_000 + (attempt - 1) * 200_000_000
            return {
                "label": kwargs["label"],
                "measurement_start_monotonic_ns": boundary_ns - 900_000_000,
                "measurement_end_monotonic_ns": boundary_ns,
                "cleanup_end_monotonic_ns": boundary_ns + 20_000_000,
            }

        def qualify(*_args: object, **kwargs: object) -> dict[str, object]:
            events.append("qualify")
            attempt = int(kwargs["attempt"])
            boundary_ns = int(kwargs["boundary_ns"])
            cleanup_end_ns = int(kwargs["cleanup_end_ns"])
            passed = attempt == 2
            return {
                "attempt": attempt,
                "passed": passed,
                "boundary": "thermal_measurement_end",
                "boundary_monotonic_ns": boundary_ns,
                "cleanup_end_monotonic_ns": cleanup_end_ns,
                "qualification_monotonic_ns": boundary_ns + 110_000_000,
                "sample_monotonic_ns": boundary_ns + 100_000_000 if passed else None,
                "sample_age_ms": 10.0 if passed else None,
                "stability_sensor": "soc012",
                "stability_value_c": 75.0 if passed else None,
                "safety_sensor": "tj",
                "safety_value_c": 89.5 if passed else None,
                "target_c": 75.0,
                "tolerance_c": 1.0,
                "telemetry": {"health": {"healthy": True}} if passed else None,
                "failure_reason": (
                    None
                    if passed
                    else f"{kwargs['label']} observed no causal post-cleanup telemetry sample"
                ),
            }

        def actual(*_args: object, **kwargs: object) -> dict[str, object]:
            events.append("actual-start")
            self.assertEqual(kwargs["window_not_before_ns"], 1_220_000_000)
            return {
                "passed": True,
                "measurement_start_monotonic_ns": 1_360_000_000,
                "sample_monotonic_ns": 1_300_000_000,
                "sample_age_ms": 60.0,
                "stability_sensor": "soc012",
                "stability_value_c": 75.0,
                "safety_sensor": "tj",
                "safety_value_c": 89.5,
                "target_c": 75.0,
                "tolerance_c": 1.0,
                "telemetry": {"health": {"healthy": True}},
                "failure_reason": None,
            }

        def process_json(_stdout: str, description: str) -> dict[str, object]:
            if description == "critical benchmark":
                return {
                    "measurement_start_monotonic_ns": 1_360_000_000,
                    "measurement_end_monotonic_ns": 2_000_000_000,
                }
            return {"worker": description}

        case = profile.single_client_cases()[0]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            profile, "mark_event", side_effect=marker
        ), mock.patch.object(
            profile, "wait_until_paused", side_effect=paused
        ), mock.patch.object(
            profile,
            "process_affinity_snapshot",
            return_value={"pid": 1, "expected_cpu": 0, "tasks": []},
        ), mock.patch.object(
            profile, "sha256_file", return_value="a" * 64
        ), mock.patch.object(
            profile,
            "run_thermal_load",
            side_effect=preheat,
        ), mock.patch.object(
            profile,
            "qualify_thermal_start",
            side_effect=qualify,
        ), mock.patch.object(
            profile, "validate_thermal_qualification_evidence"
        ), mock.patch.object(
            profile, "require_successful_thermal_qualification", return_value=1_000_000_000
        ), mock.patch.object(
            profile, "validate_thermal_qualification_handoff"
        ), mock.patch.object(
            profile, "validate_actual_thermal_start", side_effect=actual
        ), mock.patch.object(
            profile, "_require_processes_paused", side_effect=require_paused
        ), mock.patch.object(
            profile,
            "resume_processes",
            side_effect=lambda _items: events.append("resume"),
        ), mock.patch.object(
            profile, "_read_process_json", side_effect=process_json
        ), mock.patch.object(profile, "atomic_json"):
            result = profile._run_block_attempt(
                case=case,
                block=1,
                block_attempt=1,
                output_root=pathlib.Path(directory),
                bench=pathlib.Path("/bench"),
                engine_root=pathlib.Path("/engines"),
                small_uuid="MIG-small",
                big_uuid="MIG-big",
                resident_mps_pipe=pathlib.Path("/small/pipe"),
                resident_mps_log=pathlib.Path("/small/log"),
                big_mps_pipe=pathlib.Path("/big/pipe"),
                big_mps_log=pathlib.Path("/big/log"),
                monitor=object(),
                telemetry_writer=object(),
                thermal_lock=thermal_lock(),
                popen=popen,
            )

        self.assertLess(events.index("spawn:301"), events.index("preheat"))
        self.assertLess(events.index("paused:302"), events.index("preheat"))
        self.assertLess(events.index("preheat"), events.index("qualify"))
        self.assertEqual(events.count("preheat"), 2)
        self.assertEqual(events.count("qualify"), 2)
        self.assertEqual(events.count("marker:thermal_start_qualification_result"), 2)
        self.assertIsNone(qualification_result_metadata[0]["sample_monotonic_ns"])
        self.assertGreater(
            events.index("marker:guard_block_start"),
            max(
                index
                for index, event in enumerate(events)
                if event == "marker:thermal_start_qualification_result"
            ),
        )
        self.assertLess(
            events.index("marker:guard_block_measurement_release"),
            events.index("resume"),
        )
        self.assertLess(events.index("marker:guard_block_resume"), events.index("resume"))
        self.assertLess(events.index("resume"), events.index("actual-start"))
        self.assertEqual(
            marker_metadata["guard_block_resume"]["resume_semantics"],
            "issued-before-sigcont",
        )
        self.assertTrue(result["thermally_valid"])
        self.assertEqual(result["selected_thermal_attempt"], 2)
        self.assertEqual(len(result["thermal_attempts"]), 2)
        self.assertFalse(result["thermal_attempts"][0]["pre_release_passed"])
        self.assertIsNone(
            result["thermal_attempts"][0]["qualification"]["sample_monotonic_ns"]
        )
        self.assertIsNone(result["thermal_attempts"][0]["start_marker_monotonic_ns"])
        self.assertEqual(result["thermal_handoff"]["boundary"], "thermal_measurement_end")
        self.assertEqual(
            result["thermal_handoff"]["boundary_to_critical_measurement_start_ms"],
            160.0,
        )
        self.assertTrue(result["thermal_handoff"]["strictly_within_bound"])

    def test_logical_block_retries_only_until_first_thermal_valid_attempt(self) -> None:
        first = {"attempt": 1, "thermally_valid": False, "drain_max_ms": 0.1}
        second = {"attempt": 2, "thermally_valid": True, "drain_max_ms": 9.9}
        with mock.patch.object(
            profile, "_run_block_attempt", side_effect=[first, second]
        ) as run_attempt:
            result = profile.run_block(
                case=profile.single_client_cases()[0],
                block=1,
                output_root=pathlib.Path("/out"),
                bench=pathlib.Path("/bench"),
                engine_root=pathlib.Path("/engines"),
                small_uuid="MIG-small",
                big_uuid="MIG-big",
                resident_mps_pipe=pathlib.Path("/small/pipe"),
                resident_mps_log=pathlib.Path("/small/log"),
                big_mps_pipe=pathlib.Path("/big/pipe"),
                big_mps_log=pathlib.Path("/big/log"),
                monitor=object(),
                telemetry_writer=object(),
                thermal_lock=thermal_lock(),
            )
        self.assertEqual(result["selected_attempt"], 2)
        self.assertEqual(result["attempts"], [first, second])
        self.assertEqual(
            [call.kwargs["block_attempt"] for call in run_attempt.call_args_list],
            [1, 2],
        )

    def test_handoff_limit_is_strict(self) -> None:
        lock = thermal_lock()
        self.assertEqual(
            profile._handoff_elapsed_ms(1_000_000_000, 1_499_999_999, lock, "x"),
            499.999999,
        )
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            profile._handoff_elapsed_ms(
                1_000_000_000, 1_500_000_000, lock, "x"
            )

    def test_active_handoff_marks_exact_500_ms_and_clock_reordering_invalid(self) -> None:
        qualification = {
            "boundary_monotonic_ns": 1_000_000_000,
            "cleanup_end_monotonic_ns": 1_020_000_000,
            "sample_monotonic_ns": 1_100_000_000,
            "qualification_monotonic_ns": 1_110_000_000,
        }
        evidence = profile._handoff_evidence(
            qualification,
            qualification_result_ns=1_120_000_000,
            block_start_ns=1_130_000_000,
            measurement_release_ns=1_140_000_000,
            resume_issued_ns=1_150_000_000,
            measurement_start_ns=1_500_000_000,
            thermal_lock=thermal_lock(),
        )
        self.assertFalse(evidence["strictly_within_bound"])
        reordered = dict(qualification)
        reordered["sample_monotonic_ns"] = 1_010_000_000
        evidence = profile._handoff_evidence(
            reordered,
            qualification_result_ns=1_120_000_000,
            block_start_ns=1_130_000_000,
            measurement_release_ns=1_140_000_000,
            resume_issued_ns=1_150_000_000,
            measurement_start_ns=1_160_000_000,
            thermal_lock=thermal_lock(),
        )
        self.assertFalse(evidence["strictly_within_bound"])

    def test_process_state_snapshot_tolerates_exit_race(self) -> None:
        alive = self.FakeProcess(101, None)
        exited = self.FakeProcess(102, 7)
        malformed = self.FakeProcess(103, None)
        with mock.patch.object(
            profile,
            "process_state",
            side_effect=[FileNotFoundError("gone"), RuntimeError("malformed")],
        ) as process_state:
            states = profile._best_effort_process_states(
                [alive, exited, malformed]
            )
        self.assertEqual(
            states,
            {
                "101": "exited",
                "102": "exited",
                "103": "unavailable:RuntimeError",
            },
        )
        self.assertEqual(process_state.call_args_list, [mock.call(101), mock.call(103)])

    def test_terminate_continues_after_first_process_signal_and_wait_failures(self) -> None:
        class CleanupProcess:
            def __init__(self, pid: int, fail_first_wait: bool = False) -> None:
                self.pid = pid
                self.fail_first_wait = fail_first_wait
                self.wait_calls = 0
                self.sent_signals: list[int] = []
                self.killed = False
                self.terminated = False

            def poll(self) -> None:
                return None

            def send_signal(self, sent_signal: int) -> None:
                self.sent_signals.append(sent_signal)

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                self.wait_calls += 1
                if self.fail_first_wait and self.wait_calls == 1:
                    raise OSError("wait failed")
                self.terminated = True
                return 0

            def kill(self) -> None:
                self.killed = True

        first = CleanupProcess(401, fail_first_wait=True)
        second = CleanupProcess(402)
        resumed: list[int] = []

        def resume(pid: int, sent_signal: int) -> None:
            self.assertEqual(sent_signal, profile.signal.SIGCONT)
            resumed.append(pid)
            if pid == first.pid:
                raise OSError("resume failed")

        with mock.patch.object(profile.os, "kill", side_effect=resume):
            with self.assertRaisesRegex(
                RuntimeError, "pid=401:sigcont:OSError:resume failed"
            ) as raised:
                profile._terminate([first, second])  # type: ignore[arg-type]

        self.assertIn("pid=401:wait:OSError:wait failed", str(raised.exception))
        self.assertEqual(resumed, [401, 402])
        self.assertEqual(first.sent_signals, [profile.signal.SIGINT])
        self.assertTrue(first.killed)
        self.assertTrue(first.terminated)
        self.assertEqual(second.sent_signals, [profile.signal.SIGINT])
        self.assertFalse(second.killed)
        self.assertTrue(second.terminated)

    def test_abort_cleanup_errors_cannot_escape_or_hide_primary(self) -> None:
        case = profile.single_client_cases()[0]
        worker = self.FakeProcess(201, None)
        critical = self.FakeProcess(202, None)
        primary = RuntimeError("thermal start precondition timed out")
        with mock.patch.object(
            profile,
            "_best_effort_process_states",
            return_value={"201": "exited"},
        ), mock.patch.object(
            profile,
            "_terminate",
            side_effect=FileNotFoundError("already exited"),
        ), mock.patch.object(profile, "mark_event") as mark_event:
            with self.assertRaisesRegex(RuntimeError, "thermal start precondition"):
                try:
                    raise primary
                except RuntimeError:
                    profile._cleanup_aborted_block(
                        case=case,
                        block=1,
                        workers=[worker],
                        critical=critical,
                        telemetry_writer=object(),
                    )
                    raise
        metadata = mark_event.call_args.args[2]
        self.assertEqual(metadata["worker_states"], {"201": "exited"})
        self.assertEqual(
            metadata["cleanup_errors"],
            ["terminate:FileNotFoundError:already exited"],
        )

    def test_abort_marker_failure_is_best_effort(self) -> None:
        case = profile.single_client_cases()[0]
        with mock.patch.object(
            profile, "_best_effort_process_states", return_value={}
        ), mock.patch.object(profile, "_terminate"), mock.patch.object(
            profile, "mark_event", side_effect=OSError("telemetry closed")
        ):
            profile._cleanup_aborted_block(
                case=case,
                block=1,
                workers=[],
                critical=None,
                telemetry_writer=object(),
            )

    def test_campaign_primary_survives_telemetry_close_failure(self) -> None:
        primary = RuntimeError("guard block failed")
        close = mock.Mock(side_effect=OSError("tail close failed"))
        session = SimpleNamespace(
            tail_process=SimpleNamespace(pid=991), monitor=object()
        )
        case = profile.single_client_cases()[0]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            args = self.campaign_args(root)
            with mock.patch.object(
                profile.os, "sched_getaffinity", return_value={13}
            ), mock.patch.object(
                profile.os, "sched_setaffinity"
            ), mock.patch.object(
                profile, "_read_json_once", return_value=(thermal_lock(), "a" * 64)
            ), mock.patch.object(
                profile,
                "load_env",
                return_value={
                    "JDG_MIG_SMALL_UUID": "MIG-small",
                    "JDG_MIG_BIG_UUID": "MIG-big",
                    "JDG_MPS_PIPE_DIRECTORY": str(args.resident_mps_pipe),
                    "JDG_MPS_LOG_DIRECTORY": str(args.resident_mps_log),
                },
            ), mock.patch.object(
                profile, "_required_artifacts", return_value={}
            ), mock.patch.object(
                profile, "hardware_fingerprint", return_value={}
            ), mock.patch.object(
                profile, "sha256_file", return_value="b" * 64
            ), mock.patch.object(
                profile, "start_telemetry_session", return_value=session
            ), mock.patch.object(
                profile, "mark_event", return_value=1
            ), mock.patch.object(
                profile, "single_client_cases", return_value=(case,)
            ), mock.patch.object(
                profile, "held_out_cases", return_value=()
            ), mock.patch.object(
                profile, "run_block", side_effect=primary
            ), mock.patch.object(
                profile, "close_telemetry_session", close
            ):
                with self.assertRaisesRegex(RuntimeError, "guard block failed") as raised:
                    profile.run_campaign(args)
        self.assertIs(raised.exception, primary)
        self.assertEqual(close.call_count, 1)
        self.assertIn("tail close failed", " ".join(primary.__notes__))

    def test_campaign_keyboard_interrupt_still_closes_telemetry(self) -> None:
        close = mock.Mock(return_value=[])
        session = SimpleNamespace(
            tail_process=SimpleNamespace(pid=992), monitor=object()
        )
        case = profile.single_client_cases()[0]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            args = self.campaign_args(root)
            with mock.patch.object(
                profile.os, "sched_getaffinity", return_value={13}
            ), mock.patch.object(
                profile.os, "sched_setaffinity"
            ), mock.patch.object(
                profile, "_read_json_once", return_value=(thermal_lock(), "a" * 64)
            ), mock.patch.object(
                profile,
                "load_env",
                return_value={
                    "JDG_MIG_SMALL_UUID": "MIG-small",
                    "JDG_MIG_BIG_UUID": "MIG-big",
                    "JDG_MPS_PIPE_DIRECTORY": str(args.resident_mps_pipe),
                    "JDG_MPS_LOG_DIRECTORY": str(args.resident_mps_log),
                },
            ), mock.patch.object(
                profile, "_required_artifacts", return_value={}
            ), mock.patch.object(
                profile, "hardware_fingerprint", return_value={}
            ), mock.patch.object(
                profile, "sha256_file", return_value="b" * 64
            ), mock.patch.object(
                profile, "start_telemetry_session", return_value=session
            ), mock.patch.object(
                profile, "mark_event", return_value=1
            ), mock.patch.object(
                profile, "single_client_cases", return_value=(case,)
            ), mock.patch.object(
                profile, "held_out_cases", return_value=()
            ), mock.patch.object(
                profile, "run_block", side_effect=KeyboardInterrupt
            ), mock.patch.object(
                profile, "close_telemetry_session", close
            ):
                with self.assertRaises(KeyboardInterrupt):
                    profile.run_campaign(args)
        close.assert_called_once_with(session)


if __name__ == "__main__":
    unittest.main()
