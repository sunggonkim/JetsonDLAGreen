#!/usr/bin/env python3
import importlib.util
import io
import json
import pathlib
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
SPEC = importlib.util.spec_from_file_location(
    "mig_slack_governor", ROOT / "runtime" / "mig_slack_governor.py"
)
assert SPEC is not None and SPEC.loader is not None
GOVERNOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GOVERNOR
SPEC.loader.exec_module(GOVERNOR)


class MigSlackGovernorTest(unittest.TestCase):
    @staticmethod
    def parsed_args() -> object:
        argv = [
            "mig_slack_governor.py",
            "--bench",
            "/tmp/bench",
            "--engine-root",
            "/tmp/engines",
            "--mig-env",
            "/tmp/mig.env",
            "--big-mps-pipe",
            "/tmp/pipe",
            "--big-mps-log",
            "/tmp/log",
            "--output",
            "/tmp/summary.json",
        ]
        with mock.patch.object(sys, "argv", argv):
            return GOVERNOR.parse_args()

    @staticmethod
    def worker_result(pid: int = 1234) -> dict:
        return {
            "schema_version": 1,
            "role": "pressure",
            "model": "distilbert-sst2",
            "execution_environment": {
                "pid": pid,
                "cuda_visible_devices": "small",
                "mps_active_thread_percentage": 25,
                "cpu_affinity": [0],
            },
            "gpu": {
                "name": "NVIDIA Thor MIG 1g.0gb",
                "multiprocessors": 2,
            },
            "config": {
                "warmup": 100,
                "duration_seconds": 3600,
                "include_transfers": True,
                "priority": "default",
                "start_paused": True,
                "burst_size": 1,
                "period_ms": 0,
                "deadline_ms": 0,
                "guard_ms": 0,
                "gated_processes": 0,
                "stopped_processes": 0,
                "gate_mode": "stop",
                "stream_priority_value": 0,
            },
        }

    @staticmethod
    def successful_thermal_qualification(
        boundary_ns: int = 1_000_000_000,
        *,
        attempt: int = 1,
    ) -> dict:
        cleanup_ns = boundary_ns + 20_000_000
        sample_ns = cleanup_ns + 10_000_000
        qualification_ns = sample_ns + 10_000_000
        return {
            "attempt": attempt,
            "passed": True,
            "boundary": GOVERNOR.THERMAL_HANDOFF_BOUNDARY,
            "boundary_monotonic_ns": boundary_ns,
            "cleanup_end_monotonic_ns": cleanup_ns,
            "qualification_monotonic_ns": qualification_ns,
            "sample_monotonic_ns": sample_ns,
            "sample_age_ms": 10.0,
            "stability_sensor": "soc012",
            "stability_value_c": 75.0,
            "safety_sensor": "tj",
            "safety_value_c": 90.0,
            "target_c": 75.0,
            "tolerance_c": 1.0,
            "telemetry": {"health": {"healthy": True}},
            "failure_reason": None,
        }

    @staticmethod
    def successful_actual_start(measurement_start_ns: int) -> dict:
        return {
            "passed": True,
            "measurement_start_monotonic_ns": measurement_start_ns,
            "sample_monotonic_ns": measurement_start_ns - 10_000_000,
            "sample_age_ms": 10.0,
            "stability_sensor": "soc012",
            "stability_value_c": 75.0,
            "safety_sensor": "tj",
            "safety_value_c": 90.0,
            "target_c": 75.0,
            "tolerance_c": 1.0,
            "telemetry": {"health": {"healthy": True}},
            "failure_reason": None,
        }

    def test_worker_provenance_binds_mps_quota_and_visible_sms(self) -> None:
        process = mock.Mock(pid=1234)
        worker = GOVERNOR.RunningWorker(
            GOVERNOR.WorkerAction(0, "language", "resident-1g", 25),
            process,
            0,
            "small",
            100,
        )
        result = self.worker_result()
        GOVERNOR.validate_worker_result(result, worker)
        worker.period_ms = 2.5
        result["config"]["period_ms"] = 2.5
        GOVERNOR.validate_worker_result(result, worker)
        result["execution_environment"]["mps_active_thread_percentage"] = 100
        with self.assertRaisesRegex(ValueError, "execution environment"):
            GOVERNOR.validate_worker_result(result, worker)

        borrower = GOVERNOR.RunningWorker(
            GOVERNOR.WorkerAction(0, "language", "borrower-2g", 25),
            mock.Mock(pid=1234),
            0,
            "big",
            100,
        )
        borrower_result = self.worker_result()
        borrower_result["gpu"] = {
            "name": "NVIDIA Thor MIG 2g.0gb",
            "multiprocessors": 2,
        }
        borrower_result["execution_environment"]["cuda_visible_devices"] = "big"
        borrower_result["execution_environment"]["mps_active_thread_percentage"] = 25
        borrower_result["config"]["priority"] = "low"
        GOVERNOR.validate_worker_result(borrower_result, borrower)

    def test_critical_provenance_binds_workload_and_gate(self) -> None:
        args = self.parsed_args()
        with tempfile.TemporaryDirectory() as temporary:
            args.engine_root = pathlib.Path(temporary)
            engine = args.engine_root / "mig-2g" / "resnet50-v2.engine"
            engine.parent.mkdir(parents=True)
            engine.write_bytes(b"unit-test-engine-placeholder")
            result = {
                "schema_version": 1,
                "role": "benchmark",
                "model": "resnet50-v2",
                "engine": str(engine),
                "completed_requests": args.samples,
                "execution_environment": {
                    "pid": 4321,
                    "cuda_visible_devices": "big",
                    "mps_active_thread_percentage": 100,
                    "cpu_affinity": [12],
                },
                "gpu": {
                    "name": "NVIDIA Thor MIG 2g.0gb",
                    "multiprocessors": 12,
                },
                "config": {
                    "warmup": args.warmup,
                    "burst_size": args.burst_size,
                    "period_ms": args.period_ms,
                    "deadline_ms": 6.0,
                    "duration_seconds": 0,
                    "guard_ms": 2.0,
                    "gated_processes": 1,
                    "stopped_processes": 2,
                    "gate_mode": "cooperative",
                    "start_paused": True,
                    "include_transfers": True,
                    "priority": "high",
                    "stream_priority_value": -5,
                },
            }
            GOVERNOR.validate_critical_result(
                result,
                process_pid=4321,
                args=args,
                critical_uuid="big",
                deadline_ms=6.0,
                gated_processes=1,
                stopped_processes=2,
                guard_ms=2.0,
                start_paused=True,
            )
            result["model"] = "resnet10-detection"
            with self.assertRaisesRegex(RuntimeError, "workload provenance"):
                GOVERNOR.validate_critical_result(
                    result,
                    process_pid=4321,
                    args=args,
                    critical_uuid="big",
                    deadline_ms=6.0,
                    gated_processes=1,
                    stopped_processes=2,
                    guard_ms=2.0,
                    start_paused=True,
                )

    def test_scenarios_share_the_same_paired_multimodal_trace(self) -> None:
        self.assertEqual(
            GOVERNOR.offered_for_epoch(0), ("audio", "language")
        )
        self.assertEqual(
            GOVERNOR.offered_for_epoch(6), ("audio", "language")
        )
        self.assertEqual(
            GOVERNOR.offered_for_epoch(2, "independent"),
            GOVERNOR.offered_for_epoch(2, "dependent"),
        )

    def test_dependent_scenario_exposes_audio_to_language_chain(self) -> None:
        self.assertEqual(
            GOVERNOR.offered_for_epoch(0, "dependent"),
            ("audio", "language"),
        )
        self.assertEqual(
            GOVERNOR.offered_for_epoch(2, "dependent"),
            ("audio", "language", "audio", "language", "audio", "language"),
        )
        plan = GOVERNOR.plan_for(
            "fixed-full-gate",
            GOVERNOR.offered_for_epoch(0, "dependent"),
            GOVERNOR.FeedbackState(),
        )
        pipes = GOVERNOR.dependency_pipes_for_plan(plan, "dependent")
        try:
            self.assertEqual(
                [(p.upstream_tenant_id, p.downstream_tenant_id) for p in pipes],
                [(0, 1)],
            )
        finally:
            GOVERNOR.close_dependency_pipes(pipes)

    def test_independent_scenario_has_no_dependency_edges(self) -> None:
        plan = GOVERNOR.plan_for(
            "static-mig",
            GOVERNOR.offered_for_epoch(2, "independent"),
            GOVERNOR.FeedbackState(),
        )
        self.assertEqual(
            GOVERNOR.dependency_pipes_for_plan(plan, "independent"), ()
        )

    def test_static_mig_keeps_best_effort_on_small_instance(self) -> None:
        plan = GOVERNOR.plan_for(
            "static-mig",
            ("language", "audio"),
            GOVERNOR.FeedbackState(),
        )
        self.assertEqual(len(plan.residents), 2)
        self.assertEqual(len(plan.borrowers), 0)
        self.assertTrue(
            all(action.placement == "resident-1g" for action in plan.residents)
        )

    def test_same_mig_places_best_effort_with_critical(self) -> None:
        plan = GOVERNOR.plan_for(
            "same-mig",
            ("language", "audio"),
            GOVERNOR.FeedbackState(),
        )
        self.assertEqual(len(plan.residents), 0)
        self.assertEqual(len(plan.borrowers), 2)
        self.assertTrue(
            all(action.placement == "borrower-2g" for action in plan.borrowers)
        )

    def test_borrowing_preserves_identical_tenant_concurrency(self) -> None:
        plan = GOVERNOR.plan_for(
            "fixed-borrow",
            ("language", "audio", "language"),
            GOVERNOR.FeedbackState(),
        )
        self.assertEqual(len(plan.residents), 2)
        self.assertEqual(len(plan.borrowers), 1)
        self.assertEqual(len(plan.residents) + len(plan.borrowers), 3)
        self.assertEqual(
            sorted(
                action.tenant_id
                for action in (*plan.residents, *plan.borrowers)
            ),
            [0, 1, 2],
        )

    def test_resident_full_gate_keeps_all_tenants_on_small_instance(self) -> None:
        plan = GOVERNOR.plan_for(
            "resident-full-gate",
            ("language", "audio", "language"),
            GOVERNOR.FeedbackState(),
        )
        self.assertEqual(len(plan.residents), 3)
        self.assertFalse(plan.borrowers)

    def test_every_policy_uses_the_same_active_worker_count(self) -> None:
        offered = ("language", "audio", "language", "audio")
        for policy in GOVERNOR.POLICIES:
            plan = GOVERNOR.plan_for(
                policy, offered, GOVERNOR.FeedbackState()
            )
            self.assertEqual(
                len(plan.residents) + len(plan.borrowers),
                len(offered),
                policy,
            )

    def test_governor_applies_admission_quota_and_borrower_limits(self) -> None:
        state = GOVERNOR.FeedbackState(
            resident_admission_limit=3,
            resident_quota_index=1,
            borrower_limit=2,
        )
        plan = GOVERNOR.plan_for(
            "mig-governor",
            ("language", "audio", "language", "audio"),
            state,
        )
        self.assertEqual(len(plan.residents), 2)
        self.assertEqual(len(plan.borrowers), 1)
        self.assertTrue(
            all(action.quota_percent == 50 for action in plan.residents)
        )

    def test_only_controlled_borrowing_receives_a_guard(self) -> None:
        state = GOVERNOR.FeedbackState()
        plan = GOVERNOR.plan_for(
            "fixed-borrow", ("language", "audio"), state
        )
        profile = {"language": 1.5, "audio": 2.0}
        self.assertEqual(
            GOVERNOR.guard_for(
                "uncoordinated-borrow", plan, state, profile
            ),
            0.0,
        )
        self.assertEqual(
            GOVERNOR.guard_for(
                "fixed-borrow", plan, state, profile
            ),
            2.0,
        )
        self.assertEqual(
            GOVERNOR.guard_for(
                "mig-governor", plan, state, profile
            ),
            2.0,
        )

    def test_governor_guards_resident_only_epoch(self) -> None:
        state = GOVERNOR.FeedbackState()
        plan = GOVERNOR.plan_for("mig-governor", ("language",), state)
        self.assertFalse(plan.borrowers)
        self.assertEqual(
            GOVERNOR.guard_for(
                "mig-governor", plan, state, {"language": 1.5, "audio": 2.0}
            ),
            1.5,
        )

    def test_guard_bounds_serial_mps_work_per_instance(self) -> None:
        state = GOVERNOR.FeedbackState()
        plan = GOVERNOR.plan_for("mig-governor", ("audio",) * 6, state)
        self.assertEqual(
            GOVERNOR.guard_for(
                "mig-governor", plan, state, {"language": 1.5, "audio": 2.0}
            ),
            6.0,
        )

    def test_quota_aware_guard_profile_uses_action_quota(self) -> None:
        guards = {
            "resident-1g": {
                str(quota): {
                    "language": {"guard_ms": language},
                    "audio": {"guard_ms": audio},
                }
                for quota, language, audio in (
                    (25, 1.6, 6.3),
                    (50, 1.3, 3.4),
                    (100, 1.1, 1.9),
                )
            },
            "borrower-2g": {
                "100": {
                    "language": {"guard_ms": 1.0},
                    "audio": {"guard_ms": 1.5},
                }
            },
        }
        lock = {
            "schema_version": 3,
            "kind": "p9-quota-aware-guard-lock",
            "guards": guards,
        }
        profile = GOVERNOR.guard_profile_from_lock(lock)
        self.assertEqual(GOVERNOR.GUARD_LOCK_SCHEMA_VERSION, 3)
        with self.assertRaisesRegex(ValueError, "unsupported schema"):
            GOVERNOR.guard_profile_from_lock(lock | {"schema_version": 2})

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "guard-lock.json"
            payload = json.dumps(lock, sort_keys=True) + "\n"
            path.write_text(payload, encoding="utf-8")
            loaded, loaded_profile, digest = GOVERNOR.load_guard_lock(path, None)
            self.assertEqual(loaded, lock)
            self.assertEqual(loaded_profile, profile)
            path.write_text(payload + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                GOVERNOR.load_guard_lock(path, digest)
        state = GOVERNOR.FeedbackState(resident_quota_index=0)
        plan = GOVERNOR.plan_for("mig-governor", ("audio",) * 6, state, 100)
        self.assertEqual(
            GOVERNOR.guard_for("mig-governor", plan, state, profile),
            18.9,
        )
        self.assertLess(GOVERNOR.maximum_profiled_guard_ms(profile), 20.0)

    def test_governor_gates_residual_cross_mig_interference(self) -> None:
        self.assertEqual(
            GOVERNOR.gated_placements("fixed-borrow"),
            frozenset({"borrower-2g"}),
        )
        self.assertEqual(
            GOVERNOR.gated_placements("resident-full-gate"),
            frozenset({"resident-1g"}),
        )
        self.assertEqual(
            GOVERNOR.gated_placements("fixed-full-gate"),
            frozenset({"resident-1g", "borrower-2g"}),
        )
        self.assertEqual(
            GOVERNOR.gated_placements("mig-governor"),
            frozenset({"resident-1g", "borrower-2g"}),
        )
        self.assertEqual(GOVERNOR.gated_placements("static-mig"), frozenset())

    def test_violation_reclaims_borrower_and_resident_capacity(self) -> None:
        state = GOVERNOR.FeedbackState(
            resident_quota_index=2,
            borrower_limit=6,
            guard_adjustment_ms=0.0,
        )
        action = GOVERNOR.update_feedback(
            state,
            violated=True,
            critical_p99_ms=5.2,
            deadline_ms=5.0,
        )
        self.assertEqual(action, "residual-reclaim")
        self.assertEqual(state.borrower_limit, 5)
        self.assertEqual(state.resident_quota_percent, 50)
        self.assertEqual(state.guard_adjustment_ms, 0.0)

    def test_drain_pressure_reduces_concurrency_under_frozen_guard(self) -> None:
        state = GOVERNOR.FeedbackState()
        action = GOVERNOR.update_feedback(
            state,
            violated=False,
            critical_p99_ms=4.0,
            deadline_ms=5.0,
            drain_near_overrun=True,
        )
        self.assertEqual(action, "drain-reclaim")
        self.assertEqual(state.guard_adjustment_ms, 0.0)
        self.assertEqual(state.borrower_limit, 5)
        self.assertEqual(state.resident_admission_limit, 5)
        self.assertEqual(state.resident_quota_percent, 100)

    def test_safe_feedback_recovers_one_dimension_at_a_time(self) -> None:
        state = GOVERNOR.FeedbackState(
            resident_admission_limit=5,
            resident_quota_index=1,
            borrower_limit=4,
        )
        for _ in range(3):
            GOVERNOR.update_feedback(
                state,
                violated=False,
                critical_p99_ms=4.0,
                deadline_ms=5.0,
            )
        self.assertEqual(state.resident_admission_limit, 5)
        self.assertEqual(state.resident_quota_percent, 100)
        self.assertEqual(state.borrower_limit, 4)

    def test_telemetry_failure_forces_minimum_safe_state(self) -> None:
        state = GOVERNOR.FeedbackState()
        GOVERNOR.fail_closed_feedback(state)
        self.assertEqual(state.resident_admission_limit, 1)
        self.assertEqual(state.resident_quota_percent, 25)
        self.assertEqual(state.borrower_limit, 0)

    def test_cpu_parser(self) -> None:
        self.assertEqual(GOVERNOR.expand_cpu_list("0-2,5"), [0, 1, 2, 5])
        self.assertEqual(GOVERNOR.format_cpu_list([5, 1, 2, 3]), "1-3,5")
        with self.assertRaises(ValueError):
            GOVERNOR.expand_cpu_list("2-0")

    def test_validation_rejects_nonfinite_and_overlapping_cpu_inputs(self) -> None:
        arguments = self.parsed_args()
        arguments.deadline_ms = float("nan")
        with self.assertRaisesRegex(SystemExit, "must be finite"):
            GOVERNOR.validate_args(arguments)
        arguments = self.parsed_args()
        arguments.pressure_cpus = "0-11"
        with self.assertRaisesRegex(SystemExit, "must be disjoint"):
            GOVERNOR.validate_args(arguments)

    def test_live_thermal_telemetry_fails_closed(self) -> None:
        class Monitor:
            def __init__(self, result: dict) -> None:
                self.result = result

            def aggregate(self, *_args: object, **_kwargs: object) -> dict:
                return self.result

        unhealthy = {"health": {"healthy": False}}
        with self.assertRaisesRegex(RuntimeError, "became unhealthy"):
            GOVERNOR.require_live_thermal_telemetry(
                Monitor(unhealthy), 1, 1_000_000_001, 104.0, "test"
            )
        hot = {
            "health": {"healthy": True},
            "temperatures_c": {
                "soc012": {"max": 75.0},
                "tj": {"max": 104.0},
            },
        }
        with self.assertRaisesRegex(RuntimeError, "hard limit"):
            GOVERNOR.require_live_thermal_telemetry(
                Monitor(hot), 1, 1_000_000_001, 104.0, "test"
            )

    def test_post_warmup_process_barrier(self) -> None:
        process = subprocess.Popen(
            ["bash", "-c", "kill -STOP $$; sleep 30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            GOVERNOR.wait_until_paused([process], 2.0)
            self.assertIn(GOVERNOR.process_state(process.pid), {"T", "t"})
            GOVERNOR.resume_processes([process])
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGCONT)
                process.terminate()
                process.communicate(timeout=2.0)

    def test_calibration_preloads_critical_before_each_preheater(self) -> None:
        events: list[str] = []
        commands: list[list[str]] = []

        class Process:
            pid = 4321
            returncode = 0

            @staticmethod
            def poll() -> int:
                return 0

        process = Process()

        class Monitor:
            def __init__(self) -> None:
                self.qualification_results = iter((850_000_000, 950_000_000))

            def mark(self, name: str, _metadata: dict) -> object:
                events.append(f"mark:{name}")
                timestamp = (
                    next(self.qualification_results)
                    if name == "thermal_start_qualification_result"
                    else {
                        "calibration_prepare": 100_000_000,
                        "calibration_start": 1_100_000_000,
                        "thermal_actual_start_qualification_result": 2_005_000_000,
                        "calibration_measurement_window": 2_010_000_000,
                        "calibration_end": 2_100_000_000,
                    }[name]
                )
                return types.SimpleNamespace(monotonic_ns=timestamp)

        def popen(command: list[str], **_kwargs: object) -> Process:
            events.append("spawn-critical")
            commands.append(command)
            return process

        def wait_until_paused(
            processes: list[Process], _timeout: float
        ) -> None:
            self.assertEqual(processes, [process])
            events.append("critical-paused")

        def preheat(*_args: object, **kwargs: object) -> dict:
            self.assertIn("critical-paused", events)
            events.append("preheat")
            attempt = int(str(kwargs["label"]).rsplit("-", 1)[1])
            return {
                "label": kwargs["label"],
                "measurement_start_monotonic_ns": (attempt - 1) * 100_000_000,
                "measurement_end_monotonic_ns": 700_000_000
                + attempt * 100_000_000,
                "cleanup_end_monotonic_ns": 720_000_000
                + attempt * 100_000_000,
                "last_window": stable_window,
            }

        stable_window = {
            "samples": 480,
            "window_seconds": 60.0,
            "observed_span_seconds": 59.4,
            "mean_c": 75.0,
            "min_c": 75.0,
            "max_c": 75.0,
            "latest_c": 75.0,
            "slope_c_per_minute": 0.0,
            "maximum_gap_seconds": 0.1,
        }

        def resume(processes: list[Process]) -> None:
            self.assertEqual(processes, [process])
            events.append("resume-critical")

        def qualify(*_args: object, **kwargs: object) -> dict:
            events.append("qualify-start")
            attempt = int(kwargs["attempt"])
            result = self.successful_thermal_qualification(
                boundary_ns=int(kwargs["boundary_ns"]),
                attempt=attempt,
            )
            if attempt == 1:
                result["passed"] = False
                result["failure_reason"] = "unstable candidate"
            return result

        def collect(_process: Process, _timeout: float) -> dict:
            events.append("collect-critical")
            return {
                "measurement_start_monotonic_ns": 1_200_000_000,
                "measurement_end_monotonic_ns": 2_000_000_000,
            }

        args = self.parsed_args()
        args.calibration_repeats = 1
        args.thermal_target_c = 75.0
        args.thermal_hard_limit_c = 104.0
        validate = mock.Mock()
        patches = (
            mock.patch.object(
                GOVERNOR, "engine_path", return_value=pathlib.Path("/tmp/engine")
            ),
            mock.patch.object(GOVERNOR.subprocess, "Popen", side_effect=popen),
            mock.patch.object(
                GOVERNOR, "wait_until_paused", side_effect=wait_until_paused
            ),
            mock.patch.object(
                GOVERNOR,
                "require_processes_paused",
                return_value={str(process.pid): "T"},
            ),
            mock.patch.object(
                GOVERNOR,
                "process_affinity_snapshot",
                return_value={"pid": process.pid},
            ),
            mock.patch.object(GOVERNOR, "run_thermal_load", side_effect=preheat),
            mock.patch.object(GOVERNOR, "resume_processes", side_effect=resume),
            mock.patch.object(GOVERNOR, "qualify_thermal_start", side_effect=qualify),
            mock.patch.object(GOVERNOR, "collect_json", side_effect=collect),
            mock.patch.object(GOVERNOR, "validate_critical_result", validate),
            mock.patch.object(
                GOVERNOR,
                "validate_actual_thermal_start",
                side_effect=lambda *_args, **kwargs: self.successful_actual_start(
                    int(kwargs["measurement_start_ns"])
                ),
            ),
            mock.patch.object(
                GOVERNOR,
                "terminate_process",
                side_effect=lambda _process: events.append("cleanup-critical"),
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
            5
        ], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11]:
            calibrations, _, preconditions = GOVERNOR.run_calibration_set(
                args,
                {"CUDA_VISIBLE_DEVICES": "big"},
                10.0,
                Monitor(),
                "pre",
                base_env={},
                mig={},
                precondition_each_repeat=True,
            )

        self.assertLess(events.index("spawn-critical"), events.index("preheat"))
        self.assertLess(events.index("critical-paused"), events.index("preheat"))
        self.assertLess(events.index("preheat"), events.index("resume-critical"))
        self.assertIn("--start-paused", commands[0])
        self.assertEqual(
            commands[0][commands[0].index("--start-paused") + 1], "true"
        )
        self.assertTrue(validate.call_args.kwargs["start_paused"])
        self.assertEqual(len(preconditions), 1)
        self.assertEqual(events.count("preheat"), 2)
        handoff = calibrations[0]["thermal_handoff"]
        self.assertEqual(handoff["boundary_to_measurement_release_ms"], 200.0)
        self.assertEqual(handoff["boundary_to_measurement_start_ms"], 300.0)
        self.assertEqual(len(calibrations[0]["thermal_start_attempts"]), 2)
        self.assertFalse(
            calibrations[0]["thermal_start_attempts"][0]["qualification"]["passed"]
        )
        self.assertTrue(calibrations[0]["thermal_start_qualification"]["passed"])
        self.assertTrue(preconditions[0]["label"].endswith("attempt-02"))
        self.assertEqual(
            calibrations[0]["measurement_release_monotonic_ns"],
            1_100_000_000,
        )

    def test_policy_epoch_zero_preloads_all_processes_before_preheater(
        self,
    ) -> None:
        events: list[str] = []

        class Process:
            def __init__(self, pid: int) -> None:
                self.pid = pid
                self.returncode = 0

            @staticmethod
            def poll() -> int:
                return 0

        worker_process = Process(111)
        critical_process = Process(222)
        worker = GOVERNOR.RunningWorker(
            GOVERNOR.WorkerAction(0, "language", "resident-1g", 100),
            worker_process,
            0,
            "small",
            100,
        )

        class Monitor:
            def mark(self, name: str, _metadata: dict) -> object:
                events.append(f"mark:{name}")
                timestamp = {
                    "policy_start": 0,
                    "epoch_prepare": 1,
                    "thermal_start_qualification_result": 1_050_000_000,
                    "measurement_start": 1_100_000_000,
                    "cleanup_end": 2_000_000_000,
                }[name]
                return types.SimpleNamespace(monotonic_ns=timestamp)

        def start_workers(*_args: object, **_kwargs: object) -> list:
            events.append("spawn-workers")
            return [worker]

        def spawn_critical(*_args: object, **_kwargs: object) -> Process:
            events.append("spawn-critical")
            return critical_process

        def wait(processes: list[Process], _timeout: float) -> None:
            self.assertEqual(processes, [worker_process, critical_process])
            events.append("all-paused")

        def preheat(*_args: object, **kwargs: object) -> dict:
            self.assertEqual(
                kwargs["label"], "pre-static-mig-epoch-00-attempt-01"
            )
            self.assertIn("all-paused", events)
            events.append("preheat")
            return {
                "label": kwargs["label"],
                "measurement_start_monotonic_ns": 0,
                "measurement_end_monotonic_ns": 1_000_000_000,
                "cleanup_end_monotonic_ns": 1_020_000_000,
                "last_window": {
                    "samples": 480,
                    "window_seconds": 60.0,
                    "observed_span_seconds": 59.4,
                    "mean_c": 75.0,
                    "min_c": 75.0,
                    "max_c": 75.0,
                    "latest_c": 75.0,
                    "slope_c_per_minute": 0.0,
                    "maximum_gap_seconds": 0.1,
                },
            }

        def resume(processes: list[Process]) -> None:
            if processes == [worker_process]:
                events.append("resume-workers")
            elif processes == [critical_process]:
                self.assertIn("resume-workers", events)
                events.append("resume-critical")
            else:
                self.fail(f"unexpected resume set: {processes}")

        def qualify(*_args: object, **kwargs: object) -> dict:
            events.append("qualify")
            return self.successful_thermal_qualification(
                attempt=int(kwargs["attempt"])
            )

        def reject_after_release(*_args: object, **_kwargs: object) -> None:
            self.assertIn("resume-critical", events)
            raise RuntimeError("stop-after-order-check")

        args = self.parsed_args()
        args.epochs = 1
        args.thermal_target_c = 75.0
        args.thermal_hard_limit_c = 104.0
        patches = (
            mock.patch.object(GOVERNOR, "start_workers", side_effect=start_workers),
            mock.patch.object(
                GOVERNOR.subprocess, "Popen", side_effect=spawn_critical
            ),
            mock.patch.object(GOVERNOR, "wait_until_paused", side_effect=wait),
            mock.patch.object(
                GOVERNOR,
                "require_processes_paused",
                return_value={"111": "T", "222": "T"},
            ),
            mock.patch.object(
                GOVERNOR,
                "process_affinity_snapshot",
                return_value={"tasks": []},
            ),
            mock.patch.object(GOVERNOR, "run_thermal_load", side_effect=preheat),
            mock.patch.object(GOVERNOR, "resume_processes", side_effect=resume),
            mock.patch.object(GOVERNOR, "wait_until_resumed"),
            mock.patch.object(
                GOVERNOR,
                "qualify_thermal_start",
                side_effect=qualify,
            ),
            mock.patch.object(
                GOVERNOR, "collect_json", side_effect=reject_after_release
            ),
            mock.patch.object(GOVERNOR, "terminate_process"),
            mock.patch.object(GOVERNOR, "stop_workers", return_value=[]),
            mock.patch.object(
                GOVERNOR, "engine_path", return_value=pathlib.Path("/tmp/engine")
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
            5
        ], patches[6], patches[7], patches[8], patches[9], patches[10], patches[
            11
        ], patches[12], self.assertRaisesRegex(
            RuntimeError, "stop-after-order-check"
        ):
            GOVERNOR.run_policy(
                "static-mig",
                args,
                {},
                {"JDG_MIG_BIG_UUID": "big"},
                {},
                5.0,
                {},
                Monitor(),
            )

        self.assertLess(events.index("spawn-workers"), events.index("preheat"))
        self.assertLess(events.index("spawn-critical"), events.index("preheat"))
        self.assertLess(events.index("all-paused"), events.index("preheat"))
        self.assertLess(
            events.index("preheat"), events.index("mark:measurement_start")
        )
        self.assertLess(events.index("qualify"), events.index("mark:measurement_start"))
        self.assertLess(
            events.index("mark:measurement_start"), events.index("resume-workers")
        )
        self.assertLess(events.index("resume-workers"), events.index("resume-critical"))

    def test_jain_fairness(self) -> None:
        self.assertEqual(GOVERNOR.jain_fairness([10, 10]), 1.0)
        self.assertIsNone(GOVERNOR.jain_fairness([0, 0]))
        self.assertAlmostEqual(GOVERNOR.jain_fairness([10, 0]), 0.5)

    def test_service_rate_uses_each_worker_measurement_window(self) -> None:
        workers = [
            {
                "completed_requests": 100,
                "elapsed_seconds": 1.0,
                "placement": "resident-1g",
            },
            {
                "completed_requests": 100,
                "elapsed_seconds": 2.0,
                "placement": "resident-1g",
            },
        ]
        self.assertEqual(
            GOVERNOR.rate_by(workers, "placement", "resident-1g"),
            150.0,
        )

    def test_policy_rate_is_weighted_by_critical_window(self) -> None:
        epochs = [
            {"measurement_seconds": 1.0, "rate": 100.0},
            {"measurement_seconds": 3.0, "rate": 200.0},
        ]
        self.assertEqual(
            GOVERNOR.time_weighted_epoch_rate(epochs, "rate"), 175.0
        )

    def test_pooled_trace_uses_type7_percentile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.csv"
            path.write_text(
                "request,release_to_completion_ms\n"
                "0,1\n1,2\n2,3\n3,4\n4,5\n",
                encoding="utf-8",
            )
            p99, samples = GOVERNOR.pooled_trace_p99([path])
        self.assertEqual(samples, 5)
        self.assertAlmostEqual(p99, 4.96)

    def test_minimum_telemetry_sample_fraction(self) -> None:
        self.assertEqual(
            GOVERNOR.minimum_telemetry_samples(0, 1_000_000_000), 8
        )

    def test_thermal_sensor_roles_and_handoff_bound_are_fixed(self) -> None:
        args = self.parsed_args()
        self.assertEqual(args.thermal_stability_sensor, "soc012")
        self.assertEqual(args.thermal_safety_sensor, "tj")
        self.assertEqual(args.thermal_handoff_max_ms, 500.0)
        self.assertEqual(
            GOVERNOR.THERMAL_HANDOFF_BOUNDARY,
            "thermal_measurement_end",
        )
        self.assertEqual(GOVERNOR.THERMAL_QUALIFICATION_MAX_ATTEMPTS, 3)
        self.assertEqual(GOVERNOR.THERMAL_ACTIVE_STABLE_ENDPOINTS, 3)
        self.assertEqual(GOVERNOR.THERMAL_ACTIVE_STABLE_SPACING_SECONDS, 1.0)
        self.assertIn(
            "temperature:soc012", GOVERNOR.TELEMETRY_REQUIRED_FIELDS
        )
        self.assertIn("temperature:tj", GOVERNOR.TELEMETRY_REQUIRED_FIELDS)

        args.thermal_handoff_max_ms = 499.0
        with self.assertRaisesRegex(SystemExit, "invalid experiment dimensions"):
            GOVERNOR.validate_args(args)

    def test_thermal_handoff_rejects_either_500_ms_boundary(self) -> None:
        qualification = self.successful_thermal_qualification()
        evidence = GOVERNOR.validate_thermal_qualification_handoff(
            qualification,
            1_050_000_000,
            1_100_000_000,
            1_499_999_999,
        )
        self.assertTrue(evidence["strictly_within_bound"])
        self.assertEqual(
            evidence["boundary"], "thermal_measurement_end"
        )
        self.assertLess(evidence["boundary_to_measurement_start_ms"], 500.0)

        for release_ns, measurement_start_ns in (
            (1_500_000_000, 1_500_000_000),
            (1_100_000_000, 1_500_000_000),
        ):
            with self.subTest(
                release_ns=release_ns,
                measurement_start_ns=measurement_start_ns,
            ), self.assertRaisesRegex(RuntimeError, "strict bound"):
                GOVERNOR.validate_thermal_qualification_handoff(
                    qualification,
                    1_050_000_000,
                    release_ns,
                    measurement_start_ns,
                )

        failed = dict(qualification)
        failed["passed"] = False
        failed["failure_reason"] = "unstable"
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            GOVERNOR.validate_thermal_qualification_handoff(
                failed, 1_050_000_000, 1_100_000_000, 1_200_000_000
            )

    def test_thermal_qualification_selects_first_postcleanup_sample(self) -> None:
        stream = io.StringIO()
        now_ns = [1_100_000_000]
        monitor = GOVERNOR.TegrastatsMonitor(
            GOVERNOR.JsonlTelemetryWriter(stream),
            clock=lambda: now_ns[0] + 1,
            mem_available_reader=lambda: 1.0,
        )
        line = "RAM 1/2MB CPU [0%@1] soc012@75C tj@90C VIN 1mW"
        def advance(_seconds: float) -> None:
            now_ns[0] += 100_000_000
            monitor.record_line(
                line,
                monotonic_ns=now_ns[0],
                mem_available_mb=1.0,
            )

        args = self.parsed_args()
        args.thermal_target_c = 75.0
        args.thermal_hard_limit_c = 104.0
        result = GOVERNOR.qualify_thermal_start(
            args,
            monitor,
            label="exact-marker",
            attempt=1,
            boundary_ns=1_000_000_000,
            cleanup_end_ns=1_100_000_000,
            clock_ns=lambda: now_ns[0],
            sleep=advance,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["sample_monotonic_ns"], 1_200_000_000)
        self.assertEqual(result["qualification_monotonic_ns"], 1_200_000_001)
        self.assertEqual(result["boundary_monotonic_ns"], 1_000_000_000)
        self.assertTrue(result["telemetry"]["interval"]["end_inclusive"])
        marker = [
            record
            for record in map(json.loads, stream.getvalue().splitlines())
            if record["record_type"] == "marker"
        ][-1]
        self.assertEqual(marker["name"], "thermal_start_qualification")
        self.assertEqual(
            marker["metadata"],
            {
                "label": "exact-marker",
                "attempt": 1,
                "boundary_monotonic_ns": 1_000_000_000,
                "cleanup_end_monotonic_ns": 1_100_000_000,
                "sample_monotonic_ns": 1_200_000_000,
            },
        )

    def test_thermal_qualification_fails_without_fresh_causal_sample(self) -> None:
        now_ns = [1_100_000_000]
        monitor = GOVERNOR.TegrastatsMonitor(
            GOVERNOR.JsonlTelemetryWriter(io.StringIO()),
            clock=lambda: now_ns[0] + 1,
            mem_available_reader=lambda: 1.0,
        )
        def advance_without_sample(_seconds: float) -> None:
            now_ns[0] += 100_000_000

        args = self.parsed_args()
        args.thermal_target_c = 75.0
        args.thermal_hard_limit_c = 104.0
        result = GOVERNOR.qualify_thermal_start(
            args,
            monitor,
            label="duplicate-poll",
            attempt=1,
            boundary_ns=1_000_000_000,
            cleanup_end_ns=1_100_000_000,
            clock_ns=lambda: now_ns[0],
            sleep=advance_without_sample,
        )

        self.assertFalse(result["passed"])
        self.assertIsNone(result["sample_monotonic_ns"])
        self.assertIn("no causal", result["failure_reason"])

    def test_thermal_qualification_does_not_skip_bad_first_sample(self) -> None:
        monitor = GOVERNOR.TegrastatsMonitor(
            GOVERNOR.JsonlTelemetryWriter(io.StringIO()),
            clock=lambda: 1_300_000_000,
            mem_available_reader=lambda: 1.0,
        )
        monitor.record_line(
            "RAM 1/2MB CPU [0%@1] soc012@70C tj@90C VIN 1mW",
            monotonic_ns=1_200_000_000,
            mem_available_mb=1.0,
        )
        monitor.record_line(
            "RAM 1/2MB CPU [0%@1] soc012@75C tj@90C VIN 1mW",
            monotonic_ns=1_250_000_000,
            mem_available_mb=1.0,
        )
        args = self.parsed_args()
        args.thermal_target_c = 75.0
        args.thermal_hard_limit_c = 104.0
        result = GOVERNOR.qualify_thermal_start(
            args,
            monitor,
            label="first-causal",
            attempt=1,
            boundary_ns=1_000_000_000,
            cleanup_end_ns=1_100_000_000,
            clock_ns=lambda: 1_300_000_000,
            sleep=lambda _seconds: None,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["sample_monotonic_ns"], 1_200_000_000)
        self.assertEqual(result["stability_value_c"], 70.0)

    def test_actual_start_selects_latest_causal_sample(self) -> None:
        def evaluate(latest_c: float) -> dict:
            monitor = GOVERNOR.TegrastatsMonitor(
                GOVERNOR.JsonlTelemetryWriter(io.StringIO()),
                clock=lambda: 1_400_000_000,
                mem_available_reader=lambda: 1.0,
            )
            for timestamp, value in (
                (1_200_000_000, 75.0),
                (1_300_000_000, latest_c),
            ):
                monitor.record_line(
                    "RAM 1/2MB CPU [0%@1] "
                    f"soc012@{value}C tj@90C VIN 1mW",
                    monotonic_ns=timestamp,
                    mem_available_mb=1.0,
                )
            args = self.parsed_args()
            args.thermal_target_c = 75.0
            args.thermal_hard_limit_c = 104.0
            return GOVERNOR.validate_actual_thermal_start(
                args,
                monitor,
                label="actual",
                measurement_start_ns=1_350_000_000,
                window_not_before_ns=1_100_000_000,
            )

        passed = evaluate(75.5)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["sample_monotonic_ns"], 1_300_000_000)
        failed = evaluate(77.0)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["stability_value_c"], 77.0)

    def test_thermal_start_uses_soc012_but_tj_enforces_safety(self) -> None:
        class Monitor:
            def __init__(self, tj_max: float) -> None:
                self.tj_max = tj_max

            def samples(self) -> list[object]:
                parsed = types.SimpleNamespace(
                    temperatures_c={"soc012": 75.0, "tj": self.tj_max}
                )
                return [
                    types.SimpleNamespace(
                        monotonic_ns=1_000_000_000 + index * 100_000_000,
                        parsed=parsed,
                    )
                    for index in range(1, 601)
                ]

            def aggregate(self, *_args: object, **kwargs: object) -> dict:
                return {
                    "valid_samples": 600,
                    "health": {
                        "healthy": True,
                        "required_fields": list(
                            kwargs.get("required_fields", ())
                        ),
                    },
                    "temperatures_c": {
                        "soc012": {"max": 75.0},
                        "tj": {"max": self.tj_max},
                    },
                }

        args = self.parsed_args()
        args.thermal_target_c = 75.0
        args.thermal_hard_limit_c = 104.0
        summary, _ = GOVERNOR.replay_thermal_start(
            args,
            Monitor(103.9),
            label="sensor-split",
            reference_ns=61_000_000_000,
            not_before_ns=1_000_000_000,
        )
        self.assertEqual(summary["mean_c"], 75.0)
        with self.assertRaisesRegex(RuntimeError, "hard limit"):
            GOVERNOR.replay_thermal_start(
                args,
                Monitor(104.0),
                label="sensor-split",
                reference_ns=61_000_000_000,
                not_before_ns=1_000_000_000,
            )

    def test_thermal_pilot_checkpoint_requires_every_stability_bound(self) -> None:
        window = {
            "samples": 1440,
            "window_seconds": 180.0,
            "observed_span_seconds": 178.2,
            "mean_c": 89.0,
            "min_c": 88.5,
            "max_c": 89.5,
            "latest_c": 89.0,
            "slope_c_per_minute": 0.2,
            "maximum_gap_seconds": 0.3,
        }
        telemetry = {
            "valid_samples": 1440,
            "health": {
                "healthy": True,
                "required_fields": list(GOVERNOR.TELEMETRY_REQUIRED_FIELDS),
            },
            "temperatures_c": {"tj": {"max": 89.5}},
        }
        self.assertTrue(
            GOVERNOR.thermal_pilot_checkpoint_is_stable(
                window,
                telemetry,
                hard_limit_c=104.0,
                window_seconds=180.0,
                maximum_slope_c_per_minute=0.2,
            )
        )
        failures = (
            ("samples", 1439),
            ("observed_span_seconds", 178.199),
            ("slope_c_per_minute", 0.201),
            ("maximum_gap_seconds", 0.301),
        )
        for field, value in failures:
            with self.subTest(field=field):
                candidate = dict(window)
                candidate[field] = value
                self.assertFalse(
                    GOVERNOR.thermal_pilot_checkpoint_is_stable(
                        candidate,
                        telemetry,
                        hard_limit_c=104.0,
                        window_seconds=180.0,
                        maximum_slope_c_per_minute=0.2,
                    )
                )
        hot_safety = dict(telemetry)
        hot_safety["temperatures_c"] = {"tj": {"max": 104.0}}
        self.assertFalse(
            GOVERNOR.thermal_pilot_checkpoint_is_stable(
                window,
                hot_safety,
                hard_limit_c=104.0,
                window_seconds=180.0,
                maximum_slope_c_per_minute=0.2,
            )
        )
        unhealthy = dict(telemetry)
        unhealthy["health"] = {
            "healthy": False,
            "required_fields": list(GOVERNOR.TELEMETRY_REQUIRED_FIELDS),
        }
        self.assertFalse(
            GOVERNOR.thermal_pilot_checkpoint_is_stable(
                window,
                unhealthy,
                hard_limit_c=104.0,
                window_seconds=180.0,
                maximum_slope_c_per_minute=0.2,
            )
        )

    def test_thermal_checkpoint_rejects_missed_or_catch_up_boundaries(self) -> None:
        self.assertTrue(
            GOVERNOR.thermal_stability_checkpoint_is_timely(30.0, 30.0)
        )
        self.assertTrue(
            GOVERNOR.thermal_stability_checkpoint_is_timely(31.0, 30.0)
        )
        self.assertFalse(
            GOVERNOR.thermal_stability_checkpoint_is_timely(31.001, 30.0)
        )
        self.assertFalse(
            GOVERNOR.thermal_stability_checkpoint_is_timely(29.999, 30.0)
        )

    def test_thermal_pilot_formal_dimensions_and_timeout_are_fixed(self) -> None:
        args = self.parsed_args()
        args.telemetry_log = pathlib.Path("/tmp/tegrastats.txt")
        args.thermal_pilot_seconds = 600.0
        args.thermal_window_seconds = 180.0
        self.assertEqual(GOVERNOR.validate_args(args), list(GOVERNOR.POLICIES))

        invalid = (
            ("thermal_pilot_seconds", 599.0),
            ("thermal_pilot_seconds", 900.0),
            ("thermal_window_seconds", 179.0),
            ("thermal_timeout_seconds", 901.0),
            ("thermal_max_slope_c_per_minute", 0.21),
        )
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                candidate = self.parsed_args()
                candidate.telemetry_log = pathlib.Path("/tmp/tegrastats.txt")
                candidate.thermal_pilot_seconds = 600.0
                candidate.thermal_window_seconds = 180.0
                setattr(candidate, field, value)
                with self.assertRaisesRegex(
                    SystemExit, "invalid experiment dimensions"
                ):
                    GOVERNOR.validate_args(candidate)

        target = self.parsed_args()
        target.telemetry_log = pathlib.Path("/tmp/tegrastats.txt")
        target.thermal_target_c = 89.0
        target.thermal_lock_sha256 = "a" * 64
        self.assertEqual(
            target.thermal_window_seconds, 60.0
        )
        self.assertEqual(GOVERNOR.validate_args(target), list(GOVERNOR.POLICIES))

    def test_duration_pilot_emits_ordered_checks_and_stops_at_600_seconds(
        self,
    ) -> None:
        class Clock:
            def __init__(self) -> None:
                self.now_ns = 1_000_000_000

            def monotonic_ns(self) -> int:
                return self.now_ns

            def monotonic(self) -> float:
                return self.now_ns / 1_000_000_000.0

            def sleep(self, _seconds: float) -> None:
                self.now_ns += 30_000_000_000

        class Monitor:
            def __init__(self, clock: Clock) -> None:
                self.clock = clock
                self.start_ns: int | None = None
                self.markers: list[tuple[str, dict, int]] = []

            def mark(self, name: str, metadata: dict) -> object:
                if name == "thermal_start":
                    self.start_ns = self.clock.now_ns
                self.markers.append((name, metadata, self.clock.now_ns))
                return types.SimpleNamespace(monotonic_ns=self.clock.now_ns)

            def samples(self) -> list[object]:
                if self.start_ns is None:
                    return []
                count = max(
                    0,
                    (self.clock.now_ns - self.start_ns) // 100_000_000,
                )
                parsed = types.SimpleNamespace(
                    temperatures_c={"soc012": 75.0, "tj": 89.0}
                )
                return [
                    types.SimpleNamespace(
                        monotonic_ns=self.start_ns + index * 100_000_000,
                        parsed=parsed,
                    )
                    for index in range(1, count + 1)
                ]

            def aggregate(
                self, start_ns: int, end_ns: int, **kwargs: object
            ) -> dict:
                sample_count = max(0, (end_ns - start_ns) // 100_000_000)
                minimum = int(kwargs.get("minimum_valid_samples", 1))
                healthy = sample_count >= minimum
                return {
                    "valid_samples": sample_count,
                    "health": {
                        "healthy": healthy,
                        "required_fields": list(
                            kwargs.get("required_fields", ())
                        ),
                        "reasons": [] if healthy else ["insufficient_valid_samples"],
                    },
                    "temperatures_c": {
                        "soc012": {"max": 75.0},
                        "tj": {"max": 89.0},
                    },
                }

        args = self.parsed_args()
        args.thermal_window_seconds = 180.0
        args.thermal_timeout_seconds = 900.0
        args.thermal_hard_limit_c = 104.0
        clock = Clock()
        monitor = Monitor(clock)
        patches = (
            mock.patch.object(GOVERNOR, "start_workers", return_value=[]),
            mock.patch.object(GOVERNOR, "wait_until_paused"),
            mock.patch.object(GOVERNOR, "resume_processes"),
            mock.patch.object(GOVERNOR, "stop_workers", return_value=[]),
            mock.patch.object(GOVERNOR.time, "monotonic_ns", clock.monotonic_ns),
            mock.patch.object(GOVERNOR.time, "monotonic", clock.monotonic),
            mock.patch.object(GOVERNOR.time, "sleep", clock.sleep),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
            5
        ], patches[6]:
            result = GOVERNOR.run_thermal_load(
                args,
                {},
                {},
                monitor,
                label="thermal-pilot",
                duration_seconds=600.0,
            )

        checks = [
            metadata
            for name, metadata, _ in monitor.markers
            if name == "thermal_stability_check"
        ]
        boundaries = [
            (metadata, timestamp)
            for name, metadata, timestamp in monitor.markers
            if name == "thermal_stability_boundary"
        ]
        self.assertEqual(len(checks), 20)
        self.assertEqual(len(boundaries), 20)
        self.assertEqual(
            [
                name
                for name, _, _ in monitor.markers
                if name.startswith("thermal_stability_")
            ],
            [
                name
                for _ in range(20)
                for name in (
                    "thermal_stability_boundary",
                    "thermal_stability_check",
                )
            ],
        )
        self.assertEqual(
            [item["checkpoint_index"] for item in checks], list(range(20))
        )
        self.assertEqual(
            [item["scheduled_elapsed_seconds"] for item in checks],
            [30.0 * index for index in range(1, 21)],
        )
        self.assertEqual(
            set(checks[-1]),
            {
                "label",
                "checkpoint_index",
                "scheduled_elapsed_seconds",
                "actual_elapsed_seconds",
                "checkpoint_monotonic_ns",
                "passed",
                "consecutive_passes",
                "window",
            },
        )
        self.assertTrue(checks[-1]["passed"])
        self.assertGreaterEqual(checks[-1]["consecutive_passes"], 3)
        for check, (boundary, timestamp) in zip(
            checks, boundaries, strict=True
        ):
            self.assertEqual(
                set(boundary),
                {
                    "label",
                    "checkpoint_index",
                    "scheduled_elapsed_seconds",
                },
            )
            self.assertEqual(check["checkpoint_monotonic_ns"], timestamp)
        measurement_end = next(
            metadata
            for name, metadata, _ in monitor.markers
            if name == "thermal_measurement_end"
        )
        self.assertEqual(measurement_end["checkpoint_index"], 19)
        self.assertEqual(measurement_end["scheduled_elapsed_seconds"], 600.0)
        self.assertEqual(
            measurement_end["checkpoint_monotonic_ns"],
            checks[-1]["checkpoint_monotonic_ns"],
        )
        self.assertEqual(result["duration_seconds"], 600.0)
        self.assertEqual(result["stability_checks"], checks)
        self.assertEqual(result["termination_reason"], "stable-checkpoints")

    def test_target_preheater_requires_three_spaced_active_endpoints(self) -> None:
        class Clock:
            def __init__(self) -> None:
                self.now_ns = 1_000_000_000

            def monotonic_ns(self) -> int:
                return self.now_ns

            def monotonic(self) -> float:
                return self.now_ns / 1_000_000_000.0

            def sleep(self, _seconds: float) -> None:
                self.now_ns += 1_000_000_000

        class Monitor:
            def __init__(self, clock: Clock) -> None:
                self.clock = clock
                self.markers: list[tuple[str, dict, int]] = []

            def mark(self, name: str, metadata: dict) -> object:
                self.markers.append((name, metadata, self.clock.now_ns))
                return types.SimpleNamespace(monotonic_ns=self.clock.now_ns)

            def sample_window(self, *_args: object, **_kwargs: object) -> object:
                parsed = types.SimpleNamespace(
                    temperatures_c={"soc012": 75.0, "tj": 89.0}
                )
                sample = types.SimpleNamespace(
                    monotonic_ns=self.clock.now_ns,
                    parsed=parsed,
                )
                return types.SimpleNamespace(
                    samples=(sample,), interval_complete=True
                )

            @staticmethod
            def aggregate(*_args: object, **kwargs: object) -> dict:
                return {
                    "valid_samples": 600,
                    "health": {
                        "healthy": True,
                        "required_fields": list(kwargs.get("required_fields", ())),
                        "reasons": [],
                    },
                    "temperatures_c": {
                        "soc012": {"max": 75.0},
                        "tj": {"max": 89.0},
                    },
                }

        stable_window = {
            "samples": 600,
            "window_seconds": 1.0,
            "observed_span_seconds": 1.0,
            "mean_c": 75.0,
            "min_c": 75.0,
            "max_c": 75.0,
            "latest_c": 75.0,
            "slope_c_per_minute": 0.0,
            "maximum_gap_seconds": 0.1,
        }
        args = self.parsed_args()
        args.thermal_target_c = 75.0
        args.thermal_window_seconds = 1.0
        args.thermal_timeout_seconds = 10.0
        args.thermal_hard_limit_c = 104.0
        clock = Clock()
        monitor = Monitor(clock)
        patches = (
            mock.patch.object(GOVERNOR, "start_workers", return_value=[]),
            mock.patch.object(GOVERNOR, "wait_until_paused"),
            mock.patch.object(GOVERNOR, "resume_processes"),
            mock.patch.object(GOVERNOR, "stop_workers", return_value=[]),
            mock.patch.object(
                GOVERNOR,
                "_thermal_start_evidence",
                return_value=(stable_window, {"health": {"healthy": True}}, None),
            ),
            mock.patch.object(GOVERNOR.time, "monotonic_ns", clock.monotonic_ns),
            mock.patch.object(GOVERNOR.time, "monotonic", clock.monotonic),
            mock.patch.object(GOVERNOR.time, "sleep", clock.sleep),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
            5
        ], patches[6], patches[7]:
            result = GOVERNOR.run_thermal_load(
                args,
                {},
                {},
                monitor,
                label="active-target",
                target_c=75.0,
            )

        checks = [
            metadata
            for name, metadata, _timestamp in monitor.markers
            if name == "thermal_active_stability_check"
        ]
        self.assertEqual(len(checks), 3)
        self.assertEqual([check["consecutive_passes"] for check in checks], [1, 2, 3])
        self.assertEqual(
            [check["sample_monotonic_ns"] for check in checks],
            [2_000_000_000, 3_000_000_000, 4_000_000_000],
        )
        self.assertEqual(result["active_stability_checks"], checks)
        self.assertEqual(result["active_stable_endpoints"], 3)
        self.assertEqual(result["termination_reason"], "active-stability-endpoints")
        measurement_end = next(
            metadata
            for name, metadata, _timestamp in monitor.markers
            if name == "thermal_measurement_end"
        )
        self.assertEqual(measurement_end["window"], stable_window)
        self.assertEqual(
            measurement_end["boundary_sample_monotonic_ns"],
            checks[-1]["sample_monotonic_ns"],
        )

    def test_platform_thermal_limit_stays_below_passive_trip(self) -> None:
        limit = GOVERNOR.platform_thermal_hard_limit_c()
        self.assertGreater(limit, 0.0)
        self.assertLess(limit, 109.0)


if __name__ == "__main__":
    unittest.main()
