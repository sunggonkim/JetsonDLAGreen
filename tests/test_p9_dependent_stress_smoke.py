#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dependent_stress", ROOT / "scripts" / "run_p9_dependent_stress_smoke.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DependentStressSmokeTest(unittest.TestCase):
    def test_common_workload_contract_rehashes_shared_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            arrival = root / "arrival.jsonl"
            dataset = root / "dataset.jsonl"
            arrival.write_text('{"arrival_sequence":0}\n', encoding="utf-8")
            dataset.write_text('{"sample_id":"0"}\n', encoding="utf-8")
            contract_path = root / "common.json"
            contract = {
                "schema_version": 1,
                "workload_id": "resnet-control",
                "topology": "fixed-2g+1g",
                "placement": "fixed-1g-producer-2g-consumer",
                "input_tensor": "features",
                "payload_bytes": 14720,
                "arrival_trace_path": str(arrival),
                "arrival_trace_sha256": hashlib.sha256(arrival.read_bytes()).hexdigest(),
                "dataset_manifest_path": str(dataset),
                "dataset_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
            contract_path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
            loaded = MODULE.load_common_workload_contract(
                contract_path,
                workload="resnet-control",
                placement="fixed-1g-producer-2g-consumer",
                input_tensor="features",
                payload_bytes=14720,
            )
            self.assertEqual(loaded["workload_id"], "resnet-control")
            self.assertEqual(loaded["contract_sha256"], hashlib.sha256(contract_path.read_bytes()).hexdigest())

    def test_common_workload_contract_rejects_tampered_or_wrong_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            arrival = root / "arrival.jsonl"
            dataset = root / "dataset.jsonl"
            arrival.write_text("arrival\n", encoding="utf-8")
            dataset.write_text("dataset\n", encoding="utf-8")
            contract = {
                "schema_version": 1,
                "workload_id": "resnet-control",
                "topology": "fixed-2g+1g",
                "placement": "fixed-1g-producer-2g-consumer",
                "input_tensor": "features",
                "payload_bytes": 14720,
                "arrival_trace_path": str(arrival),
                "arrival_trace_sha256": hashlib.sha256(arrival.read_bytes()).hexdigest(),
                "dataset_manifest_path": str(dataset),
                "dataset_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
            path = root / "common.json"
            path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "placement"):
                MODULE.load_common_workload_contract(
                    path,
                    workload="resnet-control",
                    placement="fixed-2g-producer-1g-consumer",
                    input_tensor="features",
                    payload_bytes=14720,
                )

    def test_common_workload_contract_binds_producer_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            arrival = root / "arrival.jsonl"
            dataset = root / "dataset.jsonl"
            producer = root / "inputs.bin"
            arrival.write_text("arrival\n", encoding="utf-8")
            dataset.write_text("dataset\n", encoding="utf-8")
            producer.write_bytes(b"bytes")
            contract = {
                "schema_version": 1,
                "workload_id": "resnet-control",
                "topology": "fixed-2g+1g",
                "placement": "fixed-1g-producer-2g-consumer",
                "input_tensor": "features",
                "payload_bytes": 14720,
                "arrival_trace_path": str(arrival),
                "arrival_trace_sha256": hashlib.sha256(arrival.read_bytes()).hexdigest(),
                "dataset_manifest_path": str(dataset),
                "dataset_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "producer_input_trace_path": str(producer),
                "producer_input_trace_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
            }
            path = root / "common.json"
            path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
            loaded = MODULE.load_common_workload_contract(
                path,
                workload="resnet-control",
                placement="fixed-1g-producer-2g-consumer",
                input_tensor="features",
                payload_bytes=14720,
            )
            self.assertEqual(loaded["producer_input_trace_path"], str(producer.resolve()))
            arrival.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "arrival_trace_path"):
                MODULE.load_common_workload_contract(
                    path,
                    workload="resnet-control",
                    placement="fixed-1g-producer-2g-consumer",
                    input_tensor="features",
                    payload_bytes=14720,
                )

    def test_summary_uses_payload_validated_pipeline(self) -> None:
        row = MODULE.summarize(
            MODULE.SCENARIOS[0],
            {
                "status": "ok",
                "checksum_failures": 0,
                "iterations": 100,
                "deadline_misses": 3,
                "end_to_end_us": {"p99": 800.0},
                "deadline_mode": "wall",
                "stage_latency_us": {
                    "producer_compute_p99": 600.0,
                    "transport_ready_p99": 50.0,
                    "consumer_compute_p99": 30.0,
                    "output_verification_p99": 20.0,
                },
                "payload_bytes": 14720,
                "unique_payload_checksums": 4,
                "unique_policy_output_checksums": 4,
            },
            {"throughput_per_second": 700.0},
        )
        self.assertEqual(row["deadline_misses"], 3)
        self.assertEqual(row["background_goodput_rps"], 700.0)
        self.assertEqual(row["payload_bytes"], 14720)
        self.assertEqual(row["unique_payload_checksums"], 4)
        self.assertEqual(row["deadline_mode"], "wall")

    def test_summary_rejects_corruption(self) -> None:
        with self.assertRaisesRegex(ValueError, "correctness"):
            MODULE.summarize(
                MODULE.SCENARIOS[0],
                {"status": "ok", "checksum_failures": 1},
                {"throughput_per_second": 1.0},
            )

    def test_summary_binds_post_completion_application_output_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "outputs.bin"
            output.write_bytes(b"JDGOUT1\x00")
            row = MODULE.summarize(
                MODULE.SCENARIOS[0],
                {
                    "status": "ok", "checksum_failures": 0, "iterations": 1,
                    "deadline_misses": 0, "end_to_end_us": {"p99": 10.0},
                    "deadline_mode": "wall", "stage_latency_us": {},
                    "payload_bytes": 14720, "unique_payload_checksums": 1,
                    "unique_policy_output_checksums": 1,
                },
                {"throughput_per_second": 1.0},
                application_output_trace=output,
            )
        self.assertEqual(
            row["application_output_trace"]["capture_boundary"],
            "post-completion",
        )

    def test_mechanism_scenarios_are_not_published_system_labels(self) -> None:
        self.assertEqual(
            tuple(item.name for item in MODULE.DEFAULT_SCENARIOS),
            (
                "nvidia-mig-isolation",
                "nvidia-mps-spatial-sharing",
                "static-full-gate",
                "process-stop-ablation",
                "quiet",
            ),
        )
        self.assertEqual(
            {MODULE.PUBLIC_SYSTEM_NAMES[item.name] for item in MODULE.MECHANISM_SCENARIOS},
            {"Quota-only provisioning", "Partition-only planning", "Full-DAG quiescence"},
        )

    def test_mps_baseline_preserves_fixed_cross_mig_topology(self) -> None:
        scenario = next(
            item for item in MODULE.DEFAULT_SCENARIOS
            if item.name == "nvidia-mps-spatial-sharing"
        )
        self.assertFalse(scenario.same_instance)
        self.assertIsNone(scenario.gate_mode)

    def test_real_resnet_detection_head_binds_learned_split_payload(self) -> None:
        self.assertEqual(MODULE.WORKLOAD_PAYLOAD_BYTES["resnet-detection-head"], 1_884_160)
        self.assertEqual(MODULE.WORKLOAD_PAYLOAD_BYTES["resnet50-classification"], 802_816)
        source = (ROOT / "scripts" / "run_p9_dependent_stress_smoke.py").read_text()
        self.assertIn('choices=("resnet-control", "resnet-detection-head", "resnet50-classification", "whisper-projection")', source)
        self.assertIn('"--producer-input-trace"', source)

    def test_transport_is_explicit_and_preserved_in_summary(self) -> None:
        row = MODULE.summarize(
            MODULE.SCENARIOS[-1],
            {
                "status": "ok", "checksum_failures": 0, "iterations": 1,
                "deadline_misses": 0, "end_to_end_us": {"p99": 10.0},
                "deadline_mode": "wall", "transport": "pinned-shared-sysmem-d2h-h2d",
                "stage_latency_us": {"validation_excluded_end_to_end_p99": 9.0},
                "payload_bytes": 14720, "unique_payload_checksums": 1,
                "unique_policy_output_checksums": 1,
            },
            {"throughput_per_second": 1.0},
        )
        self.assertEqual(row["transport"], "pinned-shared-sysmem-d2h-h2d")
        source = (ROOT / "scripts" / "run_p9_dependent_stress_smoke.py").read_text()
        self.assertIn('"--transport",\n                effective_transport', source)
        self.assertIn("actuate_selected_plan", source)

    def test_mig_baseline_does_not_admit_best_effort_on_reserved_slices(self) -> None:
        mig = MODULE.SCENARIOS[0]
        self.assertEqual(mig.name, "nvidia-mig-isolation")
        self.assertFalse(mig.best_effort_admitted)

    def test_static_full_gate_is_a_descriptive_same_slo_baseline(self) -> None:
        baseline = next(item for item in MODULE.DEFAULT_SCENARIOS if item.name == "static-full-gate")
        self.assertEqual(MODULE.PUBLIC_SYSTEM_NAMES[baseline.name], "Static full gating")
        self.assertEqual(baseline.gate_mode, "stop")
        self.assertEqual(baseline.gate_scope, "pipeline")
        self.assertTrue(baseline.best_effort_admitted)

    def test_performance_mode_is_marked_non_correctness_evidence(self) -> None:
        row = MODULE.summarize(
            MODULE.SCENARIOS[-1],
            {
                "status": "ok", "checksum_failures": 0, "iterations": 2,
                "deadline_misses": 0, "end_to_end_us": {"p99": 10.0},
                "deadline_mode": "wall", "correctness_validated": False,
                "checksum_mode": "off", "payload_bytes": 14720,
                "unique_payload_checksums": 0,
                "unique_policy_output_checksums": 0,
                "stage_latency_us": {"validation_excluded_end_to_end_p99": 9.0},
            },
            {"throughput_per_second": 1.0},
        )
        self.assertFalse(row["correctness_validated"])
        self.assertEqual(row["checksum_mode"], "off")

    def test_plan_violation_requires_explicit_diagnostic_mode(self) -> None:
        source = (ROOT / "scripts" / "run_p9_dependent_stress_smoke.py").read_text()
        self.assertIn("--allow-plan-diagnostic", source)
        self.assertIn("diagnostic-only-plan-violation", source)

    def test_quiet_plan_binds_large_edge_and_slack(self) -> None:
        deadline_lock = {"path": "/deadline.json", "sha256": "d" * 64}
        plan = {
            "proposed_system": "QUIET",
            "status": "selected",
            "deadline_us": 1700.0,
            "deadline_lock": deadline_lock,
            "selected_plan": {
                "candidate_id": "q100-q100",
                "feasible": True,
                "placement": {"producer": "1g-q100", "consumer": "2g-q100"},
                "reserved_slack_us": 20.0,
                "uncovered_guard_us": 0.0,
                "dag": {"edges": [{
                    "payload_bytes": 2_304_000,
                    "transport": "registered-shared-sysmem-direct-binding",
                }]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            loaded, provenance = MODULE.validate_quiet_plan(
                path, 1700.0, deadline_lock, 100, 100, "producer",
                "whisper-projection",
            )
        self.assertEqual(loaded["proposed_system"], "QUIET")
        self.assertEqual(len(provenance["sha256"]), 64)

    def test_quiet_plan_binds_reverse_placement_variant(self) -> None:
        deadline_lock = {"path": "/deadline.json", "sha256": "d" * 64}
        plan = {
            "proposed_system": "QUIET",
            "status": "selected",
            "deadline_us": 1700.0,
            "deadline_lock": deadline_lock,
            "selected_plan": {
                "candidate_id": "q100-q100@fixed-2g-producer-1g-consumer",
                "feasible": True,
                "placement": {"producer": "2g-q100", "consumer": "1g-q100"},
                "placement_variant": "fixed-2g-producer-1g-consumer",
                "reserved_slack_us": 20.0,
                "uncovered_guard_us": 0.0,
                "dag": {"edges": [{
                    "payload_bytes": 14720,
                    "transport": "registered-shared-sysmem-direct-binding",
                }]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            loaded, _ = MODULE.validate_quiet_plan(
                path, 1700.0, deadline_lock, 100, 100, "producer",
                "resnet-control", "fixed-2g-producer-1g-consumer",
            )
        self.assertEqual(
            loaded["selected_plan"]["placement_variant"],
            "fixed-2g-producer-1g-consumer",
        )

    def test_quiet_plan_binds_consumer_protection_scope(self) -> None:
        deadline_lock = {"path": "/deadline.json", "sha256": "d" * 64}
        plan = {
            "proposed_system": "QUIET",
            "status": "selected",
            "deadline_us": 1700.0,
            "deadline_lock": deadline_lock,
            "selected_plan": {
                "candidate_id": "q100-q100@fixed-2g-producer-1g-consumer",
                "feasible": True,
                "placement": {"producer": "2g-q100", "consumer": "1g-q100"},
                "placement_variant": "fixed-2g-producer-1g-consumer",
                "protection_scope": "consumer",
                "reserved_slack_us": 20.0,
                "uncovered_guard_us": 0.0,
                "dag": {"edges": [{
                    "payload_bytes": 14720,
                    "transport": "registered-shared-sysmem-direct-binding",
                }]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            loaded, _ = MODULE.validate_quiet_plan(
                path, 1700.0, deadline_lock, 100, 100, "consumer",
                "resnet-control", "fixed-2g-producer-1g-consumer",
            )
        self.assertEqual(loaded["selected_plan"]["protection_scope"], "consumer")

    def test_plan_scope_is_available_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "plan.json"
            path.write_text(json.dumps({"selected_plan": {"protection_scope": "consumer"}}))
            self.assertEqual(MODULE.quiet_plan_protection_scope(path), "consumer")
            path.write_text(json.dumps({"selected_plan": {"protection_scope": "invalid"}}))
            with self.assertRaisesRegex(ValueError, "protection_scope"):
                MODULE.quiet_plan_protection_scope(path)

    def test_consumer_scope_requires_dependent_mode(self) -> None:
        source = (ROOT / "scripts" / "run_p9_dependent_stress_smoke.py").read_text()
        self.assertIn('choices=("producer", "consumer", "pipeline")', source)
        self.assertIn("consumer protection scope requires dependent mode", source)

    def test_wall_contract_starts_at_request_arrival(self) -> None:
        source = (ROOT / "benchmarks" / "mig_trt_pipeline.cpp").read_text()
        self.assertIn("transfer.actual_release_ns = monotonic_ns();", source)
        self.assertIn("transfer.arrival_ns = transfer.actual_release_ns;", source)
        self.assertIn(
            "result.consumer_done_ns) -\n                result.transfer.arrival_ns",
            source,
        )
        self.assertIn(
            "arrival-to-consumer-completion-excludes-correctness-validation",
            source,
        )

    def test_deadline_lock_cannot_cross_placement_variants(self) -> None:
        lock = {"contract": {
            "producer_uuid": "small",
            "consumer_uuid": "big",
        }}
        mig = {"JDG_MIG_SMALL_UUID": "small", "JDG_MIG_BIG_UUID": "big"}
        MODULE.validate_deadline_placement(
            lock, mig, "fixed-1g-producer-2g-consumer"
        )
        with self.assertRaisesRegex(ValueError, "topology"):
            MODULE.validate_deadline_placement(
                lock, mig, "fixed-2g-producer-1g-consumer"
            )


if __name__ == "__main__":
    unittest.main()
