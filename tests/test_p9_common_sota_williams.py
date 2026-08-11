import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "common_sota_williams", ROOT / "scripts/run_p9_common_sota_williams.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CommonSotaWilliamsTest(unittest.TestCase):
    def test_jdgint_trace_count_is_read_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "inputs.bin"
            trace.write_bytes(MODULE.JDGINT_MAGIC + MODULE.JDGINT_HEADER.pack(1, 7, 16))
            self.assertEqual(MODULE.producer_input_trace_count(trace), 7)

    def test_jdgint_trace_header_rejects_wrong_magic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "inputs.bin"
            trace.write_bytes(b"badbadbadbadbadbad" + struct.pack("<IIQ", 1, 7, 16))
            with self.assertRaisesRegex(ValueError, "magic"):
                MODULE.producer_input_trace_count(trace)

    def test_six_orders_balance_positions_and_predecessors(self) -> None:
        orders = MODULE.williams_orders()
        self.assertEqual(len(orders), 6)
        self.assertEqual(len(set(orders)), 6)
        for system in MODULE.SYSTEMS:
            self.assertEqual(
                [sum(order[position] == system for order in orders) for position in range(6)],
                [1] * 6,
            )
        predecessors = {
            (left, right): sum(
                order[index - 1] == left and order[index] == right
                for order in orders for index in range(1, 6)
            )
            for left in MODULE.SYSTEMS for right in MODULE.SYSTEMS if left != right
        }
        self.assertEqual(set(predecessors.values()), {1})

    def test_active_orders_are_only_numeric_frontier_rows(self) -> None:
        orders = MODULE.active_williams_orders()
        self.assertEqual(set(orders), {
            ("NVIDIA MPS", "XSched", "QUIET"),
            ("XSched", "QUIET", "NVIDIA MPS"),
            ("QUIET", "NVIDIA MPS", "XSched"),
        })
        for system in MODULE.ACTIVE_SYSTEMS:
            self.assertEqual(
                [sum(order[position] == system for order in orders) for position in range(3)],
                [1] * 3,
            )

    def test_active_runner_rejects_reverse_xsched_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "active XSched contract"):
                MODULE.main([
                    "--repo", str(ROOT),
                    "--result-dir", str(Path(directory) / "out"),
                    "--deadline-lock", str(Path(directory) / "lock.json"),
                    "--quiet-plan", str(Path(directory) / "plan.json"),
                    "--sequence-index", "0", "--active-only",
                    "--placement-variant", "fixed-2g-producer-1g-consumer",
                ])

    def test_only_quiet_is_the_proposed_public_name(self) -> None:
        self.assertEqual(MODULE.SYSTEMS[-1], "QUIET")
        self.assertNotIn("governor", " ".join(MODULE.SYSTEMS).lower())

    def test_comparison_contract_marks_noncomparable_controls(self) -> None:
        self.assertFalse(MODULE.COMPARISON_CONTRACTS["NVIDIA MIG"]["numeric_comparison_allowed"])
        self.assertFalse(MODULE.COMPARISON_CONTRACTS["Orion"]["numeric_comparison_allowed"])
        self.assertFalse(MODULE.COMPARISON_CONTRACTS["XSched"]["numeric_comparison_allowed"])
        self.assertFalse(MODULE.COMPARISON_CONTRACTS["gpulet"]["numeric_comparison_allowed"])
        self.assertTrue(MODULE.COMPARISON_CONTRACTS["QUIET"]["numeric_comparison_allowed"])
        self.assertEqual(
            MODULE.COMPARISON_CONTRACTS["NVIDIA MIG"]["topology"],
            "fixed-2g+1g-physical-isolation",
        )

    def test_xsched_receives_the_verified_deadline_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            lock = output / "deadline-lock.json"
            lock.write_text("{}\n", encoding="utf-8")

            def fake_run(command, *, cwd, env=None):
                self.assertEqual(env["DEADLINE_LOCK"], str(lock))
                self.assertEqual(
                    env["APPLICATION_OUTPUT_TRACE"],
                    str(output / "application-outputs.bin"),
                )

                self.assertNotIn("DEADLINE_US", env)
                (output / "verification.json").write_text(json.dumps({
                    "requests": 10,
                    "misses": 1,
                    "p99_us": 2.0,
                    "background_window": {"completion_goodput_rps": 3.0},
                    "deadline_mode": "wall",
                    "latency_contract": "production-wall-arrival-to-completion",
                    "production_wall_definition": MODULE.PRODUCTION_WALL_DEFINITION,
                    "correctness_validation_placement": MODULE.CORRECTNESS_PLACEMENT,
                }), encoding="utf-8")

            with mock.patch.object(MODULE, "run", side_effect=fake_run):
                row, evidence = MODULE.xsched_row(
                    ROOT, output, lock, 10, "whisper-projection"
                )
            self.assertEqual(row["system"], "XSched")
            self.assertEqual(row["background_goodput_rps"], 3.0)
            self.assertEqual(evidence, output / "verification.json")

    def test_xsched_receives_common_contract_and_consumer_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            lock = output / "deadline-lock.json"
            contract = output / "common-workload.json"
            engine = output / "head.engine"
            lock.write_text("{}\n", encoding="utf-8")
            contract.write_text("{}\n", encoding="utf-8")
            engine.write_bytes(b"engine")

            def fake_run(command, *, cwd, env=None):
                self.assertEqual(env["COMMON_WORKLOAD_CONTRACT"], str(contract))
                self.assertEqual(env["REQUIRE_COMMON_WORKLOAD"], "1")
                self.assertEqual(env["CONSUMER_ENGINE"], str(engine))
                (output / "verification.json").write_text(json.dumps({
                    "requests": 10,
                    "misses": 0,
                    "p99_us": 2.0,
                    "background_window": {"completion_goodput_rps": 3.0},
                    "deadline_mode": "wall",
                    "latency_contract": "production-wall-arrival-to-completion",
                    "production_wall_definition": MODULE.PRODUCTION_WALL_DEFINITION,
                    "correctness_validation_placement": MODULE.CORRECTNESS_PLACEMENT,
                }), encoding="utf-8")

            with mock.patch.object(MODULE, "run", side_effect=fake_run):
                MODULE.xsched_row(
                    ROOT, output, lock, 10, "resnet-detection-head",
                    contract, engine,
                )

    def test_common_workload_contract_is_active_sequence_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "common-workload.json"
            contract.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "active MPS/XSched/QUIET"):
                MODULE.main([
                    "--repo", str(ROOT),
                    "--result-dir", str(root / "out"),
                    "--deadline-lock", str(root / "lock.json"),
                    "--quiet-plan", str(root / "plan.json"),
                    "--sequence-index", "0",
                    "--common-workload-contract", str(contract),
                ])

    def test_active_learned_workload_requires_frozen_input_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "common-workload.json"
            contract.write_text("{}\n", encoding="utf-8")
            engine = root / "head.engine"
            engine.write_bytes(b"engine")
            with self.assertRaisesRegex(ValueError, "producer-input-trace"):
                MODULE.main([
                    "--repo", str(ROOT),
                    "--result-dir", str(root / "out"),
                    "--deadline-lock", str(root / "lock.json"),
                    "--quiet-plan", str(root / "plan.json"),
                    "--sequence-index", "0", "--active-only",
                    "--workload", "resnet50-classification",
                    "--consumer-engine", str(engine),
                    "--common-workload-contract", str(contract),
                ])

    def test_quiet_row_binds_requested_reverse_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            lock = output / "deadline-lock.json"
            plan = output / "selection.json"
            lock.write_text("{}\n", encoding="utf-8")
            plan.write_text("{}\n", encoding="utf-8")

            def fake_run(command, *, cwd, env=None):
                self.assertIn("--placement-variant", command)
                self.assertIn("fixed-2g-producer-1g-consumer", command)
                (output / "summary.json").write_text(json.dumps({
                    "results": [{
                        "system": "QUIET", "pipeline_requests": 10,
                        "deadline_misses": 0, "pipeline_p99_us": 4.0,
                        "background_goodput_rps": 5.0,
                        "deadline_mode": "wall",
                        "latency_contract": "production-wall-arrival-to-completion",
                        "production_wall_definition": MODULE.PRODUCTION_WALL_DEFINITION,
                        "correctness_validation_placement": MODULE.CORRECTNESS_PLACEMENT,
                    }],
                }), encoding="utf-8")

            with mock.patch.object(MODULE, "run", side_effect=fake_run):
                row, evidence = MODULE.base_row(
                    ROOT, output, "QUIET", lock, plan, 10,
                    "resnet-control", 4.0,
                    "fixed-2g-producer-1g-consumer",
                )
            self.assertEqual(row["system"], "QUIET")
            self.assertEqual(evidence, output / "summary.json")

    def test_active_xsched_rejects_validation_excluded_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            lock = output / "deadline-lock.json"
            lock.write_text("{}\n", encoding="utf-8")
            (output / "verification.json").write_text(json.dumps({
                "requests": 10,
                "misses": 0,
                "p99_us": 2.0,
                "background_window": {"completion_goodput_rps": 3.0},
                "deadline_mode": "validation-excluded",
            }), encoding="utf-8")
            with mock.patch.object(MODULE, "run"):
                with self.assertRaisesRegex(ValueError, "production-wall contract"):
                    MODULE.xsched_row(ROOT, output, lock, 10, "resnet-control")

    def test_active_xsched_accepts_wall_contract_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            lock = output / "deadline-lock.json"
            lock.write_text("{}\n", encoding="utf-8")
            (output / "verification.json").write_text(json.dumps({
                "requests": 10,
                "misses": 0,
                "p99_us": 2.0,
                "background_window": {"completion_goodput_rps": 3.0},
                "deadline_mode": "wall",
                "latency_contract": "production-wall-arrival-to-completion",
                "production_wall_definition": MODULE.PRODUCTION_WALL_DEFINITION,
                "correctness_validation_placement": MODULE.CORRECTNESS_PLACEMENT,
            }), encoding="utf-8")
            with mock.patch.object(MODULE, "run"):
                row, evidence = MODULE.xsched_row(
                    ROOT, output, lock, 10, "resnet-control"
                )
            self.assertEqual(row["deadline_mode"], "wall")
            self.assertEqual(evidence, output / "verification.json")

    def test_existing_xsched_row_reads_window_goodput(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evidence = output / "verification.json"
            evidence.write_text(json.dumps({
                "requests": 10,
                "misses": 10,
                "p99_us": 4.0,
                "background_window": {"completion_goodput_rps": 5.0},
                "deadline_mode": "wall",
                "latency_contract": "production-wall-arrival-to-completion",
                "production_wall_definition": MODULE.PRODUCTION_WALL_DEFINITION,
                "correctness_validation_placement": MODULE.CORRECTNESS_PLACEMENT,
            }), encoding="utf-8")
            row, loaded = MODULE.existing_row(output, "XSched")
            self.assertEqual(row["background_goodput_rps"], 5.0)
            self.assertEqual(loaded, evidence)


if __name__ == "__main__":
    unittest.main()
