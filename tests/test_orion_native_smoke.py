import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orion_native_smoke", ROOT / "baselines/orion/verify_native_smoke.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
UPSTREAM_COMMIT = MODULE.UPSTREAM_COMMIT
verify = MODULE.verify


class OrionNativeSmokeVerifierTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        client = {
            "schema_version": 1,
            "model": "resnet10-detection",
            "completed_requests": 2,
            "deadline_misses": 0,
            "config": {"priority": "low"},
            "execution_environment": {"cuda_visible_devices": "MIG-test"},
            "gpu": {"multiprocessors": 12},
        }
        high = json.loads(json.dumps(client))
        high["config"]["priority"] = "high"
        decisions = [
            {"schema_version": 1, "decision_sequence": 0,
             "arrival_sequence": 1, "client_id": 1, "priority": "high",
             "api": "cuLaunchKernelEx", "reordered": True,
             "profile_position": 0, "resource_class": -1, "sm_used": 0,
             "profile_duration_us": 0, "admission_reason": "unprofiled",
             "active_sm_at_admission": 0,
             "active_be_duration_us_at_admission": 0,
             "high_priority_active_at_admission": False,
             "initial_gate_clients": 2, "start_monotonic_ns": 10,
             "end_monotonic_ns": 11, "result": 0},
            {"schema_version": 1, "decision_sequence": 1,
             "arrival_sequence": 0, "client_id": 0,
             "priority": "best-effort", "api": "cuLaunchKernelEx",
             "reordered": False,
             "profile_position": 0, "resource_class": -1, "sm_used": 0,
             "profile_duration_us": 0, "admission_reason": "unprofiled",
             "active_sm_at_admission": 0,
             "active_be_duration_us_at_admission": 0,
             "high_priority_active_at_admission": False,
             "initial_gate_clients": 2,
             "start_monotonic_ns": 12, "end_monotonic_ns": 13,
             "result": 0},
        ]
        result = {
            "schema_version": 1,
            "kind": "orion-thor-native-positive-control",
            "upstream_commit": UPSTREAM_COMMIT,
            "port_stage": "driver-operation-software-queue",
            "numeric_comparison_allowed": False,
            "scheduler": {"arrivals": 2, "decisions": 2,
                          "reordered_decisions": 1},
            "best_effort": client,
            "high_priority": high,
        }
        launches = [
            {"sequence": index, "api": "cuLaunchKernelEx", "result": 0}
            for index in range(2)
        ]
        result_path = root / "result.json"
        decisions_path = root / "decisions.jsonl"
        launches_path = root / "launches.jsonl"
        result_path.write_text(json.dumps(result) + "\n")
        decisions_path.write_text("".join(json.dumps(row) + "\n" for row in decisions))
        launches_path.write_text("".join(json.dumps(row) + "\n" for row in launches))
        return result_path, decisions_path, launches_path

    def test_accepts_real_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            result = verify(*paths, 2, "MIG-test")
            self.assertEqual(result["reordered_decisions"], 1)

    def test_rejects_fifo_and_fake_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
            rows[0]["arrival_sequence"] = 0
            rows[1]["arrival_sequence"] = 1
            paths[1].write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "no later earlier arrival"):
                verify(*paths, 2, "MIG-test")


if __name__ == "__main__":
    unittest.main()
