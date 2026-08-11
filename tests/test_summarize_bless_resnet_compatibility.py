from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "analysis" / "summarize_bless_resnet_compatibility.py"
SPEC = importlib.util.spec_from_file_location("summarize_bless_resnet", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummarizeBlessResnetCompatibilityTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        q100 = root / "q100.engine"
        q100.write_bytes(b"q100")
        stderr = root / "stderr.txt"
        stderr.write_text("TensorRT: Myelin kErrorCuda\n")
        q25 = root / "q25.engine"
        q25.write_bytes(b"q25")
        q25_sha = MODULE.sha256(q25)
        lock = root / "lock.json"
        lock_value = {
            "kind": "bless-thor-tensorrt-safe-boundary-lock",
            "status": "frozen",
            "total_logical_launches": 18,
            "safe_switch_operations": [0, 6, 9, 15, 18],
            "selected_switch_operation": 9,
            "engine": {"sha256": q25_sha},
            "affinity_engines": [{"sha256": q25_sha} for _ in range(4)],
        }
        lock.write_text(json.dumps(lock_value))
        heldout = root / "heldout.json"
        heldout.write_text(json.dumps({
            "kind": "bless-thor-trt-squad-replica-functional-gate",
            "status": "passed",
            "logical_launches": 18,
            "physical_launches": 18,
            "shadow_launches": 54,
            "safe_switch_operation": 9,
            "activation_copies": 1,
            "engine": {"sha256": q25_sha},
            "boundary_lock": {"sha256": MODULE.sha256(lock)},
        }))
        schedule = root / "schedule.json"
        schedule.write_text(json.dumps({
            "kind": "bless-thor-common-tensorrt-profile-and-first-squad",
            "status": "profiled",
            "models": {"resnet": {"logical_launches": 18},
                       "distilbert": {"logical_launches": 47}},
            "squad": [{"request_id": "resnet", "kernel_index": 0}],
            "configuration": {"shares": None, "predicted_us": 1.0,
                              "estimator": "workload-equivalence"},
            "numeric_comparison_allowed": False,
        }))
        return q100, stderr, lock, heldout, schedule

    def test_summarizes_structural_incompatibility(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = MODULE.summarize(*self.fixture(Path(raw)))
            self.assertEqual(result["status"], "structural-incompatibility-characterized")
            self.assertFalse(result["numeric_comparison_allowed"])
            self.assertEqual(result["executable_q25_replica_plan"]["safe_switch_operations"],
                             [0, 6, 9, 15, 18])

    def test_rejects_missing_failure_signature(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            values = self.fixture(Path(raw))
            values[1].write_text("some other error\n")
            with self.assertRaisesRegex(ValueError, "failure signature differs"):
                MODULE.summarize(*values)


if __name__ == "__main__":
    unittest.main()
