import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.compare_p9_placement_runs import compare


def _summary(placement: str, deadline: float = 1700.0) -> dict:
    return {
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-control",
        "latency_contract": "production-wall-arrival-to-completion",
        "deadline_mode": "wall",
        "checksum_mode": "inline",
        "deadline_us": deadline,
        "iterations": 10,
        "background_period_ms": 4.0,
        "background_offered_rps": 250.0,
        "placement_variant": placement,
        "deadline_lock": {"sha256": "a" * 64},
        "results": [{
            "system": "QUIET",
            "placement_variant": placement,
            "pipeline_requests": 10,
            "deadline_misses": 0,
            "wall_pipeline_p99_us": 100.0,
            "background_goodput_rps": 240.0,
            "correctness_validated": True,
            "unique_payload_checksums": 2,
            "unique_policy_output_checksums": 2,
        }],
    }


class PlacementComparisonTest(unittest.TestCase):
    def _write(self, directory: Path, name: str, value: dict) -> Path:
        path = directory / name
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def test_contract_drift_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = compare([
                self._write(root, "forward.json", _summary("fixed-1g-producer-2g-consumer", 1700.0)),
                self._write(root, "reverse.json", _summary("fixed-2g-producer-1g-consumer", 1600.0)),
            ])
        self.assertEqual(result["slo_comparison_status"], "not-comparable-contract-drift")

    def test_same_deadline_without_shared_lock_is_unbound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reverse = _summary("fixed-2g-producer-1g-consumer")
            reverse["deadline_lock"] = None
            result = compare([
                self._write(root, "forward.json", _summary("fixed-1g-producer-2g-consumer")),
                self._write(root, "reverse.json", reverse),
            ])
        self.assertTrue(result["deadline_equal"])
        self.assertFalse(result["lock_provenance_equal"])
        self.assertEqual(result["slo_comparison_status"], "same-deadline-unbound-lock")

    def test_same_contract_is_comparable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = compare([
                self._write(root, "forward.json", _summary("fixed-1g-producer-2g-consumer")),
                self._write(root, "reverse.json", _summary("fixed-2g-producer-1g-consumer")),
            ])
        self.assertTrue(result["contract_equal"])
        self.assertEqual(result["slo_comparison_status"], "comparable")

    def test_rejects_non_wall_or_wrong_system(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = _summary("fixed-1g-producer-2g-consumer")
            value["deadline_mode"] = "validation-excluded"
            with self.assertRaises(ValueError):
                compare([self._write(root, "bad.json", value), self._write(root, "ok.json", _summary("fixed-2g-producer-1g-consumer"))])


if __name__ == "__main__":
    unittest.main()
