import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize_p9_wall_load_sweep import summarize


def _summary(path: Path, system: str, rate: float, misses: int = 0) -> None:
    value = {
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-control",
        "background_period_ms": 1000.0 / rate,
        "background_offered_rps": rate,
        "latency_contract": "production-wall-arrival-to-completion",
        "deadline_mode": "wall", "checksum_mode": "inline", "iterations": 100,
        "deadline_lock": {"sha256": "a" * 64, "deadline_us": 770.605407},
        "results": [{"system": system, "pipeline_requests": 100,
                     "deadline_misses": misses, "pipeline_p99_us": 700.0,
                     "background_goodput_rps": rate}],
    }
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class WallLoadSweepTest(unittest.TestCase):
    def test_groups_points_and_selects_slo_frontier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for rate in (125.0, 250.0):
                for system in ("NVIDIA MPS", "XSched", "QUIET"):
                    path = root / f"{system}-{rate}.json"
                    _summary(path, system, rate, misses=100 if system == "XSched" else 0)
                    paths.append(path)
            result = summarize(paths)
            self.assertEqual(result["kind"], "p9-production-wall-load-sweep")
            self.assertEqual(result["frontier"]["QUIET"]["max_slo_qualified_offered_rps"], 250.0)
            self.assertIsNone(result["frontier"]["XSched"]["max_slo_qualified_offered_rps"])
            self.assertEqual(len(result["comparator_manifest"]["sha256"]), 64)
            self.assertTrue(result["frontier"]["QUIET"]["points"][0]["numeric_comparison_allowed"])

    def test_rejects_missing_system(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("NVIDIA MPS", "QUIET"):
                path = root / f"{system}.json"
                _summary(path, system, 125.0)
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "missing systems"):
                summarize(paths)

    def test_rejects_duplicate_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, system in enumerate(("NVIDIA MPS", "XSched", "QUIET")):
                path = root / f"{index}.json"
                _summary(path, system, 125.0)
                paths.append(path)
            duplicate = root / "duplicate.json"
            _summary(duplicate, "QUIET", 125.0)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                summarize(paths + [duplicate])

    def test_rejects_unbalanced_load_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("NVIDIA MPS", "XSched", "QUIET"):
                path = root / f"{system}-125.json"
                _summary(path, system, 125.0)
                paths.append(path)
            # A second load containing only QUIET must not be usable to claim
            # a higher frontier point than the comparators were tested at.
            extra = root / "quiet-250.json"
            _summary(extra, "QUIET", 250.0)
            with self.assertRaisesRegex(ValueError, "missing systems"):
                summarize(paths + [extra])

    def test_rejects_period_offered_load_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mismatch.json"
            _summary(path, "NVIDIA MPS", 125.0)
            value = json.loads(path.read_text())
            value["background_period_ms"] = 4.0
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "period and offered"):
                summarize([path], expected_systems={"NVIDIA MPS"})


if __name__ == "__main__":
    unittest.main()
