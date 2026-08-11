#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "boer_independent_pair", ROOT / "baselines/boer/evaluate_independent_pair.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BoerIndependentPairTest(unittest.TestCase):
    def test_complementary_quota(self) -> None:
        self.assertEqual(MODULE.quota_pair(25), (25, 75))
        with self.assertRaisesRegex(ValueError, "complementary"):
            MODULE.quota_pair(100)

    def test_metrics_apply_published_p99_feasibility(self) -> None:
        producer = {
            "release_to_completion": {"p99_ms": 0.8},
            "throughput_per_second": 100.0,
        }
        background = {
            "release_to_completion": {"p99_ms": 1.5},
            "throughput_per_second": 90.0,
        }
        result = MODULE.metrics_from_results(producer, background, 2.0)
        self.assertEqual(result["feasible"], 1.0)
        self.assertEqual(result["worst_p99_ms"], 1.5)


if __name__ == "__main__":
    unittest.main()
