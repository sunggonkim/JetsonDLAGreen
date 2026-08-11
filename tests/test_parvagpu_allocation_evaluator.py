#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parva_eval", ROOT / "baselines/parvagpu/evaluate_allocation.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ParvaAllocationEvaluatorTest(unittest.TestCase):
    def test_summary_binds_measured_slos(self) -> None:
        allocation = {"allocation": [
            {"model": "resnet-producer", "physical_segment_gpc": 2},
            {"model": "distilbert-background", "physical_segment_gpc": 1},
        ]}
        result = MODULE.summarize(allocation, {
            "resnet-producer": {
                "release_to_completion": {"p99_ms": 0.5},
                "throughput_per_second": 499.0,
            },
            "distilbert-background": {
                "release_to_completion": {"p99_ms": 1.0},
                "throughput_per_second": 498.0,
            },
        }, 3.0)
        self.assertTrue(result["all_slos_met"])
        self.assertEqual(len(result["services"]), 2)


if __name__ == "__main__":
    unittest.main()
