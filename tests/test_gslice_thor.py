#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gslice_thor", ROOT / "baselines/gslice/run_thor.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GsliceThorTest(unittest.TestCase):
    def test_algorithm_one_increases_underprovisioned_share(self) -> None:
        value = MODULE.compute_demand(50, 1.7, 2.0, 500, 400)
        self.assertGreater(value, 50)

    def test_algorithm_one_holds_inside_deadband(self) -> None:
        self.assertEqual(MODULE.compute_demand(50, 1.7, 1.68, 500, 495), 50)

    def test_deadband_ignores_arrival_jitter_before_reclaim(self) -> None:
        value = MODULE.compute_demand(50, 2.0, 1.0, 500, 499.95)
        self.assertLess(value, 50)

    def test_max_min_and_discrete_engine_projection(self) -> None:
        self.assertEqual(MODULE.max_min_pair(80, 20), (80, 20))
        self.assertEqual(MODULE.max_min_pair(80, 70), (50, 50))
        producer, background = MODULE.snap_pair(76, 24)
        self.assertEqual((producer, background), (75, 25))
        self.assertLessEqual(producer + background, 100)

    def test_next_allocation_reacts_to_pipeline_miss(self) -> None:
        updated = MODULE.next_allocation(
            (50, 50),
            deadline_ms=1.7,
            pipeline_p50_ms=2.0,
            background_period_ms=2.0,
            background_mean_ms=1.5,
            background_throughput_rps=520,
        )
        self.assertGreater(updated[0], updated[1])


if __name__ == "__main__":
    unittest.main()
