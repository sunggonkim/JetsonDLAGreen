import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quiet_heldout_load", ROOT / "analysis/summarize_p9_quiet_heldout_load.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QuietHeldoutLoadTest(unittest.TestCase):
    def test_replays_current_hardware_sweep(self) -> None:
        paths = []
        for rps in (125, 375, 500, 600, 650, 700, 750):
            candidates = sorted(ROOT.glob(f"results/p9-quiet-resnet-heldout-{rps}rps-1100r-*/summary.json"))
            if not candidates:
                self.skipTest("held-out hardware sweep is absent")
            paths.append(candidates[-1])
        result = MODULE.summarize(paths)
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(result["workload"], "resnet-control")
        self.assertEqual(result["maximum_zero_miss_offered_rps"], 700.0)
        self.assertEqual(result["first_observed_failure_rps"], 750.0)
        self.assertTrue(result["monotone_failure_frontier_observed"])
        self.assertEqual(result["loads"][-1]["misses"], 1)


if __name__ == "__main__":
    unittest.main()
