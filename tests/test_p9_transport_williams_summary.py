import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "transport_williams_summary", ROOT / "analysis/summarize_p9_transport_williams.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TransportWilliamsSummaryTest(unittest.TestCase):
    def test_percentile_matches_type7(self):
        self.assertEqual(MODULE.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(MODULE.percentile([1, 2, 3, 4], 0.99), 3.97)

    def test_paired_interval(self):
        result = MODULE.mean_t95([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(result, {"mean": 1.0, "lower": 1.0, "upper": 1.0})


if __name__ == "__main__":
    unittest.main()
