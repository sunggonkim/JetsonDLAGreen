import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "numeric_sota_summary", ROOT / "analysis/summarize_p9_numeric_sota_repetitions.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NumericSotaRepetitionsTest(unittest.TestCase):
    def test_requires_complete_williams_design(self):
        with self.assertRaisesRegex(ValueError, "exactly six"):
            MODULE.summarize([])

    def test_action_contract_is_explicit(self):
        self.assertEqual(MODULE.EXPECTED_ACTIONS["QUIET"], (100, 100, "producer"))
        self.assertEqual(
            MODULE.EXPECTED_ACTIONS["Full-DAG quiescence"],
            (100, 100, "pipeline"),
        )
        self.assertEqual(
            MODULE.EXPECTED_ACTIONS["Partition-only planning"],
            (90, 10, "producer"),
        )

    def test_paired_ratio_interval(self):
        interval = MODULE.mean_t95([3.2, 3.3, 3.25, 3.21, 3.28, 3.24])
        self.assertGreater(interval["lower"], 3.0)
        self.assertLess(interval["upper"], 3.5)


if __name__ == "__main__":
    unittest.main()
