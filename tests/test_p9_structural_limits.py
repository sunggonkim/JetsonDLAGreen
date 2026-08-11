import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p9_structural_limits", ROOT / "analysis/summarize_p9_structural_limits.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StructuralLimitsTest(unittest.TestCase):
    def test_constants_bind_large_edge_and_frozen_deadline(self):
        self.assertEqual(MODULE.PAYLOAD_BYTES, 2_304_000)
        self.assertEqual(len(MODULE.DEADLINE_SHA), 64)

    def test_finite_rejects_invalid_numbers(self):
        for value in (True, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MODULE.finite(value, "test")


if __name__ == "__main__":
    unittest.main()
