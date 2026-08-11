#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_drain", ROOT / "runtime" / "profile_drain.py"
)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
SPEC.loader.exec_module(PROFILE)


class DrainProfileTest(unittest.TestCase):
    def test_envelope_rounds_up_worst_p999(self) -> None:
        self.assertEqual(PROFILE.envelope_ms([1.1, 1.4, 1.2]), 2.0)

    def test_invalid_samples_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PROFILE.envelope_ms([])
        with self.assertRaises(ValueError):
            PROFILE.envelope_ms([0.0])


if __name__ == "__main__":
    unittest.main()
