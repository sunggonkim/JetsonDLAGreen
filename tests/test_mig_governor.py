#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mig_governor", ROOT / "runtime" / "mig_governor.py"
)
assert SPEC is not None and SPEC.loader is not None
GOVERNOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOVERNOR)


class QuotaControlTest(unittest.TestCase):
    def test_compute_quota_tracks_fair_share(self) -> None:
        self.assertEqual(GOVERNOR.profiled_action("compute", 1, 2), (100, 1))
        self.assertEqual(GOVERNOR.profiled_action("compute", 2, 2), (50, 2))
        self.assertEqual(GOVERNOR.profiled_action("compute", 4, 2), (25, 4))
        self.assertEqual(GOVERNOR.profiled_action("compute", 6, 2), (25, 6))

    def test_memory_action_caps_admission(self) -> None:
        self.assertEqual(GOVERNOR.profiled_action("memory", 1, 2), (25, 1))
        self.assertEqual(GOVERNOR.profiled_action("memory", 6, 2), (25, 2))
        self.assertEqual(GOVERNOR.profiled_action("memory", 6, 1), (25, 1))

    def test_cpu_list_expansion(self) -> None:
        self.assertEqual(GOVERNOR.expand_cpu_list("0-2,5"), [0, 1, 2, 5])
        with self.assertRaises(ValueError):
            GOVERNOR.expand_cpu_list("2-0")
        with self.assertRaises(ValueError):
            GOVERNOR.expand_cpu_list("1,1")


if __name__ == "__main__":
    unittest.main()
