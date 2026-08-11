#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orion_probe", ROOT / "baselines" / "orion" / "probe.py"
)
assert SPEC is not None and SPEC.loader is not None
ORION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORION
SPEC.loader.exec_module(ORION)


class OrionProbeTest(unittest.TestCase):
    def test_signal_and_shell_segmentation_status_require_integration(self) -> None:
        for returncode in (-11, 139):
            status, reason = ORION.classify_probe(returncode)
            self.assertEqual(status, "requires-orion-managed-client-integration")
            self.assertIn("thread queues", reason)

    def test_survival_is_not_misreported_as_measured_orion(self) -> None:
        status, reason = ORION.classify_probe(0)
        self.assertEqual(status, "interceptor-process-survived")
        self.assertIn("does not prove", reason)


if __name__ == "__main__":
    unittest.main()
