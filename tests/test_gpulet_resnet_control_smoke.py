#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_gpulet_resnet",
    ROOT / "baselines/gpulet/verify_resnet_control_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GpuletResnetControlSmokeTest(unittest.TestCase):
    def test_replays_hardware_smoke_when_present(self) -> None:
        roots = sorted(ROOT.glob("results/p9-gpulet-resnet-control-5x100-eval100-*/result.json"))
        if not roots:
            self.skipTest("gpulet ResNet-control hardware smoke is absent")
        result = MODULE.verify(roots[-1].parent)
        self.assertEqual(result["upstream_commit"], MODULE.UPSTREAM_COMMIT)
        self.assertEqual(result["requests"], 100)
        self.assertGreaterEqual(result["unique_payload_checksums"], 2)
        self.assertFalse(result["token_only"])


if __name__ == "__main__":
    unittest.main()
