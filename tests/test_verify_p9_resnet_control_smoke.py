from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "analysis" / "verify_p9_resnet_control_smoke.py"
SPEC = importlib.util.spec_from_file_location("verify_resnet_control", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyResnetControlSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        (ROOT / "results/p9-resnet-layer7-control-mlp-100r-traced-v2-20260809T1420Z/result.json").is_file(),
        "requires preserved local ResNet hardware evidence",
    )
    def test_replays_preserved_hardware_smoke(self) -> None:
        result = MODULE.verify(
            ROOT / "results/p9-resnet-layer7-control-mlp-100r-traced-v2-20260809T1420Z"
        )
        self.assertEqual(result["requests"], 100)
        self.assertEqual(result["edge"]["bytes"], 14_720)
        self.assertFalse(result["token_only"])


if __name__ == "__main__":
    unittest.main()
