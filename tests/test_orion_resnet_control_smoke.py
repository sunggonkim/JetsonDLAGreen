from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "baselines/orion/verify_resnet_control_smoke.py"
SPEC = importlib.util.spec_from_file_location("verify_orion_resnet", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OrionResnetControlSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        (ROOT / "results/p9-orion-resnet-control-100r-250rps-dev-v2-20260809T1433Z/result.json").is_file()
        and (ROOT / "build-r39/jdg-orion-mig-trt-pipeline").is_file(),
        "requires preserved local Orion hardware evidence",
    )
    def test_replays_hardware_smoke(self) -> None:
        directory = ROOT / "results/p9-orion-resnet-control-100r-250rps-dev-v2-20260809T1433Z"
        result = MODULE.verify(
            directory / "result.json", directory / "pipeline.csv",
            directory / "checksums.csv", directory / "scheduler-events.jsonl",
            ROOT / "results/p9-orion-distilbert-operation-profile-20260809T110353Z/profile.json",
            ROOT / "results/p9-orion-resnet10-operation-profile-20260809T1430Z/profile.json",
            ROOT / "results/p9-orion-distilbert-operation-profile-20260809T110353Z/scheduler-profile.tsv",
            ROOT / "results/p9-orion-resnet10-operation-profile-20260809T1430Z/scheduler-profile.tsv",
            ROOT / "build-r39/jdg-orion-mig-trt-pipeline",
        )
        self.assertEqual(result["misses"], 79)
        self.assertFalse(result["token_only"])
        self.assertGreater(result["scheduler"]["event_records"], 0)


if __name__ == "__main__":
    unittest.main()
