import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "baselines/orion/verify_resnet50_imagenette_smoke.py"
SPEC = importlib.util.spec_from_file_location("verify_orion_imagenette", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OrionResnet50ImagenetteSmokeTest(unittest.TestCase):
    def test_current_smoke_replays_common_contract(self) -> None:
        base = ROOT / "results/p9-orion-resnet50-imagenette-gate100-r01-20260811"
        required = [
            base / "result.json",
            base / "pipeline.csv",
            base / "scheduler-events.jsonl",
            base / "verification.json",
            base / "accuracy-gate.json",
            ROOT / "results/p9-orion-distilbert-operation-profile-20260809T110353Z/profile.json",
            ROOT / "results/p9-orion-resnet50-imagenette-profile-20260811/profile.json",
            ROOT / "results/p9-resnet50-imagenette-calibration-r02-20260811/deadline-lock.json",
            ROOT / "results/p9-resnet50-imagenette-gate100-20260811/common-workload.json",
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("current Orion ImageNette smoke is absent")
        value = MODULE.verify(
            required[0], required[1], required[2],
            required[5], required[6], required[7], required[8], ROOT,
        )
        self.assertTrue(value["functional_gate_passed"])
        self.assertTrue(value["numeric_smoke_valid"])
        self.assertFalse(value["formal_claim_allowed"])
        self.assertEqual(value["requests"], 90)
        self.assertEqual(value["misses"], 0)
        self.assertEqual(value["scheduler_events"]["records"], 4479)
        self.assertEqual(value["common_workload"]["dataset_manifest_sha256"],
                         "2951a9f93f6b7aa91bb8d9b9cc15b7ea0a6e7a8f06663ff6983c51eb32666b2b")
        gate = json.loads((base / "accuracy-gate.json").read_text(encoding="utf-8"))
        self.assertEqual(gate["reference_accuracy"], 0.8333333333333334)
        self.assertEqual(gate["candidate_accuracy"], 0.8333333333333334)
        self.assertEqual(gate["accuracy_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
