import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bind_current_spec", ROOT / "analysis/bind_p9_current_dependent_spec.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BindCurrentDependentSpecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lock_path = ROOT / (
            "results/p9-whisper-pipeline-deadline-calibration-current-5x1000-20260810/"
            "deadline-lock.json"
        )
        if not self.lock_path.is_file():
            self.skipTest("current Whisper deadline lock is absent")
        self.lock = json.loads(self.lock_path.read_text(encoding="utf-8"))

    def test_binds_boer_command_and_contract(self) -> None:
        template = json.loads((
            ROOT / "baselines/boer/specs/p9-dependent-whisper-frozen-smoke.json"
        ).read_text(encoding="utf-8"))
        result = MODULE.bind(template, self.lock, self.lock_path, ROOT, ROOT / "results/test")
        self.assertEqual(result["contract"]["deadline_us"], self.lock["deadline_us"])
        command = result["evaluator_command"]
        self.assertEqual(command[command.index("--deadline-us") + 1], str(self.lock["deadline_us"]))

    def test_binds_parvagpu_service_slo(self) -> None:
        template = json.loads((
            ROOT / "baselines/parvagpu/specs/p9-dependent-whisper-frozen-smoke.json"
        ).read_text(encoding="utf-8"))
        result = MODULE.bind(template, self.lock, self.lock_path, ROOT, None)
        producer = next(item for item in result["services"] if item["model"] == "whisper-producer")
        self.assertEqual(producer["slo_ms"], self.lock["deadline_us"] / 1000.0)


if __name__ == "__main__":
    unittest.main()
