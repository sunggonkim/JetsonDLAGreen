from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "analysis" / "build_bless_common_schedule.py"
SPEC = importlib.util.spec_from_file_location("build_bless_common_schedule", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildBlessCommonScheduleTest(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        for model, count in (("resnet", 3), ("distilbert", 4)):
            for sms in (2, 4, 6, 8):
                directory = root / f"{model}-{sms}"
                directory.mkdir(parents=True)
                start = sms * 1_000_000
                records = []
                for operation in range(count):
                    duration = (10 - sms) * 1000 + operation
                    records.append({
                        "operation": operation,
                        "selected_sms": sms,
                        "result": 0,
                        "start_monotonic_ns": start,
                        "end_monotonic_ns": start + duration,
                    })
                    start += duration + 1
                (directory / "squad.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
        return root

    def test_builds_profiles_and_a_scheduler_decision(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = MODULE.build(self.fixture(Path(raw)))
            self.assertEqual(result["models"]["resnet"]["logical_launches"], 3)
            self.assertEqual(result["models"]["distilbert"]["logical_launches"], 4)
            self.assertEqual(len(result["squad"]), 3)
            self.assertIn(result["configuration"]["estimator"], {
                "interference-free", "workload-equivalence"
            })
            self.assertFalse(result["numeric_comparison_allowed"])

    def test_rejects_profile_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.fixture(Path(raw))
            path = root / "resnet-4" / "squad.jsonl"
            path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
            with self.assertRaisesRegex(ValueError, "launch counts differ"):
                MODULE.build(root)


if __name__ == "__main__":
    unittest.main()
