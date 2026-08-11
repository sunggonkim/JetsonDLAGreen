#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OrionDependentRunnerTest(unittest.TestCase):
    def test_runner_binds_upstream_duration_rule_and_raw_verifier(self) -> None:
        text = (ROOT / "scripts/run_p9_orion_dependent_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(".pooled_p99_us", text)
        self.assertIn("--orion-max-be-duration-us", text)
        self.assertIn("--orion-profile-aware true", text)
        self.assertIn("scheduler-profile.tsv", text)
        self.assertIn("verify_dependent_smoke.py", text)
        self.assertNotIn("run_thor.py", text)

    @unittest.skipUnless(
        (ROOT / "results/p9-orion-distilbert-operation-profile-20260809T110353Z/profile.json").is_file(),
        "requires preserved local Orion profile evidence",
    )
    def test_existing_profiles_are_real_driver_operation_profiles(self) -> None:
        for relative, model in (
            ("results/p9-orion-distilbert-operation-profile-20260809T110353Z/profile.json", "distilbert-sst2"),
            ("results/p9-orion-whisper-operation-profile-20260809T110353Z/profile.json", "whisper-tiny-encoder"),
        ):
            profile = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            self.assertEqual(profile["model"], model)
            self.assertGreater(len(profile["operations"]), 30)
            self.assertTrue(all(row["api"] == "cuLaunchKernelEx" for row in profile["operations"]))


if __name__ == "__main__":
    unittest.main()
