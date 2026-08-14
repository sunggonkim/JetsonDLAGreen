#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "published_sota",
    ROOT / "analysis/summarize_p9_published_sota_smoke.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(
    (ROOT / "results/p9-dependent-whisper-final-lock-1000r-250rps-20260809T1241/summary.json").is_file(),
    "requires preserved local published-system evidence",
)
class PublishedSotaSmokeTest(unittest.TestCase):
    def paths(self) -> tuple[Path, ...]:
        return (
            ROOT / "results/p9-dependent-whisper-final-lock-1000r-250rps-20260809T1241/summary.json",
            ROOT / "results/p9-boer-dependent-whisper-windowed-search-20260809T1238/search.json",
            ROOT / "results/p9-orion-dependent-whisper-faithful-1000r-20260809T1303/verification.json",
            ROOT / "results/p9-xsched-dependent-whisper-windowed-1000r-250rps-20260809T114154Z/verification.json",
            ROOT / "results/p9-bless-tensorrt-fidelity-v5-20260809T1352/summary.json",
            ROOT / "results/p9-parvagpu-dependent-whisper-profile-20260809/windowed-allocation.json",
        )

    def test_builds_public_comparison_without_fake_bless_numbers(self) -> None:
        result = MODULE.summarize(*self.paths())
        self.assertEqual(result["proposed_system"], "QUIET")
        names = [row["system"] for row in result["rows"]]
        self.assertEqual(names, [
            "NVIDIA MIG", "NVIDIA MPS", "BOER (Thor port)",
            "Orion (Thor port)", "XSched (Thor port)",
            "BLESS (Thor reimplementation)", "QUIET",
        ])
        bless = result["rows"][5]
        self.assertEqual(bless["evidence"], "functional-only")
        self.assertIsNone(bless["p99_us"])
        self.assertEqual(result["rows"][-1]["misses"], 0)
        self.assertFalse(result["rows"][3]["numeric_comparison_allowed"])
        self.assertEqual(
            result["rows"][3]["comparison_status"],
            "faithful-port-pending-differential-gate",
        )
        self.assertFalse(result["rows"][4]["numeric_comparison_allowed"])
        self.assertEqual(
            result["rows"][4]["comparison_status"],
            "historical-port-not-current-native-gate:formal-native-runtime-passed-slo-infeasible",
        )
        self.assertEqual(len(result["comparator_manifest"]["sha256"]), 64)

    def test_rejects_a_different_deadline_lock(self) -> None:
        paths = list(self.paths())
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "boer.json"
            raw = MODULE.load(paths[1])
            raw["contract"]["deadline_lock_sha256"] = "0" * 64
            import json
            altered.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            paths[1] = altered
            with self.assertRaisesRegex(ValueError, "BOER evidence"):
                MODULE.summarize(*paths)


if __name__ == "__main__":
    unittest.main()
