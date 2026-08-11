#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "positive_controls",
    ROOT / "analysis/summarize_p9_sota_positive_controls.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(
    (ROOT / "results/p9-boer-independent-payload-search-v1-20260809/search.json").is_file(),
    "requires preserved local positive-control evidence",
)
class SotaPositiveControlsTest(unittest.TestCase):
    def test_all_published_ports_have_intended_domain_positive_evidence(self) -> None:
        result = MODULE.summarize(
            ROOT / "results/p9-boer-independent-payload-search-v1-20260809/search.json",
            ROOT / "results/p9-parvagpu-independent-execution-v2-20260809/summary.json",
            ROOT / "results/p9-orion-profile-aware-signature-positive-20260809T110423Z/verification.json",
            ROOT / "results/p9-xsched-native-positive-20260809T101042Z/verification.json",
            ROOT / "results/p9-bless-tensorrt-fidelity-v5-20260809T1352/summary.json",
        )
        self.assertEqual(len(result["rows"]), 5)
        self.assertTrue(all(row["status"].startswith("passed-") for row in result["rows"]))
        self.assertIn("not cross-system rankings", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
