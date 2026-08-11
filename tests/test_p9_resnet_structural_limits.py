#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "resnet_structural_limits",
    ROOT / "analysis/summarize_p9_resnet_structural_limits.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ResnetStructuralLimitsTest(unittest.TestCase):
    def test_replays_current_hardware_evidence(self) -> None:
        plan_roots = sorted(ROOT.glob(
            "results/p9-quiet-resnet-slack-plan-common-deadline-*/plan.json"
        ))
        base_roots = sorted(ROOT.glob(
            "results/p9-resnet-control-common-deadline-plan-enforced-base-*/summary.json"
        ))
        if not plan_roots or not base_roots:
            self.skipTest("plan-enforced ResNet evidence is absent")
        scope_roots = sorted(ROOT.glob(
            "results/p9-quiet-scope-common-frozen-1500r-500rps-*"
        ))
        if not scope_roots:
            self.skipTest("frozen stage-scope evidence is absent")
        result = MODULE.summarize({
            "boer_independent": ROOT / "results/p9-boer-independent-payload-search-v1-20260809/search.json",
            "boer_dependent": ROOT / "results/p9-boer-dependent-payload-search-v3-common-deadline-20260809/search.json",
            "parva_independent": ROOT / "results/p9-parvagpu-independent-execution-v2-20260809/summary.json",
            "parva_dependent": ROOT / "results/p9-parvagpu-dependent-profile-v3-common-deadline-20260809/allocation.json",
            "quiet_plan": plan_roots[-1],
            "plan_enforced_base": base_roots[-1],
            "producer_scope": scope_roots[-1] / "producer/summary.json",
            "pipeline_scope": scope_roots[-1] / "pipeline/summary.json",
            "transport": ROOT / "results/p9-transport-williams-4x500-20260809/aggregate.json",
        })
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertFalse(result["findings"]["BOER"]["dependent_feasible"])
        self.assertTrue(result["findings"]["QUIET"]["plan_enforced"])
        self.assertGreater(result["findings"]["QUIET"]["reserved_slack_us"], 0)


if __name__ == "__main__":
    unittest.main()
