import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p9_static_full_gate_smoke.sh"


class StaticFullGateSmokeScriptTest(unittest.TestCase):
    def test_help_and_claim_boundary(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Static full gating", text)
        self.assertIn("numeric_comparison_allowed", text)
        self.assertIn('"ranking_allowed": False', text)
        self.assertNotIn('baseline_system": "QUIET"', text)

    def test_requires_lock_and_runs_only_static_scenario(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--deadline-lock", text)
        self.assertIn("--scenario static-full-gate", text)
        self.assertIn('execution_order") != ["Static full gating"]', text)
        self.assertIn("--checksum-mode inline", text)


if __name__ == "__main__":
    unittest.main()
