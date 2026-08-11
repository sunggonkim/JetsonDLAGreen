import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "current_whisper_evidence", ROOT / "analysis/summarize_p9_current_whisper_evidence.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CurrentWhisperEvidenceTest(unittest.TestCase):
    def test_joins_current_hardware_evidence(self) -> None:
        paths = {
            "numeric": ROOT / "results/p9-common-sota-whisper-current-smoke-aggregate-6x100-20260810/summary.json",
            "boer_independent": ROOT / "results/p9-boer-independent-payload-search-v1-20260809/search.json",
            "boer_dependent": ROOT / "results/p9-boer-dependent-whisper-current-lock-search-20260810/search.json",
            "parva_independent": ROOT / "results/p9-parvagpu-independent-execution-v2-20260809/summary.json",
            "parva_dependent": ROOT / "results/p9-parvagpu-dependent-whisper-current-lock-20260810/allocation.json",
            "transport": ROOT / "results/p9-transport-williams-4x500-20260809/aggregate.json",
        }
        if not all(path.is_file() for path in paths.values()):
            self.skipTest("current Whisper evidence is absent")
        result = MODULE.summarize(paths)
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertFalse(result["published_system_boundaries"]["BOER"]["dependent_feasible"])
        self.assertAlmostEqual(
            result["published_system_boundaries"]["BOER"]["independent_selected"]["worst_p99_ms"],
            1.4811732,
        )
        self.assertEqual(
            len(result["published_system_boundaries"]["ParvaGPU"]["independent_services"]), 2
        )
        self.assertTrue(result["published_system_boundaries"]["QUIET"]["observed_zero_miss"])


if __name__ == "__main__":
    unittest.main()
