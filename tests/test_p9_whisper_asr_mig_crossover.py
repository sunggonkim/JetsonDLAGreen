import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p9_whisper_asr_mig_crossover.py"
SPEC = importlib.util.spec_from_file_location("whisper_crossover", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WhisperAsrMigCrossoverTest(unittest.TestCase):
    def test_modes_change_only_placement_and_protection(self) -> None:
        by_name = {mode.name: mode for mode in MODULE.MODES}
        self.assertEqual(
            set(by_name),
            {"nvidia-mig", "nvidia-mps-static-split", "quiet"},
        )
        self.assertEqual(by_name["nvidia-mig"].producer_device, "big")
        self.assertFalse(by_name["nvidia-mig"].gate_background)
        self.assertEqual(
            by_name["nvidia-mps-static-split"].producer_device, "small"
        )
        self.assertFalse(by_name["nvidia-mps-static-split"].gate_background)
        self.assertEqual(by_name["quiet"].producer_device, "small")
        self.assertTrue(by_name["quiet"].gate_background)

    def test_aggregate_preserves_deadline_miss_counts(self) -> None:
        rows = [
            {
                "rate_rps": 20.0, "mode": "quiet", "requests": 100,
                "deadline_misses": miss, "p50_us": 1.0, "p99_us": 2.0,
                "queue_p99_us": 0.5, "request_goodput_rps": 20.0,
                "background_goodput_rps": 100.0, "producer_mean_us": 3.0,
                "consumer_mean_us": 4.0,
            }
            for miss in (2, 3)
        ]
        result = MODULE.aggregate(rows)
        self.assertEqual(result[0]["sessions"], 2)
        self.assertEqual(result[0]["requests"], 200)
        self.assertEqual(result[0]["deadline_misses"], 5)


if __name__ == "__main__":
    unittest.main()
