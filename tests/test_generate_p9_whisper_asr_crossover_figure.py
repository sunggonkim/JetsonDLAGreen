import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis/generate_p9_whisper_asr_crossover_figure.py"
SPEC = importlib.util.spec_from_file_location("whisper_crossover_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _raw() -> dict:
    rows = []
    for session in (1, 2, 3):
        for mode, misses, gated in (
            ("nvidia-mig", 50, 0),
            ("nvidia-mps-static-split", 10, 0),
            ("quiet", 0, 1),
        ):
            rows.append(
                {
                    "session": session, "rate_rps": 19.0, "mode": mode,
                    "deadline_misses": misses, "requests": 100,
                    "p50_us": 100000.0, "p99_us": 200000.0 + session,
                    "queue_p99_us": 1000.0, "request_goodput_rps": 18.0,
                    "background_goodput_rps": 800.0 + session,
                    "output_sha256": "a" * 64, "gated_processes": gated,
                    "producer_mean_us": 8000.0, "consumer_mean_us": 50000.0,
                    "gate_hold_p99_us": 10000.0 if gated else None,
                }
            )
    return {
        "kind": "p9-whisper-asr-mig-crossover",
        "evidence_class": "exploratory-nonthermal-directional",
        "thermal_campaign": False,
        "comparator_output_contract": "byte-identical",
        "pipeline_slots": 3,
        "deadline_us": 250000.0,
        "rows": rows,
    }


class GenerateWhisperCrossoverFigureTest(unittest.TestCase):
    def test_compaction_locks_order_and_counts(self) -> None:
        summary = MODULE.compact(_raw(), pathlib.Path(__file__), {"binary": "b" * 64})
        self.assertEqual(summary["system_order"], ["QUIET", "NVIDIA MIG", "NVIDIA MPS"])
        self.assertEqual(summary["systems"]["NVIDIA MIG"]["misses"], 150)
        self.assertEqual(summary["systems"]["QUIET"]["misses"], 0)
        self.assertEqual(summary["output_contract"], "byte-identical across all nine runs")

    def test_rejects_nonidentical_outputs(self) -> None:
        raw = _raw()
        raw["rows"][0]["output_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "outputs differ"):
            MODULE.compact(raw, pathlib.Path(__file__), {})


if __name__ == "__main__":
    unittest.main()
