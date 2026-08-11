import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize_p9_wall_smoke_repeats import summarize


def make(system, p99, misses, lock="lock"):
    return {
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-control", "deadline_mode": "wall",
        "latency_contract": "production-wall-arrival-to-completion",
        "production_wall_definition": (
            "arrival-to-consumer-completion-excludes-correctness-validation"
        ),
        "correctness_validation_placement": "post-completion",
        "checksum_mode": "inline", "deadline_us": 770.0,
        "deadline_lock": {"sha256": lock}, "background_period_ms": 4.0,
        "background_offered_rps": 250.0, "iterations": 100,
        "results": [{"system": system, "pipeline_requests": 100,
                      "deadline_misses": misses, "pipeline_p99_us": p99,
                      "background_goodput_rps": 249.0,
                      "producer_uuid": "small", "consumer_uuid": "big",
                      "producer_sms": 8, "consumer_sms": 12,
                      "deadline_mode": "wall",
                      "latency_contract": "production-wall-arrival-to-completion",
                      "checksum_mode": "inline", "correctness_validated": True}],
    }


class WallSmokeRepeatTest(unittest.TestCase):
    def test_aggregates_paired_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quiet, mps = [], []
            for index in range(2):
                q = root / f"q{index}.json"; m = root / f"m{index}.json"
                q.write_text(json.dumps(make("QUIET", 700 + index, 0)), encoding="utf-8")
                m.write_text(json.dumps(make("NVIDIA MPS", 900 + index, 50 + index)), encoding="utf-8")
                quiet.append(q); mps.append(m)
            result = summarize(quiet, mps, "NVIDIA MPS")
        self.assertEqual(result["systems"]["QUIET"]["deadline_misses"], 0)
        self.assertEqual(result["systems"]["NVIDIA MPS"]["deadline_misses"], 101)
        self.assertEqual(result["quiet_delta_vs_baseline"]["p99_mean_us"], -200.0)
        interval = result["paired_session_statistics"][
            "p99_delta_us_quiet_minus_baseline"
        ]["t95"]
        self.assertEqual(interval["n"], 2)
        self.assertLess(interval["upper"], 0.0)

    def test_rejects_mismatched_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q = root / "q.json"; m = root / "m.json"
            q.write_text(json.dumps(make("QUIET", 700, 0, "a")), encoding="utf-8")
            m.write_text(json.dumps(make("NVIDIA MPS", 900, 1, "b")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contracts"):
                summarize([q], [m], "NVIDIA MPS")


if __name__ == "__main__":
    unittest.main()
