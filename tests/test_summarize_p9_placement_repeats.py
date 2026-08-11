import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summarize_p9_placement_repeats import summarize


def fixture(placement: str, p99: float, name: str) -> dict:
    return {
        "kind": "p9-dependent-small-stress-smoke", "workload": "resnet-control",
        "latency_contract": "production-wall-arrival-to-completion", "deadline_mode": "wall",
        "checksum_mode": "inline", "deadline_us": 773.730452, "iterations": 100,
        "background_period_ms": 4.0, "background_offered_rps": 250.0,
        "placement_variant": placement, "deadline_lock": {"sha256": "a" * 64},
        "results": [{"system": "QUIET", "correctness_validated": True,
                     "pipeline_requests": 100, "deadline_misses": 0,
                     "wall_pipeline_p99_us": p99, "background_goodput_rps": 248.0,
                     "unique_payload_checksums": 4, "unique_policy_output_checksums": 4}],
        "name": name,
    }


class PlacementRepeatSummaryTest(unittest.TestCase):
    def test_pairs_equal_repeat_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for placement, offset in (("fixed-1g-producer-2g-consumer", 0.0), ("fixed-2g-producer-1g-consumer", -100.0)):
                for index in range(2):
                    path = root / f"{placement}-{index}.json"
                    path.write_text(json.dumps(fixture(placement, 700.0 + offset + index, path.name)) + "\n")
                    paths.append(path)
            result = summarize(paths)
        self.assertEqual(result["placements"]["fixed-1g-producer-2g-consumer"]["repeat_count"], 2)
        self.assertEqual(len(result["paired_deltas"]), 2)
        interval = result["paired_session_statistics"]["p99_us_reverse_minus_forward"]["t95"]
        self.assertEqual(interval["n"], 2)
        self.assertLess(interval["upper"], 0.0)

    def test_rejects_unbalanced_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for placement in ("fixed-1g-producer-2g-consumer", "fixed-2g-producer-1g-consumer"):
                path = root / f"{placement}.json"
                path.write_text(json.dumps(fixture(placement, 700.0, path.name)) + "\n")
                paths.append(path)
            extra = root / "extra.json"
            extra.write_text(json.dumps(fixture("fixed-1g-producer-2g-consumer", 700.0, extra.name)) + "\n")
            paths.append(extra)
            with self.assertRaisesRegex(ValueError, "repeat counts differ"):
                summarize(paths)


if __name__ == "__main__":
    unittest.main()
