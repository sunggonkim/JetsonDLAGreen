import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summarize_p9_whisper_wall_causal_pair import summarize


def fixture(root: Path, mode: str, lock: str = "a" * 64) -> Path:
    scenario = root / mode
    scenario.mkdir()
    raw = {
        "status": "ok", "dependency_mode": mode,
        "dependency_edge": {"present": mode == "dependent"},
    }
    pipeline = scenario / "pipeline.json"
    pipeline.write_text(json.dumps(raw) + "\n")
    summary = root / f"{mode}-summary.json"
    summary.write_text(json.dumps({
        "kind": "p9-dependent-small-stress-smoke", "workload": "whisper-projection",
        "dependency_mode": mode, "latency_contract": "production-wall-arrival-to-completion",
        "deadline_mode": "wall", "checksum_mode": "inline", "iterations": 100,
        "background_period_ms": 4.0, "deadline_us": 7661.9,
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "deadline_lock": {"sha256": lock},
        "results": [{"system": "QUIET", "pipeline_requests": 100,
                     "deadline_misses": 0, "wall_pipeline_p99_us": 7000.0,
                     "background_goodput_rps": 249.0, "correctness_validated": True,
                     "checksum_failures": 0, "payload_bytes": 2304000,
                     "unique_payload_checksums": 4, "unique_policy_output_checksums": 4,
                     "request_trace": {"path": f"{mode}/pipeline.csv"},
                     "stage_latency_us": {
                         "producer_compute_p99": 100.0,
                         "transport_ready_p99": 500.0 if mode == "dependent" else 0.0,
                         "consumer_compute_p99": 50.0,
                         "edge_transport_p99": 7.0 if mode == "dependent" else 0.1,
                         "output_verification_p99": 10.0,
                     }}],
    }) + "\n")
    return summary


class WhisperWallCausalPairTest(unittest.TestCase):
    def test_pair_replays_edge_presence_and_common_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = summarize(fixture(root, "independent"), fixture(root, "dependent"))
        self.assertEqual(result["delta_dependent_minus_independent"]["wall_p99_us"], 0.0)
        self.assertEqual(result["delta_dependent_minus_independent"]["transport_ready_p99_us"], 500.0)
        self.assertIsNone(result["causal_interpretation"]["edge_transport_fraction_of_wall_delta"])

    def test_rejects_different_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            independent = fixture(root, "independent")
            dependent = fixture(root, "dependent", "b" * 64)
            with self.assertRaisesRegex(ValueError, "different deadline locks"):
                summarize(independent, dependent)

    def test_rejects_hidden_raw_workload_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            independent = fixture(root, "independent")
            dependent = fixture(root, "dependent")
            raw_path = root / "dependent" / "pipeline.json"
            raw = json.loads(raw_path.read_text())
            raw["payload_shape"] = [9, 9, 9]
            raw_path.write_text(json.dumps(raw) + "\n")
            with self.assertRaisesRegex(ValueError, "raw contract differs"):
                summarize(independent, dependent)


if __name__ == "__main__":
    unittest.main()
