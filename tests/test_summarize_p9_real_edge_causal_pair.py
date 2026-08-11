import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize_p9_real_edge_causal_pair import summarize


def pipeline(mode: str, *, edge: bool | None = None) -> dict:
    if edge is None:
        edge = mode == "dependent"
    return {
        "status": "ok", "dependency_mode": mode,
        "pipeline": "resnet10-layer7-cov-to-control-mlp",
        "transport": "registered-shared-sysmem-direct-binding",
        "producer_uuid": "small", "consumer_uuid": "big",
        "producer_sms": 8, "consumer_sms": 12,
        "producer_quota": 100, "consumer_quota": 100,
        "payload_bytes": 14720, "producer_output_tensor": "Layer7_cov",
        "consumer_input_tensor": "features", "consumer_output_tensor": "policy_output",
        "payload_shape": [1, 4, 23, 40], "warmup": 10, "iterations": 100,
        "checksum_mode": "inline", "correctness_validated": True,
        "correctness_scope": (
            "producer-output-consumer-input-equality"
            if mode == "dependent" else "producer-activation-replay-output-oracle"
        ),
        "checksum_failures": 0, "unique_payload_checksums": 4,
        "unique_policy_output_checksums": 4, "deadline_mode": "wall",
        "deadline_us": 770.605407,
        "dependency_edge": {"present": edge, "payload_bytes": 14720,
                            "transport": "registered-shared-sysmem-direct-binding"},
        "deadline_misses": 0,
        "stage_latency_us": {"producer_compute_p99": 600.0,
                             "consumer_compute_p99": 30.0,
                             "edge_transport_p99": 0.0 if not edge else 7.0},
        "end_to_end_us": {"p99": 610.0 if not edge else 690.0, "max": 620.0},
        "handoff_us": {"p99": 0.0 if not edge else 10.0},
    }


class RealEdgeCausalPairTest(unittest.TestCase):
    def test_accepts_real_edge_toggle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            independent = root / "independent.json"
            dependent = root / "dependent.json"
            independent.write_text(json.dumps(pipeline("independent")) + "\n")
            dependent.write_text(json.dumps(pipeline("dependent")) + "\n")
            result = summarize(independent, dependent)
        self.assertEqual(result["kind"], "p9-real-edge-causal-pair")
        self.assertEqual(result["delta_dependent_minus_independent"]["edge_transport_p99_us"], 7.0)

    def test_rejects_control_only_dependent_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            independent = root / "independent.json"
            dependent = root / "dependent.json"
            independent.write_text(json.dumps(pipeline("independent")) + "\n")
            dependent.write_text(json.dumps(pipeline("dependent", edge=False)) + "\n")
            with self.assertRaisesRegex(ValueError, "edge presence"):
                summarize(independent, dependent)


if __name__ == "__main__":
    unittest.main()
