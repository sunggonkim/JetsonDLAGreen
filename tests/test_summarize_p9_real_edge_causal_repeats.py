import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize_p9_real_edge_causal_repeats import summarize_repeats


def result(mode: str, p99: float) -> dict:
    edge = mode == "dependent"
    return {
        "status": "ok", "dependency_mode": mode,
        "pipeline": "resnet10-layer7-cov-to-control-mlp",
        "transport": "registered-shared-sysmem-direct-binding",
        "producer_uuid": "small", "consumer_uuid": "big",
        "producer_sms": 8, "consumer_sms": 12, "producer_quota": 100,
        "consumer_quota": 100, "payload_bytes": 14720,
        "producer_output_tensor": "Layer7_cov", "consumer_input_tensor": "features",
        "consumer_output_tensor": "policy_output", "payload_shape": [1, 4, 23, 40],
        "warmup": 10, "iterations": 100, "checksum_mode": "inline",
        "correctness_validated": True, "checksum_failures": 0,
        "correctness_scope": (
            "producer-output-consumer-input-equality"
            if mode == "dependent" else "producer-activation-replay-output-oracle"
        ),
        "unique_payload_checksums": 4, "unique_policy_output_checksums": 4,
        "deadline_mode": "wall", "deadline_us": 770.605407,
        "dependency_edge": {"present": edge, "payload_bytes": 14720,
                            "transport": "registered-shared-sysmem-direct-binding"},
        "deadline_misses": 0,
        "stage_latency_us": {"producer_compute_p99": 600.0,
                             "consumer_compute_p99": 30.0,
                             "edge_transport_p99": 7.0 if edge else 0.0},
        "end_to_end_us": {"p99": p99, "max": p99 + 5},
        "handoff_us": {"p99": 10.0 if edge else 0.0},
    }


def learned_result(mode: str, p99: float, system: str = "QUIET") -> dict:
    row = {
        "system": system,
        "pipeline_requests": 20,
        "deadline_misses": 0,
        "wall_pipeline_p99_us": p99,
        "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
        "correctness_validation_placement": "post-completion",
        "checksum_mode": "inline",
        "transport": "registered-direct",
        "quiet_gate_scope": "producer",
        "background_period_ms": 4.0,
        "producer_quota_percent": 100,
        "background_quota_percent": 100,
        "consumer_engine": {"path": "head.engine", "sha256": "e" * 64},
        "payload_bytes": 1884160,
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "producer_uuid": "small",
        "consumer_uuid": "big",
        "producer_sms": 8,
        "consumer_sms": 12,
        "consumer_engine_mode": "external-trained-engine",
        "consumer_input_tensor": "Layer6_relu_Y",
        "transport": "registered-direct",
        "gate_scope": "producer",
        "producer_quota_percent": 100,
        "background_quota_percent": 100,
        "request_trace": {"path": "trace.csv", "sha256": "f" * 64},
        "stage_latency_us": {
            "validation_excluded_end_to_end_p50": 1200.0,
            "edge_transport_p99": 40.0 if mode == "dependent" else 0.0,
        },
        "background_goodput_rps": 100.0,
    }
    return {
        "schema_version": 1,
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-detection-head",
        "dependency_mode": mode,
        "consumer_engine_mode": "external-trained-engine",
        "consumer_input_tensor": "Layer6_relu_Y",
        "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
        "correctness_validation_placement": "post-completion",
        "checksum_mode": "inline",
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "payload_bytes": 1884160,
        "iterations": 20,
        "deadline_us": 100000.0,
        "deadline_mode": "wall",
        "results": [row],
    }


class CausalRepeatTest(unittest.TestCase):
    def test_aggregates_paired_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            independent, dependent = [], []
            for index, p99 in enumerate((600.0, 610.0), start=1):
                ip, dp = root / f"i{index}.json", root / f"d{index}.json"
                ip.write_text(json.dumps(result("independent", p99)) + "\n")
                dp.write_text(json.dumps(result("dependent", p99 + 80)) + "\n")
                independent.append(ip)
                dependent.append(dp)
            value = summarize_repeats(independent, dependent)
        self.assertEqual(value["repeat_count"], 2)
        self.assertEqual(value["delta_p99_us"]["mean"], 80.0)
        self.assertEqual(value["paired_session_ci95_us"]["n"], 2)
        self.assertEqual(value["statistical_unit"], "paired-session")

    def test_rejects_unpaired_counts(self):
        with self.assertRaisesRegex(ValueError, "counts must match"):
            summarize_repeats([Path("a")], [])

    def test_aggregates_learned_head_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            independent, dependent = [], []
            for index, p99 in enumerate((2400.0, 2500.0), start=1):
                ip, dp = root / f"li{index}.json", root / f"ld{index}.json"
                ip.write_text(json.dumps(learned_result("independent", p99)) + "\n")
                dp.write_text(json.dumps(learned_result("dependent", p99 + 100.0)) + "\n")
                independent.append(ip)
                dependent.append(dp)
            value = summarize_repeats(independent, dependent)
        self.assertEqual(value["workload"], "resnet-detection-head")
        self.assertEqual(value["delta_p99_us"]["mean"], 100.0)
        self.assertEqual(value["rows"][0]["delta_edge_transport_p99_us"], 40.0)

    def test_rejects_learned_head_payload_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ip, dp = root / "i.json", root / "d.json"
            ip.write_text(json.dumps(learned_result("independent", 2400.0)) + "\n")
            bad = learned_result("dependent", 2500.0)
            bad["results"][0]["payload_bytes"] = 14720
            dp.write_text(json.dumps(bad) + "\n")
            with self.assertRaisesRegex(ValueError, "payload differs"):
                summarize_repeats([ip], [dp])

    def test_rejects_learned_head_gate_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ip, dp = root / "i.json", root / "d.json"
            ip.write_text(json.dumps(learned_result("independent", 2400.0)) + "\n")
            bad = learned_result("dependent", 2500.0)
            bad["quiet_gate_scope"] = "pipeline"
            dp.write_text(json.dumps(bad) + "\n")
            with self.assertRaisesRegex(ValueError, "differs in quiet_gate_scope"):
                summarize_repeats([ip], [dp])

    def test_rejects_learned_head_transport_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ip, dp = root / "i.json", root / "d.json"
            ip.write_text(json.dumps(learned_result("independent", 2400.0)) + "\n")
            bad = learned_result("dependent", 2500.0)
            bad["transport"] = "pinned-bounce"
            bad["results"][0]["transport"] = "pinned-bounce"
            dp.write_text(json.dumps(bad) + "\n")
            with self.assertRaisesRegex(ValueError, "differs in transport"):
                summarize_repeats([ip], [dp])


if __name__ == "__main__":
    unittest.main()
