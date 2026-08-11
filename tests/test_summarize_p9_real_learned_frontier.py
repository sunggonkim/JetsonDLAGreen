import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize_p9_real_learned_frontier import summarize


def fixture(system: str, offered: float, misses: int = 0) -> dict:
    return {
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-detection-head",
        "dependency_mode": "dependent",
        "latency_contract": "production-wall-arrival-to-completion",
        "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
        "correctness_validation_placement": "post-completion",
        "checksum_mode": "inline",
        "consumer_engine_mode": "external-trained-engine",
        "consumer_input_tensor": "Layer6_relu_Y",
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "deadline_us": 3500.0,
        "deadline_mode": "wall",
        "background_offered_rps": offered,
        "consumer_engine": {"sha256": "a" * 64},
        "results": [{
            "system": system, "pipeline_requests": 100,
            "deadline_misses": misses, "wall_pipeline_p99_us": 2800.0,
            "background_goodput_rps": offered - 1.0,
            "correctness_validated": True, "checksum_failures": 0,
            "payload_bytes": 1884160, "producer_uuid": "p",
            "consumer_uuid": "c", "producer_sms": 8, "consumer_sms": 12,
        }],
    }


class LearnedFrontierTest(unittest.TestCase):
    def test_separates_descriptive_and_cp95(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS"):
                for offered in (125.0, 250.0, 375.0):
                    path = root / f"{system}-{int(offered)}.json"
                    path.write_text(json.dumps(fixture(system, offered)) + "\n")
                    paths.append(path)
            result = summarize(paths)
        self.assertEqual(result["systems"]["QUIET"]["descriptive_max_offered_rps"], 375.0)
        self.assertIsNone(result["systems"]["QUIET"]["formal_cp95_max_offered_rps"])

    def test_rejects_different_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, (system, offered) in enumerate(
                ([("QUIET", 125.0), ("QUIET", 250.0), ("QUIET", 375.0),
                  ("NVIDIA MPS", 125.0), ("NVIDIA MPS", 250.0), ("NVIDIA MPS", 375.0)])
            ):
                value = fixture(system, offered)
                if index == 5:
                    value["consumer_engine"]["sha256"] = "b" * 64
                path = root / f"r{index}.json"
                path.write_text(json.dumps(value) + "\n")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "different consumer engine"):
                summarize(paths)


if __name__ == "__main__":
    unittest.main()
