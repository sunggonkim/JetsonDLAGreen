#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "transport_ablation", ROOT / "analysis/summarize_p9_transport_ablation.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TransportAblationTest(unittest.TestCase):
    def test_transport_only_accepts_learned_resnet_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for index, (label, transport) in enumerate(MODULE.EXPECTED.items()):
                raw = {
                    "kind": "p9-dependent-small-stress-smoke",
                    "workload": "resnet-detection-head",
                    "transport": transport,
                    "iterations": 20,
                    "results": [{
                        "transport": transport,
                        "payload_bytes": 1_884_160,
                        "correctness_validated": True,
                        "deadline_misses": 0,
                        "pipeline_requests": 20,
                        "wall_pipeline_p99_us": 3000 + index,
                        "stage_latency_us": {
                            "validation_excluded_end_to_end_p99": 2500 + index * 100,
                            "producer_compute_p99": 600,
                            "consumer_compute_p99": 70,
                            "transport_notification_p99": 10,
                            "producer_handoff_copy_p99": index * 10,
                            "consumer_handoff_copy_p99": index * 5,
                            "edge_transport_p99": 10 + index * 15,
                        },
                    }],
                }
                path = Path(directory) / f"{label}.json"
                path.write_text(json.dumps(raw) + "\n")
                paths[label] = path
            result = MODULE.summarize_transport_only(
                paths,
                workload="resnet-detection-head",
                pipeline="resnet10-backbone-to-detection-head",
                payload_bytes=1_884_160,
            )
        self.assertEqual(result["payload_bytes"], 1_884_160)
        self.assertFalse(result["formal"])
        self.assertFalse(result["ranking_allowed"])
        self.assertEqual(result["registered_delta_us"]["pinned_minus_registered"], 100)
        self.assertEqual(result["transports"][0]["workload"], "resnet-detection-head")

    def test_transport_only_rejects_mixed_request_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for index, (label, transport) in enumerate(MODULE.EXPECTED.items()):
                count = 20 if index == 0 else 10
                raw = {
                    "kind": "p9-dependent-small-stress-smoke",
                    "workload": "resnet-detection-head",
                    "transport": transport,
                    "results": [{
                        "transport": transport,
                        "payload_bytes": 1_884_160,
                        "correctness_validated": True,
                        "deadline_misses": 0,
                        "pipeline_requests": count,
                        "wall_pipeline_p99_us": 100.0,
                        "stage_latency_us": {
                            "validation_excluded_end_to_end_p99": 100.0,
                            "producer_compute_p99": 1.0,
                            "consumer_compute_p99": 1.0,
                            "transport_notification_p99": 1.0,
                            "producer_handoff_copy_p99": 1.0,
                            "consumer_handoff_copy_p99": 1.0,
                            "edge_transport_p99": 1.0,
                        },
                    }],
                }
                path = Path(directory) / f"{label}.json"
                path.write_text(json.dumps(raw) + "\n")
                paths[label] = path
            with self.assertRaisesRegex(ValueError, "same request count"):
                MODULE.summarize_transport_only(
                    paths,
                    workload="resnet-detection-head",
                    pipeline="resnet10-backbone-to-detection-head",
                    payload_bytes=1_884_160,
                )

    def test_registered_savings_use_validation_excluded_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for index, (label, transport) in enumerate(MODULE.EXPECTED.items()):
                raw = {
                    "status": "ok",
                    "pipeline": "whisper-last-hidden-state-to-projection-mlp",
                    "transport": transport,
                    "payload_bytes": 2304000,
                    "checksum_failures": 0,
                    "unique_payload_checksums": 4,
                    "unique_policy_output_checksums": 4,
                    "producer_uuid": "producer",
                    "consumer_uuid": "consumer",
                    "end_to_end_us": {"p99": 6000 + index},
                    "stage_latency_us": {
                        "validation_excluded_end_to_end_p99": 1500 + index * 50,
                        "producer_compute_p99": 1400,
                        "consumer_compute_p99": 50,
                        "producer_payload_verification_p99": 2000,
                        "consumer_payload_verification_p99": 2000,
                        "transport_notification_p99": 10,
                        "producer_handoff_copy_p99": index * 10,
                        "consumer_handoff_copy_p99": index * 5,
                        "edge_transport_p99": 10 + index * 15,
                    },
                }
                path = Path(directory) / f"{label}.json"
                path.write_text(json.dumps(raw))
                paths[label] = path
            same = json.loads(paths["registered"].read_text())
            same["producer_uuid"] = "shared"
            same["consumer_uuid"] = "shared"
            same_path = Path(directory) / "same.json"
            same_path.write_text(json.dumps(same))
            paths["same_instance"] = same_path
            result = MODULE.summarize(paths)
        self.assertEqual(result["registered_savings_us"]["vs_pinned"], 50)
        self.assertEqual(result["registered_savings_us"]["vs_pageable"], 100)
        self.assertEqual(
            result["placement_control"]["same_instance_mps_edge_p99_us"], 10
        )
        self.assertFalse(result["application_trace_bound"])

    def test_required_application_trace_rejects_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for index, (label, transport) in enumerate(MODULE.EXPECTED.items()):
                raw = {
                    "status": "ok", "pipeline": "whisper-last-hidden-state-to-projection-mlp",
                    "transport": transport, "payload_bytes": 2304000,
                    "checksum_failures": 0, "unique_payload_checksums": 4,
                    "unique_policy_output_checksums": 4,
                    "stage_latency_us": {
                        "validation_excluded_end_to_end_p99": 1.0,
                        "producer_compute_p99": 1.0, "consumer_compute_p99": 1.0,
                        "producer_payload_verification_p99": 1.0,
                        "consumer_payload_verification_p99": 1.0,
                        "transport_notification_p99": 1.0,
                        "producer_handoff_copy_p99": 1.0,
                        "consumer_handoff_copy_p99": 1.0,
                        "edge_transport_p99": 1.0,
                    },
                    "end_to_end_us": {"p99": 1.0},
                }
                path = Path(directory) / f"{label}.json"
                path.write_text(json.dumps(raw) + "\n")
                paths[label] = path
            same = json.loads(paths["registered"].read_text())
            same["producer_uuid"] = same["consumer_uuid"] = "shared"
            same_path = Path(directory) / "same.json"
            same_path.write_text(json.dumps(same) + "\n")
            paths["same_instance"] = same_path
            with self.assertRaisesRegex(ValueError, "output trace"):
                MODULE.summarize(paths, require_application_output_trace=True)


if __name__ == "__main__":
    unittest.main()
