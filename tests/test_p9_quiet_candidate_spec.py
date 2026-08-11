#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quiet_candidate_spec", ROOT / "analysis" / "build_p9_quiet_candidate_spec.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QuietCandidateSpecTest(unittest.TestCase):
    def test_builds_candidate_from_bound_payload_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "run"
            (result / "quiet").mkdir(parents=True)
            pipeline = {
                "iterations": 100,
                "end_to_end_us": {"p99": 700.0},
            }
            (result / "quiet" / "pipeline.json").write_text(json.dumps(pipeline))
            summary = {
                "kind": "p9-dependent-small-stress-smoke",
                "producer_quota_percent": 100,
                "background_quota_percent": 25,
                "results": [{
                    "system": "QUIET",
                    "producer_quota_percent": 100,
                    "background_quota_percent": 25,
                    "pipeline_requests": 100,
                    "pipeline_p99_us": 700.0,
                    "background_goodput_rps": 200.0,
                    "gate_p99_us": 900.0,
                }],
            }
            summary_path = result / "summary.json"
            summary_path.write_text(json.dumps(summary))
            output = root / "spec.json"
            built = MODULE.build_spec([summary_path], output, 760, 1000, 5)
        self.assertEqual(built["system"], "QUIET")
        self.assertEqual(built["candidates"][0]["candidate_id"], "q100-q25")
        self.assertEqual(built["candidates"][0]["background_quota_percent"], 25)
        self.assertEqual(built["candidates"][0]["reservation_margin_us"], 5)
        self.assertEqual(
            built["candidate_search"]["claim_status"],
            "single-candidate-characterization",
        )

    def test_accepts_validation_excluded_whisper_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "run"
            (result / "quiet").mkdir(parents=True)
            (result / "quiet" / "pipeline.json").write_text(json.dumps({
                "iterations": 100,
                "end_to_end_us": {"p99": 7000.0},
                "stage_latency_us": {"validation_excluded_end_to_end_p99": 1550.0},
            }))
            summary = {
                "kind": "p9-dependent-small-stress-smoke",
                "producer_quota_percent": 100,
                "background_quota_percent": 100,
                "results": [{
                    "system": "QUIET", "producer_quota_percent": 100,
                    "background_quota_percent": 100, "pipeline_requests": 100,
                    "pipeline_p99_us": 1550.0, "deadline_mode": "validation-excluded",
                    "background_goodput_rps": 500.0, "gate_p99_us": 900.0,
                }],
            }
            summary_path = result / "summary.json"
            summary_path.write_text(json.dumps(summary))
            built = MODULE.build_spec([summary_path], root / "spec.json", 1700, 1000, 0)
        self.assertEqual(built["candidates"][0]["candidate_id"], "q100-q100")

    def test_distinguishes_quota_search_from_placement_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = []
            for index, (producer, background) in enumerate(((100, 25), (75, 25))):
                result = root / f"run{index}"
                (result / "quiet").mkdir(parents=True)
                (result / "quiet" / "pipeline.json").write_text(json.dumps({
                    "iterations": 100,
                    "end_to_end_us": {"p99": 700.0},
                }))
                summary = {
                    "kind": "p9-dependent-small-stress-smoke",
                    "producer_quota_percent": producer,
                    "background_quota_percent": background,
                    "results": [{
                        "system": "QUIET", "producer_quota_percent": producer,
                        "background_quota_percent": background,
                        "pipeline_requests": 100, "pipeline_p99_us": 700.0,
                        "background_goodput_rps": 200.0, "gate_p99_us": 900.0,
                    }],
                }
                path = result / "summary.json"
                path.write_text(json.dumps(summary))
                summaries.append(path)
            built = MODULE.build_spec(summaries, root / "spec.json", 760, 1000, 0)
        self.assertTrue(built["candidate_search"]["multi_candidate_evaluated"])
        self.assertFalse(built["candidate_search"]["placement_search_evaluated"])
        self.assertEqual(
            built["candidate_search"]["claim_status"],
            "multi-candidate-quota-search-only",
        )

    def test_binds_resnet_deadline_lock_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "run"
            (result / "quiet").mkdir(parents=True)
            (result / "quiet" / "pipeline.json").write_text(json.dumps({
                "iterations": 100,
                "payload_bytes": 14_720,
                "end_to_end_us": {"p99": 700.0},
            }))
            summary_path = result / "summary.json"
            summary_path.write_text(json.dumps({
                "kind": "p9-dependent-small-stress-smoke",
                "workload": "resnet-control",
                "producer_quota_percent": 100,
                "background_quota_percent": 100,
                "results": [{
                    "system": "QUIET",
                    "producer_quota_percent": 100,
                    "background_quota_percent": 100,
                    "pipeline_requests": 100,
                    "pipeline_p99_us": 700.0,
                    "deadline_mode": "wall",
                    "background_goodput_rps": 250.0,
                    "gate_p99_us": 800.0,
                }],
            }))
            lock_path = root / "deadline-lock.json"
            lock_path.write_text(json.dumps({
                "kind": "p9-dependent-pipeline-deadline-lock",
                "deadline_us": 770.0,
                "contract": {
                    "workload": "resnet-control",
                    "payload_bytes": 14_720,
                    "deadline_mode": "wall",
                },
            }))
            built = MODULE.build_spec(
                [summary_path], root / "spec.json", 1, 1000, 0, lock_path
            )
        self.assertEqual(built["deadline_us"], 770.0)
        self.assertEqual(len(built["deadline_lock"]["sha256"]), 64)

    def test_reverse_candidate_preserves_placement_in_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "run"
            (result / "quiet").mkdir(parents=True)
            (result / "quiet" / "pipeline.json").write_text(json.dumps({
                "iterations": 100,
                "end_to_end_us": {"p99": 700.0},
            }))
            summary_path = result / "summary.json"
            summary_path.write_text(json.dumps({
                "kind": "p9-dependent-small-stress-smoke",
                "producer_quota_percent": 100,
                "background_quota_percent": 100,
                "placement_variant": "fixed-2g-producer-1g-consumer",
                "results": [{
                    "system": "QUIET", "producer_quota_percent": 100,
                    "background_quota_percent": 100,
                    "pipeline_requests": 100, "pipeline_p99_us": 700.0,
                    "background_goodput_rps": 200.0, "gate_p99_us": 900.0,
                    "placement_variant": "fixed-2g-producer-1g-consumer",
                }],
            }))
            built = MODULE.build_spec([summary_path], root / "spec.json", 760, 1000, 0)
        self.assertEqual(
            built["candidates"][0]["placement"],
            {"producer": "2g-q100", "consumer": "1g-q100"},
        )
        self.assertEqual(built["candidate_search"]["placement_variant_count"], 1)

    def test_common_lock_binds_two_placement_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "common-lock.json"
            lock = {
                "kind": "p9-common-placement-deadline-lock",
                "deadline_us": 800.0,
                "contract": {
                    "workload": "resnet-control",
                    "payload_bytes": 14_720,
                    "deadline_mode": "wall",
                    "allowed_placements": [
                        "fixed-1g-producer-2g-consumer",
                        "fixed-2g-producer-1g-consumer",
                    ],
                },
                "allowed_placements": [
                    "fixed-1g-producer-2g-consumer",
                    "fixed-2g-producer-1g-consumer",
                ],
            }
            lock_path.write_text(json.dumps(lock) + "\n")
            summaries = []
            for index, placement in enumerate((
                "fixed-1g-producer-2g-consumer",
                "fixed-2g-producer-1g-consumer",
            )):
                run = root / f"run{index}"
                (run / "quiet").mkdir(parents=True)
                (run / "quiet" / "pipeline.json").write_text(json.dumps({
                    "status": "ok", "pipeline": "resnet10-layer7-cov-to-control-mlp",
                    "iterations": 100, "payload_bytes": 14_720,
                    "transport": "registered-shared-sysmem-direct-binding",
                    "checksum_failures": 0, "unique_payload_checksums": 4,
                    "unique_policy_output_checksums": 4, "deadline_mode": "wall",
                    "end_to_end_us": {"p99": 700.0 + index},
                    "stage_latency_us": {
                        "producer_compute_p99": 400.0,
                        "edge_transport_p99": 10.0,
                        "consumer_compute_p99": 50.0,
                        "output_verification_p99": 20.0,
                    },
                }))
                summary = {
                    "kind": "p9-dependent-small-stress-smoke",
                    "workload": "resnet-control",
                    "producer_quota_percent": 100,
                    "background_quota_percent": 100,
                    "placement_variant": placement,
                    "deadline_lock": {"sha256": hashlib.sha256(
                        lock_path.read_bytes()).hexdigest()},
                    "results": [{
                        "system": "QUIET", "producer_quota_percent": 100,
                        "background_quota_percent": 100,
                        "pipeline_requests": 100, "pipeline_p99_us": 700.0 + index,
                        "deadline_mode": "wall", "background_goodput_rps": 200.0 + index,
                        "gate_p99_us": 700.0, "placement_variant": placement,
                    }],
                }
                path = run / "summary.json"
                path.write_text(json.dumps(summary) + "\n")
                summaries.append(path)
            built = MODULE.build_spec(
                summaries, root / "spec.json", 1, 1000, 0, lock_path
            )
        self.assertEqual(built["candidate_search"]["placement_variant_count"], 2)
        self.assertTrue(built["candidate_search"]["placement_search_evaluated"])
        self.assertTrue(all("@fixed-" in item["candidate_id"] for item in built["candidates"]))

    def test_formal_requires_quota_candidates_for_each_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = []
            for index, (placement, producer, background) in enumerate((
                ("fixed-1g-producer-2g-consumer", 100, 100),
                ("fixed-2g-producer-1g-consumer", 100, 100),
            )):
                run = root / f"run{index}"
                (run / "quiet").mkdir(parents=True)
                (run / "quiet" / "pipeline.json").write_text(json.dumps({
                    "iterations": 100, "payload_bytes": 14720,
                    "checksum_failures": 0, "end_to_end_us": {"p99": 700.0},
                }))
                path = run / "summary.json"
                path.write_text(json.dumps({
                    "kind": "p9-dependent-small-stress-smoke",
                    "workload": "resnet-control", "checksum_mode": "inline",
                    "deadline_mode": "wall", "producer_quota_percent": producer,
                    "background_quota_percent": background,
                    "placement_variant": placement,
                    "results": [{
                        "system": "QUIET", "producer_quota_percent": producer,
                        "background_quota_percent": background,
                        "pipeline_requests": 100, "deadline_misses": 0,
                        "pipeline_p99_us": 700.0, "background_goodput_rps": 200.0,
                        "gate_p99_us": 700.0, "correctness_validated": True,
                        "checksum_failures": 0, "unique_payload_checksums": 4,
                        "unique_policy_output_checksums": 4,
                        "placement_variant": placement,
                    }],
                }))
                summaries.append(path)
            with self.assertRaisesRegex(ValueError, "formal candidate search"):
                MODULE.build_spec(summaries, root / "spec.json", 760, 1000, 0, formal=True)


if __name__ == "__main__":
    unittest.main()
