#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quiet_stage_dag", ROOT / "runtime" / "quiet_stage_dag.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
select_plan = MODULE.select_plan
actuate_selected_plan = MODULE.actuate_selected_plan


def profile(
    p99: float, producer: float, transport: float, consumer: float,
    pipeline: str = "resnet10-layer7-cov-to-control-mlp",
    payload_bytes: int = 14720,
) -> dict:
    return {
        "status": "ok",
        "pipeline": pipeline,
        "transport": "registered-shared-sysmem-direct-binding",
        "payload_bytes": payload_bytes,
        "iterations": 100,
        "checksum_failures": 0,
        "unique_payload_checksums": 4,
        "unique_policy_output_checksums": 4,
        "end_to_end_us": {"p99": p99},
        "stage_latency_us": {
            "producer_compute_p99": producer,
            "transport_ready_p99": transport,
            "consumer_compute_p99": consumer,
            "output_verification_p99": 10.0,
        },
    }


class QuietStageDagWallSelectionTest(unittest.TestCase):
    def test_wall_profile_does_not_use_validation_excluded_p99(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = profile(700.0, 580.0, 40.0, 30.0)
            value["deadline_mode"] = "wall"
            value["stage_latency_us"]["validation_excluded_end_to_end_p99"] = 100.0
            (root / "profile.json").write_text(json.dumps(value))
            result = select_plan({
                "schema_version": 1, "system": "QUIET", "deadline_us": 760,
                "critical_lookahead_us": 0,
                "candidates": [{
                    "candidate_id": "wall", "profile_path": "profile.json",
                    "placement": {"producer": "1g", "consumer": "2g"},
                    "background_goodput_rps": 1, "pre_release_guard_p99_us": 0,
                    "reservation_margin_us": 0,
                }],
            }, root)
        self.assertEqual(result["selected_plan"]["observed_end_to_end_p99_us"], 700.0)

    def test_classification_wall_profile_uses_measured_joint_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = profile(
                700.0, 580.0, 40.0, 30.0,
                pipeline="resnet50-backbone-to-classification-head",
                payload_bytes=802816,
            )
            value["deadline_mode"] = "wall"
            (root / "profile.json").write_text(json.dumps(value))
            result = select_plan({
                "schema_version": 1, "system": "QUIET", "deadline_us": 760,
                "critical_lookahead_us": 0,
                "candidates": [{
                    "candidate_id": "classification-wall",
                    "profile_path": "profile.json",
                    "placement": {"producer": "1g", "consumer": "2g"},
                    "background_goodput_rps": 1,
                    "pre_release_guard_p99_us": 0,
                    "reservation_margin_us": 0,
                }],
            }, root)
        self.assertEqual(
            result["selected_plan"]["tail_bound_method"],
            "measured-production-wall-p99",
        )

    def test_joint_tail_bound_replaces_component_p99_sum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = profile(700.0, 580.0, 40.0, 30.0)
            value["joint_tail_p99_us"] = 610.0
            value["stage_latency_us"]["output_verification_p99"] = 900.0
            (root / "profile.json").write_text(json.dumps(value))
            result = select_plan({
                "schema_version": 1, "system": "QUIET", "deadline_us": 760,
                "critical_lookahead_us": 0,
                "candidates": [{
                    "candidate_id": "joint-tail",
                    "profile_path": "profile.json",
                    "placement": {"producer": "1g", "consumer": "2g"},
                    "background_goodput_rps": 1,
                    "pre_release_guard_p99_us": 0,
                    "reservation_margin_us": 10,
                }],
            }, root)
        selected = result["selected_plan"]
        self.assertEqual(selected["tail_bound_method"], "joint-request-p99")
        self.assertTrue(selected["tail_bound_promotable"])
        self.assertEqual(selected["response_reservation_us"], 620.0)

    def test_explicit_risk_budget_is_a_promotable_tail_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = profile(700.0, 580.0, 40.0, 30.0)
            value["risk_budget_us"] = {
                "producer": 300.0,
                "edge": 80.0,
                "consumer": 200.0,
            }
            (root / "profile.json").write_text(json.dumps(value))
            result = select_plan({
                "schema_version": 1, "system": "QUIET", "deadline_us": 760,
                "critical_lookahead_us": 0,
                "candidates": [{
                    "candidate_id": "risk-budget",
                    "profile_path": "profile.json",
                    "placement": {"producer": "1g", "consumer": "2g"},
                    "background_goodput_rps": 1,
                    "pre_release_guard_p99_us": 0,
                    "reservation_margin_us": 10,
                }],
            }, root)
        selected = result["selected_plan"]
        self.assertEqual(selected["tail_bound_method"], "explicit-risk-budget")
        self.assertTrue(selected["tail_bound_promotable"])
        self.assertEqual(selected["response_reservation_us"], 590.0)

    def test_replays_quota_only_candidate_search_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(json.dumps(profile(700, 580, 40, 30)))
            spec = {
                "schema_version": 1,
                "system": "QUIET",
                "deadline_us": 760,
                "critical_lookahead_us": 0,
                "candidate_search": {
                    "candidate_count": 2,
                    "multi_candidate_evaluated": True,
                    "placement_variant_count": 1,
                    "placement_search_evaluated": False,
                    "claim_status": "multi-candidate-quota-search-only",
                },
                "candidates": [
                    {
                        "candidate_id": "q50-q50",
                        "profile_path": "profile.json",
                        "placement": {"producer": "1g-q50", "consumer": "2g-q100"},
                        "placement_variant": "fixed-1g-producer-2g-consumer",
                        "background_goodput_rps": 100,
                        "pre_release_guard_p99_us": 0,
                        "reservation_margin_us": 0,
                    },
                    {
                        "candidate_id": "q100-q100",
                        "profile_path": "profile.json",
                        "placement": {"producer": "1g-q100", "consumer": "2g-q100"},
                        "placement_variant": "fixed-1g-producer-2g-consumer",
                        "background_goodput_rps": 200,
                        "pre_release_guard_p99_us": 0,
                        "reservation_margin_us": 0,
                    },
                ],
            }
            result = select_plan(spec, root)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_plan"]["candidate_id"], "q100-q100")
        self.assertFalse(result["candidate_search"]["placement_search_evaluated"])


class QuietStageDagTest(unittest.TestCase):
    def test_actuation_binds_uuid_transport_quota_and_scope(self) -> None:
        plan = {
            "proposed_system": "QUIET",
            "status": "selected",
            "selected_plan": {
                "candidate_id": "q100-q25",
                "feasible": True,
                "placement": {"producer": "1g-q100", "consumer": "2g-q100"},
                "protection_scope": "producer",
                "dag": {"edges": [{
                    "transport": "registered-shared-sysmem-direct-binding",
                }]},
            },
        }
        manifest = actuate_selected_plan(
            plan,
            {"JDG_MIG_SMALL_UUID": "small", "JDG_MIG_BIG_UUID": "big"},
            "fixed-1g-producer-2g-consumer",
            "registered-direct",
            100,
            25,
            "producer",
            "resnet-control",
            "dependent",
        )
        self.assertEqual(manifest["producer_uuid"], "small")
        self.assertEqual(manifest["consumer_uuid"], "big")
        self.assertEqual(manifest["background_quota_percent"], 25)
        self.assertTrue(manifest["admission"]["before_cuda_context"])
        with self.assertRaisesRegex(ValueError, "transport differs"):
            actuate_selected_plan(
                plan,
                {"JDG_MIG_SMALL_UUID": "small", "JDG_MIG_BIG_UUID": "big"},
                "fixed-1g-producer-2g-consumer",
                "pinned-bounce",
                100,
                25,
                "producer",
                "resnet-control",
                "dependent",
            )

    def test_selects_highest_goodput_feasible_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fast.json").write_text(json.dumps(profile(690, 580, 40, 30)))
            (root / "infeasible.json").write_text(
                json.dumps(profile(800, 650, 80, 40))
            )
            spec = {
                "schema_version": 1,
                "system": "QUIET",
                "deadline_us": 760,
                "critical_lookahead_us": 900,
                "candidates": [
                    {
                        "candidate_id": "internal-fast",
                        "profile_path": "fast.json",
                        "placement": {"producer": "1g", "consumer": "2g"},
                        "background_goodput_rps": 700,
                        "pre_release_guard_p99_us": 880,
                        "reservation_margin_us": 25,
                    },
                    {
                        "candidate_id": "internal-infeasible",
                        "profile_path": "infeasible.json",
                        "placement": {"producer": "1g", "consumer": "2g"},
                        "background_goodput_rps": 900,
                        "pre_release_guard_p99_us": 0,
                        "reservation_margin_us": 0,
                    },
                ],
            }
            result = select_plan(spec, root)
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_plan"]["candidate_id"], "internal-fast")
        self.assertEqual(result["selected_plan"]["response_reservation_us"], 685)
        self.assertEqual(result["selected_plan"]["reserved_slack_us"], 75)
        self.assertEqual(result["selected_plan"]["release_lead_time_us"], 880)
        self.assertEqual(result["selected_plan"]["uncovered_guard_us"], 0)

    def test_preserves_protection_scope_for_runtime_actuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(json.dumps(profile(700, 580, 40, 30)))
            result = select_plan({
                "schema_version": 1,
                "system": "QUIET",
                "deadline_us": 760,
                "critical_lookahead_us": 0,
                "candidates": [{
                    "candidate_id": "consumer-protected",
                    "profile_path": "profile.json",
                    "placement": {"producer": "2g", "consumer": "1g"},
                    "placement_variant": "fixed-2g-producer-1g-consumer",
                    "protection_scope": "consumer",
                    "background_goodput_rps": 100,
                    "pre_release_guard_p99_us": 0,
                    "reservation_margin_us": 0,
                }],
            }, root)
        self.assertEqual(result["selected_plan"]["protection_scope"], "consumer")

    def test_uncovered_guard_is_charged_to_arrival_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(
                json.dumps(profile(690, 580, 40, 30))
            )
            spec = {
                "schema_version": 1,
                "system": "QUIET",
                "deadline_us": 760,
                "critical_lookahead_us": 100,
                "candidates": [
                    {
                        "candidate_id": "slow-drain",
                        "profile_path": "profile.json",
                        "placement": {"producer": "1g", "consumer": "2g"},
                        "background_goodput_rps": 700,
                        "pre_release_guard_p99_us": 880,
                        "reservation_margin_us": 0,
                    }
                ],
            }
            result = select_plan(spec, root)
        self.assertEqual(result["status"], "no-feasible-plan")
        self.assertEqual(result["candidates"][0]["uncovered_guard_us"], 780)
        self.assertEqual(
            result["candidates"][0]["arrival_to_completion_reservation_us"],
            1440,
        )

    def test_rejects_public_alias_and_non_payload_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = profile(690, 580, 40, 30)
            bad["unique_payload_checksums"] = 1
            (root / "bad.json").write_text(json.dumps(bad))
            spec = {
                "schema_version": 1,
                "system": "mig-governor",
                "deadline_us": 760,
                "critical_lookahead_us": 0,
                "candidates": [],
            }
            with self.assertRaisesRegex(ValueError, "only public"):
                select_plan(spec, root)
            spec["system"] = "QUIET"
            spec["candidates"] = [
                {
                    "candidate_id": "bad",
                    "profile_path": "bad.json",
                    "placement": {"producer": "1g", "consumer": "2g"},
                    "background_goodput_rps": 1,
                    "pre_release_guard_p99_us": 0,
                    "reservation_margin_us": 0,
                }
            ]
            result = select_plan(spec, root)
        self.assertEqual(result["status"], "no-feasible-plan")
        self.assertIn("changing payloads", result["candidates"][0]["error"])

    def test_accepts_whisper_large_edge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = profile(
                1535, 1462, 9, 59,
                "whisper-last-hidden-state-to-projection-mlp", 2_304_000,
            )
            (root / "whisper.json").write_text(json.dumps(value))
            spec = {
                "schema_version": 1,
                "system": "QUIET",
                "deadline_us": 1600,
                "critical_lookahead_us": 0,
                "candidates": [{
                    "candidate_id": "whisper-direct",
                    "profile_path": "whisper.json",
                    "placement": {"producer": "1g", "consumer": "2g"},
                    "background_goodput_rps": 0,
                    "pre_release_guard_p99_us": 0,
                    "reservation_margin_us": 0,
                }],
            }
            result = select_plan(spec, root)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(
            result["selected_plan"]["dag"]["stages"][0]["id"],
            "audio-encoder",
        )

    def test_rejects_stale_candidate_search_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(
                json.dumps(profile(690, 580, 40, 30))
            )
            spec = {
                "schema_version": 1,
                "system": "QUIET",
                "deadline_us": 760,
                "critical_lookahead_us": 0,
                "candidate_search": {
                    "candidate_count": 2,
                    "multi_candidate_evaluated": True,
                    "claim_status": "multi-candidate-search",
                },
                "candidates": [{
                    "candidate_id": "q100-q100",
                    "profile_path": "profile.json",
                    "placement": {"producer": "1g", "consumer": "2g"},
                    "background_goodput_rps": 1,
                    "pre_release_guard_p99_us": 0,
                    "reservation_margin_us": 0,
                }],
            }
            with self.assertRaisesRegex(ValueError, "candidate_search count"):
                select_plan(spec, root)


if __name__ == "__main__":
    unittest.main()
