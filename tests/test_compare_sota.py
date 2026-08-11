#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACE_EVIDENCE = ROOT / "analysis" / "compare_sota.py"
TRACE_EVIDENCE_SHA256 = hashlib.sha256(TRACE_EVIDENCE.read_bytes()).hexdigest()
SPEC = importlib.util.spec_from_file_location(
    "compare_sota", ROOT / "analysis" / "compare_sota.py"
)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARE
SPEC.loader.exec_module(COMPARE)


class CompareSotaTest(unittest.TestCase):
    @staticmethod
    def quiet_summary(scenario: str = "dependent") -> dict:
        return {
            "deadline_ms": 6.0,
            "common_workload": {
                "workload_id": "resnet-control",
                "topology": "fixed-2g+1g",
                "placement": "fixed-1g-producer-2g-consumer",
                "input_tensor": "Layer7_cov",
                "payload_bytes": 14720,
                "arrival_trace_sha256": "1" * 64,
                "dataset_manifest_sha256": "2" * 64,
            },
            "config": {
                "scenario": scenario,
                "epochs": 36,
                "samples_per_epoch": 800,
                "period_ms": 20.0,
                "pressure_rps_per_tenant": 0.0,
                "burst_size": 8,
                "dmr_target": 0.0005,
                "critical_placement": "2g",
                "resident_placement": "1g",
                "borrower_placement": "2g",
                "includes_transfers": True,
                "worker_max_inflight": 1,
            },
            "policies": [
                {
                    "name": "mig-governor",
                    "pressure_goodput_per_second": 900.0,
                    "deadline_miss_rate": 0.0,
                    "critical_p99_ms_max": 5.0,
                    "critical_requests": 800,
                    "deadline_misses": 0,
                }
            ],
        }

    def test_extracts_only_quiet_as_proposed_system(self) -> None:
        metrics = COMPARE.quiet_metrics(self.quiet_summary())
        self.assertEqual(metrics["pressure_goodput_per_second"], 900.0)
        self.assertEqual(metrics["p99_ms"], 5.0)

    def test_production_wall_requires_completion_before_validation_contract(self) -> None:
        summary = self.quiet_summary()
        summary["latency_contract"] = "production-wall-arrival-to-completion"
        with self.assertRaisesRegex(ValueError, "production-wall contract"):
            COMPARE.quiet_contract(summary, "dependent")
        summary["production_wall_definition"] = (
            "arrival-to-consumer-completion-excludes-correctness-validation"
        )
        contract = COMPARE.quiet_contract(summary, "dependent")
        self.assertEqual(
            contract["production_wall_definition"],
            "arrival-to-consumer-completion-excludes-correctness-validation",
        )
        self.assertEqual(
            contract["latency_contract"],
            "production-wall-arrival-to-completion",
        )

    def test_learned_workload_requires_bound_producer_trace(self) -> None:
        summary = self.quiet_summary()
        summary["common_workload"] = {
            **summary["common_workload"],
            "workload_id": "resnet50-classification",
        }
        with self.assertRaisesRegex(ValueError, "producer_input_trace"):
            COMPARE.quiet_contract(summary, "dependent")

    def test_exploratory_evidence_cannot_be_formal_ranked(self) -> None:
        summary = self.quiet_summary()
        self.assertFalse(COMPARE.formal_evidence_bound(summary))
        summary.update({
            "formal": True,
            "thermal_normalized": True,
            "ranking_allowed": True,
            "deadline_lock_sha256": "a" * 64,
            "thermal_lock_sha256": "b" * 64,
            "session_level_statistics": {
                "run_count": 14,
                "unit": "independent-session",
                "paired_williams": True,
            },
            "slo_certification": {
                "method": "one-sided-clopper-pearson-95",
                "cp95_upper_dmr": 0.0004,
            },
        })
        self.assertTrue(COMPARE.formal_evidence_bound(summary))
        summary["thermal_normalized"] = False
        self.assertFalse(COMPARE.formal_evidence_bound(summary))

    def test_same_contract_competitor_is_accepted(self) -> None:
        contract = COMPARE.quiet_contract(self.quiet_summary(), "dependent")
        competitor = {
            "system": "BOER",
            "contract": contract,
            "provenance": {
                "upstream_commit": COMPARE.SYSTEMS["BOER"]["source_commit"],
                "fidelity": COMPARE.SYSTEMS["BOER"]["required_fidelity"],
                "thor_profile_sha256": {"audio": "a" * 64},
            },
            "metrics": {
                "pressure_goodput_per_second": 1000.0,
                "deadline_miss_rate": 0.0,
                "p99_ms": 5.0,
                "critical_requests": 800,
                "deadline_misses": 0,
            },
        }
        COMPARE.validate_competitor(competitor, "BOER", contract)

    def test_unpinned_port_is_rejected(self) -> None:
        contract = COMPARE.quiet_contract(self.quiet_summary(), "dependent")
        competitor = {
            "system": "BOER",
            "contract": contract,
            "provenance": {
                "upstream_commit": "0" * 40,
                "fidelity": "algorithm-preserving-thor-port",
                "thor_profile_sha256": {"audio": "a" * 64},
            },
            "metrics": {
                "pressure_goodput_per_second": 1000.0,
                "deadline_miss_rate": 0.0,
                "p99_ms": 5.0,
                "critical_requests": 800,
                "deadline_misses": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "pinned artifact"):
            COMPARE.validate_competitor(competitor, "BOER", contract)

    def test_cross_scenario_numbers_are_rejected(self) -> None:
        contract = COMPARE.quiet_contract(self.quiet_summary(), "dependent")
        competitor = {
            "system": "Orion",
            "contract": contract | {"scenario": "independent"},
            "provenance": {
                "upstream_commit": COMPARE.SYSTEMS["Orion"]["source_commit"],
                "fidelity": COMPARE.SYSTEMS["Orion"]["required_fidelity"],
                "thor_profile_sha256": {"audio": "a" * 64},
            },
            "metrics": {
                "pressure_goodput_per_second": 1000.0,
                "deadline_miss_rate": 0.0,
                "p99_ms": 5.0,
                "critical_requests": 800,
                "deadline_misses": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "scenario"):
            COMPARE.validate_competitor(competitor, "Orion", contract)

    def test_cross_workload_numbers_are_rejected(self) -> None:
        contract = COMPARE.quiet_contract(self.quiet_summary(), "dependent")
        competitor = {
            "system": "BOER",
            "contract": contract | {
                "common_workload": {
                    **contract["common_workload"],
                    "workload_id": "whisper-projection",
                }
            },
            "provenance": {
                "upstream_commit": COMPARE.SYSTEMS["BOER"]["source_commit"],
                "fidelity": COMPARE.SYSTEMS["BOER"]["required_fidelity"],
                "thor_profile_sha256": {"audio": "a" * 64},
            },
            "metrics": {
                "pressure_goodput_per_second": 1000.0,
                "deadline_miss_rate": 0.0,
                "p99_ms": 5.0,
                "critical_requests": 800,
                "deadline_misses": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "common_workload"):
            COMPARE.validate_competitor(competitor, "BOER", contract)

    def test_orion_requires_upstream_differential_gate(self) -> None:
        contract = COMPARE.quiet_contract(self.quiet_summary(), "dependent")
        competitor = {
            "system": "Orion", "contract": contract,
            "provenance": {
                "upstream_commit": COMPARE.SYSTEMS["Orion"]["source_commit"],
                "fidelity": COMPARE.SYSTEMS["Orion"]["required_fidelity"],
                "thor_profile_sha256": {"trace": "a" * 64},
            },
            "metrics": {
                "pressure_goodput_per_second": 1000.0,
                "deadline_miss_rate": 0.0, "p99_ms": 5.0,
                "critical_requests": 800, "deadline_misses": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "differential fidelity"):
            COMPARE.validate_competitor(competitor, "Orion", contract)
        competitor["provenance"]["differential_gate"] = {
            "schema_version": 1,
            "kind": "orion-differential-fidelity-gate",
            "system": "Orion",
            "status": "passed", "reference": "pinned-upstream-scheduler",
            "upstream_commit": COMPARE.SYSTEMS["Orion"]["source_commit"],
            "decision_cases": 12, "mismatch_cases": 0,
            "numeric_comparison_allowed": True,
            "reference_checkout_verified": True,
            "reference_git_root": "/tmp/orion",
            "reference_git_head": COMPARE.SYSTEMS["Orion"]["source_commit"],
            "reference_source_relative_path": "src/scheduler/scheduler.cpp",
            "reference_source_path": "/tmp/pinned-orion-source.cc",
            "reference_source_sha256": "d" * 64,
            "reference_source_verified": True,
            "upstream_runtime_binary_path": str(TRACE_EVIDENCE),
            "upstream_runtime_binary_sha256": TRACE_EVIDENCE_SHA256,
            "upstream_runtime_binary_verified": True,
            "reference_trace_path": str(TRACE_EVIDENCE),
            "reference_trace_sha256": "b" * 64,
            "port_trace_path": str(TRACE_EVIDENCE),
            "port_trace_sha256": "c" * 64,
        }
        competitor["provenance"]["differential_gate"]["reference_trace_sha256"] = TRACE_EVIDENCE_SHA256
        competitor["provenance"]["differential_gate"]["port_trace_sha256"] = TRACE_EVIDENCE_SHA256
        competitor["provenance"]["differential_gate"]["common_workload"] = contract["common_workload"]
        competitor["provenance"]["differential_gate"]["reference_trace_provenance"] = {
            "generator": "pinned-upstream-orion-runtime",
            "sha256": "e" * 64,
            "reference_trace_sha256": TRACE_EVIDENCE_SHA256,
            "upstream_runtime_binary_path": str(TRACE_EVIDENCE),
            "upstream_runtime_binary_sha256": TRACE_EVIDENCE_SHA256,
            "common_workload_sha256": "f" * 64,
        }
        COMPARE.validate_competitor(competitor, "Orion", contract)
        competitor["provenance"]["differential_gate"]["port_trace_path"] = "/missing/orion-port.jsonl"
        with self.assertRaisesRegex(ValueError, "differential fidelity"):
            COMPARE.validate_competitor(competitor, "Orion", contract)

    def test_orion_workload_gate_mismatch_is_rejected(self) -> None:
        contract = COMPARE.quiet_contract(self.quiet_summary(), "dependent")
        gate = {
            "schema_version": 1,
            "kind": "orion-differential-fidelity-gate",
            "system": "Orion",
            "status": "passed",
            "reference": "pinned-upstream-scheduler",
            "upstream_commit": COMPARE.SYSTEMS["Orion"]["source_commit"],
            "decision_cases": 1,
            "mismatch_cases": 0,
            "numeric_comparison_allowed": True,
            "reference_checkout_verified": True,
            "reference_git_root": "/tmp/orion",
            "reference_git_head": COMPARE.SYSTEMS["Orion"]["source_commit"],
            "reference_source_relative_path": "src/scheduler/scheduler.cpp",
            "reference_source_path": "/tmp/pinned-orion-source.cc",
            "reference_source_sha256": "d" * 64,
            "reference_source_verified": True,
            "reference_trace_path": str(TRACE_EVIDENCE),
            "reference_trace_sha256": TRACE_EVIDENCE_SHA256,
            "port_trace_path": str(TRACE_EVIDENCE),
            "port_trace_sha256": TRACE_EVIDENCE_SHA256,
            "common_workload": {
                **contract["common_workload"],
                "workload_id": "whisper-projection",
            },
        }
        competitor = {
            "system": "Orion",
            "contract": contract,
            "provenance": {
                "upstream_commit": COMPARE.SYSTEMS["Orion"]["source_commit"],
                "fidelity": COMPARE.SYSTEMS["Orion"]["required_fidelity"],
                "thor_profile_sha256": {"trace": "a" * 64},
                "differential_gate": gate,
            },
            "metrics": {
                "pressure_goodput_per_second": 1.0,
                "deadline_miss_rate": 0.0,
                "p99_ms": 1.0,
                "critical_requests": 1,
                "deadline_misses": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "workload gate"):
            COMPARE.validate_competitor(competitor, "Orion", contract)
        competitor["provenance"]["differential_gate"]["port_trace_path"] = str(TRACE_EVIDENCE)
        del competitor["provenance"]["differential_gate"]["reference_checkout_verified"]
        with self.assertRaisesRegex(ValueError, "differential fidelity"):
            COMPARE.validate_competitor(competitor, "Orion", contract)

    def test_orion_gate_schema_and_commit_are_bound(self) -> None:
        contract = COMPARE.quiet_contract(self.quiet_summary(), "dependent")
        competitor = {
            "system": "Orion", "contract": contract,
            "provenance": {
                "upstream_commit": COMPARE.SYSTEMS["Orion"]["source_commit"],
                "fidelity": COMPARE.SYSTEMS["Orion"]["required_fidelity"],
                "thor_profile_sha256": {"trace": "a" * 64},
                "differential_gate": {
                    "schema_version": 1,
                    "kind": "orion-differential-fidelity-gate",
                    "system": "Orion",
                    "status": "passed",
                    "reference": "pinned-upstream-scheduler",
                    "upstream_commit": "0" * 40,
                    "decision_cases": 12,
                    "mismatch_cases": 0,
                    "numeric_comparison_allowed": True,
                    "reference_checkout_verified": True,
                    "reference_git_root": "/tmp/orion",
                    "reference_git_head": "0" * 40,
                    "reference_source_relative_path": "src/scheduler/scheduler.cpp",
                    "reference_source_path": "/tmp/pinned-orion-source.cc",
                    "reference_source_sha256": "d" * 64,
                    "reference_source_verified": True,
                    "reference_trace_path": str(TRACE_EVIDENCE),
                    "reference_trace_sha256": "b" * 64,
                    "port_trace_path": str(TRACE_EVIDENCE),
                    "port_trace_sha256": "c" * 64,
                },
            },
            "metrics": {
                "pressure_goodput_per_second": 1000.0,
                "deadline_miss_rate": 0.0, "p99_ms": 5.0,
                "critical_requests": 800, "deadline_misses": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "differential fidelity"):
            COMPARE.validate_competitor(competitor, "Orion", contract)

    def test_pantheon_requires_common_workload_accuracy_gate(self) -> None:
        contract = COMPARE.quiet_contract(self.quiet_summary(), "dependent")
        competitor = {
            "system": "Pantheon", "contract": contract,
            "provenance": {
                "upstream_commit": COMPARE.SYSTEMS["Pantheon"]["source_commit"],
                "fidelity": COMPARE.SYSTEMS["Pantheon"]["required_fidelity"],
                "thor_profile_sha256": {"trace": "a" * 64},
            },
            "metrics": {
                "pressure_goodput_per_second": 1000.0,
                "deadline_miss_rate": 0.0, "p99_ms": 5.0,
                "critical_requests": 800, "deadline_misses": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "accuracy gate"):
            COMPARE.validate_competitor(competitor, "Pantheon", contract)
        competitor["provenance"]["common_workload_adapter"] = {
            "schema_version": 1,
            "kind": "pantheon-common-workload-accuracy-gate",
            "system": "Pantheon",
            "status": "passed",
            "upstream_commit": COMPARE.SYSTEMS["Pantheon"]["source_commit"],
            "workload": "p9-dependent-tensorrt-dag",
            "deadline_us": 6000.0,
            "accuracy_equivalent": True, "shared_arrival_trace": True,
            "decision_cases": 12,
            "upstream_source_path": "/tmp/pinned-pantheon-source.cc",
            "upstream_source_sha256": "a" * 64,
            "upstream_source_verified": True,
            "runtime_binary_path": str(TRACE_EVIDENCE),
            "runtime_binary_sha256": TRACE_EVIDENCE_SHA256,
            "runtime_binary_verified": True,
            "upstream_checkout_verified": True,
            "upstream_git_head": COMPARE.SYSTEMS["Pantheon"]["source_commit"],
            "upstream_git_root": "/tmp/pinned-pantheon",
            "upstream_source_relative_path": "src/pantheon_scheduler.cc",
            "training_result_path": "/tmp/pantheon-training.json",
            "training_result_sha256": "d" * 64,
            "training_artifact_verified": True,
            "reference_trace_path": str(TRACE_EVIDENCE),
            "reference_trace_sha256": "b" * 64,
            "port_trace_path": str(TRACE_EVIDENCE),
            "port_trace_sha256": "c" * 64,
            "reference_accuracy": 0.90, "pantheon_accuracy": 0.895,
            "accuracy_delta": 0.005, "accuracy_tolerance": 0.01,
            "numeric_comparison_allowed": True,
        }
        competitor["provenance"]["common_workload_adapter"]["reference_trace_sha256"] = TRACE_EVIDENCE_SHA256
        competitor["provenance"]["common_workload_adapter"]["port_trace_sha256"] = TRACE_EVIDENCE_SHA256
        competitor["provenance"]["common_workload_adapter"]["common_workload"] = contract["common_workload"]
        COMPARE.validate_competitor(competitor, "Pantheon", contract)
        broken = json.loads(json.dumps(competitor))
        del broken["provenance"]["common_workload_adapter"]["upstream_checkout_verified"]
        with self.assertRaisesRegex(ValueError, "accuracy gate"):
            COMPARE.validate_competitor(broken, "Pantheon", contract)

    def test_runtime_sota_set_includes_top_tier_and_edge_systems(self) -> None:
        self.assertEqual(COMPARE.SYSTEMS["Orion"]["paper"], "EuroSys 2024")
        self.assertEqual(COMPARE.SYSTEMS["XSched"]["paper"], "OSDI 2025")
        self.assertEqual(COMPARE.SYSTEMS["Pantheon"]["paper"], "MobiSys 2024")
        self.assertEqual(
            COMPARE.SYSTEMS["XSched"]["required_fidelity"], "native-xqueue-port"
        )
        self.assertEqual(
            COMPARE.SYSTEMS["Pantheon"]["required_fidelity"],
            "native-block-runtime-port",
        )

    def test_manifest_claim_boundary_blocks_structural_numeric_promotion(self) -> None:
        self.assertEqual(
            COMPARE.claim_contract("BOER")["claim_level"],
            "functional-or-structural-only",
        )
        self.assertFalse(COMPARE.claim_contract("BOER")["numeric_comparison_allowed"])
        self.assertEqual(
            COMPARE.claim_contract("XSched")["claim_level"],
            "numeric-frontier",
        )
        self.assertTrue(COMPARE.claim_contract("XSched")["numeric_comparison_allowed"])
        self.assertTrue(COMPARE.claim_contract("QUIET")["numeric_comparison_allowed"])

    def test_quiet_accuracy_gate_requires_byte_bound_pass_record(self) -> None:
        summary = self.quiet_summary()
        self.assertFalse(COMPARE.application_accuracy_gate_bound(summary))
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / "gate.json"
            dataset = root / "dataset.jsonl"
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            reference_pipeline = root / "reference.csv"
            candidate_pipeline = root / "candidate.csv"
            for evidence in (dataset, reference, candidate, reference_pipeline, candidate_pipeline):
                evidence.write_text("evidence\n", encoding="utf-8")
            gate = {
                "kind": "p9-application-accuracy-gate",
                "status": "passed",
                "numeric_comparison_allowed": True,
                "minimum_accuracy": 0.90,
                "reference_accuracy": 1.0,
                "candidate_accuracy": 1.0,
                "application_input_binding_required": True,
                "application_input_binding_contract": "passed",
                "dataset_manifest_path": str(dataset),
                "dataset_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "reference_trace_path": str(reference),
                "reference_trace_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "candidate_trace_path": str(candidate),
                "candidate_trace_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "reference_pipeline_csv": {
                    "path": str(reference_pipeline),
                    "sha256": hashlib.sha256(reference_pipeline.read_bytes()).hexdigest(),
                },
                "candidate_pipeline_csv": {
                    "path": str(candidate_pipeline),
                    "sha256": hashlib.sha256(candidate_pipeline.read_bytes()).hexdigest(),
                },
            }
            raw = (json.dumps(gate) + "\n").encode()
            path.write_bytes(raw)
            summary["application_accuracy_gate"] = {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            self.assertTrue(COMPARE.application_accuracy_gate_bound(summary))

    def test_accuracy_gate_rejects_equal_but_low_absolute_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            gate_path = root / "gate.json"
            paths = {name: root / f"{name}.jsonl" for name in ("dataset", "reference", "candidate")}
            for path in paths.values():
                path.write_text("original\n", encoding="utf-8")
            gate = {
                "kind": "p9-application-accuracy-gate", "status": "passed",
                "numeric_comparison_allowed": True, "minimum_accuracy": 0.9,
                "reference_accuracy": 0.0, "candidate_accuracy": 0.0,
                "dataset_manifest_path": str(paths["dataset"]),
                "dataset_manifest_sha256": hashlib.sha256(paths["dataset"].read_bytes()).hexdigest(),
                "reference_trace_path": str(paths["reference"]),
                "reference_trace_sha256": hashlib.sha256(paths["reference"].read_bytes()).hexdigest(),
                "candidate_trace_path": str(paths["candidate"]),
                "candidate_trace_sha256": hashlib.sha256(paths["candidate"].read_bytes()).hexdigest(),
            }
            raw = (json.dumps(gate) + "\n").encode()
            gate_path.write_bytes(raw)
            summary = self.quiet_summary()
            summary["application_accuracy_gate"] = {
                "path": str(gate_path), "sha256": hashlib.sha256(raw).hexdigest(),
            }
            self.assertFalse(COMPARE.application_accuracy_gate_bound(summary))

    def test_accuracy_gate_rejects_replaced_request_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            gate_path = root / "gate.json"
            paths = {name: root / f"{name}.jsonl" for name in ("dataset", "reference", "candidate")}
            for path in paths.values():
                path.write_text("original\n", encoding="utf-8")
            gate = {
                "kind": "p9-application-accuracy-gate",
                "status": "passed",
                "numeric_comparison_allowed": True,
                "minimum_accuracy": 0.90,
                "reference_accuracy": 1.0,
                "candidate_accuracy": 1.0,
                "dataset_manifest_path": str(paths["dataset"]),
                "dataset_manifest_sha256": hashlib.sha256(paths["dataset"].read_bytes()).hexdigest(),
                "reference_trace_path": str(paths["reference"]),
                "reference_trace_sha256": hashlib.sha256(paths["reference"].read_bytes()).hexdigest(),
                "candidate_trace_path": str(paths["candidate"]),
                "candidate_trace_sha256": hashlib.sha256(paths["candidate"].read_bytes()).hexdigest(),
            }
            gate_raw = (json.dumps(gate) + "\n").encode()
            gate_path.write_bytes(gate_raw)
            summary = self.quiet_summary()
            summary["application_accuracy_gate"] = {
                "path": str(gate_path),
                "sha256": hashlib.sha256(gate_raw).hexdigest(),
            }
            paths["reference"].write_text("replaced\n", encoding="utf-8")
            self.assertFalse(COMPARE.application_accuracy_gate_bound(summary))

    def test_numeric_comparison_requires_competitor_accuracy_gate(self) -> None:
        # The direct validator checks published provenance; promotion also
        # requires a byte-bound application gate for the competitor output.
        proposed = {"metrics": {"pressure_goodput_per_second": 1.0}}
        competitor = {"metrics": {"pressure_goodput_per_second": 1.0}}
        self.assertFalse(COMPARE.numeric_accuracy_contract_bound(proposed, competitor))


if __name__ == "__main__":
    unittest.main()
