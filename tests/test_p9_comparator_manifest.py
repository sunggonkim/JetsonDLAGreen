import json
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ComparatorManifestTest(unittest.TestCase):
    @unittest.skipUnless(
        (ROOT / "results/p9-pantheon-native-positive-20260810T020937Z/verification.json").is_file(),
        "requires preserved local comparator evidence",
    )
    def test_headline_contract_has_one_proposed_system(self) -> None:
        value = json.loads((ROOT / "docs/p9-comparator-manifest.json").read_text())
        order = value["headline_order"]
        self.assertEqual(order, ["NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "Pantheon", "QUIET"])
        rows = value["rows"]
        self.assertEqual(sum(item["class"] == "proposed-system" for item in rows.values()), 1)
        self.assertEqual(value["proposed_system"], "QUIET")
        self.assertEqual(
            value["designated_published_comparators"],
            ["Orion", "XSched", "Pantheon"],
        )
        self.assertFalse(rows["Orion"]["numeric_comparison_allowed"])
        self.assertTrue(rows["XSched"]["numeric_comparison_allowed"])
        self.assertTrue(rows["XSched"]["numeric_candidate"])
        self.assertTrue(rows["Pantheon"]["numeric_comparison_allowed"])
        pantheon_control = rows["Pantheon"]["functional_positive_control"]
        control_path = ROOT / pantheon_control["path"]
        self.assertEqual(
            hashlib.sha256(control_path.read_bytes()).hexdigest(),
            pantheon_control["sha256"],
        )
        pantheon_probe = rows["Pantheon"]["training_probe"]
        probe_path = ROOT / pantheon_probe["path"]
        self.assertEqual(
            hashlib.sha256(probe_path.read_bytes()).hexdigest(),
            pantheon_probe["sha256"],
        )
        self.assertFalse(pantheon_probe["formal_training_contract"])
        self.assertFalse(pantheon_probe["accuracy_gate_passed"])
        self.assertTrue({
            "upstream_checkout_verified", "upstream_git_root",
            "upstream_git_head", "upstream_source_relative_path",
        }.issubset(set(rows["Pantheon"]["required_gate_provenance"])))
        policy = value["paper_table_policy"]
        self.assertIn("post_completion_application_output_trace", policy["formal_required_gates"])
        self.assertEqual(policy["proposed_system"], "QUIET")
        self.assertEqual(
            policy["executed_result_order"],
            [
                "QUIET", "NVIDIA MPS", "XSched", "Pantheon", "Orion",
                "BLESS", "NVIDIA MIG", "GSLICE", "gpulet", "BOER",
                "ParvaGPU", "DeepPlan",
            ],
        )
        self.assertEqual(policy["direct_ranking_order"], ["QUIET", "NVIDIA MPS", "XSched"])
        self.assertEqual(policy["numeric_frontier_order"], ["NVIDIA MPS", "QUIET"])
        self.assertNotIn("Orion", policy["numeric_frontier_order"])
        self.assertNotIn("Pantheon", policy["numeric_frontier_order"])
        evidence = rows["QUIET"]["latest_exploratory_evidence"]
        self.assertEqual(evidence["points"], 6)
        self.assertEqual(evidence["sessions_per_point"], 3)
        self.assertFalse(evidence["formal"])
        self.assertFalse(evidence["thermal_normalized"])
        self.assertFalse(evidence["application_accuracy_bound"])
        xsched = value["rows"]["XSched"]["latest_exploratory_evidence"]["latest_native_rerun"]
        self.assertFalse(xsched["formal"])
        self.assertEqual(xsched["checksum_failures"], 0)
        self.assertEqual(xsched["suspend_transitions"], 4)
        self.assertEqual(xsched["resume_transitions"], 3)
        active = value["latest_active_comparator_smoke"]
        self.assertEqual(active["systems"], ["QUIET", "NVIDIA MPS", "XSched"])
        self.assertFalse(active["formal"])
        self.assertFalse(active["ranking_allowed"])
        self.assertEqual(active["requests_per_system"], 100)
        sequence = value["latest_active_williams_sequence"]
        self.assertEqual(sequence["systems"], ["NVIDIA MPS", "XSched", "QUIET"])
        self.assertEqual(sequence["placement_variant"], "fixed-1g-producer-2g-consumer")
        self.assertFalse(sequence["formal"])
        self.assertFalse(sequence["ranking_allowed"])
        self.assertEqual(sequence["arms"]["QUIET"]["deadline_misses"], 0)
        sequence_path = ROOT / sequence["path"]
        self.assertEqual(
            hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
            sequence["sha256"],
        )
        repeats = value["latest_active_williams_repeats"]
        self.assertEqual(repeats["sequence_count"], 3)
        self.assertFalse(repeats["formal"])
        self.assertFalse(repeats["ranking_allowed"])
        repeats_path = ROOT / repeats["path"]
        self.assertEqual(
            hashlib.sha256(repeats_path.read_bytes()).hexdigest(),
            repeats["sha256"],
        )
        self.assertEqual(repeats["systems_summary"]["QUIET"]["deadline_misses"], 0)
        load = value["rows"]["QUIET"]["latest_load_frontier"]
        self.assertEqual(load["cp95_slo_qualified_points"], 0)
        self.assertFalse(load["formal"])
        load_path = ROOT / load["path"]
        self.assertEqual(hashlib.sha256(load_path.read_bytes()).hexdigest(), load["sha256"])
        current = value["latest_current_production_wall_smoke"]
        self.assertEqual(current["systems"], ["QUIET", "NVIDIA MPS", "XSched"])
        self.assertFalse(current["formal"])
        self.assertFalse(current["ranking_allowed"])
        current_path = ROOT / current["path"]
        self.assertEqual(
            hashlib.sha256(current_path.read_bytes()).hexdigest(), current["sha256"]
        )
        self.assertEqual(current["arms"]["QUIET"]["deadline_misses"], 0)

        # Production-wall evidence must bind the same request/output contract
        # for every numeric smoke arm.  The trace hashes are part of the
        # comparator boundary, not decorative metadata.
        trace_contract = current["trace_contract"]
        self.assertEqual(
            trace_contract["request_capture_boundary"], "production-wall-measurement"
        )
        self.assertEqual(trace_contract["output_capture_boundary"], "post-completion")
        self.assertEqual(
            set(trace_contract["request_traces"]),
            {"QUIET", "NVIDIA MPS", "XSched"},
        )
        self.assertEqual(
            set(trace_contract["application_output_traces"]),
            {"QUIET", "NVIDIA MPS", "XSched"},
        )
        for category in ("request_traces", "application_output_traces"):
            for system, evidence in trace_contract[category].items():
                path = ROOT / evidence["path"]
                self.assertTrue(path.is_file(), f"missing {category} for {system}")
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), evidence["sha256"]
                )

    def test_latest_fast_smoke_is_directional_and_diagnostic_is_not_promotable(self) -> None:
        value = json.loads((ROOT / "docs/p9-comparator-manifest.json").read_text())
        smoke = value["latest_fast_causal_smoke"]
        self.assertFalse(smoke["formal"])
        self.assertEqual(smoke["claim"], "directional production-wall causal smoke; not a ranked frontier")
        self.assertEqual(smoke["arms"]["QUIET-dependent"]["deadline_misses"], 0)
        independent = smoke["arms"]["QUIET-independent"]
        self.assertEqual(independent["claim_status"], "diagnostic-only-plan-violation")
        self.assertGreater(independent["gate_p99_us"], independent["lookahead_us"])

    @unittest.skipUnless(
        (ROOT / "results/p9-kitti-car-external-smoke-20260811-25aug/accuracy-threshold-sweep.json").is_file(),
        "requires preserved local external-accuracy evidence",
    )
    def test_failed_external_accuracy_gate_cannot_be_ranked(self) -> None:
        value = json.loads((ROOT / "docs/p9-comparator-manifest.json").read_text())
        gate = value["rows"]["QUIET"]["latest_external_accuracy_gate"]
        self.assertEqual(gate["status"], "rejected")
        self.assertFalse(gate["numeric_comparison_allowed"])
        self.assertEqual(gate["candidate_accuracy"]["QUIET"], 0.0)
        self.assertEqual(gate["candidate_accuracy"]["NVIDIA MPS"], 0.0)
        self.assertEqual(gate["minimum_accuracy"], 0.9)
        sweep = gate["threshold_sweep"]
        sweep_path = ROOT / sweep["path"]
        self.assertEqual(hashlib.sha256(sweep_path.read_bytes()).hexdigest(), sweep["sha256"])
        self.assertTrue(json.loads(sweep_path.read_text())["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
