#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "payload_comparison", ROOT / "analysis" / "summarize_p9_payload_comparison.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PayloadComparisonTest(unittest.TestCase):
    @staticmethod
    def native_gates() -> dict[str, dict]:
        return {
            "orion.json": {
                "kind": "orion-thor-native-positive-control-verification",
                "status": "passed",
                "numeric_comparison_allowed": False,
                "upstream_commit": MODULE.NATIVE_GATES["Orion"]["commit"],
                "driver_launches": 180,
                "reordered_decisions": 13,
            },
            "xsched.json": {
                "kind": "xsched-thor-native-positive-control",
                "numeric_comparison_allowed": False,
                "upstream_commit": MODULE.NATIVE_GATES["XSched"]["commit"],
                "be_completed_requests": 465,
                "hp_completed_requests": 100,
                "be_suspend_transitions": 4,
                "be_resume_transitions": 4,
                "measurement_overlap_ns": 2_783_999_168,
            },
            "bless.json": {
                "kind": "bless-thor-tensorrt-fidelity-gate",
                "status": "passed-functional-gates",
                "numeric_comparison_allowed": False,
                "completed_gates": [
                    "precreated-2-4-6-8-sm-tensorrt-context-replicas",
                    "selected-only-logical-tensorrt-launch-admission",
                    "shadow-replica-advancement",
                    "safe-boundary-restricted-to-unrestricted-switch",
                    "independent-safe-boundary-profile-and-held-out-replay",
                ],
                "remaining_gate": (
                    "drive the frozen safe boundaries with BLESS relative-progress "
                    "squad scheduling on the common workload"
                ),
            },
            "pantheon.json": {
                "kind": "pantheon-thor-native-positive-control",
                "numeric_comparison_allowed": False,
                "upstream_commit": MODULE.NATIVE_GATES["Pantheon"]["commit"],
                "full_exit_job": {"last_block": 1},
                "early_exit_job": {"last_block": 0},
            },
        }

    def test_public_table_has_only_one_proposed_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontier = {
                "proposed_system": "QUIET",
                "rows": [
                    {
                        "system": system,
                        "offered_rps": 250.0,
                        "background_goodput_rps": 249.0,
                        "requests": 100,
                        "misses": misses,
                        "dmr": misses / 100,
                        "post_release_p99_us": 700.0,
                        "arrival_bound_feasible": misses == 0,
                    }
                    for system, misses in (("NVIDIA MIG", 10), ("NVIDIA MPS", 20), ("QUIET", 0))
                ],
            }
            values = {
                "frontier.json": frontier,
                "boer.json": {"system": "BOER", "status": "no-feasible-configuration", "selected": None},
                "parva.json": {"system": "ParvaGPU", "feasible": False, "reason": "layout"},
                "gpulet.json": self.gpulet_numeric(),
            } | self.native_gates()
            for name, value in values.items():
                (root / name).write_text(json.dumps(value))
            (root / "payload-gate.json").write_text(json.dumps({
                "kind": "p9-resnet-layer7-control-mlp-cross-mig-smoke",
                "status": "passed",
                "requests": 100,
                "edge": {
                    "producer_tensor": "Layer7_cov",
                    "consumer_tensor": "features",
                    "shape": [1, 4, 23, 40],
                    "bytes": 14_720,
                },
                "checksum_failures": 0,
                "unique_payload_checksums": 4,
                "unique_policy_output_checksums": 4,
                "token_only": False,
            }))
            result = MODULE.summarize(
                root / "frontier.json",
                root / "boer.json",
                root / "parva.json",
                root / "orion.json",
                root / "xsched.json",
                root / "bless.json",
                root / "pantheon.json",
                root / "gpulet.json",
                250.0,
                root / "payload-gate.json",
            )
        self.assertEqual(
            tuple(row["system"] for row in result["systems"]), MODULE.PUBLIC_SYSTEMS
        )
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(
            tuple(row["system"] for row in result["headline_systems"]),
            MODULE.HEADLINE_SYSTEMS,
        )
        self.assertEqual(
            result["headline_contract"]["gpulet_role"],
            "structural-diagnostic-excluded-from-headline",
        )
        self.assertIn("Pantheon", result["headline_contract"]["functional_only_systems"])
        self.assertEqual(
            [row["system"] for row in result["systems"] if row["system"] == "QUIET"],
            ["QUIET"],
        )
        self.assertEqual(
            [row["system"] for row in result["structural_controls"]],
            ["BOER", "ParvaGPU"],
        )
        self.assertFalse(
            {"BOER", "ParvaGPU"} & {row["system"] for row in result["systems"]}
        )

    def test_accepts_corrected_whisper_stress_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stress = {
                "workload": "whisper-projection",
                "deadline_us": 1620.0,
                "background_offered_rps": 250.0,
                "results": [
                    {
                        "system": system,
                        "pipeline_requests": 1000,
                        "deadline_misses": misses,
                        "pipeline_p99_us": p99,
                        "deadline_mode": "validation-excluded",
                        "background_goodput_rps": 249.9,
                        "gate_scope": "producer",
                    }
                    for system, misses, p99 in (
                        ("NVIDIA MIG", 550, 2225.0),
                        ("NVIDIA MPS", 586, 2243.0),
                        ("QUIET", 0, 1574.0),
                    )
                ],
            }
            values = {
                "frontier.json": stress,
                "boer.json": {"system": "BOER", "status": "no-feasible-configuration"},
                "parva.json": {"system": "ParvaGPU", "feasible": False, "reason": "profile"},
                "gpulet.json": self.gpulet_numeric(),
            } | self.native_gates()
            for name, value in values.items():
                (root / name).write_text(json.dumps(value))
            result = MODULE.summarize(
                root / "frontier.json", root / "boer.json",
                root / "parva.json", root / "orion.json",
                root / "xsched.json", root / "bless.json",
                root / "pantheon.json", root / "gpulet.json", 250.0,
            )
        self.assertEqual(result["workload"]["edge_payload_bytes"], 2_304_000)
        self.assertEqual(result["systems"][-1]["deadline_p99_us"], 1574.0)
        self.assertEqual(result["edge_secondary"]["system"], "Pantheon")

    def test_gpulet_infeasible_diagnostic_is_not_numeric(self) -> None:
        row = MODULE.gpulet_numeric_row(self.gpulet_numeric(), 250.0)
        self.assertEqual(row["status"], "structural-diagnostic")
        self.assertFalse(row["numeric_comparison_allowed"])

    def test_rejects_functional_gate_as_numeric_result(self) -> None:
        gate = self.native_gates()["xsched.json"] | {
            "numeric_comparison_allowed": True
        }
        with self.assertRaisesRegex(ValueError, "cannot be used as a numeric"):
            MODULE.native_gate_row("XSched", gate)

    def test_accepts_native_xsched_same_workload_numeric_evidence(self) -> None:
        value = {
            "kind": "xsched-thor-resnet-control-numeric-smoke-verification",
            "system": "XSched (Thor port)",
            "upstream_commit": MODULE.NATIVE_GATES["XSched"]["commit"],
            "status": "passed-smoke",
            "workload": "resnet10-layer7-cov-to-control-mlp",
            "payload_bytes": 14_720,
            "requests": 100,
            "misses": 100,
            "dmr": 1.0,
            "p99_us": 1357.0,
            "deadline_us": 760.0,
            "background_goodput_rps": 25.0,
            "background_window": {"completed_requests": 3},
            "checksum_failures": 0,
            "unique_payload_checksums": 4,
            "unique_policy_output_checksums": 4,
            "scheduler": {
                "connected_clients": 3,
                "xqueue_clients": 3,
                "be_suspend_transitions": 4,
                "be_resume_transitions": 3,
            },
            "token_only": False,
        }
        row = MODULE.xsched_numeric_row(value, 250.0)
        self.assertEqual(row["system"], "XSched")
        self.assertEqual(row["misses"], 100)
        self.assertEqual(row["background_goodput_rps"], 25.0)

    def test_rejects_different_numeric_deadlines(self) -> None:
        rows = [
            {"system": name, "deadline_us": 770.0}
            for name in MODULE.PUBLIC_SYSTEMS
        ]
        rows[2]["deadline_us"] = 760.0
        with self.assertRaisesRegex(ValueError, "different deadlines"):
            MODULE.common_numeric_deadline(rows)

    @staticmethod
    def gpulet_numeric() -> dict:
        return {
            "kind": "gpulet-thor-resnet-control-numeric-smoke-verification",
            "system": "gpulet (Thor port)",
            "upstream_commit": "3c1c2aad3b33edcef20e549d5093c43af497e6ae",
            "status": "passed-smoke",
            "workload": "resnet10-layer7-cov-to-control-mlp",
            "payload_bytes": 14_720,
            "requests": 100,
            "misses": 100,
            "dmr": 1.0,
            "p99_us": 950.0,
            "deadline_us": 770.0,
            "background_goodput_rps": 249.9,
            "spatial_schedule_feasible": False,
            "selected_action": {
                "producer_quota": 90,
                "background_quota": 10,
                "semantics": "diagnostic-largest-critical-partition",
            },
            "checksum_failures": 0,
            "unique_payload_checksums": 4,
            "unique_policy_output_checksums": 4,
            "token_only": False,
        }


if __name__ == "__main__":
    unittest.main()
