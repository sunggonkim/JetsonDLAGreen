import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_p9_goal_completion", ROOT / "analysis/audit_p9_goal_completion.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P9GoalCompletionAuditTest(unittest.TestCase):
    def test_current_evidence_is_complete(self) -> None:
        required = [
            ROOT / "results/p9-resnet-layer7-control-mlp-100r-traced-v2-20260809T1420Z",
            ROOT / "results/p9-common-sota-whisper-current-nonthermal-formal-aggregate-6x1100-20260810/summary.json",
            ROOT / "paper/eurosys27/p9-main.pdf",
        ]
        if not all(path.exists() for path in required):
            self.skipTest("current P9 evidence is absent")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            self.assertEqual(MODULE.main(["--output", str(output)]), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(result["verified_requirements"], 20)
        self.assertEqual(result["deferred_requirements"], 0)
        self.assertTrue(result["objective_complete"])
        self.assertEqual(result["status"], "complete-current-thermal-formal-evidence-verified")
        whisper = next(row for row in result["requirements"] if row["id"] == "real-whisper-dependent-payload")
        self.assertEqual(whisper["evidence"]["application_accuracy_gate"]["task"], "asr")
        self.assertEqual(whisper["evidence"]["application_accuracy_gate"]["candidate_accuracy"], 0.9)
        self.assertTrue(whisper["evidence"]["application_accuracy_gate"]["output_trace_sha256_equal"])
        vision = next(row for row in result["requirements"] if row["id"] == "real-vision-application-accuracy-gate")
        vision_gate = vision["evidence"]["application_accuracy_gate"]
        self.assertEqual(vision_gate["task"], "classification")
        self.assertEqual(vision_gate["requests"], 90)
        self.assertEqual(vision_gate["reference_accuracy"], 0.8333333333333334)
        self.assertEqual(vision_gate["candidate_accuracy"], 0.8333333333333334)
        self.assertEqual(vision_gate["accuracy_delta"], 0.0)
        self.assertEqual(vision_gate["minimum_accuracy"], 0.8)
        self.assertEqual(vision_gate["application_input_binding_contract"], "passed")
        self.assertEqual(vision_gate["application_output_trace_contract"], "passed")
        followups = {row["id"]: row for row in result["deferred_followups"]}
        self.assertEqual(followups["thermal-normalized-formal-campaign"]["status"], "verified")
        self.assertEqual(followups["orion-differential-fidelity-gate"]["status"], "not-required")
        ports = next(row for row in result["requirements"] if row["id"] == "pinned-upstream-sota-ports")
        self.assertEqual(ports["status"], "verified")
        self.assertEqual(set(ports["evidence"]["ports"]), {"BOER", "ParvaGPU", "Orion", "XSched", "Pantheon"})
        sota = next(row for row in result["requirements"] if row["id"] == "nvidia-mig-mps-and-executable-sota")
        self.assertEqual(sota["status"], "partial")
        self.assertEqual(sota["evidence"]["numeric_frontier"], ["NVIDIA MPS", "QUIET"])
        repeats = sota["evidence"]["production_wall_repeats"]
        self.assertEqual(repeats["sequence_count"], 3)
        self.assertEqual(repeats["systems"], ["NVIDIA MPS", "XSched", "QUIET"])
        self.assertFalse(repeats["formal"])
        self.assertFalse(repeats["ranking_allowed"])
        formal = next(row for row in result["requirements"] if row["id"] == "order-balanced-nonthermal-formal-replay")
        self.assertEqual(formal["evidence"]["whisper"]["sequence_provenance"]["system_evidence"], 36)
        self.assertEqual(formal["evidence"]["whisper"]["sequence_provenance"]["orion_scheduler_verifications"], 6)
        causal = next(row for row in result["requirements"] if row["id"] == "real-learned-dependent-dag-causal-repeats")
        self.assertEqual(causal["status"], "verified")
        self.assertEqual(causal["evidence"]["QUIET"]["status"], "verified")
        self.assertEqual(causal["evidence"]["NVIDIA MPS"]["status"], "verified")
        frontier = next(row for row in result["requirements"] if row["id"] == "real-learned-dependent-dag-frontier-smoke")
        self.assertFalse(frontier["evidence"]["formal_cp95_qualified"])
        transport = next(row for row in result["requirements"] if row["id"] == "real-learned-transport-motivation")
        self.assertEqual(transport["evidence"]["transports"], ["registered", "pinned", "pageable"])
        self.assertFalse(transport["evidence"]["formal"])
        learned = next(row for row in result["requirements"] if row["id"] == "real-learned-dependent-dag-smoke")
        self.assertIsNotNone(learned["evidence"]["output_trace"])
        placement = next(row for row in result["requirements"] if row["id"] == "multi-candidate-placement-characterization")
        self.assertEqual(placement["evidence"]["candidate_count"], 2)
        self.assertFalse(placement["evidence"]["formal"])
        search = next(row for row in result["requirements"] if row["id"] == "multi-candidate-placement-quota-search-contract")
        self.assertEqual(search["evidence"]["candidate_count"], 6)
        self.assertTrue(search["evidence"]["formal_contract"])
        session_frontier = next(row for row in result["requirements"] if row["id"] == "multi-candidate-placement-quota-session-frontier")
        self.assertEqual(session_frontier["evidence"]["points"], 6)
        self.assertEqual(session_frontier["evidence"]["sessions"], 18)
        self.assertFalse(session_frontier["evidence"]["formal_cp95_qualified"])

    def test_formal_rejects_internal_or_missing_system(self) -> None:
        summary = {
            "kind": "p9-common-sota-williams-aggregate",
            "scope": "order-balanced-raw-replayed-nonthermal-campaign",
            "proposed_system": "QUIET",
            "systems": {},
        }
        with self.assertRaisesRegex(ValueError, "system set differs"):
            MODULE.audit_formal(summary, "resnet")


if __name__ == "__main__":
    unittest.main()
