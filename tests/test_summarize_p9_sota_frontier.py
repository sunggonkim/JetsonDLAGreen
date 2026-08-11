import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sota_frontier", ROOT / "analysis" / "summarize_p9_sota_frontier.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def aggregate(rate: float, quiet_goodput: float, quiet_qualified: bool = True) -> dict:
    def row(allowed: bool, goodput: float, qualified: bool) -> dict:
        return {
            "numeric_comparison_allowed": allowed,
            "slo_confidence_qualified": qualified,
            "background_goodput_rps_mean": goodput,
            "dmr_cp95_upper": 0.0004 if qualified else 0.01,
            "pooled_p99_us": 700.0,
        }
    return {
        "kind": "p9-common-sota-williams-aggregate",
        "proposed_system": "QUIET",
        "deadline_mode": "wall",
        "background_offered_rps": rate,
        "workload": "resnet-control",
        "deadline_us": 760.0,
        "systems": {
            "NVIDIA MIG": row(False, 0.0, False),
            "NVIDIA MPS": row(True, rate * 0.8, rate <= 200),
            "Orion": row(True, rate * 0.7, rate <= 300),
            "QUIET": row(True, quiet_goodput, quiet_qualified),
        },
    }


class SotaFrontierTest(unittest.TestCase):
    def test_selects_maximum_slo_qualified_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "accuracy-gate.json"
            reference_pipeline = root / "reference.csv"
            candidate_pipeline = root / "candidate.csv"
            reference_pipeline.write_text("request,input_sha256,wall_end_to_end_us,deadline_miss\n0," + "a" * 64 + ",1,0\n")
            candidate_pipeline.write_text(reference_pipeline.read_text())
            gate.write_text(json.dumps({
                "kind": "p9-application-accuracy-gate",
                "status": "passed",
                "numeric_comparison_allowed": True,
                "minimum_accuracy": 0.90,
                "reference_accuracy": 1.0,
                "candidate_accuracy": 1.0,
                "application_input_binding_required": True,
                "application_input_binding_contract": "passed",
                "reference_pipeline_csv": {"path": str(reference_pipeline), "sha256": hashlib.sha256(reference_pipeline.read_bytes()).hexdigest()},
                "candidate_pipeline_csv": {"path": str(candidate_pipeline), "sha256": hashlib.sha256(candidate_pipeline.read_bytes()).hexdigest()},
            }) + "\n")
            gate_record = {"path": str(gate), "sha256": hashlib.sha256(gate.read_bytes()).hexdigest()}
            paths = []
            for index, rate in enumerate((100.0, 200.0, 300.0)):
                path = root / f"aggregate-{index}.json"
                value = aggregate(rate, rate * 0.9)
                value["application_accuracy_gate"] = gate_record
                path.write_text(json.dumps(value) + "\n")
                paths.append(path)
            result = MODULE.summarize(paths, accuracy_gates={"NVIDIA MPS": gate})
        self.assertEqual(
            result["systems"]["QUIET"]["max_slo_qualified_offered_rps"],
            300.0,
        )
        self.assertIsNone(result["systems"]["NVIDIA MIG"]["max_slo_qualified_offered_rps"])
        self.assertEqual(result["offered_loads_rps"], [100.0, 200.0, 300.0])
        self.assertEqual(result["numeric_frontier_systems"], ["NVIDIA MPS", "QUIET"])
        self.assertEqual(
            result["exploratory_systems"], ["NVIDIA MIG", "Orion"]
        )
        self.assertTrue(result["ranking_allowed"])
        self.assertIn("NVIDIA MPS", result["application_accuracy_gates"])

    def test_rejects_mixed_deadline_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps(aggregate(100.0, 90.0)) + "\n")
            value = aggregate(200.0, 180.0)
            value["deadline_mode"] = "validation-excluded"
            second.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "production-wall"):
                MODULE.summarize([first, second])

    def test_manifest_cannot_be_overridden_by_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            value = aggregate(100.0, 90.0)
            value["systems"]["Orion"]["numeric_comparison_allowed"] = True
            value["systems"]["Orion"]["slo_confidence_qualified"] = True
            path.write_text(json.dumps(value) + "\n")
            result = MODULE.summarize([path])
        orion = result["systems"]["Orion"]
        self.assertFalse(orion["numeric_comparison_allowed"])
        self.assertIsNone(orion["max_slo_qualified_offered_rps"])

    def test_quiet_accuracy_gate_is_required_for_numeric_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            path.write_text(json.dumps(aggregate(100.0, 90.0)) + "\n")
            result = MODULE.summarize([path])
        quiet = result["systems"]["QUIET"]
        self.assertFalse(quiet["numeric_comparison_allowed"])
        self.assertIsNone(quiet["max_slo_qualified_offered_rps"])
        self.assertEqual(quiet["comparison_status"], "application-accuracy-gate-pending")
        mps = result["systems"]["NVIDIA MPS"]
        self.assertFalse(mps["numeric_comparison_allowed"])
        self.assertIsNone(mps["max_slo_qualified_offered_rps"])
        self.assertEqual(mps["comparison_status"], "application-accuracy-gate-pending")
        self.assertFalse(result["ranking_allowed"])

    def test_strict_output_trace_mode_rejects_prediction_only_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "accuracy-gate.json"
            gate.write_text(json.dumps({
                "kind": "p9-application-accuracy-gate",
                "status": "passed",
                "numeric_comparison_allowed": True,
                "minimum_accuracy": 0.90,
                "reference_accuracy": 1.0,
                "candidate_accuracy": 1.0,
            }) + "\n")
            gate_record = {
                "path": str(gate),
                "sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
            }
            path = root / "aggregate.json"
            value = aggregate(100.0, 90.0)
            value["application_accuracy_gate"] = gate_record
            path.write_text(json.dumps(value) + "\n")
            result = MODULE.summarize([path], require_output_traces=True)
        self.assertFalse(result["systems"]["QUIET"]["numeric_comparison_allowed"])

    def test_diagnostic_plan_violation_never_enters_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            value = aggregate(100.0, 90.0)
            value["claim_status"] = "diagnostic-only-plan-violation"
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                MODULE.summarize([path])


if __name__ == "__main__":
    unittest.main()
