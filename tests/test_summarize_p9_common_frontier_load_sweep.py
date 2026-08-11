import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summarize_p9_common_frontier_load_sweep import summarize


def fixture(system: str, load: float) -> dict:
    period = 1000.0 / load
    return {
        "kind": "p9-dependent-small-stress-smoke", "workload": "resnet-control",
        "latency_contract": "production-wall-arrival-to-completion", "deadline_mode": "wall",
        "checksum_mode": "inline", "deadline_us": 773.730452, "iterations": 100,
        "background_period_ms": period, "background_offered_rps": load,
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "deadline_lock": {"sha256": "a" * 64},
        "results": [{"system": system, "correctness_validated": True,
                     "pipeline_requests": 100, "deadline_misses": 0,
                     "wall_pipeline_p99_us": 700.0, "background_goodput_rps": load,
                     "checksum_failures": 0, "unique_payload_checksums": 4,
                     "unique_policy_output_checksums": 4}],
    }


class CommonLoadFrontierTest(unittest.TestCase):
    def test_output_trace_requirement_cannot_be_used_without_accuracy_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.json"
            path.write_text(json.dumps(fixture("QUIET", 125.0)) + "\n")
            with self.assertRaisesRegex(ValueError, "requires application accuracy"):
                summarize([path], require_output_traces=True)

    def test_balanced_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for load in (125.0, 250.0):
                for system in ("QUIET", "NVIDIA MPS", "XSched"):
                    path = root / f"{system}-{load}.json"
                    path.write_text(json.dumps(fixture(system, load)) + "\n")
                    paths.append(path)
            result = summarize(paths)
        self.assertEqual(result["offered_loads_rps"], [125.0, 250.0])
        point = result["frontier"]["QUIET"]["points"][0]
        self.assertGreater(point["cp95_upper_dmr"], 0.0005)
        self.assertFalse(point["cp95_slo_qualified"])
        self.assertFalse(result["frontier"]["QUIET"]["numeric_comparison_allowed"])
        self.assertIsNone(result["common_workload"])
        self.assertEqual(result["dmr_target"], 0.0005)
        self.assertEqual(result["numeric_frontier_systems"], ["NVIDIA MPS", "QUIET"])
        self.assertEqual(result["exploratory_systems"], ["XSched"])
        self.assertFalse(result["ranking_allowed"])
        self.assertEqual(
            result["production_wall_definition"],
            "arrival-to-consumer-completion-excludes-correctness-validation",
        )

    def test_rejects_missing_system(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS"):
                path = root / f"{system}.json"
                path.write_text(json.dumps(fixture(system, 125.0)) + "\n")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "does not contain all systems"):
                summarize(paths)

    def test_rejects_same_numeric_deadline_with_different_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for load in (125.0, 250.0, 375.0):
                for system in ("QUIET", "NVIDIA MPS", "XSched"):
                    value = fixture(system, load)
                    if system == "XSched":
                        value["deadline_lock"] = {"sha256": "b" * 64}
                    path = root / f"{system}-{int(load)}.json"
                    path.write_text(json.dumps(value) + "\n")
                    paths.append(path)
            with self.assertRaisesRegex(ValueError, "common workload/deadline contract"):
                summarize(paths)

    def test_accepts_complete_static_full_gate_sweep_as_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for load in (125.0, 250.0):
                for system in ("QUIET", "NVIDIA MPS", "XSched", "Static full gating"):
                    path = root / f"{system}-{load}.json"
                    path.write_text(json.dumps(fixture(system, load)) + "\n")
                    paths.append(path)
            result = summarize(paths)
        self.assertIn("Static full gating", result["systems"])
        self.assertFalse(result["frontier"]["Static full gating"]["numeric_comparison_allowed"])

    def test_formal_accuracy_mode_requires_a_byte_bound_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "accuracy-gate.json"
            reference_pipeline = root / "reference.csv"
            candidate_pipeline = root / "candidate.csv"
            arrival_trace = root / "arrival.jsonl"
            dataset_manifest = root / "dataset.jsonl"
            workload_contract = root / "workload.json"
            reference_pipeline.write_text("request,input_sha256,wall_end_to_end_us,deadline_miss\n0," + "a" * 64 + ",1,0\n")
            candidate_pipeline.write_text(reference_pipeline.read_text())
            arrival_trace.write_text("{\"request_id\":0}\n")
            dataset_manifest.write_text("{\"input_sha256\":\"" + "a" * 64 + "\"}\n")
            workload_contract.write_text("{}\n")
            common_workload = {
                "schema_version": 1,
                "workload_id": "resnet-control",
                "topology": "fixed-1g-producer-2g-consumer",
                "placement": "1g->2g",
                "input_tensor": "Layer7_cov",
                "payload_bytes": 14720,
                "arrival_trace_path": str(arrival_trace),
                "arrival_trace_sha256": hashlib.sha256(arrival_trace.read_bytes()).hexdigest(),
                "dataset_manifest_path": str(dataset_manifest),
                "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
                "contract_path": str(workload_contract),
                "contract_sha256": hashlib.sha256(workload_contract.read_bytes()).hexdigest(),
            }
            gate.write_text(json.dumps({
                "kind": "p9-application-accuracy-gate",
                "status": "passed",
                "numeric_comparison_allowed": True,
                "workload": "resnet-control",
                "minimum_accuracy": 0.90,
                "reference_accuracy": 1.0,
                "candidate_accuracy": 1.0,
                "dataset_manifest_path": str(dataset_manifest),
                "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
                "application_input_binding_required": True,
                "application_input_binding_contract": "passed",
                "reference_pipeline_csv": {"path": str(reference_pipeline), "sha256": hashlib.sha256(reference_pipeline.read_bytes()).hexdigest()},
                "candidate_pipeline_csv": {"path": str(candidate_pipeline), "sha256": hashlib.sha256(candidate_pipeline.read_bytes()).hexdigest()},
            }) + "\n")
            gate_sha = hashlib.sha256(gate.read_bytes()).hexdigest()
            paths = []
            for system in ("QUIET", "NVIDIA MPS", "XSched"):
                value = fixture(system, 125.0)
                value["common_workload"] = common_workload
                value["application_accuracy_gate"] = {
                    "path": str(gate), "sha256": gate_sha,
                }
                path = root / f"{system}.json"
                path.write_text(json.dumps(value) + "\n")
                paths.append(path)
            result = summarize(paths, require_application_accuracy=True)
        self.assertTrue(result["application_accuracy_required"])

    def test_formal_accuracy_mode_rejects_exploratory_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS", "XSched"):
                path = root / f"{system}.json"
                path.write_text(json.dumps(fixture(system, 125.0)) + "\n")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "application accuracy gate|common workload"):
                summarize(paths, require_application_accuracy=True)

    def test_diagnostic_plan_violation_never_enters_frontier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            value = fixture("QUIET", 125.0)
            value["claim_status"] = "diagnostic-only-plan-violation"
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                summarize([path])


if __name__ == "__main__":
    unittest.main()
