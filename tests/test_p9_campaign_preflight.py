from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p9_campaign_preflight", ROOT / "analysis/preflight_p9_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CampaignPreflightTest(unittest.TestCase):
    def test_accuracy_gate_requires_passed_post_completion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = {}
            for name in (
                "dataset", "reference", "candidate", "reference-output", "candidate-output",
                "reference-pipeline", "candidate-pipeline",
            ):
                path = root / f"{name}.jsonl"
                path.write_text("{}\n")
                evidence[name] = path
            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            gate = {
                "kind": "p9-application-accuracy-gate",
                "status": "passed",
                "numeric_comparison_allowed": True,
                "application_output_trace_required": True,
                "application_output_trace_contract": "passed",
                "application_input_binding_required": True,
                "application_input_binding_contract": "passed",
                "requests": 1,
                "accuracy_tolerance": 0.01,
                "minimum_accuracy": 0.90,
                "reference_accuracy": 1.0,
                "candidate_accuracy": 1.0,
                "accuracy_delta": 0.0,
                "dataset_manifest_path": str(evidence["dataset"]),
                "dataset_manifest_sha256": digest(evidence["dataset"]),
                "reference_trace_path": str(evidence["reference"]),
                "reference_trace_sha256": digest(evidence["reference"]),
                "candidate_trace_path": str(evidence["candidate"]),
                "candidate_trace_sha256": digest(evidence["candidate"]),
            }
            for prefix in ("reference", "candidate"):
                output = evidence[f"{prefix}-output"]
                gate[f"{prefix}_output_trace"] = {
                    "path": str(output), "sha256": digest(output),
                    "capture_boundary": "post-completion", "record_count": 2,
                }
                pipeline = evidence[f"{prefix}-pipeline"]
                gate[f"{prefix}_pipeline_csv"] = {
                    "path": str(pipeline), "sha256": digest(pipeline),
                }
            gate_path = root / "accuracy.json"
            gate_path.write_text(json.dumps(gate) + "\n")
            checked = MODULE.verify_accuracy_gate(gate_path)
            self.assertEqual(checked["requests"], 1)
            gate["status"] = "pending"
            gate_path.write_text(json.dumps(gate) + "\n")
            with self.assertRaisesRegex(ValueError, "passed formal"):
                MODULE.verify_accuracy_gate(gate_path)

    def test_accuracy_gate_must_bind_common_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ("dataset", "reference", "candidate", "reference-output",
                         "candidate-output", "reference-pipeline", "candidate-pipeline"):
                files[name] = root / f"{name}.jsonl"
                files[name].write_text("{}\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            gate = {
                "kind": "p9-application-accuracy-gate", "status": "passed",
                "numeric_comparison_allowed": True,
                "application_output_trace_required": True,
                "application_output_trace_contract": "passed",
                "application_input_binding_required": True,
                "application_input_binding_contract": "passed",
                "workload": "wrong-workload", "requests": 1,
                "accuracy_tolerance": 0.01, "minimum_accuracy": 0.90,
                "reference_accuracy": 1.0, "candidate_accuracy": 1.0,
                "accuracy_delta": 0.0,
                "dataset_manifest_path": str(files["dataset"]),
                "dataset_manifest_sha256": digest(files["dataset"]),
                "reference_trace_path": str(files["reference"]),
                "reference_trace_sha256": digest(files["reference"]),
                "candidate_trace_path": str(files["candidate"]),
                "candidate_trace_sha256": digest(files["candidate"]),
            }
            for prefix in ("reference", "candidate"):
                gate[f"{prefix}_output_trace"] = {
                    "path": str(files[f"{prefix}-output"]),
                    "sha256": digest(files[f"{prefix}-output"]),
                    "capture_boundary": "post-completion", "record_count": 2,
                }
                gate[f"{prefix}_pipeline_csv"] = {
                    "path": str(files[f"{prefix}-pipeline"]),
                    "sha256": digest(files[f"{prefix}-pipeline"]),
                }
            gate_path = root / "accuracy.json"
            gate_path.write_text(json.dumps(gate) + "\n")
            common = {
                "workload_id": "resnet-detection-head", "request_count": 1,
                "dataset_manifest": {
                    "path": str(files["dataset"].resolve()),
                    "sha256": digest(files["dataset"]),
                },
            }
            with self.assertRaisesRegex(ValueError, "workload differs"):
                MODULE.verify_accuracy_gate(gate_path, common_workload=common)
            gate["workload"] = common["workload_id"]
            gate["requests"] = 2
            gate_path.write_text(json.dumps(gate) + "\n")
            with self.assertRaisesRegex(ValueError, "request count differs"):
                MODULE.verify_accuracy_gate(gate_path, common_workload=common)

    def test_common_contract_rechecks_referenced_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrival = root / "arrival.jsonl"
            dataset = root / "dataset.jsonl"
            arrival.write_text(json.dumps({
                "schema_version": 1, "iteration": 10, "request_id": "r0",
                "arrival_sequence": 0, "input_sha256": "a" * 64,
                "expected_label": "cat",
            }) + "\n")
            dataset.write_text(json.dumps({
                "schema_version": 1, "sample_id": "s0",
                "input_sha256": "a" * 64, "expected_label": "cat",
            }) + "\n")
            contract = root / "contract.json"
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            contract.write_text(json.dumps({
                "schema_version": 1, "workload_id": "resnet-detection-head",
                "topology": "fixed-2g+1g", "placement": "1g->2g",
                "input_tensor": "Layer6_relu_Y", "payload_bytes": 1884160,
                "request_count": 1, "arrival_trace_path": str(arrival),
                "arrival_trace_sha256": digest(arrival),
                "dataset_manifest_path": str(dataset),
                "dataset_manifest_sha256": digest(dataset),
            }) + "\n")
            result = MODULE.check_common_contract(contract)
            self.assertEqual(result["request_count"], 1)
            arrival.write_text(arrival.read_text().replace('"cat"', '"dog"'))
            with self.assertRaisesRegex(ValueError, "arrival trace sha256 differs"):
                MODULE.check_common_contract(contract)

    def test_preflight_never_promotes_missing_external_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = MODULE.preflight(
                common=root / "missing-common.json",
                thermal=root / "missing-thermal.json",
                deadline=None,
                accuracy=None,
                check_gpu=False,
            )
            self.assertFalse(result["formal_ready"])
            self.assertTrue(any("common workload:" in item for item in result["missing"]))
            self.assertTrue(any("application accuracy" in item for item in result["missing"]))
            self.assertTrue(any("deadline lock" in item for item in result["missing"]))

    def test_mig_env_requires_both_instances_and_pipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipe = root / "pipe"
            pipe.mkdir()
            env = root / "mig.env"
            env.write_text(
                "JDG_MIG_SMALL_UUID=MIG-small\n"
                "JDG_MIG_BIG_UUID=MIG-big\n"
                f"JDG_MPS_PIPE_DIRECTORY={pipe}\n"
            )
            value = MODULE.check_mig_env(env)
            self.assertEqual(value["small_uuid"], "MIG-small")
            self.assertEqual(value["big_uuid"], "MIG-big")
            (root / "missing.env").write_text("JDG_MIG_SMALL_UUID=MIG-small\n")
            with self.assertRaisesRegex(ValueError, "JDG_MIG_BIG_UUID"):
                MODULE.check_mig_env(root / "missing.env")


if __name__ == "__main__":
    unittest.main()
