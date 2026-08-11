import importlib.util
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_active", ROOT / "analysis" / "summarize_p9_active_comparator_smoke.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def summary(system: str, *, lock: str = "a" * 64, requests: int = 20) -> dict:
    return {
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-control",
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "deadline_us": 773.730452,
        "deadline_lock": {"sha256": lock, "path": "/tmp/lock.json"},
            "claim_status": "exploratory-contract-smoke",
            "results": [{
                "system": system,
            "pipeline_requests": requests,
            "deadline_misses": 0,
            "wall_pipeline_p99_us": 650.0,
                "correctness_validated": True,
                "checksum_mode": "inline",
                "latency_contract": "production-wall-arrival-to-completion",
                "deadline_mode": "wall",
                "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
                "correctness_validation_placement": "post-completion",
                "background_goodput_rps": 100.0,
            "unique_payload_checksums": 4,
            "unique_policy_output_checksums": 4,
        }],
    }


class ActiveComparatorSmokeTest(unittest.TestCase):
    def write(self, directory: pathlib.Path, value: dict, *, verification: bool = False) -> pathlib.Path:
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / "lock.json"
        lock.write_text("{}\n" if value["deadline_lock"].get("sha256") == "a" * 64 else '{"different":true}\n')
        value["deadline_lock"]["path"] = str(lock)
        value["deadline_lock"]["sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
        if verification:
            checked = directory / "verification.json"
            checked.write_text("{}\n")
            value["results"][0]["sota_verification"] = {
                "path": str(checked),
                "sha256": hashlib.sha256(checked.read_bytes()).hexdigest(),
            }
        (directory / "summary.json").write_text(json.dumps(value))
        return directory

    def test_requires_one_common_contract_and_keeps_ranking_exploratory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            result = MODULE.summarize(
                self.write(root / "quiet", summary("QUIET")),
                self.write(root / "mps", summary("NVIDIA MPS")),
                self.write(root / "xsched", summary("XSched"), verification=True),
            )
        self.assertEqual(result["system_order"], ["QUIET", "NVIDIA MPS", "XSched"])
        self.assertFalse(result["ranking_allowed"])
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(result["presentation"]["proposed_system"], "QUIET")
        self.assertEqual(
            result["presentation"]["labels"]["XSched"], "XSched (Thor port)"
        )
        self.assertEqual(result["presentation"]["headline_order"][0], "QUIET")
        self.assertFalse(result["presentation"]["numeric_comparison_allowed"])
        self.assertFalse(result["contract"]["request_trace_bound"])
        self.assertIn("request/output trace evidence", result["ranking_reason"])

    def test_rejects_lock_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            with self.assertRaisesRegex(ValueError, "deadline lock SHA differs"):
                MODULE.summarize(
                    self.write(root / "quiet", summary("QUIET")),
                    self.write(root / "mps", summary("NVIDIA MPS", lock="b" * 64)),
                    self.write(root / "xsched", summary("XSched"), verification=True),
                )

    def test_rejects_diagnostic_arm(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            diagnostic = summary("QUIET")
            diagnostic["claim_status"] = "diagnostic-only-plan-violation"
            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                MODULE.summarize(
                    self.write(root / "quiet", diagnostic),
                    self.write(root / "mps", summary("NVIDIA MPS")),
                    self.write(root / "xsched", summary("XSched"), verification=True),
                )

    def test_binds_request_and_post_completion_output_traces(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            directories = {}
            for name, system in (("quiet", "QUIET"), ("mps", "NVIDIA MPS"), ("xsched", "XSched")):
                value = summary(system)
                value["results"][0].update({
                    "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
                    "correctness_validation_placement": "post-completion",
                    "payload_bytes": 14720,
                    "consumer_input_tensor": "features",
                    "consumer_engine_mode": "generated-control-policy",
                    "producer_sms": 8,
                    "consumer_sms": 12,
                })
                arm_dir = root / name
                arm_dir.mkdir(parents=True)
                request = arm_dir / "pipeline.csv"
                request.write_text("request,wall_end_to_end_us,deadline_miss\n0,1,0\n")
                output = arm_dir / "outputs.bin"
                output.write_bytes(b"JDGOUT1-test")
                value["results"][0]["request_trace"] = {
                    "path": str(request), "sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
                }
                value["results"][0]["application_output_trace"] = {
                    "path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "capture_boundary": "post-completion",
                }
                directories[name] = self.write(arm_dir, value, verification=system == "XSched")
            result = MODULE.summarize(directories["quiet"], directories["mps"], directories["xsched"])
        self.assertTrue(result["contract"]["post_completion_output_trace_bound"])
        self.assertTrue(result["contract"]["request_trace_bound"])
        self.assertEqual(result["contract"]["input_contract"]["payload_bytes"], 14720)

    def test_rejects_tampered_trace_or_workload_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            directories = {}
            for name, system in (("quiet", "QUIET"), ("mps", "NVIDIA MPS"), ("xsched", "XSched")):
                value = summary(system)
                value["results"][0].update({
                    "payload_bytes": 14720,
                    "consumer_input_tensor": "features",
                    "consumer_engine_mode": "generated-control-policy",
                    "producer_sms": 8,
                    "consumer_sms": 12,
                })
                arm_dir = root / name
                arm_dir.mkdir(parents=True)
                trace = arm_dir / "trace.csv"
                trace.write_text("trace\n")
                value["results"][0]["request_trace"] = {
                    "path": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                }
                directories[name] = self.write(arm_dir, value, verification=system == "XSched")
            (root / "mps" / "trace.csv").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "request trace: SHA-256"):
                MODULE.summarize(directories["quiet"], directories["mps"], directories["xsched"])

    def test_rejects_mismatched_payload_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            directories = []
            for name, system in (("quiet", "QUIET"), ("mps", "NVIDIA MPS"), ("xsched", "XSched")):
                value = summary(system)
                value["results"][0].update({
                    "payload_bytes": 14720 if system != "NVIDIA MPS" else 1884160,
                    "consumer_input_tensor": "features",
                    "consumer_engine_mode": "generated-control-policy",
                    "producer_sms": 8,
                    "consumer_sms": 12,
                })
                directories.append(self.write(root / name, value, verification=system == "XSched"))
            with self.assertRaisesRegex(ValueError, "workload engine or MIG capacity"):
                MODULE.summarize(*directories)

    def test_rejects_mismatched_common_workload_contract(self) -> None:
        contract = {
            "schema_version": 1,
            "workload_id": "resnet-detection-head",
            "topology": "fixed-2g+1g",
            "placement": "fixed-1g-producer-2g-consumer",
            "input_tensor": "Layer6_relu_Y",
            "payload_bytes": 1_884_160,
            "arrival_trace_path": "/tmp/arrival.jsonl",
            "arrival_trace_sha256": "a" * 64,
            "dataset_manifest_path": "/tmp/dataset.jsonl",
            "dataset_manifest_sha256": "b" * 64,
            "contract_path": "/tmp/common-workload.json",
            "contract_sha256": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            directories = []
            for name, system in (("quiet", "QUIET"), ("mps", "NVIDIA MPS"), ("xsched", "XSched")):
                value = summary(system)
                value["common_workload"] = dict(contract)
                if system == "NVIDIA MPS":
                    value["common_workload"]["arrival_trace_sha256"] = "d" * 64
                directories.append(self.write(root / name, value, verification=system == "XSched"))
            with self.assertRaisesRegex(ValueError, "common workload contract differs"):
                MODULE.summarize(*directories)


if __name__ == "__main__":
    unittest.main()
