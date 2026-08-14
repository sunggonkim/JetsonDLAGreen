from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bind_p9_active_accuracy", ROOT / "analysis/bind_p9_active_accuracy.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BindP9ActiveAccuracyTest(unittest.TestCase):
    def test_extracts_dependent_row_and_rechecks_declared_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "pipeline.csv"
            output = root / "output.bin"
            evidence = root / "summary.json"
            pipeline.write_text("request,wall_end_to_end_us,deadline_miss\n0,1,0\n")
            output.write_bytes(b"output")
            evidence.write_text(json.dumps({
                "results": [{
                    "system": "QUIET",
                    "request_trace": {
                        "path": str(pipeline),
                        "sha256": hashlib.sha256(pipeline.read_bytes()).hexdigest(),
                    },
                    "application_output_trace": {
                        "path": str(output),
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    },
                }],
            }) + "\n")
            result = MODULE.candidate_paths(evidence)
            self.assertEqual(result["system"], "QUIET")
            self.assertEqual(result["candidate_pipeline_csv"], str(pipeline.resolve()))
            self.assertEqual(result["candidate_output_trace"], str(output.resolve()))

    def test_extracts_mig_capacity_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "pipeline.csv"
            output = root / "output.bin"
            evidence = root / "summary.json"
            pipeline.write_text("request,wall_end_to_end_us,deadline_miss\n0,1,0\n")
            output.write_bytes(b"output")
            evidence.write_text(json.dumps({
                "results": [{
                    "system": "NVIDIA MIG",
                    "request_trace": {"path": str(pipeline)},
                    "application_output_trace": {"path": str(output)},
                    "best_effort_admitted": False,
                }],
            }) + "\n")
            result = MODULE.candidate_paths(evidence)
            self.assertEqual(result["system"], "NVIDIA MIG")
            self.assertEqual(result["candidate_pipeline_csv"], str(pipeline.resolve()))

    def test_extracts_xsched_pipeline_next_to_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "pipeline.csv"
            output = root / "application-outputs.bin"
            evidence = root / "verification.json"
            pipeline.write_text("request,wall_end_to_end_us,deadline_miss\n0,1,0\n")
            output.write_bytes(b"output")
            evidence.write_text(json.dumps({
                "system": "XSched (Thor port)",
                "application_output_trace": {"path": str(output)},
            }) + "\n")
            result = MODULE.candidate_paths(evidence)
            self.assertEqual(result["system"], "XSched")
            self.assertEqual(result["candidate_pipeline_csv"], str(pipeline.resolve()))

    def test_rejects_changed_declared_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "pipeline.csv"
            output = root / "output.bin"
            evidence = root / "summary.json"
            pipeline.write_text("request,wall_end_to_end_us,deadline_miss\n0,1,0\n")
            output.write_bytes(b"output")
            evidence.write_text(json.dumps({
                "results": [{
                    "system": "NVIDIA MPS",
                    "request_trace": {"path": str(pipeline)},
                    "application_output_trace": {"path": str(output), "sha256": "0" * 64},
                }],
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                MODULE.candidate_paths(evidence)


if __name__ == "__main__":
    unittest.main()
