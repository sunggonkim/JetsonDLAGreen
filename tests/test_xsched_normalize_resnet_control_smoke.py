import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from baselines.xsched.normalize_resnet_control_smoke import normalize


class XSchedNormalizeTest(unittest.TestCase):
    def test_normalizes_verified_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            lock.write_text(json.dumps({"contract": {"workload": "resnet-control"}}) + "\n")
            value = {
                "kind": "xsched-thor-resnet-control-numeric-smoke-verification",
                "status": "passed-smoke", "workload": "resnet10-layer7-cov-to-control-mlp",
                "deadline_us": 770.0, "requests": 10, "misses": 2, "p99_us": 900.0,
                "deadline_lock": {"sha256": hashlib.sha256(lock.read_bytes()).hexdigest()},
                "checksum_failures": 0, "background_goodput_rps": 25.0,
                "scheduler": {"be_suspend_transitions": 4},
            }
            verification = root / "verification.json"
            verification.write_text(json.dumps(value) + "\n")
            result = normalize(verification, lock, 4.0)
        self.assertEqual(result["results"][0]["system"], "XSched")
        self.assertEqual(result["results"][0]["deadline_misses"], 2)
        self.assertTrue(result["results"][0]["correctness_validated"])

    def test_preserves_post_completion_application_trace_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            lock.write_text(json.dumps({"contract": {"workload": "resnet-control"}}) + "\n")
            trace = root / "outputs.bin"
            trace.write_bytes(b"JDGOUT1\0")
            value = {
                "kind": "xsched-thor-resnet-control-numeric-smoke-verification",
                "status": "passed-smoke", "workload": "resnet10-layer7-cov-to-control-mlp",
                "deadline_us": 770.0, "requests": 10, "misses": 0, "p99_us": 700.0,
                "deadline_lock": {"sha256": hashlib.sha256(lock.read_bytes()).hexdigest()},
                "checksum_failures": 0, "background_goodput_rps": 25.0,
                "scheduler": {"be_suspend_transitions": 4},
                "application_output_trace": {
                    "path": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    "capture_boundary": "post-completion",
                },
            }
            verification = root / "verification.json"
            verification.write_text(json.dumps(value) + "\n")
            result = normalize(verification, lock, 4.0)
        self.assertEqual(
            result["application_output_trace"]["capture_boundary"], "post-completion"
        )
        self.assertEqual(
            result["results"][0]["sota_verification"]["application_output_trace"]["sha256"],
            value["application_output_trace"]["sha256"],
        )

    def test_rejects_lock_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"; lock.write_text("{}\n")
            verification = root / "verification.json"
            verification.write_text(json.dumps({
                "kind": "xsched-thor-resnet-control-numeric-smoke-verification",
                "status": "passed-smoke", "workload": "resnet10-layer7-cov-to-control-mlp",
                "deadline_lock": {"sha256": "0" * 64},
            }))
            with self.assertRaisesRegex(ValueError, "provenance"):
                normalize(verification, lock, 4.0)


if __name__ == "__main__":
    unittest.main()
