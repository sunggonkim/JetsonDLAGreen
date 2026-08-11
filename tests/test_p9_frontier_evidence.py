import json
import tempfile
import unittest
from pathlib import Path

from analysis.p9_frontier_evidence import validate_correctness


def base_value() -> dict:
    return {
        "workload": "resnet-control",
        "deadline_us": 773.730452,
        "checksum_mode": "inline",
    }


def base_row() -> dict:
    return {
        "system": "XSched",
        "correctness_validated": True,
        "pipeline_requests": 2,
        "deadline_misses": 1,
        "wall_pipeline_p99_us": 900.0,
    }


class FrontierEvidenceTest(unittest.TestCase):
    def test_sota_verification_is_replayed_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verification = root / "verification.json"
            verification.write_text(json.dumps({
                "kind": "xsched-test-verification",
                "status": "passed-smoke",
                "token_only": False,
                "checksum_failures": 0,
                "unique_payload_checksums": 2,
                "unique_policy_output_checksums": 2,
                "workload": "resnet10-layer7-cov-to-control-mlp",
                "deadline_us": 773.730452,
                "requests": 2,
                "misses": 1,
                "p99_us": 900.0,
            }) + "\n", encoding="utf-8")
            import hashlib
            evidence = {
                "path": str(verification),
                "sha256": hashlib.sha256(verification.read_bytes()).hexdigest(),
            }
            row = base_row() | {"sota_verification": evidence}
            result = validate_correctness(base_value(), row, root / "summary.json")
        self.assertEqual(result["source"], "sota_verification")
        self.assertEqual(result["unique_payload_checksums"], 2)

    def test_missing_sota_checksum_provenance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "verification provenance"):
            validate_correctness(base_value(), base_row(), Path("summary.json"))


if __name__ == "__main__":
    unittest.main()
