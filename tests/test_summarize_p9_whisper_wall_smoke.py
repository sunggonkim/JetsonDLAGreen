import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summarize_p9_whisper_wall_smoke import summarize


def fixture(system: str, lock: str = "a" * 64) -> dict:
    return {
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "whisper-projection",
        "latency_contract": "production-wall-arrival-to-completion",
        "deadline_mode": "wall",
        "checksum_mode": "inline",
        "iterations": 100,
        "background_period_ms": 4.0,
        "deadline_us": 7661.928109,
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "deadline_lock": {"sha256": lock},
        "results": [{
            "system": system, "pipeline_requests": 100, "deadline_misses": 0,
            "wall_pipeline_p99_us": 7000.0, "background_goodput_rps": 249.0,
            "correctness_validated": True, "checksum_failures": 0,
            "payload_bytes": 2304000, "unique_payload_checksums": 4,
            "unique_policy_output_checksums": 4,
        }],
    }


class WhisperWallSmokeTest(unittest.TestCase):
    def test_common_lock_and_inline_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS"):
                path = root / f"{system}.json"
                path.write_text(json.dumps(fixture(system)) + "\n")
                paths.append(path)
            result = summarize(paths)
        self.assertEqual(result["payload_bytes"], 2304000)
        self.assertEqual(result["deadline_lock_sha256"], "a" * 64)

    def test_rejects_mixed_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "quiet.json"
            second = root / "mps.json"
            first.write_text(json.dumps(fixture("QUIET")) + "\n")
            second.write_text(json.dumps(fixture("NVIDIA MPS", "b" * 64)) + "\n")
            with self.assertRaisesRegex(ValueError, "different deadline lock"):
                summarize([first, second])


if __name__ == "__main__":
    unittest.main()
