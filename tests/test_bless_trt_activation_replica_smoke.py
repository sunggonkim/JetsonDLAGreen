import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_bless_activation",
    ROOT / "baselines/bless/verify_trt_activation_replica_smoke.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def launch(sequence: int) -> dict:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "api": "cuLaunchKernelEx",
        "tid": 1,
        "start_monotonic_ns": 100 + sequence * 2,
        "end_monotonic_ns": 101 + sequence * 2,
        "function": "0x1",
        "stream": "0x2",
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
        "shared_mem_bytes": 0,
        "attributes": 0,
        "result": 0,
    }


class BlessActivationReplicaSmokeTest(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        engine = root / "engine.plan"
        engine.write_bytes(b"engine")
        (root / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "bless-thor-trt-activation-replica-smoke",
                    "affinity_domain_sms": [2, 4, 6, 8],
                    "activation_bytes": 1024,
                    "output_checksums": [7, 7, 7, 7],
                    "activation_source_checksum": 9,
                    "activation_destination_checksum": 9,
                    "post_copy_output_checksum": 7,
                    "restricted_to_unrestricted_copy": True,
                    "status": "passed",
                }
            )
        )
        (root / "stderr.txt").write_text("")
        records = [launch(sequence) for sequence in range(10)]
        (root / "driver-launches.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        return engine.resolve()

    def test_accepts_peer_copied_activation_and_repeated_launches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = MODULE.verify(root, self.fixture(root))
        self.assertTrue(result["restricted_to_unrestricted_peer_copy"])
        self.assertTrue(result["post_copy_inference_passed"])

    def test_rejects_activation_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = self.fixture(root)
            raw = json.loads((root / "result.json").read_text())
            raw["activation_destination_checksum"] = 10
            (root / "result.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "activation result"):
                MODULE.verify(root, engine)

    def test_rejects_divergent_post_copy_launch_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = self.fixture(root)
            records = [
                json.loads(line)
                for line in (root / "driver-launches.jsonl").read_text().splitlines()
            ]
            records[-1]["grid"] = [2, 1, 1]
            (root / "driver-launches.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            with self.assertRaisesRegex(ValueError, "sequences differ"):
                MODULE.verify(root, engine)


if __name__ == "__main__":
    unittest.main()
