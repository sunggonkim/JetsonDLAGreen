import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_activation_replay_trace",
    ROOT / "scripts" / "verify_activation_replay_trace.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def checksum(payload: bytes) -> int:
    value = 1469598103934665603
    for byte in payload:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


class ActivationReplayTraceTest(unittest.TestCase):
    def write_trace(self, root: Path, payloads: list[bytes]) -> Path:
        path = root / "activations.bin"
        with path.open("wb") as stream:
            stream.write(MODULE.MAGIC)
            stream.write(MODULE.HEADER.pack(1, len(payloads), len(payloads[0])))
            for iteration, payload in enumerate(payloads):
                digest = hashlib.sha256(f"request-{iteration}".encode()).hexdigest()
                stream.write(
                    MODULE.PREFIX.pack(
                        iteration, digest.encode("ascii"), checksum(payload)
                    )
                )
                stream.write(payload)
        return path

    def test_accepts_dense_byte_checked_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory), [b"abcd", b"wxyz"])
            result = MODULE.load(path)
        self.assertEqual(result["format"], "JDGACT1")
        self.assertEqual(result["record_count"], 2)

    def test_rejects_payload_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory), [b"abcd"])
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "checksum"):
                MODULE.load(path)

    def test_rejects_trailing_record_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory), [b"abcd"])
            path.write_bytes(path.read_bytes() + b"stale")
            with self.assertRaisesRegex(ValueError, "trailing"):
                MODULE.load(path)


if __name__ == "__main__":
    unittest.main()
