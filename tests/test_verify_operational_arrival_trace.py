import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_operational_arrival_trace",
    ROOT / "scripts" / "verify_operational_arrival_trace.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OperationalArrivalTraceTest(unittest.TestCase):
    def write_trace(self, root: Path, count: int = 2) -> Path:
        path = root / "arrival.bin"
        with path.open("wb") as stream:
            stream.write(MODULE.MAGIC)
            stream.write(MODULE.HEADER.pack(1, count, MODULE.RECORD.size))
            for sequence in range(count):
                stream.write(
                    MODULE.RECORD.pack(
                        sequence + 10,
                        sequence,
                        sequence * 2_000_000,
                        ("a" * 64).encode(),
                        f"request-{sequence}".encode().ljust(64, b"\x00"),
                    )
                )
        return path

    def test_accepts_dense_request_bound_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.load(self.write_trace(Path(directory)))
        self.assertEqual(result["format"], "JDGARR1")
        self.assertEqual(result["records"][1]["release_offset_ns"], 2_000_000)

    def test_rejects_non_dense_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory))
            raw = bytearray(path.read_bytes())
            sequence_offset = len(MODULE.MAGIC) + MODULE.HEADER.size + 4
            raw[sequence_offset : sequence_offset + 4] = (7).to_bytes(4, "little")
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "dense"):
                MODULE.load(path)

    def test_rejects_trailing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory))
            path.write_bytes(path.read_bytes() + b"stale")
            with self.assertRaisesRegex(ValueError, "trailing"):
                MODULE.load(path)


if __name__ == "__main__":
    unittest.main()
