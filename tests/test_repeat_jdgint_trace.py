import hashlib
import importlib.util
import pathlib
import struct
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/repeat_jdgint_trace.py"
SPEC = importlib.util.spec_from_file_location("repeat_jdgint_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(path: pathlib.Path, payloads: list[bytes]) -> None:
    with path.open("wb") as stream:
        stream.write(MODULE.MAGIC)
        stream.write(MODULE.HEADER.pack(1, len(payloads), len(payloads[0])))
        for iteration, payload in enumerate(payloads):
            stream.write(struct.pack("<I", iteration))
            stream.write(hashlib.sha256(payload).hexdigest().encode("ascii"))
            stream.write(payload)


class RepeatJdgintTraceTest(unittest.TestCase):
    def test_cycles_valid_records_and_rewrites_dense_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source.bin"
            output = root / "output.bin"
            _source(source, [b"aaaa", b"bbbb"])
            result = MODULE.repeat(source, output, 5)
            sample_bytes, records = MODULE._read(output)
        self.assertEqual(sample_bytes, 4)
        self.assertEqual(result["source_records"], 2)
        self.assertEqual(result["output_records"], 5)
        self.assertEqual([payload for _, payload in records],
                         [b"aaaa", b"bbbb", b"aaaa", b"bbbb", b"aaaa"])
        self.assertEqual(
            result["coverage_policy"],
            "cyclic-performance-replay-not-accuracy-expansion",
        )

    def test_rejects_payload_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source.bin"
            _source(source, [b"aaaa"])
            raw = bytearray(source.read_bytes())
            raw[-1] ^= 1
            source.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "payload hash"):
                MODULE.repeat(source, source.with_name("output.bin"), 2)


if __name__ == "__main__":
    unittest.main()
