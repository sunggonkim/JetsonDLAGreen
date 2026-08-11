import importlib.util
import pathlib
import struct
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "read_application_output_trace.py"
SPEC = importlib.util.spec_from_file_location("read_output_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trace_bytes(records: list[tuple[int, bytes]]) -> bytes:
    output = bytearray(MODULE.MAGIC)
    output += struct.pack("<I", 1)
    output += struct.pack("<Q", len(records[0][1]))
    for iteration, value in records:
        output += struct.pack("<I", iteration)
        output += value
    return bytes(output)


def asr_trace_bytes(records: list[tuple[int, list[int]]]) -> bytes:
    output = bytearray(MODULE.ASR_MAGIC)
    output += struct.pack("<II", 1, len(records))
    for iteration, tokens in records:
        output += struct.pack("<II", iteration, len(tokens))
        output += struct.pack("<" + "I" * len(tokens), *tokens)
    return bytes(output)


class ReadApplicationOutputTraceTest(unittest.TestCase):
    def test_parses_float32_argmax_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "output.bin"
            value = struct.pack("<3f", 0.1, 0.9, 0.2)
            path.write_bytes(trace_bytes([(10, value)]))
            result = MODULE.parse(path, float32_output=True)
        self.assertEqual(result["record_count"], 1)
        self.assertEqual(result["records"][0]["outputs"][0]["argmax"], 1)

    def test_rejects_partial_record_and_duplicate_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "output.bin"
            value = struct.pack("<2f", 0.1, 0.2)
            path.write_bytes(trace_bytes([(1, value)])[:-1])
            with self.assertRaisesRegex(ValueError, "partial record"):
                MODULE.parse(path)
            path.write_bytes(trace_bytes([(1, value), (1, value)]))
            with self.assertRaisesRegex(ValueError, "repeats"):
                MODULE.parse(path)

    def test_float32_values_are_available_for_detector_decoders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "output.bin"
            value = struct.pack("<2f", 0.25, 0.75)
            path.write_bytes(trace_bytes([(0, value)]))
            result = MODULE.parse(path, float32_values=True)
        self.assertEqual(result["float32_values"], True)
        self.assertEqual(result["records"][0]["outputs"][0]["values"], [0.25, 0.75])

    def test_parses_variable_length_whisper_token_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "asr.bin"
            path.write_bytes(asr_trace_bytes([(1, [2221, 13]), (0, [50257])]))
            result = MODULE.parse(path)
        self.assertEqual(result["format"], "JDGASR1")
        self.assertEqual(result["task"], "asr")
        self.assertEqual([row["iteration"] for row in result["records"]], [0, 1])
        self.assertEqual(result["records"][0]["outputs"][0]["tokens"], [50257])
        self.assertEqual(result["records"][1]["outputs"][0]["bytes"], 8)

    def test_rejects_invalid_whisper_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "asr.bin"
            path.write_bytes(asr_trace_bytes([(0, [MODULE.WHISPER_VOCAB_SIZE])]))
            with self.assertRaisesRegex(ValueError, "outside Whisper vocabulary"):
                MODULE.parse(path)


if __name__ == "__main__":
    unittest.main()
