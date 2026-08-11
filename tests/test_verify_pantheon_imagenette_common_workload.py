import struct
import tempfile
import unittest
from pathlib import Path

from analysis.verify_pantheon_imagenette_common_workload import _output, _timings


class PantheonImageNetteVerifierTest(unittest.TestCase):
    def test_post_completion_output_trace_is_dense_and_float32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.bin"
            raw = bytearray(b"JDGOUT1\0")
            raw.extend(struct.pack("<IQ", 1, 40))
            raw.extend(struct.pack("<I10f", 0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0))
            raw.extend(struct.pack("<I10f", 1, 2.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0))
            path.write_bytes(raw)
            info, predictions = _output(path, 2)
            self.assertEqual(info["capture_boundary"], "post-completion")
            self.assertEqual(predictions, [1, 0])

    def test_runtime_exit_trace_binds_actual_release_and_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.log"
            path.write_text(
                "[info] [EXEC:EXIT] HIGH_PRIORITY 100 1101 1001 10 0 0 0.8\n"
                "[info] [EXEC:EXIT] HIGH_PRIORITY 200 250 50 5 11 0 0.8\n",
                encoding="ascii",
            )
            rows = _timings(path, 2, 1000)
            self.assertEqual(rows[0]["wall_latency_us"], 1001.0)
            self.assertTrue(rows[0]["deadline_miss"])
            self.assertFalse(rows[1]["deadline_miss"])


if __name__ == "__main__":
    unittest.main()
