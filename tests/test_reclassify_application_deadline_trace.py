import csv
import tempfile
import unittest
from pathlib import Path

from analysis.reclassify_application_deadline_trace import reclassify


class ReclassifyApplicationDeadlineTraceTest(unittest.TestCase):
    def test_only_deadline_flags_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "output.csv"
            source.write_text(
                "request,input_sha256,wall_end_to_end_us,deadline_miss\n"
                "0," + "a" * 64 + ",10.0,0\n"
                "1," + "b" * 64 + ",20.0,0\n",
                encoding="utf-8",
            )
            result = reclassify(source, output, 15.0)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual([row["wall_end_to_end_us"] for row in rows], ["10.0", "20.0"])
        self.assertEqual([row["input_sha256"] for row in rows], ["a" * 64, "b" * 64])
        self.assertEqual([row["deadline_miss"] for row in rows], ["0", "1"])
        self.assertEqual(result["deadline_misses"], 1)
        self.assertTrue(result["latencies_unchanged"])


if __name__ == "__main__":
    unittest.main()
