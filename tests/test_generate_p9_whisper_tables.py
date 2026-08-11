import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_p9_whisper_tables", ROOT / "analysis/generate_p9_whisper_tables.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P9WhisperTableTest(unittest.TestCase):
    def setUp(self) -> None:
        aggregate = ROOT / "results/p9-common-sota-whisper-current-nonthermal-formal-aggregate-6x1100-20260810/summary.json"
        heldout = ROOT / "results/p9-quiet-whisper-current-heldout-load-sweep-7x1100-20260810/summary.json"
        structural = ROOT / "results/p9-current-whisper-formal-structural-evidence-20260810/summary.json"
        if not aggregate.is_file() or not heldout.is_file() or not structural.is_file():
            self.skipTest("current Whisper evidence is absent")
        self.aggregate = json.loads(aggregate.read_text(encoding="utf-8"))
        self.heldout = json.loads(heldout.read_text(encoding="utf-8"))
        self.structural = json.loads(structural.read_text(encoding="utf-8"))

    def test_renders_replayed_values_and_public_name(self) -> None:
        text = MODULE.latex(self.aggregate, self.heldout, self.structural)
        self.assertIn("NVIDIA MIG & \\multicolumn{5}{c}", text)
        self.assertNotIn("NVIDIA MIG & 1788/6600", text)
        self.assertIn("\\textbf{QUIET} & 0/6600", text)
        self.assertIn("750 & 531.12 & 0/1100", text)
        self.assertIn("BOER & 1.481 & 499.63", text)
        self.assertIn("ParvaGPU & 0.969 & 499.76", text)
        self.assertNotIn("governor", text.lower())

    def test_rejects_wrong_workload(self) -> None:
        self.aggregate["workload"] = "resnet-control"
        with self.assertRaisesRegex(ValueError, "Whisper common-workload"):
            MODULE.validate(self.aggregate, self.heldout, self.structural)

    def test_main_writes_tex_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate, heldout = root / "aggregate.json", root / "heldout.json"
            structural = root / "structural.json"
            aggregate.write_text(json.dumps(self.aggregate), encoding="utf-8")
            heldout.write_text(json.dumps(self.heldout), encoding="utf-8")
            structural.write_text(json.dumps(self.structural), encoding="utf-8")
            self.assertEqual(MODULE.main([
                "--aggregate", str(aggregate), "--heldout", str(heldout),
                "--structural", str(structural),
                "--tex-output", str(root / "table.tex"),
                "--csv-output", str(root / "table.csv"),
            ]), 0)
            self.assertTrue((root / "table.tex").is_file())
            self.assertEqual(len((root / "table.csv").read_text().splitlines()), 7)


if __name__ == "__main__":
    unittest.main()
