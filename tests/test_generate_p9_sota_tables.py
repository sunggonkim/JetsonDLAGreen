import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_p9_sota_tables", ROOT / "analysis/generate_p9_sota_tables.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(
    (ROOT / "results/p9-common-sota-williams-nonthermal-formal-raw-aggregate-6x1100-20260809T153122Z/summary.json").is_file()
    and (ROOT / "results/p9-quiet-resnet-heldout-load-aggregate-7x1100-20260810/summary.json").is_file(),
    "requires preserved local historical table evidence",
)
class P9SotaTableTest(unittest.TestCase):
    def setUp(self) -> None:
        aggregate_path = Path(
            Path("/tmp/current-common-sota-williams-formal-raw-aggregate").read_text().strip()
        )
        heldout_path = Path(Path("/tmp/current-heldout-load-aggregate").read_text().strip())
        if not aggregate_path.is_absolute():
            aggregate_path = ROOT / aggregate_path
        if not heldout_path.is_absolute():
            heldout_path = ROOT / heldout_path
        self.aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        self.heldout = json.loads(heldout_path.read_text(encoding="utf-8"))

    def test_emits_only_public_system_names_and_raw_values(self) -> None:
        text = MODULE.latex(self.aggregate, self.heldout)
        self.assertIn("\\textbf{QUIET} & 0/6600", text)
        self.assertIn("Orion & \\multicolumn{5}{c}", text)
        self.assertNotIn("Orion & 4456/6600", text)
        self.assertIn("500 & 499.68 & 0/1100", text)
        self.assertNotIn("governor", text.lower())

    def test_main_writes_tex_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "aggregate.json"
            heldout = root / "heldout.json"
            aggregate.write_text(json.dumps(self.aggregate), encoding="utf-8")
            heldout.write_text(json.dumps(self.heldout), encoding="utf-8")
            self.assertEqual(MODULE.main([
                "--aggregate", str(aggregate), "--heldout", str(heldout),
                "--tex-output", str(root / "table.tex"),
                "--csv-output", str(root / "table.csv"),
            ]), 0)
            self.assertTrue((root / "table.tex").is_file())
            self.assertEqual(len((root / "table.csv").read_text().splitlines()), 7)

    def test_rejects_internal_policy_name(self) -> None:
        self.aggregate["internal"] = "mig-governor"
        with self.assertRaisesRegex(ValueError, "internal policy"):
            MODULE.validate(self.aggregate, self.heldout)

    def test_slo_infeasible_rows_cannot_render_as_numeric(self) -> None:
        aggregate = json.loads(json.dumps(self.aggregate))
        aggregate["systems"]["NVIDIA MPS"]["numeric_comparison_allowed"] = True
        aggregate["systems"]["NVIDIA MPS"]["slo_confidence_qualified"] = False
        row = next(
            item for item in MODULE.format_rows(aggregate)
            if item["system"] == "NVIDIA MPS"
        )
        self.assertIsNone(row["p99_us"])
        self.assertIn("slo-infeasible", row["status"])

    def test_new_headline_view_excludes_legacy_gpulet(self) -> None:
        aggregate = json.loads(json.dumps(self.aggregate))
        headline = {
            name: aggregate["systems"][name]
            for name in ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "QUIET")
        }
        headline["Pantheon"] = {
            "comparison_status": "functional-only-pending-common-workload-adapter"
        }
        aggregate["headline_systems"] = {
            name: headline[name]
            for name in ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "Pantheon", "QUIET")
        }
        MODULE.validate(aggregate, self.heldout)
        rows = MODULE.format_rows(aggregate)
        self.assertEqual([row["system"] for row in rows], list(MODULE.HEADLINE_SYSTEM_ORDER))
        self.assertIsNone(next(row for row in rows if row["system"] == "Pantheon")["p99_us"])
        self.assertNotIn("gpulet", [row["system"] for row in rows])


if __name__ == "__main__":
    unittest.main()
