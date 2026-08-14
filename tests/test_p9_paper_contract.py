import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/eurosys27"


class P9PaperContractTest(unittest.TestCase):
    def test_public_name_and_evidence_boundary(self):
        paths = [PAPER / "p9-main.tex", *sorted((PAPER / "p9-sections").glob("*.tex"))]
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in ("mig-governor", "joint-governor", "jdg-governor"):
            self.assertNotIn(forbidden, text)
        self.assertIn("Only \\sys is a proposed system name", text)
        self.assertIn("A thermal-normalized formal campaign is now bound", text)

    def test_sota_ports_are_cited_and_native_only(self):
        text = (PAPER / "p9-sections/05-related-work.tex").read_text(encoding="utf-8")
        for citation in ("orion2024", "xsched2025", "pantheon2024", "edgeiso2020"):
            self.assertIn(f"\\cite{{{citation}}}", text)
        implementation = (PAPER / "p9-sections/03-implementation.tex").read_text(encoding="utf-8")
        normalized = " ".join(implementation.split())
        self.assertIn("pinned artifact's scheduler and runtime execute", normalized)
        self.assertNotIn("managed-client port", implementation)

    def test_numeric_roster_is_fixed_and_partial_evidence_is_separate(self):
        systems = (
            "QUIET", "NVIDIA MIG", "NVIDIA MPS", "XSched", "Orion", "Pantheon",
        )
        partial = ("BLESS", "GSLICE", "gpulet", "BOER", "ParvaGPU", "DeepPlan")
        table = (PAPER / "generated/p9-current-results.tex").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        manifest = json.loads((ROOT / "docs/p9-comparator-manifest.json").read_text(encoding="utf-8"))
        compact = json.loads(
            (PAPER / "generated/p9-six-system-imagenette-gate.json").read_text(
                encoding="utf-8"
            )
        )
        readme_sections = (
            readme.split("## Fixed measured comparison roster", 1)[1].split(
                "## Formal promotion ledger", 1
            )[0],
            readme.split("## Formal promotion ledger", 1)[1].split(
                "## Application semantics", 1
            )[0],
            readme.split("## Application semantics", 1)[1].split(
                "## Partial artifact evidence", 1
            )[0],
        )
        generated_sections = (
            table.split(r"\newcommand{\PnineApplicationTable}", 1)[1].split(
                r"\newcommand{\PnineFormalTable}", 1
            )[0],
            table.split(r"\newcommand{\PnineFormalTable}", 1)[1].split(
                r"\newcommand{\PnineComparatorTable}", 1
            )[0],
            table.split(r"\newcommand{\PnineComparatorTable}", 1)[1].split(
                r"\newcommand{\PninePartialEvidenceTable}", 1
            )[0],
        )
        for section in (*readme_sections, *generated_sections):
            positions = [section.index(system) for system in systems]
            self.assertEqual(positions, sorted(positions))

        policy = manifest["paper_table_policy"]
        self.assertEqual(tuple(manifest["headline_order"]), systems)
        self.assertEqual(tuple(policy["fixed_numeric_roster"]), systems)
        self.assertEqual(tuple(policy["executed_result_order"]), systems)
        self.assertEqual(tuple(compact["system_order"]), systems)
        self.assertEqual(tuple(compact["systems"]), systems)
        self.assertEqual(tuple(policy["partial_evidence_order"]), partial)
        self.assertEqual(
            policy["direct_ranking_order"],
            ["QUIET", "NVIDIA MPS", "XSched"],
        )
        reason_table = readme.split("## Why a system is not in the numeric graph", 1)[1].split(
            "## Fixed measured comparison roster", 1
        )[0]
        partial_table = readme.split("## Partial artifact evidence", 1)[1].split(
            "## QUIET mechanism validation", 1
        )[0]
        generated_partial = table.split(r"\newcommand{\PninePartialEvidenceTable}", 1)[1]
        for system in partial:
            self.assertIn(system, reason_table)
            self.assertIn(system, partial_table)
            self.assertIn(system, generated_partial)


if __name__ == "__main__":
    unittest.main()
