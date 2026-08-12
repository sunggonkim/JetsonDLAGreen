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

    def test_all_locally_executed_systems_are_visible(self):
        systems = (
            "QUIET",
            "NVIDIA MPS",
            "XSched",
            "Pantheon",
            "Orion",
            "BLESS",
            "NVIDIA MIG",
            "GSLICE",
            "gpulet",
            "BOER",
            "ParvaGPU",
            "DeepPlan",
        )
        table = (PAPER / "generated/p9-current-results.tex").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        matrix = (ROOT / "docs/sota-matrix.md").read_text(encoding="utf-8")
        reselection = (ROOT / "docs/p9-sota-reselection.md").read_text(encoding="utf-8")
        manifest = json.loads((ROOT / "docs/p9-comparator-manifest.json").read_text(encoding="utf-8"))
        executed_readme = readme.split("### All locally executed comparison systems", 1)[1].split(
            "### Directly comparable formal campaign", 1
        )[0]
        current_matrix = matrix.split("## Current reporting rule", 1)[1].split(
            "## Historical design-space context", 1
        )[0]
        current_reselection = reselection.split("## Current paper contract", 1)[1].split(
            "## Superseded pre-thermal decision record", 1
        )[0]
        for system in systems:
            self.assertIn(system, table)
            self.assertIn(system, executed_readme)
            self.assertIn(system, current_matrix)
            self.assertIn(system, current_reselection)
        self.assertEqual(tuple(manifest["paper_table_policy"]["executed_result_order"]), systems)
        self.assertEqual(
            manifest["paper_table_policy"]["direct_ranking_order"],
            ["QUIET", "NVIDIA MPS", "XSched"],
        )
        self.assertIn("A rows share the formal contract", table)
        self.assertIn("A different workload, deadline, or fidelity", executed_readme)
        self.assertIn("Execution visibility and statistical comparability", current_matrix)
        self.assertIn("Visibility:", current_reselection)
        self.assertIn("Ranking:", current_reselection)


if __name__ == "__main__":
    unittest.main()
