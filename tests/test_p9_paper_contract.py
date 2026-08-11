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


if __name__ == "__main__":
    unittest.main()
