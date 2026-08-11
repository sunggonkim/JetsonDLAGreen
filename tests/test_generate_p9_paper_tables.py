import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p9_paper_tables", ROOT / "analysis/generate_p9_paper_tables.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class P9PaperTablesTest(unittest.TestCase):
    def test_public_system_order_has_one_proposed_name(self):
        self.assertEqual(MODULE.SYSTEM_ORDER.count("QUIET"), 1)
        self.assertNotIn("mig-governor", MODULE.SYSTEM_ORDER)

    def test_local_full_dag_result_is_not_labeled_orion(self):
        self.assertEqual(MODULE.latex_name("Full-DAG quiescence"), "Full-DAG quiescence")
        self.assertNotIn("Orion", MODULE.SYSTEM_ORDER)

    def test_presentation_names_do_not_claim_unexecuted_sota(self):
        for forbidden in ("GSLICE", "gpulet", "Orion"):
            self.assertNotIn(forbidden, MODULE.SYSTEM_ORDER)


if __name__ == "__main__":
    unittest.main()
