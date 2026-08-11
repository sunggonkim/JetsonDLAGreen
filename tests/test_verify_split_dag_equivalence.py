import importlib.util
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_split_dag_equivalence.py"
SPEC = importlib.util.spec_from_file_location("verify_split_dag", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VerifySplitDagEquivalenceTest(unittest.TestCase):
    def test_help_is_available_without_onnxruntime_import(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("learned ONNX DAG split", result.stdout)

    def test_static_shape_is_inferred(self) -> None:
        self.assertEqual(MODULE._static_shape([None, 3, 368, 640]), None)
        self.assertEqual(MODULE._static_shape([1, 3, 368, 640]), (1, 3, 368, 640))
        self.assertEqual(MODULE._static_shape([1, "width", 640]), None)

    def test_nonpositive_shape_is_rejected(self) -> None:
        self.assertIsNone(MODULE._static_shape([1, 3, 0, 640]))

    def test_batch_dynamic_shape_is_concretized(self) -> None:
        self.assertEqual(MODULE._infer_input_shape([None, 3, 368, 640]), (1, 3, 368, 640))
        self.assertIsNone(MODULE._infer_input_shape([None, 3, "height", 640]))


if __name__ == "__main__":
    unittest.main()
