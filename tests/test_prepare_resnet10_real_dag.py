import importlib.util
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_resnet10_real_dag.py"
SPEC = importlib.util.spec_from_file_location("prepare_real_dag", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareResnet10RealDagTest(unittest.TestCase):
    def test_split_contract_is_learned_detection_graph(self) -> None:
        self.assertEqual(MODULE.SPLIT_TENSOR, "Layer6_relu_Y")
        self.assertEqual(MODULE.HEAD_OUTPUTS, ("Layer7_cov", "Layer7_bbox"))
        self.assertEqual(MODULE.PRODUCER_INPUT, "data")

    def test_help_is_available_without_onnx_import(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("learned ResNet10 backbone/detection-head", result.stdout)


if __name__ == "__main__":
    unittest.main()
