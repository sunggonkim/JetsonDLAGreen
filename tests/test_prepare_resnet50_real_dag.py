import importlib.util
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_resnet50_real_dag.py"
SPEC = importlib.util.spec_from_file_location("prepare_resnet50_dag", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareResnet50RealDagTest(unittest.TestCase):
    def test_contract_is_learned_classification_dag(self) -> None:
        self.assertEqual(MODULE.PRODUCER_INPUT, "gpu_0/data_0")
        self.assertEqual(MODULE.FINAL_OUTPUT, "gpu_0/softmax_1")
        self.assertEqual(MODULE.SPLIT_TENSOR, "gpu_0/res4_5_branch2c_bn_2")

    def test_help_is_available_without_onnx_import(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("learned ResNet-50 classification DAG", result.stdout)


if __name__ == "__main__":
    unittest.main()
