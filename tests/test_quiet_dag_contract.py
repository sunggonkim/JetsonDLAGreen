import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quiet_dag_contract", ROOT / "runtime" / "quiet_dag_contract.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QuietDagContractTest(unittest.TestCase):
    def test_three_stage_chain_is_validated(self) -> None:
        result = MODULE.validate_dag({
            "stages": [{"id": "vision"}, {"id": "head"}, {"id": "policy"}],
            "edges": [
                {"source": "vision", "target": "head"},
                {"source": "head", "target": "policy"},
            ],
        })
        self.assertEqual(result["topology"], "three-or-more-stage-chain")
        self.assertEqual(result["multi_stage_validation"], "passed")
        self.assertTrue(result["general_dag_claim_allowed"])

    def test_fan_out_fan_in_is_validated(self) -> None:
        result = MODULE.validate_dag({
            "stages": [
                {"id": "encoder"}, {"id": "vision"},
                {"id": "audio"}, {"id": "join"},
            ],
            "edges": [
                {"source": "encoder", "target": "vision"},
                {"source": "encoder", "target": "audio"},
                {"source": "vision", "target": "join"},
                {"source": "audio", "target": "join"},
            ],
        })
        self.assertEqual(result["topology"], "fan-out-fan-in")
        self.assertEqual(result["multi_stage_validation"], "passed")

    def test_two_stage_shape_does_not_enable_general_claim(self) -> None:
        result = MODULE.validate_dag({
            "stages": [{"id": "producer"}, {"id": "consumer"}],
            "edges": [{"source": "producer", "target": "consumer"}],
        })
        self.assertEqual(result["topology"], "two-stage-chain")
        self.assertEqual(result["multi_stage_validation"], "pending")
        self.assertFalse(result["general_dag_claim_allowed"])

    def test_cycle_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cycle"):
            MODULE.validate_dag({
                "stages": [{"id": "a"}, {"id": "b"}],
                "edges": [
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "a"},
                ],
            })


if __name__ == "__main__":
    unittest.main()
