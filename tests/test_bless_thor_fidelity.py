import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bless_thor_fidelity", ROOT / "analysis/summarize_bless_thor_fidelity.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(
    (ROOT / "results/p9-bless-native-squad-20260809T123619Z/verification.json").is_file(),
    "requires preserved local BLESS hardware evidence",
)
class BlessThorFidelityTest(unittest.TestCase):
    def test_joins_real_preserved_gates(self) -> None:
        result = MODULE.summarize(
            ROOT / "results/p9-bless-native-squad-20260809T123619Z/verification.json",
            ROOT / "results/p9-bless-trt-context-replica-traced-v2-20260809T1328Z/verification.json",
            ROOT / "results/p9-bless-trt-activation-replica-20260809T1335Z/verification.json",
            ROOT / "results/p9-bless-trt-squad-replica-heldout-20260809T1351Z/verification.json",
        )
        self.assertEqual(result["status"], "passed-functional-gates")
        self.assertFalse(result["numeric_comparison_allowed"])
        self.assertEqual(result["tensorrt_replicas"]["driver_launch_records"], 9400)
        self.assertEqual(result["logical_squad_admission"]["physical_launches"], 47)

    def test_rejects_numeric_promotion(self) -> None:
        native = MODULE.load(
            ROOT / "results/p9-bless-native-squad-20260809T123619Z/verification.json"
        )
        replica = MODULE.load(
            ROOT / "results/p9-bless-trt-context-replica-traced-v2-20260809T1328Z/verification.json"
        )
        activation = (
            ROOT / "results/p9-bless-trt-activation-replica-20260809T1335Z/verification.json"
        )
        squad = ROOT / "results/p9-bless-trt-squad-replica-heldout-20260809T1351Z/verification.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.json"
            replica_path = root / "replica.json"
            native_path.write_text(json.dumps(native))
            replica["numeric_comparison_allowed"] = True
            replica_path.write_text(json.dumps(replica))
            with self.assertRaisesRegex(ValueError, "TensorRT-context"):
                MODULE.summarize(native_path, replica_path, activation, squad)


if __name__ == "__main__":
    unittest.main()
