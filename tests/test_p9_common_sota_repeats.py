import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "common_sota_repeats", ROOT / "analysis/summarize_p9_common_sota_repeats.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CommonSotaRepeatsTest(unittest.TestCase):
    def make_summary(self, root: Path, repeat: int) -> Path:
        inputs = {}
        for name in MODULE.NUMERIC_INPUTS:
            source = root / f"{repeat}-{name}.json"
            source.write_text(json.dumps({"repeat": repeat, "name": name}), encoding="utf-8")
            inputs[name] = {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        systems = [
            {
                "system": name,
                "requests": 100,
                "misses": 0 if name == "QUIET" else 50,
                "deadline_p99_us": 700.0 + repeat,
                "deadline_us": 770.0,
                "background_goodput_rps": 240.0 + repeat,
            }
            for name in MODULE.SYSTEMS
        ]
        summary = {
            "kind": "p9-dependent-payload-six-system-smoke",
            "workload": {
                "producer": "TensorRT ResNet10 Layer7_cov",
                "edge_payload_bytes": 14_720,
            },
            "scope": "functional-smoke-not-formal-statistics",
            "offered_background_rps": 250.0,
            "common_deadline_us": 770.0,
            "proposed_system": "QUIET",
            "inputs": inputs,
            "systems": systems,
        }
        path = root / f"summary-{repeat}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def test_aggregates_independent_repeats_without_formal_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = MODULE.summarize([self.make_summary(root, 1), self.make_summary(root, 2)])
        self.assertEqual(result["scope"], "independent-hardware-repeats-not-counterbalanced-not-formal")
        self.assertEqual(result["systems"]["QUIET"]["misses"], 0)
        self.assertEqual(result["systems"]["NVIDIA MIG"]["misses"], 100)

    def test_rejects_reused_numeric_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = self.make_summary(root, 1)
            two = self.make_summary(root, 2)
            value = json.loads(two.read_text(encoding="utf-8"))
            value["inputs"]["orion_numeric"] = json.loads(one.read_text(encoding="utf-8"))["inputs"]["orion_numeric"]
            two.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reused"):
                MODULE.summarize([one, two])


if __name__ == "__main__":
    unittest.main()
