import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deepplan_thor", ROOT / "baselines/deepplan/thor_adapter.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def layer(index, layer_type, size, load, device, host):
    return {
        "index": index, "layer_type": layer_type, "size_bytes": size,
        "load_time_us": load, "cuda_exec_time_us": device,
        "cuda_host_exec_time_us": host,
    }


class DeepPlanThorTest(unittest.TestCase):
    def test_static_selects_direct_host_for_measured_transport_edge(self):
        layers = MODULE.validate_layers([layer(0, "DAGEdge", 2304000, 114.04071, 0.0, 14.05766)])
        self.assertEqual(MODULE.summarize(MODULE.static_plan(layers))["direct_host_layers"], [0])
        self.assertEqual(MODULE.summarize(MODULE.dynamic_plan(layers))["direct_host_layers"], [0])

    def test_dynamic_preserves_upstream_overload_guard(self):
        layers = MODULE.validate_layers([layer(0, "Linear", 1024, 10.0, 1.0, 20.0)])
        self.assertEqual(MODULE.summarize(MODULE.dynamic_plan(layers))["load_then_execute_layers"], [0])

    def test_cli_binds_upstream_commit_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            output = root / "plan.json"
            profile.write_text(json.dumps({
                "kind": "deepplan-thor-layer-profile",
                "layers": [layer(0, "DAGEdge", 2304000, 114.04071, 0.0, 14.05766)],
            }), encoding="utf-8")
            self.assertEqual(MODULE.main(["--profile", str(profile), "--output", str(output)]), 0)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["system"], "DeepPlan")
            self.assertEqual(value["upstream_commit"], MODULE.UPSTREAM_COMMIT)
            self.assertIn("not-yet-common-runtime", value["scope"])

    def test_rejects_nonfinite_profile(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.validate_layers([layer(0, "Linear", 1, float("nan"), 1.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
