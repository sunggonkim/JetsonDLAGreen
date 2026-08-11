import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "transport_williams", ROOT / "scripts/run_p9_transport_williams.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TransportWilliamsTest(unittest.TestCase):
    def test_order_balances_positions_and_predecessors(self):
        orders = MODULE.williams_orders()
        self.assertEqual(len(orders), 4)
        for position in range(4):
            self.assertEqual(
                set(order[position] for order in orders),
                set(MODULE.TREATMENTS),
            )
        pairs = {(order[i], order[i + 1]) for order in orders for i in range(3)}
        self.assertEqual(len(pairs), 12)

    def test_treatment_mapping_uses_expected_instances(self):
        mig = {
            "JDG_MIG_SMALL_UUID": "small",
            "JDG_MIG_BIG_UUID": "big",
            "JDG_MPS_PIPE_DIRECTORY": "/mps",
        }
        same = MODULE.treatment_args("same-instance-registered", mig)
        self.assertIn("small", same)
        self.assertIn("--consumer-mps-pipe", same)
        cross = MODULE.treatment_args("cross-mig-registered", mig)
        self.assertIn("big", cross)
        self.assertNotIn("--consumer-mps-pipe", cross)

    def test_transport_runner_binds_post_completion_application_trace(self):
        source = (ROOT / "scripts/run_p9_transport_williams.py").read_text()
        self.assertIn('"--application-output-trace"', source)
        self.assertIn('"capture_boundary": "post-completion"', source)
        legacy = (ROOT / "scripts/run_p9_mig_trt_transport_smoke.sh").read_text()
        self.assertIn('--application-output-trace', legacy)
        self.assertIn('application output trace', legacy)


if __name__ == "__main__":
    unittest.main()
