import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "numeric_sota_smoke", ROOT / "scripts/run_p9_numeric_sota_smoke.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class NumericSotaSmokeTest(unittest.TestCase):
    def test_public_rows_are_unique_and_only_one_is_proposed(self):
        names = [item.public_name for item in module.SYSTEMS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names.count("QUIET"), 1)
        self.assertNotIn("governor", " ".join(names).lower())

    def test_frozen_mechanism_actions(self):
        actions = {
            item.public_name: (item.producer_quota, item.background_quota)
            for item in module.SYSTEMS
        }
        self.assertEqual(actions["Quota-only provisioning"], (90, 10))
        self.assertEqual(actions["Partition-only planning"], (90, 10))
        self.assertEqual(actions["Full-DAG quiescence"], (100, 100))
        self.assertNotIn("Orion", actions)

    def test_williams_design_balances_positions_and_predecessors(self):
        orders = module.williams_orders()
        self.assertEqual(len(orders), 6)
        for name in module.CANONICAL_NAMES:
            self.assertEqual(
                [sum(order[position] == name for order in orders) for position in range(6)],
                [1] * 6,
            )
        pairs = {
            (order[index], order[(index + 1) % 6])
            for order in orders
            for index in range(6)
        }
        self.assertEqual(len(pairs), 30)


if __name__ == "__main__":
    unittest.main()
