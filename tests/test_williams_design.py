#!/usr/bin/env python3
import collections
import os
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class WilliamsDesignTest(unittest.TestCase):
    def test_policy_positions_and_predecessors_are_balanced(self) -> None:
        environment = os.environ.copy()
        environment["PRINT_POLICY_ORDERS"] = "1"
        output = subprocess.run(
            [str(ROOT / "scripts" / "run_p9_repeated.sh")],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        orders = [line.split(",") for line in output.splitlines() if line]
        self.assertEqual(len(orders), 14)
        policies = set(orders[0])
        self.assertEqual(len(policies), 7)
        self.assertTrue(all(set(order) == policies for order in orders))

        positions = collections.Counter(
            (position, policy)
            for order in orders
            for position, policy in enumerate(order)
        )
        self.assertTrue(all(count == 2 for count in positions.values()))

        predecessors = collections.Counter(
            pair
            for order in orders
            for pair in zip(order, order[1:])
        )
        expected_pairs = {
            (left, right)
            for left in policies
            for right in policies
            if left != right
        }
        self.assertEqual(set(predecessors), expected_pairs)
        self.assertTrue(all(count == 2 for count in predecessors.values()))


if __name__ == "__main__":
    unittest.main()
