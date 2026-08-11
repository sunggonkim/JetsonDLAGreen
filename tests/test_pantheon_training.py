#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pantheon_train_cifar10", ROOT / "baselines/pantheon/train_cifar10.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PantheonTrainingTest(unittest.TestCase):
    def test_type7_percentile(self) -> None:
        self.assertEqual(MODULE.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(MODULE.percentile([1.0, 2.0], 0.99), 1.99)

    def test_bounded_loader_does_not_consume_extra_batch(self) -> None:
        self.assertEqual(list(MODULE.bounded(range(5), 2)), [0, 1])

    def test_rejects_invalid_percentile(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid percentile"):
            MODULE.percentile([], 0.5)


if __name__ == "__main__":
    unittest.main()
