#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines/bless"))

from scheduler import (  # noqa: E402
    KernelProfile,
    RequestState,
    ScheduledKernel,
    choose_configuration,
    form_kernel_squad,
    strict_share_configurations,
)


def kernel(name: str, full: float, half: float, cumulative: float) -> KernelProfile:
    return KernelProfile(
        name=name,
        cumulative_us={50: cumulative * 1.5, 100: cumulative},
        duration_us={50: half, 100: full},
        native_share=100,
    )


class BlessSchedulerTest(unittest.TestCase):
    def test_squad_selects_smallest_relative_progress(self) -> None:
        a = RequestState("a", 100.0, 30.0, [kernel("a0", 3, 5, 20)])
        b = RequestState("b", 100.0, 10.0, [kernel("b0", 4, 6, 20)])
        squad = form_kernel_squad([a, b])
        self.assertEqual([(item.request_id, item.kernel_index) for item in squad], [("b", 0)])

    def test_squad_stops_at_maximum(self) -> None:
        kernels = [kernel(f"k{index}", 1, 2, index + 1) for index in range(10)]
        squad = form_kernel_squad([RequestState("a", 10, 0, kernels)], maximum_kernels=6)
        self.assertEqual(len(squad), 6)

    def test_squad_stops_at_request_end(self) -> None:
        a = RequestState("a", 10, 0, [kernel("a0", 1, 2, 1)])
        b = RequestState("b", 10, 1, [kernel("b0", 1, 2, 1)] * 8)
        squad = form_kernel_squad([a, b])
        self.assertEqual([(item.request_id, item.kernel_index) for item in squad], [("a", 0)])

    def test_enumerates_only_complete_spatial_partitions(self) -> None:
        configurations = strict_share_configurations(["a", "b"], [25, 50, 75])
        self.assertEqual(configurations, [
            {"a": 25, "b": 75},
            {"a": 50, "b": 50},
            {"a": 75, "b": 25},
        ])

    def test_configuration_uses_paper_estimators(self) -> None:
        squad = [
            ScheduledKernel("a", 0, kernel("a0", 7, 4, 7)),
            ScheduledKernel("b", 0, kernel("b0", 8, 5, 8)),
        ]
        selected = choose_configuration(squad, [50])
        self.assertEqual(selected.estimator, "interference-free")
        self.assertEqual(selected.shares, {"a": 50, "b": 50})
        self.assertEqual(selected.predicted_us, 5.0)

    def test_unrestricted_wins_when_restriction_is_expensive(self) -> None:
        squad = [
            ScheduledKernel("a", 0, kernel("a0", 2, 20, 2)),
            ScheduledKernel("b", 0, kernel("b0", 3, 20, 3)),
        ]
        selected = choose_configuration(squad, [50])
        self.assertEqual(selected.estimator, "workload-equivalence")
        self.assertIsNone(selected.shares)
        self.assertEqual(selected.predicted_us, 5.0)


if __name__ == "__main__":
    unittest.main()
