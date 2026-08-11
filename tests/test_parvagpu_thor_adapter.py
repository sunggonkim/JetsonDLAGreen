#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parvagpu_thor_adapter", ROOT / "baselines" / "parvagpu" / "thor_adapter.py"
)
assert SPEC is not None and SPEC.loader is not None
PARVA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PARVA
SPEC.loader.exec_module(PARVA)


class ParvaGpuThorAdapterTest(unittest.TestCase):
    @staticmethod
    def profiles() -> list:
        return [
            PARVA.ProfilePoint("audio", 1, 1, 1, 500.0, 2.0),
            PARVA.ProfilePoint("audio", 2, 1, 1, 800.0, 2.0),
            PARVA.ProfilePoint("language", 1, 1, 1, 400.0, 2.0),
            PARVA.ProfilePoint("language", 2, 1, 1, 700.0, 2.0),
        ]

    @staticmethod
    def spec(services: list[dict]) -> dict:
        return {
            "contract": {"pressure_layout": "1g+2g"},
            "available_segments_gpc": [1],
            "services": services,
        }

    def test_single_service_uses_remaining_one_g_segment(self) -> None:
        result = PARVA.run(
            self.spec([{"model": "audio", "request_rate": 300, "slo_ms": 10}]),
            self.profiles(),
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["allocation"][0]["physical_segment_gpc"], 1)

    def test_two_models_are_infeasible_in_fixed_layout(self) -> None:
        result = PARVA.run(
            self.spec(
                [
                    {"model": "audio", "request_rate": 300, "slo_ms": 10},
                    {"model": "language", "request_rate": 300, "slo_ms": 10},
                ]
            ),
            self.profiles(),
        )
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "insufficient fixed MIG segments")

    def test_configurator_maximizes_throughput_per_gpc(self) -> None:
        service = PARVA.Service("audio", 300.0, 10.0)
        requests = PARVA.configure_service(service, self.profiles())
        self.assertEqual(requests[0].point.segment_gpc, 1)

    def test_independent_fixed_layout_can_use_both_segments(self) -> None:
        spec = self.spec(
            [
                {"model": "audio", "request_rate": 300, "slo_ms": 10},
                {"model": "language", "request_rate": 300, "slo_ms": 10},
            ]
        )
        spec["available_segments_gpc"] = [1, 2]
        result = PARVA.run(spec, self.profiles())
        self.assertTrue(result["feasible"])
        self.assertEqual(
            sorted(item["physical_segment_gpc"] for item in result["allocation"]),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
