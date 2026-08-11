#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "boer_finalize", ROOT / "baselines" / "boer" / "finalize_result.py"
)
assert SPEC is not None and SPEC.loader is not None
FINALIZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FINALIZE
SPEC.loader.exec_module(FINALIZE)


class BoerFinalizeTest(unittest.TestCase):
    def test_binds_selected_replay_and_profiles(self) -> None:
        search = {
            "provenance": {
                "upstream_commit": FINALIZE.UPSTREAM_COMMIT,
                "fidelity": FINALIZE.FIDELITY,
            },
            "selected": {
                "sm_percent": 25,
                "offered_rps": 200,
                "metrics": {
                    "served_rps_0": 190.0,
                    "served_rps_1": 191.0,
                    "worst_p99_ms": 5.5,
                    "deadline_miss_rate": 0.0,
                },
            },
        }
        replay = {
            "deadline_ms": 6.0,
            "config": {
                "scenario": "independent",
                "epochs": 1,
                "samples_per_epoch": 16,
                "period_ms": 20.0,
                "pressure_rps_per_tenant": 200.0,
                "burst_size": 8,
                "dmr_target": 0.0005,
                "borrower_quota": 25,
            },
            "policies": [
                {
                    "name": "uncoordinated-borrow",
                    "goodput_by_modality": {"audio": 198.0, "language": 197.0},
                    "critical_p99_ms_max": 5.7,
                    "deadline_miss_rate": 0.01,
                    "pressure_goodput_per_second": 395.0,
                    "critical_requests": 100,
                    "deadline_misses": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            profile = pathlib.Path(temporary) / "profile.json"
            profile.write_text("{}\n", encoding="utf-8")
            result = FINALIZE.finalize(search, replay, [profile])
        self.assertEqual(result["system"], "BOER")
        self.assertEqual(result["metrics"]["pressure_goodput_per_second"], 395.0)
        self.assertEqual(result["selected"]["metrics"]["served_rps_0"], 190.0)
        self.assertEqual(result["measurement_stage"], "final-replay")

    def test_rejects_replay_with_a_different_selected_configuration(self) -> None:
        search = {
            "provenance": {
                "upstream_commit": FINALIZE.UPSTREAM_COMMIT,
                "fidelity": FINALIZE.FIDELITY,
            },
            "selected": {"sm_percent": 25, "offered_rps": 200, "metrics": {}},
        }
        replay = {
            "config": {
                "borrower_quota": 50,
                "pressure_rps_per_tenant": 200.0,
            },
            "policies": [{"name": "uncoordinated-borrow"}],
        }
        with self.assertRaisesRegex(ValueError, "quota differs"):
            FINALIZE.finalize(search, replay, [])


if __name__ == "__main__":
    unittest.main()
