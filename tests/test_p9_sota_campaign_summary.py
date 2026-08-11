#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p9_sota_campaign", ROOT / "analysis" / "summarize_p9_sota_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


class P9SotaCampaignSummaryTest(unittest.TestCase):
    def test_zero_miss_6400_requests_certifies_target(self) -> None:
        self.assertLessEqual(SUMMARY.clopper_pearson_upper(0, 6400), 0.0005)

    def test_replays_counts_and_public_system_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runs = []
            for system, policy in SUMMARY.SYSTEM_POLICY.items():
                path = root / system / "summary.json"
                path.parent.mkdir()
                path.write_text(
                    json.dumps(
                        {
                            "deadline_ms": 6.0,
                            "config": {
                                "scenario": "independent",
                                "pressure_rps_per_tenant": 100.0,
                                "epochs": 8,
                                "samples_per_epoch": 800,
                                "dmr_target": 0.0005,
                                "critical_placement": "2g",
                                "resident_placement": "1g",
                                "borrower_placement": "2g",
                                "borrower_quota": 100 if system == "QUIET" else 25,
                                "guard_override_ms": 10.0 if system == "QUIET" else None,
                            },
                            "artifacts": {
                                "benchmark_sha256": "a" * 64,
                                "implementation_sha256": {"runtime": "b" * 64},
                            },
                            "policies": [
                                {
                                    "name": policy,
                                    "critical_requests": 6400,
                                    "deadline_misses": 0,
                                    "deadline_miss_rate": 0.0,
                                    "pressure_goodput_per_second": 199.0,
                                    "critical_p99_ms_max": 5.0,
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                runs.append(
                    {
                        "system": system,
                        "scenario": "independent",
                        "offered_rps_per_tenant": 100,
                        "repeat": 1,
                        "position": 1,
                        "summary": str(path.relative_to(root)),
                    }
                )
            result = SUMMARY.summarize({"quiet_guard_ms": 10, "runs": runs}, root)
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(result["competitor"], "BOER (Thor port)")
        self.assertTrue(all(row["slo_certified"] for row in result["rows"]))


if __name__ == "__main__":
    unittest.main()
