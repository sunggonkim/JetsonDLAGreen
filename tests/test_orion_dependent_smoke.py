#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORION = ROOT / "baselines/orion"
sys.path.insert(0, str(ORION))
SPEC = importlib.util.spec_from_file_location(
    "verify_orion_dependent_smoke",
    ORION / "verify_dependent_smoke.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OrionDependentSmokeTest(unittest.TestCase):
    @staticmethod
    def result() -> dict[str, object]:
        return {
            "warmup": 2,
            "iterations": 2,
            "deadline_misses": 1,
            "stage_latency_us": {
                "validation_excluded_end_to_end_p99": 19.9,
            },
        }

    @staticmethod
    def pipeline(path: Path) -> None:
        rows = []
        for request, latency in ((2, 10.0), (3, 20.0)):
            row = {column: "1" for column in MODULE.TRACE_COLUMNS}
            row.update({
                "request": str(request),
                "validation_excluded_end_to_end_us": str(latency),
                "wall_end_to_end_us": str(latency + 1.0),
                "deadline_miss": str(int(latency > 15.0)),
            })
            rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=MODULE.TRACE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def event() -> dict[str, object]:
        return {
            "schema_version": 1,
            "decision_sequence": 3,
            "arrival_sequence": 4,
            "client_id": 0,
            "priority": "best-effort",
            "api": "cuLaunchKernelEx",
            "reordered": True,
            "profile_position": 0,
            "resource_class": 0,
            "sm_used": 4,
            "profile_duration_us": 10.0,
            "admission_reason": "complementary-with-high-priority",
            "active_sm_at_admission": 4,
            "active_be_duration_us_at_admission": 0.0,
            "high_priority_active_at_admission": True,
            "initial_gate_clients": 0,
            "start_monotonic_ns": 100,
            "end_monotonic_ns": 110,
            "result": 0,
        }

    @staticmethod
    def profiles() -> list[dict[str, object]]:
        operation = {
            "api": "cuLaunchKernelEx",
            "profile": 0,
            "sm_used": 4,
            "duration_us": 10.0,
        }
        return [{"operations": [operation]}, {"operations": [operation]}]

    def test_replays_pipeline_deadline_and_p99(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.csv"
            self.pipeline(path)
            replay = MODULE.replay_pipeline(path, self.result(), 15.0)
        self.assertEqual(replay["requests"], 2)
        self.assertEqual(replay["misses"], 1)
        self.assertAlmostEqual(replay["p99_us"], 19.9)

    def test_rejects_tampered_deadline_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.csv"
            self.pipeline(path)
            text = path.read_text(encoding="utf-8")
            path.write_text(text.rsplit(",1\n", 1)[0] + ",0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "classification differs"):
                MODULE.replay_pipeline(path, self.result(), 15.0)

    def test_replays_event_only_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps(self.event()) + "\n", encoding="utf-8")
            replay = MODULE.replay_events(
                path,
                self.profiles(),
                {
                    "trace_records": 1,
                    "decisions": 5,
                    "arrivals": 6,
                    "complementary_admissions": 1,
                    "reordered_decisions": 1,
                },
            )
        self.assertEqual(replay, {"records": 1, "complementary": 1, "reordered": 1})

    def test_rejects_event_profile_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            event = self.event()
            event["sm_used"] = 5
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profile differs"):
                MODULE.replay_events(
                    path,
                    self.profiles(),
                    {
                        "trace_records": 1,
                        "decisions": 5,
                        "arrivals": 6,
                        "complementary_admissions": 1,
                        "reordered_decisions": 1,
                    },
                )


if __name__ == "__main__":
    unittest.main()
