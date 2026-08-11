#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_xsched_dependent_smoke",
    ROOT / "baselines/xsched/verify_dependent_smoke.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class XSchedDependentSmokeTest(unittest.TestCase):
    def test_verifier_contract_is_production_wall(self) -> None:
        source = (ROOT / "baselines/xsched/verify_dependent_smoke.py").read_text()
        self.assertIn('result.get("deadline_mode") != "wall"', source)
        self.assertNotIn('result.get("deadline_mode") != "validation-excluded"', source)
        self.assertIn('capture_boundary": "post-completion"', source)

    def test_resnet_tensor_default_can_be_specialized_before_readonly(self) -> None:
        source = (ROOT / "scripts/run_p9_xsched_dependent_smoke.sh").read_text()
        self.assertIn('CONSUMER_INPUT_TENSOR="${CONSUMER_INPUT_TENSOR:-features}"', source)
        self.assertIn("readonly PRODUCER_ENGINE CONSUMER_INPUT_TENSOR", source)
        self.assertNotIn(
            'readonly CONSUMER_INPUT_TENSOR="${CONSUMER_INPUT_TENSOR:-features}"',
            source,
        )

    def test_replays_completion_goodput_in_critical_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "be.csv"
            with trace.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=MODULE.BE_COLUMNS)
                writer.writeheader()
                for index, latency in enumerate((1.0, 20.0, 1.0)):
                    writer.writerow({
                        "request": index,
                        "release_to_completion_ms": latency,
                        "gpu_service_ms": 1,
                        "queue_delay_ms": 0,
                        "gate_overhead_ms": 0,
                        "drain_ms": 0,
                        "resume_ms": 0,
                    })
            result = MODULE.replay_be_window(
                trace,
                {
                    "config": {"period_ms": 4},
                    "measurement_start_monotonic_ns": 100_000_000,
                    "measurement_end_monotonic_ns": 140_000_000,
                    "completed_requests": 3,
                },
                103_000_000,
                113_000_000,
            )
        self.assertEqual(result["offered_requests"], 3)
        self.assertEqual(result["completed_requests"], 2)
        self.assertEqual(result["completion_goodput_rps"], 200.0)

    def test_rejects_missing_preemption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                "client process 10 connected, cmdline: jdg-trt-bench\n"
                "client process 11 connected, cmdline: jdg-mig-trt-pipeline\n"
                "client process 12 connected, cmdline: jdg-mig-trt-pipeline\n"
                "XQueue (0x1) from process 10 created\n"
                "XQueue (0x2) from process 11 created\n"
                "XQueue (0x3) from process 12 created\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "did not suspend"):
                MODULE.replay_scheduler(log)

    def test_replays_three_xqueues_and_be_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                "client process 10 connected, cmdline: jdg-trt-bench\n"
                "client process 11 connected, cmdline: jdg-mig-trt-pipeline\n"
                "client process 12 connected, cmdline: jdg-mig-trt-pipeline\n"
                "XQueue (0x1) from process 10 created\n"
                "XQueue (0x2) from process 11 created\n"
                "XQueue (0x3) from process 12 created\n"
                "schedule transition pid 10 operation 1 running 1 suspended 0\n"
                "schedule transition pid 10 operation 2 running 0 suspended 1\n"
                "schedule transition pid 10 operation 3 running 1 suspended 0\n",
                encoding="utf-8",
            )
            result = MODULE.replay_scheduler(log)
        self.assertEqual(result["xqueue_clients"], 3)
        self.assertEqual(result["be_suspend_transitions"], 1)
        self.assertEqual(result["be_resume_transitions"], 1)

    def test_learned_head_contract_is_declared(self) -> None:
        self.assertEqual(
            MODULE.WORKLOADS["resnet-detection-head"]["pipeline"],
            "resnet10-backbone-to-learned-detection-head",
        )
        self.assertEqual(MODULE.WORKLOADS["resnet-detection-head"]["payload_bytes"], 1_884_160)


if __name__ == "__main__":
    unittest.main()
