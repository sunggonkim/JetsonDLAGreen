#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_bless_affinity",
    ROOT / "baselines/bless/verify_affinity_smoke.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BlessAffinitySmokeTest(unittest.TestCase):
    def make_result(self, directory: Path) -> None:
        requests = []
        for requested in range(1, 9):
            actual = ((requested + 1) // 2) * 2
            requests.append({
                "requested_sms": requested,
                "create_result": "CUDA_SUCCESS",
                "actual_sms": actual,
                "query_result": "CUDA_SUCCESS",
                "destroy_result": "CUDA_SUCCESS",
                "create_ns": 1,
            })
        (directory / "context-domain.json").write_text(json.dumps({
            "schema_version": 1,
            "kind": "bless-thor-context-domain",
            "device": "NVIDIA Thor MIG 1g.0gb",
            "multiprocessors": 8,
            "exec_affinity_supported": True,
            "requests": requests,
        }) + "\n", encoding="utf-8")
        for quota, sms in MODULE.EXPECTED:
            benchmark = {
                "schema_version": 1,
                "model": "distilbert-sst2",
                "role": "benchmark",
                "engine": f"/engines/mig-1g-q{quota}/distilbert-sst2.engine",
                "execution_environment": {"mps_active_thread_percentage": 100},
                "gpu": {"name": "NVIDIA Thor MIG 1g.0gb", "multiprocessors": 8},
                "completed_requests": 20,
                "release_to_completion": {"p99_ms": 1.0},
            }
            (directory / f"q{quota}.json").write_text(json.dumps({
                "schema_version": 1,
                "kind": "bless-thor-trt-affinity-smoke",
                "requested_sms": sms,
                "actual_sms": sms,
                "benchmark": benchmark,
            }) + "\n", encoding="utf-8")
            (directory / f"q{quota}.stderr").write_text("", encoding="utf-8")

    def test_accepts_complete_affinity_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_result(directory)
            result = MODULE.verify(directory)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["affinity_domain_sms"], [2, 4, 6, 8])
        self.assertEqual(len(result["evidence_sha256"]), 9)

    def test_rejects_a_failed_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_result(directory)
            (directory / "q25.stderr").write_text(
                "unspecified launch failure\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "failure"):
                MODULE.verify(directory)


if __name__ == "__main__":
    unittest.main()
