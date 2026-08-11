#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_orion_profiled_smoke",
    ROOT / "baselines/orion/verify_profiled_smoke.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OrionProfiledSmokeTest(unittest.TestCase):
    @staticmethod
    def profile(root: Path, name: str, resource: int) -> Path:
        scheduler = root / f"{name}.tsv"
        scheduler.write_text(
            "orion-thor-profile-v1\n"
            "position\tapi\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\t"
            "block_z\tshared_mem_bytes\tprofile\tsm_used\tduration_us\n"
            f"0\tcuLaunchKernelEx\t1\t1\t1\t32\t1\t1\t0\t{resource}\t4\t10\n",
            encoding="ascii",
        )
        path = root / f"{name}.json"
        value = {
            "kind": "orion-thor-operation-profile",
            "upstream_commit": MODULE.UPSTREAM_COMMIT,
            "numeric_comparison_allowed": False,
            "scheduler_profile": {
                "schema": "orion-thor-profile-v1",
                "path": str(scheduler),
                "sha256": hashlib.sha256(scheduler.read_bytes()).hexdigest(),
            },
            "operations": [{
                "position": 0, "api": "cuLaunchKernelEx",
                "grid": [1, 1, 1], "block": [32, 1, 1],
                "shared_mem_bytes": 0, "profile": resource,
                "sm_used": 4, "duration_us": 10.0,
            }],
        }
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        be = self.profile(root, "be", 0)
        hp = self.profile(root, "hp", 1)
        rows = []
        for index, client in enumerate((1, 0)):
            complementary = client == 0
            rows.append({
                "schema_version": 1, "decision_sequence": index,
                "arrival_sequence": index, "client_id": client,
                "priority": "high" if client else "best-effort",
                "api": "cuLaunchKernelEx", "reordered": False,
                "profile_position": 0, "resource_class": 1 if client else 0,
                "sm_used": 4, "profile_duration_us": 10.0,
                "admission_reason": "complementary-with-high-priority"
                                    if complementary else "high-priority",
                "active_sm_at_admission": 4 if complementary else 0,
                "active_be_duration_us_at_admission": 0.0,
                "high_priority_active_at_admission": complementary,
                "initial_gate_clients": 0,
                "start_monotonic_ns": 10 + index * 10,
                "end_monotonic_ns": 15 + index * 10,
                "result": 0,
            })
        decisions = root / "decisions.jsonl"
        decisions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        result = root / "result.json"
        result.write_text(json.dumps({
            "kind": "orion-thor-profile-aware-positive-control",
            "upstream_commit": MODULE.UPSTREAM_COMMIT,
            "port_stage": "profile-aware-admission",
            "numeric_comparison_allowed": False,
            "scheduler": {
                "algorithm": "orion-profile-aware", "initial_gate_clients": 0,
                "arrivals": 2, "decisions": 2, "reordered_decisions": 0,
                "high_priority_decisions": 1,
                "profiled_best_effort_admissions": 1,
                "complementary_admissions": 1, "profile_blocked_polls": 1,
            },
            "best_effort": {"completed_requests": 1},
            "high_priority": {"completed_requests": 1},
        }) + "\n", encoding="utf-8")
        return result, decisions, be, hp

    def test_replays_profile_aware_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.fixture(Path(directory))
            verified = MODULE.verify(*evidence)
        self.assertTrue(verified["functional_gate_passed"])
        self.assertFalse(verified["numeric_comparison_allowed"])
        self.assertEqual(verified["complementary_admissions"], 1)

    def test_rejects_tampered_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, decisions, be, hp = self.fixture(root)
            rows = [json.loads(line) for line in decisions.read_text().splitlines()]
            rows[1]["high_priority_active_at_admission"] = False
            decisions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "lacks active HP"):
                MODULE.verify(result, decisions, be, hp)


if __name__ == "__main__":
    unittest.main()
