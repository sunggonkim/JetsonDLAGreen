from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "analysis" / "freeze_bless_trt_safe_boundaries.py"
SPEC = importlib.util.spec_from_file_location("freeze_bless_boundaries", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FreezeBlessBoundariesTest(unittest.TestCase):
    def fixture(self, root: Path, total: int = 47) -> tuple[Path, Path]:
        profile = root / "profile"
        profile.mkdir()
        engine = root / "engine.plan"
        engine.write_bytes(b"engine")
        safe = {0, total, total // 2}
        for operation in range(total + 1):
            directory = profile / f"op-{operation}"
            directory.mkdir()
            result = {
                "schema_version": 1,
                "kind": "bless-thor-trt-squad-replica-smoke",
                "status": "passed",
                "logical_launches": total,
                "physical_launches": total,
                "shadow_launches": total * 3,
                "restricted_launches": operation,
                "unrestricted_launches": total - operation,
                "activation_copies": int(0 < operation < total),
                "signature_mismatches": 0,
                "selected_output_matches": operation in safe,
            }
            (directory / "result.json").write_text(json.dumps(result) + "\n")
            records = []
            for index in range(total):
                records.append({
                    "operation": index,
                    "selected_sms": 2 if index < operation else 8,
                    "activation_copy": 0 < operation < total and index == operation,
                    "result": 0,
                })
            (directory / "squad.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
        return profile, engine

    def test_selects_safe_midpoint_without_evaluation_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile, engine = self.fixture(Path(raw))
            lock = MODULE.build_lock(profile, engine, ROOT)
            self.assertEqual(lock["selected_switch_operation"], 23)
            self.assertEqual(lock["total_logical_launches"], 47)
            self.assertTrue(lock["held_out_validation_required"])
            self.assertIn(2, lock["unsafe_switch_operations"])

    def test_rejects_a_tampered_trace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile, engine = self.fixture(Path(raw))
            path = profile / "op-23" / "squad.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[22]["selected_sms"] = 8
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(ValueError, "trace differs"):
                MODULE.build_lock(profile, engine, ROOT)

    def test_infers_a_different_engine_launch_count(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile, engine = self.fixture(Path(raw), total=8)
            lock = MODULE.build_lock(profile, engine, ROOT)
            self.assertEqual(lock["total_logical_launches"], 8)
            self.assertEqual(lock["selected_switch_operation"], 4)
            self.assertEqual(len(lock["profile_evidence"]), 9)

    def test_rejects_an_incomplete_operation_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile, engine = self.fixture(Path(raw), total=8)
            for child in (profile / "op-8").iterdir():
                child.unlink()
            (profile / "op-8").rmdir()
            with self.assertRaisesRegex(ValueError, "operation set differs"):
                MODULE.build_lock(profile, engine, ROOT)


if __name__ == "__main__":
    unittest.main()
