from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "baselines" / "bless" / "verify_trt_squad_replica_smoke.py"
SPEC = importlib.util.spec_from_file_location("verify_bless_trt_squad", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BlessTrtSquadReplicaTest(unittest.TestCase):
    def fixture(self, directory: Path) -> tuple[Path, Path]:
        engine = directory / "engine.plan"
        engine.write_bytes(b"engine")
        result = {
            "schema_version": 1,
            "kind": "bless-thor-trt-squad-replica-smoke",
            "status": "passed",
            "affinity_domain_sms": [2, 4, 6, 8],
            "logical_launches": 47,
            "physical_launches": 47,
            "shadow_launches": 141,
            "signature_mismatches": 0,
            "restricted_launches": 23,
            "unrestricted_launches": 24,
            "activation_copies": 1,
            "last_selected_sms": 8,
            "selected_output_matches": True,
            "selected_output_checksum": 17,
            "output_checksums": [17, 17, 17, 17],
        }
        (directory / "result.json").write_text(json.dumps(result) + "\n")
        records = []
        for index in range(47):
            records.append(
                {
                    "schema_version": 1,
                    "operation": index,
                    "api": "cuLaunchKernelEx",
                    "selected_sms": 2 if index < 23 else 8,
                    "activation_copy": index == 23,
                    "grid": [1, 1, 1],
                    "block": [32, 1, 1],
                    "start_monotonic_ns": index * 2,
                    "end_monotonic_ns": index * 2 + 1,
                    "result": 0,
                }
            )
        (directory / "squad.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in records)
        )
        (directory / "stderr.txt").write_text("")
        lock = directory / "boundary-lock.json"
        lock.write_text(json.dumps({
            "schema_version": 1,
            "kind": "bless-thor-tensorrt-safe-boundary-lock",
            "status": "frozen",
            "held_out_validation_required": True,
            "selected_switch_operation": 23,
            "safe_switch_operations": [0, 23, 47],
            "engine": {"sha256": MODULE.sha256(engine)},
        }) + "\n")
        return engine, lock

    def test_accepts_safe_midpoint_switch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            engine, lock = self.fixture(directory)
            summary = MODULE.verify(directory, engine, lock)
            self.assertEqual(summary["safe_switch_operation"], 23)
            self.assertFalse(summary["numeric_comparison_allowed"])

    def test_rejects_output_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            engine, lock = self.fixture(directory)
            result = json.loads((directory / "result.json").read_text())
            result["selected_output_matches"] = False
            (directory / "result.json").write_text(json.dumps(result) + "\n")
            with self.assertRaisesRegex(ValueError, "result differs"):
                MODULE.verify(directory, engine, lock)

    def test_rejects_wrong_switch_trace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            engine, lock = self.fixture(directory)
            lines = (directory / "squad.jsonl").read_text().splitlines()
            record = json.loads(lines[22])
            record["selected_sms"] = 8
            lines[22] = json.dumps(record)
            (directory / "squad.jsonl").write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "trace differs"):
                MODULE.verify(directory, engine, lock)


if __name__ == "__main__":
    unittest.main()
