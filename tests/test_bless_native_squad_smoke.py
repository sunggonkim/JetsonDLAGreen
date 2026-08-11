#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_bless_native_squad",
    ROOT / "baselines/bless/verify_native_squad_smoke.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BlessNativeSquadSmokeTest(unittest.TestCase):
    def make_result(self, directory: Path) -> None:
        result = {
            "schema_version": 1,
            "kind": "bless-thor-native-squad-smoke",
            "algorithm": "relative-progress-kernel-squads",
            "maximum_squad_kernels": 6,
            "restricted_fraction": 0.5,
            "affinity_domain_sms": [2, 4, 6, 8],
            "requests": 2,
            "kernels_per_request": 12,
            "squads": 4,
            "checksums": [11, 22],
            "expected_checksums": [11, 22],
            "status": "passed",
        }
        records = [
            (6, "workload-equivalence", [8, 8], [6, 0]),
            (6, "workload-equivalence", [8, 8], [12, 0]),
            (6, "interference-free", [2, 6], [12, 6]),
            (6, "interference-free", [2, 6], [12, 12]),
        ]
        (directory / "native-squad.json").write_text(
            json.dumps(result) + "\n", encoding="utf-8"
        )
        with (directory / "native-squad.jsonl").open("w", encoding="utf-8") as trace:
            for sequence, (count, estimator, shares, cursor) in enumerate(records):
                trace.write(json.dumps({
                    "schema_version": 1,
                    "sequence": sequence,
                    "kernel_count": count,
                    "estimator": estimator,
                    "shares": shares,
                    "predicted_us": 1.0,
                    "cursor": cursor,
                }) + "\n")
        (directory / "native-squad.stderr").write_text("", encoding="utf-8")

    def test_accepts_complete_native_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_result(directory)
            result = MODULE.verify(directory)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["kernels"], 24)
        self.assertEqual(len(result["evidence_sha256"]), 3)

    def test_rejects_cursor_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_result(directory)
            records = [json.loads(line) for line in (
                directory / "native-squad.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            records[1]["cursor"] = [11, 0]
            (directory / "native-squad.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "trace record"):
                MODULE.verify(directory)


if __name__ == "__main__":
    unittest.main()
