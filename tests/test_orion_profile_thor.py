#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orion_profile_thor", ROOT / "baselines/orion/profile_thor.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OrionProfileThorTest(unittest.TestCase):
    @staticmethod
    def row(index: int, position: int, duration: float) -> dict:
        grid = [8 if position == 0 else 4, 1, 1]
        return {
            "schema_version": 1,
            "client_id": 1,
            "operation_index": index,
            "api": "cuLaunchKernelEx",
            "grid": grid,
            "block": [128, 1, 1],
            "grid_blocks": grid[0],
            "block_threads": 128,
            "shared_mem_bytes": 0,
            "active_blocks_per_sm": 2,
            "device_sms": 8,
            "estimated_sms": 4 if position == 0 else 2,
            "kernel_duration_us": duration,
        }

    def fixture(self, root: Path, durations: tuple[float, float]) -> dict:
        result = {
            "kind": "orion-thor-operation-profile-raw",
            "upstream_commit": MODULE.UPSTREAM_COMMIT,
            "model": "whisper-tiny-encoder",
            "samples": 2,
            "warmup": 1,
            "numeric_comparison_allowed": False,
            "client": {
                "completed_requests": 2,
                "gpu": {"multiprocessors": 8},
            },
        }
        result_path = root / "result.json"
        trace_path = root / "trace.jsonl"
        result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        rows = []
        for request in range(3):
            for position in range(2):
                rows.append(
                    self.row(
                        request * 2 + position,
                        position,
                        durations[position] + (0.1 if request == 2 else 0.0),
                    )
                )
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return MODULE.replay_mode(
            result_path,
            trace_path,
            model="whisper-tiny-encoder",
            samples=2,
            warmup=1,
        )

    def test_replays_constant_operation_sequence_and_classifies_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replays = {}
            for mode, durations in {
                "isolated": (10.0, 20.0),
                "compute": (15.0, 21.0),
                "memory": (10.5, 30.0),
            }.items():
                child = root / mode
                child.mkdir()
                replays[mode] = self.fixture(child, durations)
            profiles = MODULE.classify_profiles(replays)
        self.assertEqual([row["resource_class"] for row in profiles], ["compute", "memory"])
        self.assertEqual([row["profile"] for row in profiles], [1, 0])
        self.assertEqual([row["sm_used"] for row in profiles], [4, 2])

    def test_rejects_changed_operation_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = self.fixture(root, (10.0, 20.0))
            replays = {mode: json.loads(json.dumps(replay)) for mode in MODULE.MODES}
            replays["memory"]["positions"][0]["signature"]["grid"] = [99, 1, 1]
            with self.assertRaisesRegex(ValueError, "signature differs"):
                MODULE.classify_profiles(replays)

    def test_scheduler_profile_is_exact_and_ascii(self) -> None:
        profiles = [
            {"position": 0, "profile": 1, "sm_used": 8, "duration_us": 12.25},
            {"position": 1, "profile": 0, "sm_used": 3, "duration_us": 4.5},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.tsv"
            path.write_text(
                "orion-thor-profile-v1\n"
                "position\tapi\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\t"
                "block_z\tshared_mem_bytes\tprofile\tsm_used\tduration_us\n"
                + "".join(
                    f"{row['position']}\tcuLaunchKernelEx\t1\t1\t1\t32\t1\t1\t"
                    f"0\t{row['profile']}\t{row['sm_used']}\t"
                    f"{row['duration_us']:.9g}\n"
                    for row in profiles
                ),
                encoding="ascii",
            )
            self.assertEqual(
                path.read_bytes(),
                b"orion-thor-profile-v1\n"
                b"position\tapi\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\t"
                b"block_z\tshared_mem_bytes\tprofile\tsm_used\tduration_us\n"
                b"0\tcuLaunchKernelEx\t1\t1\t1\t32\t1\t1\t0\t1\t8\t12.25\n"
                b"1\tcuLaunchKernelEx\t1\t1\t1\t32\t1\t1\t0\t0\t3\t4.5\n",
            )


if __name__ == "__main__":
    unittest.main()
