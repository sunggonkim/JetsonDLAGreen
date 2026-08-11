#!/usr/bin/env python3

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_pipeline_deadline",
    ROOT / "analysis" / "freeze_p9_pipeline_deadline.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreezePipelineDeadlineTest(unittest.TestCase):
    def test_verify_artifacts_rehashes_lock_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "binary"
            artifact.write_bytes(b"version-a")
            lock = {"artifacts": {"binary": {
                "path": str(artifact), "sha256": digest(artifact),
            }}}
            MODULE.verify_artifacts(lock)
            artifact.write_bytes(b"version-b")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                MODULE.verify_artifacts(lock)

    def test_verify_artifacts_rejects_missing_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks artifact provenance"):
            MODULE.verify_artifacts({"artifacts": {}})

    def fixture(self, root: Path) -> Path:
        blocks = []
        for index, values in enumerate(([100.0, 110.0, 120.0], [105.0, 115.0, 125.0])):
            directory = root / f"block-{index:02d}"
            directory.mkdir()
            trace = directory / "pipeline.csv"
            with trace.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MODULE.TRACE_COLUMNS)
                writer.writeheader()
                for request, value in enumerate(values):
                    row = {name: 0 for name in MODULE.TRACE_COLUMNS}
                    row.update(
                        request=request,
                        validation_excluded_end_to_end_us=value,
                        wall_end_to_end_us=value + 10,
                    )
                    writer.writerow(row)
            result = directory / "pipeline.json"
            result.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "pipeline": "whisper-last-hidden-state-to-projection-mlp",
                        "transport": "registered-shared-sysmem-direct-binding",
                        "payload_bytes": 2304000,
                        "iterations": 3,
                        "producer_quota": 100,
                        "checksum_failures": 0,
                        "deadline_us": None,
                        "deadline_misses": 0,
                        "deadline_mode": "validation-excluded",
                        "stage_latency_us": {
                            "validation_excluded_end_to_end_p99": MODULE.percentile(list(values), 0.99)
                        },
                    }
                ),
                encoding="utf-8",
            )
            blocks.append(
                {
                    "index": index,
                    "result_path": str(result.relative_to(root)),
                    "result_sha256": digest(result),
                    "trace_path": str(trace.relative_to(root)),
                    "trace_sha256": digest(trace),
                }
            )
        summary = root / "calibration.json"
        summary.write_text(
            json.dumps(
                {
                    "kind": "p9-dependent-pipeline-deadline-calibration",
                    "config": {
                        "workload": "whisper-projection",
                        "payload_bytes": 2304000,
                        "transport": "registered-shared-sysmem-direct-binding",
                        "deadline_mode": "validation-excluded",
                        "blocks": 2,
                        "samples_per_block": 3,
                        "slo_factor": 1.10,
                        "producer_quota_percent": 100,
                        "producer_uuid": "small",
                        "consumer_uuid": "big",
                    },
                    "artifacts": {},
                    "blocks": blocks,
                }
            ),
            encoding="utf-8",
        )
        return summary

    def test_builds_replayable_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = MODULE.build_lock(self.fixture(Path(directory)))
        self.assertEqual(lock["pooled_samples"], 6)
        self.assertAlmostEqual(lock["deadline_us"], lock["pooled_p99_us"] * 1.10)

    def test_rejects_trace_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.fixture(root)
            with (root / "block-00/pipeline.csv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                MODULE.build_lock(summary)

    def test_builds_resnet_wall_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = self.fixture(root)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["config"].update(
                workload="resnet-control", payload_bytes=14_720,
                deadline_mode="wall",
            )
            for block in summary["blocks"]:
                result_path = root / block["result_path"]
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result.update(
                    pipeline="resnet10-layer7-cov-to-control-mlp",
                    payload_bytes=14_720,
                    deadline_mode="wall",
                )
                values = []
                with (root / block["trace_path"]).open(
                    newline="", encoding="utf-8"
                ) as source:
                    values = [
                        float(row["wall_end_to_end_us"])
                        for row in csv.DictReader(source)
                    ]
                result["end_to_end_us"] = {
                    "p99": MODULE.percentile(values, 0.99)
                }
                result_path.write_text(json.dumps(result), encoding="utf-8")
                block["result_sha256"] = digest(result_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            lock = MODULE.build_lock(summary_path)
        self.assertEqual(lock["contract"]["workload"], "resnet-control")
        self.assertEqual(lock["contract"]["deadline_mode"], "wall")

    def test_builds_learned_resnet_head_wall_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = self.fixture(root)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["config"].update(
                workload="resnet-detection-head", payload_bytes=1_884_160,
                deadline_mode="wall",
            )
            for block in summary["blocks"]:
                result_path = root / block["result_path"]
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result.update(
                    pipeline="resnet10-backbone-to-learned-detection-head",
                    payload_bytes=1_884_160,
                    deadline_mode="wall",
                )
                with (root / block["trace_path"]).open(
                    newline="", encoding="utf-8"
                ) as source:
                    values = [
                        float(row["wall_end_to_end_us"])
                        for row in csv.DictReader(source)
                    ]
                result["end_to_end_us"] = {
                    "p99": MODULE.percentile(values, 0.99)
                }
                result.pop("stage_latency_us", None)
                result_path.write_text(json.dumps(result), encoding="utf-8")
                block["result_sha256"] = digest(result_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            lock = MODULE.build_lock(summary_path)
        self.assertEqual(lock["contract"]["workload"], "resnet-detection-head")
        self.assertEqual(lock["contract"]["payload_bytes"], 1_884_160)

    def test_builds_whisper_wall_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = self.fixture(root)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["config"]["deadline_mode"] = "wall"
            for block in summary["blocks"]:
                result_path = root / block["result_path"]
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["deadline_mode"] = "wall"
                with (root / block["trace_path"]).open(
                    newline="", encoding="utf-8"
                ) as source:
                    values = [
                        float(row["wall_end_to_end_us"])
                        for row in csv.DictReader(source)
                    ]
                result["end_to_end_us"] = {"p99": MODULE.percentile(values, 0.99)}
                result.pop("stage_latency_us", None)
                result_path.write_text(json.dumps(result), encoding="utf-8")
                block["result_sha256"] = digest(result_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            lock = MODULE.build_lock(summary_path)
        self.assertEqual(lock["contract"]["workload"], "whisper-projection")
        self.assertEqual(lock["contract"]["deadline_mode"], "wall")

    def test_rejects_placement_profile_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = self.fixture(root)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["config"].update(
                placement_variant="fixed-2g-producer-1g-consumer",
                producer_profile="mig-1g-q100",
                consumer_profile="mig-2g-q100",
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid pipeline calibration contract"):
                MODULE.build_lock(summary_path)


if __name__ == "__main__":
    unittest.main()
