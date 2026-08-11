from __future__ import annotations

import csv
import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "baselines/xsched/verify_resnet_control_smoke.py"
SPEC = importlib.util.spec_from_file_location("verify_xsched_resnet", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class XSchedResnetControlSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        (ROOT / "results/p9-xsched-resnet-control-100r-250rps-20260809T141021Z/result.json").is_file(),
        "requires preserved local XSched hardware evidence",
    )
    def test_replays_hardware_smoke(self) -> None:
        directory = (
            ROOT
            / "results/p9-xsched-resnet-control-100r-250rps-20260809T141021Z"
        )
        result = MODULE.verify(
            directory / "result.json", directory / "pipeline.csv",
            directory / "checksums.csv", directory / "be.json",
            directory / "be.csv", directory / "server.log",
            [directory / "be.log", directory / "result.log"],
            directory / "provenance/jdg-mig-trt-pipeline",
            directory / "provenance/resnet10-detection.engine",
            directory / "provenance/distilbert-sst2.engine",
            directory / "provenance/xserver",
            directory / "provenance/libshimcuda.so",
            directory / "provenance/thor-cuda13-tensorrt.patch",
            directory / "xsched-commit.txt", 760.0,
        )
        self.assertEqual(result["misses"], 100)
        self.assertEqual(result["scheduler"]["be_suspend_transitions"], 4)
        self.assertFalse(result["token_only"])
        self.assertIsNone(result["common_workload"])

    def test_numeric_promotion_requires_common_workload_contract(self) -> None:
        directory = (
            ROOT
            / "results/p9-xsched-resnet-control-100r-250rps-20260809T141021Z"
        )
        with self.assertRaisesRegex(ValueError, "common workload"):
            MODULE.verify(
                directory / "result.json", directory / "pipeline.csv",
                directory / "checksums.csv", directory / "be.json",
                directory / "be.csv", directory / "server.log",
                [directory / "be.log", directory / "result.log"],
                directory / "provenance/jdg-mig-trt-pipeline",
                directory / "provenance/resnet10-detection.engine",
                directory / "provenance/distilbert-sst2.engine",
                directory / "provenance/xserver",
                directory / "provenance/libshimcuda.so",
                directory / "provenance/thor-cuda13-tensorrt.patch",
                directory / "xsched-commit.txt", 760.0,
                require_common_workload=True,
            )

    @unittest.skipUnless(
        (ROOT / "results/p9-xsched-resnet-control-100r-250rps-20260809T141021Z/result.json").is_file(),
        "requires preserved local XSched hardware evidence",
    )
    def test_valid_common_workload_is_rehashed_and_emitted(self) -> None:
        directory = (
            ROOT
            / "results/p9-xsched-resnet-control-100r-250rps-20260809T141021Z"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrival = root / "arrival.jsonl"
            dataset = root / "dataset.jsonl"
            arrival.write_text("{}\n", encoding="utf-8")
            dataset.write_text("{}\n", encoding="utf-8")
            contract = root / "common-workload.json"
            contract.write_text(json.dumps({
                "schema_version": 1,
                "workload_id": "resnet-detection-head",
                "topology": "fixed-2g+1g",
                "placement": "fixed-1g-producer-2g-consumer",
                "input_tensor": "Layer6_relu_Y",
                "payload_bytes": 1_884_160,
                "arrival_trace_path": str(arrival),
                "arrival_trace_sha256": hashlib.sha256(arrival.read_bytes()).hexdigest(),
                "dataset_manifest_path": str(dataset),
                "dataset_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }, indent=2) + "\n", encoding="utf-8")
            result = MODULE.verify(
                directory / "result.json", directory / "pipeline.csv",
                directory / "checksums.csv", directory / "be.json",
                directory / "be.csv", directory / "server.log",
                [directory / "be.log", directory / "result.log"],
                directory / "provenance/jdg-mig-trt-pipeline",
                directory / "provenance/resnet10-detection.engine",
                directory / "provenance/distilbert-sst2.engine",
                directory / "provenance/xserver",
                directory / "provenance/libshimcuda.so",
                directory / "provenance/thor-cuda13-tensorrt.patch",
                directory / "xsched-commit.txt", 760.0,
                common_workload_contract=contract,
                require_common_workload=True,
            )
        self.assertEqual(result["common_workload"]["workload_id"], "resnet-detection-head")
        self.assertEqual(len(result["common_workload"]["contract_sha256"]), 64)

    def test_checksum_replay_rejects_token_only_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "checksums.csv"
            with trace.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.writer(destination)
                writer.writerow(("request", "payload_checksum", "output_checksum"))
                writer.writerow((10, 1, 2))
                writer.writerow((11, 1, 2))
            result = {
                "iterations": 2,
                "warmup": 10,
                "unique_payload_checksums": 1,
                "unique_policy_output_checksums": 1,
            }
            replay = MODULE.replay_checksums(trace, result)
        self.assertEqual(replay, {"payloads": 1, "outputs": 1})
        self.assertLess(replay["payloads"], 2)

    def test_scheduler_requires_native_suspend_resume(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
