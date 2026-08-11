import importlib.util
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_common_workload_contract", ROOT / "scripts" / "build_common_workload_contract.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CommonWorkloadContractTest(unittest.TestCase):
    def _files(self, root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        dataset = root / "dataset.jsonl"
        dataset.write_text(
            json.dumps({
                "schema_version": 1, "sample_id": "s0",
                "input_sha256": "a" * 64, "expected_label": "cat",
            }) + "\n",
            encoding="utf-8",
        )
        arrival = root / "arrival.jsonl"
        arrival.write_text(
            json.dumps({
                "schema_version": 1, "iteration": 5, "request_id": "r0",
                "arrival_sequence": 0, "input_sha256": "a" * 64,
                "expected_label": "cat",
            }) + "\n",
            encoding="utf-8",
        )
        return arrival, dataset

    def test_build_binds_request_and_dataset_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arrival, dataset = self._files(pathlib.Path(directory))
            result = MODULE.build(
                workload_id="resnet-detection-head",
                topology="fixed-2g+1g",
                placement="fixed-1g-producer-2g-consumer",
                input_tensor="Layer6_relu_Y",
                payload_bytes=1_884_160,
                arrival_trace=arrival,
                dataset_manifest=dataset,
            )
            self.assertEqual(result["request_count"], 1)
            self.assertEqual(result["arrival_trace_sha256"], hashlib.sha256(arrival.read_bytes()).hexdigest())

    def test_rejects_unbound_label_and_non_dense_arrival(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            arrival, dataset = self._files(root)
            value = json.loads(arrival.read_text())
            value["arrival_sequence"] = 1
            arrival.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dense"):
                MODULE.build(
                    workload_id="resnet-detection-head",
                    topology="fixed-2g+1g",
                    placement="fixed-1g-producer-2g-consumer",
                    input_tensor="Layer6_relu_Y",
                    payload_bytes=1_884_160,
                    arrival_trace=arrival,
                    dataset_manifest=dataset,
                )

    def test_binds_producer_input_trace_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            arrival, dataset = self._files(root)
            trace = root / "inputs.bin"
            trace.write_bytes(b"JDGINT1-test-bytes\n")
            result = MODULE.build(
                workload_id="resnet50-classification",
                topology="fixed-2g+1g",
                placement="fixed-1g-producer-2g-consumer",
                input_tensor="gpu_0/res4_5_branch2c_bn_2",
                payload_bytes=802816,
                arrival_trace=arrival,
                dataset_manifest=dataset,
                producer_input_trace=trace,
            )
            self.assertEqual(result["producer_input_trace_path"], str(trace.resolve()))
            self.assertEqual(
                result["producer_input_trace_sha256"],
                hashlib.sha256(trace.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
