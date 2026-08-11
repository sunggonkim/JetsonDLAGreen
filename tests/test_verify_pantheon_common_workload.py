import json
import hashlib
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.verify_pantheon_common_workload import UPSTREAM_COMMIT, verify


def row(index: int, *, correct: bool = True) -> dict:
    return {
        "schema_version": 1,
        "request_id": f"request-{index}",
        "arrival_sequence": index,
        "input_sha256": f"{index + 1:064x}",
        "expected_label": index % 10,
        "prediction": index % 10 if correct else (index + 1) % 10,
        "correct": correct,
        "selected_exit": 1,
        "block_sequence": [0, 1],
        "wall_latency_us": 100.0 + index,
        "deadline_us": 1000.0,
        "deadline_miss": False,
    }


def write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def source_kwargs(root: Path) -> dict:
    source = root / "pantheon_scheduler.cc"
    source.write_text("pinned Pantheon source\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "pantheon_scheduler.cc"], check=True)
    subprocess.run([
        "git", "-C", str(root), "-c", "user.email=test@example.invalid",
        "-c", "user.name=Test", "commit", "-q", "--no-gpg-sign",
        "-m", "unpromoting fixture",
    ], check=True)
    return {
        "upstream_source_path": source,
        "upstream_source_sha256": digest,
        "require_pinned_checkout": False,
    }


def training_kwargs(root: Path) -> dict:
    result = root / "pantheon-training.json"
    result.write_text(json.dumps({
        "kind": "pantheon-cifar10-resnet50-training",
        "system": "Pantheon",
        "upstream_commit": UPSTREAM_COMMIT,
        "formal_training_contract": True,
        "accuracy_gate_passed": True,
        "full_output_max_abs_error": 0.0,
        "source_sha256": {"offline/een/een.py": "a" * 64},
        "dataset_sha256": {"test_batch": "b" * 64},
        "artifacts": {"profile_sha256": "c" * 64},
    }) + "\n", encoding="utf-8")
    return {"training_result_path": result}


def common_workload_kwargs(root: Path) -> dict:
    arrival = root / "arrival.jsonl"
    dataset = root / "dataset.jsonl"
    arrival.write_text("arrival\n", encoding="utf-8")
    dataset.write_text("dataset\n", encoding="utf-8")
    contract = root / "common-workload.json"
    contract.write_text(json.dumps({
        "schema_version": 1,
        "workload_id": "p9-dependent-tensorrt-dag",
        "topology": "fixed-2g+1g",
        "placement": "fixed-1g-producer-2g-consumer",
        "input_tensor": "features",
        "payload_bytes": 14720,
        "arrival_trace_path": str(arrival),
        "arrival_trace_sha256": hashlib.sha256(arrival.read_bytes()).hexdigest(),
        "dataset_manifest_path": str(dataset),
        "dataset_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }) + "\n", encoding="utf-8")
    return {"common_workload_contract": contract, "require_common_workload": True}


class PantheonCommonWorkloadGateTest(unittest.TestCase):
    def test_common_workload_contract_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(0)])
            write_trace(pantheon, [row(0)])
            result = verify(
                reference, pantheon, deadline_us=1000.0,
                **source_kwargs(root), **training_kwargs(root),
                **common_workload_kwargs(root),
            )
            self.assertEqual(result["common_workload"]["workload_id"], "p9-dependent-tensorrt-dag")
            self.assertFalse(result["numeric_comparison_allowed"])

    def test_runtime_binary_is_rehashed_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(0)])
            write_trace(pantheon, [row(0)])
            binary = root / "pantheon-runtime"
            binary.write_bytes(b"runtime bytes\n")
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            source = source_kwargs(root)
            training = training_kwargs(root)
            workload = common_workload_kwargs(root)
            result = verify(
                reference, pantheon, deadline_us=1000.0,
                runtime_binary_path=binary, runtime_binary_sha256=digest,
                **source, **training, **workload,
            )
            self.assertEqual(result["runtime_binary_path"], str(binary.resolve()))
            self.assertEqual(result["runtime_binary_sha256"], digest)
            self.assertTrue(result["runtime_binary_verified"])
            self.assertFalse(result["numeric_comparison_allowed"])
            with self.assertRaisesRegex(ValueError, "runtime binary SHA256"):
                verify(
                    reference, pantheon, deadline_us=1000.0,
                    runtime_binary_path=binary, runtime_binary_sha256="a" * 64,
                    **source, **training, **workload,
                )

    def test_common_workload_contract_is_required_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(0)])
            write_trace(pantheon, [row(0)])
            with self.assertRaisesRegex(ValueError, "common workload"):
                verify(
                    reference, pantheon, deadline_us=1000.0,
                    **source_kwargs(root), **training_kwargs(root),
                    require_common_workload=True,
                )

    def test_shared_arrivals_and_accuracy_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(index) for index in range(4)])
            port_rows = [row(index) for index in range(4)]
            port_rows[0] = row(0, correct=False)
            write_trace(pantheon, port_rows)
            result = verify(reference, pantheon, expected_cases=4,
                            accuracy_tolerance=0.3, deadline_us=1000.0,
                            **source_kwargs(root), **training_kwargs(root))
            self.assertTrue(result["shared_arrival_trace"])
            self.assertTrue(result["accuracy_equivalent"])
            self.assertFalse(result["numeric_comparison_allowed"])
            self.assertEqual(result["reference_trace_path"], str(reference.resolve()))
            self.assertEqual(result["port_trace_path"], str(pantheon.resolve()))

    def test_input_or_label_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(0), row(1)])
            changed = row(1)
            changed["expected_label"] = 99
            changed["prediction"] = 99
            write_trace(pantheon, [row(0), changed])
            with self.assertRaisesRegex(ValueError, "shared workload mismatch"):
                verify(reference, pantheon, deadline_us=1000.0,
                       **source_kwargs(root), **training_kwargs(root))

    def test_accuracy_tolerance_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(index) for index in range(10)])
            write_trace(pantheon, [row(index, correct=False) for index in range(2)]
                        + [row(index) for index in range(2, 10)])
            with self.assertRaisesRegex(ValueError, "accuracy differs"):
                verify(reference, pantheon, accuracy_tolerance=0.01,
                       deadline_us=1000.0, **source_kwargs(root),
                       **training_kwargs(root))

    def test_non_dense_or_wrong_exit_trace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            first = row(0)
            second = row(1)
            second["arrival_sequence"] = 3
            write_trace(reference, [first, second])
            wrong = row(1)
            wrong["selected_exit"] = 0
            write_trace(pantheon, [first, wrong])
            with self.assertRaisesRegex(ValueError, "sequence"):
                verify(reference, pantheon, deadline_us=1000.0,
                       **source_kwargs(root), **training_kwargs(root))

    def test_source_digest_is_recomputed_from_bound_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(0)])
            write_trace(pantheon, [row(0)])
            source = root / "pantheon_scheduler.cc"
            source.write_text("actual bytes\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify(reference, pantheon, upstream_source_path=source,
                       upstream_source_sha256="a" * 64, deadline_us=1000.0,
                       require_pinned_checkout=False, **training_kwargs(root))

    def test_strict_mode_rejects_non_pinned_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(0)])
            write_trace(pantheon, [row(0)])
            kwargs = source_kwargs(root)
            kwargs["require_pinned_checkout"] = True
            with self.assertRaisesRegex(ValueError, "pinned Pantheon commit"):
                verify(reference, pantheon, deadline_us=1000.0,
                       **kwargs, **training_kwargs(root))

    def test_development_training_artifact_cannot_enter_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            pantheon = root / "pantheon.jsonl"
            write_trace(reference, [row(0)])
            write_trace(pantheon, [row(0)])
            source = root / "pantheon_scheduler.cc"
            source.write_text("source\n", encoding="utf-8")
            training = root / "training.json"
            training.write_text(json.dumps({
                "kind": "pantheon-cifar10-resnet50-training",
                "system": "Pantheon", "upstream_commit": UPSTREAM_COMMIT,
                "formal_training_contract": False,
                "accuracy_gate_passed": False,
                "full_output_max_abs_error": 0.0,
                "source_sha256": {"x": "a" * 64},
                "dataset_sha256": {"x": "b" * 64},
                "artifacts": {"x": "c" * 64},
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "formal accuracy gate"):
                verify(reference, pantheon, upstream_source_path=source,
                       training_result_path=training, deadline_us=1000.0,
                       require_pinned_checkout=False)


if __name__ == "__main__":
    unittest.main()
