import json
import hashlib
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.verify_orion_differential_gate import UPSTREAM_COMMIT, verify


def row(index: int, *, reordered: bool = False, admitted: bool = True) -> dict:
    return {
        "schema_version": 1,
        "case_id": f"case-{index}",
        "arrival_sequence": index,
        "decision_sequence": index,
        "client_id": index % 2,
        "priority": "high" if index % 2 else "best-effort",
        "api": "cuLaunchKernelEx",
        "profile_position": index % 3,
        "sm_used": 4,
        "duration_us": 12.5,
        "admitted": admitted,
        "reordered": reordered,
        "admission_reason": "profiled" if admitted else "blocked",
    }


def write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def source_kwargs(root: Path) -> dict:
    source = root / "upstream_scheduler.cc"
    source.write_text("pinned upstream scheduler source\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "upstream_scheduler.cc"], check=True)
    subprocess.run([
        "git", "-C", str(root), "-c", "user.email=test@example.invalid",
        "-c", "user.name=Test", "commit", "-q", "--no-gpg-sign",
        "-m", "pinned upstream",
    ], check=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "reference_source_path": source,
        "reference_source_sha256": digest,
        "require_pinned_checkout": False,
    }


class OrionDifferentialGateTest(unittest.TestCase):
    def test_common_workload_contract_is_bound_before_numeric_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            write_trace(reference, [row(0)])
            write_trace(port, [row(0)])
            arrival = root / "arrival.jsonl"
            dataset = root / "dataset.jsonl"
            arrival.write_text("arrival\n", encoding="utf-8")
            dataset.write_text("dataset\n", encoding="utf-8")
            contract = root / "common.json"
            contract.write_text(json.dumps({
                "schema_version": 1,
                "workload_id": "resnet-control",
                "topology": "fixed-2g+1g",
                "placement": "fixed-1g-producer-2g-consumer",
                "input_tensor": "features",
                "payload_bytes": 14720,
                "arrival_trace_path": str(arrival),
                "arrival_trace_sha256": hashlib.sha256(arrival.read_bytes()).hexdigest(),
                "dataset_manifest_path": str(dataset),
                "dataset_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }) + "\n", encoding="utf-8")
            kwargs = source_kwargs(root)
            result = verify(
                reference, port, **kwargs,
                common_workload_contract=contract,
                require_common_workload=True,
            )
            self.assertEqual(result["common_workload"]["workload_id"], "resnet-control")
            self.assertFalse(result["numeric_comparison_allowed"])
            strict = dict(kwargs)
            strict["require_pinned_checkout"] = True
            with self.assertRaisesRegex(ValueError, "trace provenance"):
                verify(
                    reference, port, **strict,
                    common_workload_contract=contract,
                    require_common_workload=True,
                )

    def test_upstream_runtime_binary_is_bound_by_trace_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            write_trace(reference, [row(0)])
            write_trace(port, [row(0)])
            arrival = root / "arrival.jsonl"
            dataset = root / "dataset.jsonl"
            arrival.write_text("arrival\n", encoding="utf-8")
            dataset.write_text("dataset\n", encoding="utf-8")
            contract = root / "common.json"
            contract.write_text(json.dumps({
                "schema_version": 1,
                "workload_id": "resnet-control",
                "topology": "fixed-2g+1g",
                "placement": "fixed-1g-producer-2g-consumer",
                "input_tensor": "features",
                "payload_bytes": 14720,
                "arrival_trace_path": str(arrival),
                "arrival_trace_sha256": hashlib.sha256(arrival.read_bytes()).hexdigest(),
                "dataset_manifest_path": str(dataset),
                "dataset_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }) + "\n", encoding="utf-8")
            binary = root / "scheduler_eval.so"
            binary.write_bytes(b"pinned runtime binary\n")
            source = root / "upstream_scheduler.cc"
            source.write_text("pinned upstream scheduler source\n", encoding="utf-8")
            source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            binary_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            trace_digest = hashlib.sha256(reference.read_bytes()).hexdigest()
            contract_digest = hashlib.sha256(contract.read_bytes()).hexdigest()
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps({
                "schema_version": 1,
                "kind": "orion-upstream-trace-provenance",
                "upstream_commit": UPSTREAM_COMMIT,
                "generator": "pinned-upstream-orion-runtime",
                "reference_trace_path": str(reference),
                "reference_trace_sha256": trace_digest,
                "reference_source_path": str(source),
                "reference_source_sha256": source_digest,
                "upstream_runtime_binary_path": str(binary),
                "upstream_runtime_binary_sha256": binary_digest,
                "common_workload_path": str(contract),
                "common_workload_sha256": contract_digest,
            }) + "\n", encoding="utf-8")
            result = verify(
                reference,
                port,
                reference_source_path=source,
                upstream_runtime_binary_path=binary,
                common_workload_contract=contract,
                require_common_workload=True,
                require_pinned_checkout=False,
                reference_trace_provenance=provenance,
            )
            self.assertEqual(result["upstream_runtime_binary_sha256"], binary_digest)
            self.assertFalse(result["numeric_comparison_allowed"])
            provenance.write_text(
                provenance.read_text(encoding="utf-8").replace(
                    "upstream_runtime_binary_path", "missing_runtime_binary_path", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "paths are missing"):
                verify(
                    reference,
                    port,
                    reference_source_path=source,
                    upstream_runtime_binary_path=binary,
                    common_workload_contract=contract,
                    require_common_workload=True,
                    require_pinned_checkout=False,
                    reference_trace_provenance=provenance,
                )

    def test_common_workload_is_required_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            write_trace(reference, [row(0)])
            write_trace(port, [row(0)])
            with self.assertRaisesRegex(ValueError, "common workload"):
                verify(
                    reference, port, **source_kwargs(root),
                    require_common_workload=True,
                )

    def test_exact_canonical_decisions_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            rows = [row(index, reordered=index == 1) for index in range(3)]
            write_trace(reference, rows)
            write_trace(port, rows)
            result = verify(
                reference, port, **source_kwargs(root),
                expected_cases=3,
            )
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["numeric_comparison_allowed"])
            self.assertEqual(result["mismatch_cases"], 0)
            self.assertEqual(result["reference_trace_path"], str(reference.resolve()))
            self.assertEqual(result["port_trace_path"], str(port.resolve()))

    def test_admission_or_order_difference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            write_trace(reference, [row(0), row(1, reordered=True)])
            write_trace(port, [row(0), row(1, reordered=False)])
            with self.assertRaisesRegex(ValueError, "decision mismatch"):
                verify(reference, port, **source_kwargs(root))

    def test_non_dense_or_duplicate_trace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            first = row(0)
            second = row(1)
            second["arrival_sequence"] = 0
            write_trace(reference, [first, second])
            write_trace(port, [first, row(1)])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                verify(reference, port, **source_kwargs(root))

    def test_only_pinned_commit_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            write_trace(reference, [row(0)])
            write_trace(port, [row(0)])
            with self.assertRaisesRegex(ValueError, "pinned Orion"):
                verify(
                    reference, port, upstream_commit="0" * 40,
                    **source_kwargs(root),
                )
        self.assertTrue(UPSTREAM_COMMIT)

    def test_unpinned_checkout_is_non_promoting_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            write_trace(reference, [row(0)])
            write_trace(port, [row(0)])
            result = verify(reference, port, **source_kwargs(root))
        self.assertFalse(result["reference_checkout_verified"])
        self.assertFalse(result["numeric_comparison_allowed"])

    def test_strict_mode_rejects_non_pinned_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            write_trace(reference, [row(0)])
            write_trace(port, [row(0)])
            kwargs = source_kwargs(root)
            kwargs["require_pinned_checkout"] = True
            with self.assertRaisesRegex(ValueError, "pinned Orion commit"):
                verify(reference, port, **kwargs)

    def test_source_digest_is_recomputed_from_bound_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            port = root / "port.jsonl"
            write_trace(reference, [row(0)])
            write_trace(port, [row(0)])
            source = root / "upstream_scheduler.cc"
            source.write_text("actual bytes\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify(
                    reference, port, reference_source_path=source,
                    reference_source_sha256="a" * 64,
                    require_pinned_checkout=False,
                )


if __name__ == "__main__":
    unittest.main()
