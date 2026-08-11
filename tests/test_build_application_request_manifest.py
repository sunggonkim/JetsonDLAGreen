import importlib.util
import hashlib
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_application_request_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_request_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PREDICTION_SCRIPT = ROOT / "analysis" / "build_application_prediction_trace.py"
PREDICTION_SPEC = importlib.util.spec_from_file_location(
    "build_prediction_trace_for_manifest_contract", PREDICTION_SCRIPT
)
assert PREDICTION_SPEC is not None and PREDICTION_SPEC.loader is not None
PREDICTION_MODULE = importlib.util.module_from_spec(PREDICTION_SPEC)
import sys
sys.modules[PREDICTION_SPEC.name] = PREDICTION_MODULE
PREDICTION_SPEC.loader.exec_module(PREDICTION_MODULE)


class BuildApplicationRequestManifestTest(unittest.TestCase):
    def dataset(self, root: pathlib.Path) -> pathlib.Path:
        path = root / "dataset.jsonl"
        rows = [
            {"schema_version": 1, "sample_id": "a", "input_sha256": "a" * 64, "expected_label": "cat"},
            {"schema_version": 1, "sample_id": "b", "input_sha256": "b" * 64, "expected_label": "dog"},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_expands_warmup_and_preserves_external_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.dataset(pathlib.Path(directory))
            rows, provenance = MODULE.build(dataset, warmup=10, request_id_prefix="r")
        self.assertEqual([row["iteration"] for row in rows], [10, 11])
        self.assertEqual([row["arrival_sequence"] for row in rows], [0, 1])
        self.assertEqual([row["request_id"] for row in rows], ["r-000000", "r-000001"])
        self.assertEqual([row["expected_label"] for row in rows], ["cat", "dog"])
        self.assertEqual(provenance["sample_reuse"], False)

    def test_rejects_implicit_sample_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.dataset(pathlib.Path(directory))
            with self.assertRaisesRegex(ValueError, "exceed labelled dataset"):
                MODULE.build(dataset, warmup=0, requests=3)

    def test_allows_explicit_duplicate_bytes_with_same_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "dataset.jsonl"
            digest = "a" * 64
            path.write_text(
                "".join(json.dumps({
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "input_sha256": digest,
                    "expected_label": "cat",
                }) + "\n" for sample_id in ("a", "b")),
                encoding="utf-8",
            )
            rows, _ = MODULE.build(path, warmup=0)
        self.assertEqual([row["input_sha256"] for row in rows], [digest, digest])

    def test_rejects_duplicate_bytes_with_conflicting_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "dataset.jsonl"
            digest = "a" * 64
            path.write_text(
                "".join(json.dumps({
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "input_sha256": digest,
                    "expected_label": label,
                }) + "\n" for sample_id, label in (("a", "cat"), ("b", "dog"))),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                MODULE.build(path, warmup=0)

    def test_dataset_digest_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.dataset(pathlib.Path(directory))
            _, provenance = MODULE.build(dataset, warmup=0)
            self.assertEqual(
                provenance["dataset_manifest"]["sha256"],
                hashlib.sha256(dataset.read_bytes()).hexdigest(),
            )

    def test_output_is_accepted_by_prediction_trace_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            dataset = self.dataset(root)
            rows, _ = MODULE.build(dataset, warmup=10, request_id_prefix="head")
            request_manifest = root / "requests.jsonl"
            request_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            parsed = PREDICTION_MODULE._read_request_manifest(request_manifest)
        self.assertEqual(sorted(parsed), [10, 11])
        self.assertEqual(parsed[10]["request_id"], "head-000000")
        self.assertEqual(parsed[11]["expected_label"], "dog")

    def test_sample_list_binds_measured_rows_after_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            dataset = self.dataset(root)
            sample_list = root / "samples.jsonl"
            sample_rows = [
                {"iteration": 0, "sample_id": "warm", "path": "/tmp/warm", "input_sha256": "a" * 64},
                {"iteration": 1, "sample_id": "measured-a", "path": "/tmp/a", "input_sha256": "a" * 64},
                {"iteration": 2, "sample_id": "measured-b", "path": "/tmp/b", "input_sha256": "b" * 64},
            ]
            sample_list.write_text(
                "".join(json.dumps(row) + "\n" for row in sample_rows),
                encoding="utf-8",
            )
            rows, provenance = MODULE.build(
                dataset, warmup=1, requests=2, sample_list=sample_list,
            )
        self.assertEqual([row["iteration"] for row in rows], [1, 2])
        self.assertEqual([row["input_sha256"] for row in rows], ["a" * 64, "b" * 64])
        self.assertEqual(provenance["producer_sample_list"]["warmup_records_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
