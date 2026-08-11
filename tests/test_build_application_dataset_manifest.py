import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_application_dataset_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_dataset_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildApplicationDatasetManifestTest(unittest.TestCase):
    def test_hashes_selected_files_and_external_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "a.bin").write_bytes(b"alpha")
            (root / "b.bin").write_bytes(b"beta")
            labels = root / "labels.json"
            labels.write_text(json.dumps({"a.bin": "class-a", "b.bin": "class-b"}) + "\n")
            rows, provenance = MODULE.build(root, labels, pattern="*.bin")
        self.assertEqual([row["sample_id"] for row in rows], ["a.bin", "b.bin"])
        self.assertEqual([row["expected_label"] for row in rows], ["class-a", "class-b"])
        self.assertEqual(provenance["label_source"], "external-dataset-owner-map")
        self.assertFalse(provenance["automatic_filename_labels"])

    def test_rejects_missing_or_unused_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "a.bin").write_bytes(b"alpha")
            labels = root / "labels.json"
            labels.write_text(json.dumps({"other.bin": "class-a"}) + "\n")
            with self.assertRaisesRegex(ValueError, "lacks selected sample"):
                MODULE.build(root, labels, pattern="*.bin")


if __name__ == "__main__":
    unittest.main()
