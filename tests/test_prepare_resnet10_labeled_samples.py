import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_resnet10_labeled_samples.py"
SPEC = importlib.util.spec_from_file_location("prepare_resnet10_labeled_samples", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareResnet10SamplesTest(unittest.TestCase):
    def _fixture(self, *, category_id=3):
        root = Path(tempfile.mkdtemp())
        images = root / "images"
        images.mkdir()
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[:, :, 0] = 10
        image[:, :, 1] = 20
        image[:, :, 2] = 30
        self.assertTrue(cv2.imwrite(str(images / "frame.jpg"), image))
        annotations = root / "annotations.json"
        annotations.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "frame.jpg", "width": 20, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": category_id, "bbox": [2, 1, 10, 5]}],
        }) + "\n")
        category_map = root / "category-map.json"
        category_map.write_text(json.dumps({str(category_id): "Car"}) + "\n")
        return root, images, annotations, category_map

    def test_preprocesses_tensor_and_binds_external_detection_label(self):
        root, images, annotations, category_map = self._fixture()
        result = MODULE.prepare(images, annotations, category_map, root / "out")
        self.assertEqual(len(result["samples"]), 1)
        sample = json.loads((root / "out" / "samples.jsonl").read_text())
        dataset = json.loads((root / "out" / "dataset-manifest.jsonl").read_text())
        self.assertEqual(sample["iteration"], 0)
        self.assertEqual(dataset["expected_label"], '{"detections":[{"box":[64,36,320,184],"class":"Car"}]}')
        tensor = np.fromfile(root / "out" / "tensors" / "000000.f32", dtype=np.float32)
        self.assertEqual(tensor.size, 3 * 368 * 640)
        # Source BGR [10,20,30] becomes model RGB [30,20,10], scaled by 1/255.
        self.assertAlmostEqual(float(tensor[0]), 30.0 / 255.0, places=6)
        self.assertAlmostEqual(float(tensor[368 * 640]), 20.0 / 255.0, places=6)
        self.assertTrue(result["labels_external"])
        self.assertFalse(result["filename_labels_inferred"])

    def test_rejects_unmapped_annotation_category(self):
        root, images, annotations, category_map = self._fixture(category_id=7)
        category_map.write_text(json.dumps({"3": "Car"}) + "\n")
        with self.assertRaisesRegex(ValueError, "absent from the explicit category map"):
            MODULE.prepare(images, annotations, category_map, root / "out")

    def test_rejects_background_label_not_emitted_by_vendor_parser(self):
        root, images, annotations, category_map = self._fixture()
        category_map.write_text(json.dumps({"3": "Person"}) + "\n")
        with self.assertRaisesRegex(ValueError, "not emitted by the vendor ResNet10 parser"):
            MODULE.prepare(images, annotations, category_map, root / "out")

    def test_allows_only_explicitly_ignored_categories(self):
        root, images, annotations, category_map = self._fixture(category_id=7)
        # Ignored categories must be absent from the mapped model classes.
        category_map.write_text(json.dumps({"3": "Car"}) + "\n")
        result = MODULE.prepare(images, annotations, category_map, root / "out",
                                ignored_categories={7})
        dataset = json.loads((root / "out" / "dataset-manifest.jsonl").read_text())
        self.assertEqual(dataset["expected_label"], '{"detections":[]}')
        self.assertEqual(result["ignored_category_ids"], [7])

    def test_rejects_image_escape(self):
        root, images, annotations, category_map = self._fixture()
        value = json.loads(annotations.read_text())
        value["images"][0]["file_name"] = "../outside.jpg"
        annotations.write_text(json.dumps(value) + "\n")
        with self.assertRaisesRegex(ValueError, "escapes image root"):
            MODULE.prepare(images, annotations, category_map, root / "out")

    def test_rejects_annotation_dimension_mismatch(self):
        root, images, annotations, category_map = self._fixture()
        value = json.loads(annotations.read_text())
        value["images"][0]["width"] = 21
        annotations.write_text(json.dumps(value) + "\n")
        with self.assertRaisesRegex(ValueError, "metadata=21x10"):
            MODULE.prepare(images, annotations, category_map, root / "out")


if __name__ == "__main__":
    unittest.main()
