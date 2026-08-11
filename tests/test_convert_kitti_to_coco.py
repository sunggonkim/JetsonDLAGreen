import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_kitti_to_coco", ROOT / "scripts/convert_kitti_to_coco.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConvertKittiToCocoTest(unittest.TestCase):
    def _fixture(self):
        root = Path(tempfile.mkdtemp())
        images = root / "images"
        labels = root / "labels"
        images.mkdir()
        labels.mkdir()
        Image.new("RGB", (100, 50), (0, 0, 0)).save(images / "000001.png")
        labels.joinpath("000001.txt").write_text(
            "Car 0 0 0 10 5 80 40 1 1 1 1 1 1 1\n"
            "DontCare -1 -1 -10 0 0 10 10 0 0 0 0 0 0 0\n",
            encoding="utf-8",
        )
        return root, images, labels

    def test_converts_explicit_class_and_writes_category_map(self):
        root, images, labels = self._fixture()
        output = root / "annotations.json"
        result = MODULE.convert(images, labels, output, mappings=["Car=Car"])
        value = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(value["images"]), 1)
        self.assertEqual(len(value["annotations"]), 1)
        self.assertEqual(value["annotations"][0]["category_id"], 1)
        self.assertEqual(value["annotations"][0]["bbox"], [10.0, 5.0, 70.0, 35.0])
        self.assertEqual(json.loads(Path(result["category_map_path"]).read_text()), {"1": "Car"})

    def test_reuses_category_id_for_multiple_objects(self):
        root, images, labels = self._fixture()
        labels.joinpath("000001.txt").write_text(
            "Car 0 0 0 10 5 40 25 1 1 1 1 1 1 1\n"
            "Car 0 0 0 50 10 90 45 1 1 1 1 1 1 1\n",
            encoding="utf-8",
        )
        result = MODULE.convert(images, labels, root / "annotations.json", mappings=["Car=Car"])
        annotations = result["annotations"]
        self.assertEqual(len(annotations), 2)
        self.assertEqual([row["category_id"] for row in annotations], [1, 1])

    def test_rejects_unmapped_class(self):
        root, images, labels = self._fixture()
        labels.joinpath("000001.txt").write_text(
            "Truck 0 0 0 10 5 80 40 1 1 1 1 1 1 1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "neither mapped nor ignored"):
            MODULE.convert(images, labels, root / "annotations.json", mappings=["Car=Car"])

    def test_rejects_unsupported_target_label(self):
        root, images, labels = self._fixture()
        with self.assertRaisesRegex(ValueError, "not emitted by ResNet10"):
            MODULE.convert(images, labels, root / "annotations.json", mappings=["Car=Person"])


if __name__ == "__main__":
    unittest.main()
