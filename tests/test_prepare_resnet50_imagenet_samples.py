import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_resnet50_imagenet_samples.py"
SPEC = importlib.util.spec_from_file_location("prepare_resnet50_imagenet_samples", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareResnet50ImageNetSamplesTest(unittest.TestCase):
    def _fixture(self):
        root = Path(tempfile.mkdtemp())
        images = root / "images"
        images.mkdir()
        for name, color in (("n03062245_0001.JPEG-224.jpg", (20, 40, 60)), ("n03792782_0002.JPEG-224.jpg", (60, 40, 20))):
            image = np.zeros((224, 320, 3), dtype=np.uint8)
            image[:, :] = color
            Image.fromarray(image, mode="RGB").save(images / name)
        mapping = root / "synset-map.json"
        mapping.write_text(json.dumps({
            "n03062245": {"index": 503, "label": "cocktail shaker", "source": "ImageNet WNID annotation"},
            "n03792782": {"index": 671, "label": "mountain bike", "source": "ImageNet WNID annotation"},
        }) + "\n")
        return root, images, mapping

    def test_preprocesses_and_binds_explicit_labels(self):
        root, images, mapping = self._fixture()
        result = MODULE.prepare(images, mapping, root / "out")
        self.assertEqual(len(result["samples"]), 2)
        sample = json.loads((root / "out" / "samples.jsonl").read_text().splitlines()[0])
        dataset = json.loads((root / "out" / "dataset-manifest.jsonl").read_text().splitlines()[0])
        self.assertEqual(dataset["expected_label"], "cocktail shaker")
        tensor = np.fromfile(root / "out" / "tensors" / "000000.f32", dtype=np.float32)
        self.assertEqual(tensor.size, 3 * 224 * 224)
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue(result["provenance"]["labels_external"])
        self.assertFalse(result["provenance"]["filename_labels_inferred"])
        self.assertEqual(json.loads((root / "out" / "class-map.json").read_text())["503"], "cocktail shaker")
        self.assertEqual(sample["input_sha256"], result["samples"][0]["input_sha256"])

    def test_rejects_unmapped_filename_synset(self):
        root, images, mapping = self._fixture()
        mapping.write_text(json.dumps({"n03062245": {"index": 503, "label": "cocktail shaker", "source": "external"}}) + "\n")
        with self.assertRaisesRegex(ValueError, "absent from explicit map"):
            MODULE.prepare(images, mapping, root / "out")

    def test_rejects_missing_wnid(self):
        root, images, mapping = self._fixture()
        (images / "n03792782_0002.JPEG-224.jpg").rename(images / "unlabelled.jpg")
        with self.assertRaisesRegex(ValueError, "lacks an ImageNet WNID"):
            MODULE.prepare(images, mapping, root / "out")

    def test_accepts_explicit_parent_directory_wnid(self):
        root, images, mapping = self._fixture()
        parent_root = root / "imagenette" / "n03062245"
        parent_root.mkdir(parents=True)
        (images / "n03062245_0001.JPEG-224.jpg").rename(parent_root / "image.JPEG")
        with self.assertRaisesRegex(ValueError, "lacks an ImageNet WNID"):
            MODULE.prepare(root / "imagenette", mapping, root / "out")
        result = MODULE.prepare(
            root / "imagenette", mapping, root / "out-parent", wnid_from_parent=True
        )
        self.assertEqual(result["dataset"][0]["expected_label"], "cocktail shaker")
        self.assertTrue(result["provenance"]["wnid_from_parent_directory"])

    def test_limit_is_per_synset(self):
        root, images, mapping = self._fixture()
        image = Image.open(images / "n03062245_0001.JPEG-224.jpg")
        image.save(images / "n03062245_0003.JPEG-224.jpg")
        result = MODULE.prepare(images, mapping, root / "out", limit_per_synset=1)
        self.assertEqual(len(result["samples"]), 2)

    def test_binds_source_archive_digest(self):
        root, images, mapping = self._fixture()
        archive = root / "dataset.tar"
        archive.write_bytes(b"immutable archive placeholder\n")
        result = MODULE.prepare(images, mapping, root / "out", source_archive=archive)
        self.assertEqual(result["provenance"]["source_archive"]["sha256"], MODULE.sha256(archive))


if __name__ == "__main__":
    unittest.main()
