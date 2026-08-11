import json
import tempfile
import unittest
from pathlib import Path

from analysis.verify_p9_runtime_lock import sha256, verify_runtime_binding


class RuntimeLockBindingTest(unittest.TestCase):
    def _fixtures(self, directory: Path) -> tuple[Path, Path, Path, Path]:
        repo = directory / "repo"
        binary = repo / "build-r39/jdg-mig-trt-pipeline"
        source = repo / "benchmarks/mig_trt_pipeline.cpp"
        producer = directory / "producer.engine"
        consumer = directory / "consumer.engine"
        installed = directory / "different.engine"
        for path, contents in (
            (binary, b"runtime-binary"),
            (source, b"// runtime source\n"),
            (producer, b"producer-engine"),
            (consumer, b"consumer-engine"),
            (installed, b"different-engine"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        return repo, producer, consumer, installed

    def _lock(
        self,
        directory: Path,
        repo: Path,
        producer: Path,
        consumer: Path,
    ) -> Path:
        artifacts = {
            "binary": {
                "path": str(repo / "build-r39/jdg-mig-trt-pipeline"),
                "sha256": sha256(repo / "build-r39/jdg-mig-trt-pipeline"),
            },
            "source": {
                "path": str(repo / "benchmarks/mig_trt_pipeline.cpp"),
                "sha256": sha256(repo / "benchmarks/mig_trt_pipeline.cpp"),
            },
            "engine": {"path": str(producer), "sha256": sha256(producer)},
            "consumer_engine": {
                "path": str(consumer),
                "sha256": sha256(consumer),
            },
        }
        path = directory / "deadline-lock.json"
        path.write_text(json.dumps({"artifacts": artifacts}) + "\n", encoding="utf-8")
        return path

    def test_runtime_artifacts_are_rehashed_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            repo, producer, consumer, _ = self._fixtures(directory)
            lock = self._lock(directory, repo, producer, consumer)
            result = verify_runtime_binding(lock, repo, producer, consumer)
            self.assertEqual(
                result["artifacts"]["engine"]["sha256"], sha256(producer)
            )
            self.assertEqual(
                result["artifacts"]["consumer_engine"]["sha256"],
                sha256(consumer),
            )

    def test_different_launched_engine_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            repo, producer, consumer, installed = self._fixtures(directory)
            lock = self._lock(directory, repo, producer, consumer)
            with self.assertRaisesRegex(ValueError, "launched runtime"):
                verify_runtime_binding(lock, repo, installed, consumer)

    def test_omitted_consumer_is_rejected_when_lock_binds_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            repo, producer, consumer, _ = self._fixtures(directory)
            lock = self._lock(directory, repo, producer, consumer)
            with self.assertRaisesRegex(ValueError, "omits it"):
                verify_runtime_binding(lock, repo, producer)


if __name__ == "__main__":
    unittest.main()
