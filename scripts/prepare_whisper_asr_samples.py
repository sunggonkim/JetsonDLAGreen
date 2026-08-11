#!/usr/bin/env python3
"""Prepare a real, externally labelled Whisper input subset.

The source tree contains a Whisper encoder, but a shape-compatible projection
is not an ASR application.  This tool turns official LibriSpeech-style FLAC
files and their separately supplied ``*.trans.txt`` labels into the fixed
Whisper log-Mel input expected by the TensorRT encoder.

Labels are read only from the transcript sidecar files.  They are never
derived from filenames, predictions, or the generated tensor bytes.  Selection
is deterministic (canonical relative path order, optional duration bound, and
optional prefix limit) and is recorded in the provenance object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 480_000
FEATURE_SHAPE = (80, 3_000)
SAMPLE_KEYS = {"iteration", "sample_id", "path", "input_sha256"}
DATASET_KEYS = {"schema_version", "sample_id", "input_sha256", "expected_label"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for path in sorted(root.rglob("*.trans.txt")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            fields = line.split(" ", 1)
            if len(fields) != 2 or not fields[0] or not fields[1].strip():
                raise ValueError(f"invalid transcript row {path}:{line_number}")
            sample_id, transcript = fields
            if sample_id in transcripts and transcripts[sample_id] != transcript:
                raise ValueError(f"conflicting transcript for {sample_id}")
            transcripts[sample_id] = transcript
    if not transcripts:
        raise ValueError("no *.trans.txt transcript sidecars found")
    return transcripts


def _duration_seconds(path: Path) -> float:
    try:
        value = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        )
        duration = float(value.strip())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError(f"cannot determine audio duration for {path}") from error
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"audio duration is invalid for {path}")
    return duration


def _decode_audio(path: Path) -> np.ndarray:
    try:
        raw = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-",
            ]
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot decode audio file {path}") from error
    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"decoded audio is empty or non-finite for {path}")
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    window[: min(samples.size, WINDOW_SAMPLES)] = samples[:WINDOW_SAMPLES]
    return window


def _mel_features(session: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    output = session.run(None, {"audio": audio.reshape(1, WINDOW_SAMPLES)})
    if len(output) != 1:
        raise ValueError("Whisper mel model returned an unexpected output count")
    features = np.asarray(output[0], dtype=np.float32)
    if features.shape != (1, *FEATURE_SHAPE) or not np.isfinite(features).all():
        raise ValueError(f"Whisper mel output shape/value is invalid: {features.shape}")
    return np.ascontiguousarray(features[0])


def prepare(
    dataset_root: Path,
    mel_model: Path,
    output_dir: Path,
    *,
    pattern: str = "*.flac",
    max_duration_s: float | None = None,
    max_reference_words: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    mel_model = mel_model.resolve()
    output_dir = output_dir.resolve()
    if not dataset_root.is_dir() or not mel_model.is_file():
        raise ValueError("dataset root and mel model must exist")
    if max_duration_s is not None and (
        not np.isfinite(max_duration_s) or max_duration_s <= 0.0
    ):
        raise ValueError("max_duration_s must be positive and finite")
    if max_reference_words is not None and max_reference_words <= 0:
        raise ValueError("max_reference_words must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    transcripts = _read_transcripts(dataset_root)
    audio_files = sorted(
        path for path in dataset_root.rglob(pattern) if path.is_file()
    )
    selected: list[tuple[Path, float, str]] = []
    for path in audio_files:
        relative = path.relative_to(dataset_root).as_posix()
        sample_id = path.stem
        if sample_id not in transcripts:
            raise ValueError(f"transcript sidecar lacks selected audio {relative}")
        duration = _duration_seconds(path)
        if max_duration_s is not None and duration > max_duration_s:
            continue
        if max_reference_words is not None and len(transcripts[sample_id].split()) > max_reference_words:
            continue
        selected.append((path, duration, transcripts[sample_id]))
        if limit is not None and len(selected) == limit:
            break
    if not selected:
        raise ValueError("audio selection is empty")

    selected_ids = {path.stem for path, _, _ in selected}
    unused = sorted(set(transcripts) - selected_ids)
    # The source transcript file commonly contains more rows than a bounded
    # experiment.  Such rows are intentionally outside the selected subset,
    # but every selected audio must have an owner-supplied label.
    del unused

    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    mel_session = ort.InferenceSession(
        str(mel_model), providers=["CPUExecutionProvider"]
    )
    samples: list[dict[str, Any]] = []
    dataset: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for iteration, (audio, duration, transcript) in enumerate(selected):
        feature_path = feature_dir / f"{iteration:06d}.f32"
        features = _mel_features(mel_session, _decode_audio(audio))
        features.tofile(feature_path)
        digest = sha256(feature_path)
        sample_id = audio.relative_to(dataset_root).as_posix()
        samples.append({
            "iteration": iteration,
            "sample_id": sample_id,
            "path": str(feature_path),
            "input_sha256": digest,
        })
        dataset.append({
            "schema_version": 1,
            "sample_id": sample_id,
            "input_sha256": digest,
            "expected_label": transcript,
        })
        source_rows.append({
            "sample_id": sample_id,
            "audio_sha256": sha256(audio),
            "audio_seconds": duration,
            "feature_sha256": digest,
        })

    sample_path = output_dir / "samples.jsonl"
    dataset_path = output_dir / "dataset-manifest.jsonl"
    provenance_path = output_dir / "provenance.json"
    sample_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples),
        encoding="utf-8",
    )
    dataset_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dataset),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "kind": "p9-real-whisper-asr-sample-provenance",
        "dataset_root": str(dataset_root),
        "selection": {
            "pattern": pattern,
            "order": "canonical-relative-path",
            "max_duration_s": max_duration_s,
            "max_reference_words": max_reference_words,
            "limit": limit,
            "sample_reuse": False,
        },
        "label_source": "official-transcript-sidecars",
        "mel_model": {"path": str(mel_model), "sha256": sha256(mel_model)},
        "samples": source_rows,
        "sample_list": {"path": str(sample_path), "sha256": sha256(sample_path)},
        "dataset_manifest": {
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
        },
        "automatic_filename_labels": False,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        "samples": len(samples),
        "sample_list": str(sample_path),
        "dataset_manifest": str(dataset_path),
        "provenance": str(provenance_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mel-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.flac")
    parser.add_argument("--max-duration-s", type=float)
    parser.add_argument(
        "--max-reference-words", type=int,
        help="select only labels with at most this many whitespace-delimited words",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.dataset_root,
        args.mel_model,
        args.output_dir,
        pattern=args.pattern,
        max_duration_s=args.max_duration_s,
        max_reference_words=args.max_reference_words,
        limit=args.limit,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
