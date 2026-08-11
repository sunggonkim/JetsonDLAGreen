#!/usr/bin/env python3
"""Decode a JDGASR1 token trace with the pinned Whisper tokenizer.

The binary trace contains only post-completion token IDs.  This decoder binds
those IDs to the exact tokenizer JSON used by the real ASR application and
emits a hash-keyed transcript map for the accuracy gate.  It does not read
dataset labels and cannot manufacture a prediction for a missing output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_trace_reader() -> Any:
    path = Path(__file__).with_name("read_application_output_trace.py")
    spec = importlib.util.spec_from_file_location("p9_asr_output_trace_reader", path)
    if spec is None or spec.loader is None:
        raise ValueError("application output trace parser is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _byte_encoder() -> dict[int, str]:
    """Return the GPT-2 byte-to-unicode alphabet used by Whisper BPE."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(value) for value in unicode_values)))


def _token_map(tokenizer: dict[str, Any]) -> dict[int, str]:
    model = tokenizer.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError("tokenizer model vocabulary is missing")
    result: dict[int, str] = {}
    for token, token_id in model["vocab"].items():
        if not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("tokenizer vocabulary entry is invalid")
        if token_id in result and result[token_id] != token:
            raise ValueError("tokenizer vocabulary repeats an id")
        result[token_id] = token
    added_tokens = tokenizer.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer added_tokens is invalid")
    for entry in added_tokens:
        if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
            raise ValueError("tokenizer added token is invalid")
        token_id = entry.get("id")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("tokenizer added token id is invalid")
        previous = result.get(token_id)
        if previous is not None and previous != entry["content"]:
            raise ValueError("tokenizer added token conflicts with vocabulary")
        result[token_id] = entry["content"]
    return result


def decode_tokens(tokens: list[int], token_map: dict[int, str]) -> str:
    encoder = _byte_encoder()
    decoder = {symbol: value for value, symbol in encoder.items()}
    pieces: list[str] = []
    for token_id in tokens:
        token = token_map.get(token_id)
        if token is None:
            raise ValueError(f"Whisper token id {token_id} is absent from tokenizer")
        if token.startswith("<|") and token.endswith("|>"):
            continue
        try:
            pieces.append(bytes(decoder[character] for character in token).decode("utf-8"))
        except (KeyError, UnicodeDecodeError) as error:
            raise ValueError(f"Whisper token {token_id} is not valid ByteLevel UTF-8") from error
    return "".join(pieces).strip()


def decode(trace: Path, tokenizer_path: Path) -> dict[str, str]:
    reader = _load_trace_reader()
    parsed = reader.parse(trace)
    if parsed.get("format") != "JDGASR1" or parsed.get("task") != "asr":
        raise ValueError("trace is not a JDGASR1 ASR output trace")
    tokenizer = json.loads(tokenizer_path.resolve().read_bytes())
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer JSON must be an object")
    token_map = _token_map(tokenizer)
    result: dict[str, str] = {}
    for record in parsed["records"]:
        output = record["outputs"][0]
        transcript = decode_tokens(output["tokens"], token_map)
        if not transcript:
            raise ValueError(f"empty decoded transcript at iteration {record['iteration']}")
        digest = output["sha256"]
        if digest in result and result[digest] != transcript:
            raise ValueError("output hash maps to conflicting transcripts")
        result[digest] = transcript
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decode(args.trace, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"kind": "p9-whisper-token-transcript-map", "outputs": len(result), "path": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
