import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "decode_whisper_token_trace.py"
SPEC = importlib.util.spec_from_file_location("decode_whisper_token_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DecodeWhisperTokenTraceTest(unittest.TestCase):
    def test_decodes_byte_level_tokens_and_skips_special_tokens(self) -> None:
        encoder = MODULE._byte_encoder()
        token_map = {
            1: encoder[32] + "hello",
            2: encoder[32] + "world",
            3: "<|endoftext|>",
        }
        self.assertEqual(MODULE.decode_tokens([1, 2, 3], token_map), "hello world")

    def test_rejects_unknown_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent from tokenizer"):
            MODULE.decode_tokens([9], {})

    def test_tokenizer_map_accepts_vocab_and_added_tokens(self) -> None:
        value = {
            "model": {"vocab": {"a": 1}},
            "added_tokens": [{"id": 2, "content": "<|endoftext|>"}],
        }
        result = MODULE._token_map(value)
        self.assertEqual(result, {1: "a", 2: "<|endoftext|>"})


if __name__ == "__main__":
    unittest.main()
