"""Tests for the Qwen2 -> GGUF tokenizer conversion."""

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[2]


class FakeWriter:
    """Records the calls a real gguf.GGUFWriter would receive."""

    def __init__(self):
        self.tokens = None
        self.vocab_size = None

    def add_tokenizer_model(self, *a, **k): pass
    def add_tokenizer_pre(self, *a, **k): pass
    def add_token_list(self, tokens): self.tokens = tokens
    def add_token_types(self, *a, **k): pass
    def add_token_merges(self, *a, **k): pass
    def add_vocab_size(self, n): self.vocab_size = n
    def add_bos_token_id(self, *a, **k): pass
    def add_eos_token_id(self, *a, **k): pass
    def add_pad_token_id(self, *a, **k): pass
    def add_eot_token_id(self, *a, **k): pass


def _write_fake_tokenizer_files(model_dir: Path, num_named_tokens: int) -> None:
    """A tokenizer.json whose vocab only names ids 0..num_named_tokens-1.

    Mirrors real Qwen2 checkpoints, where config.json's vocab_size (the
    actual token_embd.weight row count) is padded larger than the number
    of ids the tokenizer actually names.
    """
    vocab = {f"tok{i}": i for i in range(num_named_tokens)}
    tj = {"model": {"vocab": vocab, "merges": []}, "added_tokens": []}
    tc = {"eos_token": f"tok{num_named_tokens - 1}", "pad_token": None, "bos_token": None}
    (model_dir / "tokenizer.json").write_text(json.dumps(tj))
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tc))


def test_tokenizer_token_list_padded_to_config_vocab_size(tmp_path):
    """token list length must match config.json's vocab_size, not just the
    highest token id named in tokenizer.json.

    Reproduces: real Qwen2 checkpoints can have config.json vocab_size (the
    actual embedding tensor row count) larger than the number of ids named
    in tokenizer.json (reserved/padding rows). If the GGUF's token list is
    shorter than the embedding tensor, llama.cpp fails to load the model
    with 'check_tensor_dims: tensor token_embd.weight has wrong shape'.
    """
    pytest.importorskip("gguf")
    sys.path.insert(0, str(_root / "src"))
    from utils.convert_qwen2_to_gguf import _write_qwen2_tokenizer

    _write_fake_tokenizer_files(tmp_path, num_named_tokens=10)

    writer = FakeWriter()
    _write_qwen2_tokenizer(writer, tmp_path, vocab_size=16)

    assert len(writer.tokens) == 16
    assert writer.vocab_size == 16
