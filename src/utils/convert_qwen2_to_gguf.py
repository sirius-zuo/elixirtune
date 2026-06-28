#!/usr/bin/env python3
"""
Convert a Qwen2.x MLX-quantized (4-bit) fused model to GGUF F16.
Requires: gguf>=0.10, safetensors, numpy  (all available in Python 3.11 framework)
Usage: python3.11 convert_qwen2_to_gguf.py <fused_model_dir> <output.gguf>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

import gguf

TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_CONTROL = 3

BITS = 4
GROUP_SIZE = 64
VALUES_PER_UINT32 = 32 // BITS  # 8


def dequantize_mlx_4bit(weight: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """Dequantize MLX affine 4-bit packed uint32 tensor to float16."""
    out_features, packed_in = weight.shape
    in_features = packed_in * VALUES_PER_UINT32
    # Derive group size from actual tensor shapes (scales has one entry per group)
    actual_group_size = in_features // scales.shape[1]

    # Unpack 8 × 4-bit values from each uint32
    unpacked = np.zeros((out_features, in_features), dtype=np.int32)
    mask = (1 << BITS) - 1
    for i in range(VALUES_PER_UINT32):
        unpacked[:, i::VALUES_PER_UINT32] = (weight >> (i * BITS)) & mask

    # Expand scales/biases from (out, groups) → (out, in_features)
    scales_f = np.repeat(scales.astype(np.float32), actual_group_size, axis=1)
    biases_f = np.repeat(biases.astype(np.float32), actual_group_size, axis=1)

    return (unpacked.astype(np.float32) * scales_f + biases_f).astype(np.float16)


# HuggingFace → GGUF tensor name mapping for Qwen2 architecture
# Computed once via gguf.TensorNameMap, then cached here for speed.
def _hf_to_gguf_name(hf_name: str, num_layers: int, tnm: gguf.TensorNameMap) -> str | None:
    """Return the GGUF tensor name for a HF tensor name, or None to skip."""
    # Strip trailing .weight / .bias
    suffix = ""
    base = hf_name
    for sfx in (".weight", ".bias"):
        if hf_name.endswith(sfx):
            base = hf_name[: -len(sfx)]
            suffix = sfx
            break

    # Replace layer indices: model.layers.7.X → model.layers.N.X then use tnm
    mapped = tnm.mapping.get(base)
    if mapped is None:
        return None
    _, gguf_base = mapped
    return gguf_base + suffix


def _write_qwen2_tokenizer(writer: gguf.GGUFWriter, model_dir: Path) -> None:
    """Write Qwen2 BPE tokenizer metadata from tokenizer.json."""
    tj = json.loads((model_dir / "tokenizer.json").read_text())
    tc = json.loads((model_dir / "tokenizer_config.json").read_text())

    base_vocab = tj["model"]["vocab"]   # {token_str: id}  (id 0..151642)
    added = tj.get("added_tokens", [])  # [{id, content, special}, ...]
    merges_raw = tj["model"]["merges"]  # [["A","B"], ...] or ["A B", ...]

    # Build sorted token list (by id)
    all_tokens: dict[int, str] = {v: k for k, v in base_vocab.items()}
    special_ids: set[int] = set()
    for entry in added:
        all_tokens[entry["id"]] = entry["content"]
        if entry.get("special"):
            special_ids.add(entry["id"])

    max_id = max(all_tokens)
    tokens = [all_tokens.get(i, f"[PAD{i}]") for i in range(max_id + 1)]
    token_types = [
        TOKEN_TYPE_CONTROL if i in special_ids else TOKEN_TYPE_NORMAL
        for i in range(max_id + 1)
    ]

    # Merges: stored as ["A","B"] pairs or "A B" strings → write as "A B"
    if merges_raw and isinstance(merges_raw[0], list):
        merges = [f"{a} {b}" for a, b in merges_raw]
    else:
        merges = list(merges_raw)

    # Identify special token IDs from tokenizer_config
    def _find_id(content: str | None) -> int | None:
        if content is None:
            return None
        for tid, tok in all_tokens.items():
            if tok == content:
                return tid
        return None

    eos_id = _find_id(tc.get("eos_token"))
    pad_id = _find_id(tc.get("pad_token"))
    bos_id = _find_id(tc.get("bos_token"))
    eot_id = _find_id("<|im_end|>")       # Qwen2 uses this as end-of-turn

    writer.add_tokenizer_model("gpt2")    # BPE family
    writer.add_tokenizer_pre("qwen2")     # llama.cpp pre-tokenizer variant
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_vocab_size(len(tokens))
    if bos_id is not None:
        writer.add_bos_token_id(bos_id)
    if eos_id is not None:
        writer.add_eos_token_id(eos_id)
    if pad_id is not None:
        writer.add_pad_token_id(pad_id)
    if eot_id is not None:
        writer.add_eot_token_id(eot_id)
    print(f"  Tokenizer: {len(tokens)} tokens, {len(merges)} merges, "
          f"eos={eos_id}, pad={pad_id}, eot={eot_id}")


def convert(model_dir: Path, output: Path) -> None:
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())

    num_layers: int = config["num_hidden_layers"]
    hidden_size: int = config["hidden_size"]
    num_heads: int = config["num_attention_heads"]
    num_kv_heads: int = config.get("num_key_value_heads", num_heads)
    intermediate_size: int = config["intermediate_size"]
    vocab_size: int = config["vocab_size"]
    max_pos: int = config.get("max_position_embeddings", 32768)
    rms_eps: float = config.get("rms_norm_eps", 1e-6)
    rope_theta: float = config.get("rope_theta", 10000.0)
    head_dim: int = hidden_size // num_heads

    tnm = gguf.TensorNameMap(gguf.MODEL_ARCH.QWEN2, num_layers)

    writer = gguf.GGUFWriter(str(output), arch="qwen2")

    # ── Metadata ──────────────────────────────────────────────────────────────
    writer.add_name(model_dir.name)
    writer.add_context_length(max_pos)
    writer.add_embedding_length(hidden_size)
    writer.add_block_count(num_layers)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_layer_norm_rms_eps(rms_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_rope_dimension_count(head_dim)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Writing tokenizer...")
    try:
        _write_qwen2_tokenizer(writer, model_dir)
    except Exception as e:
        print(f"  Warning: tokenizer write failed ({e}); GGUF will lack tokenizer", file=sys.stderr)

    # ── Tensors ───────────────────────────────────────────────────────────────
    sf_path = model_dir / "model.safetensors"
    if not sf_path.exists():
        # Sharded model — look at index
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]
        index = {"weight_map": {}}  # handled separately

    # Collect tensor metadata across all shards
    tensors: dict[str, tuple[Path, str]] = {}  # name → (shard, shape/dtype info)
    for shard in shard_files:
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="numpy") as f:
            for k in f.keys():
                tensors[k] = shard_path

    # Group quantized triplets: base → (weight_key, scales_key, biases_key)
    quantized_groups: dict[str, dict[str, str]] = {}
    plain_tensors: list[str] = []

    for key in tensors:
        if key.endswith(".scales"):
            base = key[: -len(".scales")]
            quantized_groups.setdefault(base, {})["scales"] = key
        elif key.endswith(".biases"):
            base = key[: -len(".biases")]
            quantized_groups.setdefault(base, {})["biases"] = key
        elif key.endswith(".weight"):
            base = key[: -len(".weight")]
            if base in quantized_groups or (base + ".scales") in tensors:
                quantized_groups.setdefault(base, {})["weight"] = key
            else:
                plain_tensors.append(key)
        else:
            plain_tensors.append(key)

    written = 0

    def _write(hf_name: str, arr: np.ndarray) -> None:
        nonlocal written
        gguf_name = _hf_to_gguf_name(hf_name, num_layers, tnm)
        if gguf_name is None:
            print(f"  skip  {hf_name} (unmapped)")
            return
        arr = arr.astype(np.float16)
        writer.add_tensor(gguf_name, arr)
        written += 1
        print(f"  [{written:4d}] {hf_name:60s} → {gguf_name}  {arr.shape}")

    print(f"Writing tensors ({len(quantized_groups)} quantized, {len(plain_tensors)} plain)...")

    # Open all shards once and cache
    open_shards: dict[Path, object] = {}
    for p in set(tensors.values()):
        open_shards[p] = safe_open(str(p), framework="numpy")

    def _get(key: str) -> np.ndarray:
        return open_shards[tensors[key]].get_tensor(key)

    for base, parts in quantized_groups.items():
        if "weight" not in parts:
            continue
        w = _get(parts["weight"])
        s = _get(parts["scales"])
        b = _get(parts["biases"])
        arr = dequantize_mlx_4bit(w, s, b)
        _write(base + ".weight", arr)

    for key in plain_tensors:
        arr = _get(key)
        _write(key, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone — {written} tensors written to {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    convert(args.model_dir, args.output)
