"""Helpers to build tiny-but-valid synthetic llama GGUF files for tests.

The model is a real 1-layer llama architecture (~1.8 MB at F16) with all the
KV metadata llama-quantize's hparam loader requires, and dims chosen so
k-quants apply cleanly (row sizes divisible by 256). llama-quantize b3821
accepts it and quantizes it in well under a second.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

from ggufpacker.layout import parse_layout


def write_tiny_llama_f16(
    path: str | Path,
    seed: int = 7,
    n_vocab: int = 512,
    n_embd: int = 256,
    n_ff: int = 512,
    n_head: int = 4,
    n_layers: int = 1,
) -> Path:
    path = Path(path)
    rng = np.random.default_rng(seed)
    w = GGUFWriter(str(path), "llama")
    w.add_context_length(512)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layers)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head)
    w.add_rope_dimension_count(n_embd // n_head)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(n_vocab)
    w.add_file_type(1)  # LlamaFileType.MOSTLY_F16

    def f16(*shape: int) -> np.ndarray:
        # small scale keeps quantized values well-behaved
        return (rng.standard_normal(shape) * 0.05).astype(np.float16)

    def f32(n: int) -> np.ndarray:
        return rng.standard_normal(n).astype(np.float32)

    w.add_tensor("token_embd.weight", f16(n_vocab, n_embd))
    w.add_tensor("output_norm.weight", f32(n_embd))
    w.add_tensor("output.weight", f16(n_vocab, n_embd))
    for i in range(n_layers):
        w.add_tensor(f"blk.{i}.attn_norm.weight", f32(n_embd))
        w.add_tensor(f"blk.{i}.attn_q.weight", f16(n_embd, n_embd))
        w.add_tensor(f"blk.{i}.attn_k.weight", f16(n_embd, n_embd))
        w.add_tensor(f"blk.{i}.attn_v.weight", f16(n_embd, n_embd))
        w.add_tensor(f"blk.{i}.attn_output.weight", f16(n_embd, n_embd))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", f32(n_embd))
        w.add_tensor(f"blk.{i}.ffn_gate.weight", f16(n_ff, n_embd))
        w.add_tensor(f"blk.{i}.ffn_up.weight", f16(n_ff, n_embd))
        w.add_tensor(f"blk.{i}.ffn_down.weight", f16(n_embd, n_ff))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def write_tiny_imatrix(
    path: str | Path,
    n_embd: int = 256,
    n_ff: int = 512,
    seed: int = 11,
    n_layers: int = 1,
) -> Path:
    """Legacy (pre-GGUF) imatrix file matching write_tiny_llama_f16's tensors.

    Format as written by llama.cpp b3821's imatrix example:
        i32 n_entries
        per entry: i32 len(name), name, i32 ncall, i32 nval, f32 vals[nval]
        i32 last_call, i32 len(dataset), dataset
    nval must equal the tensor's ne[0] (row length) or quantize ignores it.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    entries: list[tuple[str, int]] = []
    for i in range(n_layers):
        entries += [
            (f"blk.{i}.attn_q.weight", n_embd),
            (f"blk.{i}.attn_k.weight", n_embd),
            (f"blk.{i}.attn_v.weight", n_embd),
            (f"blk.{i}.attn_output.weight", n_embd),
            (f"blk.{i}.ffn_gate.weight", n_embd),
            (f"blk.{i}.ffn_up.weight", n_embd),
            (f"blk.{i}.ffn_down.weight", n_ff),
        ]
    entries.append(("output.weight", n_embd))
    dataset = b"synthetic-calibration.txt"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, nval in entries:
            nb = name.encode()
            vals = (rng.random(nval).astype(np.float32) + 0.5)  # positive importance
            f.write(struct.pack("<i", len(nb)) + nb)
            f.write(struct.pack("<ii", 1, nval))  # ncall=1, nval
            f.write(vals.tobytes())
        f.write(struct.pack("<i", 1))  # last_call
        f.write(struct.pack("<i", len(dataset)) + dataset)
    return Path(path)


def flip_payload_bytes(src: str | Path, dest: str | Path, offsets_in_payload: list[int]) -> Path:
    """Simulate publisher fp-contraction variance: flip bytes inside the
    tensor-data region only; header stays identical."""
    lay = parse_layout(src)
    data = bytearray(Path(src).read_bytes())
    for off in offsets_in_payload:
        assert 0 <= off < lay.data_size
        data[lay.data_start + off] ^= 0xFF
    Path(dest).write_bytes(bytes(data))
    return Path(dest)


def mutate_kv_string(src: str | Path, dest: str | Path, key: str, new_value: bytes) -> Path:
    """Rewrite one KV string value with a DIFFERENT length (like the local path
    embedded in `quantize.imatrix.file`), shifting the metadata section length.
    Tensor-info rel_offsets are relative to data_start, so only the alignment
    padding needs recomputing; the payload is byte-identical."""
    lay = parse_layout(src)
    off, ln = lay.kv_string_locs[key]
    data = Path(src).read_bytes()
    header = data[: lay.header_end]
    # string = u64 length prefix + bytes; prefix sits at off-8
    new_header = (
        header[: off - 8] + struct.pack("<Q", len(new_value)) + new_value + header[off + ln:]
    )
    pad = (-len(new_header)) % lay.alignment
    Path(dest).write_bytes(new_header + b"\x00" * pad + data[lay.data_start:])
    return Path(dest)
