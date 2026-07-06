from pathlib import Path

import pytest

from ggufpacker.layout import GGUFParseError, parse_layout, read_header_bytes
from tests.util_tinymodel import mutate_kv_string


def test_parse_layout_tiny_model(tiny_f16: Path):
    lay = parse_layout(tiny_f16)
    assert lay.version in (2, 3)
    assert lay.alignment == 32
    assert lay.data_start % lay.alignment == 0
    assert lay.header_end <= lay.data_start < lay.file_size
    assert len(lay.tensors) == 12
    names = [t.name for t in lay.tensors]
    assert "token_embd.weight" in names
    assert "blk.0.ffn_down.weight" in names
    # first tensor's payload starts at relative offset 0
    assert lay.tensors[0].rel_offset == 0
    # rel offsets are aligned and strictly increasing
    offs = [t.rel_offset for t in lay.tensors]
    assert offs == sorted(offs)
    assert all(o % lay.alignment == 0 for o in offs)


def test_header_plus_payload_is_whole_file(tiny_f16: Path):
    lay = parse_layout(tiny_f16)
    header = read_header_bytes(lay)
    assert len(header) == lay.data_start
    whole = Path(tiny_f16).read_bytes()
    assert header == whole[: lay.data_start]
    assert len(whole) - len(header) == lay.data_size


def test_parse_rejects_garbage(tmp_path: Path):
    p = tmp_path / "junk.gguf"
    p.write_bytes(b"\x00" * 4096)
    with pytest.raises(GGUFParseError):
        parse_layout(p)


def test_parse_rejects_truncated(tiny_f16: Path, tmp_path: Path):
    data = Path(tiny_f16).read_bytes()
    p = tmp_path / "trunc.gguf"
    p.write_bytes(data[:40])  # magic+version+counts, then nothing
    with pytest.raises(GGUFParseError):
        parse_layout(p)


def test_kv_string_mutation_preserves_layout(tiny_f16: Path, tmp_path: Path):
    """mutate_kv_string models the quantize.imatrix.file path-length variance:
    metadata section length changes, payload bytes and offsets do not."""
    out = tmp_path / "mutated.gguf"
    mutate_kv_string(tiny_f16, out, "general.architecture", b"llama" + b"x" * 41)
    a, b = parse_layout(tiny_f16), parse_layout(out)
    assert b.data_start != a.data_start
    assert a.congruence_key() == b.congruence_key()  # tensors + data size unchanged
    assert Path(tiny_f16).read_bytes()[a.data_start:] == out.read_bytes()[b.data_start:]
