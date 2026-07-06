import io
import random

import pytest

from ggufpacker.delta import apply_delta, read_region, rechunk, xor_regions


def test_rechunk_reslices_exactly():
    random.seed(1)
    parts = [random.randbytes(n) for n in (0, 1, 5, 1000, 3, 0, 42)]
    whole = b"".join(parts)
    out = list(rechunk(parts, size=7))
    assert b"".join(out) == whole
    assert all(len(c) == 7 for c in out[:-1])


def test_xor_regions_and_apply_delta_roundtrip():
    random.seed(2)
    orig = random.randbytes(100_000)
    regen = bytearray(orig)
    for off in (0, 5, 99_999, 40_000):
        regen[off] ^= 0xA5
    regen = bytes(regen)

    f_orig, f_regen = io.BytesIO(orig), io.BytesIO(regen)
    chunks = []
    nonzero = 0
    for x, nz in xor_regions(f_orig, 0, f_regen, 0, len(orig)):
        chunks.append(x)
        nonzero += nz
    assert nonzero == 4

    out = io.BytesIO()
    apply_delta(io.BytesIO(regen), 0, len(orig), chunks, out)
    assert out.getvalue() == orig


def test_apply_delta_rejects_short_stream():
    regen = b"\x00" * 1000
    with pytest.raises(ValueError):
        apply_delta(io.BytesIO(regen), 0, 1000, [b"\x00" * 999], io.BytesIO())


def test_read_region_offset_and_size():
    f = io.BytesIO(b"0123456789")
    assert b"".join(read_region(f, 2, 5, chunk=2)) == b"23456"
