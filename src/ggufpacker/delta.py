"""Streaming XOR deltas over GGUF tensor-data regions.

The delta covers the whole data region (all tensor payloads plus inter-tensor
alignment padding) of the *original* file. packer.py only produces such a delta
after proving the original and regenerated layouts are congruent (same tensor
names/order/types/dims/relative offsets, same region size), which makes the
region XOR exactly equivalent to a per-tensor XOR stream — without any
per-tensor bookkeeping at reconstruction time.

Everything streams in fixed chunks; no file is ever loaded whole.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import BinaryIO

CHUNK = 8 * 1024 * 1024


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    # int.from_bytes/XOR/to_bytes is the fastest pure-stdlib byte XOR by a wide
    # margin; avoids a numpy runtime dependency for the core path.
    n = len(a)
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(n, "little")


def rechunk(chunks: Iterable[bytes], size: int = CHUNK) -> Iterator[bytes]:
    """Re-slice an iterator of arbitrary-size chunks into fixed-size chunks."""
    buf = bytearray()
    for c in chunks:
        buf += c
        while len(buf) >= size:
            yield bytes(buf[:size])
            del buf[:size]
    if buf:
        yield bytes(buf)


def read_region(f: BinaryIO, start: int, size: int, chunk: int = CHUNK) -> Iterator[bytes]:
    f.seek(start)
    remaining = size
    while remaining:
        b = f.read(min(chunk, remaining))
        if not b:
            raise OSError("unexpected EOF while reading tensor-data region")
        remaining -= len(b)
        yield b


def xor_regions(
    f_a: BinaryIO, start_a: int, f_b: BinaryIO, start_b: int, size: int
) -> Iterator[tuple[bytes, int]]:
    """Yield (xor_chunk, nonzero_byte_count) over two equal-size file regions."""
    it_a = read_region(f_a, start_a, size)
    it_b = read_region(f_b, start_b, size)
    for a, b in zip(it_a, it_b, strict=True):
        x = _xor_bytes(a, b)
        yield x, len(x) - x.count(0)


def apply_delta(
    regen: BinaryIO, regen_start: int, size: int, delta_chunks: Iterable[bytes], out: BinaryIO
) -> None:
    """out += regen data region XOR delta stream (both exactly `size` bytes)."""
    it_r = read_region(regen, regen_start, size)
    it_d = rechunk(delta_chunks, CHUNK)
    written = 0
    for r, d in zip(it_r, it_d, strict=True):
        if len(r) != len(d):
            raise ValueError("delta stream length does not match data region")
        out.write(_xor_bytes(r, d))
        written += len(r)
    if written != size:
        raise ValueError(f"delta application wrote {written} of {size} bytes")
