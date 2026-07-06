"""Byte-level GGUF layout parser.

We parse the header ourselves (instead of relying on gguf-py's GGUFReader)
because reconstruction needs *exact byte offsets*: where the header+KV+tensor-info
section ends, where the alignment padding ends, and where the tensor-data region
begins. gguf-py exposes logical values but not a guaranteed-stable byte map.

Layout of a GGUF v2/v3 file:

    u32 magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv
    n_kv   x (string key, u32 vtype, value)
    n_tensors x (string name, u32 n_dims, u64 dims[n_dims], u32 ggml_type, u64 rel_offset)
    padding to `general.alignment` (default 32)
    tensor data region (rel_offsets are relative to its start)

Key design point (see packer.py): if two files have *congruent* tensor-info
tables (same names, order, ggml types, dims and relative offsets) then their
tensor-data regions line up byte-for-byte even when the metadata sections have
different lengths (e.g. `quantize.imatrix.file` embeds a local path). A single
XOR over the whole data region is then exactly equivalent to a per-tensor XOR
stream, including inter-tensor alignment padding. We verify congruence before
ever producing such a delta; anything non-congruent falls back to a full blob.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_SIZE = {
    _T_UINT8: 1, _T_INT8: 1, _T_UINT16: 2, _T_INT16: 2,
    _T_UINT32: 4, _T_INT32: 4, _T_FLOAT32: 4, _T_BOOL: 1,
    _T_UINT64: 8, _T_INT64: 8, _T_FLOAT64: 8,
}
_SCALAR_FMT = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h",
    _T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f", _T_BOOL: "<?",
    _T_UINT64: "<Q", _T_INT64: "<q", _T_FLOAT64: "<d",
}


class GGUFParseError(ValueError):
    """Raised when a file is not a parseable GGUF v2/v3 file."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int  # raw GGMLQuantizationType value
    rel_offset: int  # relative to data_start, alignment-padded by the writer


@dataclass(frozen=True)
class GGUFLayout:
    path: str
    file_size: int
    version: int
    alignment: int
    header_end: int  # byte offset just past the last tensor-info entry
    data_start: int  # header_end rounded up to alignment
    tensors: tuple[TensorInfo, ...]
    kv_string_locs: dict[str, tuple[int, int]]  # key -> (value byte offset, byte length)

    @property
    def data_size(self) -> int:
        return self.file_size - self.data_start

    def congruence_key(self) -> tuple:
        """Everything that must match for two data regions to line up byte-for-byte."""
        return (self.alignment, self.data_size, self.tensors)


class _Cursor:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise GGUFParseError("truncated GGUF file")
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > 1 << 31:
            raise GGUFParseError("implausible string length")
        return self.take(n).decode("utf-8", errors="replace")


def _skip_value(c: _Cursor, vtype: int) -> None:
    if vtype in _SCALAR_SIZE:
        c.take(_SCALAR_SIZE[vtype])
    elif vtype == _T_STRING:
        c.take(c.u64())
    elif vtype == _T_ARRAY:
        etype = c.u32()
        count = c.u64()
        if etype == _T_STRING:
            for _ in range(count):
                c.take(c.u64())
        elif etype in _SCALAR_SIZE:
            c.take(_SCALAR_SIZE[etype] * count)
        else:
            raise GGUFParseError(f"nested/unknown array element type {etype}")
    else:
        raise GGUFParseError(f"unknown KV value type {vtype}")


def parse_layout(path: str | Path) -> GGUFLayout:
    """Parse the metadata section of a GGUF file and return its byte layout.

    Reads the whole file header into memory (headers are KBs-to-low-MBs; tensor
    data is never touched here). Raises GGUFParseError for non-GGUF input.
    """
    path = Path(path)
    file_size = path.stat().st_size
    # Read a generous header window, growing if the metadata is unusually large
    # (big tokenizer vocab arrays can reach tens of MB).
    window = 8 * 1024 * 1024
    while True:
        with open(path, "rb") as f:
            buf = f.read(min(window, file_size))
        try:
            return _parse(buf, str(path), file_size)
        except GGUFParseError as e:
            if "truncated" in str(e) and window < file_size:
                window *= 4
                continue
            raise


def _parse(buf: bytes, path: str, file_size: int) -> GGUFLayout:
    c = _Cursor(buf)
    if c.u32() != GGUF_MAGIC:
        raise GGUFParseError("bad magic; not a GGUF file")
    version = c.u32()
    if version not in (2, 3):
        raise GGUFParseError(f"unsupported GGUF version {version}")
    n_tensors = c.u64()
    n_kv = c.u64()
    if n_tensors > 1 << 24 or n_kv > 1 << 24:
        raise GGUFParseError("implausible tensor/kv count")

    alignment = DEFAULT_ALIGNMENT
    kv_string_locs: dict[str, tuple[int, int]] = {}
    for _ in range(n_kv):
        key = c.string()
        vtype = c.u32()
        if vtype == _T_STRING:
            n = c.u64()
            kv_string_locs[key] = (c.pos, n)
            c.take(n)
        else:
            if key == "general.alignment" and vtype in _SCALAR_FMT:
                raw = c.buf[c.pos:c.pos + _SCALAR_SIZE[vtype]]
                alignment = int(struct.unpack(_SCALAR_FMT[vtype], raw)[0])
            _skip_value(c, vtype)
    if alignment <= 0 or alignment & (alignment - 1):
        raise GGUFParseError(f"invalid alignment {alignment}")

    tensors = []
    for _ in range(n_tensors):
        name = c.string()
        n_dims = c.u32()
        if n_dims > 8:
            raise GGUFParseError(f"implausible n_dims {n_dims}")
        dims = tuple(c.u64() for _ in range(n_dims))
        ggml_type = c.u32()
        rel_offset = c.u64()
        tensors.append(TensorInfo(name, dims, ggml_type, rel_offset))

    header_end = c.pos
    data_start = (header_end + alignment - 1) // alignment * alignment
    if data_start > file_size:
        raise GGUFParseError("data_start beyond end of file")
    return GGUFLayout(
        path=path,
        file_size=file_size,
        version=version,
        alignment=alignment,
        header_end=header_end,
        data_start=data_start,
        tensors=tuple(tensors),
        kv_string_locs=kv_string_locs,
    )


def read_header_bytes(layout: GGUFLayout) -> bytes:
    """All bytes before the tensor-data region, INCLUDING alignment padding.

    Storing padding bytes too means reconstruction is a plain concatenation:
    header_bytes + data_region. No padding math can go wrong at unpack time.
    """
    with open(layout.path, "rb") as f:
        return f.read(layout.data_start)
