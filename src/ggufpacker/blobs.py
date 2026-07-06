"""Content-addressed blob store.

Blobs live under <pack>/blobs/<sha256-of-stored-bytes>. The address is the hash
of the bytes ON DISK (i.e. post-compression), so integrity can be verified
without decompressing. Raw-content hashes live in the manifest where relevant.

All blobs are zstd-compressed streams; even barely-compressible payloads (F16
tensor data, imatrix) go through zstd so the reader has exactly one code path.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import zstandard

_CHUNK = 8 * 1024 * 1024


class BlobCorruptError(RuntimeError):
    """Stored blob bytes do not hash to their content address."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


class BlobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def dir(self) -> Path:
        return self.root / "blobs"

    def path_of(self, blob_id: str) -> Path:
        return self.dir / blob_id

    def _finalize(self, tmp: Path) -> str:
        blob_id = sha256_file(tmp)
        dest = self.path_of(blob_id)
        if dest.exists():
            tmp.unlink()  # content-addressed: identical bytes already stored
        else:
            os.replace(tmp, dest)
        return blob_id

    def put_bytes(self, data: bytes, level: int) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".tmp-{os.getpid()}"
        tmp.write_bytes(zstandard.ZstdCompressor(level=level).compress(data))
        return self._finalize(tmp)

    def put_stream(self, chunks: Iterator[bytes] | Iterable[bytes], level: int) -> str:
        """Compress an iterator of raw chunks into the store (used for deltas)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".tmp-{os.getpid()}"
        cctx = zstandard.ZstdCompressor(level=level)
        with open(tmp, "wb") as fout:
            with cctx.stream_writer(fout, closefd=False) as writer:
                for c in chunks:
                    writer.write(c)
        return self._finalize(tmp)

    def put_file(self, src: str | Path, level: int) -> str:
        """Stream-compress a (possibly multi-GB) file into the store."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".tmp-{os.getpid()}"
        cctx = zstandard.ZstdCompressor(level=level, threads=-1)
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            cctx.copy_stream(fin, fout, read_size=_CHUNK, write_size=_CHUNK)
        return self._finalize(tmp)

    def _verify(self, blob_id: str) -> Path:
        p = self.path_of(blob_id)
        if not p.exists():
            raise BlobCorruptError(f"blob {blob_id[:16]}... is missing")
        if sha256_file(p) != blob_id:
            raise BlobCorruptError(
                f"blob {blob_id[:16]}... is corrupt (stored-bytes hash mismatch)"
            )
        return p

    def get_bytes(self, blob_id: str) -> bytes:
        """Decompress a small blob fully into memory, verifying integrity first."""
        p = self._verify(blob_id)
        return zstandard.ZstdDecompressor().decompress(
            p.read_bytes(), max_output_size=1 << 34
        )

    def open_stream(self, blob_id: str) -> Iterator[bytes]:
        """Yield decompressed chunks of a large blob, verifying integrity first."""
        p = self._verify(blob_id)
        dctx = zstandard.ZstdDecompressor()
        with open(p, "rb") as f, dctx.stream_reader(f) as reader:
            while chunk := reader.read(_CHUNK):
                yield chunk

    def extract_to(self, blob_id: str, dest: str | Path) -> None:
        with open(dest, "wb") as out:
            for chunk in self.open_stream(blob_id):
                out.write(chunk)

    def stored_size(self, blob_id: str) -> int:
        return self.path_of(blob_id).stat().st_size
