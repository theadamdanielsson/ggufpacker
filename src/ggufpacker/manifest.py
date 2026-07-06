"""Pack manifest: everything needed to reconstruct every original file bit-exact.

The manifest is plain JSON at <pack>/manifest.json. Design rule: the manifest
holds *facts* (hashes, sizes, plans, recipes, binary identity); all bulk bytes
live in the content-addressed blob store.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "ggufpacker/0"
MANIFEST_NAME = "manifest.json"

# Plans (never lossy — blob fallback is mandatory for anything unmatchable):
#   exact : recipe regenerates the file; optional header patch when only the
#           metadata section differs (tensor payloads identical).
#   near  : recipe + zstd XOR delta over the tensor-data region (+ optional
#           header patch).
#   blob  : whole file stored zstd-compressed (source F16, imatrix, fallbacks).
PLAN_EXACT = "exact"
PLAN_NEAR = "near"
PLAN_BLOB = "blob"

ROLE_SOURCE = "source"
ROLE_IMATRIX = "imatrix"
ROLE_QUANT = "quant"


@dataclass
class FileEntry:
    filename: str
    size: int
    sha256: str
    role: str  # source | imatrix | quant
    plan: str  # exact | near | blob
    recipe: dict[str, Any] | None = None  # qtype/overrides/use_imatrix/cli_flags
    header_blob: str | None = None  # zstd'd original bytes [0, data_start)
    delta_blob: str | None = None  # zstd'd XOR over the tensor-data region
    blob: str | None = None  # zstd'd whole file (plan=blob)
    data_start: int | None = None  # original file's tensor-data offset
    note: str = ""

    def stored_blob_ids(self) -> list[str]:
        return [b for b in (self.header_blob, self.delta_blob, self.blob) if b]


@dataclass
class Manifest:
    format: str
    created: str
    tool_version: str
    quantize: dict[str, str]  # path, sha256, version banner
    files: list[FileEntry] = field(default_factory=list)

    @property
    def source(self) -> FileEntry | None:
        return next((f for f in self.files if f.role == ROLE_SOURCE), None)

    @property
    def imatrix(self) -> FileEntry | None:
        return next((f for f in self.files if f.role == ROLE_IMATRIX), None)

    def find(self, name_or_type: str) -> FileEntry | None:
        """Match by exact filename first, then by recipe qtype (case-insensitive)."""
        for f in self.files:
            if f.filename == name_or_type:
                return f
        want = name_or_type.upper()
        for f in self.files:
            if f.recipe and f.recipe.get("qtype", "").upper() == want:
                return f
        # Last resort: filename suffix match, so `unpack pack Q4_K_L` works
        # even for blob-plan files that have no recipe.
        for f in self.files:
            stem = f.filename[:-5] if f.filename.endswith(".gguf") else f.filename
            if stem.upper().endswith("-" + want):
                return f
        return None

    def save(self, pack_dir: str | Path) -> None:
        data = {
            "format": self.format,
            "created": self.created,
            "tool_version": self.tool_version,
            "quantize": self.quantize,
            "files": [asdict(f) for f in self.files],
        }
        p = Path(pack_dir) / MANIFEST_NAME
        p.write_text(json.dumps(data, indent=1) + "\n")

    @classmethod
    def load(cls, pack_dir: str | Path) -> Manifest:
        p = Path(pack_dir) / MANIFEST_NAME
        if not p.is_file():
            raise FileNotFoundError(f"not a ggufpacker: {p} missing")
        data = json.loads(p.read_text())
        if data.get("format") != FORMAT:
            raise ValueError(f"unsupported pack format {data.get('format')!r}")
        return cls(
            format=data["format"],
            created=data["created"],
            tool_version=data["tool_version"],
            quantize=data["quantize"],
            files=[FileEntry(**f) for f in data["files"]],
        )
