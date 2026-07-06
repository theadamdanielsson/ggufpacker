"""Cache-on-demand: materialize pack entries into a local cache, reuse them.

`ggufpacker get` resolves an entry exactly like `unpack` (exact filename,
then quant type, then filename-suffix match) and materializes it into

    $GGUFPACKER_CACHE (default ~/.cache/ggufpacker)/<pack identity>/<filename>

where the pack identity is the sha256 of the pack's manifest.json bytes, so a
re-packed store never aliases stale cache entries. The absolute cached path is
the function's result — the CLI prints it (and nothing else) on stdout so it
composes: `llama-server -m $(ggufpacker get pack Q4_K_M)`.

Integrity contract, same as unpack's: no unverified path is ever emitted.

- Cache hit: the cached file is re-hashed and compared against the manifest
  sha256 before the path is returned. That rehash costs ~1-2 s per GB and IS
  the integrity guarantee — the path you hand to llama-server has just been
  proven to hold the original bytes. Hits also touch the file's mtime so
  `cache ls` shows recency.
- Corrupted cached file: discarded and re-materialized, never served.
- Cache miss: reconstructed through the same verify-or-refuse machinery as
  `unpack` (ReconstructError, CLI exit 2, on any mismatch), written to a temp
  file in the cache directory and atomically renamed into place.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from .blobs import sha256_file
from .manifest import MANIFEST_NAME, Manifest
from .unpacker import Unpacker, human

META_NAME = ".pack.json"  # per-pack cache metadata; never a model filename


def cache_root() -> Path:
    env = os.environ.get("GGUFPACKER_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "ggufpacker"


def pack_identity(pack_dir: str | Path) -> str:
    """Cache identity of a pack = sha256 of its manifest.json bytes."""
    return sha256_file(Path(pack_dir) / MANIFEST_NAME)


def _log(msg: str) -> None:
    # stdout is reserved for the cached path; all progress goes to stderr.
    print(f"[get] {msg}", file=sys.stderr, flush=True)


def get(pack_dir: str | Path, name: str, llama_quantize: str | None = None, log=_log) -> Path:
    """Materialize one entry into the cache; return its absolute verified path.

    Raises FileNotFoundError (no pack), LookupError (no matching entry) or
    ReconstructError (reconstruction failed verification; nothing emitted).
    """
    pack_dir = Path(pack_dir)
    manifest = Manifest.load(pack_dir)
    entry = manifest.find(name)
    if entry is None:
        names = ", ".join(f.filename for f in manifest.files)
        raise LookupError(f"no file matching {name!r} in pack (have: {names})")

    ident = pack_identity(pack_dir)
    cdir = cache_root() / ident
    cached = cdir / entry.filename

    if cached.is_file():
        if sha256_file(cached) == entry.sha256:
            os.utime(cached)  # recency for `cache ls`
            log(f"{entry.filename}: cache hit ({entry.size:,} B, sha256 verified)")
            return cached.resolve()
        log(f"{entry.filename}: cached copy failed sha256; discarding and re-materializing")
        cached.unlink()

    cdir.mkdir(parents=True, exist_ok=True)
    _write_meta(cdir, pack_dir, ident)
    tmp = cdir / f".tmp-{os.getpid()}-{entry.filename}"
    try:
        with Unpacker(pack_dir, llama_quantize=llama_quantize, log=log) as u:
            u.reconstruct(entry, tmp)  # sha256-verified or ReconstructError
        os.replace(tmp, cached)
    finally:
        tmp.unlink(missing_ok=True)
    return cached.resolve()


def _write_meta(cdir: Path, pack_dir: Path, ident: str) -> None:
    (cdir / META_NAME).write_text(
        json.dumps({"pack_path": str(pack_dir.resolve()), "manifest_sha256": ident}) + "\n"
    )


def _read_meta(cdir: Path) -> dict | None:
    try:
        return json.loads((cdir / META_NAME).read_text())
    except (OSError, ValueError):
        return None


# -- cache ls ------------------------------------------------------------------

def ls_table() -> str:
    """Table of cached files: pack, filename, size, last-used mtime + total."""
    rows: list[tuple[str, str, str, str]] = []
    total = 0
    root = cache_root()
    if root.is_dir():
        for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
            meta = _read_meta(cdir)
            recorded = (meta or {}).get("pack_path", "")
            label = Path(recorded).name if recorded else cdir.name[:12]
            for f in sorted(p for p in cdir.iterdir()
                            if p.is_file() and not p.name.startswith(".")):
                st = f.stat()
                total += st.st_size
                rows.append((
                    label, f.name, human(st.st_size),
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                ))
    if not rows:
        return f"cache is empty ({root})"

    headers = ("PACK", "FILE", "SIZE", "LAST USED")
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(4)]
    fmt = "  ".join(f"{{:{'<' if i in (0, 1) else '>'}{widths[i]}}}" for i in range(4))
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*r) for r in rows)
    lines.append("")
    lines.append(f"{len(rows)} file(s), {human(total)} total   ({root})")
    return "\n".join(lines)


# -- cache clear ---------------------------------------------------------------

def clear(pack: str | None = None) -> int:
    """Remove all cached packs, or just one; returns how many were removed."""
    root = cache_root()
    if not root.is_dir():
        return 0
    if pack is None:
        dirs = [p for p in root.iterdir() if p.is_dir()]
    else:
        dirs = _match_pack_dirs(root, pack)
    for d in dirs:
        shutil.rmtree(d)
    return len(dirs)


def _match_pack_dirs(root: Path, pack: str) -> list[Path]:
    """Resolve a `--pack` argument: a live pack directory (hash its manifest),
    a recorded pack path/name, or a cache identity (prefix >= 8 chars)."""
    p = Path(pack)
    if (p / MANIFEST_NAME).is_file():
        d = root / pack_identity(p)
        return [d] if d.is_dir() else []
    matches = []
    for d in (q for q in root.iterdir() if q.is_dir()):
        meta = _read_meta(d)
        recorded = (meta or {}).get("pack_path", "")
        if pack in (recorded, Path(recorded).name, d.name) or (
            len(pack) >= 8 and d.name.startswith(pack)
        ):
            matches.append(d)
    return matches
