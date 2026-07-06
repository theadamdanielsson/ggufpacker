"""`ggufpacker unpack` / `verify` / `stats`: execute plans, never emit bad bytes.

Reconstruction contract (the exact inverse of packer._finalize_plan):

    header  = header_blob bytes if stored, else the regen's own header
    payload = regen tensor-data region, XOR'd with the delta stream if stored
    file    = header + payload

The stored header is the original's bytes [0, data_start) INCLUDING the
alignment padding between the last tensor-info entry and the tensor data, so
concatenation reproduces the original byte-for-byte; the original's tensor-info
offsets (which live inside that stored header) are authoritative, and pack-time
congruence checking guarantees the regen's payload bytes land on exactly those
offsets. Every emitted file's sha256 is checked against the manifest; on
mismatch the output is deleted and ReconstructError is raised (CLI exit 2).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .blobs import BlobCorruptError, BlobStore, sha256_file
from .delta import apply_delta, read_region
from .layout import parse_layout
from .manifest import PLAN_BLOB, PLAN_EXACT, PLAN_NEAR, FileEntry, Manifest
from .quantizer import QuantizeError, Quantizer


class ReconstructError(RuntimeError):
    """Reconstruction failed or verification refused to emit the file."""


def _log(msg: str) -> None:
    print(f"[unpack] {msg}", flush=True)


class Unpacker:
    """Reconstructs files from a pack; caches the extracted source + imatrix
    so `verify` does not re-extract them for every quant."""

    def __init__(self, pack_dir: str | Path, llama_quantize: str | None = None, log=_log):
        self.pack_dir = Path(pack_dir)
        self.manifest = Manifest.load(self.pack_dir)
        self.store = BlobStore(self.pack_dir)
        self.llama_quantize = llama_quantize
        self.log = log
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._source_path: Path | None = None
        self._imatrix_path: Path | None = None
        self._quantizer: Quantizer | None = None

    # -- context management ------------------------------------------------
    def __enter__(self) -> Unpacker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _tmpdir(self) -> Path:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="ggufpacker-unpack-")
        return Path(self._tmp.name)

    # -- plan execution ----------------------------------------------------
    def reconstruct(self, entry: FileEntry, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if entry.plan == PLAN_BLOB:
                self.store.extract_to(entry.blob, out_path)
            elif entry.plan in (PLAN_EXACT, PLAN_NEAR):
                regen = self._regen(entry)
                splice_from_regen(self.store, entry, regen, out_path)
                regen.unlink(missing_ok=True)
            else:
                raise ReconstructError(f"unknown plan {entry.plan!r}")
        except (BlobCorruptError, OSError, ValueError) as e:
            out_path.unlink(missing_ok=True)
            raise ReconstructError(f"{entry.filename}: {e}") from e

        got = sha256_file(out_path)
        if got != entry.sha256:
            out_path.unlink(missing_ok=True)
            raise ReconstructError(
                f"{entry.filename}: reconstructed sha256 {got[:16]}... does not match "
                f"manifest {entry.sha256[:16]}...; refusing to emit. "
                f"(Packs are machine+build-scoped in v0 — is this the same "
                f"llama-quantize build that packed it?)"
            )
        self.log(f"{entry.filename}: OK ({entry.plan}, {entry.size:,} B, sha256 verified)")

    def _regen(self, entry: FileEntry) -> Path:
        if not entry.recipe:
            raise ReconstructError(f"{entry.filename}: plan {entry.plan} but no recipe recorded")
        src = self._materialize_source()
        imx = self._materialize_imatrix() if entry.recipe.get("use_imatrix") else None
        q = self._get_quantizer()
        out = self._tmpdir() / f"regen-{entry.filename}"
        res = q.run(
            src, out, entry.recipe["qtype"],
            imatrix=imx,
            token_embedding_type=entry.recipe.get("token_embedding_type"),
            output_tensor_type=entry.recipe.get("output_tensor_type"),
        )
        if res.returncode != 0:
            raise ReconstructError(
                f"{entry.filename}: llama-quantize failed rc={res.returncode}: "
                f"{res.output_tail[-400:]}"
            )
        return out

    def _get_quantizer(self) -> Quantizer:
        if self._quantizer is None:
            try:
                self._quantizer = Quantizer.locate(
                    self.llama_quantize or self.manifest.quantize.get("path")
                )
            except QuantizeError as e:
                raise ReconstructError(str(e)) from e
            want = self.manifest.quantize.get("sha256")
            if want and self._quantizer.sha256 != want:
                self.log(
                    "WARNING: llama-quantize binary differs from the one that built "
                    f"this pack (sha256 {self._quantizer.sha256[:16]}... vs {want[:16]}...). "
                    "Quantization is build-scoped; reconstruction may fail verification."
                )
        return self._quantizer

    def _materialize_source(self) -> Path:
        if self._source_path is None:
            entry = self.manifest.source
            if entry is None:
                raise ReconstructError("pack has no source model entry")
            p = self._tmpdir() / entry.filename
            self.store.extract_to(entry.blob, p)
            if sha256_file(p) != entry.sha256:
                raise ReconstructError(f"source {entry.filename} failed sha256 after extraction")
            self._source_path = p
        return self._source_path

    def _materialize_imatrix(self) -> Path:
        if self._imatrix_path is None:
            entry = self.manifest.imatrix
            if entry is None:
                raise ReconstructError("recipe needs an imatrix but pack has none")
            p = self._tmpdir() / entry.filename
            self.store.extract_to(entry.blob, p)
            if sha256_file(p) != entry.sha256:
                raise ReconstructError(f"imatrix {entry.filename} failed sha256 after extraction")
            self._imatrix_path = p
        return self._imatrix_path

    # -- verify ------------------------------------------------------------
    def verify_all(self) -> list[tuple[str, str]]:
        """Re-execute every plan; returns [(filename, 'OK'|'FAIL: reason')]."""
        results: list[tuple[str, str]] = []
        scratch = self._tmpdir() / "verify-out"
        for entry in self.manifest.files:
            try:
                self.reconstruct(entry, scratch)
                results.append((entry.filename, "OK"))
            except ReconstructError as e:
                results.append((entry.filename, f"FAIL: {e}"))
            finally:
                scratch.unlink(missing_ok=True)
        return results


def splice_from_regen(store: BlobStore, entry: FileEntry, regen: Path, out_path: Path) -> None:
    """header (stored or regen's) + regen payload (delta-XOR'd if stored)."""
    reg_layout = parse_layout(regen)

    if entry.header_blob:
        header = store.get_bytes(entry.header_blob)
    else:
        with open(regen, "rb") as f:
            header = f.read(reg_layout.data_start)

    # Original payload size = original file size minus its data_start. When a
    # header patch is stored, len(header) IS the original data_start; without
    # one, the regen file byte-identically plays the original's role.
    payload_size = entry.size - len(header)
    if payload_size != reg_layout.data_size:
        raise ValueError(
            f"payload size mismatch: manifest implies {payload_size}, "
            f"regen has {reg_layout.data_size} (wrong/changed quantize build?)"
        )

    with open(out_path, "wb") as out, open(regen, "rb") as f_reg:
        out.write(header)
        if entry.delta_blob:
            apply_delta(
                f_reg, reg_layout.data_start, payload_size,
                store.open_stream(entry.delta_blob), out,
            )
        else:
            for chunk in read_region(f_reg, reg_layout.data_start, payload_size):
                out.write(chunk)


# -- stats -------------------------------------------------------------------

def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= 1024
    return f"{n:,.1f} TB"


def stats_table(pack_dir: str | Path) -> str:
    """The money shot: per-file plan + stored cost, totals, ratio."""
    pack_dir = Path(pack_dir)
    m = Manifest.load(pack_dir)
    store = BlobStore(pack_dir)

    rows: list[tuple[str, str, str, str, str]] = []
    total_orig = 0
    seen_blobs: set[str] = set()
    for e in m.files:
        stored = 0
        for bid in e.stored_blob_ids():
            if bid not in seen_blobs:  # content-addressing can dedup
                seen_blobs.add(bid)
                stored += store.stored_size(bid)
        total_orig += e.size
        plan = e.plan if e.role == "quant" else f"{e.plan} ({e.role})"
        detail = ""
        if e.recipe:
            detail = e.recipe["qtype"]
            if e.recipe.get("token_embedding_type") or e.recipe.get("output_tensor_type"):
                detail += "+ovr"
        rows.append((e.filename, human(e.size), plan, human(stored) if stored else "—", detail))

    manifest_size = (pack_dir / "manifest.json").stat().st_size
    total_stored = manifest_size + sum(store.stored_size(b) for b in seen_blobs)

    headers = ("FILE", "ORIGINAL", "PLAN", "STORED", "RECIPE")
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(5)]
    lines = []
    fmt = "  ".join(
        f"{{:{'<' if i in (0, 2, 4) else '>'}{widths[i]}}}" for i in range(5)
    )
    lines.append(fmt.format(*headers))
    lines.append(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        lines.append(fmt.format(*r))
    lines.append("")
    ratio = total_orig / total_stored if total_stored else 0.0
    lines.append(
        f"{len(m.files)} files: {human(total_orig)} -> {human(total_stored)}, {ratio:.1f}x"
        f"   (manifest {human(manifest_size)})"
    )
    return "\n".join(lines)
