"""`ggufpacker pack`: turn a directory of GGUF quantizations into a pack.

Per-file plan selection (never lossy; blob fallback is mandatory + automatic):

  EXACT  regen sha256 == original sha256           -> recipe only
  EXACT  tensor payloads identical, metadata not   -> recipe + header patch
  NEAR   payloads differ slightly (fp-contraction  -> recipe + header patch +
         variance between builds, 0.0-0.35% bytes)    zstd XOR delta
  BLOB   anything else (unparseable, recipe never  -> whole file, zstd
         matches, non-congruent layouts, oversized
         delta, quantize failure, no source F16)

Every non-blob plan is *proven at pack time*: the plan is executed against the
regenerated file and the reconstruction's sha256 must equal the original's
before it is recorded. What goes in the manifest has already round-tripped once.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .blobs import BlobStore, sha256_file
from .delta import xor_regions
from .layout import GGUFLayout, GGUFParseError, parse_layout, read_header_bytes
from .manifest import (
    FORMAT,
    PLAN_BLOB,
    PLAN_EXACT,
    PLAN_NEAR,
    ROLE_IMATRIX,
    ROLE_QUANT,
    ROLE_SOURCE,
    FileEntry,
    Manifest,
)
from .quantizer import QuantizeError, Quantizer
from .recipe import (
    Recipe,
    detect_overrides,
    guess_recipe,
    override_cli_name,
    tensor_type_map,
)

# zstd levels: the source F16 barely compresses (random-ish mantissas), so a
# high level buys almost nothing and costs a lot of time -> 6. Deltas/headers
# are small and sparse -> 12 (empirically where the correction deltas compress best).
SOURCE_ZSTD_LEVEL = 6
DELTA_ZSTD_LEVEL = 12

# If the compressed delta exceeds this fraction of the original file, the
# recipe clearly did not reproduce the file (published NEAR deltas measure
# ~0.0-0.35% raw diff); storing the whole file is smaller and more honest.
DELTA_COST_CEILING = 0.5

_SOURCE_TYPES = {"F32", "F16", "BF16"}


class PackError(RuntimeError):
    pass


@dataclass
class _Candidate:
    path: Path
    layout: GGUFLayout | None
    parse_error: str = ""


def _log(msg: str) -> None:
    print(f"[pack] {msg}", flush=True)


def pack(
    input_dir: str | Path,
    out_pack: str | Path,
    llama_quantize: str | None = None,
    log=_log,
) -> Manifest:
    input_dir = Path(input_dir)
    out_pack = Path(out_pack)
    if not input_dir.is_dir():
        raise PackError(f"input is not a directory: {input_dir}")
    if out_pack.exists() and any(out_pack.iterdir()):
        raise PackError(f"output already exists and is not empty: {out_pack}")
    out_pack.mkdir(parents=True, exist_ok=True)
    store = BlobStore(out_pack)

    ggufs, imatrix_path, skipped = _scan(input_dir)
    if not ggufs:
        raise PackError(f"no .gguf files found in {input_dir}")
    for s in skipped:
        log(f"ignoring non-GGUF/non-imatrix file: {s.name}")

    cands = [
        _Candidate(p, *_try_parse(p))
        for p in ggufs
    ]
    source = _pick_source(cands)
    if source is None:
        log("WARNING: no F32/F16/BF16 source GGUF found; every quant becomes a blob")

    quantizer: Quantizer | None = None
    if source is not None and any(c is not source for c in cands):
        try:
            quantizer = Quantizer.locate(llama_quantize)
            log(f"llama-quantize: {quantizer.binary} "
                f"({quantizer.version or 'no version banner'}, sha256 {quantizer.sha256[:16]})")
        except QuantizeError as e:
            log(f"WARNING: {e}; every quant becomes a blob")

    manifest = Manifest(
        format=FORMAT,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        tool_version=__version__,
        quantize=(
            {"path": quantizer.binary, "sha256": quantizer.sha256, "version": quantizer.version}
            if quantizer
            else {}
        ),
    )

    if source is not None:
        log(f"source: {source.path.name} ({source.path.stat().st_size:,} B) -> zstd blob")
        manifest.files.append(_store_whole(store, source.path, ROLE_SOURCE, SOURCE_ZSTD_LEVEL))
    if imatrix_path is not None:
        log(f"imatrix: {imatrix_path.name} -> zstd blob")
        manifest.files.append(_store_whole(store, imatrix_path, ROLE_IMATRIX, DELTA_ZSTD_LEVEL))

    with tempfile.TemporaryDirectory(prefix="ggufpacker-regen-") as tmp:
        for cand in cands:
            if cand is source:
                continue
            entry = _plan_quant(
                cand, source, imatrix_path, quantizer, store, Path(tmp), log
            )
            manifest.files.append(entry)

    manifest.save(out_pack)
    n_by_plan: dict[str, int] = {}
    for f in manifest.files:
        n_by_plan[f.plan] = n_by_plan.get(f.plan, 0) + 1
    log(f"done: {len(manifest.files)} files "
        f"({', '.join(f'{v} {k}' for k, v in sorted(n_by_plan.items()))}) -> {out_pack}")
    return manifest


def _scan(input_dir: Path) -> tuple[list[Path], Path | None, list[Path]]:
    ggufs: list[Path] = []
    imatrices: list[Path] = []
    skipped: list[Path] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() == ".gguf":
            ggufs.append(p)
        elif "imatrix" in p.name.lower():
            imatrices.append(p)
        else:
            skipped.append(p)
    imatrix = imatrices[0] if imatrices else None
    if len(imatrices) > 1:
        # v0: single imatrix per pack; extras are still preserved as-is via skip
        # (they are not gguf files, so they are simply not packed).
        skipped.extend(imatrices[1:])
    return ggufs, imatrix, skipped


def _try_parse(path: Path) -> tuple[GGUFLayout | None, str]:
    try:
        return parse_layout(path), ""
    except (GGUFParseError, OSError) as e:
        return None, str(e)


def _pick_source(cands: list[_Candidate]) -> _Candidate | None:
    """Highest-precision GGUF = dominant >=2D tensor type in {F32, F16, BF16};
    ties broken by file size (a bigger file carries more precision)."""
    from .recipe import dominant_type

    best: _Candidate | None = None
    for c in cands:
        if c.layout is None:
            continue
        if dominant_type(c.layout) in _SOURCE_TYPES:
            if best is None or c.path.stat().st_size > best.path.stat().st_size:
                best = c
    return best


def _store_whole(store: BlobStore, path: Path, role: str, level: int) -> FileEntry:
    blob_id = store.put_file(path, level)
    return FileEntry(
        filename=path.name,
        size=path.stat().st_size,
        sha256=sha256_file(path),
        role=role,
        plan=PLAN_BLOB,
        blob=blob_id,
    )


def _blob_fallback(store: BlobStore, cand: _Candidate, note: str, log) -> FileEntry:
    log(f"{cand.path.name}: BLOB ({note})")
    entry = _store_whole(store, cand.path, ROLE_QUANT, SOURCE_ZSTD_LEVEL)
    entry.note = note
    return entry


def _plan_quant(
    cand: _Candidate,
    source: _Candidate | None,
    imatrix_path: Path | None,
    quantizer: Quantizer | None,
    store: BlobStore,
    tmp: Path,
    log,
) -> FileEntry:
    name = cand.path.name
    if cand.layout is None:
        return _blob_fallback(store, cand, f"not parseable as GGUF: {cand.parse_error}", log)
    if source is None or quantizer is None:
        return _blob_fallback(store, cand, "no source model or no llama-quantize", log)

    guess = guess_recipe(name, cand.layout)
    if guess is None:
        return _blob_fallback(store, cand, "no recipe candidates (name + histogram)", log)

    pub_map = tensor_type_map(cand.layout)
    regen = tmp / f"regen-{name}"
    try:
        for qtype in guess.candidates:
            recipe, ok = _try_candidate(
                qtype, cand, pub_map, source, imatrix_path, quantizer, regen, log
            )
            if ok:
                entry = _finalize_plan(cand, regen, recipe, store, log)
                if entry is not None:
                    return entry
                return _blob_fallback(
                    store, cand, f"recipe {qtype} matched types but layouts not congruent", log
                )
        origins = f"{guess.origin}: {', '.join(guess.candidates)}"
        return _blob_fallback(store, cand, f"no recipe candidate matched ({origins})", log)
    finally:
        regen.unlink(missing_ok=True)


def _try_candidate(
    qtype: str,
    cand: _Candidate,
    pub_map: dict[str, str],
    source: _Candidate,
    imatrix_path: Path | None,
    quantizer: Quantizer,
    regen: Path,
    log,
) -> tuple[Recipe | None, bool]:
    """Run one base-type candidate (+ token-embd/output overrides if the
    published tensor-type map calls for them). True iff the regenerated
    tensor-type map matches the published one exactly."""
    recipe = Recipe(qtype=qtype, use_imatrix=imatrix_path is not None)
    res = quantizer.run(source.path, regen, qtype,
                        imatrix=imatrix_path if recipe.use_imatrix else None)
    if res.returncode != 0:
        log(f"{cand.path.name}: quantize {qtype} failed rc={res.returncode}")
        return None, False

    reg_map = tensor_type_map(parse_layout(regen))
    overrides = detect_overrides(pub_map, reg_map)
    if overrides is None:
        return None, False
    if overrides:
        recipe.token_embedding_type = (
            override_cli_name(overrides["token_embd.weight"])
            if "token_embd.weight" in overrides else None
        )
        recipe.output_tensor_type = (
            override_cli_name(overrides["output.weight"])
            if "output.weight" in overrides else None
        )
        log(f"{cand.path.name}: retry {qtype} with overrides "
            f"tok={recipe.token_embedding_type} out={recipe.output_tensor_type}")
        res = quantizer.run(
            source.path, regen, qtype,
            imatrix=imatrix_path if recipe.use_imatrix else None,
            token_embedding_type=recipe.token_embedding_type,
            output_tensor_type=recipe.output_tensor_type,
        )
        if res.returncode != 0:
            log(f"{cand.path.name}: override quantize failed rc={res.returncode}")
            return None, False
        reg_map = tensor_type_map(parse_layout(regen))
        if detect_overrides(pub_map, reg_map) != {}:
            return None, False
    return recipe, True


def _finalize_plan(
    cand: _Candidate, regen: Path, recipe: Recipe, store: BlobStore, log
) -> FileEntry | None:
    """Choose EXACT/NEAR for an accepted regen; returns None -> blob fallback.

    Reconstruction contract (mirrors unpacker.py):
        header  = stored original bytes [0, data_start) if header_blob else regen's
        payload = regen data region, XOR delta if delta_blob
        file    = header + payload
    The original's tensor-info (living inside the stored header) stays
    authoritative for offsets; congruence guarantees regen payload bytes land
    on exactly those offsets.
    """
    name = cand.path.name
    pub_sha = sha256_file(cand.path)
    entry = FileEntry(
        filename=name,
        size=cand.path.stat().st_size,
        sha256=pub_sha,
        role=ROLE_QUANT,
        plan=PLAN_EXACT,
        recipe={
            "qtype": recipe.qtype,
            "token_embedding_type": recipe.token_embedding_type,
            "output_tensor_type": recipe.output_tensor_type,
            "use_imatrix": recipe.use_imatrix,
            "cli_flags": recipe.cli_flags(),
        },
        data_start=cand.layout.data_start,  # type: ignore[union-attr]
    )

    if sha256_file(regen) == pub_sha:
        log(f"{name}: EXACT (recipe only)")
        return entry

    pub_layout = cand.layout
    assert pub_layout is not None
    reg_layout = parse_layout(regen)
    if pub_layout.congruence_key() != reg_layout.congruence_key():
        return None  # caller falls back to blob

    pub_header = read_header_bytes(pub_layout)
    reg_header = read_header_bytes(reg_layout)
    if pub_header != reg_header:
        entry.header_blob = store.put_bytes(pub_header, DELTA_ZSTD_LEVEL)

    size = pub_layout.data_size
    nonzero = 0
    delta_id: str | None = None
    with open(pub_layout.path, "rb") as f_pub, open(regen, "rb") as f_reg:

        def gen():
            nonlocal nonzero
            for x, nz in xor_regions(
                f_pub, pub_layout.data_start, f_reg, reg_layout.data_start, size
            ):
                nonzero += nz
                yield x

        delta_id = store.put_stream(gen(), DELTA_ZSTD_LEVEL)

    if nonzero == 0:
        # Payloads identical; only metadata differed (e.g. quantize.imatrix.file
        # embeds a local path). Drop the all-zero delta blob we just wrote.
        if delta_id:
            _drop_unshared_blob(store, delta_id, keep=entry.header_blob)
        log(f"{name}: EXACT (payloads identical, header patch "
            f"{store.stored_size(entry.header_blob):,} B)" if entry.header_blob
            else f"{name}: EXACT (payloads identical)")
    else:
        entry.plan = PLAN_NEAR
        entry.delta_blob = delta_id
        cost = store.stored_size(delta_id)
        if cost > entry.size * DELTA_COST_CEILING:
            _drop_unshared_blob(store, delta_id, keep=None)
            if entry.header_blob:
                _drop_unshared_blob(store, entry.header_blob, keep=None)
            log(f"{name}: delta too large ({cost:,} B for a {entry.size:,} B file)")
            return None
        pct = 100.0 * nonzero / size if size else 0.0
        log(f"{name}: NEAR (diff {pct:.4f}% of payload, delta {cost:,} B)")

    _prove(entry, regen, store)
    return entry


def _drop_unshared_blob(store: BlobStore, blob_id: str, keep: str | None) -> None:
    """Remove a blob we decided not to reference (content addressing makes this
    safe only because pack writes serially and nothing else references it yet)."""
    if blob_id != keep:
        store.path_of(blob_id).unlink(missing_ok=True)


def _prove(entry: FileEntry, regen: Path, store: BlobStore) -> None:
    """Execute the recorded plan against the regen and require a sha256 match.

    This is the pack-time round-trip proof: if this passes, unpack can only
    fail through non-determinism of the quantize step itself (same machine +
    same binary => established deterministic).
    """
    from .unpacker import splice_from_regen

    with tempfile.NamedTemporaryFile(dir=regen.parent, delete=False) as tf:
        tmp_out = Path(tf.name)
    try:
        splice_from_regen(store, entry, regen, tmp_out)
        got = sha256_file(tmp_out)
        if got != entry.sha256:
            raise PackError(
                f"internal error: plan for {entry.filename} does not round-trip "
                f"({got[:16]} != {entry.sha256[:16]})"
            )
    finally:
        tmp_out.unlink(missing_ok=True)
