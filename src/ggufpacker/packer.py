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

Multi-model directories (v0.3): files are grouped by *tensor identity* — the
set of tensor names + shapes from the GGUF header (types differ between a
quant and its source; names/shapes do not). Each quant is matched to the
F32/F16/BF16 source whose tensor identity it shares. When several sources
share one identity (e.g. base vs abliterated finetune: identical tensor maps),
the match is only taken on filename-prefix affinity (exact model stem, then
longest common prefix); anything still ambiguous is stored as a blob with a
log line — never guessed. imatrix files associate to models the same way.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .blobs import BlobCorruptError, BlobStore, sha256_file
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
    model_stem,
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


@dataclass
class _ModelGroup:
    """One model in the input directory: its chosen source + matched quants."""

    source: _Candidate
    quants: list[_Candidate] = field(default_factory=list)
    imatrix: Path | None = None

    @property
    def stem(self) -> str:
        return model_stem(self.source.path.name)


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

    ggufs, imatrices, skipped = _scan(input_dir)
    if not ggufs:
        raise PackError(f"no .gguf files found in {input_dir}")
    for s in skipped:
        log(f"ignoring non-GGUF/non-imatrix file: {s.name}")

    cands = [
        _Candidate(p, *_try_parse(p))
        for p in ggufs
    ]
    groups, unmatched = _group_models(cands, log)
    if not groups:
        log("WARNING: no F32/F16/BF16 source GGUF found; every quant becomes a blob")
    _associate_imatrices(groups, imatrices, log)

    quantizer: Quantizer | None = None
    if any(g.quants for g in groups):
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

    stored_imatrices: dict[Path, FileEntry] = {}
    with tempfile.TemporaryDirectory(prefix="ggufpacker-regen-") as tmp:
        for group in groups:
            source = group.source
            log(f"source: {source.path.name} ({source.path.stat().st_size:,} B) -> zstd blob")
            manifest.files.append(
                _store_whole(store, source.path, ROLE_SOURCE, SOURCE_ZSTD_LEVEL)
            )
            if group.imatrix is not None and group.imatrix not in stored_imatrices:
                log(f"imatrix: {group.imatrix.name} -> zstd blob")
                entry = _store_whole(store, group.imatrix, ROLE_IMATRIX, DELTA_ZSTD_LEVEL)
                stored_imatrices[group.imatrix] = entry
                manifest.files.append(entry)
            for cand in group.quants:
                entry = _plan_quant(
                    cand, source, group.imatrix, quantizer, store, Path(tmp), log
                )
                entry.source = source.path.name
                if entry.recipe and entry.recipe.get("use_imatrix") and group.imatrix:
                    entry.imatrix = group.imatrix.name
                manifest.files.append(entry)

        for cand, note in unmatched:
            if note:
                manifest.files.append(_blob_fallback(store, cand, note, log))
            else:  # unparseable: let _plan_quant produce the parse-error note
                manifest.files.append(
                    _plan_quant(cand, None, None, quantizer, store, Path(tmp), log)
                )

    for p in imatrices:
        if p not in stored_imatrices:
            log(f"imatrix: {p.name} (not associated with any model) -> zstd blob")
            manifest.files.append(_store_whole(store, p, ROLE_IMATRIX, DELTA_ZSTD_LEVEL))

    manifest.save(out_pack)
    n_by_plan: dict[str, int] = {}
    for f in manifest.files:
        n_by_plan[f.plan] = n_by_plan.get(f.plan, 0) + 1
    log(f"done: {len(manifest.files)} files "
        f"({', '.join(f'{v} {k}' for k, v in sorted(n_by_plan.items()))}) -> {out_pack}")
    return manifest


def _scan(input_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
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
    return ggufs, imatrices, skipped


def _try_parse(path: Path) -> tuple[GGUFLayout | None, str]:
    try:
        return parse_layout(path), ""
    except (GGUFParseError, OSError) as e:
        return None, str(e)


def _identity(layout: GGUFLayout) -> frozenset:
    """Tensor identity: the set of tensor names + shapes. A quant and its
    source share it (ggml types differ; names and dims do not)."""
    return frozenset((t.name, t.dims) for t in layout.tensors)


def _is_source_type(cand: _Candidate) -> bool:
    from .recipe import dominant_type

    return cand.layout is not None and dominant_type(cand.layout) in _SOURCE_TYPES


def _group_models(
    cands: list[_Candidate], log
) -> tuple[list[_ModelGroup], list[tuple[_Candidate, str]]]:
    """Group files by model: pick one source per (tensor identity, model stem)
    — highest precision = largest file — then attach each quant to the source
    whose tensor identity it shares. Ambiguity between identical-identity
    sources (finetune twins) resolves ONLY on filename-prefix affinity;
    otherwise the quant is returned in `unmatched` for blob fallback.

    Returns (groups, unmatched) where unmatched pairs a candidate with a blob
    note ("" = unparseable; the parse error is reported downstream)."""
    chosen: dict[tuple[frozenset, str], _Candidate] = {}
    for c in cands:
        if not _is_source_type(c):
            continue
        key = (_identity(c.layout), model_stem(c.path.name))
        prev = chosen.get(key)
        if prev is None or c.path.stat().st_size > prev.path.stat().st_size:
            chosen[key] = c

    groups = [_ModelGroup(source=c) for c in chosen.values()]
    by_identity: dict[frozenset, list[_ModelGroup]] = {}
    for g in groups:
        by_identity.setdefault(_identity(g.source.layout), []).append(g)

    picked_sources = {id(c) for c in chosen.values()}
    unmatched: list[tuple[_Candidate, str]] = []
    for c in cands:
        if id(c) in picked_sources:
            continue
        if c.layout is None:
            unmatched.append((c, ""))
            continue
        matches = by_identity.get(_identity(c.layout), [])
        if not matches:
            unmatched.append(
                (c, "no F32/F16/BF16 source shares this file's tensor names/shapes")
            )
        elif len(matches) == 1:
            matches[0].quants.append(c)
        else:
            best = _prefix_affinity(model_stem(c.path.name), matches)
            if best is not None:
                log(f"{c.path.name}: {len(matches)} sources share its tensor map; "
                    f"matched to {best.source.path.name} by filename prefix")
                best.quants.append(c)
            else:
                names = ", ".join(sorted(g.source.path.name for g in matches))
                unmatched.append((
                    c,
                    f"ambiguous source: {len(matches)} sources share this tensor map "
                    f"({names}) and the filename gives no unique prefix affinity; "
                    f"storing whole file rather than guessing",
                ))
    return groups, unmatched


def _prefix_affinity(stem: str, groups: list[_ModelGroup]) -> _ModelGroup | None:
    """The group whose source stem the given stem UNIQUELY prefers: exact stem
    equality first, then longest common prefix. None when tied or unrelated —
    the caller must fall back to a blob, never guess."""
    def score(g: _ModelGroup) -> tuple[int, int]:
        lcp = len(os.path.commonprefix([stem, g.stem]))
        return (1 if stem == g.stem else 0, lcp)

    scored = sorted(groups, key=score, reverse=True)
    best_score = score(scored[0])
    if best_score[1] == 0:  # no shared prefix at all: no affinity
        return None
    if len(scored) > 1 and score(scored[1]) == best_score:  # tie: refuse to guess
        return None
    return scored[0]


def _associate_imatrices(groups: list[_ModelGroup], imatrices: list[Path], log) -> None:
    """Attach each model's imatrix by filename-prefix affinity. A single
    imatrix next to a single model applies unconditionally (pre-0.3 behavior,
    covers `imatrix.dat`-style names); with several models an imatrix applies
    only to the model whose files share its prefix — one stem must be a full
    prefix of the other (a merely partial overlap like tinyA/tinyB is not
    affinity). A model without a matched imatrix packs its quants without
    one, as before."""
    if not imatrices:
        return
    if len(groups) == 1 and len(imatrices) == 1:
        groups[0].imatrix = imatrices[0]
        return
    for g in groups:
        def score(p: Path, _stem: str = g.stem) -> tuple[int, int]:
            istem = model_stem(p.name)
            lcp = len(os.path.commonprefix([istem, _stem]))
            return (1 if istem == _stem else 0, lcp)

        candidates = [p for p in imatrices if _stem_contains(model_stem(p.name), g.stem)]
        if not candidates:
            continue
        ranked = sorted(candidates, key=score, reverse=True)
        best = score(ranked[0])
        if len(ranked) > 1 and score(ranked[1]) == best:
            log(f"{g.source.path.name}: several imatrix files match equally; "
                f"packing this model's quants without an imatrix")
            continue
        g.imatrix = ranked[0]


def _stem_contains(a: str, b: str) -> bool:
    """True when one non-empty stem is a full prefix of the other."""
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


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
        if recipe.use_imatrix:
            # The header embeds `quantize.imatrix.file` — a machine-local path
            # that will differ (in LENGTH, shifting every offset) when the
            # imatrix is materialized elsewhere at unpack time. Pin the
            # original header so the plan reproduces anywhere.
            entry.header_blob = store.put_bytes(
                read_header_bytes(cand.layout), DELTA_ZSTD_LEVEL  # type: ignore[arg-type]
            )
            _prove(entry, regen, store)
            log(f"{name}: EXACT (recipe only; header pinned, imatrix path is local)")
        else:
            log(f"{name}: EXACT (recipe only)")
        return entry

    pub_layout = cand.layout
    assert pub_layout is not None
    reg_layout = parse_layout(regen)
    if pub_layout.congruence_key() != reg_layout.congruence_key():
        return None  # caller falls back to blob

    pub_header = read_header_bytes(pub_layout)
    reg_header = read_header_bytes(reg_layout)
    if pub_header != reg_header or recipe.use_imatrix:
        # (also pinned whenever the recipe uses an imatrix — see above)
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


# -- pack --prune ---------------------------------------------------------------


class PruneError(RuntimeError):
    """Prune could not run (e.g. a --keep type matches nothing)."""


class PruneRefused(PruneError):
    """A pre-deletion verification failed; nothing was deleted."""


@dataclass
class PruneResult:
    deleted: list[tuple[str, int]] = field(default_factory=list)  # (filename, bytes)
    kept: list[tuple[str, str]] = field(default_factory=list)  # (filename, reason)

    @property
    def freed(self) -> int:
        return sum(size for _, size in self.deleted)


def prune_originals(
    input_dir: str | Path,
    pack_dir: str | Path,
    manifest: Manifest,
    keep_types: list[str] | tuple[str, ...] = (),
    log=_log,
) -> PruneResult:
    """Delete original quant files after a completed, fully verified pack.

    Only reachable through `pack --prune`, which calls this with the manifest
    a successful pack() just returned in the same process — pack() raises on
    any failed plan, and every EXACT/NEAR plan it records has already been
    round-trip-proven (_prove). So an unverified prune is impossible by
    construction. Belt and braces, this still re-verifies before deleting:

    - the on-disk manifest must equal the one pack() returned;
    - each original file must still hash to its manifest sha256 (it has not
      changed since it was packed);
    - every referenced blob must hash to its content address;
    - blob-plan files are fully re-extracted and compared by sha256 (their
      deletion is safe because the blob IS the file, stored losslessly).

    Never deleted: source F32/F16/BF16 files, imatrix files, anything named
    in keep_types (matched with the same filename/suffix/recipe logic as
    unpack-by-type; a type matching several files — e.g. one per model in a
    multi-model pack — keeps all of them), and files the pack skipped.

    All verification happens before the first deletion: on any failure
    PruneRefused is raised and nothing has been deleted.
    """
    input_dir = Path(input_dir)
    pack_dir = Path(pack_dir)

    on_disk = Manifest.load(pack_dir)
    if on_disk != manifest:
        raise PruneRefused(
            "pack manifest on disk does not match the pack just written; "
            "refusing to prune"
        )

    keep_names: set[str] = set()
    for t in keep_types:
        matches = manifest.find_all(t)
        if not matches:
            raise PruneError(
                f"--keep {t!r} matches no file in the pack; refusing to prune"
            )
        keep_names.update(f.filename for f in matches)

    result = PruneResult()
    to_delete: list[tuple[FileEntry, Path]] = []
    for e in manifest.files:
        if e.role != ROLE_QUANT:
            result.kept.append((e.filename, e.role))
            continue
        if e.filename in keep_names:
            result.kept.append((e.filename, "--keep"))
            continue
        p = input_dir / e.filename
        if p.is_file():
            to_delete.append((e, p))

    store = BlobStore(pack_dir)
    for e, p in to_delete:  # verify EVERYTHING before deleting ANYTHING
        if sha256_file(p) != e.sha256:
            raise PruneRefused(
                f"{e.filename}: file changed since it was packed; refusing to prune"
            )
        for bid in e.stored_blob_ids():
            try:
                store._verify(bid)
            except BlobCorruptError as exc:
                raise PruneRefused(f"{e.filename}: {exc}; refusing to prune") from exc
        if e.plan == PLAN_BLOB:
            _verify_blob_roundtrip(store, e, pack_dir)
        elif e.plan in (PLAN_EXACT, PLAN_NEAR):
            if not e.recipe:
                raise PruneRefused(
                    f"{e.filename}: plan {e.plan} has no recipe; refusing to prune"
                )
            # round-trip proven at pack time (_prove); pack() raises otherwise
        else:
            raise PruneRefused(f"{e.filename}: unknown plan {e.plan!r}; refusing to prune")

    for e, p in to_delete:
        size = p.stat().st_size
        p.unlink()
        result.deleted.append((e.filename, size))
        log(f"pruned {e.filename} ({size:,} B)")
    return result


def _verify_blob_roundtrip(store: BlobStore, entry: FileEntry, pack_dir: Path) -> None:
    """Fully extract a blob-plan file and require its sha256 to match: the
    proof that deleting the original loses nothing."""
    with tempfile.NamedTemporaryFile(dir=pack_dir, delete=False) as tf:
        tmp = Path(tf.name)
    try:
        store.extract_to(entry.blob, tmp)
        if sha256_file(tmp) != entry.sha256:
            raise PruneRefused(
                f"{entry.filename}: stored blob does not reproduce the file; "
                f"refusing to prune"
            )
    finally:
        tmp.unlink(missing_ok=True)
