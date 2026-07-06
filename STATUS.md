# ggufpacker v0 status

Updated: 2026-07-06 (v0.3.1)

## Done

- **Hardening (v0.3.1)**, from an adversarial self-audit before launch:
  - `--prune` now verifies the **restore closure** of every EXACT/NEAR file
    before deleting anything: the stored source blob (and imatrix blob when
    the recipe uses one) is fully re-extracted and sha256-compared, not just
    content-address-checked. Previously only the quant's own blobs were
    verified, so bit-rot in the source blob could let prune delete an
    original the pack could no longer regenerate.
  - Corrupt, truncated, or newer-version `manifest.json` is now a clean
    refusal (`ManifestError` -> exit 2 with a message) on every command;
    previously stats/verify/unpack/get crashed with a raw traceback.
  - `pack -o <existing regular file>` refuses cleanly instead of raising
    `NotADirectoryError`.
  - Originals are re-stat'd (size + mtime) immediately before deletion,
    shrinking the verify-to-delete race window to microseconds.
  - Tests: 116 passing (8 new: dependency-blob corruption refusals incl.
    imatrix, missing-source refusal, manifest-error exit codes across all
    read commands, pack-onto-file refusal).

- `pack` / `unpack` / `stats` / `verify` CLI (`ggufpacker` entry point), exit
  code 2 on any verification refusal.
- **Multi-model directories (v0.3)**: files group by tensor identity (the
  set of tensor names + shapes in the GGUF header — types differ between a
  quant and its source, names/shapes do not), and each quant matches the
  F32/F16/BF16 source sharing its identity, so one directory can hold many
  models' ladders. Identical-identity twin sources (base vs abliterated
  finetune: same tensor maps) resolve only on filename-prefix affinity
  (exact model stem first, then longest common prefix, unique winner
  required); anything still ambiguous falls back to a blob with a log line —
  never guessed. imatrix files associate per model by stem containment (a
  lone imatrix beside a lone model still applies unconditionally; models
  without one pack without one). Manifest entries carry per-file
  source/imatrix references (older packs fall back to the single source);
  `stats` shows per-model subtotals when a pack holds several sources.
  Also fixed en route: EXACT recipe-only plans that use an imatrix now pin
  the original header, because the header embeds the machine-local imatrix
  path whose length shifts every offset at unpack time.
- **`pack --prune --keep TYPE[,TYPE...]` (v0.3)**: after the pack completes —
  every EXACT/NEAR plan already round-trip-proven at pack time, pack()
  raising on any failure — the originals are deleted. `--keep` types are
  matched with the same suffix-first logic as unpack-by-type and keep every
  match (one per model in a multi-model pack); source F16/BF16 and imatrix
  files are never deleted; skipped files are never touched; blob-plan files
  ARE deleted (the blob is the file, stored losslessly) but only after a
  full re-extraction + sha256 proof. Belt and braces before any deletion:
  on-disk manifest must equal the in-process one, originals must still hash
  to their packed sha256, referenced blobs must hash to their content
  address — any failure refuses (exit 2) and deletes nothing. Prints each
  deleted file and total space freed. `--prune` without a completed
  verified pack is impossible by construction (it only exists on `pack`).
- **Cache LRU size cap (v0.3)**: `cache prune --max-size N[G|M]` evicts
  least-recently-used cached files (mtime ascending; hits touch mtime)
  until the cache fits, reporting evictions + space freed. Setting
  `GGUFPACKER_CACHE_MAX` applies the same eviction automatically at the end
  of every `get` — after materializing, never evicting the file `get` is
  about to return.
- Cache-on-demand (v0.2): `get` materializes an entry into
  `$GGUFPACKER_CACHE` (default `~/.cache/ggufpacker`)/`<manifest-sha256>`/ and
  prints only the verified absolute path on stdout, so
  `llama-server -m $(ggufpacker get pack Q4_K_M)` composes. Hits are
  rehash-verified (~1-2 s/GB; the rehash is the integrity guarantee) and
  touch mtime; corrupted cache entries are re-materialized, never served;
  misses reconstruct atomically (temp file + rename) through the same
  verify-or-refuse path as `unpack` (exit 2). `exec` runs a command with
  every `{}` replaced by the cached path (appended when absent) and
  propagates the child's exit code. `cache ls` / `cache clear [--pack]`
  inspect and prune the cache.
- Byte-level GGUF layout parser (v2/v3): header end, `general.alignment`
  handling (default 32), tensor-info table, data-region offsets. Own parser
  instead of gguf-py's reader because reconstruction needs an exact byte map.
- Plan machinery, all proven at pack time by executing the plan and hashing:
  - EXACT: recipe only; or recipe + header patch when only metadata differs
    (the `quantize.imatrix.file` local-path case).
  - NEAR: recipe + zstd XOR delta over the tensor-data region + header patch.
    Delta is produced only after verifying layout congruence (same tensor
    names/order/types/dims/rel-offsets and region size), which makes the
    region XOR equivalent to a per-tensor XOR stream. Never whole-file XOR.
  - BLOB: automatic, mandatory fallback — unparseable files, unmatched
    recipes, non-congruent layouts, oversized deltas (>50% of original),
    quantize failures, missing source, missing binary. Never lossy.
- Recipe inference: filename suffix (case-insensitive), bartowski `_L`/`_XL`
  custom-variant map to base types, `token_embd.weight`/`output.weight`
  override detection from the published tensor-type map with quantize retry,
  tensor-type-histogram candidates when the filename is unhelpful.
- Content-addressed zstd blob store (address = sha256 of stored bytes, so
  corruption is detectable without decompression); streaming compression for
  multi-GB files, level 6 for source blobs, 12 for deltas/headers.
- Manifest records: filename/size/sha256/plan/recipe flags per file, plus
  llama-quantize identity (path, sha256, best-effort build banner). Unpack
  warns when the binary differs from the packer's.
- Tests: 108 passing at v0.3.0, 116 at v0.3.1 (`pytest -q`; quantize-dependent
  tests need `GGUFPACKER_TEST_LLAMA_QUANTIZE` or `llama-quantize` on PATH).
  Synthetic 1-layer llama F16 (~1.8 MB, built
  with gguf-py's GGUFWriter, accepted by llama-quantize b3821) plus a
  synthetic legacy-format imatrix. Coverage: EXACT round-trip (Q8_0, Q4_K_M),
  NEAR round-trip (flipped payload bytes restored bit-exact),
  metadata-length-change round-trip, metadata+payload combined, imatrix
  round-trip (header patch forced by the embedded path KV), `_L`-style
  override detection (Q6_K + token-embd q8_0), blob fallback (garbage file and
  recipe-less valid GGUF), corrupted-blob refusal with exit 2, manifest-hash
  refusal, unpack-by-type, stats output, layout/delta/recipe unit tests.
  Cache-on-demand coverage: miss materializes + verifies, hit skips
  reconstruction (counted) + touches mtime, corrupted cache entry
  re-materializes, corrupted pack refuses with exit 2 and prints no path,
  `get` stdout purity (path + newline, byte-exact), `exec` `{}` substitution /
  path-append / exit-code propagation / no-child-on-failure, `cache ls`
  listing and `clear` (all, by pack path, by recorded name after the pack is
  deleted). Quantize-dependent tests skip gracefully if the b3821 binary
  disappears.
  v0.3 coverage: a second synthetic architecture (2 layers, 512 embd) packed
  in one directory with the first — each quant matched to its own source,
  round-tripped, verified, per-model stats grouping, by-type ambiguity;
  identical-tensor-map twin sources resolved by filename prefix (including
  the base/base-abliterated longest-prefix trap) and blob-fallback with a
  log line when no affinity exists; per-model imatrix association (model
  without imatrix must not inherit its neighbor's); pre-0.3 manifest
  back-compat. Prune: keep/source/imatrix/skipped survival, blob-plan
  deletion, freed-bytes report, refusal on changed originals / manifest
  drift / corrupt blobs (deleting nothing), multi-model --keep, exit-code
  mapping. Cache LRU: parse_size, eviction order by mtime, protected file
  never evicted, emptied pack-dir cleanup, CLI reporting, env-var cap on
  hits and misses, invalid cap ignored with a warning.

## Not done (v0 scope cuts, deliberate)

- Single-file `.ggufpack` archive (v0 pack is a directory) — v1.
- Portable packs across machines/llama.cpp builds — pending the fp-contract
  work; v0 refuses (with a clear error) rather than emitting wrong bytes.
- Multi-shard GGUF (`-00001-of-000NN`) regeneration — shards fall back to
  blobs today.
- GGUF-format imatrix files (newer llama.cpp); the legacy `.imatrix`/`.dat`
  format b3821 consumes is supported.
- `pack --jobs N` parallel regen; per-quant progress reporting.
- Recipe search beyond histogram candidates (e.g. trying `--pure`,
  `--leave-output-tensor`).

## Done since: real-repo integration runs

Packed the full 17-quant `bartowski/Llama-3.2-1B-Instruct-GGUF` ladder
(v0.2): **19 files, 16.0 GB -> 1.8 GB (8.7x), manifest 12.4 KB.** Originals
deleted, all 17 quants regenerated from the pack and sha256-verified against
the original Hugging Face files (17/17, 283 s total). One EXACT (`Q8_0`),
sixteen NEAR (deltas 1.8-6.6 MB), zero blob fallbacks — including the
deprecated `Q4_0_4_x` repack types and all `_L`/`_XL` override variants.

**v0.3 multi-model validation (2026-07-06)**: one directory holding TWO real
published models — `bartowski/Llama-3.2-1B-Instruct-GGUF` (f16 + imatrix +
Q8_0 + Q4_K_M) and `bartowski/SmolLM2-135M-Instruct-GGUF` (f16 + imatrix +
Q8_0 + Q4_K_M) — packed with llama-quantize b3821. Every quant matched its
own model's source (manifest `source`/`imatrix` references correct for all
four), zero blob fallbacks, `stats` grouped per model with subtotals
(8 files, 4.8 GB -> 2.0 GB, 2.4x — ratio dominated by the two F16 source
blobs; the four quants alone stored 2.2 GB -> 7.4 MB). All four quants
reconstructed sha256-identical to the Hugging Face originals. Notable:
SmolLM2's repo pins llama.cpp b3991, not our b3821 — its Q8_0 still packed
EXACT (payloads identical, header patch) and its Q4_K_M landed NEAR with a
0.048% delta, so the pinned-build mismatch cost kilobytes, not a blob.
`pack --prune --keep Q4_K_M` was validated separately on real file copies:
it deleted only Q8_0 (138.1 MB freed, reported per file), kept the --keep
quant + source + imatrix, and the deleted file was then restored bit-exact
from the pack. `GGUFPACKER_CACHE_MAX=1M` evicted the LRU cached file at the
end of `get` while never evicting the file being returned.

## Next

- Finetune/variant deltas (e.g. abliterated vs base) via weight-level
  delta compression — different discipline, tracked but not scheduled.
- Single-file `.ggufpack` archive format.
- Revisit portable packs once an upstream deterministic-quantization build
  mode exists.
- Wire the real-repo round-trip as an opt-in `pytest -m realrepo` job.
