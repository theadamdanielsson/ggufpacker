# ggufpacker v0 status

Updated: 2026-07-07 (v0.5.0)

## Done

- **Conversion attestations + the chain (v0.5.0)**: the sibling predicate
  `gguf-conversion/v0` (safetensors snapshot -> F16), licensed by the
  conversion determinism proof (gguf-quant-determinism conversion.yml:
  convert_hf_to_gguf at a pinned commit is bit-reproducible across
  OS/arch/Python/numpy/torch, dense + MoE, resharded inputs).
  - `attest-conversion` proves-then-emits: re-runs the converter on the
    snapshot, byte-compares, refuses otherwise. The statement pins the
    WHOLE input closure — every non-hidden snapshot file (name, sha256,
    size), the directory name verbatim (parsed into general.* header
    fields), the converter identity (script sha256, gitRef, Python,
    key library versions). `--source-uri/--source-revision` anchor the
    snapshot to its published HF revision.
  - `verify-conversion` re-derives-or-refuses: closure digests checked
    first, files OUTSIDE the attested closure refuse too (an unattested
    README.md changes the output), then re-convert + byte-compare, with
    the same TAMPER-EVIDENT vs INCONCLUSIVE split and `--check-source`
    (primary .safetensors vs HF `/raw/` LFS pointer).
  - `verify-chain` = the trust-root deliverable: requires the quant
    statement's baseModel digest to EQUAL the conversion statement's
    subject digest (content links the chain; checked on parsed statements
    before any subprocess), then re-derives both edges. Exit 0 = quant
    bytes trace to the published snapshot; `--check-source` anchors the
    root to huggingface.co.
  - Same 0.4.2 hardening discipline on the new loader: traversal-safe
    closure names, buildIdentity required, recipe.command validated and
    the recovered `--model-name` pattern-checked before it reaches argv,
    `reDerivedDigest == subject` enforced.
  - 34 new tests (182 total), hermetic via a fake converter that mirrors
    the real input-closure semantics (dir name, file bytes, --model-name);
    chain tests run the real llama-quantize. Spec:
    docs/conversion-attestation.md.

- **Attestation hardening (v0.4.2)**, from an adversarial red-team of 0.4.1:
  - Identity anchoring: `attest --source-uri/--source-download-url` record
    the base model's canonical identity (purl + resolve URL);
    `verify-attestation --check-source` requires the attested digest to
    equal the published file's (HF `/raw/` LFS-pointer check, no download).
    Closes the laundering gap where an attacker honestly attests a quant of
    their own poisoned F16 under a familiar filename — previously the
    statement proved derivation-from-those-bytes with the root identity
    only implied by the name. Docs now state the distinction explicitly.
  - Strict statement validation before anything touches the filesystem or
    argv: safe sibling-only file names (no separators/`..`/absolute),
    hex-validated digests, `quantType`/override types pattern-checked (no
    flag injection), `reDerivedDigest == subject` enforced (was written but
    never checked), malformed nested fields refuse cleanly instead of
    raising KeyError tracebacks.
  - Re-derivation bounded by `--timeout` (default 3600 s) — statements are
    untrusted input; a hanging recipe is killed and reported as exit 1.
  - Mismatch reporting split: same-binary mismatch is TAMPER-EVIDENT (the
    attesting binary itself cannot reproduce the attested bytes); a
    different-binary mismatch is INCONCLUSIVE, qualified by whether the
    attester claimed a deterministic build. Previously one message blamed
    the build in both cases.
  - Spec correction: `sigstore attest` CLI cannot carry custom predicates
    (SLSA types only) — signing goes through the sigstore-python API or
    `cosign attest-blob`; docs fixed.
  - Known deferral: `deterministic: true` remains attester-asserted (a
    known-answer self-test needs per-gitRef golden hashes; tracked in Next).
  - 22 new tests (148 total); both published real statements load clean
    under the stricter parser.

- **Derivation attestations (v0.4.0)**: `attest` proves a quant derives
  bit-exactly from a source (re-derive + sha256 match, refuse otherwise) and
  emits an in-toto Statement v1 with a custom predicate
  (`.../attestation/gguf-derivation/v0`); `verify-attestation` re-derives and
  byte-compares against the attested digest (exit 0 proven / 2 refused).
  Spec in docs/derivation-attestation.md. Design informed by adversarial
  research (2026-07-06): no existing spec/tool does bit-exact recipe
  re-derivation of weights (OMS/sigstore = signing; Cisco Model Provenance
  Kit = statistical detection; two 2026 arXiv papers call for exactly this
  and ship nothing); in-toto custom predicate under own namespace confirmed
  as the correct, non-redundant format vs SLSA (re-derivability vs
  trust-the-builder). Portability rule discovered en route: llama-quantize
  embeds the --imatrix argument string verbatim in the output header, so
  attestable artifacts must be quantized with the bare cwd-relative imatrix
  filename; attest enforces this and hints the embedded path on refusal.
  Statements are unsigned by design (DSSE/sigstore wrapping is out of scope
  for v0). 10 new tests (126 total).

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

- `deterministic: true` as a proven claim: a known-answer self-test in
  `attest` (tiny pinned source + recipe -> per-gitRef golden hash) so the
  flag can only be written by a binary that just demonstrated determinism.
- `attest --sign` (optional `ggufpacker[sign]` extra): sigstore-python
  `StatementBuilder` -> keyless DSSE bundle `<model>.gguf-derivation.sigstore.json`
  (the `sigstore attest` CLI cannot carry custom predicates).
- Conversion attestation extensions: per-file `--check-source` anchoring
  (v0 anchors the primary .safetensors only), a lockfile digest in
  buildIdentity, and PyTorch `.bin` snapshots if ever CI-proven.
- Reusable `verify-gguf` GitHub Action (builds llama.cpp from pinned source,
  so no prebuilt binary in the trust path) + a live shields endpoint badge
  fed by scheduled verification runs.
- Header dedup: the demo pack's 17 NEAR files store 32.2 MB of zstd'd
  original headers (~1.9 MB each, dominated by the embedded tokenizer) next
  to 32.1 MB of actual deltas — the headers are near-identical across the
  ladder, so delta-coding them against one base header would nearly halve
  the non-source stored bytes.
- Finetune/variant deltas (e.g. abliterated vs base) via weight-level
  delta compression — different discipline, tracked but not scheduled.
- Single-file `.ggufpack` archive format.
- Revisit portable packs once an upstream deterministic-quantization build
  mode exists.
- Wire the real-repo round-trip as an opt-in `pytest -m realrepo` job.
