# ggufpacker v0 status

Updated: 2026-07-06

## Done

- `pack` / `unpack` / `stats` / `verify` CLI (`ggufpacker` entry point), exit
  code 2 on any verification refusal.
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
- Tests: 46 passing (`pytest -q`). Synthetic 1-layer llama F16 (~1.8 MB, built
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

## Done since: real-repo integration run

Packed the full 17-quant `bartowski/Llama-3.2-1B-Instruct-GGUF` ladder:
**19 files, 16.0 GB -> 1.8 GB (8.7x), manifest 12.4 KB.** Originals deleted,
all 17 quants regenerated from the pack and sha256-verified against the
original Hugging Face files (17/17, 283 s total). One EXACT (`Q8_0`),
sixteen NEAR (deltas 1.8-6.6 MB), zero blob fallbacks — including the
deprecated `Q4_0_4_x` repack types and all `_L`/`_XL` override variants.

## Next

- LRU size cap for the on-demand cache (`cache` prune to a max total size,
  evicting by last-used mtime) — v0.3.
- Wire the real-repo round-trip as an opt-in `pytest -m realrepo` job.
- Single-file `.ggufpack` archive format.
- Revisit portable packs once an upstream deterministic-quantization build
  mode exists.
