# ggufpack v0 status

Updated: 2026-07-06

## Done

- `pack` / `unpack` / `stats` / `verify` CLI (`ggufpack` entry point), exit
  code 2 on any verification refusal.
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
- Tests: 27 passing (`pytest -q`). Synthetic 1-layer llama F16 (~1.8 MB, built
  with gguf-py's GGUFWriter, accepted by llama-quantize b3821) plus a
  synthetic legacy-format imatrix. Coverage: EXACT round-trip (Q8_0, Q4_K_M),
  NEAR round-trip (flipped payload bytes restored bit-exact),
  metadata-length-change round-trip, metadata+payload combined, imatrix
  round-trip (header patch forced by the embedded path KV), `_L`-style
  override detection (Q6_K + token-embd q8_0), blob fallback (garbage file and
  recipe-less valid GGUF), corrupted-blob refusal with exit 2, manifest-hash
  refusal, unpack-by-type, stats output, layout/delta/recipe unit tests.
  Quantize-dependent tests skip gracefully if the b3821 binary disappears.

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

## Next

- Integration test on a real bartowski repo (Llama-3.2-1B-Instruct ladder)
  once the development agent's pipeline finishes and disk frees — expect EXACT for
  Q8_0/no-search types, NEAR elsewhere per the development measurements; wire it
  as an opt-in `pytest -m realrepo` job.
- Measure the stats money-shot on that repo (expected: ladder collapses to
  F16 + imatrix + KB-scale deltas).
- Then: single-file archive format, and revisit portability.
