# ggufpack

Pack a directory of GGUF quantizations (a typical published model repo: one
F16/BF16 source, an imatrix, and a ladder of quant files) into a compact store,
and reconstruct every original file **bit-exact**.

## How it works

`llama.cpp` quantization is deterministic on one machine + one build
(thread-count invariant). A quant file is therefore mostly *redundant*: it can
be regenerated from the source model plus a small recipe (`llama-quantize`
flags). ggufpack stores, per file, the cheapest plan that provably round-trips:

| plan  | stored                                   | when                                            |
|-------|------------------------------------------|-------------------------------------------------|
| exact | recipe only (+ header patch if needed)   | regeneration reproduces the file                 |
| near  | recipe + zstd XOR delta + header patch   | tensor payloads differ slightly (build variance) |
| blob  | whole file, zstd                         | anything unmatchable — fallback, never lossy     |

The "header patch" exists because GGUF metadata embeds machine-local strings
(`quantize.imatrix.file` holds the imatrix path the publisher used), which
changes the metadata section length. ggufpack stores the original bytes before
the tensor-data region and splices them onto the regenerated tensor payloads;
deltas are computed tensor-region-aligned, never as a whole-file XOR.

Every plan is executed once at pack time and must hash back to the original's
sha256 before it is recorded. At unpack time the output hash is checked again;
on mismatch the file is deleted and ggufpack exits with code 2 rather than
emitting wrong bytes.

## The determinism caveat (read this)

**Packs are machine + build scoped in v0.** Reconstruction runs the same
`llama-quantize` binary that packed; the manifest records the binary's sha256
and unpack warns if it differs. Across different builds/toolchains, published
quants regenerate to within ~0.0–0.35% of bytes (FP-contraction variance) —
that is what the `near` plan absorbs when packing files quantized elsewhere,
but a pack moved to a different machine may fail verification (and will refuse
to emit, by design). Portable packs are pending the fp-contract work.

## Usage

```sh
pip install -e .            # Python 3.11+; deps: gguf, zstandard

ggufpack pack   ./my-model-repo -o my-model.ggufpack --llama-quantize /path/to/llama-quantize
ggufpack stats  my-model.ggufpack
ggufpack unpack my-model.ggufpack Q4_K_M -o restored-Q4_K_M.gguf
ggufpack unpack my-model.ggufpack my-model-Q8_0.gguf -o restored.gguf
ggufpack verify my-model.ggufpack
```

A pack is a directory (single-file archive is v1): `manifest.json` plus
content-addressed zstd blobs under `blobs/`.

`stats` shows what each file costs to store; the pack total is dominated by the
zstd'd source model — the quant ladder itself collapses to recipes and small
deltas.

## What it is not

- Not lossy, ever: unmatchable files are stored whole automatically.
- Not a general GGUF compressor: one file with no source model just becomes a
  zstd blob.
- Not (yet) portable across machines or llama.cpp builds — see caveat above.

## Development

```sh
pip install -e '.[dev]'
pytest -q
ruff check src tests
```

Tests build a tiny (~2 MB) synthetic 1-layer llama model and quantize it with a
real `llama-quantize`; quantize-dependent tests skip if the binary is absent.

## License

MIT
