# GGUF conversion attestation (v0)

The claim one edge up the chain from [derivation attestation](derivation-attestation.md):
**this F16 GGUF derives bit-exactly from that safetensors snapshot via
convert_hf_to_gguf.py.** Chained with a quant attestation — linked by the F16
digest, same bytes on both sides — the two statements cover the whole path
from the published safetensors (the Hugging Face trust root) to the deployed
quant:

```
ggufpacker attest-conversion model-f16.gguf --source-dir Llama-3.2-1B-Instruct \
    --llama-cpp-dir ~/llama.cpp
ggufpacker verify-conversion model-f16.gguf.conversion.json --llama-cpp-dir ~/llama.cpp
ggufpacker verify-chain model-Q4_K_M.gguf.derivation.json model-f16.gguf.conversion.json \
    --check-source
```

Same discipline as the quant path: `attest-conversion` refuses (exit 2)
unless it has just re-run the conversion and matched the file's sha256;
`verify-conversion` refuses unless re-derivation reproduces the attested
digest; `verify-chain` refuses unless the statements are digest-linked AND
both edges re-derive. Nothing is ever asserted that was not just proven.

## Why this is possible at all

convert_hf_to_gguf.py at a pinned llama.cpp commit is bit-reproducible —
measured, not assumed: across OS, architecture, Python version, numpy/torch
versions, dense and MoE models, and resharded inputs, on public CI in
[gguf-quant-determinism](https://github.com/theadamdanielsson/gguf-quant-determinism)
(conversion workflow, hard asserts on exact sha256s). Unlike quantization it
needs no build flag: the conversion is integer/copy work by construction.

## The input closure

The conversion reads more than the weights, and everything it reads is part
of the recipe. Three findings from the determinism campaign shape the
predicate:

1. **The snapshot directory NAME is an input.** `general.name`, `basename`,
   `finetune` and `size_label` are parsed from it. Convert in a directory
   named exactly the repo basename, or the header differs.
2. **README.md is an input.** The model card is read into KV metadata
   (license, languages, organization). A snapshot with a different README
   produces a different file.
3. **Tensor order is an input.** The converter emits tensors in input
   iteration order; a permuted shard index permutes the output with zero
   content differences. Pinning every snapshot file (including
   `model.safetensors.index.json`) pins the order.

So the statement pins the *whole snapshot*: every non-hidden file by name,
sha256 and size, plus the directory name verbatim. Verification refuses if
any closure file is missing or differs — and also if the snapshot contains a
file the attester never saw, because that too can change the output.

## The statement

An unsigned [in-toto Statement v1](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md)
with predicate type:

```
https://github.com/theadamdanielsson/ggufpacker/attestation/gguf-conversion/v0
```

```json
{
 "_type": "https://in-toto.io/Statement/v1",
 "subject": [
  {"name": "Llama-3.2-1B-Instruct-f16.gguf",
   "digest": {"sha256": "<output sha256>"}}
 ],
 "predicateType": "https://github.com/theadamdanielsson/ggufpacker/attestation/gguf-conversion/v0",
 "predicate": {
  "source": {
   "directoryName": "Llama-3.2-1B-Instruct",
   "uri": "pkg:huggingface/meta-llama/Llama-3.2-1B-Instruct",
   "revision": "<commit>",
   "files": [
    {"name": "README.md", "digest": {"sha256": "..."}, "sizeBytes": 35887},
    {"name": "config.json", "digest": {"sha256": "..."}, "sizeBytes": 877},
    {"name": "model.safetensors", "digest": {"sha256": "..."}, "sizeBytes": 2471645608},
    {"name": "tokenizer.json", "digest": {"sha256": "..."}, "sizeBytes": 9085657}
   ]
  },
  "recipe": {
   "transform": "convert_hf_to_gguf",
   "outputType": "f16",
   "command": "python convert_hf_to_gguf.py --outtype f16 --outfile <output> <sourceDir>",
   "conventions": [
    "the snapshot directory is named exactly the repo basename; general.name/basename/finetune/size_label derive from it",
    "README.md, when present, is part of the input closure (read into KV metadata)",
    "tensor order follows input iteration order; a permuted shard index permutes the output"
   ]
  },
  "builder": {
   "tool": "ggml-org/llama.cpp/convert_hf_to_gguf.py",
   "buildIdentity": {
    "converterSha256": "<sha256 of convert_hf_to_gguf.py>",
    "gitRef": "<llama.cpp commit>",
    "python": "3.12.7",
    "keyLibraries": {"numpy": "2.5.1", "torch": "2.12.1", "gguf": "0.17.0"}
   }
  },
  "reproducibility": {
   "deterministic": true,
   "determinismEvidence": "https://github.com/theadamdanielsson/gguf-quant-determinism/blob/main/.github/workflows/conversion.yml",
   "reDerivedDigest": {"sha256": "<output sha256>"}
  },
  "attestedBy": "ggufpacker 0.5.0",
  "producedAt": "2026-07-07T12:00:00Z"
 }
}
```

Field semantics beyond the [derivation predicate](derivation-attestation.md)'s:

- **source.files** — the input closure, relative POSIX names inside the
  snapshot, each digest-pinned. Verifiers reject absolute names, `..`, and
  duplicates; files are only ever looked up inside
  `<search dir>/<directoryName>`.
- **source.directoryName** — recorded verbatim because it is parsed into
  header fields. Verification requires the snapshot to carry this exact name.
- **source.uri / revision** — the identity anchor, purl plus the published
  revision. `verify-conversion --check-source` resolves the primary
  `.safetensors` file at
  `huggingface.co/<repo>/resolve/<revision>/<name>` (via the `/raw/` LFS
  pointer, no download) and requires its digest to equal the attested one.
  Without it the statement proves derivation from the attested bytes, not
  from a canonical published snapshot.
- **builder.buildIdentity** — `converterSha256` identifies the exact script;
  `gitRef` is the portable pointer; `python` and `keyLibraries` are recorded
  as evidence for toolchain reconstruction (the determinism campaign found
  the output identical across the tested numpy/torch/Python spreads, so they
  are context, not pins).

## The chain

`verify-chain` takes both statements and enforces, in order:

1. **Linkage**: the quant statement's `predicate.baseModel.digest.sha256`
   equals the conversion statement's `subject.digest.sha256`. Content links
   the chain — same bytes, whatever either side named the file. Checked on
   the parsed statements alone, before any tool is located or any subprocess
   runs.
2. **Root edge**: `verify-conversion` semantics — closure digests, no
   extra files, re-run the conversion, byte-compare. With `--check-source`,
   the snapshot is anchored to its published Hugging Face revision.
3. **Leaf edge**: `verify-attestation` semantics — re-run the quantize
   recipe, byte-compare.

Exit 0 means: these quant bytes are the output of the attested recipe on an
F16 that is itself the output of the pinned converter on the pinned snapshot
— and with `--check-source`, that snapshot is what Hugging Face publishes at
that revision. The quant statement's own `baseModel` anchor is not checked
inside a chain: the F16's provenance *is* the conversion attestation.

## Limitations (v0)

- **Reproduction needs the converter's environment**: a llama.cpp checkout
  at the attested `gitRef` and a Python that imports its requirements
  (torch, transformers, gguf). The determinism evidence says version spread
  within an era does not change the bytes; era-scale drift (converter
  feature additions) does, which is what `gitRef` pins.
- **The anchor checks the weights file, not every closure file.**
  `--check-source` v0 anchors the primary `.safetensors` against its
  published digest; config/tokenizer/README are digest-pinned in the
  statement but verified against the published repo only by that one file's
  revision. Full per-file anchoring is a small extension.
- **Historical F16s verify only with era-matched converters.** Payload bytes
  have been stable for years (a 2024 bartowski F16 is tensor-identical to a
  2026 conversion), but header KVs grow with converter versions, so byte-exact
  reproduction requires the attested converter era.
- **One weights format.** Safetensors snapshots only — the path CI-proven
  deterministic. PyTorch `.bin` snapshots are out of scope for v0.
- **Unsigned**, same as the derivation predicate; DSSE/sigstore composes on
  top.
