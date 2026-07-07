# GGUF derivation attestation (v0)

A small, checkable claim: **this quant file derives bit-exactly from that
base model via this recipe.** Not "probably derives" (statistical
fingerprinting answers that), and not "somebody trustworthy published it"
(signing answers that): anyone with the pinned llama-quantize build can
re-run the recipe and byte-compare.

```
ggufpacker attest  model-Q4_K_M.gguf --source model-f16.gguf --imatrix model.imatrix
ggufpacker verify-attestation model-Q4_K_M.gguf.derivation.json
```

`attest` refuses (exit 2) unless it has just re-derived the file and matched
its sha256; `verify-attestation` refuses unless re-derivation reproduces the
attested digest. Neither ever asserts anything it did not just prove.

## Why this is possible at all

Quantization is a deterministic function of `(source, recipe, build)`.
On one machine it always was; across machines it becomes deterministic with
`-ffp-contract=off` on llama.cpp's quantization kernels — proposed upstream
in [ggml-org/llama.cpp#25353](https://github.com/ggml-org/llama.cpp/pull/25353)
and proven on public CI (three compilers, three OSes, two architectures, one
hash set per quant) in
[gguf-quant-determinism](https://github.com/theadamdanielsson/gguf-quant-determinism).

As far as I can tell that makes quantization the first model transform where
derivation can be checked bit-exactly in practice. Training and GPU inference
are not bit-deterministic across hardware, which is the obstacle recent
proposals for "reproducible builds for AI" run into (see
[arXiv:2606.03019](https://arxiv.org/abs/2606.03019),
[arXiv:2606.00279](https://arxiv.org/abs/2606.00279) — both argue for
bit-exact reconstructability from declared inputs; neither ships a
mechanism). Quantization is the piece that works today.

## The statement

An attestation is an unsigned [in-toto Statement v1](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md)
with predicate type:

```
https://github.com/theadamdanielsson/ggufpacker/attestation/gguf-derivation/v0
```

```json
{
 "_type": "https://in-toto.io/Statement/v1",
 "subject": [
  {"name": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
   "digest": {"sha256": "<output sha256>"}}
 ],
 "predicateType": "https://github.com/theadamdanielsson/ggufpacker/attestation/gguf-derivation/v0",
 "predicate": {
  "baseModel": {"name": "Llama-3.2-1B-Instruct-f16.gguf",
                "digest": {"sha256": "<source sha256>"}, "sizeBytes": 2479595360,
                "uri": "pkg:huggingface/bartowski/Llama-3.2-1B-Instruct-GGUF@<commit>",
                "downloadLocation": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/<commit>/Llama-3.2-1B-Instruct-f16.gguf"},
  "recipe": {
   "quantType": "Q4_K_M", "outputFormat": "gguf", "useImatrix": true,
   "command": "llama-quantize --imatrix Llama-3.2-1B-Instruct.imatrix <baseModel> <output> Q4_K_M",
   "imatrix": {"name": "Llama-3.2-1B-Instruct.imatrix",
               "digest": {"sha256": "<imatrix sha256>"}, "sizeBytes": 1314426}
  },
  "builder": {
   "tool": "ggml-org/llama.cpp/llama-quantize",
   "buildIdentity": {"binarySha256": "<sha256 of the binary>",
                     "versionBanner": "build = 3821 (abc1234)",
                     "gitRef": "b3821"}
  },
  "reproducibility": {
   "deterministic": true,
   "determinismEvidence": "https://github.com/ggml-org/llama.cpp/pull/25353",
   "reDerivedDigest": {"sha256": "<output sha256>"}
  },
  "attestedBy": "ggufpacker 0.4.0",
  "producedAt": "2026-07-06T21:00:00Z"
 }
}
```

Field semantics:

- **subject** — exactly one entry: the quant file, named by filename, keyed by
  sha256. `subject.digest.sha256 == predicate.reproducibility.reDerivedDigest.sha256`
  is the machine-checkable core: the attester re-derived and matched before
  emitting. Verifiers enforce the equality; a statement where they disagree is
  refused as self-contradictory.
- **baseModel / recipe.imatrix** — the derivation inputs, digest-pinned. The
  imatrix is an *input blob*, not a derivation (its generation is
  inference-based and not bit-deterministic); it is pinned, never re-derived.
- **baseModel.uri / downloadLocation** — the canonical identity anchor
  (purl form `pkg:huggingface/<org>/<repo>@<commit>` plus a concrete resolve
  URL). This matters: without it, the statement proves the quant derives from
  *the attested bytes* — it does not prove those bytes are the model the
  filename suggests. An attacker can honestly attest a quant of their own
  poisoned F16 under a familiar name. `verify-attestation --check-source`
  closes this by requiring the attested digest to equal the published file's
  digest at the attested location (for Hugging Face, checked via the `/raw/`
  LFS pointer — no model download). Attest with `--source-uri` and
  `--source-download-url` whenever the base is published.
- **File names** — always plain sibling filenames. Verifiers reject names
  containing path separators or `..`; attested files are only ever looked up
  inside the verification directory.
- **recipe** — the llama-quantize invocation minus machine-local paths.
  `tokenEmbeddingType` / `outputTensorType` appear when the quant is a
  base-type-plus-override variant (bartowski `_L`/`_XL` style).
- **builder.buildIdentity** — `binarySha256` identifies the exact attesting
  binary; `gitRef` (attester-supplied) is the portable pointer a verifier
  builds from.
- **reproducibility.deterministic** — `true` only when the attester asserts
  the binary was built with the deterministic quantize flag. When `false`,
  verification is only expected to succeed on a binary matching
  `binarySha256`; when `true`, any deterministic build of `gitRef` on any
  machine must reproduce the bytes.

## Verification procedure

1. Parse the statement; require this predicate type, exactly one subject,
   well-formed digests, safe file names, and `reDerivedDigest == subject`.
   Anything else refuses before touching the filesystem.
2. Locate `baseModel` (and `imatrix` if `useImatrix`) by attested name inside
   the verification directory; require their sha256 digests to match. If the
   subject file itself is present, require its sha256 to match too (tampered
   artifact ≠ derivation gap). With `--check-source`, also require the
   attested `baseModel` digest to equal the published file at its attested
   `downloadLocation`.
3. Run the recipe with a llama-quantize build (ideally a deterministic build
   of `builder.buildIdentity.gitRef`), bounded by a timeout — statements are
   untrusted input.
4. Byte-compare: sha256(output) must equal `subject.digest.sha256`. On
   mismatch, the report distinguishes two findings: with the *attesting
   binary itself* (same `binarySha256`), a mismatch is tamper-evident — the
   attested file cannot be the output of the attested recipe; with a
   different binary, it is inconclusive unless the attester claimed a
   deterministic build. Either way: refusal, exit 2.

## Relation to adjacent work

- **[OpenSSF Model Signing / sigstore model-transparency](https://github.com/ossf/model-signing-spec)**
  prove *who published* bytes and that they were not tampered with. This
  predicate proves *where the bytes came from*. They compose: DSSE-wrap and
  sigstore-sign this statement and you have both. Signing is deliberately out
  of scope for ggufpacker v0.
- **[Cisco Model Provenance Kit](https://github.com/cisco-ai-defense/model-provenance-kit)**
  answers the same question statistically (weight fingerprints; needs no
  cooperation from the publisher, works on any existing model). This
  predicate answers it exactly, but only for artifacts published with an
  attestation. Fingerprinting covers the models that already exist;
  attestations only cover what gets published with one from here on.
- **Hugging Face `base_model_relation: quantized`** declares this
  relationship; nothing verifies it. An attestation is the verifiable version
  of that model-card field.

## Limitations (v0)

- **Historical quants cannot be attested or verified** — they were built with
  FP contraction on, on unknown machines. The verifiable corpus starts with
  artifacts quantized deterministically from now on.
- **Cross-machine verification needs the deterministic build** (the 16-line
  patch from #25353, buildable today; frictionless once merged upstream).
  Same-binary verification works with any build.
- **Statements are unsigned.** An attestation proves derivation, not
  authorship. For signatures, note that the `sigstore attest` CLI cannot
  carry a custom predicate (it accepts only the two SLSA predicate types);
  use the sigstore-python API (`StatementBuilder` accepts any
  `predicate_type`) or `cosign attest-blob --type <this predicate URI>` to
  produce a `.sigstore.json` bundle.
- **Identity is only as strong as the anchor.** Without `--check-source`
  (or an out-of-band check of `baseModel.digest` against the publisher),
  the statement proves derivation from the attested bytes, not from a
  canonical model.
- **One transform per statement.** The safetensors → F16 conversion has its
  own sibling predicate, [gguf-conversion/v0](conversion-attestation.md);
  `verify-chain` links the two by the F16 digest and verifies the whole
  path snapshot → F16 → quant. Sharding and merges are not attested yet.
