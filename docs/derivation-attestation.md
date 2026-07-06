# GGUF derivation attestation (v0)

A small, verifiable claim: **this quant file derives bit-exactly from that
base model via this recipe.** Not "probably derives" (statistical
fingerprinting answers that), and not "somebody trustworthy published it"
(signing answers that) — *provably derives*: anyone with the pinned
llama-quantize build can re-run the recipe and byte-compare.

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

That makes quantization the one model transform where derivation can be
*proven* today. Training and GPU inference are not bit-deterministic across
hardware, which is exactly the obstacle recent proposals for
"reproducible builds for AI" run into (see
[arXiv:2606.03019](https://arxiv.org/abs/2606.03019),
[arXiv:2606.00279](https://arxiv.org/abs/2606.00279) — both call for
bit-exact reconstructability from declared inputs; neither ships a
mechanism). Quantization is the shippable slice.

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
                "digest": {"sha256": "<source sha256>"}, "sizeBytes": 2479595360},
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
  emitting.
- **baseModel / recipe.imatrix** — the derivation inputs, digest-pinned. The
  imatrix is an *input blob*, not a derivation (its generation is
  inference-based and not bit-deterministic); it is pinned, never re-derived.
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

1. Parse the statement; require this predicate type and exactly one subject.
2. Locate `baseModel` (and `imatrix` if `useImatrix`) by attested name;
   require their sha256 digests to match. If the subject file itself is
   present, require its sha256 to match too (tampered artifact ≠ derivation gap).
3. Run the recipe with a llama-quantize build (ideally a deterministic build
   of `builder.buildIdentity.gitRef`).
4. Byte-compare: sha256(output) must equal `subject.digest.sha256`.
   Anything else is a refusal, exit 2.

## Relation to adjacent work

- **[OpenSSF Model Signing / sigstore model-transparency](https://github.com/ossf/model-signing-spec)**
  prove *who published* bytes and that they were not tampered with. This
  predicate proves *where the bytes came from*. They compose: DSSE-wrap and
  sigstore-sign this statement and you have both. Signing is deliberately out
  of scope for ggufpacker v0.
- **[Cisco Model Provenance Kit](https://github.com/cisco-ai-defense/model-provenance-kit)**
  answers the same question statistically (weight fingerprints, no
  cooperation from the publisher needed — works on any existing model).
  This predicate answers it cryptographically, but only for artifacts
  published with an attestation. Detection for the past, proof for the
  future.
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
  authorship. Wrap in DSSE + sigstore for signatures.
- **One transform.** Conversion, sharding, and merges are plausibly
  attestable the same way but are not in v0.
