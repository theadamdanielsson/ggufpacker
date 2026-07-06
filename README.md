# ggufpacker

Check that a GGUF quant is actually what it claims to be — and store whole quant ladders as recipes instead of files. Same machinery for both: a quant is a deterministic function of its F16 source, so it can be re-derived and byte-compared instead of trusted.

## Prove a quant derives from its base model

Today there is no way to check that a published quant was honestly produced from the model it names. That gap is exploitable: a quant can behave clean in full precision and only misbehave after quantization ([Egashira et al., ICML 2025](https://arxiv.org/abs/2505.23786) — 88.7% success injecting insecure code this way). Signing doesn't close it (it proves who uploaded the bytes, not where they came from), and statistical fingerprinting says "probably derived", not "this file is exactly what the recipe produces".

`attest` closes it for artifacts published from now on. It re-derives the quant from its source, byte-compares, and only on a sha256 match writes an [in-toto statement](docs/derivation-attestation.md) recording the derivation. `verify-attestation` re-runs the recipe and refuses on any mismatch — a poisoned or modified quant fails verification:

```
ggufpacker attest model-Q4_K_M.gguf --source model-f16.gguf --imatrix model.imatrix
ggufpacker verify-attestation model-Q4_K_M.gguf.derivation.json
```

This works across machines, not just on the attester's box: quantization is bit-reproducible across OS/arch/compiler with a one-flag build change (proposed upstream in [ggml-org/llama.cpp#25353](https://github.com/ggml-org/llama.cpp/pull/25353)). The whole loop runs on public CI in [gguf-quant-determinism](https://github.com/theadamdanielsson/gguf-quant-determinism): a statement attested on macOS/arm64/clang is verified bit-exact on Linux/x86_64/gcc from the statement plus the public F16 alone — the 770 MB quant file is never transferred — and a tampered statement is refused. Limits and the imatrix portability rule are in the [spec](docs/derivation-attestation.md); the main one: old quants can't be attested (they were built with FP contraction on, on unknown machines), so the verifiable corpus starts with what gets published deterministically from here.

## Pack a ladder: 16.0 GiB -> 1.8 GiB, every file back bit-exact

The same fact — quants re-derive from the F16 — means there is no reason to store 15-25 near-duplicate files. Measured on a real published repo (`bartowski/Llama-3.2-1B-Instruct-GGUF`, llama.cpp `b3821`): 19 files, one `.ggufpack` store, manifest 12.4 KiB. Exact bytes: 17,157,953,114 -> 1,964,806,736 (8.73x). Originals deleted, all 17 quants regenerated from the pack, 17/17 sha256 identical to the original Hugging Face files, in 283 seconds.

| File | Original | Plan | Stored | Recipe |
|------|---------:|------|-------:|--------|
| f16 | 2.3 GiB | blob | 1.8 GiB | source |
| imatrix | 1.3 MiB | blob | 828.7 KiB | |
| IQ3_M | 626.8 MiB | near | 2.5 MiB | |
| IQ4_XS | 708.7 MiB | near | 2.1 MiB | |
| Q3_K_L | 698.6 MiB | near | 4.0 MiB | |
| Q3_K_XL | 759.3 MiB | near | 3.8 MiB | Q3_K_L + override |
| Q4_0 | 737.2 MiB | near | 2.5 MiB | |
| Q4_0_4_4 | 735.2 MiB | near | 2.5 MiB | |
| Q4_0_4_8 | 735.2 MiB | near | 2.5 MiB | |
| Q4_0_8_8 | 735.2 MiB | near | 2.5 MiB | |
| Q4_K_L | 830.9 MiB | near | 4.4 MiB | Q4_K_M + override |
| Q4_K_M | 770.3 MiB | near | 4.5 MiB | |
| Q4_K_S | 739.7 MiB | near | 5.1 MiB | |
| Q5_K_L | 929.9 MiB | near | 5.7 MiB | Q5_K_M + override |
| Q5_K_M | 869.3 MiB | near | 5.9 MiB | |
| Q5_K_S | 851.2 MiB | near | 6.6 MiB | |
| Q6_K | 974.5 MiB | near | 2.6 MiB | |
| Q6_K_L | 1.0 GiB | near | 2.5 MiB | Q6_K + override |
| Q8_0 | 1.2 GiB | EXACT | 1.8 MiB | |

![demo](https://raw.githubusercontent.com/theadamdanielsson/ggufpacker/main/docs/demo.gif)

## Quickstart

```
pip install ggufpacker
```

(Python 3.11+. A v0 pack is a *directory* named `*.ggufpack` — manifest, source blob, per-file deltas; the single-file archive is planned for v1.)

You also need a `llama-quantize` binary (from a llama.cpp build) and the F16 source present in the directory you pack. ggufpacker invokes `llama-quantize` to prove and later reproduce each file; it does not ship one.

Pack a directory:

```
ggufpacker pack ./Llama-3.2-1B-Instruct-GGUF -o llama-1b.ggufpack --llama-quantize /path/to/llama-quantize
```

Once the pack has completed — every plan proven at pack time — you can let it delete the originals in the same command. `--prune` removes the quant files the pack can regenerate (each deletion is re-verified against the pack first; source F16/BF16 and imatrix files are never deleted), and `--keep` holds back the quants you actually run:

```
ggufpacker pack ./Llama-3.2-1B-Instruct-GGUF -o llama-1b.ggufpack --prune --keep Q4_K_M
```

Inspect what was stored:

```
ggufpacker stats llama-1b.ggufpack
```

### Multi-model directories

A directory can hold several models' ladders at once — "pack your whole models folder". Files are grouped by tensor identity (the names + shapes in the GGUF header), and each quant is matched to *its own* model's F16 source; per-model imatrix files associate by filename prefix. When two sources are indistinguishable even by tensor identity (a base model and a finetune with identical tensor maps), the filename prefix breaks the tie, and anything still ambiguous is stored as a whole-file blob rather than guessed — never wrong, worst case stored bigger. `stats` shows per-model subtotals for such packs, and `--keep Q4_K_M` keeps every model's Q4_K_M.

Regenerate a single quant (by type or by filename). Output is always sha256-verified against the original; ggufpacker refuses to emit a file on mismatch:

```
ggufpacker unpack llama-1b.ggufpack Q4_K_M -o Q4_K_M.gguf
```

Or skip managing output files: `get` materializes the quant into a local cache (`~/.cache/ggufpacker`, override with `$GGUFPACKER_CACHE`) and prints the verified absolute path — and nothing else — on stdout, so it composes:

```
llama-server -m $(ggufpacker get llama-1b.ggufpack Q4_K_M)
```

Repeat calls serve straight from the cache after an integrity rehash (~1–2 s per GB; that rehash is the guarantee the path holds verified bytes). `ggufpacker exec llama-1b.ggufpack Q4_K_M -- llama-cli -m {} -p "hi"` does the same and runs the command, substituting `{}` with the cached path (appended as the last argument if no `{}` is present) and propagating its exit code. Inspect or reclaim space with `ggufpacker cache ls` / `ggufpacker cache clear [--pack PACK]`.

The cache can also keep itself under a size cap: `ggufpacker cache prune --max-size 20G` evicts least-recently-used files (by mtime, which every hit touches) until the total fits, and setting `GGUFPACKER_CACHE_MAX=20G` applies the same eviction automatically at the end of every `get` — after materializing, and never evicting the file `get` is about to return.

Re-verify the whole store end to end:

```
ggufpacker verify llama-1b.ggufpack
```

### Derivation attestations

Covered up top; the statement records source digest, imatrix digest, recipe,
build identity, and output digest. Two notes that matter in practice:
statistical fingerprinting (e.g. Cisco's Model Provenance Kit) works on any
existing file but says "probably derived" — an attestation is byte-exact but
only exists for artifacts quantized deterministically from now on. And the
imatrix must be passed as a bare cwd-relative filename when quantizing, or the
artifact isn't portable (llama-quantize embeds the path string in the header;
`attest` refuses and explains when it hits this). Full spec:
[docs/derivation-attestation.md](docs/derivation-attestation.md).

## How it works

Each file in the directory is stored under one of three plans:

- **EXACT** — recipe only. The quant reproduces byte-for-byte from the source and recipe alone, no delta needed. In the demo, `Q8_0` packed to 1.8 MiB with an EXACT plan.
- **NEAR** — recipe plus a zstd correction delta. The regenerated file is almost identical to the original; the delta patches the remaining bytes. Most k-quants land here, with deltas in the low single-digit MB.
- **blob** — the whole file, zstd-compressed. This is the automatic fallback for anything that cannot be expressed as a recipe (the F16 source itself, and files ggufpacker does not know how to regenerate). It is never lossy.

A recipe is usually just the quant type. Some variants are a base type plus a tensor-type override — for example `Q6_K_L` is recorded as `Q6_K` plus an override on the embedding/output tensors.

Every plan is executed and hash-verified **at pack time**, before it is recorded. ggufpacker actually runs the recipe, compares the result to the original by sha256, and only then writes the plan into the store. If a plan does not reproduce, ggufpacker downgrades it (NEAR, or blob) until it has something that verifies. The pack you get is one whose every file has already been proven to reconstruct bit-exact.

## Why the deltas exist at all

If quantization were perfectly reproducible, most files would pack EXACT and the deltas would be zero. They are not, and the reason is a finding worth reading on its own: GGUF quantization is **not** reproducible across machines. The same F16, the same imatrix, the same llama.cpp tag (`b3821`), and the same quant type produce different bytes on a Linux/x86_64 build versus a macOS/arm64 build — in one case 113 of 147 tensors differed, 0.196% of bytes. Locally it is perfectly deterministic (multithread equals single-thread, byte-identical); the divergence is cross-machine.

The root cause is floating-point contraction (FMA) in the k-quant scale-search loops, and a one-flag build change (`-ffp-contract=off` on the quant kernels' compilation unit) makes quantization bit-reproducible across OS, arch, and compiler. That is what the deltas are absorbing today: the gap between the machine that built the published file and the machine you are regenerating on.

- Upstream fix proposal: [ggml-org/llama.cpp#25353](https://github.com/ggml-org/llama.cpp/pull/25353).
- Cross-platform evidence, re-runnable on public CI: [gguf-quant-determinism](https://github.com/theadamdanielsson/gguf-quant-determinism). The CI matrix tests the fix two ways: strict builds of tag `b3821` (all five quant legs — including IQ4_XS and an imatrix leg — bit-identical across x86_64/gcc and arm64/clang), and **the #25353 patch itself applied to pinned llama.cpp master with default flags otherwise** — also bit-identical cross-arch, with no measurable slowdown (single-threaded quantize timings came out slightly *faster* on the patched build on both runners; treat as noise). A default MSVC build at the same commit emits exactly the patched hashes (current MSVC does not contract at its default `/fp:precise`), so in that CI matrix gcc/Linux, clang/macOS, and MSVC/Windows all produce one hash set per quant.

Switching to the deterministic build is also quality-free, measured: Q4_K_M from the default and `-ffp-contract=off` builds score within 0.0007 PPL on wikitext-2 — ~650x below quantization's own ~0.46 PPL cost and below the error estimate ([data](https://github.com/theadamdanielsson/gguf-quant-determinism#quality-effect-none-measurable)).

Once an upstream deterministic build mode lands, the NEAR deltas for future quants can go to zero and packs become portable across machines.

## Limitations

Stated up front, because they change what ggufpacker is good for today:

- **Packs are machine- and build-scoped in v0.** A pack is proven against the `llama-quantize` build that made it. Portable packs depend on the upstream deterministic mode above; until then, treat a pack as reproducible on comparable builds, not universally.
- **The F16 source must be in the directory.** Recipes regenerate quants *from* the source. No F16, no packing — there is nothing to regenerate from.
- **Regeneration is not free.** Reconstructing a file means running `llama-quantize`, roughly ~17s per file on an M-series laptop (the demo regenerated 17 quants in 283 seconds). This is a storage/bandwidth tool, not a hot path.
- **Some files fall back to blobs.** Multi-shard GGUFs and GGUF-format imatrix files are stored as zstd blobs rather than recipes. Still lossless, just not 8.7x.
- **One validation scale so far.** The bit-exact study covers 1B/1.5B-scale models (40/40 files across two families). Larger models are expected to behave identically but have not yet been measured. If you run a bigger one, the numbers are the interesting part — please report them.

## Validation

40/40 files across two model families reconstruct bit-exact:

- **Llama-3.2-1B** (17 quants, build `b3821`): 14.68 GB (SI) of quants. The correction deltas total 32.1 MB — **+1.3% of the F16 size for the entire ladder**. The pack stores 64.3 MB for those 17 files in total, because each NEAR file also carries its zstd'd original header (~1.9 MB each, dominated by the embedded tokenizer) for metadata-exact reconstruction — 2.6% of the F16 all-in.
- **Qwen2.5-1.5B** (23 files, build `b3772`): 22.18 GB -> 3.156 GB, **7.03x**; deltas total 59.8 MB.

Worst delta across all 40 files: 1.2% of the file (`IQ2_M`). `Q8_0` was EXACT (zero delta) in both families — it has no scale-search loop, so it is already deterministic everywhere.

## FAQ

**Is this lossy?**
No. Every emitted file is sha256-verified against the original, and ggufpacker refuses to emit on a mismatch. Plans are also proven at pack time. If a byte is wrong, you get an error, not a file.

**Why not just zstd the directory?**
Because byte compressors get about 1.15x on quantized weights — they are already high-entropy. ggufpacker gets 8.7x because it does not compress the weights, it regenerates them from the source.

**Isn't this what Hugging Face Xet already does?**
Xet dedups at the byte-chunk level and [reports ~2x on quant ladders](https://huggingface.co/blog/from-chunks-to-blocks) (bartowski's gemma-2-9b-it: 29 quants, 191 GB stored as ~97 GB — there is even a [demo space](https://huggingface.co/spaces/xet-team/quantization-dedup) visualizing it). That 2x is a good measure of how much raw near-duplication a ladder has when you can only match bytes. Regenerating from the source instead of matching bytes is what gets from 2x to ~9x — and it also works on your own disk, not just on the Hub's storage backend.

**Can I run inference directly from a pack?**
Not directly — a pack is cold storage. But you no longer have to manage the unpacked files yourself: `ggufpacker get pack Q4_K_M` materializes the quant into a local cache (sha256-verified, reconstructed once) and prints its path, so `llama-server -m $(ggufpacker get pack Q4_K_M)` just works, and `ggufpacker exec pack Q4_K_M -- llama-cli -m {} -p "hi"` runs the command for you. The practical pattern still stands: keep your daily-driver quant unpacked (or cached), pack the ladder you rarely touch.

**What about LoRA adapters or finetunes?**
Different problem. Those are not deterministic re-quantizations of an F16 in the same directory, so they are out of scope for v0.

## License

MIT.
