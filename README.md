# ggufpacker

Check that a GGUF quant really comes from the model it claims — and store a whole set of quants as one small pack that can rebuild every file, byte for byte.

Both work for the same reason: a quant is not an original. It's produced from a source model by a fixed procedure, so it can be rebuilt and compared instead of trusted.

## Check a quant against its source

When you download a quant today, you are trusting whoever uploaded it. There is no way to check the file was actually made from the model on the label. That gap is exploitable: researchers have shown a model can be tuned to behave normally at full precision and only turn malicious after quantization, passing the standard checks on the way ([Egashira et al., ICML 2025](https://arxiv.org/abs/2505.23786)).

ggufpacker closes that gap for new uploads. `attest` rebuilds the quant from the original model, checks that every byte matches, and only then writes a small proof file recording what was made from what. `verify-attestation` is the other side: it reads a proof file, rebuilds the quant on your machine, and refuses if anything is off. A modified or mislabeled file fails the check.

```
ggufpacker attest model-Q4_K_M.gguf --source model-f16.gguf --imatrix model.imatrix
ggufpacker verify-attestation model-Q4_K_M.gguf.derivation.json
```

The check works on anyone's machine, not just the uploader's, because quantization can be made to produce identical bytes everywhere — a one-flag llama.cpp fix proposed upstream in [#25353](https://github.com/ggml-org/llama.cpp/pull/25353). The whole loop runs on public CI in [gguf-quant-determinism](https://github.com/theadamdanielsson/gguf-quant-determinism): a proof made on a Mac is checked on a Linux machine that has only the proof file and the public source model. The quant itself — 770 MB — is never sent anywhere; 2 KB of JSON stands in for it. And a tampered proof is refused.

Two honest limits. Old quants can't be checked — they were built before the fix, and their exact bytes can't be reproduced anymore. And checking across machines needs the fixed llama.cpp build until the change lands upstream. Details, including the imatrix naming rule, are in [docs/derivation-attestation.md](docs/derivation-attestation.md).

### Check the whole path back to Hugging Face

The check above covers the last step: quant from F16. But the F16 is a derived file too — it was converted from the safetensors the model author published on Hugging Face. That conversion also produces identical bytes everywhere (no fix needed; proven on the same public CI), so it gets the same treatment. `attest-conversion` proves an F16 came from a specific published snapshot, and `verify-chain` checks both proofs together:

```
ggufpacker attest-conversion model-f16.gguf --source-dir Llama-3.2-1B-Instruct --llama-cpp-dir ~/llama.cpp
ggufpacker verify-chain model-Q4_K_M.gguf.derivation.json model-f16.gguf.conversion.json --check-source
```

`verify-chain` first requires the two proofs to be about the same F16 — matched by content, not by filename — then rebuilds both steps. With `--check-source` it also checks the snapshot against what Hugging Face publishes at the recorded revision. A pass means: these quant bytes trace back to that published model, byte for byte, with nothing swapped in anywhere along the way. Details in [docs/conversion-attestation.md](docs/conversion-attestation.md).

## Store 16 GiB of quants in 1.8 GiB

A publisher ships 15-25 quant files per model, all made from the same source. So don't store every file — store the source once, plus a short recipe per file, and rebuild on demand. Measured on a real repo (`bartowski/Llama-3.2-1B-Instruct-GGUF`, llama.cpp `b3821`): 19 files went from 16.0 GiB to 1.8 GiB (exact: 17,157,953,114 -> 1,964,806,736 bytes). The originals were deleted, and all 17 quants were rebuilt from the pack — every one matched the original Hugging Face file exactly, in 283 seconds total.

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

You can let the pack delete the originals in the same command. `--prune` only removes files the pack has just proven it can rebuild, and re-checks everything once more right before deleting. The source model and imatrix are never deleted. `--keep` holds back the quants you actually run:

```
ggufpacker pack ./Llama-3.2-1B-Instruct-GGUF -o llama-1b.ggufpack --prune --keep Q4_K_M
```

Inspect what was stored:

```
ggufpacker stats llama-1b.ggufpack
```

### Multi-model directories

A directory can hold several models at once — pack your whole models folder. Each quant is matched to its own model's source by comparing the tensors inside the files, not by trusting filenames. When two sources look identical from the inside (a base model and a finetune, say), the filename breaks the tie, and anything still ambiguous is stored whole rather than guessed: never wrong, worst case a bit bigger. `stats` shows per-model subtotals, and `--keep Q4_K_M` keeps every model's Q4_K_M.

Regenerate a single quant (by type or by filename). Output is always sha256-verified against the original; ggufpacker refuses to emit a file on mismatch:

```
ggufpacker unpack llama-1b.ggufpack Q4_K_M -o Q4_K_M.gguf
```

Or skip managing output files entirely: `get` rebuilds the quant into a local cache and prints its path — nothing else — so you can use it inline:

```
llama-server -m $(ggufpacker get llama-1b.ggufpack Q4_K_M)
```

Repeat calls come straight from the cache after a quick hash check (a second or two per GB — that check is what guarantees the path holds good bytes). `ggufpacker exec llama-1b.ggufpack Q4_K_M -- llama-cli -m {} -p "hi"` does the same and then runs the command for you, with `{}` standing in for the cached file. `ggufpacker cache ls` and `cache clear` inspect and empty the cache.

The cache can also keep itself under a size limit: `ggufpacker cache prune --max-size 20G` deletes the least-recently-used files until it fits, and setting `GGUFPACKER_CACHE_MAX=20G` does that automatically after every `get` (never deleting the file it's about to hand you).

Re-verify the whole store end to end:

```
ggufpacker verify llama-1b.ggufpack
```

### Derivation attestations

Covered up top. The proof file records what the quant was made from (source
and imatrix, each pinned by hash), the exact recipe, which build made it, and
the result hash. Two practical notes. First, tools like Cisco's Model
Provenance Kit guess a file's origin statistically and work on anything
already published — a proof file is exact, but only exists for files made
with one from now on. Second, when quantizing with an imatrix, pass just the
filename from its own directory (not a full path): llama-quantize writes the
path you typed into the output file, so a full path makes the result
machine-specific. `attest` refuses and explains when it hits this. Full spec:
[docs/derivation-attestation.md](docs/derivation-attestation.md).

## How it works

Each file in the directory is stored under one of three plans:

- **EXACT** — recipe only. The quant reproduces byte-for-byte from the source and recipe alone, no delta needed. In the demo, `Q8_0` packed to 1.8 MiB with an EXACT plan.
- **NEAR** — recipe plus a zstd correction delta. The regenerated file is almost identical to the original; the delta patches the remaining bytes. Most k-quants land here, with deltas in the low single-digit MB.
- **blob** — the whole file, zstd-compressed. This is the automatic fallback for anything that cannot be expressed as a recipe (the F16 source itself, and files ggufpacker does not know how to regenerate). It is never lossy.

A recipe is usually just the quant type. Some variants are a base type plus a tensor-type override — for example `Q6_K_L` is recorded as `Q6_K` plus an override on the embedding/output tensors.

Every plan is executed and hash-verified **at pack time**, before it is recorded. ggufpacker actually runs the recipe, compares the result to the original by sha256, and only then writes the plan into the store. If a plan does not reproduce, ggufpacker downgrades it (NEAR, or blob) until it has something that verifies. The pack you get is one whose every file has already been proven to reconstruct bit-exact.

## Why the deltas exist at all

If quantization were perfectly reproducible, most files would pack EXACT and the deltas would be zero. They are not, because GGUF quantization turns out **not** to be reproducible across machines: the same source, same imatrix, same llama.cpp version and same quant type produce different bytes on a Linux box than on a Mac (in one measurement, 113 of 147 tensors differed — about 0.2% of the bytes). On a single machine it is perfectly deterministic; the differences only appear across machines.

The cause is a compiler optimization: on some hardware the compiler fuses a multiply and an add into one instruction, which rounds slightly differently, which flips near-tie choices inside the quantizer. Turning that optimization off for one source file (`-ffp-contract=off`) makes quantization produce identical bytes everywhere. The deltas are absorbing exactly that gap today — the difference between the machine that built the published file and yours.

- Upstream fix proposal: [ggml-org/llama.cpp#25353](https://github.com/ggml-org/llama.cpp/pull/25353).
- Evidence, re-runnable on public CI: [gguf-quant-determinism](https://github.com/theadamdanielsson/gguf-quant-determinism). Without the fix, Linux and macOS builds produce different bytes from identical inputs. With it — including a CI job that applies the actual #25353 patch to current llama.cpp — Linux/gcc, macOS/clang, and Windows/MSVC all produce the same bytes for every quant type tested, with no measurable slowdown (the patched build actually timed slightly faster; treat that as noise).

The fix costs nothing in model quality, and that was measured rather than assumed: the two builds' Q4_K_M outputs score within 0.0007 perplexity of each other on wikitext-2 — about 650x smaller than the quality cost of quantization itself, and well inside measurement noise ([data](https://github.com/theadamdanielsson/gguf-quant-determinism#quality-effect-none-measurable)).

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
