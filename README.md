# ggufpacker

Pack a directory of GGUF quantizations into a compact store, and reconstruct every file bit-exact on demand.

**16.0 GB -> 1.8 GB, 8.7x** on a real published repo (`bartowski/Llama-3.2-1B-Instruct-GGUF`, llama.cpp `b3821`): 19 files, one `.ggufpack` store, manifest 12.4 KB. Originals deleted, all 17 quants regenerated from the pack, 17/17 sha256 identical to the original Hugging Face files, in 283 seconds.

| File | Original | Plan | Stored | Recipe |
|------|---------:|------|-------:|--------|
| f16 | 2.3 GB | blob | 1.8 GB | source |
| imatrix | 1.3 MB | blob | 828.7 KB | |
| IQ3_M | 626.8 MB | near | 2.5 MB | |
| IQ4_XS | 708.7 MB | near | 2.1 MB | |
| Q3_K_L | 698.6 MB | near | 4.0 MB | |
| Q3_K_XL | 759.3 MB | near | 3.8 MB | Q3_K_L + override |
| Q4_0 | 737.2 MB | near | 2.5 MB | |
| Q4_0_4_4 | 735.2 MB | near | 2.5 MB | |
| Q4_0_4_8 | 735.2 MB | near | 2.5 MB | |
| Q4_0_8_8 | 735.2 MB | near | 2.5 MB | |
| Q4_K_L | 830.9 MB | near | 4.4 MB | Q4_K_M + override |
| Q4_K_M | 770.3 MB | near | 4.5 MB | |
| Q4_K_S | 739.7 MB | near | 5.1 MB | |
| Q5_K_L | 929.9 MB | near | 5.7 MB | Q5_K_M + override |
| Q5_K_M | 869.3 MB | near | 5.9 MB | |
| Q5_K_S | 851.2 MB | near | 6.6 MB | |
| Q6_K | 974.5 MB | near | 2.6 MB | |
| Q6_K_L | 1.0 GB | near | 2.5 MB | Q6_K + override |
| Q8_0 | 1.2 GB | EXACT | 1.8 MB | |

![demo](https://raw.githubusercontent.com/theadamdanielsson/ggufpacker/main/docs/demo.gif)

## Why this works

A publisher ships 15-25 quant variants per model. Every one of them is a deterministic function of a single F16 source: run `llama-quantize` with a given type (and, for k-quants, an imatrix) and you get that variant back. So there is no reason to store 16 GB of near-duplicate weights. ggufpacker stores the F16 source once, plus a tiny recipe per file and a small zstd "correction delta", and regenerates each quant on demand.

## Quickstart

```
pip install ggufpacker
```

You also need a `llama-quantize` binary (from a llama.cpp build) and the F16 source present in the directory you pack. ggufpacker invokes `llama-quantize` to prove and later reproduce each file; it does not ship one.

Pack a directory:

```
ggufpacker pack ./Llama-3.2-1B-Instruct-GGUF -o llama-1b.ggufpack --llama-quantize /path/to/llama-quantize
```

Inspect what was stored:

```
ggufpacker stats llama-1b.ggufpack
```

Regenerate a single quant (by type or by filename). Output is always sha256-verified against the original; ggufpacker refuses to emit a file on mismatch:

```
ggufpacker unpack llama-1b.ggufpack Q4_K_M -o Q4_K_M.gguf
```

Re-verify the whole store end to end:

```
ggufpacker verify llama-1b.ggufpack
```

## How it works

Each file in the directory is stored under one of three plans:

- **EXACT** — recipe only. The quant reproduces byte-for-byte from the source and recipe alone, no delta needed. In the demo, `Q8_0` packed to 1.8 MB with an EXACT plan.
- **NEAR** — recipe plus a zstd correction delta. The regenerated file is almost identical to the original; the delta patches the remaining bytes. Most k-quants land here, with deltas in the low single-digit MB.
- **blob** — the whole file, zstd-compressed. This is the automatic fallback for anything that cannot be expressed as a recipe (the F16 source itself, and files ggufpacker does not know how to regenerate). It is never lossy.

A recipe is usually just the quant type. Some variants are a base type plus a tensor-type override — for example `Q6_K_L` is recorded as `Q6_K` plus an override on the embedding/output tensors.

Every plan is executed and hash-verified **at pack time**, before it is recorded. ggufpacker actually runs the recipe, compares the result to the original by sha256, and only then writes the plan into the store. If a plan does not reproduce, ggufpacker downgrades it (NEAR, or blob) until it has something that verifies. The pack you get is one whose every file has already been proven to reconstruct bit-exact.

## Why the deltas exist at all

If quantization were perfectly reproducible, most files would pack EXACT and the deltas would be zero. They are not, and the reason is a finding worth reading on its own: GGUF quantization is **not** reproducible across machines. The same F16, the same imatrix, the same llama.cpp tag (`b3821`), and the same quant type produce different bytes on a Linux/x86_64 build versus a macOS/arm64 build — in one case 113 of 147 tensors differed, 0.196% of bytes. Locally it is perfectly deterministic (multithread equals single-thread, byte-identical); the divergence is cross-machine.

The root cause is floating-point contraction (FMA) in the k-quant scale-search loops, and a one-flag build change (`-ffp-contract=off` on the quant kernels' compilation unit) makes quantization bit-reproducible across OS, arch, and compiler. That is what the deltas are absorbing today: the gap between the machine that built the published file and the machine you are regenerating on.

- Upstream fix proposal: [ggml-org/llama.cpp#25353](https://github.com/ggml-org/llama.cpp/pull/25353).
- Cross-platform evidence (re-runnable on public CI): [gguf-quant-determinism](https://github.com/theadamdanielsson/gguf-quant-determinism).

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

- **Llama-3.2-1B** (17 files, build `b3821`): a 14.68 GB ladder; correction deltas total 31.9 MB, i.e. **+1.3% of the F16 size for the entire ladder**.
- **Qwen2.5-1.5B** (23 files, build `b3772`): 22.18 GB -> 3.156 GB, **7.03x**.

Worst delta across all 40 files: 0.94% of the file (`IQ2_M`). `Q8_0` was EXACT (zero delta) in both families — it has no scale-search loop, so it is already deterministic everywhere.

## FAQ

**Is this lossy?**
No. Every emitted file is sha256-verified against the original, and ggufpacker refuses to emit on a mismatch. Plans are also proven at pack time. If a byte is wrong, you get an error, not a file.

**Why not just zstd the directory?**
Because byte compressors get about 1.15x on quantized weights — they are already high-entropy. ggufpacker gets 8.7x because it does not compress the weights, it regenerates them from the source.

**What about LoRA adapters or finetunes?**
Different problem. Those are not deterministic re-quantizations of an F16 in the same directory, so they are out of scope for v0.

## License

MIT.
