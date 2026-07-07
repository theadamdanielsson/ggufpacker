"""ggufpacker command-line interface.

Exit codes: 0 success, 1 usage/environment error, 2 verification refusal
(a reconstruction did not hash to the manifest value and was not emitted).
`exec` propagates the child command's exit code (128+signal if it was killed).
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # `exec` takes an arbitrary child command after `--`; split it off before
    # argparse sees it so child flags are never parsed as ours.
    exec_command: list[str] | None = None
    if argv and argv[0] == "exec" and "--" in argv:
        i = argv.index("--")
        argv, exec_command = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(
        prog="ggufpacker",
        description="Pack a directory of GGUF quantizations into a compact store; "
        "reconstruct every file bit-exact.",
    )
    ap.add_argument("--version", action="version", version=f"ggufpacker {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="pack a directory of GGUF files")
    p.add_argument("dir", help="directory containing GGUF quants (+ source F16/BF16, imatrix)")
    p.add_argument("-o", "--output", required=True, metavar="NAME.ggufpack",
                   help="output pack directory")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: $PATH lookup)")
    p.add_argument("--prune", action="store_true",
                   help="after the pack completes fully verified, delete the original "
                   "quant files (source F16/BF16 and imatrix files are never deleted; "
                   "every deletion is re-verified against the pack first)")
    p.add_argument("--keep", metavar="TYPE[,TYPE...]",
                   help="with --prune: quant types to keep on disk, matched like "
                   "unpack-by-type (e.g. --keep Q4_K_M,Q8_0)")

    p = sub.add_parser("unpack", help="reconstruct one file from a pack")
    p.add_argument("pack", help="pack directory")
    p.add_argument("name", metavar="filename|type",
                   help="original filename, or a quant type like Q4_K_M")
    p.add_argument("-o", "--output", required=True, metavar="PATH",
                   help="where to write the reconstructed file")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: path recorded in the pack)")

    p = sub.add_parser(
        "get",
        help="materialize one file into the local cache and print its absolute path",
        description="Resolve an entry exactly like unpack, materialize it into "
        "$GGUFPACKER_CACHE (default ~/.cache/ggufpacker), and print the verified "
        "absolute path — the only stdout output, so it composes: "
        "llama-server -m $(ggufpacker get PACK Q4_K_M). Cache hits are rehash-"
        "verified (~1-2 s/GB) before the path is printed.",
    )
    p.add_argument("pack", help="pack directory")
    p.add_argument("name", metavar="filename|type",
                   help="original filename, or a quant type like Q4_K_M")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: path recorded in the pack)")

    p = sub.add_parser(
        "exec",
        help="materialize into the cache, then run a command on the cached file",
        description="Run `get`, then execute COMMAND with every literal {} in its "
        "arguments replaced by the cached path; if no argument contains {}, the "
        "path is appended as the last argument. The child's exit code is "
        "propagated. Example: ggufpacker exec PACK Q4_K_M -- llama-cli -m {} -p hi",
        usage="ggufpacker exec [-h] [--llama-quantize PATH] pack filename|type -- COMMAND...",
    )
    p.add_argument("pack", help="pack directory")
    p.add_argument("name", metavar="filename|type",
                   help="original filename, or a quant type like Q4_K_M")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: path recorded in the pack)")

    p = sub.add_parser("cache", help="inspect, clear or size-cap the on-demand cache")
    csub = p.add_subparsers(dest="cache_cmd", required=True)
    csub.add_parser("ls", help="list cached files (pack, file, size, last used)")
    c = csub.add_parser("clear", help="remove all cached files, or one pack's")
    c.add_argument("--pack", metavar="PACK",
                   help="only remove this pack's entries (pack directory path, "
                   "recorded pack name, or cache identity prefix)")
    c = csub.add_parser(
        "prune",
        help="evict least-recently-used files until the cache fits a size cap",
        description="Evict least-recently-used cached files (by mtime, which every "
        "`get` hit touches) until the total is at most SIZE. Set $GGUFPACKER_CACHE_MAX "
        "to apply the same eviction automatically at the end of every `get` (the file "
        "`get` returns is never evicted).",
    )
    c.add_argument("--max-size", required=True, metavar="N[G|M]",
                   help="size cap, e.g. 20G, 500M, or plain bytes")

    p = sub.add_parser(
        "attest",
        help="prove a quant derives bit-exactly from a source; emit an attestation",
        description="Re-derive QUANT from SOURCE with llama-quantize, byte-compare, "
        "and only on a sha256 match write an in-toto Statement (JSON) recording the "
        "derivation: source digest, recipe, build identity, output digest. Refuses "
        "(exit 2) if no recipe reproduces the file. The statement is unsigned; wrap "
        "it in DSSE/sigstore separately if you need signatures.",
    )
    p.add_argument("quant", help="the published quant .gguf to attest")
    p.add_argument("--source", required=True, metavar="F16.gguf",
                   help="the base model the quant derives from")
    p.add_argument("--imatrix", metavar="PATH",
                   help="importance matrix the recipe uses, if any")
    p.add_argument("--qtype", metavar="TYPE",
                   help="quant type (default: inferred from filename/tensors)")
    p.add_argument("--token-embedding-type", metavar="T",
                   help="with --qtype: --token-embedding-type override")
    p.add_argument("--output-tensor-type", metavar="T",
                   help="with --qtype: --output-tensor-type override")
    p.add_argument("--llama-cpp-ref", metavar="REF",
                   help="llama.cpp tag/commit of the quantize build, recorded as-is")
    p.add_argument("--deterministic-build", action="store_true",
                   help="assert the binary was built with deterministic quantization "
                   "(-ffp-contract=off on ggml-quants.c; llama.cpp#25353), making the "
                   "attestation verifiable across machines")
    p.add_argument("--source-uri", metavar="PURL",
                   help="canonical identity of the base model, purl form, e.g. "
                   "pkg:huggingface/meta-llama/Llama-3.2-1B-Instruct@<commit>; "
                   "without it the statement proves derivation from the attested "
                   "bytes only, not from a canonical published model")
    p.add_argument("--source-download-url", metavar="URL",
                   help="where the base model file is published (a huggingface.co "
                   "resolve URL enables `verify-attestation --check-source`)")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: $PATH lookup)")
    p.add_argument("-o", "--output", metavar="PATH",
                   help="where to write the statement (default: QUANT.derivation.json)")

    p = sub.add_parser(
        "verify-attestation",
        help="re-derive an attested quant and byte-compare against the statement",
        description="Load an attestation written by `attest`, locate the base model "
        "and imatrix by their attested names (next to the attestation by default), "
        "check their digests, re-run the recipe, and compare the output sha256 to "
        "the attested subject digest. Exit 0 = proven; exit 2 = refused/mismatch.",
    )
    p.add_argument("attestation", help="the .derivation.json statement")
    p.add_argument("--dir", metavar="DIR",
                   help="directory holding the attested files (default: the "
                   "attestation's directory)")
    p.add_argument("--check-source", action="store_true",
                   help="also require the attested baseModel digest to equal the "
                   "published file at its attested downloadLocation (huggingface.co "
                   "URLs; checked via the /raw/ LFS pointer, no model download)")
    p.add_argument("--timeout", type=float, default=3600.0, metavar="SECONDS",
                   help="bound on the re-derivation quantize run (default 3600)")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: $PATH lookup)")

    p = sub.add_parser("stats", help="show per-file plans, stored cost and ratio")
    p.add_argument("pack", help="pack directory")

    p = sub.add_parser("verify", help="re-execute every plan and check all hashes")
    p.add_argument("pack", help="pack directory")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: path recorded in the pack)")

    args = ap.parse_args(argv)
    if args.cmd == "exec":
        args.exec_command = exec_command
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "pack":
        from .packer import PackError, pack

        keep = [t.strip() for t in (args.keep or "").split(",") if t.strip()]
        if keep and not args.prune:
            print("error: --keep requires --prune", file=sys.stderr)
            return 1
        try:
            manifest = pack(args.dir, args.output, llama_quantize=args.llama_quantize)
        except PackError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if args.prune:
            return _prune_after_pack(args, manifest, keep)
        return 0

    if args.cmd == "unpack":
        from .manifest import AmbiguousNameError, ManifestError
        from .unpacker import ReconstructError, Unpacker

        try:
            with Unpacker(args.pack, llama_quantize=args.llama_quantize) as u:
                entry = u.manifest.find(args.name)
                if entry is None:
                    names = ", ".join(f.filename for f in u.manifest.files)
                    print(f"error: no file matching {args.name!r} in pack "
                          f"(have: {names})", file=sys.stderr)
                    return 1
                u.reconstruct(entry, args.output)
        except (FileNotFoundError, AmbiguousNameError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except ManifestError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        except ReconstructError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        return 0

    if args.cmd == "get":
        rc, path = _cached_get(args)
        if rc == 0:
            print(path)  # the ONLY stdout output: composes with $(...)
        return rc

    if args.cmd == "exec":
        import subprocess

        if not args.exec_command:
            print("error: exec needs a command after '--', e.g. "
                  "ggufpacker exec PACK Q4_K_M -- llama-server -m {}", file=sys.stderr)
            return 1
        rc, path = _cached_get(args)
        if rc != 0:
            return rc
        cmd = [a.replace("{}", str(path)) for a in args.exec_command]
        if cmd == args.exec_command:  # no {} anywhere: append the path
            cmd.append(str(path))
        child = subprocess.run(cmd).returncode
        return 128 - child if child < 0 else child  # -SIGTERM -> 143, shell style

    if args.cmd == "cache":
        from .cache import clear, ls_table, parse_size, prune_to_size
        from .unpacker import human

        if args.cache_cmd == "ls":
            print(ls_table())
            return 0
        if args.cache_cmd == "clear":
            n = clear(args.pack)
            if args.pack and n == 0:
                print(f"error: nothing cached for {args.pack!r}", file=sys.stderr)
                return 1
            print(f"removed {n} cached pack(s)")
            return 0
        if args.cache_cmd == "prune":
            try:
                max_bytes = parse_size(args.max_size)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            stats = prune_to_size(max_bytes)
            for path, size in stats.evicted:
                print(f"evicted {path.name} ({human(size)})")
            print(f"evicted {len(stats.evicted)} file(s), freed {human(stats.freed)}; "
                  f"cache now {human(stats.remaining)} (cap {args.max_size})")
            return 0
        raise AssertionError(f"unhandled cache command {args.cache_cmd}")

    if args.cmd == "attest":
        import json as _json

        from .attest import AttestError, attest
        from .quantizer import QuantizeError

        try:
            result = attest(
                args.quant,
                source_path=args.source,
                imatrix_path=args.imatrix,
                llama_quantize=args.llama_quantize,
                qtype=args.qtype,
                token_embedding_type=args.token_embedding_type,
                output_tensor_type=args.output_tensor_type,
                llama_cpp_ref=args.llama_cpp_ref,
                deterministic_build=args.deterministic_build,
                source_uri=args.source_uri,
                source_download_url=args.source_download_url,
                log=lambda m: print(m, file=sys.stderr),
            )
        except (FileNotFoundError, QuantizeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except AttestError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        out = args.output or f"{args.quant}.derivation.json"
        with open(out, "w") as f:
            _json.dump(result.statement, f, indent=1)
            f.write("\n")
        r = result.recipe
        detail = r.qtype + (
            " +overrides" if r.token_embedding_type or r.output_tensor_type else ""
        )
        print(f"proven: {detail} reproduces "
              f"{result.statement['subject'][0]['name']} bit-exact "
              f"({result.seconds:.1f}s); attestation written to {out}",
              file=sys.stderr)
        return 0

    if args.cmd == "verify-attestation":
        from .attest import AttestationInvalid, VerifyFailed, verify
        from .quantizer import QuantizeError

        try:
            verify(
                args.attestation,
                llama_quantize=args.llama_quantize,
                search_dir=args.dir,
                timeout=args.timeout,
                check_source=args.check_source,
                log=lambda m: print(m, file=sys.stderr),
            )
        except (FileNotFoundError, QuantizeError, OSError) as e:
            # OSError covers network failures during --check-source: an
            # unreachable registry is an environment problem, not a verdict.
            print(f"error: {e}", file=sys.stderr)
            return 1
        except (AttestationInvalid, VerifyFailed) as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print("verified: re-derivation reproduces the attested sha256 bit-exact")
        return 0

    if args.cmd == "stats":
        from .manifest import ManifestError
        from .unpacker import stats_table

        try:
            print(stats_table(args.pack))
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except ManifestError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        return 0

    if args.cmd == "verify":
        from .manifest import ManifestError
        from .unpacker import Unpacker

        try:
            with Unpacker(args.pack, llama_quantize=args.llama_quantize) as u:
                results = u.verify_all()
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except ManifestError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        bad = 0
        for name, status in results:
            print(f"{'PASS' if status == 'OK' else 'FAIL'}  {name}"
                  + ("" if status == "OK" else f"  ({status})"))
            bad += status != "OK"
        print(f"{len(results) - bad}/{len(results)} files verified")
        return 2 if bad else 0

    raise AssertionError(f"unhandled command {args.cmd}")


def _prune_after_pack(args: argparse.Namespace, manifest, keep: list[str]) -> int:
    """pack --prune back half: delete verified originals, report exactly what
    was deleted and how much space was freed. Only ever called right after a
    successful pack() in this same process."""
    from .packer import PruneError, PruneRefused, prune_originals
    from .unpacker import human

    try:
        res = prune_originals(args.dir, args.output, manifest, keep)
    except PruneRefused as e:
        print(f"REFUSED: {e} (nothing was deleted)", file=sys.stderr)
        return 2
    except PruneError as e:
        print(f"error: {e} (nothing was deleted)", file=sys.stderr)
        return 1
    for name, size in res.deleted:
        print(f"deleted {name} ({human(size)})")
    kept = ", ".join(f"{name} [{why}]" for name, why in res.kept)
    print(f"pruned {len(res.deleted)} file(s), freed {human(res.freed)}"
          + (f"; kept: {kept}" if kept else ""))
    return 0


def _cached_get(args: argparse.Namespace):
    """Shared get/exec front half: (exit_code, cached_path|None). Never emits
    an unverified path; failures mirror unpack (1 usage, 2 refusal)."""
    from .cache import get
    from .manifest import ManifestError
    from .unpacker import ReconstructError

    try:
        return 0, get(args.pack, args.name, llama_quantize=args.llama_quantize)
    except (FileNotFoundError, LookupError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1, None
    except ManifestError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2, None
    except ReconstructError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2, None


if __name__ == "__main__":
    raise SystemExit(main())
