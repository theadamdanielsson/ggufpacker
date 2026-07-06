"""ggufpacker command-line interface.

Exit codes: 0 success, 1 usage/environment error, 2 verification refusal
(a reconstruction did not hash to the manifest value and was not emitted).
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
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

    p = sub.add_parser("unpack", help="reconstruct one file from a pack")
    p.add_argument("pack", help="pack directory")
    p.add_argument("name", metavar="filename|type",
                   help="original filename, or a quant type like Q4_K_M")
    p.add_argument("-o", "--output", required=True, metavar="PATH",
                   help="where to write the reconstructed file")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: path recorded in the pack)")

    p = sub.add_parser("stats", help="show per-file plans, stored cost and ratio")
    p.add_argument("pack", help="pack directory")

    p = sub.add_parser("verify", help="re-execute every plan and check all hashes")
    p.add_argument("pack", help="pack directory")
    p.add_argument("--llama-quantize", metavar="PATH",
                   help="llama-quantize binary (default: path recorded in the pack)")

    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "pack":
        from .packer import PackError, pack

        try:
            pack(args.dir, args.output, llama_quantize=args.llama_quantize)
        except PackError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "unpack":
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
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except ReconstructError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        return 0

    if args.cmd == "stats":
        from .unpacker import stats_table

        try:
            print(stats_table(args.pack))
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "verify":
        from .unpacker import Unpacker

        try:
            with Unpacker(args.pack, llama_quantize=args.llama_quantize) as u:
                results = u.verify_all()
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        bad = 0
        for name, status in results:
            print(f"{'PASS' if status == 'OK' else 'FAIL'}  {name}"
                  + ("" if status == "OK" else f"  ({status})"))
            bad += status != "OK"
        print(f"{len(results) - bad}/{len(results)} files verified")
        return 2 if bad else 0

    raise AssertionError(f"unhandled command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
