"""Wrapper around the llama-quantize binary.

Packs are machine+build-scoped in v0: llama.cpp quantization is perfectly
deterministic for one binary on one machine (established experimentally,
thread-count invariant), but different builds/toolchains may differ by up to
~0.35% of bytes via FP contraction. We therefore record the binary's identity
(path, sha256, version banner) in the manifest and warn loudly at unpack time
if the binary in use does not match.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .blobs import sha256_file


class QuantizeError(RuntimeError):
    pass


@dataclass
class QuantizeResult:
    returncode: int
    seconds: float
    argv: list[str]
    output_tail: str


@dataclass
class Quantizer:
    binary: str
    sha256: str
    version: str  # best-effort build banner, "" if not detectable

    @classmethod
    def locate(cls, explicit: str | None) -> Quantizer:
        cand = explicit or shutil.which("llama-quantize")
        if not cand or not Path(cand).is_file():
            raise QuantizeError(
                "llama-quantize binary not found; pass --llama-quantize PATH"
            )
        return cls(binary=str(Path(cand).resolve()), sha256=sha256_file(cand),
                   version=cls._probe_version(cand))

    @staticmethod
    def _probe_version(binary: str) -> str:
        # llama-quantize has no --version; its usage/startup banner sometimes
        # carries "build = NNNN (sha)". Best effort only — sha256 is the real id.
        try:
            p = subprocess.run([binary], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        m = re.search(r"build\s*=\s*\S+\s*\([0-9a-f]+\)", p.stdout + p.stderr)
        return m.group(0) if m else ""

    def run(
        self,
        src_gguf: str | Path,
        out_gguf: str | Path,
        qtype: str,
        imatrix: str | Path | None = None,
        token_embedding_type: str | None = None,
        output_tensor_type: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> QuantizeResult:
        """Invoke llama-quantize. `imatrix` is passed VERBATIM: llama-quantize
        embeds that exact string in the output header (quantize.imatrix.file),
        so the string is part of the output bytes. Callers that need portable
        output pass a bare filename plus `cwd` (the attest path does).
        `timeout` (seconds) bounds the subprocess — verification runs recipes
        from untrusted statements and must not hang forever."""
        import time

        argv: list[str] = [self.binary]
        if imatrix is not None:
            argv += ["--imatrix", str(imatrix)]
        if token_embedding_type:
            argv += ["--token-embedding-type", token_embedding_type]
        if output_tensor_type:
            argv += ["--output-tensor-type", output_tensor_type]
        argv += [str(src_gguf), str(out_gguf), qtype]
        t0 = time.time()
        try:
            p = subprocess.run(argv, capture_output=True, text=True,
                               cwd=str(cwd) if cwd else None, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise QuantizeError(
                f"llama-quantize exceeded the {timeout:.0f}s timeout"
            ) from e
        return QuantizeResult(
            returncode=p.returncode,
            seconds=time.time() - t0,
            argv=argv,
            output_tail=(p.stdout + p.stderr)[-2000:],
        )
