"""Wrapper around llama.cpp's convert_hf_to_gguf.py (safetensors -> F16 GGUF).

The conversion step below quantization. Measured bit-reproducible at a pinned
llama.cpp commit across OS, architecture, Python and dependency versions, dense
and MoE models, and resharded inputs -- so, like quantization, it can be
attested by re-derive-or-refuse rather than signature. The input closure is
wider than the weights: the source DIRECTORY NAME feeds general.name/basename/
finetune/size_label, and README.md (if present) is read into KV metadata, so
both are part of what a verifier must reproduce.

Identity is recorded as: the converter script's sha256, the llama.cpp gitRef it
came from, the Python version, and the resolved versions of the key libraries
that touch the numeric path. sha256 of the output is the real check; the rest is
evidence for a skeptic reconstructing the toolchain.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .blobs import sha256_file

_KEY_LIBS = ("numpy", "torch", "transformers", "gguf", "safetensors", "sentencepiece")


class ConvertError(RuntimeError):
    pass


@dataclass
class ConvertResult:
    returncode: int
    seconds: float
    argv: list[str]
    output_tail: str


@dataclass
class Converter:
    script: str            # absolute path to convert_hf_to_gguf.py
    sha256: str            # sha256 of that script
    git_ref: str           # llama.cpp commit/tag the script came from ("" if unknown)
    python: str            # python interpreter used to run it
    python_version: str
    key_libraries: dict[str, str] = field(default_factory=dict)

    @classmethod
    def locate(
        cls,
        llama_cpp_dir: str | None,
        script: str | None = None,
        python: str | None = None,
        git_ref: str | None = None,
    ) -> Converter:
        """Find convert_hf_to_gguf.py inside a llama.cpp checkout (or an explicit
        path) and the Python that will run it. The converter's own deps
        (torch/transformers/...) must be importable by that Python; we record
        their versions, we do not install them."""
        path: Path | None = None
        if script:
            path = Path(script)
        elif llama_cpp_dir:
            path = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
        if path is None or not path.is_file():
            raise ConvertError(
                "convert_hf_to_gguf.py not found; pass --llama-cpp-dir DIR or "
                "--converter PATH"
            )
        py = python or sys.executable
        ref = git_ref or cls._git_ref(path.parent)
        return cls(
            script=str(path.resolve()),
            sha256=sha256_file(path),
            git_ref=ref,
            python=py,
            python_version=cls._python_version(py),
            key_libraries=cls._key_libraries(py),
        )

    @staticmethod
    def _git_ref(repo_dir: Path) -> str:
        try:
            p = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=15,
            )
            return p.stdout.strip() if p.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            return ""

    @staticmethod
    def _python_version(py: str) -> str:
        try:
            p = subprocess.run(
                [py, "-c", "import sys;print('%d.%d.%d'%sys.version_info[:3])"],
                capture_output=True, text=True, timeout=30,
            )
            return p.stdout.strip() if p.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            return ""

    @staticmethod
    def _key_libraries(py: str) -> dict[str, str]:
        code = (
            "import importlib.metadata as m\n"
            f"names={list(_KEY_LIBS)!r}\n"
            "out={}\n"
            "for n in names:\n"
            "    try: out[n]=m.version(n)\n"
            "    except Exception: pass\n"
            "import json;print(json.dumps(out))"
        )
        try:
            p = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=60)
            if p.returncode == 0:
                import json
                return json.loads(p.stdout.strip() or "{}")
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        return {}

    def run(
        self,
        source_dir: str | Path,
        out_gguf: str | Path,
        outtype: str = "f16",
        model_name: str | None = None,
        timeout: float | None = None,
    ) -> ConvertResult:
        """Invoke convert_hf_to_gguf.py on `source_dir`. The directory's own
        name is a load-bearing input (metadata heuristics parse it), so callers
        must materialize the snapshot in a directory named the repo basename;
        this wrapper passes the directory as given and does not rename it."""
        argv = [self.python, self.script, "--outtype", outtype,
                "--outfile", str(out_gguf)]
        if model_name:
            argv += ["--model-name", model_name]
        argv += [str(source_dir)]
        t0 = time.time()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise ConvertError(
                f"convert_hf_to_gguf exceeded the {timeout:.0f}s timeout"
            ) from e
        return ConvertResult(
            returncode=p.returncode,
            seconds=time.time() - t0,
            argv=argv,
            output_tail=(p.stdout + p.stderr)[-2000:],
        )
