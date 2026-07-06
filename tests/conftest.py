from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.util_tinymodel import write_tiny_llama_f16

# A llama-quantize binary (b3821 was used during development). Point
# GGUFPACK_TEST_LLAMA_QUANTIZE at yours; quantize-dependent tests skip when
# no binary is available.
QBIN = Path(
    os.environ.get(
        "GGUFPACK_TEST_LLAMA_QUANTIZE",
        "/private/tmp/local/-Users-adamdanielsson/work"
        "/scratchpad/ggufpack-phase0/llama.cpp/build/bin/llama-quantize",
    )
)


def _quantize_available() -> bool:
    return QBIN.is_file() and os.access(QBIN, os.X_OK)


needs_quantize = pytest.mark.skipif(
    not _quantize_available(), reason="llama-quantize binary not available"
)


@pytest.fixture(scope="session")
def qbin() -> str:
    if not _quantize_available():
        pytest.skip("llama-quantize binary not available")
    return str(QBIN)


@pytest.fixture(scope="session")
def tiny_f16(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("tinymodel")
    return write_tiny_llama_f16(d / "tiny-f16.gguf")


@pytest.fixture(scope="session")
def tiny_quants(tmp_path_factory: pytest.TempPathFactory, tiny_f16: Path, qbin: str) -> dict:
    """Quantize the tiny model once per session; {'Q8_0': path, 'Q4_K_M': path}."""
    d = tmp_path_factory.mktemp("tinyquants")
    out = {}
    for qtype in ("Q8_0", "Q4_K_M"):
        p = d / f"tiny-{qtype}.gguf"
        r = subprocess.run([qbin, str(tiny_f16), str(p), qtype], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        out[qtype] = p
    return out


@pytest.fixture()
def model_dir(tmp_path: Path, tiny_f16: Path) -> Path:
    """A fresh 'published repo' directory holding the source F16."""
    d = tmp_path / "repo"
    d.mkdir()
    shutil.copyfile(tiny_f16, d / tiny_f16.name)
    return d
