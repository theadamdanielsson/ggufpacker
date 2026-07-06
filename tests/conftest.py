from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ggufpacker.manifest import FORMAT, FileEntry, Manifest
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


@pytest.fixture()
def variant_pack(model_dir: Path, tiny_f16: Path, tmp_path: Path, qbin: str) -> Path:
    """A real pack with bartowski-style variants sharing a recipe base type:
    tiny-Q4_K_M.gguf (recipe Q4_K_M) and tiny-Q4_K_L.gguf (recipe base Q4_K_M
    + token-embedding override). 'L' sorts before 'M', so the manifest's file
    order mirrors the pack that exposed the recipe-first resolution bug."""
    for name, extra in (
        ("tiny-Q4_K_M.gguf", []),
        ("tiny-Q4_K_L.gguf", ["--token-embedding-type", "q8_0"]),
    ):
        r = subprocess.run(
            [qbin, *extra, str(tiny_f16), str(model_dir / name), "Q4_K_M"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stdout + r.stderr

    from ggufpacker.packer import pack

    pack_dir = tmp_path / "variants.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    # The trap this fixture exists for: both entries carry recipe base Q4_K_M.
    m = Manifest.load(pack_dir)
    for name in ("tiny-Q4_K_M.gguf", "tiny-Q4_K_L.gguf"):
        e = m.find(name)
        assert e.recipe and e.recipe["qtype"] == "Q4_K_M", (name, e.recipe)
    return pack_dir


@pytest.fixture()
def ambiguous_pack(tmp_path: Path) -> Path:
    """Manifest-only pack where two variants share recipe base Q4_K_M and NO
    filename carries a -Q4_K_M suffix: a by-type query for Q4_K_M is genuinely
    ambiguous. Name resolution runs before any blob access, so no blobs are
    needed to exercise the refusal."""
    pack_dir = tmp_path / "ambiguous.ggufpack"
    pack_dir.mkdir()
    common = dict(size=1, sha256="0" * 64, role="quant", plan="exact")
    Manifest(
        format=FORMAT, created="now", tool_version="test", quantize={},
        files=[
            FileEntry(filename="X-Q4_K_L.gguf",
                      recipe={"qtype": "Q4_K_M", "token_embedding_type": "q8_0"},
                      **common),
            FileEntry(filename="X-Q4_K_XL.gguf",
                      recipe={"qtype": "Q4_K_M", "token_embedding_type": "q8_0",
                              "output_tensor_type": "q8_0"},
                      **common),
        ],
    ).save(pack_dir)
    return pack_dir
