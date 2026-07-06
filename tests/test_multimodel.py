"""Multi-model directories: every quant must match ITS OWN model's source by
tensor identity (names + shapes); identical-identity twin sources resolve only
on filename-prefix affinity and NEVER by guessing; stats groups per model."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ggufpacker.manifest import AmbiguousNameError, Manifest
from ggufpacker.packer import pack
from ggufpacker.unpacker import Unpacker, stats_table
from tests.conftest import needs_quantize
from tests.util_tinymodel import write_tiny_imatrix, write_tiny_llama_f16


def _roundtrip(pack_dir: Path, name: str, original: Path, out_dir: Path, qbin: str | None):
    out = out_dir / f"out-{name}"
    with Unpacker(pack_dir, llama_quantize=qbin) as u:
        u.reconstruct(u.manifest.find(name), out)
    assert out.read_bytes() == original.read_bytes(), f"{name} not bit-exact"


# ------------------------------------------------ two architectures, one dir

@needs_quantize
def test_each_quant_matches_its_own_source(multi_model_dir: Path, tmp_path: Path, qbin: str):
    pack_dir = tmp_path / "p.ggufpack"
    pack(multi_model_dir, pack_dir, llama_quantize=qbin)

    m = Manifest.load(pack_dir)
    assert sorted(s.filename for s in m.sources) == ["tinyA-f16.gguf", "tinyB-f16.gguf"]

    for name, want_source in [
        ("tinyA-Q8_0.gguf", "tinyA-f16.gguf"),
        ("tinyA-Q4_K_M.gguf", "tinyA-f16.gguf"),
        ("tinyB-Q8_0.gguf", "tinyB-f16.gguf"),
        ("tinyB-Q4_K_M.gguf", "tinyB-f16.gguf"),
    ]:
        e = m.find(name)
        assert e is not None and e.source == want_source, (name, e and e.source)
        # same machine + same binary: a correctly matched source regenerates
        # bit-exact, so anything but EXACT would mean a wrong/loose match
        assert e.plan == "exact", (name, e.plan, e.note)


@needs_quantize
def test_multi_model_roundtrip_and_verify(multi_model_dir: Path, tmp_path: Path, qbin: str):
    pack_dir = tmp_path / "p.ggufpack"
    pack(multi_model_dir, pack_dir, llama_quantize=qbin)

    for name in ("tinyA-Q4_K_M.gguf", "tinyB-Q4_K_M.gguf"):
        _roundtrip(pack_dir, name, multi_model_dir / name, tmp_path, qbin)

    with Unpacker(pack_dir, llama_quantize=qbin) as u:
        results = u.verify_all()
    assert all(status == "OK" for _, status in results), results


@needs_quantize
def test_multi_model_by_type_is_ambiguous(multi_model_dir: Path, tmp_path: Path, qbin: str):
    pack_dir = tmp_path / "p.ggufpack"
    pack(multi_model_dir, pack_dir, llama_quantize=qbin)

    m = Manifest.load(pack_dir)
    with pytest.raises(AmbiguousNameError) as exc:
        m.find("Q8_0")
    assert "tinyA-Q8_0.gguf" in str(exc.value) and "tinyB-Q8_0.gguf" in str(exc.value)


@needs_quantize
def test_multi_model_stats_groups_per_model(multi_model_dir: Path, tmp_path: Path, qbin: str):
    pack_dir = tmp_path / "p.ggufpack"
    pack(multi_model_dir, pack_dir, llama_quantize=qbin)

    table = stats_table(pack_dir)
    assert "[tinyA]" in table and "[tinyB]" in table
    assert table.count("subtotal:") == 2
    # group membership: tinyA's rows come after the [tinyA] label, before [tinyB]
    a = table.index("[tinyA]")
    b = table.index("[tinyB]")
    assert a < table.index("tinyA-Q8_0.gguf") < b
    assert b < table.index("tinyB-Q8_0.gguf")


# ------------------------------------------- ambiguous identical-identity twins

@pytest.fixture()
def twin_sources_dir(tmp_path: Path) -> Path:
    """Two sources with IDENTICAL tensor maps (same arch, different weights):
    the base-vs-abliterated case. Tensor identity cannot tell them apart."""
    d = tmp_path / "twins"
    d.mkdir()
    write_tiny_llama_f16(d / "base-f16.gguf", seed=21)
    write_tiny_llama_f16(d / "base-abliterated-f16.gguf", seed=22)
    return d


@needs_quantize
def test_twin_sources_resolved_by_filename_prefix(
    twin_sources_dir: Path, tmp_path: Path, qbin: str
):
    d = twin_sources_dir
    for src, quant in [
        ("base-f16.gguf", "base-Q8_0.gguf"),
        ("base-abliterated-f16.gguf", "base-abliterated-Q8_0.gguf"),
    ]:
        r = subprocess.run([qbin, str(d / src), str(d / quant), "Q8_0"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

    pack_dir = tmp_path / "p.ggufpack"
    pack(d, pack_dir, llama_quantize=qbin)

    m = Manifest.load(pack_dir)
    # the trap: naive longest-common-prefix would match base-Q8_0 to
    # base-abliterated-f16 ("base-" > "base"); exact stem equality must win
    e = m.find("base-Q8_0.gguf")
    assert e.source == "base-f16.gguf"
    assert e.plan == "exact", (e.plan, e.note)  # exact == provably right source
    e = m.find("base-abliterated-Q8_0.gguf")
    assert e.source == "base-abliterated-f16.gguf"
    assert e.plan == "exact", (e.plan, e.note)

    for name in ("base-Q8_0.gguf", "base-abliterated-Q8_0.gguf"):
        _roundtrip(pack_dir, name, d / name, tmp_path, qbin)


@needs_quantize
def test_twin_sources_no_affinity_blob_fallback_never_guesses(
    twin_sources_dir: Path, tmp_path: Path, qbin: str
):
    d = twin_sources_dir
    # a quant of base whose name shares NO prefix with either source
    r = subprocess.run([qbin, str(d / "base-f16.gguf"), str(d / "gamma-Q8_0.gguf"), "Q8_0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    logs: list[str] = []
    pack_dir = tmp_path / "p.ggufpack"
    pack(d, pack_dir, llama_quantize=qbin, log=logs.append)

    e = Manifest.load(pack_dir).find("gamma-Q8_0.gguf")
    assert e.plan == "blob" and e.blob is not None
    assert e.source is None, "an ambiguous quant must never be assigned a source"
    assert "ambiguous source" in e.note
    assert any("gamma-Q8_0.gguf" in line and "ambiguous source" in line for line in logs), logs

    # blob fallback is still lossless
    _roundtrip(pack_dir, "gamma-Q8_0.gguf", d / "gamma-Q8_0.gguf", tmp_path, None)


# ---------------------------------------------------------- per-model imatrix

@needs_quantize
def test_per_model_imatrix_association(
    tmp_path: Path, tiny_f16: Path, tiny_b_f16: Path, tiny_b_quants: dict, qbin: str
):
    """Model A ships an imatrix (its quant was made with it); model B ships
    none. A's quants must record A's imatrix; B's must pack without one."""
    d = tmp_path / "repo"
    d.mkdir()
    shutil.copyfile(tiny_f16, d / "tinyA-f16.gguf")
    imx = write_tiny_imatrix(d / "tinyA.imatrix", seed=11)
    r = subprocess.run(
        [qbin, "--imatrix", str(imx), str(d / "tinyA-f16.gguf"),
         str(d / "tinyA-Q4_K_M.gguf"), "Q4_K_M"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    shutil.copyfile(tiny_b_f16, d / "tinyB-f16.gguf")
    shutil.copyfile(tiny_b_quants["Q4_K_M"], d / "tinyB-Q4_K_M.gguf")

    pack_dir = tmp_path / "p.ggufpack"
    pack(d, pack_dir, llama_quantize=qbin)

    m = Manifest.load(pack_dir)
    assert [f.filename for f in m.files if f.role == "imatrix"] == ["tinyA.imatrix"]

    ea = m.find("tinyA-Q4_K_M.gguf")
    assert ea.recipe["use_imatrix"] is True
    assert ea.imatrix == "tinyA.imatrix"
    assert ea.source == "tinyA-f16.gguf"

    eb = m.find("tinyB-Q4_K_M.gguf")
    assert eb.recipe["use_imatrix"] is False
    assert eb.imatrix is None
    assert eb.source == "tinyB-f16.gguf"
    assert eb.plan == "exact", (eb.plan, eb.note)  # B must NOT inherit A's imatrix

    for name in ("tinyA-Q4_K_M.gguf", "tinyB-Q4_K_M.gguf"):
        _roundtrip(pack_dir, name, d / name, tmp_path, qbin)


# --------------------------------------------------------- manifest back-compat

@needs_quantize
def test_pre_multimodel_manifest_still_unpacks(
    model_dir: Path, tiny_quants: dict, tmp_path: Path, qbin: str
):
    """Packs written before v0.3 have no per-file source/imatrix references;
    resolution must fall back to the pack's single source entry."""
    shutil.copyfile(tiny_quants["Q8_0"], model_dir / "tiny-Q8_0.gguf")
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    mpath = pack_dir / "manifest.json"
    data = json.loads(mpath.read_text())
    for f in data["files"]:
        f.pop("source", None)
        f.pop("imatrix", None)
    mpath.write_text(json.dumps(data))

    m = Manifest.load(pack_dir)
    e = m.find("tiny-Q8_0.gguf")
    assert e.source is None and e.imatrix is None
    _roundtrip(pack_dir, "tiny-Q8_0.gguf", model_dir / "tiny-Q8_0.gguf", tmp_path, qbin)
