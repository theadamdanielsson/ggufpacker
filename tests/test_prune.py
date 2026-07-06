"""`pack --prune --keep`: originals are deleted only after a completed pack
whose every plan verified; sources, imatrix files and --keep types survive;
any pre-deletion verification failure refuses and deletes nothing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ggufpacker.cli import main as cli_main
from ggufpacker.packer import PruneError, PruneRefused, pack, prune_originals
from tests.conftest import needs_quantize
from tests.util_tinymodel import write_tiny_imatrix

GARBAGE = bytes(range(256)) * 64


@pytest.fixture()
def imatrix_repo(model_dir: Path, tiny_f16: Path, tmp_path: Path, qbin: str) -> Path:
    """A repo with source F16 + imatrix + two quants + one blob-fallback file.
    All REAL COPIES (never hardlinks): deletion must be observable."""
    imx = write_tiny_imatrix(model_dir / "tiny.imatrix", seed=11)
    for qtype in ("Q8_0", "Q4_K_M"):
        r = subprocess.run(
            [qbin, "--imatrix", str(imx), str(tiny_f16),
             str(model_dir / f"tiny-{qtype}.gguf"), qtype],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stdout + r.stderr
    (model_dir / "tiny-MYSTERY.gguf").write_bytes(GARBAGE)  # blob-plan quant
    (model_dir / "README.md").write_text("not packed\n")  # skipped, never touched
    return model_dir


# ------------------------------------------------------------ happy path (CLI)

@needs_quantize
def test_prune_keep_deletes_all_but_keep_sources_imatrix(
    imatrix_repo: Path, tmp_path: Path, qbin: str, capsys
):
    capsys.readouterr()
    rc = cli_main([
        "pack", str(imatrix_repo), "-o", str(tmp_path / "p.ggufpack"),
        "--llama-quantize", qbin, "--prune", "--keep", "Q8_0",
    ])
    assert rc == 0

    assert not (imatrix_repo / "tiny-Q4_K_M.gguf").exists(), "unkept quant must be deleted"
    assert not (imatrix_repo / "tiny-MYSTERY.gguf").exists(), \
        "blob-plan files are stored losslessly; deleting them is safe"
    assert (imatrix_repo / "tiny-Q8_0.gguf").exists(), "--keep type must survive"
    assert (imatrix_repo / "tiny-f16.gguf").exists(), "source is never deleted"
    assert (imatrix_repo / "tiny.imatrix").exists(), "imatrix is never deleted"
    assert (imatrix_repo / "README.md").exists(), "skipped files are never touched"

    out = capsys.readouterr().out
    assert "deleted tiny-Q4_K_M.gguf" in out
    assert "deleted tiny-MYSTERY.gguf" in out
    assert "freed" in out
    assert "tiny-Q8_0.gguf [--keep]" in out
    assert "tiny-f16.gguf [source]" in out
    assert "tiny.imatrix [imatrix]" in out


@needs_quantize
def test_prune_without_keep_deletes_every_quant(imatrix_repo: Path, tmp_path: Path, qbin: str):
    rc = cli_main([
        "pack", str(imatrix_repo), "-o", str(tmp_path / "p.ggufpack"),
        "--llama-quantize", qbin, "--prune",
    ])
    assert rc == 0
    for name in ("tiny-Q8_0.gguf", "tiny-Q4_K_M.gguf", "tiny-MYSTERY.gguf"):
        assert not (imatrix_repo / name).exists(), name
    assert (imatrix_repo / "tiny-f16.gguf").exists()
    assert (imatrix_repo / "tiny.imatrix").exists()


@needs_quantize
def test_prune_reports_freed_bytes_exactly(imatrix_repo: Path, tmp_path: Path, qbin: str):
    expect = sum(
        (imatrix_repo / n).stat().st_size
        for n in ("tiny-Q8_0.gguf", "tiny-Q4_K_M.gguf", "tiny-MYSTERY.gguf")
    )
    m = pack(imatrix_repo, tmp_path / "p.ggufpack", llama_quantize=qbin)
    res = prune_originals(imatrix_repo, tmp_path / "p.ggufpack", m, [])
    assert res.freed == expect
    assert sorted(n for n, _ in res.deleted) == [
        "tiny-MYSTERY.gguf", "tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf",
    ]


# ---------------------------------------------------------------- multi-model

@needs_quantize
def test_prune_keep_type_keeps_it_for_every_model(
    multi_model_dir: Path, tmp_path: Path, qbin: str
):
    """--keep Q4_K_M in a two-model dir keeps BOTH models' Q4_K_M."""
    rc = cli_main([
        "pack", str(multi_model_dir), "-o", str(tmp_path / "p.ggufpack"),
        "--llama-quantize", qbin, "--prune", "--keep", "Q4_K_M",
    ])
    assert rc == 0
    for kept in ("tinyA-Q4_K_M.gguf", "tinyB-Q4_K_M.gguf",
                 "tinyA-f16.gguf", "tinyB-f16.gguf"):
        assert (multi_model_dir / kept).exists(), kept
    for gone in ("tinyA-Q8_0.gguf", "tinyB-Q8_0.gguf"):
        assert not (multi_model_dir / gone).exists(), gone


# ------------------------------------------------------------------- refusals

def test_keep_without_prune_is_an_error(tmp_path: Path, capsys):
    capsys.readouterr()
    rc = cli_main(["pack", str(tmp_path), "-o", str(tmp_path / "p"), "--keep", "Q8_0"])
    assert rc == 1
    assert "--keep requires --prune" in capsys.readouterr().err


@needs_quantize
def test_keep_matching_nothing_refuses_and_deletes_nothing(
    imatrix_repo: Path, tmp_path: Path, qbin: str
):
    m = pack(imatrix_repo, tmp_path / "p.ggufpack", llama_quantize=qbin)
    before = sorted(p.name for p in imatrix_repo.iterdir())
    with pytest.raises(PruneError, match="matches no file"):
        prune_originals(imatrix_repo, tmp_path / "p.ggufpack", m, ["IQ9_Z"])
    assert sorted(p.name for p in imatrix_repo.iterdir()) == before


@needs_quantize
def test_prune_refuses_when_original_changed_since_pack(
    imatrix_repo: Path, tmp_path: Path, qbin: str
):
    m = pack(imatrix_repo, tmp_path / "p.ggufpack", llama_quantize=qbin)
    with open(imatrix_repo / "tiny-Q8_0.gguf", "ab") as f:
        f.write(b"x")  # file changed after packing
    before = sorted(p.name for p in imatrix_repo.iterdir())
    with pytest.raises(PruneRefused, match="changed since it was packed"):
        prune_originals(imatrix_repo, tmp_path / "p.ggufpack", m, [])
    assert sorted(p.name for p in imatrix_repo.iterdir()) == before, \
        "a refusal must delete nothing at all"


@needs_quantize
def test_prune_refuses_on_manifest_drift(imatrix_repo: Path, tmp_path: Path, qbin: str):
    """The artificial 'unverified pack': the manifest on disk is not the one
    pack() just returned. Prune must refuse by construction."""
    pack_dir = tmp_path / "p.ggufpack"
    m = pack(imatrix_repo, pack_dir, llama_quantize=qbin)
    mpath = pack_dir / "manifest.json"
    mpath.write_text(mpath.read_text().replace("exact", "near", 1))
    with pytest.raises(PruneRefused, match="does not match"):
        prune_originals(imatrix_repo, pack_dir, m, [])
    assert (imatrix_repo / "tiny-Q8_0.gguf").exists()


def test_prune_refuses_on_corrupt_blob_and_deletes_nothing(model_dir: Path, tmp_path: Path):
    """Blob-plan deletion requires a full re-extraction proof; a corrupted
    stored blob must refuse the whole prune."""
    (model_dir / "tiny-MYSTERY.gguf").write_bytes(GARBAGE)
    pack_dir = tmp_path / "p.ggufpack"
    m = pack(model_dir, pack_dir, llama_quantize=None)

    blob_id = m.find("tiny-MYSTERY.gguf").blob
    blob_path = pack_dir / "blobs" / blob_id
    raw = bytearray(blob_path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    blob_path.write_bytes(bytes(raw))

    with pytest.raises(PruneRefused):
        prune_originals(model_dir, pack_dir, m, [])
    assert (model_dir / "tiny-MYSTERY.gguf").exists()


def test_cli_maps_prune_refusal_to_exit_2(model_dir: Path, tmp_path: Path, capsys,
                                          monkeypatch: pytest.MonkeyPatch):
    """Belt and braces: if prune ever refuses mid-CLI, the exit code is 2 and
    the refusal is printed (a real mid-command corruption cannot be staged
    from outside, so the refusal is injected)."""
    import ggufpacker.packer as packer_mod

    (model_dir / "tiny-MYSTERY.gguf").write_bytes(GARBAGE)

    def refuse(*a, **k):
        raise PruneRefused("injected refusal")

    monkeypatch.setattr(packer_mod, "prune_originals", refuse)
    capsys.readouterr()
    rc = cli_main(["pack", str(model_dir), "-o", str(tmp_path / "p.ggufpack"), "--prune"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "nothing was deleted" in err
    assert (model_dir / "tiny-MYSTERY.gguf").exists()


# --------------------------------------------------------------- source safety

@needs_quantize
def test_prune_never_deletes_a_demoted_source_copy(
    model_dir: Path, tiny_f16: Path, tmp_path: Path, qbin: str
):
    """Two copies of the F16 under one stem: one becomes THE source, the other
    is packed as a quant (recipe F16). Deleting the second copy is fine — it
    regenerates — but the chosen source must always survive."""
    shutil.copyfile(tiny_f16, model_dir / "tiny-BF16.gguf")  # misleading name, F16 bytes
    pack_dir = tmp_path / "p.ggufpack"
    m = pack(model_dir, pack_dir, llama_quantize=qbin)

    source_names = {f.filename for f in m.files if f.role == "source"}
    assert len(source_names) == 1
    prune_originals(model_dir, pack_dir, m, [])
    for name in source_names:
        assert (model_dir / name).exists(), "the pack's source must never be pruned"
