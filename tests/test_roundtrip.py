"""End-to-end plan round-trips: every plan class must reproduce files bit-exact,
and verification must refuse to emit anything that does not hash correctly."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ggufpack.cli import main as cli_main
from ggufpack.manifest import Manifest
from ggufpack.packer import pack
from ggufpack.unpacker import ReconstructError, Unpacker, stats_table
from tests.conftest import needs_quantize
from tests.util_tinymodel import flip_payload_bytes, mutate_kv_string, write_tiny_imatrix


def _entry(pack_dir: Path, name: str):
    e = Manifest.load(pack_dir).find(name)
    assert e is not None, name
    return e


def _unpack_and_compare(pack_dir: Path, name: str, original: Path, out: Path, qbin: str | None):
    with Unpacker(pack_dir, llama_quantize=qbin) as u:
        u.reconstruct(u.manifest.find(name), out)
    assert out.read_bytes() == original.read_bytes(), f"{name} not bit-exact"


# ---------------------------------------------------------------- EXACT

@needs_quantize
def test_exact_roundtrip(model_dir: Path, tiny_quants: dict, tmp_path: Path, qbin: str):
    for qtype, src in tiny_quants.items():
        shutil.copyfile(src, model_dir / f"tiny-{qtype}.gguf")

    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    for qtype in tiny_quants:
        e = _entry(pack_dir, f"tiny-{qtype}.gguf")
        assert e.plan == "exact"
        assert e.header_blob is None and e.delta_blob is None and e.blob is None
        assert e.recipe["qtype"] == qtype
        _unpack_and_compare(
            pack_dir, f"tiny-{qtype}.gguf", model_dir / f"tiny-{qtype}.gguf",
            tmp_path / f"out-{qtype}.gguf", qbin,
        )


@needs_quantize
def test_unpack_by_type_and_verify(model_dir: Path, tiny_quants: dict, tmp_path: Path, qbin: str):
    shutil.copyfile(tiny_quants["Q4_K_M"], model_dir / "tiny-Q4_K_M.gguf")
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    # unpack by quant type, case-insensitive
    out = tmp_path / "by-type.gguf"
    with Unpacker(pack_dir, llama_quantize=qbin) as u:
        u.reconstruct(u.manifest.find("q4_k_m"), out)
    assert out.read_bytes() == (model_dir / "tiny-Q4_K_M.gguf").read_bytes()

    with Unpacker(pack_dir, llama_quantize=qbin) as u:
        results = u.verify_all()
    assert all(status == "OK" for _, status in results), results


# ---------------------------------------------------------------- NEAR

@needs_quantize
def test_near_roundtrip_publisher_variance(
    model_dir: Path, tiny_quants: dict, tmp_path: Path, qbin: str
):
    """Simulated fp-contraction variance: a few payload bytes differ from what
    our binary regenerates -> NEAR plan (delta), restored bit-exact."""
    published = model_dir / "tiny-Q8_0.gguf"
    flip_payload_bytes(tiny_quants["Q8_0"], published, [100, 5_000, 300_000])

    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    e = _entry(pack_dir, "tiny-Q8_0.gguf")
    assert e.plan == "near"
    assert e.delta_blob is not None
    # payloads differ but headers are identical -> no header patch needed
    assert e.header_blob is None
    _unpack_and_compare(pack_dir, "tiny-Q8_0.gguf", published, tmp_path / "out.gguf", qbin)


# --------------------------------------------------- metadata-length variance

@needs_quantize
def test_metadata_differs_roundtrip(
    model_dir: Path, tiny_quants: dict, tmp_path: Path, qbin: str
):
    """A KV string value changes length (the quantize.imatrix.file local-path
    case): metadata section shifts, payloads identical -> EXACT + header patch."""
    published = model_dir / "tiny-Q8_0.gguf"
    mutate_kv_string(
        tiny_quants["Q8_0"], published,
        "general.architecture", b"llama" + b"x" * 41,
    )

    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    e = _entry(pack_dir, "tiny-Q8_0.gguf")
    assert e.plan == "exact"
    assert e.header_blob is not None  # metadata patch stored
    assert e.delta_blob is None  # payloads identical: no tensor delta
    _unpack_and_compare(pack_dir, "tiny-Q8_0.gguf", published, tmp_path / "out.gguf", qbin)


@needs_quantize
def test_metadata_and_payload_differ_roundtrip(
    model_dir: Path, tiny_quants: dict, tmp_path: Path, qbin: str
):
    """Both axes at once: shifted metadata AND payload variance -> NEAR with
    header patch; must still restore bit-exact."""
    step = tmp_path / "step.gguf"
    mutate_kv_string(tiny_quants["Q8_0"], step, "general.architecture", b"llama/longer/path")
    published = model_dir / "tiny-Q8_0.gguf"
    flip_payload_bytes(step, published, [0, 12_345])

    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    e = _entry(pack_dir, "tiny-Q8_0.gguf")
    assert e.plan == "near"
    assert e.header_blob is not None and e.delta_blob is not None
    _unpack_and_compare(pack_dir, "tiny-Q8_0.gguf", published, tmp_path / "out.gguf", qbin)


# ---------------------------------------------------------------- imatrix

@needs_quantize
def test_imatrix_roundtrip_header_patch(
    model_dir: Path, tiny_f16: Path, tmp_path: Path, qbin: str
):
    """Published imatrix quants embed the publisher's local path in the
    `quantize.imatrix.file` KV, so the metadata section never matches ours:
    the plan must be EXACT + header patch, and unpack must materialize the
    packed imatrix to regenerate."""
    pub_imx = write_tiny_imatrix(tmp_path / "publisher-dir" / "tiny.imatrix", seed=11)

    published = model_dir / "tiny-Q4_K_M.gguf"
    r = subprocess.run(
        [qbin, "--imatrix", str(pub_imx), str(tiny_f16), str(published), "Q4_K_M"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    # same imatrix content ships in the repo dir, under a different path
    shutil.copyfile(pub_imx, model_dir / "tiny.imatrix")

    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    m = Manifest.load(pack_dir)
    assert m.imatrix is not None and m.imatrix.plan == "blob"
    e = _entry(pack_dir, "tiny-Q4_K_M.gguf")
    assert e.plan == "exact"
    assert e.recipe["use_imatrix"] is True
    assert e.header_blob is not None, "path-bearing KV must force a header patch"
    assert e.delta_blob is None, "payloads must be identical (same machine+build)"
    _unpack_and_compare(pack_dir, "tiny-Q4_K_M.gguf", published, tmp_path / "out.gguf", qbin)


# ------------------------------------------------- bartowski _L/_XL overrides

@needs_quantize
def test_custom_variant_override_detection(
    model_dir: Path, tiny_f16: Path, tmp_path: Path, qbin: str
):
    """A published Q6_K_L (= Q6_K + --token-embedding-type q8_0) must be
    recognized: base type from the name map, override from the tensor-type map."""
    published = model_dir / "tiny-Q6_K_L.gguf"
    r = subprocess.run(
        [qbin, "--token-embedding-type", "q8_0", str(tiny_f16), str(published), "Q6_K"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    e = _entry(pack_dir, "tiny-Q6_K_L.gguf")
    assert e.plan == "exact"
    assert e.recipe["qtype"] == "Q6_K"
    assert e.recipe["token_embedding_type"] == "q8_0"
    _unpack_and_compare(pack_dir, "tiny-Q6_K_L.gguf", published, tmp_path / "out.gguf", qbin)


# ---------------------------------------------------------------- BLOB

def test_blob_fallback_roundtrip(model_dir: Path, tmp_path: Path):
    """Unmatchable file (not even a GGUF) -> stored whole, restored bit-exact.
    Needs no quantize binary: the fallback path must never depend on it."""
    garbage = model_dir / "tiny-MYSTERY.gguf"
    garbage.write_bytes(bytes(range(256)) * 64)

    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=None)

    e = _entry(pack_dir, "tiny-MYSTERY.gguf")
    assert e.plan == "blob" and e.blob is not None
    _unpack_and_compare(pack_dir, "tiny-MYSTERY.gguf", garbage, tmp_path / "out.gguf", None)


def test_valid_gguf_with_unmatchable_recipe_falls_back_to_blob(
    model_dir: Path, tiny_f16: Path, tmp_path: Path
):
    """A real GGUF whose recipe cannot be regenerated (here: no quantize run
    possible because we point --llama-quantize at a nonexistent path) must be
    stored as a blob rather than dropped or corrupted."""
    other = model_dir / "tiny-Q4_K_M.gguf"
    shutil.copyfile(tiny_f16, other)  # valid GGUF, misleading name

    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=str(tmp_path / "no-such-binary"))

    e = _entry(pack_dir, "tiny-Q4_K_M.gguf")
    assert e.plan == "blob" and e.blob is not None
    _unpack_and_compare(pack_dir, "tiny-Q4_K_M.gguf", other, tmp_path / "out.gguf", None)


# ------------------------------------------------------------ refusal paths

def test_corrupted_blob_refused_exit_2(model_dir: Path, tmp_path: Path, capsys):
    garbage = model_dir / "tiny-MYSTERY.gguf"
    garbage.write_bytes(bytes(range(256)) * 64)
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=None)

    blob_id = _entry(pack_dir, "tiny-MYSTERY.gguf").blob
    blob_path = pack_dir / "blobs" / blob_id
    raw = bytearray(blob_path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    blob_path.write_bytes(bytes(raw))

    out = tmp_path / "out.gguf"
    rc = cli_main(["unpack", str(pack_dir), "tiny-MYSTERY.gguf", "-o", str(out)])
    assert rc == 2
    assert not out.exists(), "refusal must not leave an output file behind"
    assert "REFUSED" in capsys.readouterr().err


def test_manifest_hash_mismatch_refused(model_dir: Path, tmp_path: Path):
    """Even with intact blobs, a final-sha mismatch must refuse to emit."""
    garbage = model_dir / "tiny-MYSTERY.gguf"
    garbage.write_bytes(bytes(range(256)) * 64)
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=None)

    mpath = pack_dir / "manifest.json"
    data = json.loads(mpath.read_text())
    for f in data["files"]:
        if f["filename"] == "tiny-MYSTERY.gguf":
            f["sha256"] = "0" * 64
    mpath.write_text(json.dumps(data))

    out = tmp_path / "out.gguf"
    with Unpacker(pack_dir) as u:
        with pytest.raises(ReconstructError):
            u.reconstruct(u.manifest.find("tiny-MYSTERY.gguf"), out)
    assert not out.exists()


# ---------------------------------------------------------------- stats

@needs_quantize
def test_stats_output(model_dir: Path, tiny_quants: dict, tmp_path: Path, qbin: str):
    shutil.copyfile(tiny_quants["Q8_0"], model_dir / "tiny-Q8_0.gguf")
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    table = stats_table(pack_dir)
    assert "tiny-Q8_0.gguf" in table
    assert "exact" in table and "blob (source)" in table
    # totals line: "N files: X -> Y, Z.Zx"
    last = table.strip().splitlines()[-1]
    assert "->" in last and "x" in last
