"""attest / verify-attestation: prove-then-emit derivation statements.

An attestation may only exist if the recipe just reproduced the quant
bit-exactly, and verification may only pass if re-derivation reproduces the
attested digest. Tampering with any input, the subject, or the statement
itself must be a clean refusal (exit 2), never a pass and never a traceback.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ggufpacker.attest import PREDICATE_TYPE, STATEMENT_TYPE
from ggufpacker.cli import main as cli_main
from tests.conftest import needs_quantize
from tests.util_tinymodel import write_tiny_imatrix


@pytest.fixture()
def attested(model_dir: Path, tiny_f16: Path, qbin: str, capsys):
    """A quantized file + its attestation, produced through the real CLI.

    Quantized under the portability convention (imatrix passed as a bare
    filename, cwd at its directory) — the header then embeds just the name,
    which is what makes the artifact attestable and relocatable."""
    imx = write_tiny_imatrix(model_dir / "tiny.imatrix", seed=11)
    quant = model_dir / "tiny-Q4_K_M.gguf"
    r = subprocess.run(
        [qbin, "--imatrix", imx.name, str(model_dir / tiny_f16.name),
         str(quant), "Q4_K_M"],
        capture_output=True, text=True, cwd=model_dir,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    rc = cli_main([
        "attest", str(quant),
        "--source", str(model_dir / tiny_f16.name),
        "--imatrix", str(imx),
        "--llama-quantize", qbin,
        "--llama-cpp-ref", "b3821",
    ])
    capsys.readouterr()
    assert rc == 0
    statement = quant.with_name(quant.name + ".derivation.json")
    assert statement.is_file()
    return model_dir, quant, statement


# ------------------------------------------------------------------ statement

@needs_quantize
def test_attest_emits_a_valid_intoto_statement(attested):
    model_dir, quant, statement = attested
    data = json.loads(statement.read_text())
    assert data["_type"] == STATEMENT_TYPE
    assert data["predicateType"] == PREDICATE_TYPE
    (subject,) = data["subject"]
    assert subject["name"] == quant.name
    assert len(subject["digest"]["sha256"]) == 64
    pred = data["predicate"]
    assert pred["recipe"]["quantType"] == "Q4_K_M"
    assert pred["recipe"]["useImatrix"] is True
    assert pred["recipe"]["imatrix"]["name"] == "tiny.imatrix"
    assert pred["builder"]["buildIdentity"]["gitRef"] == "b3821"
    assert pred["reproducibility"]["deterministic"] is False  # not asserted
    assert (pred["reproducibility"]["reDerivedDigest"]["sha256"]
            == subject["digest"]["sha256"])
    assert pred["baseModel"]["name"] == "tiny-f16.gguf"


@needs_quantize
def test_attest_refuses_a_file_no_recipe_reproduces(model_dir: Path, tiny_f16: Path,
                                                    qbin: str, capsys):
    """A tampered quant must never get an attestation."""
    quant = model_dir / "tiny-Q8_0.gguf"
    r = subprocess.run(
        [qbin, str(model_dir / tiny_f16.name), str(quant), "Q8_0"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    raw = bytearray(quant.read_bytes())
    raw[-1] ^= 0x01  # flip one payload byte
    quant.write_bytes(bytes(raw))

    capsys.readouterr()
    rc = cli_main([
        "attest", str(quant),
        "--source", str(model_dir / tiny_f16.name),
        "--llama-quantize", qbin,
    ])
    err = capsys.readouterr().err
    assert rc == 2
    assert "REFUSED" in err
    assert not quant.with_name(quant.name + ".derivation.json").exists()


@needs_quantize
def test_attest_refuses_nonportable_imatrix_path_with_hint(
    model_dir: Path, tiny_f16: Path, qbin: str, capsys
):
    """A quant whose header embeds a non-bare imatrix path cannot be attested
    portably; the refusal must name the embedded path and the fix."""
    imx = write_tiny_imatrix(model_dir / "tiny.imatrix", seed=11)
    quant = model_dir / "tiny-Q4_K_M.gguf"
    r = subprocess.run(
        [qbin, "--imatrix", str(imx), str(model_dir / tiny_f16.name),
         str(quant), "Q4_K_M"],  # absolute imatrix path -> embedded verbatim
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    capsys.readouterr()
    rc = cli_main([
        "attest", str(quant),
        "--source", str(model_dir / tiny_f16.name),
        "--imatrix", str(imx),
        "--llama-quantize", qbin,
    ])
    err = capsys.readouterr().err
    assert rc == 2
    assert "embeds imatrix path" in err
    assert not quant.with_name(quant.name + ".derivation.json").exists()


# ---------------------------------------------------------------- verification

@needs_quantize
def test_verify_attestation_roundtrip(attested, qbin: str, capsys):
    _, _, statement = attested
    capsys.readouterr()
    rc = cli_main(["verify-attestation", str(statement), "--llama-quantize", qbin])
    out = capsys.readouterr().out
    assert rc == 0
    assert "verified" in out


@needs_quantize
def test_verify_refuses_on_tampered_source(attested, qbin: str, capsys):
    model_dir, _, statement = attested
    src = model_dir / "tiny-f16.gguf"
    with open(src, "ab") as f:
        f.write(b"x")
    capsys.readouterr()
    rc = cli_main(["verify-attestation", str(statement), "--llama-quantize", qbin])
    err = capsys.readouterr().err
    assert rc == 2
    assert "base model sha256 mismatch" in err


@needs_quantize
def test_verify_refuses_on_tampered_imatrix(attested, qbin: str, capsys):
    model_dir, _, statement = attested
    with open(model_dir / "tiny.imatrix", "ab") as f:
        f.write(b"x")
    capsys.readouterr()
    rc = cli_main(["verify-attestation", str(statement), "--llama-quantize", qbin])
    err = capsys.readouterr().err
    assert rc == 2
    assert "imatrix sha256 mismatch" in err


@needs_quantize
def test_verify_refuses_on_tampered_subject_file(attested, qbin: str, capsys):
    model_dir, quant, statement = attested
    raw = bytearray(quant.read_bytes())
    raw[-1] ^= 0x01
    quant.write_bytes(bytes(raw))
    capsys.readouterr()
    rc = cli_main(["verify-attestation", str(statement), "--llama-quantize", qbin])
    err = capsys.readouterr().err
    assert rc == 2
    assert "does not match the attested sha256" in err


@needs_quantize
def test_verify_refuses_on_forged_digest(attested, qbin: str, capsys):
    """Editing the attested digest breaks BOTH the on-disk subject check and
    re-derivation — the forged claim can never verify."""
    model_dir, quant, statement = attested
    data = json.loads(statement.read_text())
    data["subject"][0]["digest"]["sha256"] = "0" * 64
    statement.write_text(json.dumps(data))
    quant.unlink()  # even without the artifact present, re-derivation refuses
    capsys.readouterr()
    rc = cli_main(["verify-attestation", str(statement), "--llama-quantize", qbin])
    err = capsys.readouterr().err
    assert rc == 2
    assert "re-derivation does not reproduce" in err


@needs_quantize
def test_verify_refuses_foreign_or_corrupt_statements(attested, qbin: str, capsys,
                                                      tmp_path: Path):
    _, _, statement = attested
    cases = {
        "not-json.json": "{ nope",
        "wrong-type.json": json.dumps({"_type": "x", "predicateType": PREDICATE_TYPE}),
        "wrong-predicate.json": json.dumps(
            {"_type": STATEMENT_TYPE, "predicateType": "https://example.com/other/v1",
             "subject": [{"name": "a", "digest": {"sha256": "0" * 64}}],
             "predicate": {}}),
    }
    for name, content in cases.items():
        p = tmp_path / name
        p.write_text(content)
        capsys.readouterr()
        rc = cli_main(["verify-attestation", str(p), "--llama-quantize", qbin])
        assert rc == 2, name
        assert "REFUSED" in capsys.readouterr().err, name


@needs_quantize
def test_verify_with_dir_relocation(attested, qbin: str, tmp_path: Path, capsys):
    """Attested files found via --dir: the statement is portable."""
    model_dir, quant, statement = attested
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for f in (quant, model_dir / "tiny-f16.gguf", model_dir / "tiny.imatrix"):
        shutil.copyfile(f, moved / f.name)
    capsys.readouterr()
    rc = cli_main(["verify-attestation", str(statement), "--dir", str(moved),
                   "--llama-quantize", qbin])
    assert rc == 0
