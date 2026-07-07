"""0.4.2 hardening: hostile statements are refused cleanly before they can
touch the filesystem, reach the quantize argv, or run unbounded; identity
anchoring (--check-source) accepts only a published-digest match."""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest

import ggufpacker.attest as attest_mod
from ggufpacker.attest import (
    PREDICATE_TYPE,
    STATEMENT_TYPE,
    AttestationInvalid,
    load_statement,
)
from ggufpacker.cli import main as cli_main
from tests.util_tinymodel import write_tiny_imatrix


def _statement(**overrides):
    """A structurally valid statement to mutate per test."""
    sha = "a" * 64
    base = {
        "_type": STATEMENT_TYPE,
        "subject": [{"name": "m-Q4_K_M.gguf", "digest": {"sha256": sha}}],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "baseModel": {"name": "m-f16.gguf", "digest": {"sha256": "b" * 64}},
            "recipe": {"quantType": "Q4_K_M", "useImatrix": False},
            "builder": {"buildIdentity": {"binarySha256": "c" * 64}},
            "reproducibility": {"deterministic": False,
                                "reDerivedDigest": {"sha256": sha}},
        },
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, data) -> Path:
    p = tmp_path / "s.derivation.json"
    p.write_text(json.dumps(data))
    return p


def _mutated(tmp_path: Path, mutate) -> Path:
    d = _statement()
    mutate(d)
    return _write(tmp_path, d)


# ------------------------------------------------------- hostile-field refusals

@pytest.mark.parametrize("name", [
    "../../../etc/hosts", "/etc/hosts", "a/b.gguf", "..", " padded.gguf",
])
def test_unsafe_names_are_refused(tmp_path: Path, name: str):
    for place in ("subject", "baseModel"):
        def mutate(d, place=place):
            if place == "subject":
                d["subject"][0]["name"] = name
            else:
                d["predicate"]["baseModel"]["name"] = name
        with pytest.raises(AttestationInvalid, match="unsafe file name"):
            load_statement(_mutated(tmp_path, mutate))


def test_unsafe_imatrix_name_is_refused(tmp_path: Path):
    def mutate(d):
        d["predicate"]["recipe"]["useImatrix"] = True
        d["predicate"]["recipe"]["imatrix"] = {
            "name": "../../x.imatrix", "digest": {"sha256": "d" * 64}}
    with pytest.raises(AttestationInvalid, match="unsafe file name"):
        load_statement(_mutated(tmp_path, mutate))


@pytest.mark.parametrize("qtype", [
    "Q4_K_M; rm -rf /", "--allow-requantize", "Q4 K M", "", None, "x" * 40,
])
def test_malformed_quant_types_are_refused(tmp_path: Path, qtype):
    with pytest.raises(AttestationInvalid, match="quantType"):
        load_statement(_mutated(
            tmp_path, lambda d: d["predicate"]["recipe"].update(quantType=qtype)))


def test_malformed_override_type_is_refused(tmp_path: Path):
    with pytest.raises(AttestationInvalid, match="tokenEmbeddingType"):
        load_statement(_mutated(
            tmp_path,
            lambda d: d["predicate"]["recipe"].update(
                tokenEmbeddingType="--exclude-weights")))


def test_missing_nested_digest_is_a_clean_refusal(tmp_path: Path, capsys):
    """Previously a KeyError traceback; must be AttestationInvalid -> exit 2."""
    p = _mutated(tmp_path, lambda d: d["predicate"]["baseModel"].pop("digest"))
    with pytest.raises(AttestationInvalid, match="sha256"):
        load_statement(p)
    capsys.readouterr()
    rc = cli_main(["verify-attestation", str(p)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "REFUSED" in err


def test_self_contradictory_rederived_digest_is_refused(tmp_path: Path):
    p = _mutated(
        tmp_path,
        lambda d: d["predicate"]["reproducibility"].update(
            reDerivedDigest={"sha256": "f" * 64}))
    with pytest.raises(AttestationInvalid, match="contradicts itself"):
        load_statement(p)


def test_statement_without_rederived_digest_still_loads(tmp_path: Path):
    """reDerivedDigest is enforced when present, not required (back-compat)."""
    p = _mutated(
        tmp_path,
        lambda d: d["predicate"]["reproducibility"].pop("reDerivedDigest"))
    load_statement(p)


# ------------------------------------------------------------------- timeout

def test_verify_timeout_is_a_clean_error(tmp_path: Path, model_dir: Path,
                                         tiny_f16: Path, capsys):
    """A hanging quantize must be killed at the timeout and reported as an
    environment error (exit 1), not hang the verifier."""
    from ggufpacker.blobs import sha256_file

    slow = tmp_path / "slow-llama-quantize"
    slow.write_text("#!/bin/sh\nsleep 30\n")
    slow.chmod(slow.stat().st_mode | stat.S_IEXEC)

    src = model_dir / tiny_f16.name
    d = _statement()
    d["predicate"]["baseModel"] = {
        "name": src.name, "digest": {"sha256": sha256_file(src)}}
    p = model_dir / "s.derivation.json"
    p.write_text(json.dumps(d))

    capsys.readouterr()
    rc = cli_main(["verify-attestation", str(p), "--llama-quantize", str(slow),
                   "--timeout", "1"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "timeout" in err


# ------------------------------------------------------------- --check-source

def test_check_source_confirms_matching_published_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sha = "b" * 64
    d = _statement()
    d["predicate"]["baseModel"]["downloadLocation"] = (
        "https://huggingface.co/org/repo/resolve/main/m-f16.gguf")
    monkeypatch.setattr(attest_mod, "_hf_published_sha256", lambda url: sha)
    attest_mod._check_source_identity(d["predicate"]["baseModel"],
                                      log=lambda m: None)


def test_check_source_refuses_on_published_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    d = _statement()
    d["predicate"]["baseModel"]["downloadLocation"] = (
        "https://huggingface.co/org/repo/resolve/main/m-f16.gguf")
    monkeypatch.setattr(attest_mod, "_hf_published_sha256", lambda url: "e" * 64)
    with pytest.raises(attest_mod.VerifyFailed, match="identity check failed"):
        attest_mod._check_source_identity(d["predicate"]["baseModel"],
                                          log=lambda m: None)


def test_check_source_requires_a_download_location(tmp_path: Path):
    d = _statement()
    with pytest.raises(AttestationInvalid, match="downloadLocation"):
        attest_mod._check_source_identity(d["predicate"]["baseModel"],
                                          log=lambda m: None)


def test_check_source_rejects_non_hf_urls(tmp_path: Path):
    d = _statement()
    d["predicate"]["baseModel"]["downloadLocation"] = "https://evil.example/x.gguf"
    with pytest.raises(AttestationInvalid, match="huggingface.co"):
        attest_mod._check_source_identity(d["predicate"]["baseModel"],
                                          log=lambda m: None)


# --------------------------------------------------- identity fields recorded

@pytest.mark.skipif(
    not __import__("tests.conftest", fromlist=["_quantize_available"])
    ._quantize_available(),
    reason="llama-quantize binary not available",
)
def test_attest_records_source_identity(model_dir: Path, tiny_f16: Path,
                                        qbin: str, capsys):
    imx = write_tiny_imatrix(model_dir / "tiny.imatrix", seed=11)
    quant = model_dir / "tiny-Q4_K_M.gguf"
    r = subprocess.run(
        [qbin, "--imatrix", imx.name, str(model_dir / tiny_f16.name),
         str(quant), "Q4_K_M"],
        capture_output=True, text=True, cwd=model_dir,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    capsys.readouterr()
    rc = cli_main([
        "attest", str(quant),
        "--source", str(model_dir / tiny_f16.name),
        "--imatrix", str(imx),
        "--llama-quantize", qbin,
        "--source-uri", "pkg:huggingface/test/tiny@deadbeef",
        "--source-download-url",
        "https://huggingface.co/test/tiny/resolve/deadbeef/tiny-f16.gguf",
    ])
    capsys.readouterr()
    assert rc == 0
    data = json.loads(
        quant.with_name(quant.name + ".derivation.json").read_text())
    bm = data["predicate"]["baseModel"]
    assert bm["uri"] == "pkg:huggingface/test/tiny@deadbeef"
    assert bm["downloadLocation"].endswith("/tiny-f16.gguf")
