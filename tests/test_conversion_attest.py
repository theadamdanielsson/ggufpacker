"""gguf-conversion/v0: attest-conversion / verify-conversion / verify-chain.

The converter under test is a FAKE convert_hf_to_gguf.py: a deterministic
function of exactly the real one's input closure (directory NAME, every
non-hidden file's name and bytes, optional --model-name) with no torch
dependency, so closure semantics are exercised hermetically. The chain test
quantizes for real, so its fake converter copies a genuine GGUF out of the
snapshot instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ggufpacker.attest import AttestationInvalid
from ggufpacker.blobs import sha256_file
from ggufpacker.cli import main as cli_main
from ggufpacker.conversion_attest import (
    PREDICATE_TYPE,
    STATEMENT_TYPE,
    load_conversion_statement,
)
from tests.conftest import needs_quantize
from tests.util_tinymodel import write_tiny_llama_f16

# Deterministic function of the attested closure: dir name + --model-name +
# every non-hidden file (sorted, name and bytes). Same interface as the real
# converter: python SCRIPT --outtype T --outfile OUT [--model-name N] SRC.
CLOSURE_CONVERTER = """\
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--outtype", required=True)
ap.add_argument("--outfile", required=True)
ap.add_argument("--model-name")
ap.add_argument("source")
a = ap.parse_args()
src = Path(a.source)
buf = bytearray()
buf += ("DIR:" + src.name + "\\n").encode()
if a.model_name:
    buf += ("NAME:" + a.model_name + "\\n").encode()
for p in sorted(src.rglob("*")):
    rel = p.relative_to(src)
    if p.is_file() and not any(part.startswith(".") for part in rel.parts):
        buf += rel.as_posix().encode() + b"\\x00" + p.read_bytes() + b"\\x00"
Path(a.outfile).write_bytes(bytes(buf))
"""

# For the chain test: the "conversion" must yield a real GGUF that
# llama-quantize accepts, so this one copies the snapshot's weights file.
COPY_CONVERTER = """\
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--outtype", required=True)
ap.add_argument("--outfile", required=True)
ap.add_argument("--model-name")
ap.add_argument("source")
a = ap.parse_args()
Path(a.outfile).write_bytes((Path(a.source) / "model.safetensors").read_bytes())
"""


def _run_converter(conv: Path, snap: Path, out: Path) -> None:
    r = subprocess.run(
        [sys.executable, str(conv), "--outtype", "f16",
         "--outfile", str(out), str(snap)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.fixture()
def convbits(tmp_path: Path):
    """Snapshot dir + fake converter + the F16 it produces from it."""
    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text(CLOSURE_CONVERTER)
    snap = tmp_path / "tiny-repo"
    snap.mkdir()
    (snap / "config.json").write_text('{"architectures": ["TinyForCausalLM"]}')
    (snap / "model.safetensors").write_bytes(b"\x00weights\x01" * 128)
    (snap / "README.md").write_text("# tiny model card\n")
    f16 = tmp_path / "tiny-repo-f16.gguf"
    _run_converter(conv, snap, f16)
    return tmp_path, conv, snap, f16


def _attest(conv: Path, snap: Path, f16: Path, *extra: str) -> int:
    return cli_main([
        "attest-conversion", str(f16),
        "--source-dir", str(snap),
        "--converter", str(conv),
        *extra,
    ])


def _statement_path(f16: Path) -> Path:
    return f16.with_name(f16.name + ".conversion.json")


# ------------------------------------------------------------------- attest

def test_attest_emits_a_valid_statement(convbits, capsys):
    tmp_path, conv, snap, f16 = convbits
    assert _attest(conv, snap, f16) == 0
    capsys.readouterr()
    data = load_conversion_statement(_statement_path(f16))
    assert data["_type"] == STATEMENT_TYPE
    assert data["predicateType"] == PREDICATE_TYPE
    assert data["subject"][0]["digest"]["sha256"] == sha256_file(f16)
    pred = data["predicate"]
    assert pred["source"]["directoryName"] == "tiny-repo"
    assert [f["name"] for f in pred["source"]["files"]] == [
        "README.md", "config.json", "model.safetensors",
    ]
    for f in pred["source"]["files"]:
        assert f["digest"]["sha256"] == sha256_file(snap / f["name"])
    assert pred["builder"]["buildIdentity"]["converterSha256"] == sha256_file(conv)
    assert (pred["reproducibility"]["reDerivedDigest"]["sha256"]
            == data["subject"][0]["digest"]["sha256"])


def test_attest_records_source_identity(convbits, capsys):
    tmp_path, conv, snap, f16 = convbits
    assert _attest(
        conv, snap, f16,
        "--source-uri", "pkg:huggingface/tiny/tiny-repo",
        "--source-revision", "abc123",
    ) == 0
    capsys.readouterr()
    src = load_conversion_statement(_statement_path(f16))["predicate"]["source"]
    assert src["uri"] == "pkg:huggingface/tiny/tiny-repo"
    assert src["revision"] == "abc123"


def test_attest_refuses_an_f16_the_snapshot_does_not_produce(convbits, capsys):
    tmp_path, conv, snap, f16 = convbits
    f16.write_bytes(f16.read_bytes() + b"tampered")
    assert _attest(conv, snap, f16) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not _statement_path(f16).exists()


def test_directory_name_is_a_conversion_input(convbits, capsys):
    tmp_path, conv, snap, f16 = convbits
    renamed = snap.with_name("tiny-repo-RENAMED")
    snap.rename(renamed)
    assert _attest(conv, renamed, f16) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_readme_is_part_of_the_closure(convbits, capsys):
    tmp_path, conv, snap, f16 = convbits
    (snap / "README.md").write_text("# a different model card\n")
    assert _attest(conv, snap, f16) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_model_name_is_part_of_the_recipe(convbits, capsys):
    tmp_path, conv, snap, f16 = convbits
    named = tmp_path / "named-f16.gguf"
    r = subprocess.run(
        [sys.executable, str(conv), "--outtype", "f16", "--outfile", str(named),
         "--model-name", "Tiny-1B", str(snap)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    # Without the flag the output differs; with it, it reproduces.
    assert _attest(conv, snap, named) == 2
    assert _attest(conv, snap, named, "--model-name", "Tiny-1B") == 0
    capsys.readouterr()
    cmd = load_conversion_statement(
        _statement_path(named))["predicate"]["recipe"]["command"]
    assert "--model-name Tiny-1B" in cmd


def test_hidden_files_are_outside_the_closure(convbits, capsys):
    tmp_path, conv, snap, f16 = convbits
    (snap / ".cache").mkdir()
    (snap / ".cache" / "junk").write_text("x")
    (snap / ".gitattributes").write_text("* text\n")
    assert _attest(conv, snap, f16) == 0
    capsys.readouterr()
    names = [f["name"] for f in load_conversion_statement(
        _statement_path(f16))["predicate"]["source"]["files"]]
    assert names == ["README.md", "config.json", "model.safetensors"]


# ------------------------------------------------------------------- verify

@pytest.fixture()
def attested(convbits, capsys):
    tmp_path, conv, snap, f16 = convbits
    assert _attest(conv, snap, f16) == 0
    capsys.readouterr()
    return tmp_path, conv, snap, f16, _statement_path(f16)


def _verify(conv: Path, statement: Path, *extra: str) -> int:
    return cli_main([
        "verify-conversion", str(statement), "--converter", str(conv), *extra,
    ])


def test_verify_green(attested, capsys):
    tmp_path, conv, snap, f16, statement = attested
    assert _verify(conv, statement) == 0
    assert "verified" in capsys.readouterr().out


def test_verify_refuses_a_tampered_closure_file(attested, capsys):
    tmp_path, conv, snap, f16, statement = attested
    (snap / "config.json").write_text('{"architectures": ["EvilForCausalLM"]}')
    assert _verify(conv, statement) == 2
    assert "closure sha256 mismatch" in capsys.readouterr().err


def test_verify_refuses_a_file_outside_the_closure(attested, capsys):
    tmp_path, conv, snap, f16, statement = attested
    (snap / "extra.txt").write_text("not attested")
    assert _verify(conv, statement) == 2
    assert "outside the attested closure" in capsys.readouterr().err


def test_verify_errors_on_a_missing_snapshot(attested, capsys):
    tmp_path, conv, snap, f16, statement = attested
    shutil.rmtree(snap)
    assert _verify(conv, statement) == 1
    assert "snapshot directory not found" in capsys.readouterr().err


def test_verify_refuses_a_tampered_subject_on_disk(attested, capsys):
    tmp_path, conv, snap, f16, statement = attested
    f16.write_bytes(f16.read_bytes() + b"tampered")
    assert _verify(conv, statement) == 2
    assert "does not match the attested sha256" in capsys.readouterr().err


def test_verify_is_tamper_evident_with_the_attesting_converter(attested, capsys):
    """Statement rewritten to a digest the attested conversion cannot produce,
    subject file absent: the attesting converter itself re-derives something
    else, which is tamper evidence, not an inconclusive toolchain gap."""
    tmp_path, conv, snap, f16, statement = attested
    data = json.loads(statement.read_text())
    fake = "f" * 64
    data["subject"][0]["digest"]["sha256"] = fake
    data["predicate"]["reproducibility"]["reDerivedDigest"]["sha256"] = fake
    statement.write_text(json.dumps(data))
    f16.unlink()
    assert _verify(conv, statement) == 2
    assert "TAMPER-EVIDENT" in capsys.readouterr().err


def test_verify_mismatch_with_a_different_converter_is_inconclusive(
        attested, capsys):
    tmp_path, conv, snap, f16, statement = attested
    data = json.loads(statement.read_text())
    fake = "f" * 64
    data["subject"][0]["digest"]["sha256"] = fake
    data["predicate"]["reproducibility"]["reDerivedDigest"]["sha256"] = fake
    data["predicate"]["builder"]["buildIdentity"]["converterSha256"] = "0" * 64
    statement.write_text(json.dumps(data))
    f16.unlink()
    assert _verify(conv, statement) == 2
    assert "INCONCLUSIVE" in capsys.readouterr().err


# ------------------------------------------------- hostile statement refusals

def _conv_statement(**overrides):
    sha = "a" * 64
    base = {
        "_type": STATEMENT_TYPE,
        "subject": [{"name": "m-f16.gguf", "digest": {"sha256": sha}}],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "source": {
                "directoryName": "m-repo",
                "files": [
                    {"name": "model.safetensors", "digest": {"sha256": "b" * 64}},
                ],
            },
            "recipe": {
                "transform": "convert_hf_to_gguf",
                "outputType": "f16",
                "command": ("python convert_hf_to_gguf.py --outtype f16 "
                            "--outfile <output> <sourceDir>"),
            },
            "builder": {
                "tool": "ggml-org/llama.cpp/convert_hf_to_gguf.py",
                "buildIdentity": {"converterSha256": "c" * 64},
            },
            "reproducibility": {
                "deterministic": True,
                "reDerivedDigest": {"sha256": sha},
            },
        },
    }
    base.update(overrides)
    return base


def _write_statement(tmp_path: Path, data) -> Path:
    p = tmp_path / "s.conversion.json"
    p.write_text(json.dumps(data))
    return p


def test_the_baseline_hostile_statement_is_itself_valid(tmp_path):
    load_conversion_statement(_write_statement(tmp_path, _conv_statement()))


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(predicateType="https://example.com/other/v0"),
    lambda d: d.update(subject=d["subject"] * 2),
    lambda d: d["subject"][0].update(name="../escape.gguf"),
    lambda d: d["predicate"]["source"].update(directoryName="a/b"),
    lambda d: d["predicate"]["source"].update(directoryName="../up"),
    lambda d: d["predicate"]["source"].update(files=[]),
    lambda d: d["predicate"]["source"]["files"][0].update(name="../../etc/hosts"),
    lambda d: d["predicate"]["source"]["files"][0].update(name="/etc/hosts"),
    lambda d: d["predicate"]["source"]["files"].append(
        dict(d["predicate"]["source"]["files"][0])),
    lambda d: d["predicate"]["source"]["files"][0].update(digest={"sha256": "xyz"}),
    lambda d: d["predicate"].pop("builder"),
    lambda d: d["predicate"]["builder"].pop("buildIdentity"),
    lambda d: d["predicate"]["recipe"].pop("command"),
    lambda d: d["predicate"]["recipe"].update(
        command="python x.py --model-name --evil s"),
    lambda d: d["predicate"]["reproducibility"].update(
        reDerivedDigest={"sha256": "d" * 64}),
], ids=[
    "wrong-predicate-type", "two-subjects", "traversal-subject-name",
    "dirname-with-separator", "dirname-traversal", "empty-closure",
    "closure-traversal", "closure-absolute", "closure-duplicate",
    "closure-bad-digest", "no-builder", "no-build-identity", "no-command",
    "option-shaped-model-name", "self-contradictory-rederived",
])
def test_hostile_statements_are_refused(tmp_path, mutate):
    data = _conv_statement()
    mutate(data)
    with pytest.raises(AttestationInvalid):
        load_conversion_statement(_write_statement(tmp_path, data))


def test_hostile_statement_via_cli_is_a_clean_exit_2(tmp_path, capsys):
    data = _conv_statement()
    data["predicate"]["source"]["files"][0]["name"] = "../../etc/hosts"
    p = _write_statement(tmp_path, data)
    assert cli_main(["verify-conversion", str(p)]) == 2
    assert "REFUSED" in capsys.readouterr().err


# -------------------------------------------------------------------- chain

def _quant_statement(base_model_sha: str):
    """A structurally valid gguf-derivation/v0 statement (mirrors the
    hardening tests' shape) whose baseModel digest is under test control."""
    sha = "9" * 64
    return {
        "_type": STATEMENT_TYPE,
        "subject": [{"name": "m-Q4_K_M.gguf", "digest": {"sha256": sha}}],
        "predicateType": ("https://github.com/theadamdanielsson/ggufpacker"
                          "/attestation/gguf-derivation/v0"),
        "predicate": {
            "baseModel": {"name": "m-f16.gguf",
                          "digest": {"sha256": base_model_sha}},
            "recipe": {"quantType": "Q4_K_M", "useImatrix": False},
            "builder": {"buildIdentity": {"binarySha256": "c" * 64}},
            "reproducibility": {"deterministic": False,
                                "reDerivedDigest": {"sha256": sha}},
        },
    }


def test_chain_refuses_unlinked_statements_before_any_tool_runs(
        tmp_path, capsys):
    """Digest linkage is checked on the parsed statements alone: no converter,
    no quantizer, no snapshot needed to learn the chain is broken."""
    qp = tmp_path / "q.derivation.json"
    qp.write_text(json.dumps(_quant_statement("d" * 64)))
    cp = _write_statement(tmp_path, _conv_statement())  # subject "a"*64
    assert cli_main(["verify-chain", str(qp), str(cp)]) == 2
    assert "chain broken" in capsys.readouterr().err


@needs_quantize
def test_chain_green_snapshot_to_quant(tmp_path, qbin, capsys):
    """The full path: snapshot -> F16 (fake copying converter, real GGUF
    bytes) -> quant (real llama-quantize), two linked attestations, one
    verify-chain."""
    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text(COPY_CONVERTER)
    snap = tmp_path / "tiny-repo"
    snap.mkdir()
    write_tiny_llama_f16(snap / "model.safetensors")

    f16 = tmp_path / "tiny-f16.gguf"
    f16.write_bytes((snap / "model.safetensors").read_bytes())
    assert cli_main([
        "attest-conversion", str(f16),
        "--source-dir", str(snap), "--converter", str(conv),
    ]) == 0

    quant = tmp_path / "tiny-Q4_K_M.gguf"
    r = subprocess.run([qbin, str(f16), str(quant), "Q4_K_M"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert cli_main([
        "attest", str(quant), "--source", str(f16),
        "--qtype", "Q4_K_M", "--llama-quantize", qbin,
    ]) == 0
    capsys.readouterr()

    assert cli_main([
        "verify-chain",
        str(quant.with_name(quant.name + ".derivation.json")),
        str(_statement_path(f16)),
        "--llama-quantize", qbin, "--converter", str(conv),
    ]) == 0
    out = capsys.readouterr()
    assert "chain verified" in out.out
    assert "link ok" in out.err


@needs_quantize
def test_chain_refuses_a_tampered_snapshot(tmp_path, qbin, capsys):
    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text(COPY_CONVERTER)
    snap = tmp_path / "tiny-repo"
    snap.mkdir()
    write_tiny_llama_f16(snap / "model.safetensors")
    f16 = tmp_path / "tiny-f16.gguf"
    f16.write_bytes((snap / "model.safetensors").read_bytes())
    assert cli_main(["attest-conversion", str(f16),
                     "--source-dir", str(snap), "--converter", str(conv)]) == 0
    quant = tmp_path / "tiny-Q4_K_M.gguf"
    r = subprocess.run([qbin, str(f16), str(quant), "Q4_K_M"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert cli_main(["attest", str(quant), "--source", str(f16),
                     "--qtype", "Q4_K_M", "--llama-quantize", qbin]) == 0
    capsys.readouterr()

    with open(snap / "model.safetensors", "ab") as fh:
        fh.write(b"\xff")
    assert cli_main([
        "verify-chain",
        str(quant.with_name(quant.name + ".derivation.json")),
        str(_statement_path(f16)),
        "--llama-quantize", qbin, "--converter", str(conv),
    ]) == 2
    assert "closure sha256 mismatch" in capsys.readouterr().err
