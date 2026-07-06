"""A pack whose manifest.json cannot be trusted — corrupt JSON, an unknown
format tag, or fields from a newer ggufpacker — must be a clean refusal
(exit 2, message on stderr), never a traceback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ggufpacker.cli import main as cli_main
from ggufpacker.manifest import Manifest, ManifestError
from ggufpacker.packer import pack


@pytest.fixture()
def blob_pack(model_dir: Path, tmp_path: Path) -> Path:
    """A real (source-only) pack that needs no llama-quantize binary."""
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=None)
    return pack_dir


def _read_commands(pack_dir: Path, out: Path) -> list[list[str]]:
    return [
        ["stats", str(pack_dir)],
        ["verify", str(pack_dir)],
        ["unpack", str(pack_dir), "tiny-f16.gguf", "-o", str(out)],
        ["get", str(pack_dir), "tiny-f16.gguf"],
    ]


def test_corrupt_manifest_json_refuses_with_exit_2(blob_pack: Path, tmp_path: Path, capsys):
    (blob_pack / "manifest.json").write_text("{ this is not valid json ")
    for cmd in _read_commands(blob_pack, tmp_path / "out.gguf"):
        capsys.readouterr()
        rc = cli_main(cmd)
        captured = capsys.readouterr()
        assert rc == 2, cmd
        assert "manifest" in captured.err, cmd
        assert captured.out.strip() == "", f"{cmd}: nothing may reach stdout"


def test_unknown_format_tag_refuses_with_exit_2(blob_pack: Path, tmp_path: Path, capsys):
    mp = blob_pack / "manifest.json"
    data = json.loads(mp.read_text())
    data["format"] = "ggufpack/99"
    mp.write_text(json.dumps(data))
    for cmd in _read_commands(blob_pack, tmp_path / "out.gguf"):
        capsys.readouterr()
        rc = cli_main(cmd)
        err = capsys.readouterr().err
        assert rc == 2, cmd
        assert "ggufpack/99" in err, cmd


def test_newer_manifest_fields_refuse_with_exit_2(blob_pack: Path, tmp_path: Path, capsys):
    """Forward compatibility: a pack written by a newer ggufpacker (extra file
    fields this version does not know) must refuse, not drop fields or crash."""
    mp = blob_pack / "manifest.json"
    data = json.loads(mp.read_text())
    data["files"][0]["field_from_v9"] = "x"
    mp.write_text(json.dumps(data))
    for cmd in _read_commands(blob_pack, tmp_path / "out.gguf"):
        capsys.readouterr()
        rc = cli_main(cmd)
        err = capsys.readouterr().err
        assert rc == 2, cmd
        assert "newer version" in err, cmd


def test_manifest_load_raises_manifest_error_directly(blob_pack: Path):
    (blob_pack / "manifest.json").write_text("[1, 2, 3]")
    with pytest.raises(ManifestError):
        Manifest.load(blob_pack)


def test_pack_into_existing_regular_file_is_a_clean_error(
    model_dir: Path, tmp_path: Path, capsys
):
    target = tmp_path / "afile"
    target.write_text("do not clobber me")
    capsys.readouterr()
    rc = cli_main(["pack", str(model_dir), "-o", str(target)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not a directory" in err
    assert target.read_text() == "do not clobber me"
