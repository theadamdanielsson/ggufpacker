"""Cache-on-demand (`get` / `exec` / `cache ls|clear`): the cached path must be
verified before it is printed, hits must not re-run reconstruction, corrupted
cache entries must be re-materialized, and stdout must carry the path alone."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ggufpacker.blobs import sha256_file
from ggufpacker.cache import META_NAME, get, pack_identity
from ggufpacker.cli import main as cli_main
from ggufpacker.manifest import Manifest
from ggufpacker.packer import pack
from ggufpacker.unpacker import Unpacker

GARBAGE = bytes(range(256)) * 64


@pytest.fixture(autouse=True)
def cache_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "gpcache"
    monkeypatch.setenv("GGUFPACKER_CACHE", str(d))
    return d


@pytest.fixture()
def blob_pack(model_dir: Path, tmp_path: Path) -> Path:
    """A pack needing no quantize binary: source F16 + one blob-plan file."""
    (model_dir / "tiny-MYSTERY.gguf").write_bytes(GARBAGE)
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=None)
    return pack_dir


@pytest.fixture()
def reconstruct_calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls = {"n": 0}
    orig = Unpacker.reconstruct

    def counting(self, entry, out_path):
        calls["n"] += 1
        return orig(self, entry, out_path)

    monkeypatch.setattr(Unpacker, "reconstruct", counting)
    return calls


# ---------------------------------------------------------------- get

def test_get_miss_materializes_prints_path_and_verifies(
    blob_pack: Path, cache_env: Path, capsys
):
    capsys.readouterr()  # drain pack-time logs
    rc = cli_main(["get", str(blob_pack), "tiny-MYSTERY.gguf"])
    assert rc == 0

    out = capsys.readouterr().out
    path = Path(out.strip())
    assert path.is_absolute() and path.is_file()
    assert cache_env.resolve() in path.parents
    assert path.name == "tiny-MYSTERY.gguf"
    assert path.read_bytes() == GARBAGE
    entry = Manifest.load(blob_pack).find("tiny-MYSTERY.gguf")
    assert sha256_file(path) == entry.sha256


def test_get_hit_skips_reconstruction_and_touches_mtime(
    blob_pack: Path, reconstruct_calls: dict
):
    p1 = get(blob_pack, "tiny-MYSTERY.gguf")
    assert reconstruct_calls["n"] == 1

    old = 1_000_000_000  # 2001; well in the past
    os.utime(p1, (old, old))
    p2 = get(blob_pack, "tiny-MYSTERY.gguf")
    assert p2 == p1
    assert reconstruct_calls["n"] == 1, "cache hit must not re-run reconstruction"
    assert p1.stat().st_mtime > old, "hit must touch mtime for `cache ls` recency"


def test_get_corrupted_cached_file_rematerializes(
    blob_pack: Path, reconstruct_calls: dict
):
    p = get(blob_pack, "tiny-MYSTERY.gguf")
    assert reconstruct_calls["n"] == 1

    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    p.write_bytes(bytes(raw))

    p2 = get(blob_pack, "tiny-MYSTERY.gguf")
    assert reconstruct_calls["n"] == 2, "bad cached bytes must trigger re-materialization"
    assert p2 == p and p2.read_bytes() == GARBAGE


def test_get_resolves_by_type_like_unpack(blob_pack: Path):
    # manifest.find's filename-suffix fallback: 'mystery' -> tiny-MYSTERY.gguf
    p = get(blob_pack, "mystery")
    assert p.name == "tiny-MYSTERY.gguf" and p.read_bytes() == GARBAGE


def test_get_stdout_is_exactly_the_path(blob_pack: Path, capsys):
    capsys.readouterr()
    assert cli_main(["get", str(blob_pack), "tiny-MYSTERY.gguf"]) == 0
    out1 = capsys.readouterr().out
    assert out1 == out1.strip() + "\n" and Path(out1.strip()).is_file()

    # hit path too: still nothing but the path on stdout
    assert cli_main(["get", str(blob_pack), "tiny-MYSTERY.gguf"]) == 0
    assert capsys.readouterr().out == out1


def test_get_unknown_name_exits_1(blob_pack: Path, capsys):
    capsys.readouterr()
    rc = cli_main(["get", str(blob_pack), "no-such-file"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == "", "no path may be printed on failure"
    assert "no file matching" in captured.err


def test_get_corrupted_pack_refuses_exit_2_no_path(blob_pack: Path, capsys):
    blob_id = Manifest.load(blob_pack).find("tiny-MYSTERY.gguf").blob
    blob_path = blob_pack / "blobs" / blob_id
    raw = bytearray(blob_path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    blob_path.write_bytes(bytes(raw))

    capsys.readouterr()
    rc = cli_main(["get", str(blob_pack), "tiny-MYSTERY.gguf"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == "", "an unverified path must never be emitted"
    assert "REFUSED" in captured.err


# ---------------------------------------------------------------- exec

def test_exec_substitutes_placeholder_and_succeeds(blob_pack: Path):
    rc = cli_main(["exec", str(blob_pack), "tiny-MYSTERY.gguf",
                   "--", "/bin/sh", "-c", "test -f {}"])
    assert rc == 0


def test_exec_appends_path_when_no_placeholder(blob_pack: Path):
    # no {} anywhere -> the cached path is appended as the last argument ($0 here)
    rc = cli_main(["exec", str(blob_pack), "tiny-MYSTERY.gguf",
                   "--", "/bin/sh", "-c", 'test -f "$0"'])
    assert rc == 0


def test_exec_propagates_child_exit_code(blob_pack: Path):
    rc = cli_main(["exec", str(blob_pack), "tiny-MYSTERY.gguf",
                   "--", "/bin/sh", "-c", "test -f {} && exit 7"])
    assert rc == 7


def test_exec_without_separator_or_command_exits_1(blob_pack: Path, capsys):
    capsys.readouterr()
    assert cli_main(["exec", str(blob_pack), "tiny-MYSTERY.gguf"]) == 1
    assert "after '--'" in capsys.readouterr().err
    assert cli_main(["exec", str(blob_pack), "tiny-MYSTERY.gguf", "--"]) == 1


def test_exec_get_failure_short_circuits(blob_pack: Path, tmp_path: Path, capsys):
    marker = tmp_path / "ran"
    rc = cli_main(["exec", str(blob_pack), "no-such-file",
                   "--", "/bin/sh", "-c", f"touch {marker}"])
    assert rc == 1
    assert not marker.exists(), "child must not run when get fails"


# ---------------------------------------------------------------- cache ls/clear

def test_cache_ls_lists_files_and_total(blob_pack: Path, capsys):
    get(blob_pack, "tiny-MYSTERY.gguf")
    capsys.readouterr()
    assert cli_main(["cache", "ls"]) == 0
    out = capsys.readouterr().out
    assert "tiny-MYSTERY.gguf" in out
    assert "p.ggufpack" in out  # pack label from recorded pack path
    assert "total" in out
    assert META_NAME not in out


def test_cache_ls_empty(capsys):
    capsys.readouterr()
    assert cli_main(["cache", "ls"]) == 0
    assert "cache is empty" in capsys.readouterr().out


def test_cache_clear_single_pack(
    blob_pack: Path, model_dir: Path, tmp_path: Path, cache_env: Path
):
    (model_dir / "tiny-OTHER.gguf").write_bytes(GARBAGE[::-1])
    other_pack = tmp_path / "q.ggufpack"
    pack(model_dir, other_pack, llama_quantize=None)

    p1 = get(blob_pack, "tiny-MYSTERY.gguf")
    p2 = get(other_pack, "tiny-OTHER.gguf")
    assert p1.is_file() and p2.is_file()

    assert cli_main(["cache", "clear", "--pack", str(blob_pack)]) == 0
    assert not p1.exists() and not (cache_env / pack_identity(blob_pack)).exists()
    assert p2.is_file(), "other pack's cache entries must survive"


def test_cache_clear_by_recorded_name_after_pack_deleted(
    blob_pack: Path, cache_env: Path
):
    p = get(blob_pack, "tiny-MYSTERY.gguf")
    ident = pack_identity(blob_pack)
    meta = json.loads((cache_env / ident / META_NAME).read_text())
    assert meta["manifest_sha256"] == ident

    import shutil

    shutil.rmtree(blob_pack)  # pack gone; clear by recorded name still works
    assert cli_main(["cache", "clear", "--pack", "p.ggufpack"]) == 0
    assert not p.exists()


def test_cache_clear_all_and_unknown_pack(blob_pack: Path, cache_env: Path, capsys):
    p = get(blob_pack, "tiny-MYSTERY.gguf")
    assert cli_main(["cache", "clear"]) == 0
    assert not p.exists()
    assert not any(cache_env.iterdir())

    capsys.readouterr()
    assert cli_main(["cache", "clear", "--pack", "never-cached"]) == 1
    assert "nothing cached" in capsys.readouterr().err


# ---------------------------------------------------------------- with quants

def test_get_real_quant_by_type(model_dir: Path, tiny_quants: dict, tmp_path: Path,
                                qbin: str, capsys):
    import shutil

    shutil.copyfile(tiny_quants["Q4_K_M"], model_dir / "tiny-Q4_K_M.gguf")
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=qbin)

    capsys.readouterr()
    rc = cli_main(["get", str(pack_dir), "q4_k_m", "--llama-quantize", qbin])
    assert rc == 0
    path = Path(capsys.readouterr().out.strip())
    assert path.read_bytes() == (model_dir / "tiny-Q4_K_M.gguf").read_bytes()
