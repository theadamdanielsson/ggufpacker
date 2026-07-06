"""Cache LRU size cap: `cache prune --max-size` evicts by mtime (oldest
first) until the cache fits; $GGUFPACKER_CACHE_MAX applies the same eviction
at the end of every `get`, never evicting the file just returned."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ggufpacker.cache import get, parse_size, prune_to_size
from ggufpacker.cli import main as cli_main
from ggufpacker.packer import pack

GARBAGE = bytes(range(256)) * 64  # 16,384 B


@pytest.fixture(autouse=True)
def cache_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "gpcache"
    monkeypatch.setenv("GGUFPACKER_CACHE", str(d))
    monkeypatch.delenv("GGUFPACKER_CACHE_MAX", raising=False)
    return d


@pytest.fixture()
def two_file_pack(model_dir: Path, tmp_path: Path) -> Path:
    (model_dir / "tiny-ALPHA.gguf").write_bytes(GARBAGE)
    (model_dir / "tiny-BETA.gguf").write_bytes(GARBAGE[::-1])
    pack_dir = tmp_path / "p.ggufpack"
    pack(model_dir, pack_dir, llama_quantize=None)
    return pack_dir


def _fake_cache(root: Path, spec: list[tuple[str, str, int, int]]) -> dict[str, Path]:
    """Build a synthetic cache: (packdir, filename, size, mtime) per file."""
    out = {}
    for packdir, name, size, mtime in spec:
        d = root / packdir
        d.mkdir(parents=True, exist_ok=True)
        (d / ".pack.json").write_text("{}")
        f = d / name
        f.write_bytes(b"\0" * size)
        os.utime(f, (mtime, mtime))
        out[name] = f
    return out


# ------------------------------------------------------------------ parse_size

@pytest.mark.parametrize("text,expect", [
    ("100", 100),
    ("2K", 2048),
    ("500M", 500 * 1024**2),
    ("20G", 20 * 1024**3),
    ("1T", 1024**4),
    ("2.5G", int(2.5 * 1024**3)),
    ("500m", 500 * 1024**2),
    ("500MB", 500 * 1024**2),
])
def test_parse_size(text: str, expect: int):
    assert parse_size(text) == expect


@pytest.mark.parametrize("bad", ["", "G", "-1G", "ten", "1..5G", "5X"])
def test_parse_size_rejects_garbage(bad: str):
    with pytest.raises(ValueError):
        parse_size(bad)


# ------------------------------------------------------------ eviction order

def test_prune_evicts_least_recently_used_first(cache_env: Path):
    files = _fake_cache(cache_env, [
        ("packA", "old.gguf", 1000, 1_000),
        ("packA", "newer.gguf", 1000, 3_000),
        ("packB", "oldest.gguf", 1000, 500),
        ("packB", "newest.gguf", 1000, 4_000),
    ])
    stats = prune_to_size(2000)
    assert [p.name for p, _ in stats.evicted] == ["oldest.gguf", "old.gguf"]
    assert stats.freed == 2000 and stats.remaining == 2000
    assert not files["oldest.gguf"].exists() and not files["old.gguf"].exists()
    assert files["newer.gguf"].exists() and files["newest.gguf"].exists()


def test_prune_under_cap_evicts_nothing(cache_env: Path):
    files = _fake_cache(cache_env, [("packA", "a.gguf", 1000, 1_000)])
    stats = prune_to_size(1000)
    assert stats.evicted == [] and stats.remaining == 1000
    assert files["a.gguf"].exists()


def test_prune_removes_emptied_pack_dirs_and_metadata(cache_env: Path):
    _fake_cache(cache_env, [
        ("packA", "a.gguf", 1000, 1_000),
        ("packB", "b.gguf", 1000, 2_000),
    ])
    stats = prune_to_size(1000)
    assert [p.name for p, _ in stats.evicted] == ["a.gguf"]
    assert not (cache_env / "packA").exists(), "emptied pack dir + metadata removed"
    assert (cache_env / "packB" / "b.gguf").exists()


def test_prune_never_evicts_protected_file_even_if_over_cap(cache_env: Path):
    files = _fake_cache(cache_env, [
        ("packA", "precious.gguf", 5000, 1_000),  # oldest AND over cap alone
        ("packA", "other.gguf", 1000, 2_000),
    ])
    stats = prune_to_size(100, protect=files["precious.gguf"])
    assert files["precious.gguf"].exists()
    assert [p.name for p, _ in stats.evicted] == ["other.gguf"]
    assert stats.remaining == 5000, "cache may stay over cap rather than evict the protected file"


# ------------------------------------------------------------------- CLI

def test_cli_cache_prune_reports_evictions(cache_env: Path, capsys):
    _fake_cache(cache_env, [
        ("packA", "old.gguf", 1000, 1_000),
        ("packA", "new.gguf", 1000, 2_000),
    ])
    capsys.readouterr()
    rc = cli_main(["cache", "prune", "--max-size", "1000"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "evicted old.gguf" in out
    assert "freed" in out and "cache now" in out


def test_cli_cache_prune_bad_size_exits_1(capsys):
    capsys.readouterr()
    rc = cli_main(["cache", "prune", "--max-size", "lots"])
    assert rc == 1
    assert "invalid size" in capsys.readouterr().err


# ------------------------------------------------------- GGUFPACKER_CACHE_MAX

def test_env_cap_evicts_lru_after_get_but_never_the_returned_file(
    two_file_pack: Path, monkeypatch: pytest.MonkeyPatch
):
    p1 = get(two_file_pack, "tiny-ALPHA.gguf")
    assert p1.is_file()

    # cap smaller than one file: after the next get, ALPHA (older) must go,
    # but BETA — the file being returned — must survive the eviction pass
    monkeypatch.setenv("GGUFPACKER_CACHE_MAX", "1K")
    p2 = get(two_file_pack, "tiny-BETA.gguf")
    assert p2.is_file(), "the file just returned is never evicted"
    assert not p1.exists(), "older cached file must be LRU-evicted"


def test_env_cap_applies_on_cache_hits_too(
    two_file_pack: Path, monkeypatch: pytest.MonkeyPatch
):
    p1 = get(two_file_pack, "tiny-ALPHA.gguf")
    p2 = get(two_file_pack, "tiny-BETA.gguf")
    assert p1.exists() and p2.exists()

    monkeypatch.setenv("GGUFPACKER_CACHE_MAX", "1K")
    p2_again = get(two_file_pack, "tiny-BETA.gguf")  # hit
    assert p2_again == p2 and p2.exists()
    assert not p1.exists(), "hits also enforce the cap (evicting the LRU file)"


def test_env_cap_invalid_value_is_ignored_with_warning(
    two_file_pack: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.setenv("GGUFPACKER_CACHE_MAX", "banana")
    capsys.readouterr()
    p = get(two_file_pack, "tiny-ALPHA.gguf")
    assert p.is_file(), "a bad cap must not break get"
    assert "ignoring GGUFPACKER_CACHE_MAX" in capsys.readouterr().err
