from __future__ import annotations

import pytest

from ggufpacker.manifest import FORMAT, AmbiguousNameError, FileEntry, Manifest


def test_format_tag_is_stable() -> None:
    """The on-disk format tag is part of the artifact, not the branding.

    It must never change with a project rename: existing .ggufpack stores
    carry this exact string, and changing it silently orphans them all
    (this regressed once, during the ggufpack -> ggufpacker rename).
    """
    assert FORMAT == "ggufpack/0"


# ---------------------------------------------------------- name resolution

def _entry(filename: str, recipe: dict | None = None) -> FileEntry:
    return FileEntry(filename=filename, size=1, sha256="0" * 64,
                     role="quant", plan="exact", recipe=recipe)


def _manifest(*files: FileEntry) -> Manifest:
    return Manifest(format=FORMAT, created="now", tool_version="test",
                    quantize={}, files=list(files))


def _variant_manifest() -> Manifest:
    """Mirrors the real bartowski failure: Q4_K_L's recipe is BASE Q4_K_M plus
    a token-embedding override, and 'L' iterates before 'M' in the file list
    (which is how the old recipe-first resolver picked the wrong file)."""
    return _manifest(
        _entry("X-Q4_K_L.gguf",
               {"qtype": "Q4_K_M", "token_embedding_type": "q8_0"}),
        _entry("X-Q4_K_M.gguf", {"qtype": "Q4_K_M"}),
    )


def test_find_by_type_prefers_filename_suffix_over_recipe_base() -> None:
    m = _variant_manifest()
    assert m.find("Q4_K_M").filename == "X-Q4_K_M.gguf"
    assert m.find("q4_k_m").filename == "X-Q4_K_M.gguf"  # case-insensitive


def test_find_variant_by_its_own_suffix() -> None:
    m = _variant_manifest()
    assert m.find("Q4_K_L").filename == "X-Q4_K_L.gguf"
    assert m.find("q4_k_l").filename == "X-Q4_K_L.gguf"


def test_find_exact_filename_always_wins() -> None:
    m = _variant_manifest()
    assert m.find("X-Q4_K_L.gguf").filename == "X-Q4_K_L.gguf"
    assert m.find("X-Q4_K_M.gguf").filename == "X-Q4_K_M.gguf"


def test_find_recipe_base_still_resolves_when_unambiguous() -> None:
    # Only the L variant exists: querying its base type must still find it.
    m = _manifest(
        _entry("X-Q4_K_L.gguf",
               {"qtype": "Q4_K_M", "token_embedding_type": "q8_0"}),
    )
    assert m.find("Q4_K_M").filename == "X-Q4_K_L.gguf"


def test_find_ambiguous_recipe_base_refuses_with_candidates() -> None:
    # Two variants share base Q4_K_M and neither filename carries the suffix:
    # picking either silently would hand the user the wrong file.
    m = _manifest(
        _entry("X-Q4_K_L.gguf",
               {"qtype": "Q4_K_M", "token_embedding_type": "q8_0"}),
        _entry("X-Q4_K_XL.gguf",
               {"qtype": "Q4_K_M", "token_embedding_type": "q8_0",
                "output_tensor_type": "q8_0"}),
    )
    with pytest.raises(AmbiguousNameError) as exc:
        m.find("Q4_K_M")
    msg = str(exc.value)
    assert "ambiguous type 'Q4_K_M'" in msg
    assert "X-Q4_K_L.gguf" in msg and "X-Q4_K_XL.gguf" in msg
    assert "use a filename" in msg


def test_find_ambiguous_suffix_refuses_with_candidates() -> None:
    m = _manifest(
        _entry("A-Q4_K_M.gguf", {"qtype": "Q4_K_M"}),
        _entry("B-Q4_K_M.gguf", {"qtype": "Q4_K_M"}),
    )
    with pytest.raises(AmbiguousNameError) as exc:
        m.find("Q4_K_M")
    msg = str(exc.value)
    assert "A-Q4_K_M.gguf" in msg and "B-Q4_K_M.gguf" in msg


def test_find_no_match_returns_none() -> None:
    assert _variant_manifest().find("Q8_0") is None


def test_ambiguous_name_error_is_a_lookup_error() -> None:
    # get/exec surface resolution failures through a LookupError catch; the
    # ambiguity refusal must ride the same path (CLI exit 1, no path printed).
    assert issubclass(AmbiguousNameError, LookupError)
