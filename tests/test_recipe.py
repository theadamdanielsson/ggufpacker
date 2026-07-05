from pathlib import Path

from ggufpack.layout import parse_layout
from ggufpack.recipe import (
    detect_overrides,
    dominant_type,
    guess_recipe,
    override_cli_name,
)


def test_filename_suffix_direct(tiny_f16: Path):
    lay = parse_layout(tiny_f16)
    g = guess_recipe("Llama-3.2-1B-Instruct-Q4_K_M.gguf", lay)
    assert g is not None and g.origin == "filename" and g.candidates == ("Q4_K_M",)


def test_filename_suffix_case_insensitive(tiny_f16: Path):
    lay = parse_layout(tiny_f16)
    g = guess_recipe("model-q8_0.gguf", lay)
    assert g is not None and g.candidates == ("Q8_0",)


def test_filename_custom_variants_map_to_base(tiny_f16: Path):
    lay = parse_layout(tiny_f16)
    for suffix, base in (("Q4_K_L", "Q4_K_M"), ("Q3_K_XL", "Q3_K_L"), ("Q6_K_L", "Q6_K")):
        g = guess_recipe(f"model-{suffix}.gguf", lay)
        assert g is not None and g.candidates == (base,), suffix


def test_histogram_fallback_when_name_unhelpful(tiny_f16: Path):
    # tiny_f16 is dominated by F16 2-D tensors -> histogram says F16
    lay = parse_layout(tiny_f16)
    assert dominant_type(lay) == "F16"
    g = guess_recipe("model-final-v2.gguf", lay)
    assert g is not None and g.origin == "histogram" and g.candidates == ("F16",)


def test_detect_overrides():
    pub = {"token_embd.weight": "Q8_0", "output.weight": "Q6_K", "blk.0.attn_q.weight": "Q4_K"}
    reg_match = dict(pub)
    assert detect_overrides(pub, reg_match) == {}

    reg_ovr = dict(pub, **{"token_embd.weight": "Q4_K"})
    assert detect_overrides(pub, reg_ovr) == {"token_embd.weight": "Q8_0"}

    reg_bad = dict(pub, **{"blk.0.attn_q.weight": "Q5_K"})
    assert detect_overrides(pub, reg_bad) is None

    assert detect_overrides(pub, {"other.weight": "Q4_K"}) is None


def test_override_cli_name():
    assert override_cli_name("Q8_0") == "q8_0"
    assert override_cli_name("IQ4_NL") == "iq4_nl"
