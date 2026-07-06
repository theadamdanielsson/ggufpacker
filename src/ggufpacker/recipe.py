"""Recipe inference: which llama-quantize invocation reproduces a published file.

Two signals, in order:
1. Filename suffix ("...-Q4_K_M.gguf"), including bartowski's custom _L/_XL
   variants which are a base type plus --token-embedding-type/--output-tensor-type
   overrides (detected from the published file's tensor-type map, not the name).
2. Tensor-type histogram of the file itself, when the name is unhelpful. K-quant
   "mixtures" share a dominant block type (e.g. Q4_K covers Q4_K_S and Q4_K_M),
   so a histogram yields an ordered candidate list to try, not a single answer.

Nothing here is trusted blindly: packer.py always compares the regenerated
tensor-type map against the published one and falls back to a stored blob when
the recipe cannot be made to match.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from gguf.constants import GGMLQuantizationType

from .layout import GGUFLayout

# bartowski custom variants -> base llama-quantize type (validated in Phase 0/1
# experiments; the _L/_XL part is expressed via tensor-type overrides).
CUSTOM_BASE: dict[str, str] = {
    "Q2_K_L": "Q2_K",
    "Q3_K_XL": "Q3_K_L",
    "Q4_K_L": "Q4_K_M",
    "Q4_K_XL": "Q4_K_M",
    "Q5_K_L": "Q5_K_M",
    "Q6_K_L": "Q6_K",
}

# llama-quantize CLI type names as of b3821.
KNOWN_QTYPES: frozenset[str] = frozenset({
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q2_K_S", "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_K", "Q4_K_S", "Q4_K_M", "Q5_K", "Q5_K_S", "Q5_K_M", "Q6_K",
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS",
    "TQ1_0", "TQ2_0",
    "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8",
    "F16", "BF16", "F32",
})

# Dominant GGML block type -> ordered llama-quantize candidates (most common
# in the wild first). K-quant mixtures (S/M/L) share the same dominant type.
_HISTOGRAM_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Q4_0": ("Q4_0",), "Q4_1": ("Q4_1",), "Q5_0": ("Q5_0",), "Q5_1": ("Q5_1",),
    "Q8_0": ("Q8_0",),
    "Q2_K": ("Q2_K", "Q2_K_S"),
    "Q3_K": ("Q3_K_M", "Q3_K_L", "Q3_K_S"),
    "Q4_K": ("Q4_K_M", "Q4_K_S"),
    "Q5_K": ("Q5_K_M", "Q5_K_S"),
    "Q6_K": ("Q6_K",),
    "IQ1_S": ("IQ1_S",), "IQ1_M": ("IQ1_M",),
    "IQ2_XXS": ("IQ2_XXS",), "IQ2_XS": ("IQ2_XS",),
    "IQ2_S": ("IQ2_M", "IQ2_S"),  # IQ2_M is an IQ2_S-block mixture
    "IQ3_XXS": ("IQ3_XXS",),
    "IQ3_S": ("IQ3_M", "IQ3_S", "IQ3_XS"),
    "IQ4_NL": ("IQ4_NL",), "IQ4_XS": ("IQ4_XS",),
    "TQ1_0": ("TQ1_0",), "TQ2_0": ("TQ2_0",),
    "F16": ("F16",), "BF16": ("BF16",), "F32": ("F32",),
}

_SUFFIX_RE = re.compile(r"-([A-Za-z0-9_]+)\.gguf$", re.IGNORECASE)

# Tensors llama-quantize can override per-tensor from the CLI.
OVERRIDABLE_TENSORS = frozenset({"token_embd.weight", "output.weight"})


@dataclass
class Recipe:
    """One llama-quantize invocation, minus machine-local paths."""

    qtype: str
    token_embedding_type: str | None = None  # lowercase ggml type name, e.g. "q8_0"
    output_tensor_type: str | None = None
    use_imatrix: bool = True

    def cli_flags(self) -> list[str]:
        """Recipe as recorded in the manifest (paths get bound at run time)."""
        flags: list[str] = []
        if self.use_imatrix:
            flags += ["--imatrix", "<IMATRIX>"]
        if self.token_embedding_type:
            flags += ["--token-embedding-type", self.token_embedding_type]
        if self.output_tensor_type:
            flags += ["--output-tensor-type", self.output_tensor_type]
        flags += ["<SOURCE>", "<OUT>", self.qtype]
        return flags


@dataclass
class RecipeGuess:
    """Ordered llama-quantize base-type candidates for one published file."""

    candidates: tuple[str, ...]
    origin: str  # "filename" | "histogram"
    notes: list[str] = field(default_factory=list)


def model_stem(filename: str) -> str:
    """Model-identity part of a filename: extension dropped, trailing quant-type
    suffix (`-Q4_K_M`, `-f16`, custom `_L`/`_XL` variants) stripped.

    "Llama-3.2-1B-Instruct-Q4_K_M.gguf" -> "Llama-3.2-1B-Instruct"
    "Llama-3.2-1B-Instruct-f16.gguf"    -> "Llama-3.2-1B-Instruct"
    "Llama-3.2-1B-Instruct.imatrix"     -> "Llama-3.2-1B-Instruct"

    Used for filename-prefix affinity tiebreaks and display grouping only —
    never for correctness decisions (those rest on tensor identity + the
    verify-or-refuse machinery).
    """
    name = filename.rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    m = _SUFFIX_RE.search(name + ".gguf")  # reuse the quant-suffix pattern
    if m:
        suffix = m.group(1).upper()
        if suffix in KNOWN_QTYPES or suffix in CUSTOM_BASE:
            name = name[: m.start()]
    return name


def type_name(ggml_type: int) -> str:
    try:
        return GGMLQuantizationType(ggml_type).name
    except ValueError:
        return f"UNKNOWN_{ggml_type}"


def tensor_type_map(layout: GGUFLayout) -> dict[str, str]:
    return {t.name: type_name(t.ggml_type) for t in layout.tensors}


def dominant_type(layout: GGUFLayout) -> str:
    """Most common tensor type among >=2-D tensors (1-D norms stay F32 in every
    quant and would drown out the signal on small models)."""
    counts = Counter(
        type_name(t.ggml_type) for t in layout.tensors if len(t.dims) >= 2
    )
    if not counts:
        counts = Counter(type_name(t.ggml_type) for t in layout.tensors)
    return counts.most_common(1)[0][0] if counts else "UNKNOWN"


def guess_recipe(filename: str, layout: GGUFLayout) -> RecipeGuess | None:
    """Return candidate base types, or None if we have no plausible recipe."""
    m = _SUFFIX_RE.search(filename)
    if m:
        suffix = m.group(1).upper()
        if suffix in CUSTOM_BASE:
            return RecipeGuess(
                candidates=(CUSTOM_BASE[suffix],),
                origin="filename",
                notes=[f"custom variant {suffix} -> base {CUSTOM_BASE[suffix]}"],
            )
        if suffix in KNOWN_QTYPES:
            return RecipeGuess(candidates=(suffix,), origin="filename")

    dom = dominant_type(layout)
    cands = _HISTOGRAM_CANDIDATES.get(dom)
    if cands:
        return RecipeGuess(
            candidates=cands,
            origin="histogram",
            notes=[f"filename gave no known type; dominant tensor type {dom}"],
        )
    return None


def detect_overrides(
    published: dict[str, str], regen: dict[str, str]
) -> dict[str, str] | None:
    """Compare tensor-type maps after a base-type attempt.

    Returns {} if they match, {tensor: published_type_name} if the mismatch is
    confined to CLI-overridable tensors, None if the recipe cannot match.
    """
    if set(published) != set(regen):
        return None
    diff = {n: published[n] for n in published if published[n] != regen[n]}
    if not diff:
        return {}
    if set(diff) <= OVERRIDABLE_TENSORS:
        return diff
    return None


def override_cli_name(ggml_name: str) -> str:
    """GGML type enum name -> the lowercase name ggml_type_name() uses,
    which is what --token-embedding-type/--output-tensor-type parse."""
    return ggml_name.lower()
