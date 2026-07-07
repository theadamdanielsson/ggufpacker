"""ggufpacker: derivation attestations for GGUF quants and conversions —
chained, they trace a quant back to the published safetensors — and quant
ladders packed as recipes.

Every file is reconstructed bit-exact or not at all; every claim is proven
by re-derivation or refused.
"""

__version__ = "0.5.0"
