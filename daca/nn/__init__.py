"""DACA Neural Network Module.

Core NN layers with Ascend-specific fixes and optimizations.

Key components:
- FlashAttention: Native kernel wrapper with BMM fallback
- LayerNorm: fp32 upcast to avoid CANN fusion bug
- RMSNorm: Manual decomposition using rsqrt + mul + mul
- activations: Missing ops like SiLU, SwiGLU
- RotaryEmbedding: Position embeddings for attention
"""

from daca.nn.attention import (
    FlashAttention,
    scaled_dot_product_attention,
    repeat_kv,
)
from daca.nn.layernorm import (
    LayerNorm,
    enable_fp32_upcast,
    disable_fp32_upcast,
)
from daca.nn.rmsnorm import RMSNorm
from daca.nn.activations import (
    silu,
    swiglu,
    inject_silu,
    inject_swiglu,
    remove_silu,
    remove_swiglu,
)
from daca.nn.rotary import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from daca.nn.embedding import Embedding
from daca.nn.softmax import (
    softmax,
    log_softmax,
)

__all__ = [
    # Attention
    "FlashAttention",
    "scaled_dot_product_attention",
    "repeat_kv",
    # LayerNorm
    "LayerNorm",
    "enable_fp32_upcast",
    "disable_fp32_upcast",
    # RMSNorm
    "RMSNorm",
    # Activations
    "silu",
    "swiglu",
    "inject_silu",
    "inject_swiglu",
    "remove_silu",
    "remove_swiglu",
    # Rotary
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    # Embedding
    "Embedding",
    # Softmax
    "softmax",
    "log_softmax",
]
