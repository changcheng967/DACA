"""Rotary Position Embeddings for Ascend NPUs.

Implements Rotary Position Embeddings (RoPE) for attention mechanisms.
Works with both standard and grouped-query attention.

Example:
    from daca.nn import RotaryEmbedding, apply_rotary_pos_emb

    rotary = RotaryEmbedding(dim=64, max_seq_len=2048)
    cos, sin = rotary(seq_len=512)
    q_rotated = apply_rotary_pos_emb(query, cos, sin)
"""

import logging
from typing import Tuple, Any

logger = logging.getLogger("daca.nn.rotary")


class RotaryEmbedding:
    """Rotary Position Embeddings.

    Generates cos/sin embeddings for rotary position encoding.
    Efficient implementation using precomputed frequencies.

    Attributes:
        dim: Dimension of the embedding (usually head_dim).
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.

    Reference:
        Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary
        Position Embedding" https://arxiv.org/abs/2104.09864

    Example:
        >>> rotary = RotaryEmbedding(dim=64, max_seq_len=2048)
        >>> cos, sin = rotary(512)
        >>> q_rotated = apply_rotary_pos_emb(q, cos, sin)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
    ):
        """Initialize RotaryEmbedding.

        Args:
            dim: Dimension of the embedding (head_dim).
            max_seq_len: Maximum sequence length.
            base: Base for computing inverse frequencies.
        """
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: 1 / (base^(2i/d))
        self._inv_freq = None
        self._cos_cached = None
        self._sin_cached = None

    def _get_inv_freq(self) -> Any:
        """Compute inverse frequencies."""
        try:
            import mindspore as ms
            from mindspore import Tensor
            import mindspore.ops as ops
            import mindspore.common.dtype as mstype
            import numpy as np
        except ImportError:
            raise ImportError("MindSpore is required for RotaryEmbedding")

        if self._inv_freq is not None:
            return self._inv_freq

        # Compute: 1 / (base^(2i/d))
        freq_indices = np.arange(0, self.dim, 2, dtype=np.float32)
        inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))

        self._inv_freq = Tensor(inv_freq, mstype.float32)
        return self._inv_freq

    def construct(self, seq_len: int) -> Tuple[Any, Any]:
        """Compute cos/sin embeddings for given sequence length.

        Args:
            seq_len: Current sequence length.

        Returns:
            Tuple of (cos, sin) tensors, each of shape (seq_len, dim).
        """
        try:
            import mindspore.ops as ops
            import mindspore.common.dtype as mstype
        except ImportError:
            raise ImportError("MindSpore is required for RotaryEmbedding")

        inv_freq = self._get_inv_freq()

        # Position indices: [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        t = ops.arange(seq_len, dtype=mstype.float32)

        # Compute freqs: t @ inv_freq
        # Shape: (seq_len, dim/2)
        freqs = ops.outer(t, inv_freq)

        # Duplicate for full dimension
        # Shape: (seq_len, dim)
        emb = ops.concat([freqs, freqs], axis=-1)

        cos = ops.cos(emb)
        sin = ops.sin(emb)

        return cos, sin

    def __call__(self, seq_len: int) -> Tuple[Any, Any]:
        """Call construct method."""
        return self.construct(seq_len)


def apply_rotary_pos_emb(
    x: Any,
    cos: Any,
    sin: Any,
    position_ids: Any = None,
) -> Any:
    """Apply rotary position embeddings to input tensor.

    Rotates pairs of features by the position-dependent angle.

    Args:
        x: Input tensor of shape (..., seq_len, head_dim).
        cos: Cosine embeddings of shape (seq_len, head_dim).
        sin: Sine embeddings of shape (seq_len, head_dim).
        position_ids: Optional position indices (for caching).

    Returns:
        Rotated tensor with same shape as input.

    Example:
        >>> cos, sin = rotary(seq_len=512)
        >>> q_rotated = apply_rotary_pos_emb(query, cos, sin)
    """
    try:
        import mindspore.ops as ops
        import mindspore.common.dtype as mstype
    except ImportError:
        raise ImportError("MindSpore is required for apply_rotary_pos_emb")

    # Get shapes
    x_shape = x.shape
    seq_len = x_shape[-2]
    head_dim = x_shape[-1]

    # Reshape cos/sin for broadcasting
    # cos/sin: (seq_len, dim) -> (1, 1, seq_len, dim) for 4D input
    while len(cos.shape) < len(x_shape):
        cos = ops.expand_dims(cos, 0)
        sin = ops.expand_dims(sin, 0)

    # Handle position_ids for indexing
    if position_ids is not None:
        cos = ops.gather(cos, position_ids, axis=-2)
        sin = ops.gather(sin, position_ids, axis=-2)

    # Apply rotation
    # rotate_half: swap first and second half, negate first half
    # x = [x1, x2] -> [-x2, x1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    rotated = ops.concat([-x2, x1], axis=-1)

    # Apply cos/sin
    # x * cos + rotated * sin
    return x * cos + rotated * sin


def rotate_half(x: Any) -> Any:
    """Rotate half the hidden dims of the input.

    Helper for rotary embeddings.

    Args:
        x: Input tensor (..., head_dim).

    Returns:
        Rotated tensor with same shape.
    """
    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for rotate_half")

    head_dim = x.shape[-1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    return ops.concat([-x2, x1], axis=-1)


def compute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> Tuple[Any, Any]:
    """Compute RoPE frequencies for all positions.

    Functional interface for computing cos/sin embeddings.

    Args:
        dim: Dimension of the embedding.
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.

    Returns:
        Tuple of (cos, sin) tensors of shape (max_seq_len, dim).

    Example:
        >>> cos, sin = compute_rope_freqs(dim=64, max_seq_len=2048)
    """
    rotary = RotaryEmbedding(dim=dim, max_seq_len=max_seq_len, base=int(base))
    return rotary(max_seq_len)
