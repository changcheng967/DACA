"""FlashAttention implementation for Ascend NPUs.

The native FlashAttentionScore kernel WORKS on 910ProA with CANN 8.3.
This module provides a wrapper with fallback to BMM for edge cases.

WHY: FlashAttention is critical for transformer performance. The native
kernel is available and working, so we use it with fallback support.

Example:
    from daca.nn import FlashAttention

    attn = FlashAttention(head_dim=64, num_heads=32)
    output = attn(query, key, value)
"""

import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger("daca.nn.attention")


class FlashAttention:
    """FlashAttention layer using native Ascend kernel.

    Uses the native FlashAttentionScore kernel when available (works on
    910ProA with CANN 8.3), with BMM fallback for unsupported configs.

    Attributes:
        head_dim: Dimension of each attention head.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        scale: Scaling factor for attention scores.

    Example:
        >>> attn = FlashAttention(head_dim=64, num_heads=32)
        >>> q = ms.Tensor((batch, 32, seq, 64), ms.float16)
        >>> k = ms.Tensor((batch, 32, seq, 64), ms.float16)
        >>> v = ms.Tensor((batch, 32, seq, 64), ms.float16)
        >>> out = attn(q, k, v)
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        use_flash: bool = True,
    ):
        """Initialize FlashAttention.

        Args:
            head_dim: Dimension of each attention head.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            scale: Custom scale factor. Default is 1/sqrt(head_dim).
            use_flash: Whether to use FlashAttention kernel (default True).
        """
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)
        self._use_flash = use_flash
        self._flash_available = None

    def _check_flash_available(self) -> bool:
        """Check if native FlashAttention is available."""
        if self._flash_available is not None:
            return self._flash_available

        try:
            import mindspore.ops as ops
            # Check if FlashAttentionScore exists
            if hasattr(ops, 'FlashAttentionScore'):
                self._flash_available = True
                logger.debug("Native FlashAttentionScore available")
            else:
                self._flash_available = False
                logger.debug("FlashAttentionScore not available, will use BMM")
        except ImportError:
            self._flash_available = False

        return self._flash_available

    def construct(
        self,
        query: Any,
        key: Any,
        value: Any,
        mask: Optional[Any] = None,
        *,
        attn_mask: Optional[Any] = None,
    ) -> Any:
        """Compute attention.

        Args:
            query: Query tensor of shape (batch, heads, seq, dim).
            key: Key tensor of shape (batch, heads, seq, dim).
            value: Value tensor of shape (batch, heads, seq, dim).
            mask: Optional attention mask.
            attn_mask: Alternative name for mask.

        Returns:
            Output tensor of shape (batch, heads, seq, dim).
        """
        # Use native FlashAttention if available
        if self._use_flash and self._check_flash_available():
            return self._flash_forward(query, key, value, mask or attn_mask)
        else:
            return self._bmm_forward(query, key, value, mask or attn_mask)

    def __call__(
        self,
        query: Any,
        key: Any,
        value: Any,
        mask: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Call construct method."""
        return self.construct(query, key, value, mask, **kwargs)

    def _flash_forward(
        self,
        query: Any,
        key: Any,
        value: Any,
        mask: Optional[Any] = None,
    ) -> Any:
        """Forward pass using native FlashAttention kernel."""
        try:
            import mindspore.ops as ops

            # FlashAttentionScore expects specific input format
            # Input: (batch, seq, num_heads, head_dim) or (batch, heads, seq, dim)
            flash = ops.FlashAttentionScore(
                scale_value=self.scale,
                keep_prob=1.0 - self.dropout,
            )

            # Transpose from (batch, heads, seq, dim) to (batch, seq, heads, dim)
            batch_size = query.shape[0]
            num_heads = query.shape[1]
            seq_len = query.shape[2]

            query_t = ops.transpose(query, (0, 2, 1, 3))
            key_t = ops.transpose(key, (0, 2, 1, 3))
            value_t = ops.transpose(value, (0, 2, 1, 3))

            # Call FlashAttention
            output = flash(
                query_t, key_t, value_t,
                attn_mask=mask,
                actual_seq_qlen=None,
                actual_seq_kvlen=None,
            )

            # Transpose back to (batch, heads, seq, dim)
            if isinstance(output, tuple):
                output = output[0]  # First element is the attention output

            output = ops.transpose(output, (0, 2, 1, 3))
            return output

        except Exception as e:
            logger.warning(f"FlashAttention failed, falling back to BMM: {e}")
            return self._bmm_forward(query, key, value, mask)

    def _bmm_forward(
        self,
        query: Any,
        key: Any,
        value: Any,
        mask: Optional[Any] = None,
    ) -> Any:
        """Forward pass using BatchMatMul (fallback)."""
        try:
            import mindspore.ops as ops
            from mindspore import Tensor
            import mindspore.common.dtype as mstype
        except ImportError:
            raise ImportError("MindSpore is required for attention")

        # Compute attention scores: Q @ K^T
        # (batch, heads, seq, dim) @ (batch, heads, dim, seq) -> (batch, heads, seq, seq)
        key_t = ops.transpose(key, (0, 1, 3, 2))
        scores = ops.BatchMatMul(transpose_b=True)(query, key)

        # Scale
        scores = ops.mul(scores, self.scale)

        # Apply mask if provided
        if mask is not None:
            # Mask should be broadcastable to (batch, heads, seq, seq)
            scores = ops.add(scores, mask)

        # Softmax with fp32 upcast for stability
        original_dtype = scores.dtype
        scores_fp32 = ops.cast(scores, mstype.float32)
        attn_weights = ops.softmax(scores_fp32, axis=-1)
        attn_weights = ops.cast(attn_weights, original_dtype)

        # Apply dropout
        if self.dropout > 0:
            attn_weights = ops.dropout(attn_weights, p=self.dropout)

        # Apply attention to values: scores @ V
        output = ops.BatchMatMul()(attn_weights, value)

        return output


def scaled_dot_product_attention(
    query: Any,
    key: Any,
    value: Any,
    scale: Optional[float] = None,
    mask: Optional[Any] = None,
    dropout_p: float = 0.0,
) -> Any:
    """Compute scaled dot-product attention.

    Functional interface for attention computation.

    Args:
        query: Query tensor (..., seq_q, head_dim).
        key: Key tensor (..., seq_k, head_dim).
        value: Value tensor (..., seq_k, head_dim).
        scale: Scale factor. Default: 1/sqrt(head_dim).
        mask: Optional attention mask.
        dropout_p: Dropout probability.

    Returns:
        Attention output (..., seq_q, head_dim).

    Example:
        >>> from daca.nn import scaled_dot_product_attention
        >>> out = scaled_dot_product_attention(q, k, v)
    """
    try:
        import mindspore.ops as ops
        import mindspore.common.dtype as mstype
    except ImportError:
        raise ImportError("MindSpore is required for attention")

    head_dim = query.shape[-1]
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)

    # Q @ K^T
    key_t = ops.transpose(key, tuple(range(len(key.shape) - 2)) + (-1, -2))
    scores = ops.matmul(query, key_t)

    # Scale
    scores = ops.mul(scores, scale)

    # Apply mask
    if mask is not None:
        scores = ops.add(scores, mask)

    # Softmax (with fp32 upcast)
    original_dtype = scores.dtype
    scores_fp32 = ops.cast(scores, mstype.float32)
    attn_weights = ops.softmax(scores_fp32, axis=-1)
    attn_weights = ops.cast(attn_weights, original_dtype)

    # Dropout
    if dropout_p > 0:
        attn_weights = ops.dropout(attn_weights, p=dropout_p)

    # @ V
    output = ops.matmul(attn_weights, value)

    return output


def repeat_kv(hidden_states: Any, n_rep: int) -> Any:
    """Repeat key/value tensors for Grouped Query Attention.

    In GQA, the number of key/value heads may be less than query heads.
    This function repeats key/value heads to match query head count.

    Args:
        hidden_states: Tensor of shape (batch, num_kv_heads, seq, head_dim).
        n_rep: Number of times to repeat (num_q_heads / num_kv_heads).

    Returns:
        Tensor of shape (batch, num_q_heads, seq, head_dim).

    Example:
        >>> # For 32 query heads, 8 key/value heads (GQA-8)
        >>> k_expanded = repeat_kv(k, n_rep=4)
    """
    if n_rep == 1:
        return hidden_states

    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for repeat_kv")

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

    # (batch, num_kv_heads, seq, head_dim) -> (batch, num_kv_heads, 1, seq, head_dim)
    hidden_states = ops.expand_dims(hidden_states, 2)

    # (batch, num_kv_heads, 1, seq, head_dim) -> (batch, num_kv_heads, n_rep, seq, head_dim)
    hidden_states = ops.tile(hidden_states, (1, 1, n_rep, 1, 1))

    # (batch, num_kv_heads, n_rep, seq, head_dim) -> (batch, num_kv_heads * n_rep, seq, head_dim)
    hidden_states = ops.reshape(
        hidden_states,
        (batch, num_kv_heads * n_rep, seq_len, head_dim)
    )

    return hidden_states
