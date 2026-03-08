"""Numerically stable softmax for Ascend NPUs.

Provides softmax with automatic fp32 upcast for numerical stability,
especially important for large vocabulary or attention computations.

Example:
    from daca.nn import softmax

    probs = softmax(logits, axis=-1)
"""

import logging
from typing import Optional, Any

logger = logging.getLogger("daca.nn.softmax")


def softmax(x: Any, axis: int = -1, dtype: Optional[Any] = None) -> Any:
    """Compute softmax with numerical stability.

    Upcasts to fp32 for computation to avoid overflow/underflow
    in fp16, then casts back to original or specified dtype.

    Args:
        x: Input tensor.
        axis: Axis to compute softmax over.
        dtype: Output dtype. If None, uses input dtype.

    Returns:
        Softmax probabilities with same shape as input.

    Example:
        >>> from daca.nn import softmax
        >>> probs = softmax(logits, axis=-1)
    """
    try:
        import mindspore.ops as ops
        import mindspore.common.dtype as mstype
    except ImportError:
        raise ImportError("MindSpore is required for softmax")

    original_dtype = x.dtype
    compute_dtype = dtype if dtype is not None else original_dtype

    # Upcast to fp32 for numerical stability
    if compute_dtype == mstype.float16:
        x = ops.cast(x, mstype.float32)

    # Compute softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    x_max = ops.max(x, axis=axis, keep_dims=True)[0]
    x_exp = ops.exp(x - x_max)
    x_sum = ops.sum(x_exp, axis=axis, keep_dims=True)

    result = x_exp / x_sum

    # Cast back to desired dtype
    if dtype is not None:
        result = ops.cast(result, dtype)
    elif original_dtype == mstype.float16:
        result = ops.cast(result, original_dtype)

    return result


def log_softmax(x: Any, axis: int = -1) -> Any:
    """Compute log softmax with numerical stability.

    More numerically stable than log(softmax(x)).

    Args:
        x: Input tensor.
        axis: Axis to compute over.

    Returns:
        Log softmax values.

    Example:
        >>> from daca.nn import log_softmax
        >>> log_probs = log_softmax(logits, axis=-1)
    """
    try:
        import mindspore.ops as ops
        import mindspore.common.dtype as mstype
    except ImportError:
        raise ImportError("MindSpore is required for log_softmax")

    original_dtype = x.dtype

    # Upcast to fp32 for numerical stability
    if original_dtype == mstype.float16:
        x = ops.cast(x, mstype.float32)

    # Compute log_softmax: x - max(x) - log(sum(exp(x - max(x))))
    x_max = ops.max(x, axis=axis, keep_dims=True)[0]
    x_shifted = x - x_max
    x_exp = ops.exp(x_shifted)
    x_sum = ops.sum(x_exp, axis=axis, keep_dims=True)
    log_sum = ops.log(x_sum)

    result = x_shifted - log_sum

    # Cast back
    if original_dtype == mstype.float16:
        result = ops.cast(result, original_dtype)

    return result


def scaled_dot_product_attention(
    query: Any,
    key: Any,
    value: Any,
    scale: Optional[float] = None,
    attn_mask: Optional[Any] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Any:
    """Scaled dot-product attention helper.

    Combines the common attention pattern into a single function.

    Args:
        query: Query tensor (..., seq_q, head_dim).
        key: Key tensor (..., seq_k, head_dim).
        value: Value tensor (..., seq_k, head_dim).
        scale: Scale factor. Default: 1/sqrt(head_dim).
        attn_mask: Optional attention mask.
        dropout_p: Dropout probability.
        is_causal: Whether to apply causal mask.

    Returns:
        Attention output (..., seq_q, head_dim).

    Example:
        >>> from daca.nn import scaled_dot_product_attention
        >>> out = scaled_dot_product_attention(q, k, v, is_causal=True)
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

    # Apply causal mask if needed
    if is_causal:
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        causal_mask = _create_causal_mask(seq_q, seq_k, scores.dtype, scores.device if hasattr(scores, 'device') else None)
        if attn_mask is None:
            attn_mask = causal_mask
        else:
            attn_mask = ops.minimum(attn_mask, causal_mask)

    # Apply mask
    if attn_mask is not None:
        scores = ops.add(scores, attn_mask)

    # Softmax (with fp32 upcast via our softmax)
    attn_weights = softmax(scores, axis=-1)

    # Dropout
    if dropout_p > 0:
        attn_weights = ops.dropout(attn_weights, p=dropout_p)

    # @ V
    output = ops.matmul(attn_weights, value)

    return output


def _create_causal_mask(
    seq_q: int,
    seq_k: int,
    dtype: Any,
    device: Any = None,
) -> Any:
    """Create causal attention mask.

    Args:
        seq_q: Query sequence length.
        seq_k: Key sequence length.
        dtype: Data type for the mask.
        device: Device for the mask tensor.

    Returns:
        Mask tensor where -inf indicates masked positions.
    """
    try:
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        import mindspore.ops as ops
        import numpy as np
    except ImportError:
        raise ImportError("MindSpore is required for causal mask")

    # Create mask: mask[i, j] = 1 if j > i else 0
    # For attention: we want to mask future positions
    # Positions to mask get -inf
    i = np.arange(seq_q).reshape(-1, 1)
    j = np.arange(seq_k).reshape(1, -1)

    # For causal: j > i means future, should be masked
    mask = (j > i).astype(np.float32) * -1e9  # -inf approximation

    return Tensor(mask, dtype)
