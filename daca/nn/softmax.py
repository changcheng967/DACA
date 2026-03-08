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
