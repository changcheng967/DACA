"""BatchMatMul operations for Ascend NPUs.

Provides batched matrix multiplication with proper handling
of 3D and higher-dimensional tensors.

Example:
    from daca.blas import bmm

    # (batch, m, k) @ (batch, k, n) -> (batch, m, n)
    result = bmm(a, b)
"""

import logging
from typing import Tuple, Any

logger = logging.getLogger("daca.blas.bmm")


def bmm(a: Any, b: Any) -> Any:
    """Batched matrix multiplication.

    Performs batched matrix multiplication on 3D tensors:
    (batch, m, k) @ (batch, k, n) -> (batch, m, n)

    Args:
        a: First tensor of shape (batch, m, k).
        b: Second tensor of shape (batch, k, n).

    Returns:
        Result tensor of shape (batch, m, n).

    Raises:
        ValueError: If shapes are incompatible.
        ValueError: If tensors are not 3D.

    Example:
        >>> import mindspore as ms
        >>> from daca.blas import bmm
        >>> a = ms.Tensor.ones((2, 3, 4), ms.float16)
        >>> b = ms.Tensor.ones((2, 4, 5), ms.float16)
        >>> c = bmm(a, b)
        >>> print(c.shape)
        (2, 3, 5)
    """
    try:
        import mindspore as ms
        import mindspore.ops as ops
        from mindspore import Tensor
    except ImportError:
        raise ImportError("MindSpore is required for bmm operations")

    # Validate inputs
    if not isinstance(a, (Tensor, ms.Tensor)):
        a = Tensor(a)
    if not isinstance(b, (Tensor, ms.Tensor)):
        b = Tensor(b)

    shape_a = a.shape
    shape_b = b.shape

    # Validate 3D tensors
    if len(shape_a) != 3:
        raise ValueError(f"bmm requires 3D tensors, got shape {shape_a} for a")
    if len(shape_b) != 3:
        raise ValueError(f"bmm requires 3D tensors, got shape {shape_b} for b")

    # Validate batch dimensions match
    if shape_a[0] != shape_b[0]:
        raise ValueError(
            f"bmm batch dimensions must match: {shape_a[0]} != {shape_b[0]}"
        )

    # Validate K dimensions
    k_a = shape_a[2]
    k_b = shape_b[1]

    if k_a != k_b:
        raise ValueError(
            f"bmm shapes incompatible: {shape_a} @ {shape_b} "
            f"(K dimensions: {k_a} != {k_b})"
        )

    # Use MindSpore BatchMatMul
    return ops.BatchMatMul()(a, b)


def batch_matmul(a: Any, b: Any) -> Any:
    """Alias for bmm.

    Provided for API compatibility.

    Args:
        a: First tensor of shape (batch, m, k).
        b: Second tensor of shape (batch, k, n).

    Returns:
        Result tensor of shape (batch, m, n).
    """
    return bmm(a, b)


def bmm_with_broadcast(a: Any, b: Any) -> Any:
    """Batched matrix multiplication with broadcasting support.

    Supports broadcasting of batch dimensions, similar to torch.bmm
    but with more flexible shape handling.

    Args:
        a: First tensor of shape (..., m, k).
        b: Second tensor of shape (..., k, n).

    Returns:
        Result tensor of shape (..., m, n).

    Example:
        >>> # Broadcast batch dims
        >>> a = ms.Tensor.ones((1, 8, 64, 32), ms.float16)  # (1, heads, seq, dim)
        >>> b = ms.Tensor.ones((4, 8, 32, 64), ms.float16)  # (batch, heads, dim, seq)
        >>> c = bmm_with_broadcast(a, b)
        >>> print(c.shape)
        (4, 8, 64, 64)
    """
    try:
        import mindspore as ms
        import mindspore.ops as ops
        from mindspore import Tensor
    except ImportError:
        raise ImportError("MindSpore is required for bmm operations")

    # Validate inputs
    if not isinstance(a, (Tensor, ms.Tensor)):
        a = Tensor(a)
    if not isinstance(b, (Tensor, ms.Tensor)):
        b = Tensor(b)

    shape_a = a.shape
    shape_b = b.shape

    # Validate at least 3D
    if len(shape_a) < 3 or len(shape_b) < 3:
        raise ValueError(
            f"bmm_with_broadcast requires 3D+ tensors, got {shape_a} and {shape_b}"
        )

    # Check K dimension
    k_a = shape_a[-1]
    k_b = shape_b[-2]

    if k_a != k_b:
        raise ValueError(
            f"bmm shapes incompatible: {shape_a} @ {shape_b} "
            f"(K dimensions: {k_a} != {k_b})"
        )

    # For exactly 3D, use standard bmm
    if len(shape_a) == 3 and len(shape_b) == 3:
        return bmm(a, b)

    # For higher dimensions, use matmul which supports broadcasting
    return ops.matmul(a, b)


def validate_bmm_shapes(
    shape_a: Tuple[int, ...],
    shape_b: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Validate shapes for batched matrix multiplication.

    Args:
        shape_a: Shape of first tensor.
        shape_b: Shape of second tensor.

    Returns:
        Tuple of expected output shape and intermediate K dimension.

    Raises:
        ValueError: If shapes are incompatible.
    """
    if len(shape_a) < 3 or len(shape_b) < 3:
        raise ValueError(f"bmm requires 3D+ tensors, got {shape_a} and {shape_b}")

    k_a = shape_a[-1]
    k_b = shape_b[-2]

    if k_a != k_b:
        raise ValueError(
            f"bmm K dimensions incompatible: {k_a} != {k_b}"
        )

    # Output shape: broadcast batch dims + (m, n)
    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]

    # Compute broadcast batch shape
    max_batch_len = max(len(batch_a), len(batch_b))
    broadcast_batch = []

    for i in range(max_batch_len):
        dim_a = batch_a[-(i + 1)] if i < len(batch_a) else 1
        dim_b = batch_b[-(i + 1)] if i < len(batch_b) else 1

        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            raise ValueError(
                f"bmm batch dimensions cannot be broadcast: {batch_a} and {batch_b}"
            )

        broadcast_batch.append(max(dim_a, dim_b))

    output_shape = tuple(reversed(broadcast_batch)) + (shape_a[-2], shape_b[-1])

    return output_shape, (k_a,)
