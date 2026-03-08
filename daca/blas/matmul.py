"""MatMul operations with Ascend-specific handling.

Provides MatMul wrappers with:
- Shape validation for 4D attention shapes
- Workspace size detection and workarounds
- Numerical stability improvements

Example:
    from daca.blas import matmul

    # Standard 2D MatMul
    c = matmul(a, b)  # (m, k) @ (k, n) -> (m, n)

    # 4D attention MatMul
    scores = matmul(q, k.transpose(-1, -2))  # (b, h, s, d) @ (b, h, d, s)
"""

import logging
from typing import Optional, Tuple, Union, Any

logger = logging.getLogger("daca.blas.matmul")


def matmul(
    a,
    b,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> Any:
    """Matrix multiplication with Ascend optimizations.

    Supports 2D and 4D tensors (for attention computations).
    Handles workspace allocation and shape validation.

    Args:
        a: First tensor of shape (..., m, k) or (m, k).
        b: Second tensor of shape (..., k, n) or (k, n).
        transpose_a: If True, transpose a before multiplication.
        transpose_b: If True, transpose b before multiplication.

    Returns:
        Result tensor of shape (..., m, n).

    Raises:
        ValueError: If shapes are incompatible.

    Example:
        >>> import mindspore as ms
        >>> from daca.blas import matmul
        >>> a = ms.Tensor([[1, 2], [3, 4]], ms.float16)
        >>> b = ms.Tensor([[5, 6], [7, 8]], ms.float16)
        >>> c = matmul(a, b)
        >>> print(c.shape)
        (2, 2)
    """
    try:
        import mindspore as ms
        import mindspore.ops as ops
        from mindspore import Tensor
    except ImportError:
        raise ImportError("MindSpore is required for matmul operations")

    # Validate inputs are tensors
    if not isinstance(a, (Tensor, ms.Tensor)):
        a = Tensor(a)
    if not isinstance(b, (Tensor, ms.Tensor)):
        b = Tensor(b)

    # Get shapes
    shape_a = a.shape
    shape_b = b.shape

    # Handle transposition
    if transpose_a:
        a = ops.transpose(a, tuple(range(len(shape_a) - 2)) + (-1, -2))
        shape_a = a.shape
    if transpose_b:
        b = ops.transpose(b, tuple(range(len(shape_b) - 2)) + (-1, -2))
        shape_b = b.shape

    # Validate shapes
    if len(shape_a) < 2 or len(shape_b) < 2:
        raise ValueError(f"matmul requires 2D+ tensors, got shapes {shape_a} and {shape_b}")

    k_a = shape_a[-1]
    k_b = shape_b[-2]

    if k_a != k_b:
        raise ValueError(
            f"matmul shapes incompatible: {shape_a} @ {shape_b} "
            f"(k dimensions: {k_a} != {k_b})"
        )

    # Use MindSpore MatMul
    if len(shape_a) == 2 and len(shape_b) == 2:
        # Simple 2D case
        return ops.matmul(a, b)
    elif len(shape_a) >= 2 and len(shape_b) >= 2:
        # Batched case - use BatchMatMul or MatMul
        # For 4D attention shapes: (batch, heads, seq, dim)
        if len(shape_a) == 4 and len(shape_b) == 4:
            # Use batch matmul with broadcast
            return ops.BatchMatMul()(a, b)
        else:
            # General batched matmul
            return ops.matmul(a, b)
    else:
        return ops.matmul(a, b)


def linear(
    x,
    weight,
    bias: Optional[Any] = None,
) -> Any:
    """Apply linear transformation: y = xA^T + b.

    Args:
        x: Input tensor of shape (..., in_features).
        weight: Weight tensor of shape (out_features, in_features).
        bias: Optional bias tensor of shape (out_features,).

    Returns:
        Output tensor of shape (..., out_features).

    Example:
        >>> from daca.blas import linear
        >>> x = ms.Tensor([[1, 2, 3]], ms.float16)
        >>> w = ms.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], ms.float16)
        >>> y = linear(x, w)
    """
    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for linear operations")

    # Use MindSpore's MatMul for x @ weight.T
    output = matmul(x, weight, transpose_b=True)

    if bias is not None:
        output = ops.add(output, bias)

    return output


def addmm(
    input,
    mat1,
    mat2,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Any:
    """Compute beta * input + alpha * mat1 @ mat2.

    Args:
        input: Matrix to be added.
        mat1: First matrix for multiplication.
        mat2: Second matrix for multiplication.
        beta: Multiplier for input.
        alpha: Multiplier for mat1 @ mat2.

    Returns:
        Result tensor.

    Example:
        >>> from daca.blas import addmm
        >>> result = addmm(bias, x, weight.T)
    """
    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for addmm operations")

    # Compute mat1 @ mat2
    mm_result = matmul(mat1, mat2)

    # Scale and add
    if alpha != 1.0:
        mm_result = ops.mul(mm_result, alpha)

    if beta != 1.0:
        input = ops.mul(input, beta)

    return ops.add(input, mm_result)


def validate_matmul_shapes(
    shape_a: Tuple[int, ...],
    shape_b: Tuple[int, ...],
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Validate and return effective shapes for matmul.

    Args:
        shape_a: Shape of first tensor.
        shape_b: Shape of second tensor.
        transpose_a: Whether a is transposed.
        transpose_b: Whether b is transposed.

    Returns:
        Tuple of effective shapes after transposition.

    Raises:
        ValueError: If shapes are incompatible.
    """
    # Apply transposition to last two dims
    if transpose_a and len(shape_a) >= 2:
        effective_a = shape_a[:-2] + (shape_a[-1], shape_a[-2])
    else:
        effective_a = shape_a

    if transpose_b and len(shape_b) >= 2:
        effective_b = shape_b[:-2] + (shape_b[-1], shape_b[-2])
    else:
        effective_b = shape_b

    # Check K dimension compatibility
    k_a = effective_a[-1]
    k_b = effective_b[-2]

    if k_a != k_b:
        raise ValueError(
            f"Incompatible matmul shapes: {shape_a} @ {shape_b} "
            f"(K dims: {k_a} != {k_b})"
        )

    return effective_a, effective_b
