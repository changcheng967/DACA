"""Missing activations for Ascend NPUs.

MindSpore 2.7.1 is missing ops.SiLU and ops.SwiGLU. This module
provides manual implementations and namespace injection.

WHY: ops.SiLU returns "module 'mindspore.ops' has no attribute 'SiLU'".
Manual implementation: silu(x) = x * sigmoid(x).

Example:
    from daca.nn import silu, swiglu

    y = silu(x)  # x * sigmoid(x)
    z = swiglu(x)  # split + silu + mul
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("daca.nn.activations")

# Track injection state
_silu_injected: bool = False
_swiglu_injected: bool = False
_original_ops_silu: Optional[Any] = None
_original_ops_swiglu: Optional[Any] = None


def silu(x: Any) -> Any:
    """SiLU (Sigmoid Linear Unit) activation.

    Also known as Swish. Manual implementation: x * sigmoid(x).

    Args:
        x: Input tensor.

    Returns:
        SiLU activation: x * sigmoid(x).

    Note:
        MindSpore 2.7.1 does not have ops.SiLU, so we implement manually.

    Example:
        >>> from daca.nn import silu
        >>> y = silu(x)
    """
    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for silu")

    return x * ops.sigmoid(x)


def swiglu(x: Any, dim: int = -1) -> Any:
    """SwiGLU activation for feed-forward networks.

    SwiGLU splits input, applies SiLU to first half, then multiplies.
    Commonly used in LLaMA-style models.

    Args:
        x: Input tensor.
        dim: Dimension to split along (default: -1).

    Returns:
        SwiGLU activation: silu(a) * b where x = concat(a, b).

    Example:
        >>> from daca.nn import swiglu
        >>> # For hidden_size=4096, intermediate_size=11008
        >>> # x has shape (batch, seq, 11008)
        >>> y = swiglu(x)  # Output shape: (batch, seq, 5504)
    """
    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for swiglu")

    # Split into two halves
    half_size = x.shape[dim] // 2
    a, b = ops.split(x, (half_size, half_size), axis=dim)

    # Apply SiLU to first half, multiply by second half
    return silu(a) * b


def geglu(x: Any, dim: int = -1) -> Any:
    """GeGLU activation for feed-forward networks.

    GeGLU splits input, applies GeLU to first half, then multiplies.

    Args:
        x: Input tensor.
        dim: Dimension to split along.

    Returns:
        GeGLU activation: gelu(a) * b.
    """
    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for geglu")

    half_size = x.shape[dim] // 2
    a, b = ops.split(x, (half_size, half_size), axis=dim)

    return ops.gelu(a) * b


# Re-export available MindSpore activations
def gelu(x: Any) -> Any:
    """GeLU activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.gelu(x)
    except ImportError:
        raise ImportError("MindSpore is required for gelu")


def relu(x: Any) -> Any:
    """ReLU activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.relu(x)
    except ImportError:
        raise ImportError("MindSpore is required for relu")


def sigmoid(x: Any) -> Any:
    """Sigmoid activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.sigmoid(x)
    except ImportError:
        raise ImportError("MindSpore is required for sigmoid")


def tanh(x: Any) -> Any:
    """Tanh activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.tanh(x)
    except ImportError:
        raise ImportError("MindSpore is required for tanh")


def mish(x: Any) -> Any:
    """Mish activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.mish(x)
    except ImportError:
        raise ImportError("MindSpore is required for mish")


def fast_gelu(x: Any) -> Any:
    """FastGeLU activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.fast_gelu(x)
    except ImportError:
        raise ImportError("MindSpore is required for fast_gelu")


def hswish(x: Any) -> Any:
    """HSwish activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.hswish(x)
    except ImportError:
        raise ImportError("MindSpore is required for hswish")


def hsigmoid(x: Any) -> Any:
    """HSigmoid activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.hsigmoid(x)
    except ImportError:
        raise ImportError("MindSpore is required for hsigmoid")


def prelu(x: Any, weight: Any) -> Any:
    """PReLU activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.prelu(x, weight)
    except ImportError:
        raise ImportError("MindSpore is required for prelu")


def selu(x: Any) -> Any:
    """SeLU activation (re-export from MindSpore)."""
    try:
        import mindspore.ops as ops
        return ops.selu(x)
    except ImportError:
        raise ImportError("MindSpore is required for selu")


def inject_silu() -> None:
    """Inject SiLU into mindspore.ops namespace.

    Adds ops.silu and ops.SiLU to the namespace if they don't exist.

    Example:
        >>> from daca.nn import inject_silu
        >>> inject_silu()
        >>> import mindspore.ops as ops
        >>> y = ops.silu(x)  # Now works!
    """
    global _silu_injected, _original_ops_silu

    if _silu_injected:
        logger.debug("SiLU already injected")
        return

    try:
        import mindspore.ops as ops

        # Store original if exists
        if hasattr(ops, 'silu'):
            _original_ops_silu = ops.silu

        # Add silu function
        ops.silu = silu

        # Add SiLU class (for API compatibility)
        class SiLU:
            """SiLU activation class."""
            def __call__(self, x):
                return silu(x)

        ops.SiLU = SiLU

        _silu_injected = True
        logger.info("Injected SiLU into mindspore.ops namespace")

    except ImportError:
        logger.warning("MindSpore not available, SiLU injection skipped")
    except Exception as e:
        logger.warning(f"Failed to inject SiLU: {e}")


def inject_swiglu() -> None:
    """Inject SwiGLU into mindspore.ops namespace.

    Example:
        >>> from daca.nn import inject_swiglu
        >>> inject_swiglu()
        >>> import mindspore.ops as ops
        >>> y = ops.swiglu(x)
    """
    global _swiglu_injected, _original_ops_swiglu

    if _swiglu_injected:
        logger.debug("SwiGLU already injected")
        return

    try:
        import mindspore.ops as ops

        # Store original if exists
        if hasattr(ops, 'swiglu'):
            _original_ops_swiglu = ops.swiglu

        # Add swiglu function
        ops.swiglu = swiglu

        # Add SwiGLU class
        class SwiGLU:
            """SwiGLU activation class."""
            def __call__(self, x, dim=-1):
                return swiglu(x, dim=dim)

        ops.SwiGLU = SwiGLU

        _swiglu_injected = True
        logger.info("Injected SwiGLU into mindspore.ops namespace")

    except ImportError:
        logger.warning("MindSpore not available, SwiGLU injection skipped")
    except Exception as e:
        logger.warning(f"Failed to inject SwiGLU: {e}")


def remove_silu() -> None:
    """Remove injected SiLU from mindspore.ops namespace.

    Example:
        >>> from daca.nn import remove_silu
        >>> remove_silu()
    """
    global _silu_injected, _original_ops_silu

    if not _silu_injected:
        logger.debug("SiLU not injected")
        return

    try:
        import mindspore.ops as ops

        if _original_ops_silu is not None:
            ops.silu = _original_ops_silu
        elif hasattr(ops, 'silu'):
            delattr(ops, 'silu')

        if hasattr(ops, 'SiLU'):
            delattr(ops, 'SiLU')

        _silu_injected = False
        _original_ops_silu = None
        logger.info("Removed SiLU from mindspore.ops namespace")

    except Exception as e:
        logger.warning(f"Failed to remove SiLU: {e}")


def remove_swiglu() -> None:
    """Remove injected SwiGLU from mindspore.ops namespace.

    Example:
        >>> from daca.nn import remove_swiglu
        >>> remove_swiglu()
    """
    global _swiglu_injected, _original_ops_swiglu

    if not _swiglu_injected:
        logger.debug("SwiGLU not injected")
        return

    try:
        import mindspore.ops as ops

        if _original_ops_swiglu is not None:
            ops.swiglu = _original_ops_swiglu
        elif hasattr(ops, 'swiglu'):
            delattr(ops, 'swiglu')

        if hasattr(ops, 'SwiGLU'):
            delattr(ops, 'SwiGLU')

        _swiglu_injected = False
        _original_ops_swiglu = None
        logger.info("Removed SwiGLU from mindspore.ops namespace")

    except Exception as e:
        logger.warning(f"Failed to remove SwiGLU: {e}")
