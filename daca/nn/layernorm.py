"""LayerNorm with fp32 upcast for Ascend NPUs.

CANN has a bug where fp16 LayerNorm incorrectly routes through
FlashAttentionScore kernel, causing rank mismatch crashes.
This module provides LayerNorm that upcasts to fp32 internally.

WHY: fp16 LayerNorm crashes with "FlashAttentionScore rank error".
The workaround is to upcast to fp32, normalize, then cast back.

Example:
    from daca.nn import LayerNorm

    ln = LayerNorm(hidden_size=768)
    normalized = ln(hidden_states)  # fp32 upcast internally
"""

import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger("daca.nn.layernorm")

# Track fp32 upcast state
_fp32_upcast_enabled: bool = False
_original_layer_norm: Optional[Any] = None


class LayerNorm:
    """LayerNorm with automatic fp32 upcast.

    Normalizes across the last dimension, with automatic upcasting
    to fp32 to avoid CANN fusion bugs on Ascend.

    Attributes:
        normalized_shape: Shape of normalized dimensions.
        epsilon: Small constant for numerical stability.
        elementwise_affine: Whether to use learnable affine parameters.

    Example:
        >>> ln = LayerNorm(768, epsilon=1e-6)
        >>> x = ms.Tensor((batch, seq, 768), ms.float16)
        >>> y = ln(x)  # Internally uses fp32
    """

    def __init__(
        self,
        normalized_shape: int,
        epsilon: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: Any = None,
    ):
        """Initialize LayerNorm.

        Args:
            normalized_shape: Size of the last dimension to normalize.
            epsilon: Small constant for numerical stability.
            elementwise_affine: Whether to use learnable gamma/beta.
            dtype: Data type for parameters.
        """
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine

        # Initialize parameters
        try:
            import mindspore as ms
            from mindspore import Parameter, Tensor
            import mindspore.common.dtype as mstype
            import mindspore.ops as ops

            param_dtype = dtype if dtype is not None else mstype.float32

            if elementwise_affine:
                # Gamma (weight): ones
                self.weight = Parameter(
                    Tensor(ops.ones((normalized_shape,), param_dtype), param_dtype),
                    name="weight"
                )
                # Beta (bias): zeros
                self.bias = Parameter(
                    Tensor(ops.zeros((normalized_shape,), param_dtype), param_dtype),
                    name="bias"
                )
            else:
                self.weight = None
                self.bias = None

        except ImportError:
            self.weight = None
            self.bias = None

    def construct(self, hidden_states: Any) -> Any:
        """Apply LayerNorm with fp32 upcast.

        CANN bug workaround: Upcast to fp32 before normalization,
        then cast back to original dtype.

        Args:
            hidden_states: Input tensor (..., normalized_shape).

        Returns:
            Normalized tensor with same shape and dtype as input.
        """
        try:
            import mindspore.ops as ops
            import mindspore.common.dtype as mstype
        except ImportError:
            raise ImportError("MindSpore is required for LayerNorm")

        original_dtype = hidden_states.dtype

        # Upcast to fp32 for normalization (avoids CANN bug)
        if original_dtype == mstype.float16:
            hidden_states = ops.cast(hidden_states, mstype.float32)

        # Compute mean and variance
        mean = ops.mean(hidden_states, axis=-1, keep_dims=True)
        variance = ops.mean(
            ops.pow(hidden_states - mean, 2),
            axis=-1,
            keep_dims=True
        )

        # Normalize
        hidden_states = (hidden_states - mean) / ops.sqrt(variance + self.epsilon)

        # Apply affine transformation
        if self.elementwise_affine and self.weight is not None:
            # Cast parameters to match hidden_states
            weight = ops.cast(self.weight, hidden_states.dtype)
            bias = ops.cast(self.bias, hidden_states.dtype)
            hidden_states = hidden_states * weight + bias

        # Cast back to original dtype
        if original_dtype == mstype.float16:
            hidden_states = ops.cast(hidden_states, original_dtype)

        return hidden_states

    def __call__(self, hidden_states: Any) -> Any:
        """Call construct method."""
        return self.construct(hidden_states)


def enable_fp32_upcast() -> None:
    """Enable fp32 upcast for all LayerNorm operations.

    This patches MindSpore's LayerNorm to automatically upcast
    fp16 inputs to fp32 before normalization.

    Example:
        >>> from daca.nn import enable_fp32_upcast
        >>> enable_fp32_upcast()
    """
    global _fp32_upcast_enabled, _original_layer_norm

    if _fp32_upcast_enabled:
        logger.debug("LayerNorm fp32 upcast already enabled")
        return

    try:
        import mindspore.nn as nn

        # Store original LayerNorm
        _original_layer_norm = nn.LayerNorm

        # Create wrapper that uses fp32 upcast
        class LayerNormWrapper(nn.LayerNorm):
            """LayerNorm wrapper with fp32 upcast."""

            def construct(self, x):
                import mindspore.ops as ops
                import mindspore.common.dtype as mstype

                original_dtype = x.dtype

                # Upcast to fp32 if needed
                if original_dtype == mstype.float16:
                    x = ops.cast(x, mstype.float32)

                # Call original construct
                result = super().construct(x)

                # Cast back
                if original_dtype == mstype.float16:
                    result = ops.cast(result, original_dtype)

                return result

        # Apply patch
        nn.LayerNorm = LayerNormWrapper

        _fp32_upcast_enabled = True
        logger.info("LayerNorm fp32 upcast enabled")

    except ImportError:
        logger.warning("MindSpore not available, LayerNorm patch skipped")
    except Exception as e:
        logger.warning(f"Failed to enable LayerNorm fp32 upcast: {e}")


def disable_fp32_upcast() -> None:
    """Disable fp32 upcast and restore original LayerNorm.

    Example:
        >>> from daca.nn import disable_fp32_upcast
        >>> disable_fp32_upcast()
    """
    global _fp32_upcast_enabled, _original_layer_norm

    if not _fp32_upcast_enabled:
        logger.debug("LayerNorm fp32 upcast not enabled")
        return

    try:
        import mindspore.nn as nn

        if _original_layer_norm is not None:
            nn.LayerNorm = _original_layer_norm
            _original_layer_norm = None

        _fp32_upcast_enabled = False
        logger.info("LayerNorm fp32 upcast disabled")

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to disable LayerNorm fp32 upcast: {e}")
