"""RMSNorm implementation for Ascend NPUs.

Manual decomposition using rsqrt + mul + mul to avoid potential
issues with native RMSNorm operators.

WHY: Manual implementation avoids dependencies on potentially
broken ops and gives us full control over the computation.

Example:
    from daca.nn import RMSNorm

    rms = RMSNorm(hidden_size=768, epsilon=1e-6)
    normalized = rms(hidden_states)
"""

import logging
from typing import Any

logger = logging.getLogger("daca.nn.rmsnorm")


class RMSNorm:
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes by the root mean square without centering
    (no mean subtraction). More efficient than LayerNorm.

    Attributes:
        hidden_size: Size of the last dimension.
        epsilon: Small constant for numerical stability.
        weight: Learnable scale parameter.

    Reference:
        Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
        https://arxiv.org/abs/1910.07467

    Example:
        >>> rms = RMSNorm(768, epsilon=1e-6)
        >>> x = ms.Tensor((batch, seq, 768), ms.float16)
        >>> y = rms(x)
    """

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        dtype: Any = None,
    ):
        """Initialize RMSNorm.

        Args:
            hidden_size: Size of the dimension to normalize.
            epsilon: Small constant for numerical stability.
            dtype: Data type for weight parameter.
        """
        self.hidden_size = hidden_size
        self.epsilon = epsilon

        # Initialize weight parameter
        try:
            import mindspore as ms
            from mindspore import Parameter, Tensor
            import mindspore.common.dtype as mstype
            import mindspore.ops as ops

            param_dtype = dtype if dtype is not None else mstype.float32

            # Weight: ones
            self.weight = Parameter(
                Tensor(ops.ones((hidden_size,), param_dtype), param_dtype),
                name="weight"
            )

        except ImportError:
            self.weight = None

    def construct(self, hidden_states: Any) -> Any:
        """Apply RMSNorm.

        Args:
            hidden_states: Input tensor (..., hidden_size).

        Returns:
            Normalized tensor with same shape as input.
        """
        try:
            import mindspore.ops as ops
            import mindspore.common.dtype as mstype
        except ImportError:
            raise ImportError("MindSpore is required for RMSNorm")

        original_dtype = hidden_states.dtype

        # Compute variance (mean of squared values)
        variance = ops.mean(
            ops.pow(hidden_states, 2),
            axis=-1,
            keep_dims=True
        )

        # Normalize: x / sqrt(variance + epsilon)
        hidden_states = hidden_states * ops.rsqrt(variance + self.epsilon)

        # Apply weight
        if self.weight is not None:
            weight = ops.cast(self.weight, hidden_states.dtype)
            hidden_states = hidden_states * weight

        return hidden_states

    def __call__(self, hidden_states: Any) -> Any:
        """Call construct method."""
        return self.construct(hidden_states)


def rms_norm(hidden_states: Any, weight: Any, epsilon: float = 1e-6) -> Any:
    """Functional RMSNorm.

    Args:
        hidden_states: Input tensor.
        weight: Scale parameter.
        epsilon: Small constant for numerical stability.

    Returns:
        Normalized tensor.

    Example:
        >>> from daca.nn.rmsnorm import rms_norm
        >>> normalized = rms_norm(x, weight)
    """
    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for rms_norm")

    # Compute variance
    variance = ops.mean(ops.pow(hidden_states, 2), axis=-1, keep_dims=True)

    # Normalize
    hidden_states = hidden_states * ops.rsqrt(variance + epsilon)

    # Scale
    hidden_states = hidden_states * weight

    return hidden_states
