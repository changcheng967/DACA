"""bf16 → fp16 transparent shim for Ascend NPUs.

Ascend 910ProA does NOT support bfloat16. This module intercepts
Tensor.astype calls requesting bfloat16 and redirects them to float16.

WHY: bf16 causes `aclnnCastGetWorkspaceSize` crash on 910ProA hardware.
The shim ensures bf16 never reaches CANN by converting at Python level.

Example:
    from daca.runtime import enable_bf16_shim

    enable_bf16_shim()

    # Now bf16 automatically converts to fp16
    import mindspore as ms
    x = ms.Tensor([1.0, 2.0], ms.bfloat16)  # Actually creates fp16
"""

import logging
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger("daca.runtime.dtype")

# Shim state
_shim_enabled: bool = False
_interception_count: int = 0
_original_astype: Optional[Callable] = None


class BFloat16Shim:
    """Manager for bf16 → fp16 transparent conversion.

    This class manages the monkey-patch of mindspore.Tensor.astype
    to intercept bfloat16 requests and redirect to float16.

    Attributes:
        enabled: Whether the shim is currently active.
        interception_count: Number of bf16 conversions intercepted.

    Example:
        >>> shim = BFloat16Shim()
        >>> shim.enable()
        >>> # bf16 now converts to fp16
        >>> shim.disable()
        >>> # Original behavior restored
    """

    def __init__(self):
        """Initialize the shim manager."""
        self._enabled = False
        self._interception_count = 0
        self._original_astype: Optional[Callable] = None
        self._warned = False

    @property
    def enabled(self) -> bool:
        """Check if shim is enabled."""
        return self._enabled

    @property
    def interception_count(self) -> int:
        """Get number of intercepted bf16 conversions."""
        return self._interception_count

    def enable(self) -> None:
        """Enable the bf16 → fp16 shim.

        Monkey-patches mindspore.Tensor.astype to intercept bfloat16
        and redirect to float16.

        Raises:
            ImportError: If mindspore is not available.
        """
        global _shim_enabled, _interception_count, _original_astype

        if self._enabled:
            logger.debug("BFloat16 shim already enabled")
            return

        try:
            import mindspore as ms
            from mindspore import Tensor
            import mindspore.common.dtype as mstype
        except ImportError:
            logger.warning("MindSpore not available, bf16 shim cannot be enabled")
            return

        # Store original astype
        if self._original_astype is None:
            self._original_astype = Tensor.astype

        # Create wrapper function
        shim_self = self

        def patched_astype(self_tensor, dtype) -> Any:
            """Patched astype that intercepts bf16 → fp16."""
            nonlocal shim_self

            # Check if target dtype is bfloat16
            target_is_bf16 = False

            # Handle different ways to specify bfloat16
            if dtype == mstype.bfloat16:
                target_is_bf16 = True
            elif isinstance(dtype, str) and dtype.lower() in ("bfloat16", "bf16"):
                target_is_bf16 = True
            elif hasattr(dtype, "__name__") and dtype.__name__ == "bfloat16":
                target_is_bf16 = True

            if target_is_bf16:
                shim_self._interception_count += 1

                if not shim_self._warned:
                    logger.warning(
                        "bfloat16 is not supported on Ascend 910ProA. "
                        "DACA automatically converting to float16. "
                        "This message will not be shown again."
                    )
                    shim_self._warned = True

                # Redirect to float16
                dtype = mstype.float16

            # Call original astype with potentially modified dtype
            return shim_self._original_astype(self_tensor, dtype)

        # Apply the patch
        Tensor.astype = patched_astype

        self._enabled = True
        _shim_enabled = True
        _original_astype = self._original_astype

        logger.info("BFloat16 → Float16 shim enabled")

    def disable(self) -> None:
        """Disable the bf16 shim and restore original astype."""
        global _shim_enabled, _original_astype

        if not self._enabled:
            logger.debug("BFloat16 shim not enabled")
            return

        if self._original_astype is not None:
            try:
                from mindspore import Tensor
                Tensor.astype = self._original_astype
            except ImportError:
                pass

        self._enabled = False
        _shim_enabled = False

        logger.info("BFloat16 shim disabled, original astype restored")

    def reset_count(self) -> int:
        """Reset interception counter and return previous count.

        Returns:
            Previous interception count.
        """
        count = self._interception_count
        self._interception_count = 0
        return count


# Global shim instance
_shim = BFloat16Shim()


def enable_bf16_shim() -> None:
    """Enable the global bf16 → fp16 shim.

    This is the primary API for enabling bf16 conversion.
    Called automatically by daca.patch().

    Example:
        >>> from daca.runtime import enable_bf16_shim
        >>> enable_bf16_shim()
        >>> # Now x.astype(ms.bfloat16) creates fp16
    """
    _shim.enable()


def disable_bf16_shim() -> None:
    """Disable the global bf16 shim.

    Restores original Tensor.astype behavior.
    Called automatically by daca.unpatch().

    Example:
        >>> from daca.runtime import disable_bf16_shim
        >>> disable_bf16_shim()
    """
    _shim.disable()


def is_bf16_supported() -> bool:
    """Check if bfloat16 is supported on current hardware.

    For Ascend 910ProA, this always returns False.

    Returns:
        False - bf16 is not supported on 910ProA hardware.

    Example:
        >>> from daca.runtime import is_bf16_supported
        >>> if not is_bf16_supported():
        ...     print("bf16 not available, using fp16")
    """
    return False


def get_bf16_interception_count() -> int:
    """Get the number of bf16 → fp16 conversions intercepted.

    Returns:
        Number of times astype was called with bfloat16.

    Example:
        >>> from daca.runtime import get_bf16_interception_count
        >>> count = get_bf16_interception_count()
        >>> print(f"Converted {count} bf16 requests to fp16")
    """
    return _shim.interception_count


def is_shim_enabled() -> bool:
    """Check if the bf16 shim is currently enabled.

    Returns:
        True if shim is active, False otherwise.
    """
    return _shim.enabled
