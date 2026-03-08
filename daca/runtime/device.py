"""Device management for Ascend NPUs.

Provides device selection and management utilities similar to torch.cuda
but for Ascend NPUs.

Example:
    from daca.runtime import set_device, device_count

    print(f"Available NPUs: {device_count()}")
    set_device(0)  # Use first NPU
"""

import logging
from typing import Optional, ContextManager
from contextlib import contextmanager

logger = logging.getLogger("daca.runtime.device")

# Track current device
_current_device: int = 0


def device_count() -> int:
    """Get the number of available Ascend NPUs.

    Returns:
        Number of NPUs available, or 0 if none detected.

    Example:
        >>> from daca.runtime import device_count
        >>> count = device_count()
        >>> print(f"Found {count} NPUs")
    """
    # Check environment first (for distributed setups)
    import os

    rank_size = os.environ.get("RANK_SIZE")
    if rank_size:
        try:
            return int(rank_size)
        except ValueError:
            pass

    # Try MindSpore
    try:
        import mindspore as ms
        from mindspore import context

        # Get device info from MindSpore
        # Note: MindSpore doesn't have a direct device_count API
        # Use get_context or environment
        device_info = context.get_context("device_id")
        if device_info is not None:
            # At least one device available
            # Check for multiple via RANK_SIZE or default to 1
            return int(os.environ.get("RANK_SIZE", "1"))
    except Exception:
        pass

    # Fallback to detection
    from daca.runtime.detect import get_npu_info
    try:
        info = get_npu_info()
        return info.get("count", 0)
    except RuntimeError:
        return 0


def current_device() -> int:
    """Get the current device index.

    Returns:
        Index of the currently active device (0-indexed).

    Example:
        >>> from daca.runtime import current_device
        >>> print(f"Using device {current_device()}")
    """
    global _current_device

    # Try MindSpore first
    try:
        import mindspore as ms
        from mindspore import context
        device_id = context.get_context("device_id")
        if device_id is not None:
            return device_id
    except Exception:
        pass

    return _current_device


def set_device(device: int) -> None:
    """Set the active Ascend NPU device.

    Args:
        device: Device index (0-indexed).

    Raises:
        ValueError: If device index is invalid.

    Example:
        >>> from daca.runtime import set_device
        >>> set_device(0)  # Use first NPU
        >>> set_device(1)  # Use second NPU
    """
    global _current_device

    count = device_count()
    if device < 0 or (count > 0 and device >= count):
        raise ValueError(f"Invalid device index {device}. Available devices: 0-{max(0, count-1)}")

    _current_device = device

    # Set in MindSpore
    try:
        import mindspore as ms
        from mindspore import context
        context.set_context(device_id=device)
        logger.debug(f"Set MindSpore device to {device}")
    except Exception as e:
        logger.debug(f"Could not set MindSpore device: {e}")

    # Set environment for other tools
    import os
    os.environ["ASCEND_DEVICE_ID"] = str(device)


def get_device(device: Optional[int] = None) -> int:
    """Get device index, defaulting to current device if not specified.

    Args:
        device: Optional device index. If None, returns current device.

    Returns:
        Device index.

    Example:
        >>> from daca.runtime import get_device
        >>> get_device()  # Current device
        0
        >>> get_device(2)  # Specified device
        2
    """
    if device is None:
        return current_device()
    return device


def synchronize(device: Optional[int] = None) -> None:
    """Synchronize CPU with NPU.

    Blocks until all operations on the device complete.
    This is useful for accurate timing measurements.

    Args:
        device: Device to synchronize. If None, uses current device.

    Example:
        >>> from daca.runtime import synchronize
        >>> synchronize()  # Wait for all operations to complete
    """
    device = get_device(device)

    try:
        import mindspore as ms
        from mindspore import context, Tensor
        import mindspore.ops as ops

        # MindSpore doesn't have explicit sync, but we can force a small op
        original_device = context.get_context("device_id")
        context.set_context(device_id=device)

        # Force synchronization via a small operation
        x = Tensor([0.0])
        _ = ops.Add()(x, x)

        if original_device is not None:
            context.set_context(device_id=original_device)

        logger.debug(f"Synchronized device {device}")
    except Exception as e:
        logger.debug(f"Synchronization warning: {e}")


class Device:
    """Context manager for device selection.

    Allows temporarily switching to a different device within a context.

    Attributes:
        device: Device index.

    Example:
        >>> from daca.runtime import Device
        >>> with Device(0):
        ...     # Operations here use device 0
        ...     pass
        >>> # Original device restored
    """

    def __init__(self, device: int):
        """Initialize device context manager.

        Args:
            device: Device index to use within context.
        """
        self.device = device
        self._prev_device: Optional[int] = None

    def __enter__(self) -> "Device":
        """Enter device context."""
        self._prev_device = current_device()
        set_device(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit device context and restore previous device."""
        if self._prev_device is not None:
            set_device(self._prev_device)

    def __repr__(self) -> str:
        return f"Device({self.device})"


@contextmanager
def device_context(device: int):
    """Context manager for device selection (function form).

    Args:
        device: Device index to use within context.

    Yields:
        Device index.

    Example:
        >>> from daca.runtime import device_context
        >>> with device_context(1):
        ...     # Operations here use device 1
        ...     pass
    """
    prev_device = current_device()
    try:
        set_device(device)
        yield device
    finally:
        set_device(prev_device)
