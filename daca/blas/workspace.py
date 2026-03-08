"""Workspace management for Ascend NPU operations.

Some CANN operations require workspace memory allocation. This module
provides utilities for managing workspace allocation safely.

Example:
    from daca.blas import WorkspaceManager, preallocate_workspace

    # Pre-allocate workspace for large operations
    preallocate_workspace(size=1024 * 1024 * 256)  # 256MB
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("daca.blas.workspace")

# Global workspace pool
_workspace_pool: Dict[str, Any] = {}
_preallocated_workspace: Optional[Any] = None
_preallocated_size: int = 0


class WorkspaceManager:
    """Manager for operation workspace allocation.

    Some Ascend operations require workspace memory. This class
    provides pre-allocation and pooling to reduce allocation overhead.

    Attributes:
        default_size: Default workspace size in bytes.
        pool: Pool of pre-allocated workspaces.

    Example:
        >>> manager = WorkspaceManager(default_size=256 * 1024 * 1024)
        >>> with manager.get_workspace(size_hint=128 * 1024 * 1024) as ws:
        ...     # Use workspace for operation
        ...     pass
    """

    def __init__(self, default_size: int = 256 * 1024 * 1024):
        """Initialize workspace manager.

        Args:
            default_size: Default workspace size in bytes (default: 256MB).
        """
        self.default_size = default_size
        self._pool: Dict[str, Any] = {}
        self._allocated: List[Any] = []

    def allocate(self, size: Optional[int] = None, name: Optional[str] = None) -> Any:
        """Allocate workspace tensor.

        Args:
            size: Size in bytes. Uses default_size if None.
            name: Optional name for pooling.

        Returns:
            Workspace tensor.

        Example:
            >>> ws = manager.allocate(128 * 1024 * 1024, name="attention")
        """
        try:
            import mindspore as ms
            from mindspore import Tensor, context
            import mindspore.common.dtype as mstype
        except ImportError:
            raise ImportError("MindSpore is required for workspace allocation")

        size = size or self.default_size

        # Check pool first
        if name and name in self._pool:
            return self._pool[name]

        # Calculate tensor shape (in float32 elements)
        num_elements = size // 4  # 4 bytes per float32
        shape = (num_elements,)

        try:
            workspace = Tensor(
                ms.numpy.zeros(shape, dtype=mstype.float32),
                dtype=mstype.float32
            )
            logger.debug(f"Allocated workspace of {size / (1024**2):.2f} MB")

            if name:
                self._pool[name] = workspace

            return workspace

        except Exception as e:
            logger.warning(f"Failed to allocate workspace: {e}")
            # Return None - operation should handle this
            return None

    def get(self, name: str, size: Optional[int] = None) -> Optional[Any]:
        """Get workspace from pool or allocate new.

        Args:
            name: Workspace name.
            size: Size if needs allocation.

        Returns:
            Workspace tensor or None if unavailable.
        """
        if name in self._pool:
            return self._pool[name]
        return self.allocate(size, name)

    def release(self, name: str) -> bool:
        """Release workspace from pool.

        Args:
            name: Workspace name.

        Returns:
            True if workspace was released, False if not found.
        """
        if name in self._pool:
            del self._pool[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all pooled workspaces."""
        self._pool.clear()
        logger.debug("Workspace pool cleared")

    def __enter__(self) -> "WorkspaceManager":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and clear pool."""
        self.clear()


def preallocate_workspace(size: int = 256 * 1024 * 1024) -> Optional[Any]:
    """Pre-allocate global workspace for large operations.

    Pre-allocating workspace can improve performance by reducing
    allocation overhead during compute-intensive operations.

    Args:
        size: Workspace size in bytes (default: 256MB).

    Returns:
        Pre-allocated workspace tensor, or None if allocation failed.

    Example:
        >>> from daca.blas import preallocate_workspace
        >>> ws = preallocate_workspace(512 * 1024 * 1024)  # 512MB
    """
    global _preallocated_workspace, _preallocated_size

    try:
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
    except ImportError:
        logger.warning("MindSpore not available for workspace allocation")
        return None

    # Calculate shape for float32
    num_elements = size // 4
    shape = (num_elements,)

    try:
        _preallocated_workspace = Tensor(
            ms.numpy.zeros(shape, dtype=mstype.float32),
            dtype=mstype.float32
        )
        _preallocated_size = size

        logger.info(f"Pre-allocated workspace: {size / (1024**2):.2f} MB")
        return _preallocated_workspace

    except Exception as e:
        logger.warning(f"Failed to pre-allocate workspace: {e}")
        return None


def get_workspace(size_hint: Optional[int] = None) -> Optional[Any]:
    """Get pre-allocated workspace or allocate new.

    Args:
        size_hint: Minimum size required in bytes.

    Returns:
        Workspace tensor, or None if unavailable.

    Example:
        >>> from daca.blas import get_workspace
        >>> ws = get_workspace(128 * 1024 * 1024)
    """
    global _preallocated_workspace, _preallocated_size

    # Check if pre-allocated workspace is sufficient
    if _preallocated_workspace is not None:
        if size_hint is None or size_hint <= _preallocated_size:
            return _preallocated_workspace

    # Need to allocate new workspace
    size = size_hint or (256 * 1024 * 1024)

    try:
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype

        num_elements = size // 4
        shape = (num_elements,)

        return Tensor(
            ms.numpy.zeros(shape, dtype=mstype.float32),
            dtype=mstype.float32
        )
    except Exception as e:
        logger.debug(f"Failed to allocate workspace: {e}")
        return None


def clear_workspace_pool() -> None:
    """Clear all pooled workspaces.

    Example:
        >>> from daca.blas import clear_workspace_pool
        >>> clear_workspace_pool()
    """
    global _preallocated_workspace, _preallocated_size, _workspace_pool

    _workspace_pool.clear()
    _preallocated_workspace = None
    _preallocated_size = 0

    logger.debug("All workspaces cleared")


def estimate_workspace_size(
    m: int,
    k: int,
    n: int,
    dtype_size: int = 2,  # fp16 = 2 bytes
) -> int:
    """Estimate workspace size for MatMul operation.

    Args:
        m: First matrix rows.
        k: Inner dimension.
        n: Second matrix columns.
        dtype_size: Size of data type in bytes.

    Returns:
        Estimated workspace size in bytes.

    Example:
        >>> size = estimate_workspace_size(1024, 1024, 1024)
        >>> print(f"Need {size / (1024**2):.2f} MB workspace")
    """
    # Conservative estimate: 3x output size for temp buffers
    output_size = m * n * dtype_size
    return output_size * 3
