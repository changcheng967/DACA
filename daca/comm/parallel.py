"""Multi-NPU communication using HCCL.

Provides collective communication operations for distributed training
on multiple Ascend NPUs.

Example:
    from daca.comm import initialize_parallel, all_reduce, get_rank

    initialize_parallel()
    rank = get_rank()
    result = all_reduce(tensor)
"""

import os
import logging
from typing import Optional, List, Any

logger = logging.getLogger("daca.comm.parallel")

# Parallel state
_initialized: bool = False
_rank: int = 0
_world_size: int = 1
_backend: Optional[str] = None


def initialize_parallel(
    backend: str = "hccl",
    init_method: Optional[str] = None,
) -> None:
    """Initialize parallel communication.

    Sets up HCCL for multi-NPU communication.

    Args:
        backend: Communication backend (default: "hccl").
        init_method: Optional initialization method URL.

    Example:
        >>> from daca.comm import initialize_parallel
        >>> initialize_parallel()
    """
    global _initialized, _rank, _world_size, _backend

    if _initialized:
        logger.warning("Parallel already initialized")
        return

    _backend = backend

    # Get rank and world size from environment (set by launcher)
    _rank = int(os.environ.get("RANK_ID", os.environ.get("RANK", "0")))
    _world_size = int(os.environ.get("RANK_SIZE", os.environ.get("WORLD_SIZE", "1")))

    try:
        import mindspore as ms
        from mindspore import context
        from mindspore.communication import init, get_rank, get_group_size

        # Initialize HCCL
        if backend == "hccl":
            init()

            # Get rank/world_size from MindSpore
            try:
                _rank = get_rank()
                _world_size = get_group_size()
            except Exception:
                pass  # Use environment values

        _initialized = True
        logger.info(f"Initialized parallel: rank={_rank}, world_size={_world_size}")

    except ImportError:
        logger.warning("MindSpore communication not available")
        _initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize parallel: {e}")
        _initialized = True


def is_initialized() -> bool:
    """Check if parallel communication is initialized.

    Returns:
        True if initialized, False otherwise.
    """
    return _initialized


def get_rank() -> int:
    """Get current process rank.

    Returns:
        Rank index (0-indexed).
    """
    global _rank

    if not _initialized:
        return int(os.environ.get("RANK_ID", os.environ.get("RANK", "0")))

    try:
        from mindspore.communication import get_rank as ms_get_rank
        return ms_get_rank()
    except Exception:
        return _rank


def get_world_size() -> int:
    """Get total number of processes.

    Returns:
        World size (number of NPUs).
    """
    global _world_size

    if not _initialized:
        return int(os.environ.get("RANK_SIZE", os.environ.get("WORLD_SIZE", "1")))

    try:
        from mindspore.communication import get_group_size
        return get_group_size()
    except Exception:
        return _world_size


def all_reduce(tensor: Any, op: str = "sum") -> Any:
    """All-reduce operation.

    Reduces tensor across all processes and broadcasts result.

    Args:
        tensor: Input tensor.
        op: Reduction operation ("sum", "avg", "max", "min").

    Returns:
        Reduced tensor.

    Example:
        >>> from daca.comm import all_reduce
        >>> reduced = all_reduce(tensor, op="sum")
    """
    try:
        import mindspore.ops as ops
        from mindspore.communication import AllReduce, ReduceOp
    except ImportError:
        logger.warning("MindSpore communication not available, returning input")
        return tensor

    # Map op string to ReduceOp
    op_map = {
        "sum": ReduceOp.SUM,
        "avg": ReduceOp.SUM,  # Divide after
        "max": ReduceOp.MAX,
        "min": ReduceOp.MIN,
    }

    reduce_op = op_map.get(op.lower(), ReduceOp.SUM)

    # Perform all-reduce
    all_reduce_op = AllReduce()
    result = all_reduce_op(tensor)

    # Handle average
    if op.lower() == "avg":
        result = result / get_world_size()

    return result


def all_gather(tensor: Any, dim: int = 0) -> Any:
    """All-gather operation.

    Gathers tensors from all processes and concatenates.

    Args:
        tensor: Input tensor.
        dim: Dimension to concatenate along.

    Returns:
        Gathered tensor with shape expanded along dim.

    Example:
        >>> from daca.comm import all_gather
        >>> gathered = all_gather(tensor, dim=0)
    """
    try:
        import mindspore.ops as ops
        from mindspore.communication import AllGather
    except ImportError:
        logger.warning("MindSpore communication not available, returning input")
        return tensor

    # Perform all-gather
    all_gather_op = AllGather()
    result = all_gather_op(tensor)

    return result


def broadcast(tensor: Any, src: int = 0) -> Any:
    """Broadcast tensor from source rank.

    Args:
        tensor: Tensor to broadcast (only used on src).
        src: Source rank to broadcast from.

    Returns:
        Broadcast tensor.

    Example:
        >>> from daca.comm import broadcast
        >>> data = broadcast(tensor, src=0)
    """
    try:
        import mindspore.ops as ops
        from mindspore.communication import Broadcast
    except ImportError:
        logger.warning("MindSpore communication not available, returning input")
        return tensor

    # Perform broadcast
    broadcast_op = Broadcast(src)
    result = broadcast_op(tensor)

    return result


def reduce_scatter(tensor: Any, dim: int = 0) -> Any:
    """Reduce-scatter operation.

    Reduces tensor and scatters result across processes.

    Args:
        tensor: Input tensor.
        dim: Dimension to scatter along.

    Returns:
        Scattered chunk of reduced tensor.
    """
    try:
        import mindspore.ops as ops
        from mindspore.communication import ReduceScatter
    except ImportError:
        logger.warning("MindSpore communication not available, returning input")
        return tensor

    reduce_scatter_op = ReduceScatter()
    result = reduce_scatter_op(tensor)

    return result


def barrier() -> None:
    """Synchronization barrier.

    Blocks until all processes reach this point.

    Example:
        >>> from daca.comm import barrier
        >>> barrier()  # Wait for all processes
    """
    global _initialized

    if not _initialized:
        return

    try:
        # MindSpore doesn't have explicit barrier, use HCCL
        import mindspore.ops as ops

        # Create a small tensor and all-reduce as barrier
        dummy = ops.zeros((1,), dtype=ms.float32)
        all_reduce(dummy, op="sum")

    except Exception as e:
        logger.debug(f"Barrier warning: {e}")


def destroy_parallel() -> None:
    """Destroy parallel communication group.

    Example:
        >>> from daca.comm import destroy_parallel
        >>> destroy_parallel()
    """
    global _initialized, _rank, _world_size, _backend

    if not _initialized:
        return

    try:
        from mindspore.communication import release

        release()
        logger.info("Destroyed parallel communication group")

    except Exception as e:
        logger.debug(f"Destroy warning: {e}")

    _initialized = False
    _rank = 0
    _world_size = 1
    _backend = None
