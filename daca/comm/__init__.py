"""DACA Communication Module.

Multi-NPU communication utilities using HCCL.

Example:
    from daca.comm import initialize_parallel, all_reduce
"""

from daca.comm.parallel import (
    initialize_parallel,
    is_initialized,
    get_rank,
    get_world_size,
    all_reduce,
    all_gather,
    broadcast,
    reduce_scatter,
    barrier,
)

__all__ = [
    "initialize_parallel",
    "is_initialized",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "all_gather",
    "broadcast",
    "reduce_scatter",
    "barrier",
]
