"""DACA BLAS Module.

Optimized matrix operations for Ascend NPUs with workspace handling
and shape validation.

Example:
    from daca.blas import matmul, bmm

    # 2D MatMul
    result = matmul(a, b)

    # BatchMatMul
    result = bmm(batch_a, batch_b)
"""

from daca.blas.matmul import (
    matmul,
    linear,
    addmm,
)
from daca.blas.bmm import (
    bmm,
    batch_matmul,
)
from daca.blas.workspace import (
    WorkspaceManager,
    get_workspace,
    preallocate_workspace,
)

__all__ = [
    # MatMul
    "matmul",
    "linear",
    "addmm",
    # BatchMatMul
    "bmm",
    "batch_matmul",
    # Workspace
    "WorkspaceManager",
    "get_workspace",
    "preallocate_workspace",
]
