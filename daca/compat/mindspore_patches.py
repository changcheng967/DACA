"""MindSpore patches for missing operators and bug workarounds.

Patches the MindSpore ops namespace to add missing operators
and fix known issues.

WHY: ops.SiLU missing, LayerNorm fp16 crashes. These patches
add missing functionality and work around bugs.

Example:
    from daca.compat import apply_ms_patches

    apply_ms_patches()
"""

import logging
from typing import Optional, Any, Dict, Callable

logger = logging.getLogger("daca.compat.mindspore_patches")

# Track patch state
_patches_applied: bool = False
_original_ops: Dict[str, Any] = {}
_original_nn: Dict[str, Any] = {}


def apply_all() -> None:
    """Apply all MindSpore patches.

    Applies:
    1. ops namespace patches (SiLU, SwiGLU)
    2. LayerNorm fp32 upcast patch

    Example:
        >>> from daca.compat.mindspore_patches import apply_all
        >>> apply_all()
    """
    global _patches_applied

    if _patches_applied:
        logger.debug("MindSpore patches already applied")
        return

    patch_ops_namespace()
    patch_layernorm()

    _patches_applied = True
    logger.info("Applied all MindSpore patches")


def revert_all() -> None:
    """Revert all MindSpore patches.

    Example:
        >>> from daca.compat.mindspore_patches import revert_all
        >>> revert_all()
    """
    global _patches_applied

    if not _patches_applied:
        logger.debug("MindSpore patches not applied")
        return

    revert_ops_namespace()
    revert_layernorm()

    _patches_applied = False
    logger.info("Reverted all MindSpore patches")


def patch_ops_namespace() -> None:
    """Patch mindspore.ops namespace with missing operators.

    Adds:
    - ops.silu: SiLU activation
    - ops.SiLU: SiLU class
    - ops.swiglu: SwiGLU activation
    - ops.SwiGLU: SwiGLU class

    Example:
        >>> from daca.compat.mindspore_patches import patch_ops_namespace
        >>> patch_ops_namespace()
        >>> import mindspore.ops as ops
        >>> y = ops.silu(x)  # Now works!
    """
    global _original_ops

    try:
        import mindspore.ops as ops
    except ImportError:
        logger.warning("MindSpore not available, ops patch skipped")
        return

    # Store originals
    for name in ["silu", "SiLU", "swiglu", "SwiGLU"]:
        if hasattr(ops, name):
            _original_ops[name] = getattr(ops, name)

    # Import implementations from daca.nn.activations
    from daca.nn.activations import silu, swiglu

    # Add functions
    ops.silu = silu
    ops.swiglu = swiglu

    # Add classes for API compatibility
    class SiLU:
        """SiLU activation class."""
        def __call__(self, x):
            return silu(x)

    class SwiGLU:
        """SwiGLU activation class."""
        def __call__(self, x, dim=-1):
            return swiglu(x, dim=dim)

    ops.SiLU = SiLU
    ops.SwiGLU = SwiGLU

    logger.info("Patched mindspore.ops namespace with SiLU and SwiGLU")


def revert_ops_namespace() -> None:
    """Revert ops namespace patches.

    Example:
        >>> from daca.compat.mindspore_patches import revert_ops_namespace
        >>> revert_ops_namespace()
    """
    global _original_ops

    try:
        import mindspore.ops as ops

        for name in ["silu", "SiLU", "swiglu", "SwiGLU"]:
            if name in _original_ops:
                setattr(ops, name, _original_ops[name])
            elif hasattr(ops, name):
                delattr(ops, name)

        _original_ops.clear()
        logger.info("Reverted ops namespace patches")

    except ImportError:
        pass


def patch_layernorm() -> None:
    """Patch LayerNorm for fp32 upcast.

    Patches nn.LayerNorm to upcast to fp32 before normalization,
    avoiding the CANN fusion bug.

    Example:
        >>> from daca.compat.mindspore_patches import patch_layernorm
        >>> patch_layernorm()
    """
    global _original_nn

    try:
        import mindspore.nn as nn
    except ImportError:
        logger.warning("MindSpore not available, LayerNorm patch skipped")
        return

    # Store original
    _original_nn["LayerNorm"] = nn.LayerNorm

    # Create wrapper that uses fp32 upcast
    OriginalLayerNorm = nn.LayerNorm

    class LayerNormWrapper(OriginalLayerNorm):
        """LayerNorm with fp32 upcast for Ascend."""

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

    nn.LayerNorm = LayerNormWrapper
    logger.info("Patched nn.LayerNorm for fp32 upcast")


def revert_layernorm() -> None:
    """Revert LayerNorm patch.

    Example:
        >>> from daca.compat.mindspore_patches import revert_layernorm
        >>> revert_layernorm()
    """
    global _original_nn

    try:
        import mindspore.nn as nn

        if "LayerNorm" in _original_nn:
            nn.LayerNorm = _original_nn["LayerNorm"]
            del _original_nn["LayerNorm"]
            logger.info("Reverted LayerNorm patch")

    except ImportError:
        pass


def add_op_to_namespace(name: str, op_func: Callable) -> None:
    """Add a custom operator to the MindSpore ops namespace.

    Args:
        name: Operator name.
        op_func: Operator function.

    Example:
        >>> from daca.compat.mindspore_patches import add_op_to_namespace
        >>> add_op_to_namespace("my_op", my_op_function)
    """
    try:
        import mindspore.ops as ops

        if hasattr(ops, name):
            _original_ops[name] = getattr(ops, name)

        setattr(ops, name, op_func)
        logger.debug(f"Added {name} to ops namespace")

    except ImportError:
        pass
