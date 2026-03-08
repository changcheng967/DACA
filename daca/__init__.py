"""DACA - DaVinci Accelerated Compute Architecture.

DACA is to Ascend what CUDA is to NVIDIA and ROCm is to AMD.
A compute platform library that makes Ascend 910ProA NPUs fully usable
for AI workloads by closing operator gaps, fixing CANN bugs, optimizing
performance, and enabling CUDA-ecosystem code to run on Ascend.

Example:
    import daca

    # Apply all compatibility patches
    daca.patch()

    # Your MindSpore code now works on Ascend
    # ...

    # When done, restore original state
    daca.unpatch()
"""

__version__ = "0.1.3"
__author__ = "DACA Contributors"
__license__ = "Apache-2.0"

import logging
from typing import Optional, Dict, Any

# Configure logging for DACA
logger = logging.getLogger("daca")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Track patch state
_patched: bool = False
_original_state: Dict[str, Any] = {}

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "runtime":
        from daca import runtime
        return runtime
    elif name == "blas":
        from daca import blas
        return blas
    elif name == "nn":
        from daca import nn
        return nn
    elif name == "comm":
        from daca import comm
        return comm
    elif name == "compile":
        from daca import compile
        return compile
    elif name == "compat":
        from daca import compat
        return compat
    elif name == "autotune":
        from daca import autotune
        return autotune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def info() -> None:
    """Display DACA information banner.

    Shows version, author, and basic environment information.
    This is the only function that uses print() instead of logging,
    as it's meant for interactive use.

    Example:
        >>> import daca
        >>> daca.info()
        ╔═══════════════════════════════════════════════════════════╗
        ║  DACA - DaVinci Accelerated Compute Architecture          ║
        ║  Version: 0.1.1                                           ║
        ║  Ascend NPU Platform Library                              ║
        ╚═══════════════════════════════════════════════════════════╝
    """
    banner = f"""
╔═══════════════════════════════════════════════════════════════╗
║  DACA - DaVinci Accelerated Compute Architecture              ║
║  Version: {__version__:<52}║
║  Ascend NPU Platform Library                                  ║
║                                                               ║
║  "DACA is to Ascend what CUDA is to NVIDIA"                   ║
║                                                               ║
║  DaCAAttention: Chunked Online Softmax Attention              ║
║  (FlashAttention-equivalent, pure MindSpore, w/ backward)     ║
║                                                               ║
║  v0.1.3: DaCAAttention IS nn.Cell - full autograd + MF patch  ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)

    # Try to detect hardware
    try:
        from daca.runtime import detect_npu, get_npu_info, check_cann_version

        print("Hardware Detection:")
        if detect_npu():
            info = get_npu_info()
            print(f"  NPUs detected: {info.get('count', 'unknown')}")
            print(f"  Device: {info.get('name', 'unknown')}")
            print(f"  Memory: {info.get('memory_gb', 'unknown')} GB per device")

            cann_info = check_cann_version()
            print(f"  CANN version: {cann_info.get('version', 'unknown')}")
        else:
            print("  No Ascend NPUs detected (or running in CPU mode)")
    except ImportError:
        print("  Runtime detection not available (MindSpore not installed)")

    print("\nUsage:")
    print("  daca.patch()    - Apply all compatibility patches")
    print("  daca.unpatch()  - Remove all patches")
    print("  daca.benchmark() - Run performance benchmarks")
    print()


def patch() -> None:
    """Apply all DACA compatibility patches.

    This applies all workarounds for known Ascend/CANN issues:

    1. bf16 → fp16 transparent shim - Intercepts Tensor.astype calls
       that request bfloat16 and redirects to float16, since bf16 is
       not supported on 910ProA hardware.

    2. SiLU/SwiGLU activation injection - Adds missing SiLU and SwiGLU
       to the MindSpore ops namespace using manual implementations.

    3. LayerNorm fp32 fast-path - Patches LayerNorm to upcast to fp32
       before normalization to avoid CANN fusion bug that crashes fp16.

    4. CANN fusion pass disabling - Disables FlashAttentionFusion V1/V2
       which cause rank mismatch errors due to aggressive fusion.

    5. Graph mode safe env vars - Sets environment variables for stable
       graph mode compilation.

    6. MindSpore ops namespace patching - Adds missing operators.

    7. MindFormers compatibility patches - Fixes bf16 configs and
       broken LayerNorm usage in MindFormers models.

    All patches are reversible via daca.unpatch().

    Raises:
        RuntimeError: If patches are already applied.

    Example:
        >>> import daca
        >>> daca.patch()
        >>> # Now bf16 → fp16, SiLU works, LayerNorm doesn't crash
        >>> daca.unpatch()  # Restore original state
    """
    global _patched

    if _patched:
        logger.warning("DACA patches already applied. Call daca.unpatch() first to re-apply.")
        return

    logger.info("Applying DACA compatibility patches...")

    # Store original state for unpatch
    _original_state.clear()
    _original_state["patched"] = True

    try:
        # 1. Enable bf16 → fp16 shim
        from daca.runtime.dtype import enable_bf16_shim
        enable_bf16_shim()
        logger.debug("Enabled bf16 → fp16 shim")

        # 2. Inject SiLU/SwiGLU activations
        from daca.nn.activations import inject_silu, inject_swiglu
        inject_silu()
        inject_swiglu()
        logger.debug("Injected SiLU/SwiGLU activations")

        # 3. Enable LayerNorm fp32 upcast
        from daca.nn.layernorm import enable_fp32_upcast
        enable_fp32_upcast()
        logger.debug("Enabled LayerNorm fp32 upcast")

        # 4. Disable broken CANN fusions
        from daca.compile.fusion import disable_flash_attention_fusion
        disable_flash_attention_fusion()
        logger.debug("Disabled FlashAttention fusion")

        # 5. Set graph mode safe env vars
        from daca.compile.graph_mode import set_safe_env
        set_safe_env()
        logger.debug("Set graph mode safe environment variables")

        # 6. Apply MindSpore patches
        from daca.compat.mindspore_patches import apply_all as apply_ms_patches
        apply_ms_patches()
        logger.debug("Applied MindSpore patches")

        # 7. Apply MindFormers patches
        from daca.compat.mindformers_patches import apply_all as apply_mf_patches
        apply_mf_patches()
        logger.debug("Applied MindFormers patches")

        _patched = True
        logger.info("DACA patches applied successfully.")

    except Exception as e:
        logger.error(f"Failed to apply patches: {e}")
        # Attempt to rollback
        unpatch()
        raise RuntimeError(f"Failed to apply DACA patches: {e}") from e


def unpatch() -> None:
    """Remove all DACA compatibility patches.

    Restores the original state of all patched modules and functions.
    Safe to call even if patches were not applied.

    Example:
        >>> import daca
        >>> daca.patch()
        >>> # ... use patched environment ...
        >>> daca.unpatch()  # Restore original state
    """
    global _patched

    if not _patched:
        logger.debug("No patches to remove.")
        return

    logger.info("Removing DACA compatibility patches...")

    errors = []

    # Remove patches in reverse order
    try:
        from daca.compat.mindformers_patches import revert_all as revert_mf_patches
        revert_mf_patches()
    except Exception as e:
        errors.append(f"MindFormers patches: {e}")

    try:
        from daca.compat.mindspore_patches import revert_all as revert_ms_patches
        revert_ms_patches()
    except Exception as e:
        errors.append(f"MindSpore patches: {e}")

    try:
        from daca.compile.graph_mode import unset_safe_env
        unset_safe_env()
    except Exception as e:
        errors.append(f"Graph mode env: {e}")

    try:
        from daca.compile.fusion import enable_flash_attention_fusion
        enable_flash_attention_fusion()
    except Exception as e:
        errors.append(f"FlashAttention fusion: {e}")

    try:
        from daca.nn.layernorm import disable_fp32_upcast
        disable_fp32_upcast()
    except Exception as e:
        errors.append(f"LayerNorm fp32 upcast: {e}")

    try:
        from daca.nn.activations import remove_silu, remove_swiglu
        remove_silu()
        remove_swiglu()
    except Exception as e:
        errors.append(f"Activation injections: {e}")

    try:
        from daca.runtime.dtype import disable_bf16_shim
        disable_bf16_shim()
    except Exception as e:
        errors.append(f"bf16 shim: {e}")

    _patched = False
    _original_state.clear()

    if errors:
        logger.warning(f"Some patches could not be fully reverted: {'; '.join(errors)}")
    else:
        logger.info("DACA patches removed successfully.")


def is_patched() -> bool:
    """Check if DACA patches are currently applied.

    Returns:
        True if patches are applied, False otherwise.

    Example:
        >>> import daca
        >>> daca.is_patched()
        False
        >>> daca.patch()
        >>> daca.is_patched()
        True
    """
    return _patched


def benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Run DACA performance benchmarks.

    Tests key operations to measure performance on the current hardware.

    Args:
        verbose: If True, print results to stdout.

    Returns:
        Dictionary with benchmark results.

    Example:
        >>> import daca
        >>> results = daca.benchmark()
        MatMul (1024x1024): 12.5 ms
        BatchMatMul (32x128x64): 3.2 ms
        FlashAttention (seq=512, heads=32): 45.7 ms
        ...
    """
    from daca.autotune.benchmark import run_all_benchmarks
    return run_all_benchmarks(verbose=verbose)


# Public API
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "info",
    "patch",
    "unpatch",
    "is_patched",
    "benchmark",
]
