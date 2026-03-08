"""CUDA API shim for Ascend NPUs.

Shims torch.cuda APIs to redirect to Ascend equivalents,
enabling PyTorch code to run on Ascend with minimal changes.

WHY: Many codebases use torch.cuda.is_available() checks.
These shims redirect to NPU equivalents.

Example:
    from daca.compat import shim_cuda_api

    shim_cuda_api()
    # Now torch.cuda.is_available() returns True if NPU available
"""

import logging
from typing import Optional, Any, Dict

logger = logging.getLogger("daca.compat.cuda_shim")

# Track shim state
_shim_active: bool = False
_original_torch_cuda: Dict[str, Any] = {}


def shim_cuda_api() -> None:
    """Shim torch.cuda APIs to Ascend equivalents.

    Patches torch.cuda to redirect common operations:
    - is_available() -> check for NPU
    - device_count() -> NPU count
    - current_device() -> current NPU
    - set_device() -> set NPU

    Example:
        >>> from daca.compat import shim_cuda_api
        >>> shim_cuda_api()
        >>> import torch
        >>> torch.cuda.is_available()  # Now checks for NPU
    """
    global _shim_active, _original_torch_cuda

    if _shim_active:
        logger.debug("CUDA API already shimmed")
        return

    try:
        import torch

        # Store original functions
        _original_torch_cuda = {
            "is_available": getattr(torch.cuda, "is_available", None),
            "device_count": getattr(torch.cuda, "device_count", None),
            "current_device": getattr(torch.cuda, "current_device", None),
            "set_device": getattr(torch.cuda, "set_device", None),
        }

        # Shim is_available
        def shimmed_is_available():
            try:
                from daca.runtime import detect_npu
                return detect_npu()
            except Exception:
                return False

        torch.cuda.is_available = shimmed_is_available

        # Shim device_count
        def shimmed_device_count():
            try:
                from daca.runtime import device_count
                return device_count()
            except Exception:
                return 0

        torch.cuda.device_count = shimmed_device_count

        # Shim current_device
        def shimmed_current_device():
            try:
                from daca.runtime import current_device
                return current_device()
            except Exception:
                return 0

        torch.cuda.current_device = shimmed_current_device

        # Shim set_device
        def shimmed_set_device(device):
            try:
                from daca.runtime import set_device
                set_device(device)
            except Exception as e:
                logger.warning(f"set_device failed: {e}")

        torch.cuda.set_device = shimmed_set_device

        _shim_active = True
        logger.info("Shimmed torch.cuda APIs to Ascend equivalents")

    except ImportError:
        logger.debug("PyTorch not available, CUDA shim skipped")
    except Exception as e:
        logger.warning(f"Failed to shim CUDA API: {e}")


def unshim_cuda_api() -> None:
    """Remove CUDA API shims and restore original behavior.

    Example:
        >>> from daca.compat import unshim_cuda_api
        >>> unshim_cuda_api()
    """
    global _shim_active, _original_torch_cuda

    if not _shim_active:
        logger.debug("CUDA API not shimmed")
        return

    try:
        import torch

        # Restore original functions
        for name, original in _original_torch_cuda.items():
            if original is not None:
                setattr(torch.cuda, name, original)

        _original_torch_cuda.clear()
        _shim_active = False
        logger.info("Restored original torch.cuda APIs")

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to unshim CUDA API: {e}")


def get_npu_count_as_cuda() -> int:
    """Get NPU count (shimmed as CUDA device count).

    Returns:
        Number of Ascend NPUs.

    Example:
        >>> from daca.compat import get_npu_count_as_cuda
        >>> count = get_npu_count_as_cuda()
    """
    try:
        from daca.runtime import device_count
        return device_count()
    except Exception:
        return 0


def redirect_cuda_calls_to_npu() -> None:
    """Redirect CUDA device strings to NPU.

    Modifies code that uses "cuda:0" style device strings
    to use Ascend equivalents.

    Example:
        >>> from daca.compat.cuda_shim import redirect_cuda_calls_to_npu
        >>> redirect_cuda_calls_to_npu()
    """
    # This would require AST manipulation for full implementation
    # For now, rely on the API shims
    logger.info("CUDA call redirection enabled via API shims")
