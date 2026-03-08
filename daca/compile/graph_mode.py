"""Graph mode configuration for Ascend NPUs.

Sets environment variables for stable graph mode compilation.

WHY: Graph mode on Ascend can fail with certain configurations.
Setting specific environment variables improves stability.

Example:
    from daca.compile import enable_graph_mode, set_safe_env

    set_safe_env()  # Set safe env vars
    enable_graph_mode()
"""

import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("daca.compile.graph_mode")

# Original environment for restoration
_original_env: Dict[str, str] = {}
_safe_env_set: bool = False


def set_safe_env() -> None:
    """Set environment variables for stable graph mode.

    Configures MindSpore and CANN environment for reliable graph compilation.

    Environment variables set:
        - MS_DEV_RUNTIME_CONF: Enable runtime config
        - ASCEND_SLOG_PRINT_TO_STDOUT: Log to stdout
        - ASCEND_GLOBAL_LOG_LEVEL: Set log level
        - MS_ASCEND_CHECK_OVERFLOW_MODE: Overflow handling

    Example:
        >>> from daca.compile import set_safe_env
        >>> set_safe_env()
    """
    global _safe_env_set, _original_env

    if _safe_env_set:
        logger.debug("Safe environment already set")
        return

    # Save original values
    env_vars = [
        "MS_DEV_RUNTIME_CONF",
        "ASCEND_SLOG_PRINT_TO_STDOUT",
        "ASCEND_GLOBAL_LOG_LEVEL",
        "MS_ASCEND_CHECK_OVERFLOW_MODE",
        "GRAPH_OP_RUN",
        "ENABLE_MS_DEBUGGER",
        "MS_JIT_MODULES",
        "MS_JIT_LEVEL",
    ]

    for var in env_vars:
        if var in os.environ:
            _original_env[var] = os.environ[var]

    # Set safe values
    safe_settings = {
        "MS_DEV_RUNTIME_CONF": "1",  # Enable runtime config
        "ASCEND_SLOG_PRINT_TO_STDOUT": "1",  # Print logs to stdout
        "ASCEND_GLOBAL_LOG_LEVEL": "3",  # Warning level
        "MS_ASCEND_CHECK_OVERFLOW_MODE": "0",  # Disable overflow check (faster)
        "GRAPH_OP_RUN": "1",  # Enable graph mode op run
        "ENABLE_MS_DEBUGGER": "0",  # Disable debugger for stability
    }

    for var, value in safe_settings.items():
        os.environ[var] = value

    _safe_env_set = True
    logger.info("Set safe graph mode environment variables")


def unset_safe_env() -> None:
    """Restore original environment variables.

    Removes all safe environment settings and restores original values.

    Example:
        >>> from daca.compile import unset_safe_env
        >>> unset_safe_env()
    """
    global _safe_env_set, _original_env

    if not _safe_env_set:
        logger.debug("Safe environment not set")
        return

    # Remove set variables
    env_vars = [
        "MS_DEV_RUNTIME_CONF",
        "ASCEND_SLOG_PRINT_TO_STDOUT",
        "ASCEND_GLOBAL_LOG_LEVEL",
        "MS_ASCEND_CHECK_OVERFLOW_MODE",
        "GRAPH_OP_RUN",
        "ENABLE_MS_DEBUGGER",
    ]

    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    # Restore original values
    for var, value in _original_env.items():
        os.environ[var] = value

    _original_env.clear()
    _safe_env_set = False
    logger.info("Restored original environment variables")


def enable_graph_mode(
    device_target: str = "Ascend",
    device_id: Optional[int] = None,
    max_call_depth: int = 1000,
) -> None:
    """Enable MindSpore graph mode with safe configuration.

    Graph mode provides better performance but requires careful configuration
    on Ascend NPUs.

    Args:
        device_target: Target device ("Ascend" or "CPU").
        device_id: Device index. If None, uses default.
        max_call_depth: Maximum recursion depth in graph.

    Example:
        >>> from daca.compile import enable_graph_mode
        >>> enable_graph_mode()
    """
    # First set safe environment
    set_safe_env()

    try:
        import mindspore as ms
        from mindspore import context

        # Configure context
        config = {
            "mode": context.GRAPH_MODE,
            "device_target": device_target,
        }

        if device_id is not None:
            config["device_id"] = device_id

        context.set_context(**config)

        logger.info(f"Enabled graph mode on {device_target}")

    except ImportError:
        logger.warning("MindSpore not available, graph mode not enabled")
    except Exception as e:
        logger.error(f"Failed to enable graph mode: {e}")
        raise


def disable_graph_mode() -> None:
    """Switch back to PyNative (eager) mode.

    Example:
        >>> from daca.compile import disable_graph_mode
        >>> disable_graph_mode()
    """
    try:
        import mindspore as ms
        from mindspore import context

        context.set_context(mode=context.PYNATIVE_MODE)
        logger.info("Switched to PyNative mode")

    except ImportError:
        logger.warning("MindSpore not available")
    except Exception as e:
        logger.error(f"Failed to disable graph mode: {e}")


class GraphCell:
    """Wrapper for safe graph mode cell execution.

    Provides a context manager and wrapper for running operations
    in graph mode with proper error handling.

    Example:
        >>> from daca.compile import GraphCell
        >>> with GraphCell():
        ...     # Operations here run in graph mode
        ...     pass
    """

    def __init__(
        self,
        device_target: str = "Ascend",
        device_id: Optional[int] = None,
    ):
        """Initialize GraphCell wrapper.

        Args:
            device_target: Target device.
            device_id: Device index.
        """
        self.device_target = device_target
        self.device_id = device_id
        self._original_mode = None

    def __enter__(self) -> "GraphCell":
        """Enter graph mode context."""
        try:
            from mindspore import context
            self._original_mode = context.get_context("mode")
            enable_graph_mode(self.device_target, self.device_id)
        except Exception as e:
            logger.warning(f"Could not enable graph mode: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit graph mode context and restore original mode."""
        try:
            from mindspore import context
            if self._original_mode is not None:
                context.set_context(mode=self._original_mode)
        except Exception:
            pass

    def __call__(self, fn):
        """Decorate function to run in graph mode.

        Args:
            fn: Function to wrap.

        Returns:
            Wrapped function.
        """
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper
