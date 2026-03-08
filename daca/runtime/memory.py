"""Memory tracking for Ascend NPUs.

Provides memory usage monitoring and tracking utilities.

Example:
    from daca.runtime import MemoryTracker

    tracker = MemoryTracker()
    with tracker.track("forward_pass"):
        # ... operations ...
        pass
    print(tracker.summary())
"""

import logging
import time
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("daca.runtime.memory")

# Global memory stats
_peak_memory: int = 0
_current_memory: int = 0


@dataclass
class MemoryRecord:
    """Record of memory usage at a point in time."""

    name: str
    timestamp: float
    allocated_bytes: int
    peak_bytes: int
    delta_bytes: int = 0


class MemoryTracker:
    """Track memory usage during operations.

    Provides detailed memory tracking with named regions and
    summary statistics.

    Attributes:
        records: List of memory records.
        enabled: Whether tracking is enabled.

    Example:
        >>> tracker = MemoryTracker()
        >>> with tracker.track("layer1"):
        ...     # ... allocate tensors ...
        ...     pass
        >>> print(tracker.summary())
    """

    def __init__(self, enabled: bool = True):
        """Initialize memory tracker.

        Args:
            enabled: Whether tracking is enabled.
        """
        self.enabled = enabled
        self.records: List[MemoryRecord] = []
        self._baseline: int = 0
        self._start_time: float = 0

    def _get_memory_allocated(self) -> int:
        """Get current memory allocated in bytes.

        Returns:
            Current memory allocation in bytes.
        """
        try:
            import mindspore as ms
            from mindspore import context

            # MindSpore doesn't have a direct memory query API
            # Use HCCL or npu-smi if available
            import os
            import subprocess

            device_id = context.get_context("device_id") or 0

            # Try npu-smi
            for smi_path in ["/usr/local/Ascend/bin/npu-smi", "/usr/bin/npu-smi"]:
                if os.path.exists(smi_path):
                    try:
                        result = subprocess.run(
                            [smi_path, "info", "-t", "usages", "-i", str(device_id)],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            # Parse memory usage from output
                            import re
                            match = re.search(r"Memory-Usage\s*[:\s]+(\d+)\s*/\s*(\d+)\s*MB", result.stdout)
                            if match:
                                used_mb = int(match.group(1))
                                return used_mb * 1024 * 1024
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
        except Exception:
            pass

        # Fallback: estimate from tensor count (rough approximation)
        return 0

    def record(self, name: str) -> MemoryRecord:
        """Record current memory usage.

        Args:
            name: Label for this record.

        Returns:
            MemoryRecord with current stats.
        """
        if not self.enabled:
            return MemoryRecord(name=name, timestamp=time.time(), allocated_bytes=0, peak_bytes=0)

        current = self._get_memory_allocated()
        global _peak_memory
        if current > _peak_memory:
            _peak_memory = current

        record = MemoryRecord(
            name=name,
            timestamp=time.time(),
            allocated_bytes=current,
            peak_bytes=_peak_memory,
            delta_bytes=current - (self.records[-1].allocated_bytes if self.records else self._baseline)
        )

        self.records.append(record)
        return record

    @contextmanager
    def track(self, name: str):
        """Context manager to track memory for a region.

        Args:
            name: Label for this region.

        Yields:
            MemoryRecord (populated after context exits).

        Example:
            >>> with tracker.track("attention"):
            ...     # ... attention computation ...
            ...     pass
        """
        start_record = self.record(f"{name}_start")
        start_time = time.time()

        record = MemoryRecord(
            name=name,
            timestamp=start_time,
            allocated_bytes=start_record.allocated_bytes,
            peak_bytes=_peak_memory
        )

        try:
            yield record
        finally:
            end_record = self.record(f"{name}_end")
            record.allocated_bytes = end_record.allocated_bytes
            record.peak_bytes = _peak_memory
            record.delta_bytes = end_record.allocated_bytes - start_record.allocated_bytes

    def reset(self) -> None:
        """Clear all records."""
        self.records.clear()
        self._baseline = self._get_memory_allocated()

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Dictionary with memory statistics.
        """
        if not self.records:
            return {
                "total_records": 0,
                "peak_memory_mb": 0,
                "current_memory_mb": 0,
            }

        peak = max(r.peak_bytes for r in self.records)
        current = self.records[-1].allocated_bytes if self.records else 0

        return {
            "total_records": len(self.records),
            "peak_memory_mb": peak / (1024 * 1024),
            "current_memory_mb": current / (1024 * 1024),
            "records": [
                {
                    "name": r.name,
                    "allocated_mb": r.allocated_bytes / (1024 * 1024),
                    "delta_mb": r.delta_bytes / (1024 * 1024),
                }
                for r in self.records
            ]
        }

    def __repr__(self) -> str:
        return f"MemoryTracker(records={len(self.records)}, enabled={self.enabled})"


def get_memory_usage(device: Optional[int] = None) -> Dict[str, int]:
    """Get current memory usage.

    Args:
        device: Device index. If None, uses current device.

    Returns:
        Dictionary with 'allocated', 'peak', 'total' in bytes.

    Example:
        >>> from daca.runtime import get_memory_usage
        >>> usage = get_memory_usage()
        >>> print(f"Using {usage['allocated'] / 1e9:.2f} GB")
    """
    global _current_memory, _peak_memory

    # Try to get actual memory usage
    try:
        import os
        import subprocess

        device_id = device if device is not None else 0

        for smi_path in ["/usr/local/Ascend/bin/npu-smi", "/usr/bin/npu-smi"]:
            if os.path.exists(smi_path):
                try:
                    result = subprocess.run(
                        [smi_path, "info", "-t", "usages", "-i", str(device_id)],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        import re
                        # Parse: Memory-Usage : 1024 / 32768 MB
                        match = re.search(r"Memory-Usage\s*[:\s]+(\d+)\s*/\s*(\d+)\s*MB", result.stdout)
                        if match:
                            used = int(match.group(1)) * 1024 * 1024
                            total = int(match.group(2)) * 1024 * 1024
                            _current_memory = used
                            if used > _peak_memory:
                                _peak_memory = used
                            return {
                                "allocated": used,
                                "peak": _peak_memory,
                                "total": total,
                            }
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
    except Exception:
        pass

    # Fallback
    return {
        "allocated": _current_memory,
        "peak": _peak_memory,
        "total": 32 * 1024 * 1024 * 1024,  # Assume 32GB
    }


def get_max_memory_allocated() -> int:
    """Get peak memory allocated since last reset.

    Returns:
        Peak memory in bytes.

    Example:
        >>> from daca.runtime import get_max_memory_allocated
        >>> peak = get_max_memory_allocated()
        >>> print(f"Peak: {peak / 1e9:.2f} GB")
    """
    return _peak_memory


def reset_peak_memory_stats() -> None:
    """Reset peak memory tracking.

    Example:
        >>> from daca.runtime import reset_peak_memory_stats
        >>> reset_peak_memory_stats()
        >>> # Peak memory counter reset
    """
    global _peak_memory
    _peak_memory = 0


def empty_cache() -> None:
    """Attempt to release cached memory.

    Note: MindSpore manages memory automatically. This is a no-op
    provided for API compatibility with PyTorch.

    Example:
        >>> from daca.runtime import empty_cache
        >>> empty_cache()
    """
    # MindSpore doesn't have explicit cache clearing
    # Memory is managed by the runtime
    logger.debug("empty_cache called (no-op for MindSpore)")
