"""DACA Runtime Module.

Hardware detection, device management, and memory tracking for Ascend NPUs.

Example:
    from daca.runtime import detect_npu, get_npu_info, set_device

    if detect_npu():
        info = get_npu_info()
        print(f"Found {info['count']} NPUs")

        # Set active device
        set_device(0)
"""

from daca.runtime.detect import (
    detect_npu,
    get_npu_info,
    check_cann_version,
    is_openi_env,
    get_platform_info,
)
from daca.runtime.device import (
    Device,
    set_device,
    get_device,
    device_count,
    current_device,
    synchronize,
)
from daca.runtime.dtype import (
    BFloat16Shim,
    enable_bf16_shim,
    disable_bf16_shim,
    is_bf16_supported,
    get_bf16_interception_count,
)
from daca.runtime.memory import (
    MemoryTracker,
    get_memory_usage,
    get_max_memory_allocated,
    reset_peak_memory_stats,
)

__all__ = [
    # Hardware detection
    "detect_npu",
    "get_npu_info",
    "check_cann_version",
    "is_openi_env",
    "get_platform_info",
    # Device management
    "Device",
    "set_device",
    "get_device",
    "device_count",
    "current_device",
    "synchronize",
    # bf16 shim
    "BFloat16Shim",
    "enable_bf16_shim",
    "disable_bf16_shim",
    "is_bf16_supported",
    "get_bf16_interception_count",
    # Memory tracking
    "MemoryTracker",
    "get_memory_usage",
    "get_max_memory_allocated",
    "reset_peak_memory_stats",
]
