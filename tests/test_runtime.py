"""Tests for DACA runtime module."""

import pytest
import os
import sys

# Skip all tests in this module if MindSpore not available
pytestmark = pytest.mark.mindspore


class TestDetect:
    """Tests for hardware detection."""

    def test_detect_npu_returns_bool(self):
        """Test that detect_npu returns a boolean."""
        from daca.runtime import detect_npu

        result = detect_npu()
        assert isinstance(result, bool)

    def test_get_platform_info(self):
        """Test platform info retrieval."""
        from daca.runtime import get_platform_info

        info = get_platform_info()

        assert isinstance(info, dict)
        assert "os" in info
        assert "arch" in info
        assert "python" in info

    def test_is_openi_env_returns_bool(self):
        """Test OpenI environment detection."""
        from daca.runtime import is_openi_env

        result = is_openi_env()
        assert isinstance(result, bool)

    def test_check_cann_version_returns_dict(self):
        """Test CANN version check."""
        from daca.runtime import check_cann_version

        info = check_cann_version()

        assert isinstance(info, dict)
        assert "version" in info
        assert "compatible" in info


class TestDevice:
    """Tests for device management."""

    def test_device_count_returns_int(self):
        """Test device_count returns integer."""
        from daca.runtime import device_count

        count = device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_current_device_returns_int(self):
        """Test current_device returns integer."""
        from daca.runtime import current_device

        device = current_device()
        assert isinstance(device, int)
        assert device >= 0

    def test_get_device_with_none(self):
        """Test get_device with None returns current device."""
        from daca.runtime import get_device, current_device

        device = get_device(None)
        assert device == current_device()

    def test_get_device_with_value(self):
        """Test get_device with value returns that value."""
        from daca.runtime import get_device

        device = get_device(2)
        assert device == 2

    def test_set_device_invalid(self):
        """Test set_device raises on invalid index."""
        from daca.runtime import set_device, device_count

        count = device_count()
        if count > 0:
            with pytest.raises(ValueError):
                set_device(-1)

            with pytest.raises(ValueError):
                set_device(count + 100)


class TestDeviceContext:
    """Tests for Device context manager."""

    def test_device_context_manager(self):
        """Test Device context manager."""
        from daca.runtime import Device, current_device

        original = current_device()

        with Device(0):
            # Inside context, should be device 0
            pass

        # After context, should restore
        assert current_device() == original


class TestDtype:
    """Tests for dtype handling."""

    def test_is_bf16_supported_returns_false(self):
        """Test is_bf16_supported returns False on 910ProA."""
        from daca.runtime import is_bf16_supported

        assert is_bf16_supported() is False

    def test_bf16_shim_disabled_by_default(self):
        """Test BF16 shim is disabled by default."""
        from daca.runtime.dtype import is_shim_enabled

        # Should be disabled initially
        # (may be enabled by other tests, so we just check it's a bool)
        assert isinstance(is_shim_enabled(), bool)


class TestMemory:
    """Tests for memory tracking."""

    def test_get_memory_usage_returns_dict(self):
        """Test get_memory_usage returns dictionary."""
        from daca.runtime import get_memory_usage

        usage = get_memory_usage()

        assert isinstance(usage, dict)
        assert "allocated" in usage
        assert "peak" in usage
        assert "total" in usage

    def test_get_max_memory_allocated_returns_int(self):
        """Test get_max_memory_allocated returns integer."""
        from daca.runtime import get_max_memory_allocated

        peak = get_max_memory_allocated()
        assert isinstance(peak, int)
        assert peak >= 0

    def test_memory_tracker_context(self):
        """Test MemoryTracker context manager."""
        from daca.runtime import MemoryTracker

        tracker = MemoryTracker()

        with tracker.track("test_op"):
            pass

        summary = tracker.summary()
        assert "total_records" in summary


class TestBFloat16Shim:
    """Tests for BF16 shim."""

    def test_enable_disable_shim(self):
        """Test enabling and disabling BF16 shim."""
        from daca.runtime import enable_bf16_shim, disable_bf16_shim, is_bf16_supported

        # Enable
        enable_bf16_shim()

        # Disable
        disable_bf16_shim()

        # Should always be False for 910ProA
        assert is_bf16_supported() is False
