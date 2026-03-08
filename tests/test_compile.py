"""Tests for DACA compile module."""

import pytest
import os


class TestGraphMode:
    """Tests for graph mode utilities."""

    def test_set_safe_env(self):
        """Test set_safe_env sets environment variables."""
        from daca.compile import set_safe_env, unset_safe_env

        set_safe_env()

        try:
            assert os.environ.get("MS_DEV_RUNTIME_CONF") == "1"
        finally:
            unset_safe_env()

    def test_unset_safe_env(self):
        """Test unset_safe_env removes environment variables."""
        from daca.compile import set_safe_env, unset_safe_env

        set_safe_env()
        unset_safe_env()

        # Should be removed
        assert os.environ.get("MS_DEV_RUNTIME_CONF") != "1"

    def test_enable_graph_mode(self):
        """Test enable_graph_mode."""
        pytest.importorskip("mindspore")
        from daca.compile import enable_graph_mode, disable_graph_mode

        enable_graph_mode()

        try:
            from mindspore import context
            assert context.get_context("mode") == context.GRAPH_MODE
        finally:
            disable_graph_mode()

    def test_graph_cell_context_manager(self):
        """Test GraphCell context manager."""
        pytest.importorskip("mindspore")
        from daca.compile import GraphCell

        with GraphCell(device_target="CPU"):
            pass  # Should work


class TestFusion:
    """Tests for fusion control."""

    def test_disable_flash_attention_fusion(self):
        """Test disable_flash_attention_fusion."""
        from daca.compile import disable_flash_attention_fusion, enable_flash_attention_fusion

        disable_flash_attention_fusion()

        try:
            assert os.environ.get("MS_DEV_DISABLE_FLASH_ATTENTION_FUSION") == "1"
        finally:
            enable_flash_attention_fusion()

    def test_enable_flash_attention_fusion(self):
        """Test enable_flash_attention_fusion."""
        from daca.compile import disable_flash_attention_fusion, enable_flash_attention_fusion

        disable_flash_attention_fusion()
        enable_flash_attention_fusion()

        assert os.environ.get("MS_DEV_DISABLE_FLASH_ATTENTION_FUSION") != "1"

    def test_get_fusion_status(self):
        """Test get_fusion_status."""
        from daca.compile.fusion import get_fusion_status

        status = get_fusion_status()

        assert isinstance(status, dict)
        assert "flash_attention" in status


class TestFusionConfig:
    """Tests for FusionConfig."""

    def test_fusion_config_create(self):
        """Test FusionConfig creation."""
        from daca.compile import FusionConfig

        config = FusionConfig(
            disable_flash_attention=True,
            disable_matmul=False,
        )

        assert config.disable_flash_attention is True
        assert config.disable_matmul is False

    def test_fusion_config_apply(self):
        """Test FusionConfig apply."""
        from daca.compile import FusionConfig

        config = FusionConfig(disable_flash_attention=True)
        config.apply()

        try:
            assert os.environ.get("MS_DEV_DISABLE_FLASH_ATTENTION_FUSION") == "1"
        finally:
            # Clean up
            if "MS_DEV_DISABLE_FLASH_ATTENTION_FUSION" in os.environ:
                del os.environ["MS_DEV_DISABLE_FLASH_ATTENTION_FUSION"]

    def test_fusion_config_custom_passes(self):
        """Test FusionConfig with custom passes."""
        from daca.compile import FusionConfig

        config = FusionConfig(
            custom_passes=["custom_op1", "custom_op2"]
        )
        config.apply()

        try:
            assert os.environ.get("MS_DEV_DISABLE_CUSTOM_OP1_FUSION") == "1"
            assert os.environ.get("MS_DEV_DISABLE_CUSTOM_OP2_FUSION") == "1"
        finally:
            # Clean up
            for key in list(os.environ.keys()):
                if key.startswith("MS_DEV_DISABLE_"):
                    del os.environ[key]
