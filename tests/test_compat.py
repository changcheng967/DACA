"""Tests for DACA compatibility module."""

import pytest
import json
import tempfile
import os


class TestConfigRewriter:
    """Tests for config rewriter."""

    def test_rewrite_config_bf16_to_fp16(self):
        """Test rewriting bf16 to fp16 in config dict."""
        from daca.compat import rewrite_config

        config = {
            "dtype": "bfloat16",
            "hidden_size": 768,
        }

        rewritten = rewrite_config(config)

        assert rewritten["dtype"] == "float16"
        assert rewritten["hidden_size"] == 768

    def test_rewrite_config_nested(self):
        """Test rewriting nested config."""
        from daca.compat import rewrite_config

        config = {
            "model": {
                "dtype": "bfloat16",
                "layers": 12,
            },
            "training": {
                "compute_dtype": "bf16",
            },
        }

        rewritten = rewrite_config(config)

        assert rewritten["model"]["dtype"] == "float16"
        assert rewritten["training"]["compute_dtype"] == "fp16"

    def test_rewrite_config_mstype(self):
        """Test rewriting mstype.bfloat16."""
        from daca.compat import rewrite_config

        config = {
            "dtype": "mstype.bfloat16",
        }

        rewritten = rewrite_config(config)

        assert rewritten["dtype"] == "mstype.float16"

    def test_config_rewriter_class(self):
        """Test ConfigRewriter class."""
        from daca.compat import ConfigRewriter

        rewriter = ConfigRewriter()

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"dtype": "bfloat16"}, f)
            temp_path = f.name

        try:
            # Rewrite
            output_path = rewriter.rewrite_json(temp_path)

            # Read back
            with open(output_path) as f:
                rewritten = json.load(f)

            assert rewritten["dtype"] == "float16"

        finally:
            os.unlink(temp_path)


class TestMindSporePatches:
    """Tests for MindSpore patches."""

    def test_apply_all(self, daca_patched):
        """Test applying all MindSpore patches."""
        from daca.compat import apply_ms_patches

        # Should not raise
        apply_ms_patches()

    def test_patch_ops_namespace(self):
        """Test patching ops namespace."""
        from daca.compat import patch_ops_namespace, revert_ops_namespace

        patch_ops_namespace()

        try:
            # Check if SiLU was added
            pytest.importorskip("mindspore")
            import mindspore.ops as ops
            assert hasattr(ops, "silu")
        finally:
            revert_ops_namespace()


class TestMindFormersPatches:
    """Tests for MindFormers patches."""

    def test_apply_all(self, daca_patched):
        """Test applying all MindFormers patches."""
        from daca.compat import apply_mf_patches

        # Should not raise
        apply_mf_patches()

    def test_rewrite_kwargs_dtype(self):
        """Test kwargs dtype rewriting."""
        pytest.importorskip("mindspore")
        import mindspore.common.dtype as mstype
        from daca.compat.mindformers_patches import _rewrite_kwargs_dtype

        kwargs = {
            "dtype": mstype.bfloat16,
            "hidden_size": 768,
        }

        rewritten = _rewrite_kwargs_dtype(kwargs)

        assert rewritten["dtype"] == mstype.float16


class TestCUDAShim:
    """Tests for CUDA API shim."""

    def test_shim_cuda_api(self):
        """Test CUDA API shim."""
        from daca.compat import shim_cuda_api, unshim_cuda_api

        shim_cuda_api()

        try:
            torch = pytest.importorskip("torch")
            # is_available should return True if NPU available
            result = torch.cuda.is_available()
            assert isinstance(result, bool)

        finally:
            unshim_cuda_api()

    def test_get_npu_count_as_cuda(self):
        """Test getting NPU count."""
        from daca.compat import get_npu_count_as_cuda

        count = get_npu_count_as_cuda()
        assert isinstance(count, int)
        assert count >= 0


# Fixtures
@pytest.fixture
def daca_patched():
    """Apply and revert DACA patches."""
    import daca
    daca.patch()
    yield
    daca.unpatch()
