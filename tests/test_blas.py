"""Tests for DACA BLAS module."""

import pytest

# Skip all tests in this module if MindSpore not available
pytestmark = pytest.mark.mindspore


class TestMatMul:
    """Tests for MatMul operations."""

    def test_matmul_2d(self):
        """Test 2D MatMul."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.blas import matmul

        a = Tensor(ms.numpy.ones((2, 3)), mstype.float16)
        b = Tensor(ms.numpy.ones((3, 4)), mstype.float16)

        result = matmul(a, b)

        assert result.shape == (2, 4)

    def test_matmul_4d_attention(self):
        """Test 4D MatMul for attention."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.blas import matmul

        # (batch, heads, seq, dim)
        q = Tensor(ms.numpy.ones((2, 4, 8, 16)), mstype.float16)
        k = Tensor(ms.numpy.ones((2, 4, 16, 8)), mstype.float16)

        result = matmul(q, k)

        assert result.shape == (2, 4, 8, 8)

    def test_matmul_invalid_shapes(self):
        """Test MatMul with invalid shapes raises."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.blas import matmul

        a = Tensor(ms.numpy.ones((2, 3)), mstype.float16)
        b = Tensor(ms.numpy.ones((4, 5)), mstype.float16)  # K mismatch

        with pytest.raises(ValueError):
            matmul(a, b)

    def test_linear(self):
        """Test linear operation."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.blas import linear

        x = Tensor(ms.numpy.ones((2, 3)), mstype.float16)
        w = Tensor(ms.numpy.ones((4, 3)), mstype.float16)
        b = Tensor(ms.numpy.zeros((4,)), mstype.float16)

        result = linear(x, w, b)

        assert result.shape == (2, 4)


class TestBatchMatMul:
    """Tests for BatchMatMul operations."""

    def test_bmm_basic(self):
        """Test basic BatchMatMul."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.blas import bmm

        a = Tensor(ms.numpy.ones((2, 3, 4)), mstype.float16)
        b = Tensor(ms.numpy.ones((2, 4, 5)), mstype.float16)

        result = bmm(a, b)

        assert result.shape == (2, 3, 5)

    def test_batch_matmul_alias(self):
        """Test batch_matmul alias."""
        from daca.blas import bmm, batch_matmul

        assert bmm == batch_matmul

    def test_bmm_non_3d_raises(self):
        """Test BMM with non-3D tensor raises."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.blas import bmm

        a = Tensor(ms.numpy.ones((2, 3)), mstype.float16)
        b = Tensor(ms.numpy.ones((2, 3)), mstype.float16)

        with pytest.raises(ValueError):
            bmm(a, b)


class TestWorkspace:
    """Tests for workspace management."""

    def test_workspace_manager_create(self):
        """Test WorkspaceManager creation."""
        from daca.blas import WorkspaceManager

        manager = WorkspaceManager()
        assert manager.default_size == 256 * 1024 * 1024

    def test_workspace_manager_custom_size(self):
        """Test WorkspaceManager with custom size."""
        from daca.blas import WorkspaceManager

        manager = WorkspaceManager(default_size=128 * 1024 * 1024)
        assert manager.default_size == 128 * 1024 * 1024

    def test_estimate_workspace_size(self):
        """Test workspace size estimation."""
        from daca.blas.workspace import estimate_workspace_size

        size = estimate_workspace_size(1024, 1024, 1024)

        # Should be multiple of output size
        assert size > 0


class TestValidateShapes:
    """Tests for shape validation."""

    def test_validate_matmul_shapes(self):
        """Test MatMul shape validation."""
        from daca.blas.matmul import validate_matmul_shapes

        shape_a, shape_b = validate_matmul_shapes((2, 3), (3, 4))

        assert shape_a == (2, 3)
        assert shape_b == (3, 4)

    def test_validate_matmul_shapes_transpose(self):
        """Test MatMul shape validation with transpose."""
        from daca.blas.matmul import validate_matmul_shapes

        shape_a, shape_b = validate_matmul_shapes((2, 3), (4, 3), transpose_b=True)

        assert shape_a == (2, 3)
        assert shape_b == (3, 4)

    def test_validate_matmul_incompatible(self):
        """Test validation raises on incompatible shapes."""
        from daca.blas.matmul import validate_matmul_shapes

        with pytest.raises(ValueError):
            validate_matmul_shapes((2, 3), (4, 5))
