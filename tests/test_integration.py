"""Integration tests for DACA.

Tests end-to-end functionality combining multiple modules.
"""

import pytest

# Skip all tests in this module if MindSpore not available
pytestmark = pytest.mark.mindspore


class TestPatchUnpatch:
    """Tests for patch/unpatch cycle."""

    def test_patch_unpatch_cycle(self):
        """Test complete patch/unpatch cycle."""
        import daca

        # Patch
        daca.patch()
        assert daca.is_patched() is True

        # Unpatch
        daca.unpatch()
        assert daca.is_patched() is False

    def test_patch_twice_warning(self):
        """Test that patching twice logs warning."""
        import daca

        daca.patch()
        daca.patch()  # Should warn but not crash

        daca.unpatch()

    def test_unpatch_when_not_patched(self):
        """Test unpatch when not patched."""
        import daca

        # Should not crash
        if daca.is_patched():
            daca.unpatch()

        daca.unpatch()  # Should be safe


class TestTransformerBlock:
    """Tests for transformer block components."""

    def test_attention_with_rope(self):
        """Test attention with rotary embeddings."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.nn import FlashAttention, RotaryEmbedding, apply_rotary_pos_emb

        batch, heads, seq, dim = 2, 4, 8, 16

        q = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)
        k = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)
        v = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)

        # Apply RoPE
        rotary = RotaryEmbedding(dim=dim, max_seq_len=seq)
        cos, sin = rotary(seq)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Attention
        attn = FlashAttention(head_dim=dim, num_heads=heads)
        output = attn(q, k, v)

        assert output.shape == (batch, heads, seq, dim)

    def test_mlp_with_swiglu(self):
        """Test MLP with SwiGLU activation."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.nn import swiglu
        from daca.blas import linear

        batch, seq, hidden = 2, 8, 32
        intermediate = hidden * 4

        x = Tensor(ms.numpy.random.randn(batch, seq, hidden), mstype.float16)

        # Project up
        w1 = Tensor(ms.numpy.random.randn(intermediate, hidden), mstype.float16)
        x_up = linear(x, w1)

        # SwiGLU
        x_act = swiglu(x_up, dim=-1)

        # Should have shape (batch, seq, intermediate // 2)
        assert x_act.shape == (batch, seq, intermediate // 2)


class TestMemoryTracking:
    """Tests for memory tracking integration."""

    def test_memory_tracker_with_operations(self):
        """Test memory tracker with actual operations."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        from daca.runtime import MemoryTracker

        tracker = MemoryTracker()

        with tracker.track("allocation"):
            x = Tensor(ms.numpy.random.randn(1000, 1000), mstype.float16)

        summary = tracker.summary()

        assert summary["total_records"] >= 2  # start and end


class TestBenchmarking:
    """Tests for benchmarking integration."""

    def test_benchmark_op_basic(self):
        """Test basic operation benchmarking."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        import mindspore.ops as ops
        from daca.autotune import benchmark_op

        x = Tensor(ms.numpy.random.randn(100, 100), mstype.float16)
        y = Tensor(ms.numpy.random.randn(100, 100), mstype.float16)

        result = benchmark_op(
            lambda: ops.matmul(x, y),
            name="test_matmul",
            warmup=2,
            repeat=5,
        )

        assert result.name == "test_matmul"
        assert result.mean_ms > 0
        assert result.repeat_runs == 5


class TestFullModel:
    """Tests for complete model functionality."""

    def test_simple_transformer_forward(self):
        """Test forward pass of a simple transformer."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        import mindspore.nn as nn
        from daca.nn import LayerNorm, FlashAttention, silu

        class SimpleBlock(nn.Cell):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.attn = FlashAttention(head_dim=hidden_size // num_heads, num_heads=num_heads)
                self.norm1 = LayerNorm(hidden_size)
                self.norm2 = LayerNorm(hidden_size)
                self.ffn_up = nn.Dense(hidden_size, hidden_size * 4)
                self.ffn_down = nn.Dense(hidden_size * 4, hidden_size)

            def construct(self, x):
                # Attention
                batch, seq, hidden = x.shape
                heads = 4

                x_reshaped = x.reshape(batch, seq, heads, -1).transpose(0, 2, 1, 3)
                attn_out = self.attn(x_reshaped, x_reshaped, x_reshaped)
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, hidden)

                x = self.norm1(x + attn_out)

                # FFN
                ffn_out = self.ffn_up(x)
                ffn_out = silu(ffn_out)
                ffn_out = self.ffn_down(ffn_out)

                x = self.norm2(x + ffn_out)

                return x

        batch, seq, hidden = 2, 8, 64
        x = Tensor(ms.numpy.random.randn(batch, seq, hidden), mstype.float16)

        block = SimpleBlock(hidden, num_heads=4)
        output = block(x)

        assert output.shape == x.shape


class TestCompatibility:
    """Tests for compatibility features."""

    def test_bf16_conversion(self, daca_patched):
        """Test automatic BF16 to FP16 conversion."""
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype

        # Request BF16
        x = Tensor([1.0, 2.0, 3.0], mstype.bfloat16)

        # Should actually be FP16
        assert x.dtype == mstype.float16

    def test_silu_injection(self, daca_patched):
        """Test SiLU injection."""
        import mindspore.ops as ops

        # Should have silu after patch
        assert hasattr(ops, "silu")

        # Should work
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype

        x = Tensor([1.0, 2.0, 3.0], mstype.float16)
        y = ops.silu(x)

        assert y.shape == x.shape


# Fixtures
@pytest.fixture
def daca_patched():
    """Apply and revert DACA patches."""
    import daca
    daca.patch()
    yield
    daca.unpatch()
