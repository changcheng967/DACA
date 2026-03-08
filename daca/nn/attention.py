"""Chunked Online Softmax Attention for Ascend 910ProA.

CRITICAL: FlashAttentionScore has NO backward pass on 910ProA (Atlas A2 only).
This module implements the FlashAttention algorithm in pure MindSpore ops using
chunked online softmax, which avoids materializing the full S×S attention matrix.

WHY: Naive BMM-Softmax-BMM materializes the full [B,H,S,S] attention matrix.
For Qwen3-8B (seq=4096, 32 heads): ~1GB per layer, ~36GB across 36 layers.
This exceeds the 32GB HBM on each 910ProA NPU → OOM crash.

The chunked approach processes Q and K/V in small tiles, using the online
softmax trick (Milakov & Gimelshein 2018) to combine partial softmax results
without ever storing the full attention matrix.

Memory: O(q_chunk × kv_chunk) instead of O(S²).
With chunks of 256: ~4MB vs ~1GB per layer.

Algorithm:
    For each q_block:
        Initialize: m = -inf (running max), l = 0 (running sum), o = 0 (running output)
        For each kv_block:
            s = q_block @ k_block.T * scale
            m_new = max(m, rowmax(s))
            p = exp(s - m_new)  # numerically stable
            l_new = exp(m - m_new) * l + rowsum(p)
            o = diag(exp(m - m_new)) * (l / l_new) * o + diag(1/l_new) * p @ v_block
            m, l = m_new, l_new
        output[q_block_range] = o

Reference: FlashAttention (Dao et al. 2022), FastAttention (Milakov & Gimelshein 2018)

Example:
    from daca.nn import DaCAAttention

    attn = DaCAAttention(num_heads=32, num_kv_heads=8, head_dim=128)
    output = attn(query, key, value)  # No OOM, supports backward
"""

import logging
import math
from typing import Optional, Tuple, Any, Union

logger = logging.getLogger("daca.nn.attention")

# Preferred input layout: [B, H, S, D] (batch, heads, sequence, dimension)
# This matches MindSpore's BatchMatMul convention and is most efficient.


class DaCAAttention:
    """Chunked Online Softmax Attention (FlashAttention-equivalent, pure MindSpore).

    Implements the FlashAttention algorithm using only MindSpore ops that have
    backward support on 910ProA. No custom kernels, no sudo required.

    Key features:
    - Memory efficient: O(chunk²) instead of O(seq²)
    - Backward support: All ops have gradients on 910ProA
    - GQA support: Handles different num_heads vs num_kv_heads
    - Causal masking: Per-chunk causal mask with block skip optimization
    - Numerical stability: Online softmax with fp32 upcast

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads (for GQA).
        head_dim: Dimension of each attention head.
        q_chunk_size: Chunk size for query dimension.
        kv_chunk_size: Chunk size for key/value dimension.
        causal: Whether to apply causal (autoregressive) mask.
        dropout_rate: Dropout probability (applied after softmax).
        scale: Scaling factor for attention scores (default: 1/sqrt(head_dim)).

    Example:
        >>> # Standard attention (no GQA)
        >>> attn = DaCAAttention(num_heads=32, num_kv_heads=32, head_dim=128)
        >>> q = ms.Tensor((batch, 32, seq, 128), ms.float16)
        >>> k = ms.Tensor((batch, 32, seq, 128), ms.float16)
        >>> v = ms.Tensor((batch, 32, seq, 128), ms.float16)
        >>> out = attn(q, k, v)  # [B, 32, S, 128]
        >>>
        >>> # GQA (Grouped Query Attention) - like LLaMA
        >>> attn = DaCAAttention(num_heads=32, num_kv_heads=8, head_dim=128)
        >>> # KV heads are automatically repeated to match Q heads
        >>> out = attn(q, k, v)
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_chunk_size: int = 256,
        kv_chunk_size: int = 256,
        causal: bool = True,
        dropout_rate: float = 0.0,
        scale: Optional[float] = None,
    ):
        """Initialize DaCAAttention.

        Args:
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads. If different from num_heads,
                          enables GQA (Grouped Query Attention).
            head_dim: Dimension of each attention head.
            q_chunk_size: Chunk size for query dimension. Default 256 fits well
                          in DaVinci Cube L1 buffer.
            kv_chunk_size: Chunk size for key/value dimension. Default 256.
            causal: Whether to apply causal masking for autoregressive models.
            dropout_rate: Dropout probability applied after softmax.
            scale: Scale factor for attention scores. Default: 1/sqrt(head_dim).

        Raises:
            ValueError: If num_heads is not divisible by num_kv_heads.
        """
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads}) for GQA"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_chunk_size = q_chunk_size
        self.kv_chunk_size = kv_chunk_size
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.n_rep = num_heads // num_kv_heads  # GQA repeat factor

        logger.debug(
            f"DaCAAttention: heads={num_heads}, kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, q_chunk={q_chunk_size}, kv_chunk={kv_chunk_size}, "
            f"causal={causal}, scale={self.scale:.6f}"
        )

    def construct(
        self,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Optional[Any] = None,
    ) -> Any:
        """Compute chunked online softmax attention.

        Args:
            query: Query tensor of shape [B, H_q, S, D] or [B, S, H_q, D].
            key: Key tensor of shape [B, H_kv, S, D] or [B, S, H_kv, D].
            value: Value tensor of shape [B, H_kv, S, D] or [B, S, H_kv, D].
            attention_mask: Optional attention mask (not yet implemented).

        Returns:
            Output tensor with same shape/layout as input query.

        Note:
            Preferred input layout is [B, H, S, D] for efficiency.
            [B, S, H, D] is accepted but will be transposed internally.
        """
        try:
            import mindspore as ms
            import mindspore.ops as ops
            from mindspore import Tensor
            import mindspore.common.dtype as mstype
        except ImportError:
            raise ImportError("MindSpore is required for DaCAAttention")

        # Detect and normalize input layout
        q_shape = query.shape
        k_shape = key.shape
        v_shape = value.shape

        # Determine layout: [B, H, S, D] vs [B, S, H, D]
        # In [B, H, S, D]: dim 1 is num_heads (small), dim 2 is seq (large)
        # In [B, S, H, D]: dim 1 is seq (large), dim 2 is num_heads (small)
        if len(q_shape) == 4:
            if q_shape[1] == self.num_heads:
                # Layout: [B, H, S, D] - preferred
                input_layout = "BHSD"
                batch_size = q_shape[0]
                seq_len = q_shape[2]
            elif q_shape[2] == self.num_heads:
                # Layout: [B, S, H, D] - need transpose
                input_layout = "BSHD"
                batch_size = q_shape[0]
                seq_len = q_shape[1]
                # Transpose to [B, H, S, D]
                query = ops.transpose(query, (0, 2, 1, 3))
                key = ops.transpose(key, (0, 2, 1, 3))
                value = ops.transpose(value, (0, 2, 1, 3))
            else:
                raise ValueError(
                    f"Cannot determine input layout. Query shape: {q_shape}, "
                    f"expected num_heads={self.num_heads} at position 1 or 2"
                )
        else:
            raise ValueError(f"Expected 4D input tensors, got query shape: {q_shape}")

        # Handle GQA: repeat KV heads to match Q heads
        if self.n_rep > 1:
            key = self._repeat_kv(key, self.n_rep)
            value = self._repeat_kv(value, self.n_rep)

        # Compute attention using chunked online softmax
        output = self._chunked_attention(query, key, value, batch_size, seq_len)

        # Transpose back if input was [B, S, H, D]
        if input_layout == "BSHD":
            output = ops.transpose(output, (0, 2, 1, 3))

        return output

    def __call__(
        self,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Call construct method."""
        return self.construct(query, key, value, attention_mask, **kwargs)

    def _repeat_kv(self, hidden_states: Any, n_rep: int) -> Any:
        """Repeat key/value heads for GQA.

        Args:
            hidden_states: [B, H_kv, S, D]
            n_rep: Number of times to repeat each head

        Returns:
            Tensor of shape [B, H_kv * n_rep, S, D]
        """
        if n_rep == 1:
            return hidden_states

        try:
            import mindspore.ops as ops
        except ImportError:
            raise ImportError("MindSpore is required")

        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

        # [B, H_kv, S, D] -> [B, H_kv, 1, S, D]
        hidden_states = ops.expand_dims(hidden_states, 2)

        # [B, H_kv, 1, S, D] -> [B, H_kv, n_rep, S, D]
        hidden_states = ops.tile(hidden_states, (1, 1, n_rep, 1, 1))

        # [B, H_kv, n_rep, S, D] -> [B, H_kv * n_rep, S, D]
        hidden_states = ops.reshape(
            hidden_states,
            (batch, num_kv_heads * n_rep, seq_len, head_dim)
        )

        return hidden_states

    def _chunked_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        batch_size: int,
        seq_len: int,
    ) -> Any:
        """Compute attention using chunked online softmax.

        This is the core FlashAttention algorithm implemented in pure MindSpore.
        Processes Q in chunks of q_chunk_size, K/V in chunks of kv_chunk_size.

        Args:
            query: [B, H_q, S, D]
            key: [B, H_q, S, D] (already repeated for GQA)
            value: [B, H_q, S, D] (already repeated for GQA)
            batch_size: Batch size B
            seq_len: Sequence length S

        Returns:
            Output tensor [B, H_q, S, D]
        """
        try:
            import mindspore as ms
            import mindspore.ops as ops
            from mindspore import Tensor
            import mindspore.common.dtype as mstype
            import numpy as np
        except ImportError:
            raise ImportError("MindSpore is required for chunked attention")

        num_heads = self.num_heads
        head_dim = self.head_dim
        q_chunk = self.q_chunk_size
        kv_chunk = self.kv_chunk_size
        scale = self.scale
        causal = self.causal

        # Number of chunks
        num_q_chunks = (seq_len + q_chunk - 1) // q_chunk
        num_kv_chunks = (seq_len + kv_chunk - 1) // kv_chunk

        # Initialize output tensor
        output = ops.zeros((batch_size, num_heads, seq_len, head_dim), query.dtype)

        # Process each query chunk
        for i in range(num_q_chunks):
            q_start = i * q_chunk
            q_end = min((i + 1) * q_chunk, seq_len)
            actual_q_chunk = q_end - q_start

            # Extract query chunk: [B, H, q_chunk, D]
            q_chunk_tensor = query[:, :, q_start:q_end, :]

            # Initialize running statistics for this query chunk
            # m: running max for numerical stability [B, H, q_chunk, 1]
            # l: running sum of exp(scores - max) [B, H, q_chunk, 1]
            # o: running output [B, H, q_chunk, D]
            m = ops.fill(mstype.float32, (batch_size, num_heads, actual_q_chunk, 1), float("-inf"))
            l = ops.zeros((batch_size, num_heads, actual_q_chunk, 1), mstype.float32)
            o = ops.zeros((batch_size, num_heads, actual_q_chunk, head_dim), mstype.float32)

            # Process each key/value chunk
            for j in range(num_kv_chunks):
                kv_start = j * kv_chunk
                kv_end = min((j + 1) * kv_chunk, seq_len)
                actual_kv_chunk = kv_end - kv_start

                # Causal mask optimization: skip fully masked blocks
                if causal:
                    # If kv_block starts after q_block ends, entire block is masked
                    if kv_start > q_end - 1:
                        continue  # Skip this KV block entirely

                # Extract key/value chunks: [B, H, kv_chunk, D]
                k_chunk_tensor = key[:, :, kv_start:kv_end, :]
                v_chunk_tensor = value[:, :, kv_start:kv_end, :]

                # Compute attention scores for this block
                # q_chunk: [B, H, q_chunk, D]
                # k_chunk: [B, H, kv_chunk, D]
                # scores: [B, H, q_chunk, kv_chunk]
                # Use BatchMatMul for efficiency
                scores = ops.BatchMatMul(transpose_b=True)(q_chunk_tensor, k_chunk_tensor)

                # Apply scale
                scores = scores * scale

                # Apply causal mask if needed
                if causal:
                    # Only apply mask if this block is partially masked
                    if kv_end > q_start:
                        mask = self._create_causal_mask_block(
                            q_start, q_end, kv_start, kv_end, scores.dtype
                        )
                        scores = scores + mask

                # Online softmax update
                # Upcast to fp32 for numerical stability
                scores_fp32 = ops.cast(scores, mstype.float32)

                # Compute new max: m_new = max(m, rowmax(scores))
                m_block = ops.max(scores_fp32, axis=-1, keep_dims=True)[0]  # [B, H, q_chunk, 1]
                m_new = ops.maximum(m, m_block)

                # Compute exp(scores - m_new)
                exp_scores = ops.exp(scores_fp32 - m_new)

                # Apply dropout if enabled (during training)
                if self.dropout_rate > 0:
                    exp_scores = ops.dropout(exp_scores, p=self.dropout_rate)

                # Update running sum: l_new = exp(m - m_new) * l + rowsum(exp_scores)
                l_scale = ops.exp(m - m_new)  # [B, H, q_chunk, 1]
                l_new = l_scale * l + ops.sum(exp_scores, axis=-1, keep_dims=True)

                # Compute attention-weighted values for this block
                # exp_scores: [B, H, q_chunk, kv_chunk]
                # v_chunk: [B, H, kv_chunk, D]
                # block_output: [B, H, q_chunk, D]
                block_output = ops.BatchMatMul()(exp_scores, ops.cast(v_chunk_tensor, mstype.float32))

                # Update running output
                # o = (l_scale * l / l_new) * o + (1 / l_new) * block_output
                # This is the online softmax rescaling formula
                o_scale = l_scale * l / (l_new + 1e-10)  # Add epsilon for stability
                o = o_scale * o + block_output / (l_new + 1e-10)

                # Update running statistics
                m = m_new
                l = l_new

            # Final output for this query chunk (already normalized by l)
            # Cast back to original dtype
            output[:, :, q_start:q_end, :] = ops.cast(o, query.dtype)

        return output

    def _create_causal_mask_block(
        self,
        q_start: int,
        q_end: int,
        kv_start: int,
        kv_end: int,
        dtype: Any,
    ) -> Any:
        """Create causal mask for a single attention block.

        For causal attention, position i can only attend to positions <= i.

        Args:
            q_start: Start index of query positions
            q_end: End index of query positions (exclusive)
            kv_start: Start index of key/value positions
            kv_end: End index of key/value positions (exclusive)
            dtype: Data type for the mask tensor

        Returns:
            Mask tensor of shape [1, 1, q_chunk, kv_chunk]
            -inf where attention should be masked, 0 elsewhere
        """
        try:
            import mindspore as ms
            from mindspore import Tensor
            import mindspore.common.dtype as mstype
            import numpy as np
        except ImportError:
            raise ImportError("MindSpore is required")

        q_len = q_end - q_start
        kv_len = kv_end - kv_start

        # Create position indices
        q_positions = np.arange(q_start, q_end).reshape(-1, 1)  # [q_len, 1]
        kv_positions = np.arange(kv_start, kv_end).reshape(1, -1)  # [1, kv_len]

        # Causal mask: mask[i, j] = -inf if j > i, else 0
        # Position j can be attended by position i only if j <= i
        mask = np.where(kv_positions > q_positions, float("-inf"), 0.0)
        mask = mask.astype(np.float32)

        # Expand to [1, 1, q_len, kv_len] for broadcasting
        mask = mask.reshape(1, 1, q_len, kv_len)

        return Tensor(mask, dtype=mstype.float32)

    def forward(self, query: Any, key: Any, value: Any, **kwargs) -> Any:
        """Alias for construct() for compatibility."""
        return self.construct(query, key, value, **kwargs)


# Keep FlashAttention as an alias for backward compatibility
# Note: This is NOT the native FlashAttentionScore - it's the chunked implementation
FlashAttention = DaCAAttention


def scaled_dot_product_attention(
    query: Any,
    key: Any,
    value: Any,
    scale: Optional[float] = None,
    mask: Optional[Any] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Any:
    """Compute scaled dot-product attention using chunked online softmax.

    Functional interface that creates a temporary DaCAAttention instance.

    Args:
        query: Query tensor [B, H, S, D] or [B, S, H, D].
        key: Key tensor [B, H, S, D] or [B, S, H, D].
        value: Value tensor [B, H, S, D] or [B, S, H, D].
        scale: Scale factor. Default: 1/sqrt(head_dim).
        mask: Optional attention mask (not yet supported).
        dropout_p: Dropout probability.
        is_causal: Whether to apply causal mask.

    Returns:
        Attention output with same shape as query.

    Example:
        >>> from daca.nn import scaled_dot_product_attention
        >>> out = scaled_dot_product_attention(q, k, v, is_causal=True)
    """
    # Infer dimensions from input
    q_shape = query.shape
    if len(q_shape) != 4:
        raise ValueError(f"Expected 4D input, got shape {q_shape}")

    # Detect layout
    if q_shape[1] < q_shape[2]:
        # [B, H, S, D] - H is smaller
        num_heads = q_shape[1]
        head_dim = q_shape[3]
    else:
        # [B, S, H, D] - S is smaller or equal
        num_heads = q_shape[2]
        head_dim = q_shape[3]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Create temporary attention module
    attn = DaCAAttention(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        causal=is_causal,
        dropout_rate=dropout_p,
        scale=scale,
    )

    return attn(query, key, value, attention_mask=mask)


def repeat_kv(hidden_states: Any, n_rep: int) -> Any:
    """Repeat key/value tensors for Grouped Query Attention.

    In GQA, the number of key/value heads may be less than query heads.
    This function repeats key/value heads to match query head count.

    Args:
        hidden_states: Tensor of shape (batch, num_kv_heads, seq, head_dim).
        n_rep: Number of times to repeat (num_q_heads / num_kv_heads).

    Returns:
        Tensor of shape (batch, num_q_heads, seq, head_dim).

    Example:
        >>> # For 32 query heads, 8 key/value heads (GQA-8)
        >>> k_expanded = repeat_kv(k, n_rep=4)
    """
    if n_rep == 1:
        return hidden_states

    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for repeat_kv")

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

    # (batch, num_kv_heads, seq, head_dim) -> (batch, num_kv_heads, 1, seq, head_dim)
    hidden_states = ops.expand_dims(hidden_states, 2)

    # (batch, num_kv_heads, 1, seq, head_dim) -> (batch, num_kv_heads, n_rep, seq, head_dim)
    hidden_states = ops.tile(hidden_states, (1, 1, n_rep, 1, 1))

    # (batch, num_kv_heads, n_rep, seq, head_dim) -> (batch, num_kv_heads * n_rep, seq, head_dim)
    hidden_states = ops.reshape(
        hidden_states,
        (batch, num_kv_heads * n_rep, seq_len, head_dim)
    )

    return hidden_states
