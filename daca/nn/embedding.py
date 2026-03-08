"""Embedding layer for Ascend NPUs.

Wraps MindSpore's embedding with padding index handling and
proper initialization.

Example:
    from daca.nn import Embedding

    emb = Embedding(vocab_size=32000, embedding_dim=4096)
    embedded = emb(token_ids)
"""

import logging
from typing import Optional, Any

logger = logging.getLogger("daca.nn.embedding")


class Embedding:
    """Embedding layer with Ascend optimizations.

    Maps token indices to dense vectors using a learnable embedding table.

    Attributes:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of the embeddings.
        padding_idx: Optional index to zero out.
        weight: Learnable embedding table.

    Example:
        >>> emb = Embedding(32000, 4096)
        >>> x = emb(token_ids)  # (batch, seq) -> (batch, seq, 4096)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        dtype: Any = None,
        scale_by_sqrt_dim: bool = False,
    ):
        """Initialize Embedding.

        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of the embeddings.
            padding_idx: Index to zero out (for padding tokens).
            dtype: Data type for the embedding table.
            scale_by_sqrt_dim: Whether to scale outputs by sqrt(dim).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.scale_by_sqrt_dim = scale_by_sqrt_dim

        # Initialize embedding table
        try:
            import mindspore as ms
            from mindspore import Parameter, Tensor
            import mindspore.common.dtype as mstype
            import mindspore.ops as ops
            import numpy as np

            param_dtype = dtype if dtype is not None else mstype.float32

            # Initialize with normal distribution
            # std = 1.0 / sqrt(embedding_dim) is common
            std = 1.0 / np.sqrt(embedding_dim)
            weight = np.random.normal(0, std, (vocab_size, embedding_dim)).astype(np.float32)

            self.weight = Parameter(
                Tensor(weight, param_dtype),
                name="weight"
            )

            # Zero out padding_idx row
            if padding_idx is not None:
                self._zero_padding_row()

        except ImportError:
            self.weight = None

    def _zero_padding_row(self) -> None:
        """Zero out the padding index row."""
        if self.padding_idx is None or self.weight is None:
            return

        try:
            import mindspore.ops as ops
            import mindspore as ms

            # Create mask for padding row
            mask = ops.ones((self.vocab_size, self.embedding_dim), self.weight.dtype)
            mask[self.padding_idx, :] = 0

            self.weight = self.weight * mask

        except Exception as e:
            logger.warning(f"Failed to zero padding row: {e}")

    def construct(self, input_ids: Any) -> Any:
        """Look up embeddings for token indices.

        Args:
            input_ids: Token indices of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim).
        """
        try:
            import mindspore.ops as ops
        except ImportError:
            raise ImportError("MindSpore is required for Embedding")

        # Use Gather for embedding lookup
        # input_ids: (batch, seq) -> (batch, seq, dim)
        embeddings = ops.gather(self.weight, input_ids, 0)

        # Scale by sqrt(dim) if configured
        if self.scale_by_sqrt_dim:
            import numpy as np
            embeddings = embeddings * np.sqrt(self.embedding_dim)

        return embeddings

    def __call__(self, input_ids: Any) -> Any:
        """Call construct method."""
        return self.construct(input_ids)


def embedding(
    input_ids: Any,
    weight: Any,
    padding_idx: Optional[int] = None,
) -> Any:
    """Functional embedding lookup.

    Args:
        input_ids: Token indices.
        weight: Embedding table.
        padding_idx: Optional padding index to zero out.

    Returns:
        Embedded tensor.

    Example:
        >>> from daca.nn.embedding import embedding
        >>> emb = embedding(token_ids, weight_table)
    """
    try:
        import mindspore.ops as ops
    except ImportError:
        raise ImportError("MindSpore is required for embedding")

    # Gather from weight table
    output = ops.gather(weight, input_ids, 0)

    # Zero out padding if specified
    if padding_idx is not None:
        import numpy as np
        # Create mask
        mask = (input_ids != padding_idx).astype(output.dtype)
        mask = ops.expand_dims(mask, -1)
        output = output * mask

    return output
