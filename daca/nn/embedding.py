"""Embedding layer for Ascend NPUs.

Wraps MindSpore's embedding with padding index handling and
proper initialization.

Example:
    from daca.nn import Embedding

    emb = Embedding(vocab_size=32000, embedding_dim=4096)
    embedded = emb(token_ids)
"""

import logging
import math
from typing import Optional, Any

logger = logging.getLogger("daca.nn.embedding")

try:
    import mindspore.nn as nn
    _HAS_NN = True
except ImportError:
    _HAS_NN = False
    # Create a dummy base class for when MindSpore is not available
    class nn:
        class Cell:
            """Dummy Cell class."""
            def __init__(self):
                pass
            def construct(self, *args, **kwargs):
                raise NotImplementedError("MindSpore is required")


class Embedding(nn.Cell):
    """Embedding layer with Ascend optimizations.

    Maps token indices to dense vectors using a learnable embedding table.
    Inherits from mindspore.nn.Cell for proper autograd support.

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
        super().__init__()

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
            weight_data = np.random.normal(0, std, (vocab_size, embedding_dim)).astype(np.float32)

            self.weight = Parameter(
                Tensor(weight_data, param_dtype),
                name="weight"
            )

            # Store padding_idx for masking during forward pass
            # (zeroing at init doesn't persist through training)

        except ImportError:
            self.weight = None

    def construct(self, input_ids: Any) -> Any:
        """Look up embeddings for token indices.

        Args:
            input_ids: Token indices of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim).
        """
        try:
            import mindspore.ops as ops
            import mindspore.common.dtype as mstype
        except ImportError:
            raise ImportError("MindSpore is required for Embedding")

        # Use Gather for embedding lookup
        # input_ids: (batch, seq) -> (batch, seq, dim)
        embeddings = ops.gather(self.weight, input_ids, 0)

        # Zero out padding_idx if specified
        if self.padding_idx is not None:
            # Create mask: 0 at padding_idx, 1 elsewhere
            mask = ops.ones((self.vocab_size, self.embedding_dim), self.weight.dtype)
            mask = ops.tensor_scatter_update(
                mask,
                ops.expand_dims(ops.scalar_to_tensor(self.padding_idx, mstype.int32), 0),
                ops.zeros((1, self.embedding_dim), self.weight.dtype)
            )
            # Apply mask (this zeros out the embedding at padding_idx for all positions)
            embeddings = embeddings * ops.gather(mask, input_ids, 0)

        # Scale by sqrt(dim) if configured
        if self.scale_by_sqrt_dim:
            embeddings = embeddings * math.sqrt(self.embedding_dim)

        return embeddings


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
        # Create mask
        mask = (input_ids != padding_idx)
        mask = ops.cast(mask, output.dtype)
        mask = ops.expand_dims(mask, -1)
        output = output * mask

    return output
