#!/usr/bin/env python
"""Training Example with DACA.

This example shows how to train a simple model on Ascend NPU using DACA.

Run this on a machine with Ascend NPU:
    python train_with_daca.py
"""

import daca


def create_simple_model(hidden_size: int = 256, num_layers: int = 2):
    """Create a simple transformer-like model."""
    try:
        import mindspore as ms
        import mindspore.nn as nn
        from daca.nn import LayerNorm, FlashAttention, silu

        class TransformerBlock(nn.Cell):
            """Simple transformer block."""

            def __init__(self, hidden_size: int, num_heads: int = 4):
                super().__init__()
                self.head_dim = hidden_size // num_heads
                self.num_heads = num_heads

                self.attn = FlashAttention(head_dim=self.head_dim, num_heads=num_heads)
                self.norm1 = LayerNorm(hidden_size)
                self.norm2 = LayerNorm(hidden_size)

                self.ffn_up = nn.Dense(hidden_size, hidden_size * 4)
                self.ffn_down = nn.Dense(hidden_size * 4, hidden_size)

            def construct(self, x):
                batch, seq, hidden = x.shape

                # Attention
                x_4d = x.reshape(batch, seq, self.num_heads, self.head_dim)
                x_4d = x_4d.transpose(0, 2, 1, 3)  # (batch, heads, seq, dim)

                attn_out = self.attn(x_4d, x_4d, x_4d)
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, hidden)

                x = self.norm1(x + attn_out)

                # FFN
                ffn_out = self.ffn_up(x)
                ffn_out = silu(ffn_out)
                ffn_out = self.ffn_down(ffn_out)

                x = self.norm2(x + ffn_out)

                return x

        class SimpleModel(nn.Cell):
            """Simple model for demonstration."""

            def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
                super().__init__()

                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.blocks = nn.CellList([
                    TransformerBlock(hidden_size)
                    for _ in range(num_layers)
                ])
                self.output = nn.Dense(hidden_size, vocab_size)

            def construct(self, input_ids):
                x = self.embedding(input_ids)

                for block in self.blocks:
                    x = block(x)

                logits = self.output(x)
                return logits

        return SimpleModel(
            vocab_size=1000,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

    except ImportError:
        print("MindSpore not available")
        return None


def train_step(model, loss_fn, optimizer, data, labels):
    """Single training step."""
    try:
        import mindspore as ms

        def forward_fn(data, labels):
            logits = model(data)
            loss = loss_fn(logits, labels)
            return loss, logits

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        (loss, _), grads = grad_fn(data, labels)
        optimizer(grads)

        return loss

    except ImportError:
        return None


def main():
    """Run training example."""
    print("=" * 60)
    print("DACA Training Example")
    print("=" * 60)

    # Apply DACA patches
    print("\nApplying DACA patches...")
    daca.patch()

    try:
        import mindspore as ms
        from mindspore import Tensor, context
        import mindspore.common.dtype as mstype
        import mindspore.nn as nn

        # Set context
        context.set_context(device_target="Ascend", device_id=0)

        # Create model
        print("\nCreating model...")
        model = create_simple_model(hidden_size=256, num_layers=2)
        print(f"Model created with {len(model.blocks)} layers")

        # Create optimizer
        print("\nSetting up optimizer...")
        optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4)

        # Create loss function
        loss_fn = nn.CrossEntropyLoss()

        # Generate dummy data
        print("\nGenerating training data...")
        batch_size = 4
        seq_len = 16
        vocab_size = 1000

        data = Tensor(ms.numpy.random.randint(0, vocab_size, (batch_size, seq_len)), mstype.int32)
        labels = Tensor(ms.numpy.random.randint(0, vocab_size, (batch_size, seq_len)), mstype.int32)

        # Training loop
        print("\nTraining for 5 steps...")
        model.set_train(True)

        for step in range(5):
            loss = train_step(model, loss_fn, optimizer, data, labels)
            if loss is not None:
                print(f"  Step {step + 1}: loss = {float(loss):.4f}")

        print("\nTraining completed!")

        # Test inference
        print("\nTesting inference...")
        model.set_train(False)

        test_input = Tensor(ms.numpy.random.randint(0, vocab_size, (1, seq_len)), mstype.int32)
        output = model(test_input)
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

        print("\n" + "=" * 60)
        print("Training example completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nMindSpore not available: {e}")
        print("Install MindSpore to run this example")

    finally:
        # Clean up
        print("\nCleaning up...")
        daca.unpatch()


if __name__ == "__main__":
    main()
