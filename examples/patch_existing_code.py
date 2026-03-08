#!/usr/bin/env python
"""Example: Patching Existing Code for Ascend.

This example shows how to make minimal changes to existing PyTorch-style
code to run on Ascend using DACA.

Before (PyTorch):
    import torch

    model = MyModel().cuda()
    x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    y = model(x)

After (MindSpore + DACA):
    import daca
    daca.patch()

    import mindspore as ms
    from mindspore import context

    context.set_context(device_target="Ascend")
    model = MyModel()
    x = ms.Tensor(ms.numpy.random.randn(32, 128), ms.float16)
    y = model(x)
"""

import daca


def example_before_pytorch():
    """Example of original PyTorch code.

    This won't run without PyTorch and CUDA.
    """
    code = """
# Original PyTorch code
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.norm(self.linear(x))

# Create model and move to GPU
model = MyModel(768).cuda()
model = model.half()  # Convert to FP16

# Create input
x = torch.randn(32, 128, 768, device="cuda", dtype=torch.float16)

# Forward pass
output = model(x)
"""
    print("Original PyTorch code:")
    print("-" * 40)
    print(code)


def example_after_daca():
    """Example of migrated code using DACA."""
    code = """
# Migrated code for Ascend with DACA
import daca
daca.patch()  # Apply all compatibility patches

import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from daca.nn import LayerNorm  # DACA's FP32-safe LayerNorm

class MyModel(nn.Cell):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Dense(hidden_size, hidden_size)
        self.norm = LayerNorm(hidden_size)  # Uses FP32 internally

    def construct(self, x):
        return self.norm(self.linear(x))

# Set device context (no .cuda() needed)
context.set_context(device_target="Ascend")

# Create model (already on NPU via context)
model = MyModel(768)
model = model.to_float(mstype.float16)  # Convert to FP16

# Create input
x = Tensor(ms.numpy.random.randn(32, 128, 768), mstype.float16)

# Forward pass
output = model(x)
"""
    print("Migrated code with DACA:")
    print("-" * 40)
    print(code)


def example_actual_run():
    """Actually run the migrated code."""
    print("\nRunning migrated code...")
    print("=" * 60)

    # Apply patches
    daca.patch()

    try:
        import mindspore as ms
        import mindspore.nn as nn
        from mindspore import context, Tensor
        import mindspore.common.dtype as mstype
        from daca.nn import LayerNorm

        # Set context
        context.set_context(device_target="Ascend")

        # Define model
        class MyModel(nn.Cell):
            def __init__(self, hidden_size):
                super().__init__()
                self.linear = nn.Dense(hidden_size, hidden_size)
                self.norm = LayerNorm(hidden_size)

            def construct(self, x):
                return self.norm(self.linear(x))

        # Create model
        print("\n1. Creating model...")
        model = MyModel(256)
        model = model.to_float(mstype.float16)
        print("   Model created and converted to FP16")

        # Create input
        print("\n2. Creating input tensor...")
        x = Tensor(ms.numpy.random.randn(2, 4, 256), mstype.float16)
        print(f"   Input shape: {x.shape}, dtype: {x.dtype}")

        # Forward pass
        print("\n3. Running forward pass...")
        output = model(x)
        print(f"   Output shape: {output.shape}, dtype: {output.dtype}")

        print("\n✓ Success! Model ran on Ascend NPU.")

    except ImportError:
        print("\nMindSpore not available - showing code example only")

    finally:
        daca.unpatch()


def main():
    """Run the example."""
    print("=" * 60)
    print("DACA Code Migration Example")
    print("=" * 60)

    # Show before/after
    example_before_pytorch()
    print("\n")
    example_after_daca()

    # Actually run it
    example_actual_run()

    print("\n" + "=" * 60)
    print("Key Migration Steps:")
    print("  1. Add 'import daca; daca.patch()' at the start")
    print("  2. Replace torch with mindspore")
    print("  3. Replace .cuda() with context.set_context()")
    print("  4. Replace nn.LayerNorm with daca.nn.LayerNorm")
    print("  5. Use ms.float16 instead of torch.float16 (BF16 not supported)")
    print("=" * 60)


if __name__ == "__main__":
    main()
