#!/usr/bin/env python
"""DACA Quickstart Example.

This example demonstrates basic DACA usage for running AI models
on Huawei Ascend NPUs.

Run this on a machine with Ascend NPU:
    python quickstart.py
"""

import daca


def main():
    """Run quickstart example."""
    # Display DACA info
    print("=" * 60)
    print("DACA Quickstart Example")
    print("=" * 60)

    # Show DACA information
    daca.info()

    # Apply all compatibility patches
    print("\nApplying DACA patches...")
    daca.patch()
    print("Patches applied successfully!")

    # Try importing MindSpore
    try:
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
        import mindspore.ops as ops

        print("\nMindSpore is available!")

        # Create a simple tensor
        print("\n1. Creating a simple tensor...")
        x = Tensor([1.0, 2.0, 3.0], mstype.float16)
        print(f"   x = {x}")
        print(f"   dtype = {x.dtype}")

        # Test SiLU (injected by DACA)
        print("\n2. Testing SiLU activation (injected by DACA)...")
        y = ops.silu(x)
        print(f"   silu(x) = {y}")

        # Test LayerNorm
        print("\n3. Testing LayerNorm with FP32 upcast...")
        from daca.nn import LayerNorm

        hidden_states = Tensor(ms.numpy.random.randn(2, 4, 8), mstype.float16)
        ln = LayerNorm(8)
        normalized = ln(hidden_states)
        print(f"   Input shape: {hidden_states.shape}")
        print(f"   Output shape: {normalized.shape}")

        # Test FlashAttention
        print("\n4. Testing FlashAttention...")
        from daca.nn import FlashAttention

        batch, heads, seq, dim = 2, 4, 8, 16
        q = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)
        k = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)
        v = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)

        attn = FlashAttention(head_dim=dim, num_heads=heads)
        output = attn(q, k, v)
        print(f"   Query shape: {q.shape}")
        print(f"   Output shape: {output.shape}")

        # Test MatMul
        print("\n5. Testing MatMul...")
        from daca.blas import matmul

        a = Tensor(ms.numpy.random.randn(64, 128), mstype.float16)
        b = Tensor(ms.numpy.random.randn(128, 64), mstype.float16)
        c = matmul(a, b)
        print(f"   A: {a.shape}, B: {b.shape}")
        print(f"   Result: {c.shape}")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except ImportError:
        print("\nMindSpore not available - skipping tensor operations")
        print("Install MindSpore to run the full example")

    finally:
        # Clean up
        print("\nCleaning up...")
        daca.unpatch()
        print("Patches removed. Done!")


if __name__ == "__main__":
    main()
