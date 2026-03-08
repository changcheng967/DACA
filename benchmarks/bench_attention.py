"""Attention benchmarks for DACA.

Benchmarks FlashAttention and BMM attention implementations.
"""

import time
import argparse
import json
from typing import Dict, Any

try:
    import mindspore as ms
    from mindspore import Tensor, context
    import mindspore.ops as ops
    import mindspore.common.dtype as mstype
    HAS_MS = True
except ImportError:
    HAS_MS = False


def benchmark_flash_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    warmup: int = 5,
    repeat: int = 50,
) -> Dict[str, Any]:
    """Benchmark FlashAttention."""
    if not HAS_MS:
        return {"error": "MindSpore not available"}

    # Create tensors
    q = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)
    k = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)
    v = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)

    try:
        # Try FlashAttentionScore
        flash = ops.FlashAttentionScore(scale_value=1.0 / (head_dim ** 0.5))

        # Transpose for FlashAttention
        q_t = ops.transpose(q, (0, 2, 1, 3))
        k_t = ops.transpose(k, (0, 2, 1, 3))
        v_t = ops.transpose(v, (0, 2, 1, 3))

        # Warmup
        for _ in range(warmup):
            _ = flash(q_t, k_t, v_t)

        # Measure
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = flash(q_t, k_t, v_t)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        import statistics
        return {
            "name": f"flash_attention_{batch_size}x{num_heads}x{seq_len}x{head_dim}",
            "type": "flash_attention",
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "mean_ms": statistics.mean(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        }

    except Exception as e:
        return {
            "name": f"flash_attention_{batch_size}x{num_heads}x{seq_len}x{head_dim}",
            "error": str(e),
        }


def benchmark_bmm_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    warmup: int = 5,
    repeat: int = 50,
) -> Dict[str, Any]:
    """Benchmark BMM-based attention."""
    if not HAS_MS:
        return {"error": "MindSpore not available"}

    # Create tensors
    q = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)
    k = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)
    v = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)

    scale = 1.0 / (head_dim ** 0.5)

    def attention(q, k, v):
        # Q @ K^T
        scores = ops.BatchMatMul(transpose_b=True)(q, k)
        scores = scores * scale
        # Softmax
        attn = ops.softmax(scores, axis=-1)
        # @ V
        return ops.BatchMatMul()(attn, v)

    # Warmup
    for _ in range(warmup):
        _ = attention(q, k, v)

    # Measure
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = attention(q, k, v)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    import statistics
    return {
        "name": f"bmm_attention_{batch_size}x{num_heads}x{seq_len}x{head_dim}",
        "type": "bmm_attention",
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """Run all attention benchmarks."""
    configs = [
        (1, 32, 128, 64),
        (1, 32, 512, 64),
        (1, 32, 1024, 64),
        (4, 32, 512, 64),
        (8, 32, 512, 64),
    ]

    results = {"flash_attention": [], "bmm_attention": []}

    for batch, heads, seq, dim in configs:
        # FlashAttention
        fa_result = benchmark_flash_attention(batch, heads, seq, dim)
        results["flash_attention"].append(fa_result)

        if verbose and "error" not in fa_result:
            print(f"FlashAttention {batch}x{heads}x{seq}x{dim}: {fa_result['mean_ms']:.3f} ms")

        # BMM attention
        bmm_result = benchmark_bmm_attention(batch, heads, seq, dim)
        results["bmm_attention"].append(bmm_result)

        if verbose and "error" not in bmm_result:
            print(f"BMM Attention {batch}x{heads}x{seq}x{dim}: {bmm_result['mean_ms']:.3f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="DACA Attention Benchmarks")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    if not HAS_MS:
        print("MindSpore not available")
        return

    context.set_context(device_target="Ascend")

    results = run_benchmarks(verbose=not args.quiet)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
