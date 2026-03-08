"""Attention benchmarks for DACA.

Benchmarks chunked (DaCAAttention) vs naive BMM attention implementations.
Shows time, memory, and TFLOPS comparison.

NOTE: Native FlashAttentionScore is NOT benchmarked because it has NO
backward pass on 910ProA (Atlas A2 only). DaCAAttention is the
FlashAttention-equivalent implementation that supports training.
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


def estimate_memory_attention(seq_len: int, num_heads: int, head_dim: int, batch: int = 1) -> Dict[str, float]:
    """Estimate memory usage for attention.

    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        batch: Batch size

    Returns:
        Dictionary with memory estimates in MB
    """
    # Naive attention materializes full S×S attention matrix
    # [B, H, S, S] in fp16 = 2 bytes
    naive_attention_matrix = batch * num_heads * seq_len * seq_len * 2  # bytes
    naive_mb = naive_attention_matrix / (1024 * 1024)

    # Chunked attention only materializes chunk×chunk
    # [B, H, chunk, chunk] where chunk=256
    chunk_size = 256
    chunked_attention_matrix = batch * num_heads * chunk_size * chunk_size * 2  # bytes
    chunked_mb = chunked_attention_matrix / (1024 * 1024)

    return {
        "naive_mb": naive_mb,
        "chunked_mb": chunked_mb,
        "savings_ratio": naive_mb / chunked_mb if chunked_mb > 0 else 0,
    }


def benchmark_chunked_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    q_chunk_size: int = 256,
    kv_chunk_size: int = 256,
    causal: bool = True,
    warmup: int = 5,
    repeat: int = 50,
) -> Dict[str, Any]:
    """Benchmark chunked online softmax attention (DaCAAttention)."""
    if not HAS_MS:
        return {"error": "MindSpore not available"}

    from daca.nn import DaCAAttention

    # Create tensors
    q = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)
    k = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)
    v = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)

    try:
        attn = DaCAAttention(
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_chunk_size=q_chunk_size,
            kv_chunk_size=kv_chunk_size,
            causal=causal,
        )

        # Warmup
        for _ in range(warmup):
            _ = attn(q, k, v)

        # Measure
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = attn(q, k, v)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        import statistics

        # Estimate memory
        mem_estimate = estimate_memory_attention(seq_len, num_heads, head_dim, batch_size)

        # Calculate TFLOPS
        # Attention FLOPs: ~4 * B * H * S^2 * D (2 for QK^T, 2 for AV)
        flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
        tflops = (flops / (statistics.mean(times) / 1000)) / 1e12 if statistics.mean(times) > 0 else 0

        return {
            "name": f"chunked_attention_{batch_size}x{num_heads}x{seq_len}x{head_dim}",
            "type": "chunked_attention",
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "q_chunk_size": q_chunk_size,
            "kv_chunk_size": kv_chunk_size,
            "causal": causal,
            "mean_ms": statistics.mean(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
            "tflops": tflops,
            "estimated_memory_mb": mem_estimate["chunked_mb"],
        }

    except Exception as e:
        return {
            "name": f"chunked_attention_{batch_size}x{num_heads}x{seq_len}x{head_dim}",
            "error": str(e),
        }


def benchmark_naive_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    warmup: int = 5,
    repeat: int = 50,
) -> Dict[str, Any]:
    """Benchmark naive BMM-Softmax-BMM attention.

    WARNING: This will OOM for large seq_len (>1024 typically).
    """
    if not HAS_MS:
        return {"error": "MindSpore not available"}

    # Check if this will likely OOM
    mem_estimate = estimate_memory_attention(seq_len, num_heads, head_dim, batch_size)
    if mem_estimate["naive_mb"] > 2048:  # > 2GB for just the attention matrix
        return {
            "name": f"naive_attention_{batch_size}x{num_heads}x{seq_len}x{head_dim}",
            "error": f"Skipped: would require {mem_estimate['naive_mb']:.0f}MB for attention matrix alone (likely OOM)",
            "estimated_memory_mb": mem_estimate["naive_mb"],
        }

    # Create tensors
    q = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)
    k = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)
    v = Tensor(ms.numpy.random.randn(batch_size, num_heads, seq_len, head_dim), mstype.float16)

    scale = 1.0 / (head_dim ** 0.5)

    def attention(q, k, v):
        # Q @ K^T
        scores = ops.BatchMatMul(transpose_b=True)(q, k)
        scores = scores * scale
        # Softmax (with fp32 upcast for stability)
        scores_fp32 = ops.cast(scores, mstype.float32)
        attn = ops.softmax(scores_fp32, axis=-1)
        attn = ops.cast(attn, mstype.float16)
        # @ V
        return ops.BatchMatMul()(attn, v)

    try:
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

        # Calculate TFLOPS
        flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
        tflops = (flops / (statistics.mean(times) / 1000)) / 1e12 if statistics.mean(times) > 0 else 0

        return {
            "name": f"naive_attention_{batch_size}x{num_heads}x{seq_len}x{head_dim}",
            "type": "naive_attention",
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "mean_ms": statistics.mean(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
            "tflops": tflops,
            "estimated_memory_mb": mem_estimate["naive_mb"],
        }

    except Exception as e:
        return {
            "name": f"naive_attention_{batch_size}x{num_heads}x{seq_len}x{head_dim}",
            "error": str(e),
            "estimated_memory_mb": mem_estimate["naive_mb"],
        }


def run_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """Run all attention benchmarks."""
    configs = [
        # (batch, heads, seq, dim) - start with smaller configs
        (1, 4, 128, 64),    # Small
        (1, 8, 256, 64),    # Small-medium
        (1, 16, 512, 64),   # Medium
        (1, 32, 1024, 64),  # Large (naive will be slow but work)
        (1, 32, 2048, 128), # Very large (naive might OOM)
        (1, 32, 4096, 128), # Qwen3-8B size (naive WILL OOM)
    ]

    results = {"chunked_attention": [], "naive_attention": [], "memory_comparison": []}

    for batch, heads, seq, dim in configs:
        mem = estimate_memory_attention(seq, heads, dim, batch)

        if verbose:
            print(f"\n=== Config: batch={batch}, heads={heads}, seq={seq}, dim={dim} ===")
            print(f"Estimated memory - Naive: {mem['naive_mb']:.1f}MB, Chunked: {mem['chunked_mb']:.1f}MB (savings: {mem['savings_ratio']:.0f}x)")

        # Chunked attention (should always work)
        chunked_result = benchmark_chunked_attention(batch, heads, seq, dim)
        results["chunked_attention"].append(chunked_result)

        if verbose and "error" not in chunked_result:
            print(f"Chunked Attention: {chunked_result['mean_ms']:.3f} ms, {chunked_result['tflops']:.2f} TFLOPS")
        elif verbose:
            print(f"Chunked Attention: ERROR - {chunked_result.get('error', 'unknown')}")

        # Naive attention (may OOM for large seq)
        naive_result = benchmark_naive_attention(batch, heads, seq, dim)
        results["naive_attention"].append(naive_result)

        if verbose and "error" not in naive_result:
            print(f"Naive Attention:   {naive_result['mean_ms']:.3f} ms, {naive_result['tflops']:.2f} TFLOPS")
        elif verbose:
            print(f"Naive Attention:   ERROR - {naive_result.get('error', 'unknown')}")

        results["memory_comparison"].append({
            "config": f"{batch}x{heads}x{seq}x{dim}",
            "naive_mb": mem["naive_mb"],
            "chunked_mb": mem["chunked_mb"],
            "savings_ratio": mem["savings_ratio"],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="DACA Attention Benchmarks")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--config", type=str, help="Single config to benchmark (format: batch,heads,seq,dim)")
    args = parser.parse_args()

    if not HAS_MS:
        print("MindSpore not available")
        return

    context.set_context(device_target="Ascend")

    if args.config:
        # Run single config
        batch, heads, seq, dim = map(int, args.config.split(","))
        result = benchmark_chunked_attention(batch, heads, seq, dim)
        print(json.dumps(result, indent=2))
        return

    results = run_benchmarks(verbose=not args.quiet)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("MEMORY COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Config':<25} {'Naive (MB)':<15} {'Chunked (MB)':<15} {'Savings':<10}")
        print("-" * 60)
        for m in results["memory_comparison"]:
            print(f"{m['config']:<25} {m['naive_mb']:<15.1f} {m['chunked_mb']:<15.1f} {m['savings_ratio']:<10.0f}x")


if __name__ == "__main__":
    main()
