"""MatMul benchmarks for DACA.

Benchmarks various MatMul configurations to measure performance.
"""

import time
import argparse
import json
from typing import List, Dict, Any

try:
    import mindspore as ms
    from mindspore import Tensor, context
    import mindspore.ops as ops
    import mindspore.common.dtype as mstype
    HAS_MS = True
except ImportError:
    HAS_MS = False


def benchmark_matmul(m: int, n: int, k: int, dtype, warmup: int = 10, repeat: int = 100) -> Dict[str, Any]:
    """Benchmark a single MatMul configuration."""
    if not HAS_MS:
        return {"error": "MindSpore not available"}

    # Create matrices
    a = Tensor(ms.numpy.random.randn(m, k), dtype)
    b = Tensor(ms.numpy.random.randn(k, n), dtype)

    # Warmup
    for _ in range(warmup):
        _ = ops.matmul(a, b)

    # Measure
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = ops.matmul(a, b)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    import statistics
    return {
        "shape": f"{m}x{k}@{k}x{n}",
        "dtype": str(dtype),
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "tflops": (2 * m * n * k) / (statistics.mean(times) * 1e9) if statistics.mean(times) > 0 else 0,
    }


def run_benchmarks(sizes: List[tuple] = None, verbose: bool = True) -> List[Dict]:
    """Run all MatMul benchmarks."""
    if sizes is None:
        sizes = [
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            # Attention shapes
            (1, 64, 64),  # Single head
            (32, 512, 64),  # Multi-head
            (32, 1024, 64),  # Long sequence
        ]

    results = []

    for m, n, k in sizes:
        # FP16
        result_fp16 = benchmark_matmul(m, n, k, mstype.float16)
        result_fp16["name"] = f"matmul_{m}x{n}x{k}_fp16"
        results.append(result_fp16)

        if verbose:
            print(f"MatMul {m}x{k}@{k}x{n} (FP16): {result_fp16['mean_ms']:.3f} ms, {result_fp16['tflops']:.1f} TFLOPS")

        # FP32
        result_fp32 = benchmark_matmul(m, n, k, mstype.float32)
        result_fp32["name"] = f"matmul_{m}x{n}x{k}_fp32"
        results.append(result_fp32)

        if verbose:
            print(f"MatMul {m}x{k}@{k}x{n} (FP32): {result_fp32['mean_ms']:.3f} ms, {result_fp32['tflops']:.1f} TFLOPS")

    return results


def main():
    parser = argparse.ArgumentParser(description="DACA MatMul Benchmarks")
    parser.add_argument("--sizes", type=str, help="Comma-separated sizes (e.g., 256x256x256,512x512x512)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    if not HAS_MS:
        print("MindSpore not available")
        return

    context.set_context(device_target="Ascend")

    sizes = None
    if args.sizes:
        sizes = []
        for size_str in args.sizes.split(","):
            m, n, k = map(int, size_str.split("x"))
            sizes.append((m, n, k))

    results = run_benchmarks(sizes, verbose=not args.quiet)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
