"""LayerNorm benchmarks for DACA.

Compares DACA's FP32-upcast LayerNorm with native implementation.
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


def benchmark_layernorm_native(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype,
    warmup: int = 5,
    repeat: int = 50,
) -> Dict[str, Any]:
    """Benchmark native LayerNorm."""
    if not HAS_MS:
        return {"error": "MindSpore not available"}

    x = Tensor(ms.numpy.random.randn(batch_size, seq_len, hidden_size), dtype)

    try:
        # Warmup
        for _ in range(warmup):
            _ = ops.layer_norm(x, (hidden_size,), None, None, 1e-5)[0]

        # Measure
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = ops.layer_norm(x, (hidden_size,), None, None, 1e-5)[0]
            end = time.perf_counter()
            times.append((end - start) * 1000)

        import statistics
        return {
            "name": f"native_layernorm_{batch_size}x{seq_len}x{hidden_size}",
            "type": "native",
            "dtype": str(dtype),
            "mean_ms": statistics.mean(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        }

    except Exception as e:
        return {"name": "native_layernorm", "error": str(e)}


def benchmark_layernorm_daca(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype,
    warmup: int = 5,
    repeat: int = 50,
) -> Dict[str, Any]:
    """Benchmark DACA's FP32-upcast LayerNorm."""
    if not HAS_MS:
        return {"error": "MindSpore not available"}

    from daca.nn import LayerNorm

    x = Tensor(ms.numpy.random.randn(batch_size, seq_len, hidden_size), dtype)
    ln = LayerNorm(hidden_size)

    # Warmup
    for _ in range(warmup):
        _ = ln(x)

    # Measure
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = ln(x)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    import statistics
    return {
        "name": f"daca_layernorm_{batch_size}x{seq_len}x{hidden_size}",
        "type": "daca",
        "dtype": str(dtype),
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """Run all LayerNorm benchmarks."""
    configs = [
        (1, 512, 768),   # BERT-base
        (1, 512, 1024),  # BERT-large
        (1, 512, 4096),  # LLaMA
        (4, 512, 768),
        (8, 512, 768),
    ]

    results = {"native": [], "daca": []}

    for batch, seq, hidden in configs:
        # DACA LayerNorm (works for both FP16 and FP32)
        daca_result = benchmark_layernorm_daca(batch, seq, hidden, mstype.float16)
        results["daca"].append(daca_result)

        if verbose:
            print(f"DACA LayerNorm {batch}x{seq}x{hidden}: {daca_result['mean_ms']:.3f} ms")

        # Native FP32 (for comparison)
        native_result = benchmark_layernorm_native(batch, seq, hidden, mstype.float32)
        results["native"].append(native_result)

        if verbose and "error" not in native_result:
            print(f"Native LayerNorm {batch}x{seq}x{hidden} (FP32): {native_result['mean_ms']:.3f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="DACA LayerNorm Benchmarks")
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
