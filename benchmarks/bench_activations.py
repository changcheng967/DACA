"""Activation benchmarks for DACA.

Benchmarks SiLU and other activations.
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


def benchmark_activation(
    name: str,
    activation_func,
    input_shape: tuple,
    dtype,
    warmup: int = 5,
    repeat: int = 100,
) -> Dict[str, Any]:
    """Benchmark an activation function."""
    if not HAS_MS:
        return {"error": "MindSpore not available"}

    x = Tensor(ms.numpy.random.randn(*input_shape), dtype)

    # Warmup
    for _ in range(warmup):
        _ = activation_func(x)

    # Measure
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = activation_func(x)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    import statistics
    return {
        "name": f"{name}_{input_shape}",
        "activation": name,
        "shape": str(input_shape),
        "dtype": str(dtype),
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """Run all activation benchmarks."""
    from daca.nn import silu

    activations = {
        "sigmoid": ops.sigmoid,
        "relu": ops.relu,
        "gelu": ops.gelu,
        "tanh": ops.tanh,
        "silu_daca": silu,
        "fast_gelu": ops.fast_gelu if hasattr(ops, "fast_gelu") else ops.gelu,
    }

    shapes = [
        (1024,),
        (4096,),
        (1, 4096),
        (32, 4096),
        (1, 512, 768),
    ]

    results = {}

    for name, func in activations.items():
        results[name] = []
        for shape in shapes:
            result = benchmark_activation(name, func, shape, mstype.float16)
            results[name].append(result)

            if verbose:
                print(f"{name} {shape}: {result['mean_ms']:.3f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="DACA Activation Benchmarks")
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
