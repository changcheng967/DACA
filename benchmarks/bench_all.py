"""Run all DACA benchmarks.

This script runs the complete benchmark suite and generates a report.
"""

import argparse
import json
import time
from typing import Dict, Any

try:
    import mindspore as ms
    from mindspore import context
    HAS_MS = True
except ImportError:
    HAS_MS = False


def run_all_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """Run all benchmarks and return results."""
    if not HAS_MS:
        print("MindSpore not available")
        return {"error": "MindSpore not available"}

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "Ascend 910ProA",
    }

    context.set_context(device_target="Ascend")

    if verbose:
        print("\n" + "=" * 60)
        print("DACA Benchmark Suite")
        print("=" * 60 + "\n")

    # Run each benchmark suite
    try:
        if verbose:
            print("--- MatMul Benchmarks ---")

        from benchmarks.bench_matmul import run_benchmarks as run_matmul
        results["matmul"] = run_matmul(verbose=verbose)
    except Exception as e:
        results["matmul"] = {"error": str(e)}

    try:
        if verbose:
            print("\n--- Attention Benchmarks ---")

        from benchmarks.bench_attention import run_benchmarks as run_attention
        results["attention"] = run_attention(verbose=verbose)
    except Exception as e:
        results["attention"] = {"error": str(e)}

    try:
        if verbose:
            print("\n--- LayerNorm Benchmarks ---")

        from benchmarks.bench_layernorm import run_benchmarks as run_layernorm
        results["layernorm"] = run_layernorm(verbose=verbose)
    except Exception as e:
        results["layernorm"] = {"error": str(e)}

    try:
        if verbose:
            print("\n--- Activation Benchmarks ---")

        from benchmarks.bench_activations import run_benchmarks as run_activations
        results["activations"] = run_activations(verbose=verbose)
    except Exception as e:
        results["activations"] = {"error": str(e)}

    if verbose:
        print("\n" + "=" * 60)
        print("Benchmarks Complete")
        print("=" * 60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="DACA Benchmark Suite")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file (default: benchmark_results.json)")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    results = run_all_benchmarks(verbose=not args.quiet)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    if not args.quiet:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
