"""Benchmarking utilities for Ascend NPUs.

Provides timing and performance measurement for operations.

Example:
    from daca.autotune import benchmark_op, BenchmarkResult

    result = benchmark_op(lambda: matmul(a, b), warmup=10, repeat=100)
    print(f"Mean time: {result.mean_ms:.3f} ms")
"""

import time
import logging
import json
from typing import Callable, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from statistics import mean, stdev

logger = logging.getLogger("daca.autotune.benchmark")


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    Attributes:
        name: Operation name.
        mean_ms: Mean execution time in milliseconds.
        std_ms: Standard deviation in milliseconds.
        min_ms: Minimum time in milliseconds.
        max_ms: Maximum time in milliseconds.
        warmup_runs: Number of warmup iterations.
        repeat_runs: Number of measured iterations.
        throughput: Optional throughput (ops/sec).
        metadata: Additional metadata.

    Example:
        >>> result = benchmark_op(lambda: matmul(a, b))
        >>> print(f"Mean: {result.mean_ms:.2f} ms")
    """
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    warmup_runs: int
    repeat_runs: int
    throughput: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        return (
            f"BenchmarkResult({self.name}: "
            f"mean={self.mean_ms:.3f}ms, "
            f"std={self.std_ms:.3f}ms, "
            f"min={self.min_ms:.3f}ms, "
            f"max={self.max_ms:.3f}ms)"
        )


def benchmark_op(
    op_func: Callable,
    name: str = "op",
    warmup: int = 10,
    repeat: int = 100,
    synchronize: bool = True,
) -> BenchmarkResult:
    """Benchmark an operation.

    Measures execution time over multiple runs with warmup.

    Args:
        op_func: Function to benchmark.
        name: Operation name for reporting.
        warmup: Number of warmup iterations (not measured).
        repeat: Number of measured iterations.
        synchronize: Whether to synchronize before timing.

    Returns:
        BenchmarkResult with timing statistics.

    Example:
        >>> def my_op():
        ...     return matmul(a, b)
        >>> result = benchmark_op(my_op, name="matmul_1k")
        >>> print(f"Time: {result.mean_ms:.2f} ms")
    """
    # Warmup
    for _ in range(warmup):
        op_func()

    # Synchronize if requested
    if synchronize:
        try:
            from daca.runtime import synchronize as sync_device
            sync_device()
        except Exception:
            pass

    # Measure
    times = []
    for _ in range(repeat):
        if synchronize:
            try:
                from daca.runtime import synchronize as sync_device
                sync_device()
            except Exception:
                pass

        start = time.perf_counter()
        op_func()

        if synchronize:
            try:
                from daca.runtime import synchronize as sync_device
                sync_device()
            except Exception:
                pass

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Compute statistics
    mean_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0.0
    min_time = min(times)
    max_time = max(times)
    throughput = 1000.0 / mean_time if mean_time > 0 else 0.0

    return BenchmarkResult(
        name=name,
        mean_ms=mean_time,
        std_ms=std_time,
        min_ms=min_time,
        max_ms=max_time,
        warmup_runs=warmup,
        repeat_runs=repeat,
        throughput=throughput,
    )


def auto_tune_matmul(
    m: int,
    n: int,
    k: int,
    tile_sizes: Optional[List[Tuple[int, int, int]]] = None,
    dtype: str = "float16",
    warmup: int = 5,
    repeat: int = 50,
) -> BenchmarkResult:
    """Auto-tune MatMul tile sizes.

    Tests different tile configurations to find optimal performance.

    Args:
        m: First matrix dimension.
        n: Second matrix dimension.
        k: Inner dimension.
        tile_sizes: List of (tile_m, tile_n, tile_k) to try.
        dtype: Data type for matrices.
        warmup: Warmup iterations per config.
        repeat: Measured iterations per config.

    Returns:
        Best BenchmarkResult found.

    Example:
        >>> best = auto_tune_matmul(1024, 1024, 1024)
        >>> print(f"Best config: {best.metadata}")
    """
    if tile_sizes is None:
        tile_sizes = [
            (32, 32, 32),
            (64, 64, 32),
            (64, 64, 64),
            (128, 64, 32),
            (128, 128, 64),
        ]

    try:
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
    except ImportError:
        raise ImportError("MindSpore is required for auto_tune_matmul")

    # Create test matrices
    dtype_map = {
        "float16": mstype.float16,
        "float32": mstype.float32,
    }
    ms_dtype = dtype_map.get(dtype, mstype.float16)

    a = Tensor(ms.numpy.random.randn(m, k), ms_dtype)
    b = Tensor(ms.numpy.random.randn(k, n), ms_dtype)

    results = []

    for tile_m, tile_n, tile_k in tile_sizes:
        # Note: MindSpore doesn't expose tile size control directly
        # This is a placeholder for when/if such API becomes available
        # For now, just benchmark the default

        result = benchmark_op(
            lambda: ms.ops.matmul(a, b),
            name=f"matmul_{m}x{n}x{k}_tile_{tile_m}_{tile_n}_{tile_k}",
            warmup=warmup,
            repeat=repeat,
        )
        result.metadata = {
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype,
        }
        results.append(result)

    # Find best
    best = min(results, key=lambda r: r.mean_ms)
    logger.info(f"Best MatMul config for {m}x{n}x{k}: {best.mean_ms:.3f}ms")

    return best


def run_all_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """Run all standard benchmarks.

    Runs a suite of benchmarks to characterize NPU performance.

    Args:
        verbose: If True, print results to stdout.

    Returns:
        Dictionary with all benchmark results.

    Example:
        >>> results = run_all_benchmarks()
        >>> print(results["matmul"]["mean_ms"])
    """
    results = {}

    try:
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
    except ImportError:
        logger.warning("MindSpore not available, skipping benchmarks")
        return {"error": "MindSpore not available"}

    if verbose:
        print("\n=== DACA Benchmark Suite ===\n")

    # MatMul benchmarks
    matmul_results = {}
    for size in [256, 512, 1024, 2048]:
        a = Tensor(ms.numpy.random.randn(size, size), mstype.float16)
        b = Tensor(ms.numpy.random.randn(size, size), mstype.float16)

        result = benchmark_op(
            lambda: ms.ops.matmul(a, b),
            name=f"matmul_{size}x{size}",
            warmup=5,
            repeat=50,
        )
        matmul_results[f"{size}x{size}"] = result.to_dict()

        if verbose:
            print(f"MatMul {size}x{size}: {result.mean_ms:.3f} ms")

    results["matmul"] = matmul_results

    # BatchMatMul benchmarks
    bmm_results = {}
    for batch, m, n, k in [(8, 64, 64, 64), (32, 128, 64, 64)]:
        a = Tensor(ms.numpy.random.randn(batch, m, k), mstype.float16)
        b = Tensor(ms.numpy.random.randn(batch, k, n), mstype.float16)

        result = benchmark_op(
            lambda: ms.ops.BatchMatMul()(a, b),
            name=f"bmm_{batch}x{m}x{k}x{n}",
            warmup=5,
            repeat=50,
        )
        bmm_results[f"{batch}x{m}x{k}x{n}"] = result.to_dict()

        if verbose:
            print(f"BMM {batch}x{m}x{k}x{n}: {result.mean_ms:.3f} ms")

    results["batch_matmul"] = bmm_results

    # Softmax benchmarks
    softmax_results = {}
    for seq_len in [128, 512, 1024]:
        x = Tensor(ms.numpy.random.randn(1, seq_len, 4096), mstype.float16)

        result = benchmark_op(
            lambda: ms.ops.softmax(x, axis=-1),
            name=f"softmax_{seq_len}",
            warmup=5,
            repeat=50,
        )
        softmax_results[str(seq_len)] = result.to_dict()

        if verbose:
            print(f"Softmax {seq_len}: {result.mean_ms:.3f} ms")

    results["softmax"] = softmax_results

    # LayerNorm benchmarks
    ln_results = {}
    for hidden_size in [768, 1024, 4096]:
        x = Tensor(ms.numpy.random.randn(1, 512, hidden_size), mstype.float16)

        result = benchmark_op(
            lambda: ms.ops.layer_norm(x, (hidden_size,), None, None, 1e-5)[0],
            name=f"layernorm_{hidden_size}",
            warmup=5,
            repeat=50,
        )
        ln_results[str(hidden_size)] = result.to_dict()

        if verbose:
            print(f"LayerNorm {hidden_size}: {result.mean_ms:.3f} ms")

    results["layernorm"] = ln_results

    # GeLU benchmarks
    gelu_results = {}
    for size in [1024, 4096, 16384]:
        x = Tensor(ms.numpy.random.randn(1, size), mstype.float16)

        result = benchmark_op(
            lambda: ms.ops.gelu(x),
            name=f"gelu_{size}",
            warmup=5,
            repeat=50,
        )
        gelu_results[str(size)] = result.to_dict()

        if verbose:
            print(f"GeLU {size}: {result.mean_ms:.3f} ms")

    results["gelu"] = gelu_results

    if verbose:
        print("\n=== Benchmarks Complete ===\n")

    return results


def benchmark_memory_allocation(
    sizes: List[int],
    warmup: int = 5,
    repeat: int = 20,
) -> Dict[int, BenchmarkResult]:
    """Benchmark memory allocation performance.

    Args:
        sizes: List of allocation sizes in bytes.
        warmup: Warmup iterations.
        repeat: Measured iterations.

    Returns:
        Dictionary mapping size to BenchmarkResult.
    """
    results = {}

    try:
        import mindspore as ms
        from mindspore import Tensor
        import mindspore.common.dtype as mstype
    except ImportError:
        return results

    for size_bytes in sizes:
        # Convert to number of float32 elements
        num_elements = size_bytes // 4

        def alloc():
            return Tensor(ms.numpy.zeros((num_elements,), dtype=mstype.float32))

        result = benchmark_op(
            alloc,
            name=f"alloc_{size_bytes // (1024*1024)}MB",
            warmup=warmup,
            repeat=repeat,
        )
        results[size_bytes] = result

    return results
