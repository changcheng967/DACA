"""DACA Autotune Module.

Auto-benchmarking and tuning utilities.

Example:
    from daca.autotune import benchmark_op, auto_tune_matmul
"""

from daca.autotune.benchmark import (
    benchmark_op,
    auto_tune_matmul,
    run_all_benchmarks,
    BenchmarkResult,
)

__all__ = [
    "benchmark_op",
    "auto_tune_matmul",
    "run_all_benchmarks",
    "BenchmarkResult",
]
