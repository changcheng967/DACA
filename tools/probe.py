#!/usr/bin/env python
"""DACA Hardware Probe.

Probes the Ascend NPU hardware and tests operator availability.
Generates a probe_data.json file with results.

Usage:
    python tools/probe.py

Output:
    probe_data.json - JSON file with test results
"""

import json
import time
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List


def create_probe_data() -> Dict[str, Any]:
    """Create probe data structure."""
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": {},
        "hardware": {},
        "ops": {},
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
        }
    }


def probe_platform() -> Dict[str, str]:
    """Probe platform information."""
    import platform
    import os

    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "hostname": platform.node(),
    }


def probe_hardware() -> Dict[str, Any]:
    """Probe hardware information."""
    try:
        from daca.runtime import detect_npu, get_npu_info, check_cann_version

        hardware = {
            "npu_detected": detect_npu(),
        }

        if hardware["npu_detected"]:
            try:
                info = get_npu_info()
                hardware.update(info)
            except Exception as e:
                hardware["npu_info_error"] = str(e)

        try:
            cann_info = check_cann_version()
            hardware["cann"] = cann_info
        except Exception as e:
            hardware["cann_error"] = str(e)

        return hardware

    except ImportError:
        return {"error": "daca.runtime not available"}


def test_operation(name: str, test_func, *args, **kwargs) -> Dict[str, Any]:
    """Test a single operation."""
    result = {"ok": False}

    try:
        start = time.time()
        output = test_func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000

        result["ok"] = True
        result["ms"] = round(elapsed, 1)
        if output is not None:
            result["output_shape"] = str(output.shape) if hasattr(output, "shape") else str(type(output))

    except Exception as e:
        result["err"] = str(e)

    return result


def probe_ops(probe_data: Dict[str, Any]) -> None:
    """Probe operator availability."""
    try:
        import mindspore as ms
        from mindspore import Tensor, context
        import mindspore.ops as ops
        import mindspore.common.dtype as mstype

        # Set context
        try:
            context.set_context(device_target="Ascend")
        except Exception:
            context.set_context(device_target="CPU")

        ops_tests = probe_data["ops"]

        # Test data types
        print("\nProbing data types...")

        ops_tests["bf16_cast"] = test_operation(
            "bf16_cast",
            lambda: Tensor([1.0], mstype.bfloat16).astype(mstype.float16)
        )

        ops_tests["fp16_cast"] = test_operation(
            "fp16_cast",
            lambda: Tensor([1.0], mstype.float32).astype(mstype.float16)
        )

        # Test activations
        print("Probing activations...")

        x = Tensor(ms.numpy.random.randn(100, 100), mstype.float16)

        ops_tests["sigmoid"] = test_operation("sigmoid", lambda: ops.sigmoid(x))
        ops_tests["gelu"] = test_operation("gelu", lambda: ops.gelu(x))
        ops_tests["relu"] = test_operation("relu", lambda: ops.relu(x))
        ops_tests["tanh"] = test_operation("tanh", lambda: ops.tanh(x))

        # Test SiLU (manual)
        ops_tests["silu_manual"] = test_operation(
            "silu_manual",
            lambda: x * ops.sigmoid(x)
        )

        # Test ops.SiLU availability
        ops_tests["silu"] = {
            "ok": hasattr(ops, "SiLU"),
            "err": "module 'mindspore.ops' has no attribute 'SiLU'" if not hasattr(ops, "SiLU") else None
        }

        # Test matrix operations
        print("Probing matrix operations...")

        a = Tensor(ms.numpy.random.randn(256, 256), mstype.float16)
        b = Tensor(ms.numpy.random.randn(256, 256), mstype.float16)

        ops_tests["matmul"] = test_operation("matmul", lambda: ops.matmul(a, b))

        a_3d = Tensor(ms.numpy.random.randn(8, 64, 64), mstype.float16)
        b_3d = Tensor(ms.numpy.random.randn(8, 64, 64), mstype.float16)

        ops_tests["bmm"] = test_operation("bmm", lambda: ops.BatchMatMul()(a_3d, b_3d))

        # Test 4D matmul (attention shapes)
        q = Tensor(ms.numpy.random.randn(2, 4, 8, 16), mstype.float16)
        k = Tensor(ms.numpy.random.randn(2, 4, 8, 16), mstype.float16)

        ops_tests["mm_4d"] = test_operation(
            "mm_4d",
            lambda: ops.matmul(q, k.transpose(0, 1, 3, 2))
        )

        # Test normalization
        print("Probing normalization...")

        x_ln = Tensor(ms.numpy.random.randn(2, 8, 64), mstype.float16)

        ops_tests["ln_fp16"] = test_operation(
            "ln_fp16",
            lambda: ops.layer_norm(x_ln, (64,), None, None, 1e-5)[0]
        )

        x_ln_fp32 = x_ln.astype(mstype.float32)
        ops_tests["ln_fp32"] = test_operation(
            "ln_fp32",
            lambda: ops.layer_norm(x_ln_fp32, (64,), None, None, 1e-5)[0]
        )

        # Test RMSNorm (manual)
        ops_tests["rmsnorm"] = test_operation(
            "rmsnorm",
            lambda: x_ln * ops.rsqrt(ops.mean(ops.pow(x_ln, 2), axis=-1, keep_dims=True) + 1e-5)
        )

        # Test FlashAttention
        print("Probing FlashAttention...")

        if hasattr(ops, "FlashAttentionScore"):
            flash = ops.FlashAttentionScore(scale_value=0.25)
            q_fa = q.transpose(0, 2, 1, 3)
            k_fa = k.transpose(0, 2, 1, 3)
            v_fa = Tensor(ms.numpy.random.randn(2, 4, 8, 16), mstype.float16).transpose(0, 2, 1, 3)

            ops_tests["fa_native"] = test_operation(
                "fa_native",
                lambda: flash(q_fa, k_fa, v_fa)
            )
        else:
            ops_tests["fa_native"] = {"ok": False, "err": "FlashAttentionScore not available"}

        # Test softmax
        print("Probing softmax...")

        ops_tests["softmax_fp16"] = test_operation(
            "softmax_fp16",
            lambda: ops.softmax(x, axis=-1)
        )

        # Test rotary
        print("Probing rotary embedding...")

        pos = Tensor(ms.numpy.arange(8), mstype.float32)
        ops_tests["rotary"] = test_operation(
            "rotary",
            lambda: ops.sin(pos)
        )

        # Test tile (for GQA repeat_kv)
        print("Probing tile/reshape...")

        x_tile = Tensor(ms.numpy.random.randn(2, 4, 8, 16), mstype.float16)
        ops_tests["tile_kv"] = test_operation(
            "tile_kv",
            lambda: ops.tile(ops.expand_dims(x_tile, 2), (1, 1, 2, 1, 1))
        )

        # Test gather (embedding)
        print("Probing gather...")

        indices = Tensor(ms.numpy.random.randint(0, 1000, (2, 8)), mstype.int32)
        weight = Tensor(ms.numpy.random.randn(1000, 64), mstype.float16)

        ops_tests["gather"] = test_operation(
            "gather",
            lambda: ops.gather(weight, indices, 0)
        )

        # Test memory allocation
        print("Probing memory allocation...")

        ops_tests["alloc_2gb"] = test_operation(
            "alloc_2gb",
            lambda: Tensor(ms.numpy.zeros((512 * 1024 * 1024,), mstype.float32))
        )

    except ImportError:
        probe_data["ops"]["error"] = "MindSpore not available"


def main():
    """Run the hardware probe."""
    parser = argparse.ArgumentParser(description="DACA Hardware Probe")
    parser.add_argument("--output", "-o", type=str, default="probe_data.json",
                        help="Output JSON file (default: probe_data.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("DACA Hardware Probe")
    print("=" * 60)

    # Create probe data structure
    probe_data = create_probe_data()

    # Probe platform
    print("\nProbing platform...")
    probe_data["platform"] = probe_platform()
    if args.verbose:
        for key, value in probe_data["platform"].items():
            print(f"  {key}: {value}")

    # Probe hardware
    print("\nProbing hardware...")
    probe_data["hardware"] = probe_hardware()
    if args.verbose:
        print(f"  NPU detected: {probe_data['hardware'].get('npu_detected', False)}")

    # Probe operators
    print("\nProbing operators...")
    probe_ops(probe_data)

    # Calculate summary
    total = len(probe_data["ops"])
    passed = sum(1 for r in probe_data["ops"].values() if r.get("ok", False))
    failed = total - passed

    probe_data["summary"]["total_tests"] = total
    probe_data["summary"]["passed"] = passed
    probe_data["summary"]["failed"] = failed

    # Print summary
    print("\n" + "=" * 60)
    print("Probe Results Summary")
    print("=" * 60)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    print("\nFailed operations:")
    for name, result in probe_data["ops"].items():
        if not result.get("ok", False):
            print(f"  - {name}: {result.get('err', 'unknown error')}")

    print("\nPassed operations (with timing):")
    for name, result in probe_data["ops"].items():
        if result.get("ok", False) and "ms" in result:
            print(f"  - {name}: {result['ms']:.1f} ms")

    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(probe_data, f, indent=2)

    print("Done!")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
