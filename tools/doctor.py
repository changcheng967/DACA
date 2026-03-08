#!/usr/bin/env python
"""DACA Environment Doctor.

Diagnoses common environment issues and provides recommendations.

Usage:
    python tools/doctor.py
"""

import os
import sys
import subprocess
from typing import List, Tuple


class DoctorCheck:
    """A single diagnostic check."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.message = ""
        self.recommendation = ""

    def run(self) -> bool:
        """Run the check. Returns True if passed."""
        raise NotImplementedError


class CheckMindSpore(DoctorCheck):
    """Check if MindSpore is installed."""

    def __init__(self):
        super().__init__(
            "mindspore",
            "Check if MindSpore is installed"
        )

    def run(self) -> bool:
        try:
            import mindspore as ms
            self.passed = True
            self.message = f"MindSpore {ms.__version__} installed"
        except ImportError:
            self.passed = False
            self.message = "MindSpore not installed"
            self.recommendation = "Install MindSpore: pip install mindspore"
        return self.passed


class CheckCANN(DoctorCheck):
    """Check if CANN is installed."""

    def __init__(self):
        super().__init__(
            "cann",
            "Check if CANN is installed"
        )

    def run(self) -> bool:
        # Check environment variables
        asc_toolkit = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")

        if asc_toolkit:
            self.passed = True
            self.message = f"CANN found at {asc_toolkit}"
            return True

        # Check common paths
        common_paths = [
            "/usr/local/Ascend/ascend-toolkit",
            "/usr/local/Ascend",
            "/home/ma-user/Ascend/ascend-toolkit",
        ]

        for path in common_paths:
            if os.path.exists(path):
                self.passed = True
                self.message = f"CANN found at {path}"
                return True

        self.passed = False
        self.message = "CANN installation not found"
        self.recommendation = "Install CANN 8.3+ from Huawei Ascend website"
        return False


class CheckNPUDevice(DoctorCheck):
    """Check if NPU devices are accessible."""

    def __init__(self):
        super().__init__(
            "npu_device",
            "Check if NPU devices are accessible"
        )

    def run(self) -> bool:
        # Check device files
        device_files = [f"/dev/davinci{i}" for i in range(8)]

        found_devices = [d for d in device_files if os.path.exists(d)]

        if found_devices:
            self.passed = True
            self.message = f"Found {len(found_devices)} NPU device(s)"
            return True

        # Check through MindSpore
        try:
            from daca.runtime import detect_npu
            if detect_npu():
                self.passed = True
                self.message = "NPU detected via MindSpore"
                return True
        except Exception:
            pass

        self.passed = False
        self.message = "No NPU devices found"
        self.recommendation = "Ensure Ascend NPU is properly installed and accessible"
        return False


class CheckPythonVersion(DoctorCheck):
    """Check Python version."""

    def __init__(self):
        super().__init__(
            "python_version",
            "Check Python version compatibility"
        )

    def run(self) -> bool:
        import sys

        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor >= 8:
            self.passed = True
            self.message = f"Python {version_str} (compatible)"
        else:
            self.passed = False
            self.message = f"Python {version_str} (incompatible)"
            self.recommendation = "DACA requires Python 3.8 or later"

        return self.passed


class CheckNumpyVersion(DoctorCheck):
    """Check NumPy version."""

    def __init__(self):
        super().__init__(
            "numpy_version",
            "Check NumPy version"
        )

    def run(self) -> bool:
        try:
            import numpy as np
            version = np.__version__
            major, minor = map(int, version.split(".")[:2])

            if major > 1 or (major == 1 and minor >= 20):
                self.passed = True
                self.message = f"NumPy {version}"
            else:
                self.passed = False
                self.message = f"NumPy {version} (too old)"
                self.recommendation = "Upgrade NumPy: pip install numpy>=1.20"

        except ImportError:
            self.passed = False
            self.message = "NumPy not installed"
            self.recommendation = "Install NumPy: pip install numpy"

        return self.passed


class CheckDACAPatch(DoctorCheck):
    """Check if DACA patches work."""

    def __init__(self):
        super().__init__(
            "daca_patch",
            "Test DACA patch functionality"
        )

    def run(self) -> bool:
        try:
            import daca

            # Test patch
            daca.patch()

            # Test basic operations
            try:
                import mindspore as ms
                from mindspore import Tensor
                import mindspore.common.dtype as mstype

                # Test tensor creation
                x = Tensor([1.0, 2.0], mstype.float16)

                daca.unpatch()

                self.passed = True
                self.message = "DACA patch/unpatch working correctly"

            except ImportError:
                self.passed = True
                self.message = "DACA patch works (MindSpore not available for full test)"

        except Exception as e:
            self.passed = False
            self.message = f"DACA patch error: {str(e)}"
            self.recommendation = "Check DACA installation: pip install -e ."

        return self.passed


class CheckBF16Shim(DoctorCheck):
    """Check BF16 shim functionality."""

    def __init__(self):
        super().__init__(
            "bf16_shim",
            "Test BF16 → FP16 shim"
        )

    def run(self) -> bool:
        try:
            import daca
            from daca.runtime import is_bf16_supported

            daca.patch()

            # Check that BF16 is reported as unsupported
            if not is_bf16_supported():
                self.passed = True
                self.message = "BF16 shim active (correctly redirects to FP16)"
            else:
                self.passed = False
                self.message = "BF16 reported as supported (incorrect for 910ProA)"

            daca.unpatch()

        except Exception as e:
            self.passed = False
            self.message = f"BF16 shim test error: {str(e)}"

        return self.passed


class CheckCANNVersion(DoctorCheck):
    """Check CANN version compatibility."""

    def __init__(self):
        super().__init__(
            "cann_version",
            "Check CANN version compatibility"
        )

    def run(self) -> bool:
        try:
            from daca.runtime import check_cann_version

            info = check_cann_version()

            if info.get("compatible", False):
                self.passed = True
                self.message = f"CANN {info.get('version', 'unknown')} (compatible)"
            else:
                self.passed = False
                self.message = f"CANN {info.get('version', 'unknown')} (may not be compatible)"
                self.recommendation = "DACA is designed for CANN 8.3+"

        except Exception as e:
            self.passed = False
            self.message = f"CANN version check error: {str(e)}"

        return self.passed


def run_doctor():
    """Run all diagnostic checks."""
    print("=" * 60)
    print("DACA Environment Doctor")
    print("=" * 60)
    print()

    checks = [
        CheckPythonVersion(),
        CheckNumpyVersion(),
        CheckMindSpore(),
        CheckCANN(),
        CheckCANNVersion(),
        CheckNPUDevice(),
        CheckDACAPatch(),
        CheckBF16Shim(),
    ]

    results = []

    for check in checks:
        print(f"Checking: {check.description}...")
        passed = check.run()
        results.append((check, passed))

        if passed:
            print(f"  ✓ PASS: {check.message}")
        else:
            print(f"  ✗ FAIL: {check.message}")
            if check.recommendation:
                print(f"    → {check.recommendation}")
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All checks passed! DACA is ready to use.")
        return 0
    else:
        print("\n✗ Some checks failed. See recommendations above.")
        return 1


def main():
    """Main entry point."""
    return run_doctor()


if __name__ == "__main__":
    sys.exit(main())
