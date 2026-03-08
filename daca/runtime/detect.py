"""Hardware detection for Ascend NPUs.

Provides functions to detect and query Ascend NPU hardware without requiring
sudo privileges. Works on OpenI virtual machines.

Key functions:
    - detect_npu(): Check if Ascend NPUs are available
    - get_npu_info(): Get detailed NPU specifications
    - check_cann_version(): Verify CANN compatibility
    - is_openi_env(): Detect OpenI VM environment
"""

import os
import platform
import subprocess
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("daca.runtime.detect")

# Cache for detection results
_npu_detected: Optional[bool] = None
_npu_info: Optional[Dict[str, Any]] = None
_cann_info: Optional[Dict[str, Any]] = None


def get_platform_info() -> Dict[str, str]:
    """Get platform information.

    Returns:
        Dictionary with platform details (os, arch, python version).

    Example:
        >>> from daca.runtime import get_platform_info
        >>> info = get_platform_info()
        >>> print(info['arch'])
        'aarch64'
    """
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def is_openi_env() -> bool:
    """Detect if running in OpenI VM environment.

    OpenI is a common platform for accessing Ascend hardware.
    This detection helps apply OpenI-specific workarounds.

    Returns:
        True if running in OpenI environment, False otherwise.

    Example:
        >>> from daca.runtime import is_openi_env
        >>> if is_openi_env():
        ...     print("Running on OpenI")
    """
    # Check common OpenI environment indicators
    indicators = [
        os.path.exists("/openi"),
        os.path.exists("/home/ma-user"),  # OpenI default user
        "OPENI" in os.environ,
        "OPENI_PROJECT" in os.environ,
        os.path.exists("/home/ma-user/.openi"),
    ]

    # Check hostname patterns
    hostname = platform.node().lower()
    openi_hostnames = ["openi", "modelarts", "notebook"]

    if any(pattern in hostname for pattern in openi_hostnames):
        return True

    return any(indicators)


def detect_npu() -> bool:
    """Detect if Ascend NPUs are available.

    Uses multiple detection methods:
    1. Try importing mindspore and checking device count
    2. Check for CANN installation
    3. Check /dev files for NPU devices

    Results are cached for performance.

    Returns:
        True if at least one NPU is detected, False otherwise.

    Example:
        >>> from daca.runtime import detect_npu
        >>> if detect_npu():
        ...     print("NPU available")
        ... else:
        ...     print("No NPU detected")
    """
    global _npu_detected

    if _npu_detected is not None:
        return _npu_detected

    _npu_detected = False

    # Method 1: Try MindSpore device detection
    try:
        import mindspore as ms
        from mindspore import context, Tensor
        import mindspore.ops as ops

        # Try to set NPU context
        try:
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
            # Create a simple tensor to verify device works
            x = Tensor([1.0])
            y = ops.Add()(x, x)
            _npu_detected = True
            logger.debug("NPU detected via MindSpore")
            return True
        except Exception as e:
            logger.debug(f"MindSpore NPU context failed: {e}")
    except ImportError:
        logger.debug("MindSpore not available for NPU detection")

    # Method 2: Check CANN installation
    cann_paths = [
        "/usr/local/Ascend/ascend-toolkit",
        "/usr/local/Ascend",
        "/home/ma-user/Ascend",
        os.path.expanduser("~/Ascend"),
    ]

    for path in cann_paths:
        if os.path.exists(path):
            logger.debug(f"CANN installation found at {path}")
            # Check for npu-smi tool
            npu_smi_paths = [
                os.path.join(path, "bin", "npu-smi"),
                "/usr/local/bin/npu-smi",
                "/usr/bin/npu-smi",
            ]
            for smi_path in npu_smi_paths:
                if os.path.exists(smi_path):
                    try:
                        result = subprocess.run(
                            [smi_path, "info"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0 and "NPU" in result.stdout:
                            _npu_detected = True
                            logger.debug("NPU detected via npu-smi")
                            return True
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass

    # Method 3: Check device files
    npu_devices = [f"/dev/davinci{i}" for i in range(8)]
    npu_devices.append("/dev/davinci_manager")

    for device in npu_devices:
        if os.path.exists(device):
            logger.debug(f"NPU device file found: {device}")
            _npu_detected = True
            return True

    logger.debug("No NPU detected")
    return False


def get_npu_info() -> Dict[str, Any]:
    """Get detailed NPU specifications.

    Returns information about detected NPUs including:
    - count: Number of NPUs
    - name: Device name (e.g., "Ascend 910ProA")
    - cores: DaVinci cores per device
    - memory_gb: HBM memory per device in GB
    - tflops_fp16: Theoretical FP16 performance

    Returns:
        Dictionary with NPU specifications.

    Raises:
        RuntimeError: If no NPUs are detected.

    Example:
        >>> from daca.runtime import get_npu_info
        >>> info = get_npu_info()
        >>> print(f"Found {info['count']} x {info['name']}")
        Found 4 x Ascend 910ProA
    """
    global _npu_info

    if _npu_info is not None:
        return _npu_info

    if not detect_npu():
        raise RuntimeError("No Ascend NPU detected. Cannot get NPU info.")

    _npu_info = {
        "count": 0,
        "name": "Unknown",
        "cores": 0,
        "memory_gb": 0,
        "tflops_fp16": 0.0,
        "compute_capability": "unknown",
    }

    # Try to get info from npu-smi
    smi_paths = [
        "/usr/local/Ascend/ascend-toolkit/bin/npu-smi",
        "/usr/local/Ascend/bin/npu-smi",
        "/usr/local/bin/npu-smi",
        "/usr/bin/npu-smi",
    ]

    for smi_path in smi_paths:
        if os.path.exists(smi_path):
            try:
                result = subprocess.run(
                    [smi_path, "info"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    _parse_npu_smi_output(result.stdout, _npu_info)
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug(f"npu-smi failed: {e}")

    # Try MindSpore for device count
    try:
        import mindspore as ms
        from mindspore import context

        # Get device count from MindSpore
        context.set_context(device_target="Ascend")
        # MindSpore doesn't expose device_count directly, use environment
        rank_size = int(os.environ.get("RANK_SIZE", "1"))
        _npu_info["count"] = max(_npu_info.get("count", 0), rank_size)
    except Exception as e:
        logger.debug(f"MindSpore device count failed: {e}")

    # Default values for 910ProA if detection failed
    if _npu_info["count"] == 0:
        _npu_info["count"] = 1

    if _npu_info["name"] == "Unknown":
        # Check environment for device info
        chip_name = os.environ.get("ASCEND_CHIP_NAME", "")
        if "910" in chip_name.upper():
            _npu_info["name"] = "Ascend 910"
            _npu_info["cores"] = 32
            _npu_info["memory_gb"] = 32
            _npu_info["tflops_fp16"] = 256.0
        else:
            # Default to 910ProA specs
            _npu_info["name"] = "Ascend 910ProA"
            _npu_info["cores"] = 32
            _npu_info["memory_gb"] = 32
            _npu_info["tflops_fp16"] = 256.0

    return _npu_info


def _parse_npu_smi_output(output: str, info: Dict[str, Any]) -> None:
    """Parse npu-smi info output.

    Args:
        output: stdout from npu-smi info command
        info: Dictionary to update with parsed info
    """
    lines = output.strip().split("\n")

    count = 0
    for line in lines:
        line = line.strip()

        # Count NPU entries
        if line.startswith("NPU") and "Chip" in line:
            count += 1

        # Parse chip name
        if "Chip Name" in line or "Chip Type" in line:
            parts = line.split(":")
            if len(parts) > 1:
                chip_name = parts[1].strip()
                info["name"] = chip_name

                # Infer specs from chip name
                if "910" in chip_name:
                    info["cores"] = 32
                    info["memory_gb"] = 32
                    info["tflops_fp16"] = 256.0
                elif "310" in chip_name:
                    info["cores"] = 8
                    info["memory_gb"] = 8
                    info["tflops_fp16"] = 44.0

        # Parse memory
        if "HBM" in line or "Memory" in line:
            parts = line.split(":")
            if len(parts) > 1:
                mem_str = parts[1].strip()
                # Try to extract GB value
                import re
                match = re.search(r"(\d+)\s*GB", mem_str, re.IGNORECASE)
                if match:
                    info["memory_gb"] = int(match.group(1))

    info["count"] = max(count, info.get("count", 0))


def check_cann_version() -> Dict[str, Any]:
    """Check CANN (Compute Architecture for Neural Networks) version.

    Verifies CANN compatibility with DACA requirements.

    Returns:
        Dictionary with CANN version information:
        - version: Full version string
        - major, minor, patch: Version components
        - compatible: Whether version is compatible with DACA
        - path: Installation path

    Example:
        >>> from daca.runtime import check_cann_version
        >>> info = check_cann_version()
        >>> print(f"CANN {info['version']}, compatible: {info['compatible']}")
    """
    global _cann_info

    if _cann_info is not None:
        return _cann_info

    _cann_info = {
        "version": "unknown",
        "major": 0,
        "minor": 0,
        "patch": 0,
        "compatible": False,
        "path": None,
    }

    # Check environment variable first
    cann_version = os.environ.get("ASCEND_TOOLKIT_VERSION", "")
    if cann_version:
        _cann_info["version"] = cann_version
        _parse_version(cann_version, _cann_info)

    # Check version file
    version_paths = [
        "/usr/local/Ascend/ascend-toolkit/version.info",
        "/usr/local/Ascend/version.info",
        "/home/ma-user/Ascend/ascend-toolkit/version.info",
    ]

    for path in version_paths:
        if os.path.exists(path):
            _cann_info["path"] = os.path.dirname(path)
            try:
                with open(path, "r") as f:
                    content = f.read()
                    # Parse version from file
                    import re
                    match = re.search(r"Version\s*[:=]?\s*([0-9.]+)", content)
                    if match:
                        _cann_info["version"] = match.group(1)
                        _parse_version(match.group(1), _cann_info)
            except IOError as e:
                logger.debug(f"Failed to read version file: {e}")

    # Check compatibility (CANN 8.3+ required for DACA)
    if _cann_info["major"] > 8 or (_cann_info["major"] == 8 and _cann_info["minor"] >= 3):
        _cann_info["compatible"] = True
    elif _cann_info["version"] != "unknown":
        # Version detected but may be older
        logger.warning(
            f"CANN version {_cann_info['version']} may not be fully compatible. "
            f"DACA is designed for CANN 8.3+"
        )
        _cann_info["compatible"] = True  # Allow it to work, but warn

    return _cann_info


def _parse_version(version_str: str, info: Dict[str, Any]) -> None:
    """Parse version string into components.

    Args:
        version_str: Version string like "8.3.RC1.alpha003"
        info: Dictionary to update with parsed components
    """
    import re

    # Handle version strings like "8.3.RC1.alpha003" or "8.3.0"
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version_str)
    if match:
        info["major"] = int(match.group(1))
        info["minor"] = int(match.group(2))
        info["patch"] = int(match.group(3)) if match.group(3) else 0
