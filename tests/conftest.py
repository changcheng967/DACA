"""Pytest configuration and fixtures for DACA tests."""

import pytest
import sys

# Check if MindSpore is available
try:
    import mindspore as ms
    HAS_MINDSPORE = True
except ImportError:
    HAS_MINDSPORE = False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "npu: marks tests that require NPU hardware"
    )
    config.addinivalue_line(
        "markers", "mindspore: marks tests that require MindSpore"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and availability."""
    skip_slow = pytest.mark.skip(reason="Slow test (use --runslow to run)")
    skip_npu = pytest.mark.skip(reason="NPU hardware required")
    skip_ms = pytest.mark.skip(reason="MindSpore required")

    for item in items:
        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)

        # Skip NPU tests if no NPU available
        if "npu" in item.keywords and not HAS_MINDSPORE:
            item.add_marker(skip_npu)

        # Skip MindSpore tests if not available
        if "mindspore" in item.keywords and not HAS_MINDSPORE:
            item.add_marker(skip_ms)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="Run slow tests"
    )


@pytest.fixture(scope="session")
def mindspore_available():
    """Check if MindSpore is available."""
    return HAS_MINDSPORE


@pytest.fixture(scope="session")
def npu_available():
    """Check if NPU is available."""
    if not HAS_MINDSPORE:
        return False

    try:
        from daca.runtime import detect_npu
        return detect_npu()
    except Exception:
        return False


@pytest.fixture
def daca_patched():
    """Fixture that applies DACA patches for a test."""
    import daca
    daca.patch()
    yield
    daca.unpatch()


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    if not HAS_MINDSPORE:
        pytest.skip("MindSpore required")

    import mindspore as ms
    from mindspore import Tensor
    import mindspore.common.dtype as mstype

    return Tensor(ms.numpy.random.randn(2, 4, 8), mstype.float16)


@pytest.fixture
def sample_attention_inputs():
    """Create sample inputs for attention testing."""
    if not HAS_MINDSPORE:
        pytest.skip("MindSpore required")

    import mindspore as ms
    from mindspore import Tensor
    import mindspore.common.dtype as mstype

    batch, heads, seq, dim = 2, 4, 8, 16

    q = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)
    k = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)
    v = Tensor(ms.numpy.random.randn(batch, heads, seq, dim), mstype.float16)

    return q, k, v
