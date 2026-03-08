# DACA - DaVinci Accelerated Compute Architecture

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.7%2B-orange.svg)](https://www.mindspore.cn/)

**DACA is to Ascend what CUDA is to NVIDIA and ROCm is to AMD.**

A compute platform library that makes Ascend 910ProA NPUs fully usable for AI workloads by closing operator gaps, fixing CANN bugs, optimizing performance, and enabling CUDA-ecosystem code to run on Ascend.

## Features

- **Pure Python** - No sudo, no custom kernels, runs on OpenI virtual machines
- **Training Ready** - All NN modules are nn.Cell with full backward pass support
- **bf16 Shim** - Transparent bf16 → fp16 conversion (hardware doesn't support bf16)
- **LayerNorm Fix** - fp32 upcast workaround for CANN fusion bug
- **Missing Operators** - SiLU, SwiGLU, and other missing ops implemented
- **FlashAttention** - Chunked online softmax (pure MindSpore, full autograd)
- **Graph Mode** - Safe environment variables for stable compilation
- **MindFormers Compatible** - Patches for seamless integration

## Quick Start

```bash
# Install
pip install -e .

# Verify installation
python -c "import daca; daca.info()"
```

### Basic Usage

```python
import daca

# Apply all compatibility patches
daca.patch()

# Your MindSpore code now works on Ascend
import mindspore as ms
from mindspore import ops

# bf16 automatically converted to fp16
x = ms.Tensor([1.0, 2.0, 3.0], ms.bfloat16)  # → fp16 internally

# SiLU now available
y = ops.silu(x)  # Works via x * sigmoid(x)

# LayerNorm won't crash
from daca.nn import LayerNorm
ln = LayerNorm(hidden_size, epsilon=1e-5)
normalized = ln(x)  # Uses fp32 upcast internally

# Use FlashAttention
from daca.nn import FlashAttention
attn = FlashAttention(num_heads=32, num_kv_heads=8, head_dim=64)  # GQA-ready
output = attn(query, key, value)  # Full autograd support (nn.Cell)

# When done, restore original state
daca.unpatch()
```

## Why DACA?

### The Problem

Huawei Ascend 910ProA is powerful hardware (256 TFLOPS FP16, 32GB HBM2), but the software stack has gaps:

| Issue | Impact |
|-------|--------|
| bf16 unsupported | Crashes with `aclnnCastGetWorkspaceSize` error |
| ops.SiLU missing | Code expecting SiLU fails |
| LayerNorm fp16 broken | CANN routes through FlashAttention, crashes |
| CANN auto-fusion bugs | Aggressive fusion causes rank mismatches |

### The Solution

DACA patches these issues at the Python level:

```python
def patch():
    """Apply all DACA compatibility patches."""
    runtime.dtype.enable_bf16_shim()      # bf16 → fp16
    nn.activations.inject_silu()          # Add SiLU to ops namespace
    nn.layernorm.enable_fp32_upcast()     # LayerNorm fp32 fast-path
    compile.fusion.disable_fa_fusion()    # Disable broken CANN fusions
    compile.graph_mode.set_safe_env()     # Graph mode env vars
    compat.mindspore_patches.apply_all()  # MindSpore namespace patches
    compat.mindformers_patches.apply_all() # MindFormers fixes
```

## Hardware Requirements

- **NPU**: Ascend 910ProA (TSMC N7+)
- **Cores**: 32× DaVinci cores per NPU
- **Memory**: 32GB HBM2 per NPU
- **Performance**: 256 TFLOPS FP16
- **CANN**: 8.3.RC1.alpha003
- **MindSpore**: 2.7.1+
- **Platform**: aarch64-linux, no root required

## Installation

### From Source

```bash
git clone https://github.com/changcheng967/DACA.git
cd DACA
pip install -e .
```

### With MindSpore

```bash
pip install -e ".[mindspore]"
```

### Development

```bash
pip install -e ".[dev]"
```

## API Reference

### Core Functions

```python
import daca

# Show DACA info banner
daca.info()

# Apply all patches
daca.patch()

# Remove all patches
daca.unpatch()

# Check if patched
daca.is_patched()  # → bool

# Run benchmarks
daca.benchmark()
```

### Runtime Module

```python
from daca.runtime import (
    detect_npu,          # Detect Ascend NPUs
    get_npu_info,        # Get NPU specs
    check_cann_version,  # Verify CANN compatibility
    is_openi_env,        # Detect OpenI VM
    set_device,          # Set active device
    device_count,        # Get NPU count
    MemoryTracker,       # Track memory usage
)

# Detect hardware
if detect_npu():
    info = get_npu_info()
    print(f"Found {info['count']} NPUs")
    print(f"Memory: {info['memory_gb']}GB each")
```

### NN Module

```python
from daca.nn import (
    FlashAttention,   # FlashAttention wrapper
    LayerNorm,        # fp32 upcast LayerNorm
    RMSNorm,          # Manual RMSNorm decomposition
    silu,             # x * sigmoid(x)
    swiglu,           # SwiGLU activation
    RotaryEmbedding,  # Rotary position embeddings
    Embedding,        # Embedding wrapper
    softmax,          # Numerically stable softmax
)

# FlashAttention
attn = FlashAttention(head_dim=64, num_heads=32, dropout=0.0)
output = attn(query, key, value, mask=mask)

# LayerNorm (fp32 upcast internally)
ln = LayerNorm(hidden_size=768, epsilon=1e-6)
normalized = ln(hidden_states)

# RMSNorm
rms = RMSNorm(hidden_size=768, epsilon=1e-6)
normalized = rms(hidden_states)

# Rotary embeddings
rotary = RotaryEmbedding(dim=64, max_seq_len=2048)
cos, sin = rotary(seq_len)
q_rotated = apply_rotary_pos_emb(query, cos, sin)
```

### BLAS Module

```python
from daca.blas import (
    matmul,        # MatMul with workspace handling
    bmm,           # BatchMatMul
    batch_matmul,  # Alias for bmm
)

# 2D MatMul
result = matmul(a, b)

# 4D attention shapes
q = ms.Tensor(shape=(batch, heads, seq, dim))
k = ms.Tensor(shape=(batch, heads, seq, dim))
scores = matmul(q, k.transpose(0, 1, 3, 2))

# BatchMatMul
result = bmm(a, b)  # (b, n, m) @ (b, m, p) → (b, n, p)
```

### Compile Module

```python
from daca.compile import (
    enable_graph_mode,           # Graph mode with safe env vars
    disable_flash_attention_fusion,  # Disable CANN FA fusion
    FusionConfig,                # Granular fusion control
)

# Enable graph mode safely
enable_graph_mode()

# Disable broken fusions
disable_flash_attention_fusion()
```

### Compatibility Module

```python
from daca.compat import (
    rewrite_config,  # bf16 → fp16 in dicts
    ConfigRewriter,  # JSON/YAML config rewriting
)

# Rewrite config dict
config = {"dtype": "bfloat16", "hidden_size": 768}
rewritten = rewrite_config(config)  # dtype → float16

# Rewrite config file
rewriter = ConfigRewriter()
rewriter.rewrite_file("config.json", "config_fp16.json")
```

## Probing Hardware

```bash
# Run hardware capability probe
python tools/probe.py

# Output: probe_data.json with test results
```

Example output:

```json
{
  "ops": {
    "bf16_cast": {"ok": false, "err": "aclnnCastGetWorkspaceSize call failed"},
    "silu": {"ok": false, "err": "module 'mindspore.ops' has no attribute 'SiLU'"},
    "sigmoid": {"ok": true, "ms": 95.3},
    "fa_native": {"ok": true, "ms": 1803.1},
    "ln_fp32": {"ok": true, "ms": 9904.2}
  }
}
```

## Diagnosing Issues

```bash
# Run environment diagnostics
python tools/doctor.py
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=daca
```

## Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/bench_all.py

# Run specific benchmark
python benchmarks/bench_matmul.py
python benchmarks/bench_attention.py
```

## Project Structure

```
DACA/
├── daca/
│   ├── __init__.py          # Main entry point
│   ├── runtime/             # Hardware detection, device mgmt
│   ├── blas/                # Matrix operations
│   ├── nn/                  # Neural network layers
│   ├── comm/                # Multi-NPU communication
│   ├── compile/             # Graph mode, fusion control
│   ├── compat/              # Third-party patches
│   ├── autotune/            # Benchmarking
│   └── docs/                # Documentation
├── tests/                   # Test suite
├── benchmarks/              # Performance benchmarks
├── examples/                # Usage examples
├── tools/                   # probe.py, doctor.py
├── setup.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Known Limitations

1. **bf16 completely unsupported** - Hardware limitation, shim converts to fp16
2. **LayerNorm fp16 broken** - CANN fusion bug, workaround uses fp32 upcast
3. **CANN auto-fusion bugs** - Aggressive fusion causes crashes, disabled by DACA

## Available vs Missing Operators

### Available (Native)
Sigmoid, GeLU, ReLU, Tanh, Mish, FastGeLU, HSigmoid, HSwish, PReLU, SeLU, Softmax, RMSNorm, RotaryEmbedding, FlashAttentionScore, MatMul, BatchMatMul

### Not Available (Shimmed)
SiLU → `x * sigmoid(x)`, SwiGLU → manual split/silu/mul, bf16 → fp16

## Contributing

Contributions are welcome! Please read our contributing guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Huawei Ascend team for the hardware
- MindSpore community
- OpenI platform for providing access to Ascend hardware
