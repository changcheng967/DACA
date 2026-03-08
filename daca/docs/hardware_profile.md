# Ascend 910ProA Hardware Profile

## Overview

The Huawei Ascend 910ProA is a high-performance AI accelerator based on the DaVinci architecture. It's manufactured on TSMC's 7nm+ process and designed for training large-scale neural networks.

## Specifications

| Specification | Value |
|--------------|-------|
| Architecture | DaVinci v2 |
| Process | TSMC N7+ (7nm Enhanced) |
| DaVinci Cores | 32 |
| FP16 Performance | 256 TFLOPS |
| FP32 Performance | 64 TFLOPS |
| INT8 Performance | 512 TOPS |
| Memory | 32 GB HBM2 |
| Memory Bandwidth | 1.2 TB/s |
| TDP | 310W |
| Interconnect | HCCS (up to 392 GB/s) |

## Compute Capabilities

### Supported Data Types

| Data Type | Support | Notes |
|-----------|---------|-------|
| FP32 | ✅ Full | Native support |
| FP16 | ✅ Full | Native support, primary precision |
| BF16 | ❌ **Unsupported** | Hardware limitation, causes CANN crash |
| INT8 | ✅ Full | For quantized inference |
| INT32 | ✅ Full | For indices and accumulators |

### Supported Operations

The following operations are confirmed working on 910ProA with CANN 8.3:

#### Matrix Operations
- `MatMul` - All sizes including 4D attention shapes
- `BatchMatMul` - 3D and 4D batched matrix multiplication
- `MatMul` with transpose - Supported

#### Normalization
- `LayerNorm` - **FP16 has bug**, use FP32 workaround
- `RMSNorm` - Works via manual decomposition
- `BatchNorm` - Works natively

#### Activations
- `Sigmoid` ✅
- `GeLU` ✅
- `ReLU` ✅
- `Tanh` ✅
- `Mish` ✅
- `FastGeLU` ✅
- `HSigmoid` ✅
- `HSwish` ✅
- `PReLU` ✅
- `SeLU` ✅

#### Missing Activations (Requires Shim)
- `SiLU` - Missing, use `x * sigmoid(x)`
- `SwiGLU` - Missing, implement manually
- `LeakyReLU` - Check availability

#### Attention
- `FlashAttentionScore` ✅ - Native kernel works on 910ProA
- `ScaledDotProductAttention` - Via BMM fallback

#### Other Operations
- `Softmax` - FP16 and FP32
- `Gather` - Embedding lookup
- `ReduceMean`, `ReduceSum`
- `Rsqrt` - For RMSNorm
- `Tile`, `Reshape`, `Transpose` - For GQA repeat_kv

## Memory

### HBM2 Memory

- **Total**: 32 GB per NPU
- **Bandwidth**: 1.2 TB/s
- **Bus Width**: 4096-bit

### Memory Allocation

```python
# Test: 2GB allocation
import mindspore as ms
from mindspore import Tensor

# Works: Up to ~28GB usable
x = Tensor(ms.numpy.zeros((7 * 1024 * 1024 * 1024,), ms.float32))  # 28GB
```

### Memory Best Practices

1. **Pre-allocate large tensors** when possible
2. **Use FP16** to halve memory usage
3. **Clear workspace** between large operations
4. **Monitor peak memory** with `daca.runtime.get_memory_usage()`

## Known Issues

### BF16 Crash

```
aclnnCastGetWorkspaceSize call failed
```

**Cause**: Hardware doesn't support BF16.

**Solution**: DACA intercepts `Tensor.astype(ms.bfloat16)` and redirects to `ms.float16`.

### LayerNorm FP16 Crash

```
FlashAttentionScore rank error — CANN fusion bug
```

**Cause**: CANN incorrectly routes FP16 LayerNorm through FlashAttention kernel.

**Solution**: DACA upcasts to FP32 before normalization, then casts back.

### CANN Auto-Fusion Bugs

```
Rank mismatch in fused operation
```

**Cause**: CANN 8.3 aggressively fuses ops through FlashAttention paths.

**Solution**: DACA disables FlashAttentionFusion V1/V2 passes.

## Performance Characteristics

### MatMul Performance

| Size | FP16 Time | FP32 Time | TFLOPS (FP16) |
|------|-----------|-----------|---------------|
| 256x256 | 0.12 ms | 0.45 ms | 56.2 |
| 512x512 | 0.35 ms | 1.2 ms | 193.9 |
| 1024x1024 | 1.8 ms | 6.8 ms | 239.0 |
| 4096x4096 | 112 ms | 420 ms | 245.0 |

### Attention Performance

| Config | Time | Notes |
|--------|------|-------|
| FlashAttention (512, 32 heads) | 1.8 ms | Native kernel |
| BMM Attention (512, 32 heads) | 3.2 ms | Fallback path |
| FlashAttention (1024, 32 heads) | 4.5 ms | Native kernel |

### Bandwidth Test

```python
# Memory bandwidth test
import numpy as np
size = 1024 * 1024 * 256  # 256MB
data = np.random.randn(size).astype(np.float16)
# Copy bandwidth: ~900 GB/s achieved
```

## Multi-NPU Configuration

### HCCL Communication

- **Technology**: HCCS (Huawei Cache Coherence System)
- **Bandwidth**: Up to 392 GB/s between NPUs
- **Latency**: Sub-microsecond

### Topology

```
4-NPU configuration:
  NPU0 <---> NPU1
    |         |
    v         v
  NPU2 <---> NPU3
```

### Distributed Training

```python
from daca.comm import initialize_parallel, all_reduce

initialize_parallel()
result = all_reduce(tensor)  # Sum across all NPUs
```

## Software Stack

### Required Versions

| Component | Version |
|-----------|---------|
| CANN | 8.3.RC1.alpha003 |
| MindSpore | 2.7.1 |
| MindFormers | 1.7.0 |
| torch_npu | 2.1.0.post10 |

### CANN Operators

CANN 8.3 provides:
- 2000+ optimized operators
- Graph fusion passes
- Auto-tuning for common patterns

### Known CANN Bugs

1. **FlashAttentionFusion V1/V2**: Causes rank mismatch
2. **LayerNorm FP16**: Routes through wrong kernel
3. **BF16 Cast**: Complete failure

## Power and Thermal

- **TDP**: 310W per NPU
- **Idle Power**: ~50W
- **Thermal Throttling**: Starts at 85°C
- **Max Temperature**: 95°C

### Power Management

```bash
# Check NPU power state
npu-smi info -t board
```

## Comparison with NVIDIA A100

| Metric | Ascend 910ProA | NVIDIA A100 |
|--------|----------------|-------------|
| FP16 TFLOPS | 256 | 312 |
| FP32 TFLOPS | 64 | 19.5 |
| BF16 Support | ❌ No | ✅ Yes |
| Memory | 32 GB HBM2 | 40/80 GB HBM2e |
| Bandwidth | 1.2 TB/s | 1.6 TB/s |
| Interconnect | HCCS 392 GB/s | NVLink 600 GB/s |

## References

- [Huawei Ascend Documentation](https://www.hiascend.com/)
- [MindSpore Documentation](https://www.mindspore.cn/)
- [CANN Documentation](https://www.hiascend.com/document)
