# DACA Operator Gap Analysis

## Overview

This document tracks the gap between common AI operators and their availability on Ascend 910ProA with CANN 8.3. It serves as a reference for understanding which operations work natively and which require workarounds.

## Operator Status

### Matrix Operations

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| MatMul | ✅ Working | Yes | - | All sizes including 4D |
| BatchMatMul | ✅ Working | Yes | - | 3D tensors |
| AddMM | ✅ Working | Yes | - | beta*x + alpha*(mat1@mat2) |
| Linear | ✅ Working | Yes | - | bias add supported |

### Normalization

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| LayerNorm FP32 | ✅ Working | Yes | - | Recommended |
| LayerNorm FP16 | ❌ Broken | Yes | FP32 upcast | CANN fusion bug |
| BatchNorm | ✅ Working | Yes | - | All variants |
| InstanceNorm | ✅ Working | Yes | - | - |
| GroupNorm | ✅ Working | Yes | - | - |
| RMSNorm | ✅ Working | No | Manual decomp | rsqrt + mul + mul |

### Activations

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| Sigmoid | ✅ Working | Yes | - | Fast |
| ReLU | ✅ Working | Yes | - | Fast |
| GeLU | ✅ Working | Yes | - | ~1.2ms |
| FastGeLU | ✅ Working | Yes | - | Faster approx |
| Tanh | ✅ Working | Yes | - | - |
| Mish | ✅ Working | Yes | - | - |
| HSwish | ✅ Working | Yes | - | Mobile-friendly |
| HSigmoid | ✅ Working | Yes | - | Mobile-friendly |
| PReLU | ✅ Working | Yes | - | Parameterized |
| SeLU | ✅ Working | Yes | - | Self-normalizing |
| **SiLU** | ❌ Missing | No | `x * sigmoid(x)` | Not in ops namespace |
| SwiGLU | ❌ Missing | No | Manual split+silu+mul | FFN activation |
| GeGLU | ❌ Missing | No | Manual split+gelu+mul | FFN activation |
| LeakyReLU | ⚠️ Check | ? | - | Verify availability |
| ELU | ⚠️ Check | ? | - | Verify availability |
| CELU | ⚠️ Check | ? | - | Verify availability |

### Attention

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| FlashAttentionScore | ✅ Working | Yes | - | Native kernel works |
| ScaledDotProductAttention | ✅ Working | No | BMM path | Manual implementation |
| MultiHeadAttention | ✅ Working | Yes | - | MindSpore builtin |
| CrossAttention | ✅ Working | Yes | - | Encoder-decoder |
| SelfAttention | ✅ Working | Yes | - | Standard |
| Sliding Window Attention | ⚠️ Partial | No | Manual mask | Custom implementation |
| FlashAttention-2 | ❓ Unknown | ? | - | Check CANN version |

### Embeddings

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| Embedding | ✅ Working | Yes | - | ops.Gather |
| PositionalEmbedding | ✅ Working | No | Manual | Add position tensor |
| RotaryEmbedding | ✅ Working | No | Manual | RoPE implementation |
| ALiBi | ⚠️ Partial | No | Manual | Bias-based |
| Relative PE | ⚠️ Partial | No | Manual | Custom implementation |

### Softmax

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| Softmax FP16 | ✅ Working | Yes | - | Stable |
| Softmax FP32 | ✅ Working | Yes | - | More precise |
| LogSoftmax | ✅ Working | Yes | - | For loss computation |
| Softmax with mask | ✅ Working | No | Add mask before | Manual |

### Pooling

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| MaxPool2D | ✅ Working | Yes | - | - |
| AvgPool2D | ✅ Working | Yes | - | - |
| MaxPool1D | ✅ Working | Yes | - | - |
| AvgPool1D | ✅ Working | Yes | - | - |
| AdaptiveAvgPool | ✅ Working | Yes | - | - |
| AdaptiveMaxPool | ✅ Working | Yes | - | - |

### Reduction

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| ReduceSum | ✅ Working | Yes | - | - |
| ReduceMean | ✅ Working | Yes | - | - |
| ReduceMax | ✅ Working | Yes | - | - |
| ReduceMin | ✅ Working | Yes | - | - |
| ArgMax | ✅ Working | Yes | - | - |
| ArgMin | ✅ Working | Yes | - | - |

### Data Type Operations

| Operation | Status | Native | Workaround | Notes |
|-----------|--------|--------|------------|-------|
| Cast FP16→FP32 | ✅ Working | Yes | - | - |
| Cast FP32→FP16 | ✅ Working | Yes | - | - |
| **Cast *→BF16** | ❌ Crash | Yes | BF16→FP16 shim | Hardware limitation |
| Cast INT8→FP16 | ✅ Working | Yes | - | - |

## Workaround Implementations

### SiLU (Swish)

```python
# MindSpore 2.7.1 doesn't have ops.SiLU
def silu(x):
    return x * ops.sigmoid(x)
```

**Performance**: ~1.2ms vs theoretical ~0.5ms if native

### SwiGLU

```python
def swiglu(x, dim=-1):
    """SwiGLU activation for FFN."""
    half = x.shape[dim] // 2
    a, b = ops.split(x, (half, half), axis=dim)
    return (a * ops.sigmoid(a)) * b  # silu(a) * b
```

### RMSNorm

```python
def rmsnorm(x, weight, epsilon=1e-6):
    """Manual RMSNorm decomposition."""
    variance = ops.mean(ops.pow(x, 2), axis=-1, keep_dims=True)
    x_norm = x * ops.rsqrt(variance + epsilon)
    return x_norm * weight
```

### LayerNorm FP16 (Bug Workaround)

```python
def layernorm_fp16_safe(x, weight, bias, epsilon=1e-5):
    """LayerNorm with FP32 upcast for Ascend."""
    original_dtype = x.dtype
    x_fp32 = ops.cast(x, mstype.float32)
    normalized = ops.layer_norm(x_fp32, ..., epsilon)
    return ops.cast(normalized, original_dtype)
```

## CANN Fusion Bugs

### FlashAttentionFusion V1/V2

**Symptom**: Rank mismatch errors
**Cause**: CANN aggressively fuses ops through FlashAttention paths
**Fix**: Disable fusion passes

```python
os.environ["MS_DEV_DISABLE_FLASH_ATTENTION_FUSION"] = "1"
os.environ["ASCEND_DISABLE_FLASH_ATTENTION_FUSION"] = "1"
```

### LayerNorm Fusion Bug

**Symptom**: `FlashAttentionScore rank error`
**Cause**: FP16 LayerNorm incorrectly routed through FA kernel
**Fix**: Use FP32 upcast

## Performance Impact of Workarounds

| Operation | Native | Workaround | Overhead |
|-----------|--------|------------|----------|
| SiLU | ~0.5ms | ~1.2ms | 2.4x |
| SwiGLU | ~1.0ms | ~2.5ms | 2.5x |
| RMSNorm | ~0.8ms | ~1.8ms | 2.25x |
| LayerNorm FP16 | ~0.5ms | ~2.0ms (FP32) | 4x |

## Recommendations

1. **Use FP16 everywhere** except LayerNorm
2. **Prefer native ops** when available
3. **Use FlashAttention** kernel (it works!)
4. **Avoid BF16** completely
5. **Test new ops** before assuming they work

## Contributing

If you discover new working or broken operators, please update this document and submit a PR.

### Testing Template

```python
import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops

def test_op(op_name, op_func, input_shape=(16, 64)):
    """Test if an operation works."""
    x = Tensor(ms.numpy.random.randn(*input_shape), ms.float16)
    try:
        start = time.time()
        y = op_func(x)
        elapsed = (time.time() - start) * 1000
        return {"ok": True, "ms": elapsed}
    except Exception as e:
        return {"ok": False, "err": str(e)}

# Example test
result = test_op("silu", lambda x: ops.silu(x))
print(result)
```
