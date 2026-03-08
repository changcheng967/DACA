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
| FlashAttentionScore (forward) | ✅ Working | Yes | - | Forward pass works |
| **FlashAttentionScore (backward)** | ❌ **NO BACKWARD** | No | Chunked attention | Atlas A2 only, 910ProA has NO backward |
| **Chunked Online Softmax** | ✅ Working | No | DaCAAttention | FlashAttention-equivalent, pure MindSpore |
| ScaledDotProductAttention | ✅ Working | No | BMM path | Manual implementation (OOM for large S) |
| MultiHeadAttention | ✅ Working | Yes | - | MindSpore builtin |
| CrossAttention | ✅ Working | Yes | - | Encoder-decoder |
| SelfAttention | ✅ Working | Yes | - | Standard |
| Sliding Window Attention | ⚠️ Partial | No | Manual mask | Custom implementation |

#### ⚠️ CRITICAL: FlashAttentionScore Has NO Backward Pass on 910ProA

The native `ops.FlashAttentionScore` kernel is **forward-only** on Ascend 910ProA. The backward
pass (needed for training) is only available on Atlas A2 hardware.

**Impact**: Using native FlashAttention during training will crash with:
```
RuntimeError: The gradient operator [FlashAttentionScoreGrad] not found
```

**Solution**: Use `DaCAAttention` which implements the FlashAttention algorithm
in pure MindSpore ops that all have backward support on 910ProA.

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
3. **Use DaCAAttention** for all attention (FlashAttention-equivalent, supports backward)
4. **Avoid native FlashAttentionScore** during training (no backward on 910ProA)
5. **Avoid BF16** completely
6. **Test new ops** before assuming they work

## Memory Comparison: Attention Methods

For Qwen3-8B style attention (seq=4096, num_heads=32, head_dim=128):

| Method | Memory per layer | Total 36 layers | Fits 32GB? | Backward? |
|--------|-----------------|-----------------|------------|-----------|
| Naive BMM-Softmax-BMM | ~1 GB | ~36 GB | ❌ **NO** | ✅ Yes |
| Chunked (chunk=256) | ~4 MB | ~144 MB | ✅ **YES** | ✅ Yes |
| Native FlashAttention | ~0 (fused) | ~0 | N/A | ❌ **NO** |

### Why Naive Attention OOMs

The naive attention implementation materializes the full attention matrix:

```python
# Naive attention - DO NOT USE for large sequences
scores = q @ k.transpose(-1, -2) * scale  # [B, H, S, S] - materializes full matrix!
attn = softmax(scores)
output = attn @ v
```

For S=4096, H=32, batch=1:
- scores shape: [1, 32, 4096, 4096] = 536M elements
- FP16: 536M × 2 bytes = **~1 GB per layer**
- 36 layers × 1 GB = **~36 GB total** > 32 GB HBM → OOM

### Chunked Online Softmax (DaCAAttention)

Instead of materializing the full S×S matrix, process in chunks:

```python
# Chunked attention - DaCAAttention
for q_block in range(0, S, chunk_size):
    for kv_block in range(0, S, chunk_size):
        # Only materialize [B, H, chunk, chunk] at a time
        scores = q_block @ kv_block.T * scale  # [B, H, 256, 256] - tiny!
        # Online softmax update...
```

Memory: O(chunk²) instead of O(S²) = 256² vs 4096² = **256KB vs 1GB**

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
