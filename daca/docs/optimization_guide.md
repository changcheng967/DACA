# DACA Optimization Guide

## Overview

This guide provides performance optimization tips for running AI workloads on Ascend 910ProA NPUs with DACA.

## General Principles

### 1. Use FP16 Whenever Possible

FP16 is the native precision for Ascend 910ProA:

```python
import mindspore as ms
import mindspore.common.dtype as mstype

# Use FP16 for tensors
x = ms.Tensor(data, dtype=mstype.float16)

# FP16 for model parameters
model = MyModel()
model = model.to_float(mstype.float16)
```

**Exception**: LayerNorm should use FP32 internally (DACA handles this).

### 2. Avoid BF16 Completely

BF16 is NOT supported on 910ProA. DACA intercepts and converts to FP16:

```python
# This would crash without DACA
x = ms.Tensor(data, ms.bfloat16)  # DACA converts to FP16

# Better: Be explicit
x = ms.Tensor(data, ms.float16)
```

### 3. Use Graph Mode for Inference

Graph mode provides better optimization:

```python
from mindspore import context
from daca.compile import enable_graph_mode

enable_graph_mode()  # Sets safe env vars

# Or manually:
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
```

### 4. Prefer Native FlashAttention

The native FlashAttention kernel works on 910ProA:

```python
from daca.nn import FlashAttention

attn = FlashAttention(head_dim=64, num_heads=32)
output = attn(query, key, value)
```

## Memory Optimization

### Reduce Memory Usage

```python
# 1. Use FP16 instead of FP32
model = model.to_float(mstype.float16)

# 2. Use gradient checkpointing for large models
from mindspore.nn import Cell
# Implement custom checkpointing or use MindSpore's

# 3. Clear workspace between operations
from daca.blas import clear_workspace_pool
clear_workspace_pool()
```

### Monitor Memory Usage

```python
from daca.runtime import get_memory_usage, MemoryTracker

# Check current usage
usage = get_memory_usage()
print(f"Using {usage['allocated'] / 1e9:.2f} GB")

# Track memory during operations
tracker = MemoryTracker()
with tracker.track("forward"):
    output = model(input)
print(tracker.summary())
```

### Pre-allocate Workspace

For large operations, pre-allocate workspace:

```python
from daca.blas import preallocate_workspace

# Pre-allocate 512MB workspace
preallocate_workspace(512 * 1024 * 1024)
```

## Compute Optimization

### Use Optimized Kernels

| Operation | Recommended Approach | Performance |
|-----------|---------------------|-------------|
| MatMul | `ops.matmul` | 256 TFLOPS |
| Attention | `FlashAttention` | 2x faster than BMM |
| Softmax | `daca.nn.softmax` | Stable |
| LayerNorm | `daca.nn.LayerNorm` | FP32 safe |
| SiLU | `daca.nn.silu` | Manual but fast |

### Batch Operations

Process data in batches to maximize throughput:

```python
# Bad: Process one at a time
for sample in data:
    output = model(sample)

# Good: Batch processing
batch = pack_batch(data, batch_size=32)
output = model(batch)
```

### Avoid Small Operations

Small operations have overhead:

```python
# Bad: Many small operations
for i in range(100):
    y = ops.add(x[i], 1)

# Good: Vectorized operation
y = ops.add(x, 1)
```

## Model Optimization

### Layer Norm Optimization

DACA's LayerNorm uses FP32 upcast for stability:

```python
from daca.nn import LayerNorm

# This automatically upcasts to FP32, then back
ln = LayerNorm(hidden_size=768)
output = ln(hidden_states)
```

### Attention Optimization

Use FlashAttention for transformers:

```python
from daca.nn import FlashAttention, scaled_dot_product_attention

# Option 1: FlashAttention class
attn = FlashAttention(head_dim=64, num_heads=32)
output = attn(q, k, v)

# Option 2: Functional API
output = scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Use RMSNorm Instead of LayerNorm

RMSNorm is faster (no mean subtraction):

```python
from daca.nn import RMSNorm

# Replace LayerNorm with RMSNorm where possible
norm = RMSNorm(hidden_size=768)
output = norm(hidden_states)
```

## Graph Mode Optimization

### Enable Graph Mode Safely

```python
from daca.compile import enable_graph_mode, set_safe_env

# Set safe environment variables first
set_safe_env()

# Enable graph mode
enable_graph_mode()
```

### Static Shape Prefered

Graph mode prefers static shapes:

```python
# Prefer fixed shapes
x = ms.Tensor(shape=(batch, seq, dim), dtype=mstype.float16)

# Avoid dynamic shapes when possible
```

### Use Cell for Graph Compilation

```python
from mindspore import nn

class MyModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layers = nn.CellList([...])

    def construct(self, x):
        # Static computation graph
        return self.layers(x)

model = MyModel()
# Compile once, run many times
```

## Data Pipeline Optimization

### Efficient Data Loading

```python
from mindspore.dataset import GeneratorDataset

# Use efficient data loading
dataset = GeneratorDataset(
    data_generator,
    column_names=["data", "label"],
    shuffle=True,
    num_parallel_workers=4,
)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=2)
```

### Avoid CPU Bottlenecks

```python
# Prefetch to overlap data loading and compute
dataset = dataset.prefetch(2)

# Use multiple workers
dataset = GeneratorDataset(..., num_parallel_workers=8)
```

## Benchmarking

### Profile Your Model

```python
from daca.autotune import benchmark_op, BenchmarkResult

# Benchmark an operation
result = benchmark_op(
    lambda: model(input),
    name="forward_pass",
    warmup=10,
    repeat=100,
)
print(result)
```

### Run All Benchmarks

```python
from daca.autotune import run_all_benchmarks

results = run_all_benchmarks(verbose=True)
```

### Identify Bottlenecks

```python
from daca.runtime import MemoryTracker

tracker = MemoryTracker()
with tracker.track("layer1"):
    x = layer1(x)
with tracker.track("layer2"):
    x = layer2(x)
with tracker.track("layer3"):
    x = layer3(x)

# Find which layer uses most memory/time
print(tracker.summary())
```

## Common Performance Issues

### Issue: Slow First Run

**Cause**: Graph compilation time
**Solution**: Warmup with a small batch

```python
# Warmup
dummy = ms.Tensor(ms.numpy.zeros((1, 128, 768)), ms.float16)
_ = model(dummy)  # Compile graph

# Now fast
output = model(real_input)
```

### Issue: Memory Fragmentation

**Cause**: Frequent allocation/deallocation
**Solution**: Pre-allocate and reuse

```python
# Pre-allocate buffers
buffer = ms.Tensor(ms.numpy.zeros((batch, seq, dim)), ms.float16)

# Reuse buffer
def forward(x):
    ops.assign(buffer, x)  # Reuse
    return model(buffer)
```

### Issue: CANN Fusion Crashes

**Cause**: Aggressive fusion passes
**Solution**: Disable problematic fusions

```python
from daca.compile import disable_flash_attention_fusion

disable_flash_attention_fusion()
```

## Performance Checklist

- [ ] Using FP16 everywhere except LayerNorm
- [ ] Avoiding BF16 completely
- [ ] Using FlashAttention for attention
- [ ] Using Graph mode for inference
- [ ] Pre-allocating workspace for large ops
- [ ] Batching operations
- [ ] Using efficient data loading
- [ ] Disabling problematic CANN fusions
- [ ] Profiling to find bottlenecks
- [ ] Warming up graph mode

## Expected Performance

On 4× Ascend 910ProA with optimal configuration:

| Model | Batch Size | Seq Length | Throughput |
|-------|------------|------------|------------|
| LLaMA-7B | 8 | 2048 | ~1500 tokens/s |
| LLaMA-13B | 4 | 2048 | ~800 tokens/s |
| BERT-Large | 32 | 512 | ~2000 samples/s |

## Further Reading

- [Hardware Profile](hardware_profile.md) - 910ProA specifications
- [Operator Gap](operator_gap.md) - Working/broken operations
- [CUDA Migration](vs_cuda.md) - PyTorch to MindSpore guide
