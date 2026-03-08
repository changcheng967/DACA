# CUDA to Ascend Migration Guide

## Overview

This guide helps you migrate PyTorch/CUDA code to MindSpore/Ascend using DACA.

## Key Differences

### Terminology Mapping

| CUDA/PyTorch | Ascend/MindSpore | Notes |
|--------------|------------------|-------|
| GPU | NPU | Neural Processing Unit |
| CUDA | CANN | Compute Architecture |
| torch.cuda | mindspore.context | Device management |
| torch.Tensor | mindspore.Tensor | Tensor class |
| torch.nn | mindspore.nn | Neural network layers |
| torch.optim | mindspore.nn.Adam, etc. | Optimizers |
| DataLoader | Dataset + DataLoader | Data pipeline |
| BF16 | ❌ Not supported | Use FP16 |

## Quick Migration

### Step 1: Apply DACA Patches

```python
import daca
daca.patch()

# Your existing code now works with workarounds applied
```

### Step 2: Replace CUDA-specific Code

```python
# Before (PyTorch)
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# After (MindSpore)
import mindspore as ms
from mindspore import context
context.set_context(device_target="Ascend", device_id=0)
# No explicit .to() needed - context handles device
```

### Step 3: Convert Tensors

```python
# Before
x = torch.tensor([1.0, 2.0], dtype=torch.float16, device="cuda")

# After
import mindspore.common.dtype as mstype
x = ms.Tensor([1.0, 2.0], dtype=mstype.float16)
```

## Common Patterns

### Model Definition

```python
# PyTorch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.norm(self.linear(x))

# MindSpore
import mindspore.nn as nn
from daca.nn import LayerNorm  # DACA's safe LayerNorm

class MyModel(nn.Cell):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Dense(hidden_size, hidden_size)
        self.norm = LayerNorm(hidden_size)  # FP32 upcast internally

    def construct(self, x):
        return self.norm(self.linear(x))
```

### Training Loop

```python
# PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

# MindSpore
from mindspore import Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits

optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4)
criterion = SoftmaxCrossEntropyWithLogits()

model_for_train = Model(model, criterion, optimizer)
model_for_train.train epochs, dataset
```

### Data Loading

```python
# PyTorch
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(data, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# MindSpore
from mindspore.dataset import TensorDataset, GeneratorDataset

dataset = GeneratorDataset(data_generator, ["data", "label"])
dataset = dataset.batch(32).shuffle(buffer_size=1000)
```

## API Reference Mapping

### Activation Functions

| PyTorch | MindSpore | DACA Notes |
|---------|-----------|------------|
| `F.relu` | `ops.relu` | ✅ Direct |
| `F.gelu` | `ops.gelu` | ✅ Direct |
| `F.silu` | `ops.silu` | ⚠️ Injected by DACA |
| `F.softmax` | `ops.softmax` | ✅ Direct |
| `F.sigmoid` | `ops.sigmoid` | ✅ Direct |

### Layer Types

| PyTorch | MindSpore | Notes |
|---------|-----------|-------|
| `nn.Linear` | `nn.Dense` | Same concept |
| `nn.LayerNorm` | `nn.LayerNorm` | ⚠️ Use daca.nn.LayerNorm |
| `nn.Dropout` | `nn.Dropout` | Same |
| `nn.Embedding` | `nn.Embedding` | Same |
| `nn.Conv2d` | `nn.Conv2d` | Same |
| `nn.MultiheadAttention` | `nn.MultiheadAttention` | Same |

### Tensor Operations

| PyTorch | MindSpore | Notes |
|---------|-----------|-------|
| `x.cuda()` | No equivalent | Use context |
| `x.cpu()` | No equivalent | Always CPU |
| `x.to(device)` | No equivalent | Context-based |
| `x.half()` | `x.astype(mstype.float16)` | Explicit |
| `x.float()` | `x.astype(mstype.float32)` | Explicit |
| `x.bfloat16()` | ❌ Not supported | Use FP16 |
| `x.shape` | `x.shape` | Same |
| `x.view()` | `x.reshape()` | Different name |
| `x.permute()` | `x.transpose()` | Different API |

## BF16 Migration

Since BF16 is not supported, all BF16 code must be converted to FP16:

```python
# Before (PyTorch with BF16)
model = model.to(torch.bfloat16)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input)

# After (MindSpore with FP16)
# DACA handles this automatically
context.set_context(device_target="Ascend")
# Use FP16 explicitly
x = x.astype(mstype.float16)
output = model(x)
```

### DACA Auto-Conversion

DACA intercepts BF16 requests:

```python
import daca
daca.patch()

# This would normally crash, but DACA converts to FP16
x = ms.Tensor([1.0], ms.bfloat16)  # Actually creates FP16
```

## SiLU/SwiGLU Migration

PyTorch has native SiLU, but MindSpore 2.7.1 doesn't:

```python
# PyTorch
import torch.nn.functional as F
x = F.silu(x)

# MindSpore with DACA
import daca
daca.patch()

import mindspore.ops as ops
x = ops.silu(x)  # Injected by DACA
```

## Distributed Training

### PyTorch DDP

```python
# PyTorch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = DDP(model, device_ids=[local_rank])
```

### MindSpore Parallel

```python
# MindSpore
from mindspore.communication import init
from mindspore import context

init()
context.set_auto_parallel_context(parallel_mode="data_parallel")
```

### DACA Helper

```python
from daca.comm import initialize_parallel, all_reduce

initialize_parallel()  # Sets up HCCL
result = all_reduce(tensor, op="sum")
```

## Debugging

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `aclnnCastGetWorkspaceSize failed` | BF16 used | DACA converts to FP16 |
| `FlashAttentionScore rank error` | FP16 LayerNorm | Use daca.nn.LayerNorm |
| `module has no attribute 'SiLU'` | Missing op | DACA patches it |
| `Out of memory` | Memory limit | Use smaller batch |

### Checking GPU vs NPU

```python
# PyTorch
if torch.cuda.is_available():
    print("CUDA available")

# MindSpore with DACA
from daca.runtime import detect_npu
if detect_npu():
    print("Ascend NPU available")
```

## Performance Tips

1. **Use Graph Mode** for inference
   ```python
   context.set_context(mode=context.GRAPH_MODE)
   ```

2. **Use FP16** for training
   ```python
   context.set_context(device_target="Ascend")
   ```

3. **Avoid BF16** completely

4. **Use FlashAttention** for transformers
   ```python
   from daca.nn import FlashAttention
   attn = FlashAttention(head_dim=64, num_heads=32)
   ```

5. **Batch operations** when possible

## Full Example

```python
# PyTorch original
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)

model = Transformer(512, 8).cuda()
x = torch.randn(32, 10, 512, device="cuda", dtype=torch.float16)
out = model(x)

# MindSpore + DACA migrated
import daca
daca.patch()

import mindspore as ms
import mindspore.nn as nn
from daca.nn import LayerNorm

class Transformer(nn.Cell):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = LayerNorm(d_model)

    def construct(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)

context.set_context(device_target="Ascend")
model = Transformer(512, 8)
x = ms.Tensor(ms.numpy.random.randn(32, 10, 512), ms.float16)
out = model(x)
```
