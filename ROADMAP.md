# DACA Roadmap

This document outlines the planned development trajectory for DACA.

## Version 0.1.0 (Current - March 2026)

### Focus: Foundation and Bug Workarounds

**Goals:**
- Establish core patching infrastructure
- Workaround critical CANN bugs
- Enable basic AI model training
- Pure Python implementation

**Deliverables:**
- [x] bf16 → fp16 transparent shim
- [x] SiLU/SwiGLU activation injection
- [x] LayerNorm fp32 fast-path
- [x] FlashAttention fusion disable
- [x] Hardware probe tool
- [x] Basic benchmark suite

## Version 0.2.0 (Planned - Q2 2026)

### Focus: Performance Optimization

**Goals:**
- Improve operator performance
- Add kernel selection heuristics
- Optimize memory allocation patterns

**Planned Features:**
- [ ] Auto-tuning framework for op configurations
- [ ] Memory pool for reduced allocation overhead
- [ ] Fused kernel emulation for common patterns
- [ ] Improved MatMul tiling strategies
- [ ] Attention pattern optimization (kv-cache, sliding window)

**Technical Details:**
```python
# Planned API
from daca.autotune import AutoTuner

tuner = AutoTuner()
tuner.tune_matmul(m=1024, n=1024, k=1024)  # Find optimal config
tuner.save("tuning_results.json")
```

## Version 0.3.0 (Planned - Q3 2026)

### Focus: Model Support Expansion

**Goals:**
- Support more model architectures
- Improve MindFormers integration
- Add model-specific optimizations

**Planned Features:**
- [ ] LLaMA architecture optimizations
- [ ] Transformer training recipes
- [ ] Gradient checkpointing helpers
- [ ] Mixed precision training utilities
- [ ] Model profiling tools

**Technical Details:**
```python
# Planned API
from daca.models import LLaMAConfig, LLaMAModel

config = LLaMAConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_hidden_layers=32,
)
model = LLaMAModel(config)

# Automatic optimization
from daca.training import optimize_for_npu
model = optimize_for_npu(model)
```

## Version 0.4.0 (Planned - Q4 2026)

### Focus: Distributed Training

**Goals:**
- Multi-NPU training support
- Distributed checkpoint handling
- Communication optimization

**Planned Features:**
- [ ] Data parallel training helpers
- [ ] Tensor parallel primitives
- [ ] Pipeline parallel support
- [ ] Gradient compression
- [ ] Async communication overlap

**Technical Details:**
```python
# Planned API
from daca.distributed import DistributedManager

dm = DistributedManager(backend="hccl")
dm.setup()

# Automatic gradient sync
model = dm.wrap_model(model)
optimizer = dm.wrap_optimizer(optimizer)
```

## Version 0.5.0 (Planned - Q1 2027)

### Focus: Inference Optimization

**Goals:**
- High-performance inference
- Model quantization support
- Batching strategies

**Planned Features:**
- [ ] INT8 quantization helpers
- [ ] KV-cache management
- [ ] Continuous batching
- [ ] Speculative decoding
- [ ] Paged attention emulation

**Technical Details:**
```python
# Planned API
from daca.inference import InferenceEngine, QuantizationConfig

quant_config = QuantizationConfig(mode="int8")
engine = InferenceEngine(model, quant_config)
outputs = engine.generate(prompts, max_length=512)
```

## Version 1.0.0 (Planned - 2027)

### Focus: Production Release

**Goals:**
- Stable API guarantees
- Comprehensive documentation
- Production deployment support

**Planned Features:**
- [ ] API stability guarantees
- [ ] Comprehensive model zoo
- [ ] Deployment containers
- [ ] Monitoring and observability
- [ ] Performance SLAs

## Long-term Vision

### DACA as the Standard Ascend Platform

**Mission:** Make Ascend NPUs as accessible as NVIDIA GPUs for AI development.

**Key Initiatives:**

1. **Ecosystem Compatibility**
   - Seamless PyTorch model migration
   - HuggingFace integration
   - ONNX model support

2. **Performance Leadership**
   - Match or exceed CUDA performance for key workloads
   - Hardware-specific optimizations
   - Continuous benchmark suite

3. **Developer Experience**
   - One-line model deployment
   - Comprehensive debugging tools
   - Active community support

4. **Enterprise Features**
   - Multi-tenant isolation
   - Security hardening
   - Compliance certifications

## Contributing to the Roadmap

We welcome community input on roadmap priorities! Ways to contribute:

1. **Feature Requests** - Open an issue with the `feature-request` label
2. **Use Cases** - Share your DACA use cases to help prioritize
3. **Performance Data** - Submit benchmark results from your hardware
4. **Code Contributions** - PRs welcome for any roadmap items

## Release Schedule

| Version | Target Date | Status |
|---------|-------------|--------|
| 0.1.0 | March 2026 | Released |
| 0.2.0 | Q2 2026 | Planning |
| 0.3.0 | Q3 2026 | Conceptual |
| 0.4.0 | Q4 2026 | Conceptual |
| 0.5.0 | Q1 2027 | Conceptual |
| 1.0.0 | 2027 | Conceptual |

---

*This roadmap is subject to change based on community feedback, hardware availability, and CANN updates.*
