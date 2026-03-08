# DACA Changelog

All notable changes to DACA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-08

### Fixed

#### CRITICAL: Chunked Online Softmax Attention
- **Memory-efficient attention for 910ProA**: Replaced naive BMM-Softmax-BMM with
  chunked online softmax attention (FlashAttention algorithm in pure MindSpore)
- **OOM fix for large sequences**: Naive attention materializes full S×S matrix.
  For Qwen3-8B (S=4096, H=32): ~1GB/layer × 36 layers = 36GB > 32GB HBM → OOM.
  Chunked approach uses only ~4MB/layer.
- **Backward pass support**: Native FlashAttentionScore has NO backward pass on
  910ProA (Atlas A2 only). Chunked implementation uses only ops with gradients.
- Added `DaCAAttention` class with:
  - Chunked online softmax algorithm (Milakov & Gimelshein 2018)
  - GQA support (different num_heads vs num_kv_heads)
  - Causal masking with block-skip optimization
  - Numerical stability via fp32 upcast for softmax
  - Both [B,H,S,D] and [B,S,H,D] input layouts

### Changed

#### Documentation Updates
- Updated `daca/docs/operator_gap.md`:
  - Documented FlashAttentionScore NO backward pass on 910ProA
  - Added memory comparison table (naive vs chunked vs native)
  - Explained chunked attention algorithm

#### Neural Network Module (`daca.nn`)
- `DaCAAttention` is now the primary attention class
- `FlashAttention` is an alias for `DaCAAttention` (backward compatible)
- `scaled_dot_product_attention` now uses chunked attention

### Memory Comparison

| Method | Memory per layer (S=4096, H=32) | Total 36 layers | Fits 32GB? |
|--------|--------------------------------|-----------------|------------|
| Naive BMM-Softmax-BMM | ~1GB | ~36GB | **NO** |
| Chunked (chunk=256) | ~4MB | ~144MB | **YES** |
| Native FlashAttention | ~0 (fused) | ~0 | N/A (broken) |

### Technical Details

The chunked online softmax algorithm processes Q and K/V in small tiles:

```
For each q_block (256 tokens):
    Initialize: m = -inf (running max), l = 0 (running sum), o = 0 (running output)
    For each kv_block (256 tokens):
        s = q_block @ k_block.T * scale        # [B,H,256,256] — only 256KB!
        m_new = max(m, rowmax(s))
        p = exp(s - m_new)                      # numerically stable
        l_new = exp(m - m_new) * l + rowsum(p)
        o = rescale(o) + p @ v_block
        m, l = m_new, l_new
    output[q_block_range] = o
```

Memory: O(q_chunk × kv_chunk) = O(256 × 256) instead of O(4096 × 4096).

## [0.1.0] - 2026-03-08

### Added

#### Core Framework
- Initial release of DACA (DaVinci Accelerated Compute Architecture)
- `daca.patch()` - Apply all compatibility patches in one call
- `daca.unpatch()` - Remove all patches and restore original state
- `daca.info()` - Display DACA banner and environment information
- `daca.benchmark()` - Run performance benchmarks
- `daca.is_patched()` - Check if patches are currently applied

#### Runtime Module (`daca.runtime`)
- Hardware detection without sudo
  - `detect_npu()` - Detect Ascend NPUs
  - `get_npu_info()` - Get NPU specs (cores, memory, TFLOPS)
  - `check_cann_version()` - Verify CANN compatibility
  - `is_openi_env()` - Detect OpenI VM environment
- Device management
  - `Device` class with context manager support
  - `set_device()`, `get_device()`, `device_count()`
- bf16 → fp16 transparent shim
  - `BFloat16Shim` class managing the patch
  - Monkey-patch `Tensor.astype` to intercept bf16
  - `is_bf16_supported()` returns False (documented limitation)
- Memory tracking
  - `MemoryTracker` class for allocation monitoring
  - `get_memory_usage()`, `get_max_memory_allocated()`

#### BLAS Module (`daca.blas`)
- MatMul with workspace handling
  - `matmul()` wrapper with shape validation
  - Handle 4D attention shapes (batch, heads, seq, dim)
  - Workspace size detection and workarounds
- BatchMatMul
  - `bmm()` for 3D tensors
  - `batch_matmul()` alias
- Workspace management
  - `WorkspaceManager` for safe allocation
  - Pre-allocated workspace pools

#### Neural Network Module (`daca.nn`)
- FlashAttention wrapper
  - `FlashAttention` using native kernel when available
  - BMM fallback path for unsupported configurations
  - GQA support with repeat_kv
- LayerNorm with fp32 upcast
  - `LayerNorm` that casts to fp32, normalizes, casts back
  - Avoids CANN fusion bug that crashes fp16 LayerNorm
- RMSNorm manual decomposition
  - `RMSNorm` using rsqrt + mul + mul
  - No dependency on potentially broken ops
- Missing activations
  - `silu(x)` = x * sigmoid(x)
  - `swiglu(x)` = split + silu + mul
  - All native activations re-exported
- Rotary embeddings
  - `RotaryEmbedding` with position frequencies
  - `apply_rotary_pos_emb()` for attention
- Embedding wrapper
  - `Embedding` using ops.Gather
  - Handle padding indices
- Numerically stable softmax
  - `softmax()` with fp32 upcast
  - `scaled_dot_product_attention()` helper

#### Communication Module (`daca.comm`)
- Multi-NPU helpers
  - `initialize_parallel()` - Setup HCCL
  - `all_reduce()`, `all_gather()`, `broadcast()`
  - `get_rank()`, `get_world_size()`

#### Compile Module (`daca.compile`)
- Graph mode workarounds
  - `enable_graph_mode()` with safe env vars
  - `GraphCell` wrapper for safe compilation
- Fusion control
  - `disable_flash_attention_fusion()` - Disable V1/V2
  - `FusionConfig` for granular control

#### Compatibility Module (`daca.compat`)
- CUDA shim
  - Shim `torch.cuda.is_available()` → False
  - Shim `torch.cuda.device_count()` → NPU count
  - Device string rewriting
- MindSpore patches
  - Add `ops.SiLU` to namespace
  - Add `ops.SwiGLU` to namespace
  - Patch LayerNorm for fp32 upcast
- MindFormers patches
  - Patch bf16 configs → fp16
  - Fix broken LayerNorm usage
- Config rewriting
  - `rewrite_config()` - bf16 → fp16 in dicts
  - `ConfigRewriter` for JSON/YAML files

#### Autotune Module (`daca.autotune`)
- Op benchmarking
  - `benchmark_op()` - Time an operation
  - `auto_tune_matmul()` - Find optimal block sizes
  - `BenchmarkResult` dataclass

#### Tools
- `tools/probe.py` - Hardware capability probe
  - Tests all known working/broken ops
  - Generates probe_data.json with results
- `tools/doctor.py` - Diagnose environment issues
  - Check CANN version
  - Verify MindSpore installation
  - Detect hardware issues

#### Benchmarks
- `benchmarks/bench_matmul.py` - MatMul benchmarks
- `benchmarks/bench_attention.py` - Attention benchmarks
- `benchmarks/bench_layernorm.py` - LayerNorm benchmarks
- `benchmarks/bench_activations.py` - Activation benchmarks
- `benchmarks/bench_all.py` - Run all benchmarks

#### Tests
- `tests/test_runtime.py` - Runtime module tests
- `tests/test_blas.py` - BLAS module tests
- `tests/test_nn.py` - NN module tests
- `tests/test_compat.py` - Compat module tests
- `tests/test_compile.py` - Compile module tests
- `tests/test_integration.py` - End-to-end tests

#### Examples
- `examples/quickstart.py` - Basic usage
- `examples/train_with_daca.py` - Training example
- `examples/patch_existing_code.py` - Migration example

#### Documentation
- `daca/docs/hardware_profile.md` - 910ProA specs
- `daca/docs/operator_gap.md` - Missing ops and workarounds
- `daca/docs/vs_cuda.md` - Migration guide from CUDA
- `daca/docs/optimization_guide.md` - Performance tips

### Known Issues

1. **bf16 completely unsupported** - `aclnnCastGetWorkspaceSize` crash. Hardware limitation.
2. **ops.SiLU missing** - Not in MindSpore 2.7.1 ops namespace. Manual `x * sigmoid(x)` works.
3. **LayerNorm fp16 broken** - CANN incorrectly routes through FlashAttentionScore kernel. fp32 LayerNorm works.
4. **CANN auto-fusion bugs** - CANN 8.3 aggressively fuses unrelated ops through FlashAttention paths.

### Hardware Support

- Target: 4× Ascend 910ProA (TSMC N7+)
- 32× DaVinci cores per NPU
- 256 TFLOPS FP16 per NPU
- 32GB HBM2 per NPU
- CANN 8.3.RC1.alpha003
- MindSpore 2.7.1
- aarch64-linux, no root access required
