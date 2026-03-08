"""DACA Compatibility Module.

Patching third-party libraries for Ascend compatibility.

Example:
    from daca.compat import rewrite_config, ConfigRewriter
"""

from daca.compat.cuda_shim import (
    shim_cuda_api,
    unshim_cuda_api,
    get_npu_count_as_cuda,
)
from daca.compat.mindspore_patches import (
    apply_all as apply_ms_patches,
    revert_all as revert_ms_patches,
    patch_ops_namespace,
    patch_layernorm,
)
from daca.compat.mindformers_patches import (
    apply_all as apply_mf_patches,
    revert_all as revert_mf_patches,
    patch_bf16_config,
    patch_layernorm_usage,
)
from daca.compat.config_rewriter import (
    rewrite_config,
    ConfigRewriter,
)

__all__ = [
    # CUDA shim
    "shim_cuda_api",
    "unshim_cuda_api",
    "get_npu_count_as_cuda",
    # MindSpore patches
    "apply_ms_patches",
    "revert_ms_patches",
    "patch_ops_namespace",
    "patch_layernorm",
    # MindFormers patches
    "apply_mf_patches",
    "revert_mf_patches",
    "patch_bf16_config",
    "patch_layernorm_usage",
    # Config rewriter
    "rewrite_config",
    "ConfigRewriter",
]
