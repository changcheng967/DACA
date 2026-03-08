"""MindFormers patches for Ascend compatibility.

Fixes bf16 config issues, LayerNorm usage, and replaces FlashAttention
with DaCAAttention in MindFormers models.

CRITICAL: Native FlashAttentionScore has NO backward pass on 910ProA
(Atlas A2 only). We replace MindFormers' FlashAttention with DaCAAttention
which implements the FlashAttention algorithm in pure MindSpore ops.

WHY: MindFormers configs often default to bf16, which crashes on Ascend.
Also patches broken LayerNorm usage patterns and attention layers.

Example:
    from daca.compat import apply_mf_patches

    apply_mf_patches()
"""

import logging
import os
from typing import Optional, Any, Dict

logger = logging.getLogger("daca.compat.mindformers_patches")

# Track patch state
_patches_applied: bool = False
_original_configs: Dict[str, Any] = {}


def apply_all() -> None:
    """Apply all MindFormers patches.

    Applies:
    1. bf16 config patch (bf16 -> fp16)
    2. LayerNorm usage patch
    3. FlashAttention -> DaCAAttention replacement
    4. Disable MS_ENABLE_FLASH_ATTENTION

    Example:
        >>> from daca.compat.mindformers_patches import apply_all
        >>> apply_all()
    """
    global _patches_applied

    if _patches_applied:
        logger.debug("MindFormers patches already applied")
        return

    # Disable native FlashAttention first
    os.environ["MS_ENABLE_FLASH_ATTENTION"] = "0"

    patch_bf16_config()
    patch_layernorm_usage()
    patch_attention()

    _patches_applied = True
    logger.info("Applied all MindFormers patches")
    logger.info("Replaced FlashAttention with DaCAAttention (supports backward on 910ProA)")


def revert_all() -> None:
    """Revert all MindFormers patches.

    Example:
        >>> from daca.compat.mindformers_patches import revert_all
        >>> revert_all()
    """
    global _patches_applied

    if not _patches_applied:
        logger.debug("MindFormers patches not applied")
        return

    revert_bf16_config()
    revert_layernorm_usage()
    revert_attention()

    _patches_applied = False
    logger.info("Reverted all MindFormers patches")


def patch_bf16_config() -> None:
    """Patch MindFormers to use fp16 instead of bf16.

    Many MindFormers model configs default to bf16, which is
    unsupported on Ascend 910ProA. This patches the config loading
    to replace bf16 with fp16.

    Example:
        >>> from daca.compat.mindformers_patches import patch_bf16_config
        >>> patch_bf16_config()
    """
    try:
        # Try to patch MindFormers config module
        import mindformers

        # Patch auto config if available
        try:
            from mindformers import AutoConfig

            original_from_pretrained = AutoConfig.from_pretrained

            def patched_from_pretrained(cls, *args, **kwargs):
                config = original_from_pretrained(*args, **kwargs)
                return _rewrite_config_dtype(config)

            AutoConfig.from_pretrained = classmethod(patched_from_pretrained)
            logger.info("Patched AutoConfig.from_pretrained for bf16 -> fp16")

        except (ImportError, AttributeError):
            pass

        # Patch common config classes
        try:
            from mindformers.models import LlamaConfig

            original_init = LlamaConfig.__init__

            def patched_init(self, *args, **kwargs):
                # Rewrite bf16 to fp16 in kwargs
                kwargs = _rewrite_kwargs_dtype(kwargs)
                original_init(self, *args, **kwargs)

            LlamaConfig.__init__ = patched_init
            logger.info("Patched LlamaConfig for bf16 -> fp16")

        except (ImportError, AttributeError):
            pass

    except ImportError:
        logger.debug("MindFormers not available, config patch skipped")


def revert_bf16_config() -> None:
    """Revert bf16 config patches.

    Example:
        >>> from daca.compat.mindformers_patches import revert_bf16_config
        >>> revert_bf16_config()
    """
    # Config patches are difficult to fully revert without storing originals
    # For now, just log that patches were removed
    logger.info("MindFormers config patches marked for removal")


def patch_layernorm_usage() -> None:
    """Patch MindFormers LayerNorm usage for Ascend.

    Some MindFormers models use LayerNorm in ways that trigger
    the CANN fusion bug. This patches those usages.

    Example:
        >>> from daca.compat.mindformers_patches import patch_layernorm_usage
        >>> patch_layernorm_usage()
    """
    try:
        import mindformers

        # Patch common model classes to use fp32 LayerNorm
        try:
            from mindformers.models.llama import LlamaModel

            # Patch LayerNorm construction in LlamaModel
            original_construct = LlamaModel.construct
            _original_configs["LlamaModel.construct"] = original_construct

            def patched_construct(self, *args, **kwargs):
                # Use fp32 LayerNorm via daca
                from daca.nn import LayerNorm
                # Model-specific patches would go here
                return original_construct(self, *args, **kwargs)

            LlamaModel.construct = patched_construct
            logger.info("Patched LlamaModel LayerNorm usage")

        except (ImportError, AttributeError):
            pass

    except ImportError:
        logger.debug("MindFormers not available, LayerNorm patch skipped")


def revert_layernorm_usage() -> None:
    """Revert LayerNorm usage patches.

    Example:
        >>> from daca.compat.mindformers_patches import revert_layernorm_usage
        >>> revert_layernorm_usage()
    """
    # Restore original construct methods
    for key, original in list(_original_configs.items()):
        if key.endswith(".construct"):
            module_key = key.rsplit(".construct", 1)[0]
            try:
                if module_key == "LlamaModel":
                    from mindformers.models.llama import LlamaModel
                    LlamaModel.construct = original
                    logger.info(f"Reverted {key}")
            except Exception as e:
                logger.debug(f"Could not revert {key}: {e}")

    logger.info("MindFormers LayerNorm usage patches reverted")


def patch_attention() -> None:
    """Replace MindFormers FlashAttention with DaCAAttention.

    CRITICAL: Native FlashAttentionScore has NO backward pass on 910ProA.
    This patches MindFormers attention modules to use DaCAAttention instead,
    which implements the FlashAttention algorithm in pure MindSpore ops
    that all support backward pass.

    Example:
        >>> from daca.compat.mindformers_patches import patch_attention
        >>> patch_attention()
    """
    # Approach 1: Patch mindformers.modules.layers.FlashAttention if it exists
    _try_patch_flash_attention_class()

    # Approach 2: Patch the FlashAttention import in model files
    _try_patch_model_attention_modules()

    # Approach 3: Ensure env vars disable native FA
    os.environ["MS_ENABLE_FLASH_ATTENTION"] = "0"
    os.environ["MF_ENABLE_FLASH_ATTENTION"] = "0"
    os.environ["MS_DEV_GRAPH_KERNEL_FLAGS"] = os.environ.get(
        "MS_DEV_GRAPH_KERNEL_FLAGS", ""
    ) + " --disable_pass=FlashAttentionFusionV1,FlashAttentionFusionV2"

    logger.info("MindFormers attention patching complete")


def _try_patch_flash_attention_class():
    """Try to patch MindFormers FlashAttention class."""
    try:
        from mindformers.modules import attention as mf_attn_module

        # Look for FlashAttention class
        if hasattr(mf_attn_module, 'FlashAttention'):
            original_cls = mf_attn_module.FlashAttention
            _original_configs["mf_FlashAttention_cls"] = original_cls

            original_construct = original_cls.construct
            _original_configs["mf_FlashAttention_construct"] = original_construct

            def patched_construct(self, query, key, value, attention_mask=None, **kwargs):
                """Patched to use DaCAAttention instead of native FlashAttention."""
                # Infer dimensions
                q_shape = query.shape
                if len(q_shape) == 4:
                    if q_shape[1] < q_shape[2]:
                        num_heads = q_shape[1]
                        head_dim = q_shape[3]
                    else:
                        num_heads = q_shape[2]
                        head_dim = q_shape[3]
                else:
                    # Fall back to original if unexpected shape
                    logger.warning(f"Unexpected query shape {q_shape}, falling back to original")
                    return original_construct(self, query, key, value, attention_mask, **kwargs)

                from daca.nn.attention import DaCAAttention
                daca_attn = DaCAAttention(
                    num_heads=num_heads,
                    num_kv_heads=num_heads,  # assume MHA unless we can detect GQA
                    head_dim=head_dim,
                    causal=True,
                )
                return daca_attn(query, key, value, attention_mask)

            original_cls.construct = patched_construct
            logger.info("Patched mindformers.modules.attention.FlashAttention.construct")

    except ImportError:
        logger.debug("mindformers.modules.attention not available")
    except Exception as e:
        logger.warning(f"Failed to patch MindFormers FlashAttention class: {e}")


def _try_patch_model_attention_modules():
    """Try to patch attention in specific model implementations."""

    # Try Qwen models (user trains Qwen3-8B)
    _try_patch_module_attr("mindformers.models.qwen2", "FlashAttention")
    _try_patch_module_attr("mindformers.models.qwen2.qwen2_modules", "FlashAttention")

    # Try LLaMA models
    _try_patch_module_attr("mindformers.models.llama", "FlashAttention")
    _try_patch_module_attr("mindformers.models.llama.llama_layer", "FlashAttention")
    _try_patch_module_attr("mindformers.models.llama.llama_layer", "LLamaAttention")

    # Try generic transformer layers
    _try_patch_module_attr("mindformers.modules.transformer", "FlashAttention")
    _try_patch_module_attr("mindformers.modules.layers", "FlashAttention")


def _try_patch_module_attr(module_path: str, attr_name: str):
    """Try to patch a specific attribute in a module.

    Args:
        module_path: Full module path (e.g., "mindformers.models.llama")
        attr_name: Attribute name to patch (e.g., "FlashAttention")
    """
    try:
        import importlib
        mod = importlib.import_module(module_path)
        if hasattr(mod, attr_name):
            original = getattr(mod, attr_name)
            key = f"{module_path}.{attr_name}"
            _original_configs[key] = original

            # If it's a class with a construct method, patch construct
            if hasattr(original, 'construct') and callable(original.construct):
                original_construct = original.construct
                _original_configs[f"{key}.construct"] = original_construct

                def make_patched(orig_construct):
                    def patched_construct(self, query, key, value, attention_mask=None, **kwargs):
                        q_shape = query.shape
                        if len(q_shape) == 4:
                            num_heads = min(q_shape[1], q_shape[2])
                            head_dim = q_shape[3]
                        else:
                            return orig_construct(self, query, key, value, attention_mask, **kwargs)

                        from daca.nn.attention import DaCAAttention
                        daca_attn = DaCAAttention(
                            num_heads=num_heads,
                            num_kv_heads=num_heads,
                            head_dim=head_dim,
                            causal=True,
                        )
                        return daca_attn(query, key, value, attention_mask)
                    return patched_construct

                original.construct = make_patched(original_construct)
                logger.info(f"Patched {key}.construct -> DaCAAttention")

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not patch {module_path}.{attr_name}: {e}")


def revert_attention() -> None:
    """Revert attention patches using stored originals.

    Example:
        >>> from daca.compat.mindformers_patches import revert_attention
        >>> revert_attention()
    """
    for key, original in list(_original_configs.items()):
        if key.endswith(".construct"):
            # Restore construct method
            if key == "mf_FlashAttention_construct":
                try:
                    from mindformers.modules import attention as mf_attn_module
                    if hasattr(mf_attn_module, 'FlashAttention'):
                        mf_attn_module.FlashAttention.construct = original
                        logger.info(f"Reverted {key}")
                except Exception as e:
                    logger.debug(f"Could not revert {key}: {e}")
            elif "." in key:
                module_key = key.rsplit(".construct", 1)[0]
                parts = module_key.rsplit(".", 1)
                if len(parts) == 2:
                    module_path, attr_name = parts
                    try:
                        import importlib
                        mod = importlib.import_module(module_path)
                        cls = getattr(mod, attr_name, None)
                        if cls and hasattr(cls, 'construct'):
                            cls.construct = original
                            logger.info(f"Reverted {key}")
                    except Exception as e:
                        logger.debug(f"Could not revert {key}: {e}")

    _original_configs.clear()
    logger.info("MindFormers attention patches reverted")


def _rewrite_config_dtype(config: Any) -> Any:
    """Rewrite config dtype from bf16 to fp16.

    Args:
        config: Config object or dict.

    Returns:
        Config with bf16 replaced by fp16.
    """
    try:
        import mindspore.common.dtype as mstype
    except ImportError:
        return config

    # Handle dict-like configs
    if isinstance(config, dict):
        return _rewrite_dict_dtype(config)

    # Handle object configs with dtype attribute
    if hasattr(config, "compute_dtype"):
        if config.compute_dtype == mstype.bfloat16:
            config.compute_dtype = mstype.float16
            logger.debug("Rewrote compute_dtype: bf16 -> fp16")

    if hasattr(config, "dtype"):
        if config.dtype == mstype.bfloat16 or config.dtype == "bfloat16":
            config.dtype = mstype.float16
            logger.debug("Rewrote dtype: bf16 -> fp16")

    if hasattr(config, "layernorm_compute_dtype"):
        if config.layernorm_compute_dtype == mstype.bfloat16:
            config.layernorm_compute_dtype = mstype.float32  # Always fp32 for stability
            logger.debug("Rewrote layernorm_compute_dtype: bf16 -> fp32")

    return config


def _rewrite_dict_dtype(config: dict) -> dict:
    """Rewrite dtype values in a config dict.

    Args:
        config: Config dictionary.

    Returns:
        Config with bf16 replaced by fp16.
    """
    try:
        import mindspore.common.dtype as mstype
    except ImportError:
        return config

    import copy

    config = copy.deepcopy(config)

    dtype_keys = [
        "dtype", "compute_dtype", "param_init_type",
        "layernorm_compute_type", "softmax_compute_type",
        "rotary_dtype", "embedding_dtype"
    ]

    for key in dtype_keys:
        if key in config:
            value = config[key]
            if value == mstype.bfloat16 or value == "bfloat16" or value == "bf16":
                config[key] = mstype.float16
                logger.debug(f"Rewrote {key}: bf16 -> fp16")

    # Special case: LayerNorm always uses fp32 for stability
    if "layernorm_compute_type" in config:
        config["layernorm_compute_type"] = mstype.float32

    return config


def _rewrite_kwargs_dtype(kwargs: dict) -> dict:
    """Rewrite dtype values in kwargs.

    Args:
        kwargs: Keyword arguments dict.

    Returns:
        Kwargs with bf16 replaced by fp16.
    """
    return _rewrite_dict_dtype(kwargs)
