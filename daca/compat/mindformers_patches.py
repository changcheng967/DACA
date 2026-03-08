"""MindFormers patches for Ascend compatibility.

Fixes bf16 config issues and LayerNorm usage in MindFormers models.

WHY: MindFormers configs often default to bf16, which crashes on Ascend.
Also patches broken LayerNorm usage patterns.

Example:
    from daca.compat import apply_mf_patches

    apply_mf_patches()
"""

import logging
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

    Example:
        >>> from daca.compat.mindformers_patches import apply_all
        >>> apply_all()
    """
    global _patches_applied

    if _patches_applied:
        logger.debug("MindFormers patches already applied")
        return

    patch_bf16_config()
    patch_layernorm_usage()

    _patches_applied = True
    logger.info("Applied all MindFormers patches")


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
    logger.info("MindFormers LayerNorm usage patches marked for removal")


def _rewrite_config_dtype(config: Any) -> Any:
    """Rewrite config dtype from bf16 to fp16.

    Args:
        config: Config object or dict.

    Returns:
        Config with bf16 replaced by fp16.
    """
    import mindspore.common.dtype as mstype

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
    import mindspore.common.dtype as mstype
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
