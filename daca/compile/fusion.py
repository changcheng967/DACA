"""CANN fusion control for Ascend NPUs.

Disables problematic CANN fusion passes, particularly FlashAttentionFusion
which causes rank mismatch errors.

CRITICAL: Native FlashAttentionScore has NO backward pass on 910ProA.
Using native FA during training will crash with:
    RuntimeError: The gradient operator [FlashAttentionScoreGrad] not found

DaCAAttention (daca.nn.attention) implements FlashAttention-equivalent
attention in pure MindSpore ops that all have backward support.

WHY: CANN 8.3 aggressively fuses ops through FlashAttention paths,
causing crashes. Disabling these fusions prevents the errors.

Example:
    from daca.compile import disable_flash_attention_fusion

    disable_flash_attention_fusion()
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("daca.compile.fusion")

# Track fusion state
_fusion_disabled: bool = False
_original_env: Dict[str, str] = {}


@dataclass
class FusionConfig:
    """Configuration for CANN fusion passes.

    Controls which fusion passes are enabled/disabled.

    Attributes:
        disable_flash_attention: Disable FlashAttentionFusion V1/V2.
        disable_matmul: Disable MatMul fusion.
        disable_elementwise: Disable elementwise fusion.
        disable_reduce: Disable reduce fusion.
        custom_passes: List of custom passes to disable.

    Example:
        >>> config = FusionConfig(disable_flash_attention=True)
        >>> config.apply()
    """
    disable_flash_attention: bool = True
    disable_matmul: bool = False
    disable_elementwise: bool = False
    disable_reduce: bool = False
    custom_passes: List[str] = field(default_factory=list)

    def apply(self) -> None:
        """Apply fusion configuration to environment."""
        env_settings = {}

        if self.disable_flash_attention:
            # Disable FlashAttention fusion V1 and V2
            env_settings["MS_DEV_DISABLE_FLASH_ATTENTION_FUSION"] = "1"
            env_settings["ASCEND_DISABLE_FLASH_ATTENTION_FUSION"] = "1"

        if self.disable_matmul:
            env_settings["MS_DEV_DISABLE_MATMUL_FUSION"] = "1"

        if self.disable_elementwise:
            env_settings["MS_DEV_DISABLE_ELEMENTWISE_FUSION"] = "1"

        if self.disable_reduce:
            env_settings["MS_DEV_DISABLE_REDUCE_FUSION"] = "1"

        # Custom passes
        for pass_name in self.custom_passes:
            env_settings[f"MS_DEV_DISABLE_{pass_name.upper()}_FUSION"] = "1"

        # Apply to environment
        for key, value in env_settings.items():
            os.environ[key] = value

        logger.info(f"Applied fusion config: {len(env_settings)} passes configured")


def disable_flash_attention_fusion() -> None:
    """Disable FlashAttention fusion passes.

    CANN 8.3 has a bug where it aggressively fuses unrelated ops
    through FlashAttention paths, causing rank mismatch errors.
    This disables those problematic fusion passes.

    Also disables MS_ENABLE_FLASH_ATTENTION to prevent CANN from
    trying to use the native FlashAttentionScore kernel (which has
    NO backward pass on 910ProA - Atlas A2 only).

    Environment variables set:
        - MS_DEV_DISABLE_FLASH_ATTENTION_FUSION=1
        - ASCEND_DISABLE_FLASH_ATTENTION_FUSION=1
        - MS_ENABLE_FLASH_ATTENTION=0

    Example:
        >>> from daca.compile import disable_flash_attention_fusion
        >>> disable_flash_attention_fusion()
        >>> # Now CANN won't try to fuse through FlashAttention
    """
    global _fusion_disabled, _original_env

    if _fusion_disabled:
        logger.debug("FlashAttention fusion already disabled")
        return

    # Save original values
    env_vars = [
        "MS_DEV_DISABLE_FLASH_ATTENTION_FUSION",
        "ASCEND_DISABLE_FLASH_ATTENTION_FUSION",
        "MS_ENABLE_FLASH_ATTENTION",
        "MS_DEV_GRAPH_KERNEL_FLAGS",
    ]
    for var in env_vars:
        if var in os.environ:
            _original_env[var] = os.environ[var]

    # Disable FlashAttention fusion V1 and V2
    os.environ["MS_DEV_DISABLE_FLASH_ATTENTION_FUSION"] = "1"
    os.environ["ASCEND_DISABLE_FLASH_ATTENTION_FUSION"] = "1"

    # Disable native FlashAttention (no backward on 910ProA)
    os.environ["MS_ENABLE_FLASH_ATTENTION"] = "0"

    # Disable FlashAttention-related graph kernel fusions
    # These can cause issues when CANN tries to route random ops through FA paths
    existing_flags = os.environ.get("MS_DEV_GRAPH_KERNEL_FLAGS", "")
    disable_flags = "--disable_flash_attention_fusion=true"
    if "disable_flash_attention_fusion" not in existing_flags:
        os.environ["MS_DEV_GRAPH_KERNEL_FLAGS"] = f"{existing_flags} {disable_flags}".strip()

    _fusion_disabled = True
    logger.info("Disabled FlashAttention fusion passes (V1 and V2)")
    logger.info("Disabled native FlashAttention (MS_ENABLE_FLASH_ATTENTION=0)")
    logger.warning(
        "Native FlashAttentionScore has NO backward pass on 910ProA. "
        "Use DaCAAttention (daca.nn.attention) for training."
    )


def enable_flash_attention_fusion() -> None:
    """Re-enable FlashAttention fusion passes.

    WARNING: Do not use during training! Native FlashAttentionScore
    has NO backward pass on 910ProA. This function is provided for
    inference-only use cases or testing.

    Restores original fusion settings.

    Example:
        >>> from daca.compile import enable_flash_attention_fusion
        >>> enable_flash_attention_fusion()
    """
    global _fusion_disabled, _original_env

    if not _fusion_disabled:
        logger.debug("FlashAttention fusion not disabled")
        return

    # Remove disabled flags
    env_vars = [
        "MS_DEV_DISABLE_FLASH_ATTENTION_FUSION",
        "ASCEND_DISABLE_FLASH_ATTENTION_FUSION",
        "MS_ENABLE_FLASH_ATTENTION",
        "MS_DEV_GRAPH_KERNEL_FLAGS",
    ]
    for var in env_vars:
        if var in os.environ and var not in _original_env:
            del os.environ[var]

    # Restore original values
    for var, value in _original_env.items():
        os.environ[var] = value

    _original_env.clear()
    _fusion_disabled = False
    logger.info("Re-enabled FlashAttention fusion passes")
    logger.warning(
        "Native FlashAttentionScore has NO backward pass on 910ProA. "
        "Only use for inference!"
    )


def disable_all_fusions() -> None:
    """Disable all known CANN fusion passes.

    Use with caution - this may impact performance.

    Example:
        >>> from daca.compile.fusion import disable_all_fusions
        >>> disable_all_fusions()
    """
    config = FusionConfig(
        disable_flash_attention=True,
        disable_matmul=True,
        disable_elementwise=True,
        disable_reduce=True,
    )
    config.apply()
    logger.warning("All CANN fusions disabled - performance may be impacted")


def get_fusion_status() -> Dict[str, bool]:
    """Get current fusion pass status.

    Returns:
        Dictionary mapping fusion pass names to enabled status.

    Example:
        >>> from daca.compile.fusion import get_fusion_status
        >>> status = get_fusion_status()
        >>> print(status["flash_attention"])
        False
    """
    return {
        "flash_attention": os.environ.get("MS_DEV_DISABLE_FLASH_ATTENTION_FUSION") != "1",
        "matmul": os.environ.get("MS_DEV_DISABLE_MATMUL_FUSION") != "1",
        "elementwise": os.environ.get("MS_DEV_DISABLE_ELEMENTWISE_FUSION") != "1",
        "reduce": os.environ.get("MS_DEV_DISABLE_REDUCE_FUSION") != "1",
    }
