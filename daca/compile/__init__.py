"""DACA Compile Module.

Graph mode and fusion control for Ascend NPUs.

Example:
    from daca.compile import enable_graph_mode, disable_flash_attention_fusion
"""

from daca.compile.graph_mode import (
    enable_graph_mode,
    disable_graph_mode,
    set_safe_env,
    unset_safe_env,
    GraphCell,
)
from daca.compile.fusion import (
    disable_flash_attention_fusion,
    enable_flash_attention_fusion,
    FusionConfig,
)

__all__ = [
    # Graph mode
    "enable_graph_mode",
    "disable_graph_mode",
    "set_safe_env",
    "unset_safe_env",
    "GraphCell",
    # Fusion
    "disable_flash_attention_fusion",
    "enable_flash_attention_fusion",
    "FusionConfig",
]
