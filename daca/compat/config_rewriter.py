"""Config rewriting utilities.

Rewrites configuration files to replace bf16 with fp16,
ensuring compatibility with Ascend NPUs.

Example:
    from daca.compat import rewrite_config, ConfigRewriter

    # Functional API
    config = rewrite_config({"dtype": "bfloat16"})

    # Class-based API
    rewriter = ConfigRewriter()
    rewriter.rewrite_file("config.json", "config_fp16.json")
"""

import json
import os
import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("daca.compat.config_rewriter")


def rewrite_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite config dict to replace bf16 with fp16.

    Traverses the config dict and replaces all bf16 references
    with fp16 for Ascend compatibility.

    Args:
        config: Configuration dictionary.

    Returns:
        Rewritten config dictionary.

    Example:
        >>> config = {"dtype": "bfloat16", "hidden_size": 768}
        >>> rewritten = rewrite_config(config)
        >>> print(rewritten["dtype"])
        'float16'
    """
    import copy

    config = copy.deepcopy(config)
    return _rewrite_recursive(config)


def _rewrite_recursive(obj: Any) -> Any:
    """Recursively rewrite bf16 to fp16 in nested structures.

    Args:
        obj: Object to rewrite.

    Returns:
        Rewritten object.
    """
    if isinstance(obj, dict):
        return {k: _rewrite_value(k, _rewrite_recursive(v)) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_rewrite_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return _rewrite_string(obj)
    else:
        return obj


def _rewrite_value(key: str, value: Any) -> Any:
    """Rewrite a single value based on key and value.

    Args:
        key: Config key.
        value: Config value.

    Returns:
        Rewritten value.
    """
    # Special handling for dtype keys
    dtype_keys = [
        "dtype", "compute_dtype", "param_init_type",
        "layernorm_compute_type", "softmax_compute_type",
        "rotary_dtype", "embedding_dtype", "type",
    ]

    key_lower = key.lower()
    is_dtype_key = any(dk in key_lower for dk in ["dtype", "type", "precision"])

    if is_dtype_key and isinstance(value, str):
        return _rewrite_string(value)

    # Special case: LayerNorm always fp32
    if "layernorm" in key_lower and isinstance(value, str):
        if "bf16" in value.lower() or "bfloat" in value.lower():
            return "float32"

    return value


def _rewrite_string(s: str) -> str:
    """Rewrite bf16 string references to fp16.

    Args:
        s: String to rewrite.

    Returns:
        Rewritten string.
    """
    # Common patterns
    patterns = [
        (r"\bbf16\b", "fp16"),
        (r"\bbfloat16\b", "float16"),
        (r"\bbfloat\b", "float16"),
        (r"mindspore\.bfloat16", "mindspore.float16"),
        (r"mstype\.bfloat16", "mstype.float16"),
    ]

    for pattern, replacement in patterns:
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

    return s


class ConfigRewriter:
    """Config file rewriter.

    Reads config files, rewrites bf16 to fp16, and saves.

    Attributes:
        replacements: List of (pattern, replacement) tuples.
        backup_original: Whether to backup original files.

    Example:
        >>> rewriter = ConfigRewriter(backup_original=True)
        >>> rewriter.rewrite_file("config.json", "config_fp16.json")
    """

    def __init__(
        self,
        backup_original: bool = True,
        custom_replacements: Optional[List[tuple]] = None,
    ):
        """Initialize ConfigRewriter.

        Args:
            backup_original: Whether to create .bak files.
            custom_replacements: Additional (pattern, replacement) pairs.
        """
        self.backup_original = backup_original
        self.replacements = [
            (r"\bbf16\b", "fp16"),
            (r"\bbfloat16\b", "float16"),
            (r"\bbfloat\b", "float16"),
            (r"mindspore\.bfloat16", "mindspore.float16"),
            (r"mstype\.bfloat16", "mstype.float16"),
        ]

        if custom_replacements:
            self.replacements.extend(custom_replacements)

    def rewrite_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Rewrite a config file.

        Args:
            input_path: Path to input config file.
            output_path: Path to output file. If None, overwrites input.

        Returns:
            Path to output file.

        Example:
            >>> rewriter = ConfigRewriter()
            >>> output = rewriter.rewrite_file("config.json")
        """
        # Read input file
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Rewrite content
        rewritten = self._rewrite_content(content)

        # Determine output path
        if output_path is None:
            output_path = input_path
            if self.backup_original:
                backup_path = f"{input_path}.bak"
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Created backup: {backup_path}")

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rewritten)

        logger.info(f"Rewrote config: {input_path} -> {output_path}")
        return output_path

    def rewrite_json(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Rewrite a JSON config file with proper formatting.

        Args:
            input_path: Path to input JSON file.
            output_path: Path to output file.

        Returns:
            Path to output file.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        rewritten = rewrite_config(config)

        if output_path is None:
            output_path = input_path

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rewritten, f, indent=2)

        logger.info(f"Rewrote JSON config: {input_path} -> {output_path}")
        return output_path

    def _rewrite_content(self, content: str) -> str:
        """Apply all replacements to content.

        Args:
            content: Original content.

        Returns:
            Rewritten content.
        """
        for pattern, replacement in self.replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        return content

    def rewrite_directory(
        self,
        directory: str,
        extensions: List[str] = [".json", ".yaml", ".yml", ".py"],
        output_directory: Optional[str] = None,
    ) -> List[str]:
        """Rewrite all config files in a directory.

        Args:
            directory: Directory to process.
            extensions: File extensions to process.
            output_directory: Output directory. If None, modifies in place.

        Returns:
            List of processed file paths.
        """
        processed = []

        for root, dirs, files in os.walk(directory):
            for filename in files:
                if any(filename.endswith(ext) for ext in extensions):
                    input_path = os.path.join(root, filename)

                    if output_directory:
                        rel_path = os.path.relpath(input_path, directory)
                        output_path = os.path.join(output_directory, rel_path)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    else:
                        output_path = None

                    try:
                        if filename.endswith(".json"):
                            self.rewrite_json(input_path, output_path)
                        else:
                            self.rewrite_file(input_path, output_path)
                        processed.append(input_path)
                    except Exception as e:
                        logger.warning(f"Failed to process {input_path}: {e}")

        logger.info(f"Processed {len(processed)} files in {directory}")
        return processed
