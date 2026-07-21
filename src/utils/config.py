"""Configuration loader and manager.

This module handles:
- Loading YAML configuration files
- Merging common config with model-specific config
- Configuration validation
- Accessing nested config values

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/common.yaml", "configs/densenet121.yaml")
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*.

    Values in *override* take precedence.  Nested dicts are merged
    recursively; all other types are replaced.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged dictionary (a new dict — inputs are not mutated).
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        logger.warning("YAML file is empty: %s", path)
        return {}

    logger.info("Loaded config: %s", path)
    return data


def load_config(
    common_path: Path | str,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load and merge the common config with an optional model-specific config.

    Args:
        common_path: Path to ``configs/common.yaml``.
        model_path: Path to a model-specific YAML (e.g.
            ``configs/densenet121.yaml``).  If ``None``, only the common
            config is returned.

    Returns:
        Merged configuration dictionary.
    """
    config = load_yaml(common_path)

    if model_path is not None:
        model_config = load_yaml(model_path)
        config = _deep_merge(config, model_config)
        logger.info("Merged model config: %s", model_path)

    return config
