"""Reproducibility and seed management.

This module handles:
- Setting global random seeds (Python, NumPy, TensorFlow)
- Deterministic operation configuration
- Ensuring reproducible data splits and augmentation

Usage:
    from src.utils.seed import set_global_seed
    set_global_seed(42)
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds across all relevant libraries for reproducibility.

    Sets seeds for:
    - Python's built-in ``random`` module
    - NumPy
    - TensorFlow (if installed)
    - ``PYTHONHASHSEED`` environment variable

    Args:
        seed: Integer seed value.
    """
    # Python built-in random.
    random.seed(seed)

    # NumPy.
    np.random.seed(seed)

    # Python hash seed (affects set/dict ordering in some contexts).
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow (optional import — works on CPU-only machines).
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info("TensorFlow seed set to %d", seed)
    except ImportError:
        logger.info("TensorFlow not installed — TF seed not set.")

    logger.info(
        "Global seed set to %d (Python, NumPy%s)",
        seed,
        ", TensorFlow" if "tensorflow" in dir() else "",
    )
