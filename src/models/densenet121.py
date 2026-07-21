"""DenseNet121 architecture for tissue classification.

This module handles:
- Building DenseNet121 with transfer learning (ImageNet weights)
- Custom classification head for histology classes
- Fine-tuning strategy (freeze/unfreeze layers)

The implementation will use a two-phase training approach:
  1. Train only the classification head (base frozen)
  2. Fine-tune the full network with a low learning rate

Author: Yassine
"""

from __future__ import annotations

from typing import Any


def build_densenet121(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str = "imagenet",
    dropout_rate: float = 0.30,
    dense_units: int = 256,
) -> Any:
    """Build a DenseNet121 model with a custom classification head.

    Args:
        num_classes: Number of output classes (22 for this dataset).
        input_shape: Input image dimensions ``(H, W, C)``.
        weights: Pre-trained weights to load (``"imagenet"`` or ``None``).
        dropout_rate: Dropout rate before the final dense layer.
        dense_units: Number of units in the intermediate dense layer.

    Returns:
        A compiled Keras Model (will be implemented in the next task).

    Raises:
        NotImplementedError: Skeleton only — training task pending.
    """
    raise NotImplementedError(
        "DenseNet121 build will be implemented in the training task."
    )


def freeze_base(model: Any) -> None:
    """Freeze the DenseNet121 base layers for head-only training.

    Args:
        model: The Keras model built by :func:`build_densenet121`.

    Raises:
        NotImplementedError: Skeleton only.
    """
    raise NotImplementedError("Will be implemented in the training task.")


def unfreeze_for_fine_tuning(
    model: Any,
    learning_rate: float = 1e-5,
) -> Any:
    """Unfreeze the base and recompile with a lower learning rate.

    Args:
        model: The Keras model after head training.
        learning_rate: Learning rate for fine-tuning phase.

    Returns:
        Recompiled model ready for fine-tuning.

    Raises:
        NotImplementedError: Skeleton only.
    """
    raise NotImplementedError("Will be implemented in the training task.")
