"""ResNet50V2 architecture for tissue classification.

This module handles:
- Building ResNet50V2 with transfer learning (ImageNet weights)
- Custom classification head for histology classes
- Fine-tuning strategy (freeze/unfreeze layers)

Author: Mabena

Note:
    This is a skeleton file. Mabena is responsible for the full
    implementation and is free to choose her own training strategy,
    validation approach, and hyperparameters. The only shared
    constraints are the test set, class mapping, and evaluation
    metrics defined in configs/common.yaml.
"""

from __future__ import annotations

from typing import Any


def build_resnet50v2(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str = "imagenet",
    dropout_rate: float = 0.30,
    dense_units: int = 256,
) -> Any:
    """Build a ResNet50V2 model with a custom classification head.

    Args:
        num_classes: Number of output classes (22 for this dataset).
        input_shape: Input image dimensions ``(H, W, C)``.
        weights: Pre-trained weights to load (``"imagenet"`` or ``None``).
        dropout_rate: Dropout rate before the final dense layer.
        dense_units: Number of units in the intermediate dense layer.

    Returns:
        A compiled Keras Model (to be implemented by Mabena).

    Raises:
        NotImplementedError: Skeleton only — Mabena will implement.
    """
    raise NotImplementedError(
        "ResNet50V2 build will be implemented by Mabena."
    )
