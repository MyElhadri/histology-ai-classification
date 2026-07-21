"""Grad-CAM (Gradient-weighted Class Activation Mapping).

Provides visual explanations for CNN predictions by highlighting
image regions that most influenced the model's decision.

This is a SKELETON — the full implementation will be completed after
both DenseNet121 and ResNet50V2 models are trained.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Author: Yassine
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def compute_gradcam_heatmap(
    model: Any,
    image: np.ndarray,
    class_index: int | None = None,
    layer_name: str | None = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image.

    Args:
        model: A compiled Keras model.
        image: Preprocessed image array of shape ``(1, H, W, 3)``.
        class_index: Target class index.  If ``None``, uses the
            predicted class.
        layer_name: Name of the convolutional layer to use.  If ``None``,
            the last conv layer is used automatically.

    Returns:
        Heatmap array of shape ``(H, W)`` with values in ``[0, 1]``.

    Raises:
        NotImplementedError: This is a skeleton — implementation pending.
    """
    raise NotImplementedError(
        "Grad-CAM will be implemented after model training is complete."
    )


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an image.

    Args:
        image: Original image as a uint8 array ``(H, W, 3)``.
        heatmap: Heatmap array ``(H, W)`` with values in ``[0, 1]``.
        alpha: Blending factor for the overlay.

    Returns:
        Blended image as uint8 array ``(H, W, 3)``.

    Raises:
        NotImplementedError: This is a skeleton — implementation pending.
    """
    raise NotImplementedError(
        "Grad-CAM overlay will be implemented after model training."
    )


def save_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path | str,
    alpha: float = 0.4,
) -> None:
    """Save a Grad-CAM visualization to disk.

    Args:
        image: Original image.
        heatmap: Grad-CAM heatmap.
        output_path: Path to save the output image.
        alpha: Blending factor.

    Raises:
        NotImplementedError: This is a skeleton — implementation pending.
    """
    raise NotImplementedError(
        "Grad-CAM save will be implemented after model training."
    )
