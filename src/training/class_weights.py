"""Compute per-class weights for imbalanced datasets.

This module is **independent of TensorFlow** so that it can be tested
locally without a GPU or the TF package installed.

The "balanced" method mirrors ``sklearn.utils.class_weight.compute_class_weight``
with ``class_weight="balanced"``:

    weight_c = n_samples / (n_classes × n_samples_c)

Where:
    n_samples   = total number of training samples
    n_classes   = number of expected classes
    n_samples_c = number of training samples belonging to class c
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Sequence

logger = logging.getLogger(__name__)


def compute_balanced_class_weights(
    labels: Sequence[int],
    num_classes: int,
) -> dict[int, float]:
    """Compute balanced class weights from integer labels.

    Args:
        labels: Sequence of integer class labels (e.g. ``[0, 1, 5, 5, 2, …]``).
            **Not mutated** by this function.
        num_classes: The expected number of distinct classes (e.g. 22).
            Every integer in ``range(num_classes)`` must appear at least
            once in *labels*.

    Returns:
        A dictionary ``{class_id: weight}`` where:
        - keys are ``int`` in ``range(num_classes)``
        - values are positive ``float``

    Raises:
        ValueError: If any expected class is missing from *labels*.
        ValueError: If *num_classes* < 1 or *labels* is empty.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")

    counts = Counter(labels)

    # Check that every expected class is present
    missing = sorted(set(range(num_classes)) - set(counts.keys()))
    if missing:
        raise ValueError(
            f"The following classes are missing from the training labels: "
            f"{missing}.  All {num_classes} classes (0..{num_classes - 1}) "
            f"must be represented in the training set."
        )

    n_samples = len(labels)
    weights: dict[int, float] = {}

    for class_id in range(num_classes):
        n_c = counts[class_id]
        w = n_samples / (num_classes * n_c)
        # Sanity: reject NaN / Inf (should not happen with n_c > 0)
        if math.isnan(w) or math.isinf(w):
            raise ValueError(
                f"Invalid weight for class {class_id}: {w} "
                f"(n_samples={n_samples}, n_c={n_c})"
            )
        weights[class_id] = float(w)

    return weights
