"""Dataset splitting — shared test set and DenseNet121 fold generation.

This module produces two separate artefacts:

1. **test_manifest.csv** — the 20 % test split, shared by ALL models.
   Both DenseNet121 and ResNet50V2 MUST evaluate on this identical set.

2. **densenet121_folds.csv** — 5-fold stratified cross-validation
   assignments for the 80 % development set.  This file is specific
   to Yassine's DenseNet121 pipeline.  Mabena is free to define her
   own validation strategy for ResNet50V2.

Columns in test_manifest.csv:
    image_id, image_path, class_name, class_id, split

Columns in densenet121_folds.csv:
    image_id, image_path, class_name, class_id, split, fold

Convention:
    split = "test"        → common test set
    split = "development" → training / validation pool
    fold  = 0..4          → DenseNet121 cross-validation fold
    fold  = -1            → test images (present only in test_manifest)

Author: Yassine
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)

# Minimum number of images a class must have to support
# the chosen protocol (test split + 5-fold CV).
_MIN_IMAGES_FOR_5FOLD_PROTOCOL = 10  # 5 folds × 2 images minimum


def create_splits(
    manifest_df: pd.DataFrame,
    output_dir: Path,
    *,
    test_size: float = 0.20,
    num_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the shared test set and DenseNet121 fold assignments.

    Args:
        manifest_df: Full dataset manifest (from :func:`build_manifest.build_manifest`).
        output_dir: Directory where manifest CSVs will be written.
        test_size: Fraction of data reserved for the common test set.
        num_folds: Number of cross-validation folds for DenseNet121.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(test_df, folds_df)``.

    Raises:
        ValueError: If any class has too few images for the protocol.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validate class sizes
    # ------------------------------------------------------------------
    class_counts = manifest_df["class_name"].value_counts()
    small_classes = class_counts[class_counts < _MIN_IMAGES_FOR_5FOLD_PROTOCOL]

    if not small_classes.empty:
        details = ", ".join(
            f"{name} ({count})" for name, count in small_classes.items()
        )
        raise ValueError(
            f"The following classes have fewer than "
            f"{_MIN_IMAGES_FOR_5FOLD_PROTOCOL} images and cannot support "
            f"a {int(test_size*100)}% test split + {num_folds}-fold CV: "
            f"{details}. "
            f"Consider merging rare classes or reducing the number of folds."
        )

    # ------------------------------------------------------------------
    # 1. Stratified train/test split → shared test set
    # ------------------------------------------------------------------
    dev_df, test_df = train_test_split(
        manifest_df,
        test_size=test_size,
        random_state=seed,
        stratify=manifest_df["class_id"],
    )

    test_df = test_df.copy()
    test_df["split"] = "test"

    test_path = output_dir / "test_manifest.csv"
    test_df.to_csv(test_path, index=False)
    logger.info(
        "Shared test manifest saved: %s (%d images)",
        test_path, len(test_df),
    )

    # ------------------------------------------------------------------
    # 2. 5-fold stratified CV on development set (DenseNet121 only)
    # ------------------------------------------------------------------
    dev_df = dev_df.copy()
    dev_df["split"] = "development"
    dev_df["fold"] = -1  # default; will be overwritten

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold_idx, (_train_idx, _val_idx) in enumerate(
        skf.split(dev_df, dev_df["class_id"])
    ):
        dev_df.iloc[_val_idx, dev_df.columns.get_loc("fold")] = fold_idx

    # Sanity check: every development image has a valid fold
    assert (dev_df["fold"] >= 0).all(), "Some development images have no fold."
    assert (dev_df["fold"] < num_folds).all(), "Fold index out of range."

    folds_path = output_dir / "densenet121_folds.csv"
    dev_df.to_csv(folds_path, index=False)
    logger.info(
        "DenseNet121 folds manifest saved: %s (%d images, %d folds)",
        folds_path, len(dev_df), num_folds,
    )

    # ------------------------------------------------------------------
    # Log summary
    # ------------------------------------------------------------------
    logger.info(
        "Split summary — Development: %d (%.0f%%), Test: %d (%.0f%%)",
        len(dev_df),
        100 * len(dev_df) / len(manifest_df),
        len(test_df),
        100 * len(test_df) / len(manifest_df),
    )

    for fold_idx in range(num_folds):
        fold_count = (dev_df["fold"] == fold_idx).sum()
        logger.info("  Fold %d: %d images", fold_idx, fold_count)

    return test_df, dev_df
