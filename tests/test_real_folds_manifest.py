"""Tests to verify the real densenet121_folds.csv is correctly generated."""

from pathlib import Path

import pandas as pd
import pytest

FOLDS_PATH = Path("data/manifests/densenet121_folds.csv")
EXPECTED_TOTAL = 432
EXPECTED_FOLDS = {0, 1, 2, 3, 4}


@pytest.fixture
def df_folds() -> pd.DataFrame:
    """Load the real folds CSV, skip if not generated."""
    if not FOLDS_PATH.exists():
        pytest.skip("densenet121_folds.csv not yet generated")
    return pd.read_csv(FOLDS_PATH)


def test_total_images(df_folds: pd.DataFrame) -> None:
    """Exactly 432 images."""
    assert len(df_folds) == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL} rows, got {len(df_folds)}"
    )


def test_unique_image_ids(df_folds: pd.DataFrame) -> None:
    """All image_ids are unique."""
    assert df_folds["image_id"].nunique() == EXPECTED_TOTAL


def test_fold_values(df_folds: pd.DataFrame) -> None:
    """Folds are exactly 0, 1, 2, 3, 4."""
    actual_folds = set(df_folds["fold"].unique())
    assert actual_folds == EXPECTED_FOLDS, f"Unexpected fold values: {actual_folds}"


def test_no_missing_values(df_folds: pd.DataFrame) -> None:
    """No NaN values in any mandatory column."""
    mandatory = ["image_id", "image_path", "class_name", "class_id", "fold"]
    for col in mandatory:
        assert df_folds[col].isna().sum() == 0, f"NaN values found in column '{col}'"


def test_all_classes_in_all_folds(df_folds: pd.DataFrame) -> None:
    """Every class appears in every fold."""
    pivot = df_folds.groupby(["class_name", "fold"]).size().unstack(fill_value=0)
    missing = pivot[pivot.min(axis=1) == 0]
    assert len(missing) == 0, (
        f"Some classes are missing from at least one fold:\n{missing}"
    )


def test_class_name_class_id_coherence(df_folds: pd.DataFrame) -> None:
    """Each class_name maps to exactly one class_id."""
    mapping = df_folds.groupby("class_name")["class_id"].nunique()
    bad = mapping[mapping > 1]
    assert len(bad) == 0, f"Inconsistent class_name → class_id mappings:\n{bad}"
