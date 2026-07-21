"""Tests for DenseNet121 folds creation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.create_folds import create_folds


@pytest.fixture
def mock_manifest(tmp_path: Path) -> Path:
    """Create a mock manifest with enough samples for 5 folds."""
    manifest_path = tmp_path / "mock_manifest.csv"
    records = []
    
    # 2 classes, 10 images each
    for cls_idx, cls_name in enumerate(["class_A", "class_B"]):
        for img_idx in range(10):
            records.append({
                "image_id": f"img_{cls_name}_{img_idx}",
                "image_path": f"{cls_name}/img_{img_idx}.png",
                "class_name": cls_name,
                "class_id": cls_idx
            })
            
    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)
    return manifest_path


def test_folds_range(mock_manifest: Path, tmp_path: Path) -> None:
    folds_path = tmp_path / "folds.csv"
    create_folds(mock_manifest, folds_path, num_folds=5)
    df = pd.read_csv(folds_path)
    
    assert df["fold"].between(0, 4).all()


def test_every_image_in_one_fold(mock_manifest: Path, tmp_path: Path) -> None:
    folds_path = tmp_path / "folds.csv"
    create_folds(mock_manifest, folds_path, num_folds=5)
    df = pd.read_csv(folds_path)
    
    # Check for NaNs
    assert df["fold"].notna().all()
    # image_id uniqueness ensures each image is listed exactly once
    assert df["image_id"].is_unique


def test_classes_represented_in_folds(mock_manifest: Path, tmp_path: Path) -> None:
    folds_path = tmp_path / "folds.csv"
    create_folds(mock_manifest, folds_path, num_folds=5)
    df = pd.read_csv(folds_path)
    
    for fold in range(5):
        fold_df = df[df["fold"] == fold]
        # Should have both class 0 and 1
        assert set(fold_df["class_id"]) == {0, 1}
