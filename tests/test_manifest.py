"""Tests for dataset manifest creation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.build_manifest import build_manifest

REQUIRED_COLUMNS = {"image_id", "image_path", "class_name", "class_id"}


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset."""
    root = tmp_path / "dataset"
    classes = ["class_A", "class_B"]
    for i, cls in enumerate(classes):
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for j in range(3):
            img_path = cls_dir / f"image_{j}.png"
            img_path.touch()
    return root


def test_manifest_columns(mock_dataset: Path, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    build_manifest(mock_dataset, manifest_path)
    df = pd.read_csv(manifest_path)
    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_image_id_unique(mock_dataset: Path, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    build_manifest(mock_dataset, manifest_path)
    df = pd.read_csv(manifest_path)
    assert df["image_id"].is_unique


def test_class_consistency(mock_dataset: Path, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    build_manifest(mock_dataset, manifest_path)
    df = pd.read_csv(manifest_path)
    
    mapping = df.groupby("class_name")["class_id"].nunique()
    assert (mapping == 1).all(), "class_name to class_id mapping is not 1:1"
