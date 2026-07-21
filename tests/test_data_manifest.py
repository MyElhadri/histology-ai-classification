"""Tests for dataset manifest and split generation.

Uses temporary mock data to verify manifest correctness without
depending on the real dataset being present.

Validates:
  - Manifest column schema
  - image_id uniqueness
  - class_name / class_id consistency
  - Test manifest: all rows have split="test"
  - DenseNet121 folds manifest: folds 0..4, split="development"
  - No overlap between test and development sets
  - Class mapping completeness
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

# We import the modules under test.
from src.data.build_manifest import build_manifest
from src.data.split_dataset import create_splits

# Required columns in the full dataset manifest.
MANIFEST_COLUMNS = {
    "image_id", "image_path", "class_name", "class_id",
    "width", "height", "channels", "file_size", "file_hash", "source",
}


@pytest.fixture()
def mock_dataset(tmp_path: Path) -> Path:
    """Create a minimal mock dataset with 3 classes × 20 tiny images each.

    This gives 60 images total — enough for a 20% test split (12 test)
    and 5-fold CV on the remaining 48.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow is required for this test.")

    dataset_root = tmp_path / "mock_dataset"

    for class_idx in range(3):
        class_name = f"class_{class_idx:02d}"
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True)

        for img_idx in range(20):
            # Create a tiny 8×8 RGB image with a unique pixel value.
            img = Image.new("RGB", (8, 8), color=(class_idx * 40, img_idx * 10, 128))
            img.save(class_dir / f"img_{img_idx:03d}.png")

    return dataset_root


@pytest.fixture()
def manifest_output(tmp_path: Path) -> Path:
    """Return a temporary directory for manifest outputs."""
    out = tmp_path / "output"
    out.mkdir()
    return out


class TestBuildManifest:
    """Tests for the manifest builder."""

    def test_manifest_has_required_columns(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df, _ = build_manifest(mock_dataset, manifest_output)
        assert MANIFEST_COLUMNS.issubset(set(df.columns)), (
            f"Missing columns: {MANIFEST_COLUMNS - set(df.columns)}"
        )

    def test_image_id_is_unique(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df, _ = build_manifest(mock_dataset, manifest_output)
        assert df["image_id"].is_unique, "image_id must be unique"

    def test_class_name_class_id_consistency(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df, mapping = build_manifest(mock_dataset, manifest_output)
        for _, row in df.iterrows():
            assert mapping[row["class_name"]] == row["class_id"], (
                f"Mismatch: {row['class_name']} → expected {mapping[row['class_name']]}, "
                f"got {row['class_id']}"
            )

    def test_class_mapping_saved(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        _, mapping = build_manifest(mock_dataset, manifest_output)
        mapping_path = manifest_output / "class_mapping.json"
        assert mapping_path.is_file()

        with open(mapping_path, "r") as f:
            loaded = json.load(f)
        assert loaded == mapping

    def test_class_mapping_is_deterministic(
        self, mock_dataset: Path, tmp_path: Path
    ) -> None:
        """Running build_manifest twice produces the same class mapping."""
        out1 = tmp_path / "out1"
        out1.mkdir()
        out2 = tmp_path / "out2"
        out2.mkdir()

        _, mapping1 = build_manifest(mock_dataset, out1)
        _, mapping2 = build_manifest(mock_dataset, out2)
        assert mapping1 == mapping2

    def test_image_paths_are_relative(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df, _ = build_manifest(mock_dataset, manifest_output)
        for path in df["image_path"]:
            assert not Path(path).is_absolute(), (
                f"Path should be relative: {path}"
            )


class TestSplits:
    """Tests for the shared test split and DenseNet121 folds."""

    def _get_manifest(
        self, mock_dataset: Path, manifest_output: Path
    ) -> pd.DataFrame:
        df, _ = build_manifest(mock_dataset, manifest_output)
        return df

    def test_test_manifest_all_rows_are_test(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df = self._get_manifest(mock_dataset, manifest_output)
        test_df, _ = create_splits(df, manifest_output, seed=42)
        assert (test_df["split"] == "test").all()

    def test_folds_manifest_all_rows_are_development(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df = self._get_manifest(mock_dataset, manifest_output)
        _, folds_df = create_splits(df, manifest_output, seed=42)
        assert (folds_df["split"] == "development").all()

    def test_folds_in_valid_range(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df = self._get_manifest(mock_dataset, manifest_output)
        _, folds_df = create_splits(df, manifest_output, num_folds=5, seed=42)
        assert folds_df["fold"].between(0, 4).all(), (
            f"Folds out of range: {folds_df['fold'].unique()}"
        )

    def test_no_overlap_between_test_and_development(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df = self._get_manifest(mock_dataset, manifest_output)
        test_df, folds_df = create_splits(df, manifest_output, seed=42)

        test_ids = set(test_df["image_id"])
        dev_ids = set(folds_df["image_id"])
        overlap = test_ids & dev_ids
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_images_accounted_for(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df = self._get_manifest(mock_dataset, manifest_output)
        test_df, folds_df = create_splits(df, manifest_output, seed=42)
        assert len(test_df) + len(folds_df) == len(df)

    def test_test_size_approximately_correct(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df = self._get_manifest(mock_dataset, manifest_output)
        test_df, _ = create_splits(df, manifest_output, test_size=0.20, seed=42)
        ratio = len(test_df) / len(df)
        assert 0.15 <= ratio <= 0.25, f"Test ratio {ratio:.2f} not ≈ 0.20"

    def test_splits_are_reproducible(
        self, mock_dataset: Path, tmp_path: Path
    ) -> None:
        """Same seed produces identical splits."""
        out1 = tmp_path / "s1"
        out1.mkdir()
        out2 = tmp_path / "s2"
        out2.mkdir()

        df, _ = build_manifest(mock_dataset, out1)
        test1, folds1 = create_splits(df, out1, seed=42)
        test2, folds2 = create_splits(df, out2, seed=42)

        assert list(test1["image_id"]) == list(test2["image_id"])
        assert list(folds1["image_id"]) == list(folds2["image_id"])

    def test_stratification_preserves_class_proportions(
        self, mock_dataset: Path, manifest_output: Path
    ) -> None:
        df = self._get_manifest(mock_dataset, manifest_output)
        test_df, _ = create_splits(df, manifest_output, seed=42)

        # Each class should have roughly test_size proportion in test
        for class_name in df["class_name"].unique():
            total = (df["class_name"] == class_name).sum()
            in_test = (test_df["class_name"] == class_name).sum()
            ratio = in_test / total
            assert 0.10 <= ratio <= 0.35, (
                f"Class {class_name}: test ratio {ratio:.2f} is extreme"
            )
