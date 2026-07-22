"""Tests for NuInsSeg 22-class dataset migration."""

import json
from pathlib import Path
import pandas as pd
import pytest

DATA_ROOT = Path("data/raw/nuinsseg_human_22_original")
MANIFEST_PATH = Path("data/manifests/original_22_dataset_manifest.csv")
REQUIRED_COLUMNS = {"image_id", "image_path", "class_name", "class_id", "original_filename", "width", "height", "channels", "file_hash"}

# Pytest might run before the script is executed by the user in a fresh environment, 
# but for the context of this specific check, we assume the dataset has been generated.
@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Dataset not yet generated.")
def test_exactly_22_classes():
    """Verify exactly 22 class folders exist."""
    folders = [f for f in DATA_ROOT.iterdir() if f.is_dir()]
    assert len(folders) == 22

@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Dataset not yet generated.")
def test_no_mouse_or_placenta():
    """Verify murine and placenta tissues are excluded."""
    for folder in DATA_ROOT.iterdir():
        if folder.is_dir():
            name = folder.name.lower()
            assert "mouse" not in name, f"Mouse tissue found: {name}"
            assert "placenta" not in name, f"Placenta found: {name}"

@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Dataset not yet generated.")
def test_no_empty_class():
    """Verify no class folder is empty."""
    for folder in DATA_ROOT.iterdir():
        if folder.is_dir():
            files = list(folder.glob("*"))
            assert len(files) > 0, f"Class {folder.name} is empty!"

@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not yet generated.")
def test_manifest_columns():
    """Verify mandatory columns in manifest."""
    df = pd.read_csv(MANIFEST_PATH)
    assert REQUIRED_COLUMNS.issubset(df.columns)

@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not yet generated.")
def test_unique_image_id():
    """Verify image_id is unique."""
    df = pd.read_csv(MANIFEST_PATH)
    assert df["image_id"].is_unique

@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not yet generated.")
def test_coherent_class_mapping():
    """Verify 1:1 mapping between class_name and class_id."""
    df = pd.read_csv(MANIFEST_PATH)
    mapping = df.groupby("class_name")["class_id"].nunique()
    assert (mapping == 1).all(), "class_name to class_id mapping is not 1:1"
