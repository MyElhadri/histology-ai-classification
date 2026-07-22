"""Tests to verify that configs/densenet121.yaml loads correctly via config.py."""

import pytest
from pathlib import Path

from src.utils.config import load_yaml

CONFIG_PATH = Path("configs/densenet121.yaml")


@pytest.fixture
def config() -> dict:
    """Load the config, skip if file is missing."""
    if not CONFIG_PATH.exists():
        pytest.skip("configs/densenet121.yaml not found")
    return load_yaml(CONFIG_PATH)


def test_config_loads(config: dict) -> None:
    """Config must be a non-empty dictionary."""
    assert isinstance(config, dict)
    assert len(config) > 0


def test_num_classes(config: dict) -> None:
    assert config["data"]["num_classes"] == 22


def test_image_size(config: dict) -> None:
    assert config["data"]["image_size"] == [224, 224]


def test_num_folds(config: dict) -> None:
    assert config["validation"]["num_folds"] == 5


def test_seed(config: dict) -> None:
    assert config["project"]["seed"] == 42


def test_dataset_root_path(config: dict) -> None:
    """Dataset root must point to the new 22-class dataset."""
    root = config["data"]["dataset_root"]
    assert "nuinsseg_human_22_original" in root


def test_manifest_path(config: dict) -> None:
    """Manifest path must reference the new manifest."""
    manifest = config["data"]["manifest_path"]
    assert "original_22_dataset_manifest" in manifest


def test_folds_path(config: dict) -> None:
    """Folds path must reference densenet121_folds.csv."""
    folds = config["data"]["folds_path"]
    assert "densenet121_folds" in folds
