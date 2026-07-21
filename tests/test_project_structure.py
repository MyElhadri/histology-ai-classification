"""Structural tests — verify essential project files and directories exist.

These tests ensure the repository layout is correct after setup.
They do NOT depend on the dataset being present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Project root is two levels up from tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDirectoryStructure:
    """Verify that critical directories exist."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "configs",
            "data",
            "data/manifests",
            "data/raw",
            "data/processed",
            "data/external",
            "data/external/smartphone_microscope",
            "docs",
            "experiments",
            "experiments/densenet121",
            "experiments/resnet50v2",
            "models",
            "models/checkpoints",
            "models/saved",
            "models/exports",
            "notebooks",
            "notebooks/colab",
            "notebooks/exploration",
            "reports",
            "reports/figures",
            "reports/metrics",
            "reports/predictions",
            "scripts",
            "src",
            "src/data",
            "src/models",
            "src/training",
            "src/evaluation",
            "src/explainability",
            "src/utils",
            "tests",
            "api",
            "mobile",
        ],
    )
    def test_directory_exists(self, rel_path: str) -> None:
        path = PROJECT_ROOT / rel_path
        assert path.is_dir(), f"Directory missing: {rel_path}"


class TestEssentialFiles:
    """Verify that essential files exist."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "README.md",
            "LICENSE",
            "requirements.txt",
            "requirements-colab.txt",
            "pyproject.toml",
            ".gitignore",
            # Configs
            "configs/common.yaml",
            "configs/densenet121.yaml",
            "configs/resnet50v2.yaml",
            "configs/config.example.yaml",
            "configs/logging.yaml",
            # Source modules
            "src/__init__.py",
            "src/data/__init__.py",
            "src/data/build_manifest.py",
            "src/data/audit.py",
            "src/data/split_dataset.py",
            "src/data/explore_dataset.py",
            "src/models/__init__.py",
            "src/models/densenet121.py",
            "src/models/resnet50v2.py",
            "src/training/__init__.py",
            "src/evaluation/__init__.py",
            "src/explainability/__init__.py",
            "src/explainability/gradcam.py",
            "src/utils/__init__.py",
            "src/utils/config.py",
            "src/utils/runtime.py",
            "src/utils/seed.py",
            # Scripts
            "scripts/prepare_data.py",
            # Docs
            "docs/colab_training.md",
            "docs/architecture_guide.md",
            "docs/methodology.md",
        ],
    )
    def test_file_exists(self, rel_path: str) -> None:
        path = PROJECT_ROOT / rel_path
        assert path.is_file(), f"File missing: {rel_path}"


class TestConfigFiles:
    """Verify YAML configuration files are valid and loadable."""

    @pytest.mark.parametrize(
        "config_file",
        [
            "configs/common.yaml",
            "configs/densenet121.yaml",
            "configs/resnet50v2.yaml",
        ],
    )
    def test_yaml_is_loadable(self, config_file: str) -> None:
        import yaml

        path = PROJECT_ROOT / config_file
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{config_file} did not parse as a dict"

    def test_common_yaml_has_required_keys(self) -> None:
        import yaml

        path = PROJECT_ROOT / "configs/common.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "project" in data
        assert data["project"]["seed"] == 42
        assert "data" in data
        assert data["data"]["num_classes"] == 22
        assert data["data"]["test_size"] == 0.20
        assert "evaluation" in data

    def test_densenet121_yaml_has_validation_section(self) -> None:
        import yaml

        path = PROJECT_ROOT / "configs/densenet121.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "validation" in data
        assert data["validation"]["strategy"] == "stratified_kfold"
        assert data["validation"]["num_folds"] == 5
