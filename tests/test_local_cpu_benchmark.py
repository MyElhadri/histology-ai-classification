"""Lightweight tests for local CPU benchmark configuration and logic.

These tests verify configuration parameters, error handling for missing dataset,
duration estimation logic, and preservation of the original Experiment D configuration.
NO actual model training is launched during pytest execution.
"""

import pytest
from pathlib import Path
from src.utils.config import load_yaml
from scripts.run_local_cpu_benchmark import validate_dataset, calculate_estimations


def test_benchmark_yaml_exists_and_valid():
    """Verify benchmark YAML file exists and contains required parameters."""
    yaml_path = Path("configs/experiments/densenet121_local_cpu_benchmark.yaml")
    assert yaml_path.is_file(), "densenet121_local_cpu_benchmark.yaml does not exist"

    cfg = load_yaml(yaml_path)

    assert cfg.get("experiment_name") == "densenet121_local_cpu_benchmark"
    assert cfg.get("training", {}).get("batch_size") == 4
    assert cfg.get("training", {}).get("head_epochs") in (2, 10)
    assert cfg.get("training", {}).get("fine_tuning_epochs") in (3, 40)
    assert cfg.get("validation", {}).get("run_mode") == "single_fold"
    assert cfg.get("validation", {}).get("target_fold") == 0


def test_original_exp_d_remains_unchanged():
    """Verify original Experiment D configuration remains untouched."""
    yaml_path = Path("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    assert yaml_path.is_file(), "densenet121_exp_d_rich_aug_article_head.yaml does not exist"

    cfg_d = load_yaml(yaml_path)

    assert cfg_d.get("project", {}).get("name") == "histology-ai-classification"
    assert cfg_d.get("training", {}).get("batch_size") == 16
    assert cfg_d.get("training", {}).get("head_epochs") == 10
    assert cfg_d.get("training", {}).get("fine_tuning_epochs") == 40
    assert cfg_d.get("model", {}).get("classifier_head", {}).get("type") == "article_inspired"
    assert cfg_d.get("fine_tuning", {}).get("strategy") == "full"


def test_dataset_validation_missing_directory(tmp_path):
    """Verify validate_dataset raises FileNotFoundError if directory does not exist."""
    non_existent = tmp_path / "missing_dataset_directory"

    with pytest.raises(FileNotFoundError):
        validate_dataset(non_existent)


def test_dataset_validation_valid_directory(tmp_path):
    """Verify validate_dataset correctly counts image files."""
    ds_dir = tmp_path / "test_ds"
    ds_dir.mkdir()
    (ds_dir / "img1.png").touch()
    (ds_dir / "img2.jpg").touch()
    (ds_dir / "readme.txt").touch()

    count = validate_dataset(ds_dir)
    assert count == 2


def test_duration_estimations_calculation():
    """Verify calculation of full fold duration estimates."""
    avg_head = 12.5
    avg_ft = 25.0

    estimates = calculate_estimations(avg_head, avg_ft)

    assert estimates["estimated_full_fold_20_epochs"] == 625.0
    assert estimates["estimated_full_fold_30_epochs"] == 875.0
    assert estimates["estimated_full_fold_40_epochs"] == 1125.0
