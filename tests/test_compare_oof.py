"""Unit tests for DenseNet vs EfficientNet OOF complementarity script."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compare_densenet_efficientnet_oof import analyze_complementarity, load_predictions


def create_mock_oof_csv(file_path: Path, num_images: int = 10, offset_pred: bool = False, pred_prob: float = 0.80) -> None:
    """Create a mock OOF CSV file."""
    rows = []
    for i in range(num_images):
        true_lbl = i % 5
        pred_lbl = (true_lbl + 1) % 5 if (offset_pred and i % 2 == 0) else true_lbl
        probs = [0.01] * 22
        probs[pred_lbl] = pred_prob

        row = {
            "image_path": f"img_{i:03d}.tif",
            "true_class": f"class_{true_lbl}",
            "true_label": true_lbl,
            "predicted_class": f"class_{pred_lbl}",
            "predicted_label": pred_lbl,
            "correct": bool(true_lbl == pred_lbl),
        }
        for c in range(22):
            row[f"prob_{c}"] = probs[c]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


def test_load_predictions_and_alignment(tmp_path: Path) -> None:
    """Test loading and aligning OOF predictions from separate files/directories."""
    dense_file = tmp_path / "dense_oof.csv"
    eff_dir = tmp_path / "eff_preds"
    eff_dir.mkdir(parents=True, exist_ok=True)
    eff_file = eff_dir / "fold_0_oof_predictions.csv"

    create_mock_oof_csv(dense_file, num_images=10, offset_pred=False)
    create_mock_oof_csv(eff_file, num_images=10, offset_pred=True)

    dense_df, eff_df = load_predictions(dense_file, eff_dir)
    assert len(dense_df) == 10
    assert len(eff_df) == 10
    assert list(dense_df["image_path"]) == list(eff_df["image_path"])


def test_analyze_complementarity(tmp_path: Path) -> None:
    """Test complementarity metric calculation and simple 50/50 ensemble."""
    dense_file = tmp_path / "dense_oof.csv"
    eff_dir = tmp_path / "eff_preds"
    eff_dir.mkdir(parents=True, exist_ok=True)
    eff_file = eff_dir / "fold_0_oof_predictions.csv"

    create_mock_oof_csv(dense_file, num_images=10, offset_pred=False, pred_prob=0.90)  # 100% accuracy, high confidence
    create_mock_oof_csv(eff_file, num_images=10, offset_pred=True, pred_prob=0.60)   # 50% accuracy, lower confidence

    dense_df, eff_df = load_predictions(dense_file, eff_dir)
    results = analyze_complementarity(dense_df, eff_df)

    assert results["images_evaluated"] == 10
    assert results["single_model_metrics"]["densenet121"]["accuracy"] == 1.0
    assert results["single_model_metrics"]["efficientnetv2b0"]["accuracy"] == 0.5
    assert results["complementarity_breakdown"]["both_correct"] == 5
    assert results["complementarity_breakdown"]["densenet_only_correct"] == 5
    assert results["complementarity_breakdown"]["efficientnet_only_correct"] == 0
    assert results["complementarity_breakdown"]["agreement_rate"] == 0.5
    assert results["simple_ensemble_50_50"]["accuracy"] == 1.0  # DenseNet 0.80 prob dominates EffNet 0.05 prob on errors


def test_compare_script_execution_without_pythonpath(tmp_path: Path) -> None:
    """Test that compare_densenet_efficientnet_oof.py runs with --help from outside repo without PYTHONPATH."""
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "compare_densenet_efficientnet_oof.py"

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    res = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert res.returncode == 0, f"Script failed with exit code {res.returncode}. Output:\n{res.stderr}"
    assert "ModuleNotFoundError" not in res.stderr
    assert "ModuleNotFoundError" not in res.stdout
    assert "usage:" in res.stdout.lower() or "--help" in res.stdout
