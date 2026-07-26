"""Unit tests for InceptionV3 screening report generator script."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.generate_inceptionv3_screening_report import generate_inceptionv3_screening_report


def create_mock_screening_oof(file_path: Path, num_images: int = 259) -> None:
    """Create mock screening OOF predictions CSV for 259 screening images."""
    rows = []
    for i in range(num_images):
        true_lbl = i % 22
        pred_lbl = true_lbl if i % 10 != 0 else (true_lbl + 1) % 22
        probs = [0.01] * 22
        probs[pred_lbl] = 0.80

        row = {
            "image_path": f"data/raw/nuinsseg_human_22_original/class_{true_lbl}/img_{i:03d}.png",
            "image_id": f"img_nuinsseg_{i:06d}",
            "fold": 0 if i < 87 else (3 if i < 173 else 4),
            "true_label": true_lbl,
            "true_class": f"class_{true_lbl}",
            "predicted_label": pred_lbl,
            "predicted_class": f"class_{pred_lbl}",
            "correct": bool(true_lbl == pred_lbl),
        }
        for c in range(22):
            row[f"prob_{c}"] = probs[c]
        rows.append(row)

    df = pd.DataFrame(rows)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def test_generate_inceptionv3_screening_report(tmp_path: Path) -> None:
    """Test generating all presentation screening report artifacts."""
    oof_dir = tmp_path / "reports" / "inceptionv3" / "predictions"
    oof_csv = oof_dir / "inceptionv3_screening_oof_predictions.csv"
    create_mock_screening_oof(oof_csv, num_images=259)

    overall = generate_inceptionv3_screening_report(tmp_path)
    assert overall["total_samples"] == 259
    assert overall["accuracy"] > 0.0

    pres_dir = tmp_path / "reports" / "inceptionv3" / "presentation_screening"
    required_artifacts = [
        "confusion_matrix_screening_counts.png",
        "confusion_matrix_screening_normalized.png",
        "confusion_matrix_screening_counts.csv",
        "confusion_matrix_screening_normalized.csv",
        "classification_report_screening.csv",
        "classification_report_screening.json",
        "overall_metrics_screening.json",
        "top_confusions_screening.csv",
        "model_statistics.json",
        "presentation_summary.md",
    ]
    for artifact_name in required_artifacts:
        artifact_file = pres_dir / artifact_name
        assert artifact_file.is_file(), f"Missing required artifact: {artifact_name}"
        assert artifact_file.stat().st_size > 0
