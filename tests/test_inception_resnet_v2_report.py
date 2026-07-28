"""Unit tests for InceptionResNetV2 Exp A screening report generation.

Tests:
- Report generated from synthetic OOF CSVs (no real model needed)
- Confusion matrix files created (PNG + CSV)
- Classification report (JSON + CSV)
- Top confusions CSV
- Overall metrics JSON with scope warnings
- Model statistics JSON
- Presentation summary markdown
- Scope warning: 259 images ≠ 432 images

No scientific training, no ImageNet download.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARCH_NAME = "inception_resnet_v2"
NUM_CLASSES = 22
SCREENING_FOLDS = [3, 0, 4]
FOLD_SIZES = {3: 86, 0: 87, 4: 86}


def _create_synthetic_oof_csv(fold: int, output_dir: Path, seed: int = 42) -> None:
    """Create synthetic OOF CSV for a given fold."""
    np.random.seed(seed + fold)
    n = FOLD_SIZES[fold]
    pred_dir = output_dir / "reports" / ARCH_NAME / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    true_labels = np.random.randint(0, NUM_CLASSES, size=n)
    pred_labels = np.random.randint(0, NUM_CLASSES, size=n)

    rows = []
    for i in range(n):
        probs = np.random.dirichlet(np.ones(NUM_CLASSES))
        # Force predicted_label = argmax
        probs_adjusted = probs.copy()
        max_idx = pred_labels[i]
        probs_adjusted[max_idx] = probs_adjusted.max() + 0.1
        probs_adjusted /= probs_adjusted.sum()

        row = {
            "image_path": f"fold_{fold}/img_{i}.png",
            "image_id": f"fold_{fold}_img_{i}",
            "fold": fold,
            "true_label": int(true_labels[i]),
            "true_class": f"class_{true_labels[i]}",
            "predicted_label": int(np.argmax(probs_adjusted)),
            "predicted_class": f"class_{np.argmax(probs_adjusted)}",
            "correct": bool(true_labels[i] == np.argmax(probs_adjusted)),
        }
        for c in range(NUM_CLASSES):
            row[f"prob_{c}"] = float(probs_adjusted[c])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = pred_dir / f"fold_{fold}_oof_predictions.csv"
    df.to_csv(csv_path, index=False)


@pytest.fixture
def synthetic_output_dir(tmp_path: Path) -> Path:
    """Create synthetic OOF CSVs for folds 3, 0, 4."""
    for fold in SCREENING_FOLDS:
        _create_synthetic_oof_csv(fold, tmp_path)
    return tmp_path


class TestScreeningReportGeneration:
    def test_report_runs_without_error(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        result = generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        assert isinstance(result, dict)

    def test_confusion_matrix_png_counts_created(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        assert (pres_dir / "confusion_matrix_screening_counts.png").is_file()

    def test_confusion_matrix_png_normalized_created(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        assert (pres_dir / "confusion_matrix_screening_normalized.png").is_file()

    def test_confusion_matrix_csv_files_created(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        assert (pres_dir / "confusion_matrix_screening_counts.csv").is_file()
        assert (pres_dir / "confusion_matrix_screening_normalized.csv").is_file()

    def test_classification_report_json_created(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        assert (pres_dir / "classification_report_screening.json").is_file()

    def test_top_confusions_csv_created(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        assert (pres_dir / "top_confusions_screening.csv").is_file()

    def test_overall_metrics_json_created(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        assert (pres_dir / "overall_metrics_screening.json").is_file()

    def test_model_statistics_json_created(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        assert (pres_dir / "model_statistics.json").is_file()

    def test_presentation_summary_markdown_created(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        assert (pres_dir / "presentation_summary.md").is_file()

    def test_overall_metrics_has_scope_warning(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        with open(pres_dir / "overall_metrics_screening.json") as f:
            overall = json.load(f)
        assert "scope_warning" in overall
        # 259 images scope
        assert "259" in str(overall["scope"]) or "259" in str(overall.get("total_samples", ""))

    def test_overall_metrics_scope_259_images(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        with open(pres_dir / "overall_metrics_screening.json") as f:
            overall = json.load(f)
        assert overall["total_samples"] == 259

    def test_presentation_summary_mentions_screening_folds(self, synthetic_output_dir: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        generate_inception_resnet_v2_screening_report(synthetic_output_dir)
        pres_dir = synthetic_output_dir / "reports" / ARCH_NAME / "presentation_screening"
        content = (pres_dir / "presentation_summary.md").read_text()
        # Should mention the folds (3, 0, 4) and scope warning
        assert "3" in content and "0" in content and "4" in content
        assert "259" in content

    def test_report_raises_if_oof_csv_missing(self, tmp_path: Path) -> None:
        from scripts.generate_inception_resnet_v2_screening_report import (
            generate_inception_resnet_v2_screening_report,
        )
        with pytest.raises(FileNotFoundError):
            generate_inception_resnet_v2_screening_report(tmp_path)
