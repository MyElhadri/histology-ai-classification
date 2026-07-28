import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_resnet50v2_gtex_report import process_split_report


@pytest.fixture
def mock_results_dir(tmp_path: Path):
    metrics = {
        "accuracy": 0.95,
        "balanced_accuracy": 0.94,
        "macro_f1": 0.93,
        "cohens_kappa": 0.92,
        "confusion_matrix": [[10, 1], [2, 12]],
        "classification_report": {
            "bladder": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9, "support": 11},
            "brain": {"precision": 0.92, "recall": 0.85, "f1-score": 0.88, "support": 14},
            "accuracy": 0.95,
            "macro avg": {"precision": 0.91, "recall": 0.87, "f1-score": 0.89, "support": 25},
            "weighted avg": {"precision": 0.91, "recall": 0.95, "f1-score": 0.93, "support": 25}
        },
        "top_confusions": [
            {"true_class": "bladder", "predicted_class": "brain", "count": 1},
            {"true_class": "brain", "predicted_class": "bladder", "count": 2}
        ]
    }
    
    with open(tmp_path / "validation_metrics.json", "w") as f:
        json.dump(metrics, f)
        
    return tmp_path


def test_generate_report_outputs(mock_results_dir: Path):
    class_names = ["bladder", "brain"]
    process_split_report("validation", mock_results_dir, class_names)
    
    assert (mock_results_dir / "validation_confusion_matrix_counts.png").exists()
    assert (mock_results_dir / "validation_confusion_matrix_counts.csv").exists()
    assert (mock_results_dir / "validation_confusion_matrix_normalized.png").exists()
    assert (mock_results_dir / "validation_confusion_matrix_normalized.csv").exists()
    assert (mock_results_dir / "validation_classification_report.csv").exists()
    assert (mock_results_dir / "validation_top_confusions.csv").exists()
    assert (mock_results_dir / "validation_presentation_summary.md").exists()
