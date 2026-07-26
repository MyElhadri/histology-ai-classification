"""Report and presentation artifacts generator for InceptionV3 Expérience A screening.

Generates confusion matrices (PNG/CSV), classification reports (CSV/JSON),
top confusions, overall metrics, model statistics, and presentation summary markdown
from Out-Of-Fold (OOF) predictions across screening folds 0, 3, and 4.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def generate_inceptionv3_screening_report(output_dir: Path) -> dict[str, Any]:
    """Generate all screening report artifacts from OOF prediction CSVs.

    Args:
        output_dir: Root output directory containing reports/inceptionv3.

    Returns:
        Dictionary of overall metrics.
    """
    out_dir = Path(output_dir).resolve()
    oof_dir = out_dir / "reports" / "inceptionv3" / "predictions"
    screening_oof_file = oof_dir / "inceptionv3_screening_oof_predictions.csv"

    if not screening_oof_file.is_file():
        # Fallback to combining fold_0, fold_3, fold_4 CSVs
        fold_csvs = [oof_dir / f"fold_{f}_oof_predictions.csv" for f in [0, 3, 4]]
        missing_csvs = [str(f) for f in fold_csvs if not f.is_file()]
        if missing_csvs:
            raise FileNotFoundError(f"Missing screening OOF prediction CSVs: {missing_csvs}")
        oof_df = pd.concat([pd.read_csv(f) for f in fold_csvs], ignore_index=True)
    else:
        oof_df = pd.read_csv(screening_oof_file)

    pres_dir = out_dir / "reports" / "inceptionv3" / "presentation_screening"
    pres_dir.mkdir(parents=True, exist_ok=True)

    y_true = oof_df["true_label"].values
    y_pred = oof_df["predicted_label"].values
    total_samples = len(oof_df)
    correct_samples = int(np.sum(y_true == y_pred))

    labels = sorted(list(set(y_true)))
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    # 1. Confusion Matrix Plots (Matplotlib 300 DPI, no seaborn)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    im = ax.imshow(cm_counts, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("InceptionV3 Exp A — Screening OOF folds 0, 3, 4 (Counts)", fontsize=14, pad=15)
    plt.colorbar(im)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(pres_dir / "confusion_matrix_screening_counts.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("InceptionV3 Exp A — Screening OOF folds 0, 3, 4 (Normalized)", fontsize=14, pad=15)
    plt.colorbar(im)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(pres_dir / "confusion_matrix_screening_normalized.png", dpi=300)
    plt.close(fig)

    # 2. Save Confusion Matrix CSVs
    pd.DataFrame(cm_counts, index=labels, columns=labels).to_csv(pres_dir / "confusion_matrix_screening_counts.csv")
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(pres_dir / "confusion_matrix_screening_normalized.csv")

    # 3. Classification Reports
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(pres_dir / "classification_report_screening.csv")
    with open(pres_dir / "classification_report_screening.json", "w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)

    # 4. Top Confusions
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm_counts[i, j] > 0:
                confusions.append({
                    "true_class": labels[i],
                    "predicted_class": labels[j],
                    "count": int(cm_counts[i, j])
                })
    df_conf = pd.DataFrame(confusions).sort_values("count", ascending=False) if confusions else pd.DataFrame(columns=["true_class", "predicted_class", "count"])
    df_conf.to_csv(pres_dir / "top_confusions_screening.csv", index=False)

    # 5. Overall Metrics & Model Statistics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    overall = {
        "model": "InceptionV3",
        "experiment": "inceptionv3-exp-a-fair-comparison",
        "scope": "Screening Folds 0, 3, 4",
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "cohens_kappa": round(kappa, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "weighted_f1": round(weighted_f1, 4),
    }

    with open(pres_dir / "overall_metrics_screening.json", "w", encoding="utf-8") as fp:
        json.dump(overall, fp, indent=2)

    model_stats = {
        "architecture": "InceptionV3",
        "input_shape": [224, 224, 3],
        "num_classes": 22,
        "preprocessing": "Rescaling(scale=1.0/127.5, offset=-1.0)",
        "input_range": "[0, 255] float32 -> [-1, 1] float32",
        "backbone_bn_frozen": True,
        "screening_folds": [0, 3, 4],
        "total_screening_samples": total_samples,
    }
    with open(pres_dir / "model_statistics.json", "w", encoding="utf-8") as fp:
        json.dump(model_stats, fp, indent=2)

    # 6. Presentation Summary Markdown
    summary_md = f"""# InceptionV3 Expérience A — Rapport de Screening (Folds 0, 3, 4)

## Résumé Général
- **Modèle** : InceptionV3 (ImageNet pre-trained)
- **Échantillons OOF de screening** : {total_samples} images (259 attendues)
- **Échantillons correctement classés** : {correct_samples} / {total_samples}
- **Accuracy** : {acc:.4f}
- **Balanced Accuracy** : {bal_acc:.4f}
- **Cohen's Kappa** : {kappa:.4f}
- **Macro F1** : {macro_f1:.4f}
- **Weighted F1** : {weighted_f1:.4f}

## Référence DenseNet121 Exp D (Mêmes Folds 0, 3, 4)
- **Mean Accuracy** : ~0.8610
- **Mean Macro F1** : ~0.7889
- **Mean Weighted F1** : ~0.8538

## Recommandation Screening
Comparaison scientifique requise avant de décider l'entraînement des folds 1 et 2.
"""
    with open(pres_dir / "presentation_summary.md", "w", encoding="utf-8") as fp:
        fp.write(summary_md)

    logger.info(f"InceptionV3 screening report artifacts generated successfully in {pres_dir}")
    return overall


def main() -> None:
    """CLI entrypoint for generating InceptionV3 screening report."""
    parser = argparse.ArgumentParser(description="Generate InceptionV3 screening report")
    parser.add_argument("--output-dir", required=True, help="Root directory containing models and reports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    generate_inceptionv3_screening_report(Path(args.output_dir))


if __name__ == "__main__":
    main()
