"""Report and presentation artifacts generator for InceptionResNetV2 Expérience A screening.

Generates from OOF prediction CSVs (folds 3, 0, 4):
- Confusion matrices (PNG 300 DPI / CSV) — counts and normalized
- Classification report (CSV / JSON)
- Top confusions (CSV)
- Overall metrics (JSON)
- Model statistics (JSON)
- Presentation summary (Markdown)
- Training curves per fold (per phase, with phase boundaries)

Note: Screening covers 259 images (folds 3, 0, 4) — NOT the full 432-image validation.
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

import matplotlib
matplotlib.use("Agg")
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

ARCH_NAME = "inception_resnet_v2"
SCREENING_FOLDS = [3, 0, 4]
EXPECTED_TOTAL = 259


def load_class_mapping(output_dir: Path) -> dict[int, str]:
    """Load authoritative class_id → class_name mapping."""
    mapping_path = PROJECT_ROOT / "data" / "manifests" / "class_mapping.json"
    if mapping_path.is_file():
        with open(mapping_path, "r", encoding="utf-8") as f:
            name_to_id = json.load(f)
        return {int(v): k for k, v in name_to_id.items()}

    # Fallback: derive from folds CSV
    folds_csv = PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"
    if folds_csv.is_file():
        df = pd.read_csv(folds_csv)
        return df.drop_duplicates("class_id").set_index("class_id")["class_name"].to_dict()

    return {}


def generate_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: dict[int, str],
    pres_dir: Path,
    model_name: str = "InceptionResNetV2",
    screening_folds_label: str = "Screening OOF folds 3, 0, 4",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate confusion matrix PNG files (counts and normalized).

    Args:
        y_true: True label array.
        y_pred: Predicted label array.
        class_names: Mapping from class_id to class_name.
        pres_dir: Output directory for presentation artifacts.
        model_name: Model name for plot titles.
        screening_folds_label: Screening scope label for titles.

    Returns:
        Tuple of (cm_counts, cm_normalized) arrays.
    """
    labels = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))
    label_names = [class_names.get(lbl, str(lbl)) for lbl in labels]

    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    # Counts matrix
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    im = ax.imshow(cm_counts, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(
        f"{model_name} Exp A\n{screening_folds_label} — Counts",
        fontsize=14, pad=15,
    )
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(label_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(pres_dir / "confusion_matrix_screening_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Normalized matrix
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title(
        f"{model_name} Exp A\n{screening_folds_label} — Normalized",
        fontsize=14, pad=15,
    )
    plt.colorbar(im, ax=ax)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(label_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(pres_dir / "confusion_matrix_screening_normalized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save CSVs
    pd.DataFrame(cm_counts, index=label_names, columns=label_names).to_csv(
        pres_dir / "confusion_matrix_screening_counts.csv"
    )
    pd.DataFrame(cm_norm, index=label_names, columns=label_names).to_csv(
        pres_dir / "confusion_matrix_screening_normalized.csv"
    )

    logger.info(f"Confusion matrices saved to {pres_dir}")
    return cm_counts, cm_norm


def generate_training_curves(output_dir: Path) -> None:
    """Generate training curve plots per fold with phase boundaries.

    Creates per-fold plots:
    - Loss per epoch (phases 1, 2, 3 marked)
    - Accuracy per epoch (phases 1, 2, 3 marked)
    - Learning rate per epoch if available

    Does NOT invent curves; skips silently if history is missing.

    Args:
        output_dir: Root output directory.
    """
    curves_dir = output_dir / "reports" / ARCH_NAME / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    hist_dir = output_dir / "reports" / ARCH_NAME / "history"
    if not hist_dir.is_dir():
        logger.info("No history directory found; skipping training curves.")
        return

    for fold in SCREENING_FOLDS:
        hist_file = hist_dir / f"fold_{fold}_history.json"
        if not hist_file.is_file():
            logger.info(f"History for fold {fold} not found; skipping.")
            continue

        with open(hist_file, "r", encoding="utf-8") as fp:
            hist = json.load(fp)

        phase_boundaries = hist.get("phase_boundaries", [])

        epochs = range(1, len(hist.get("loss", [])) + 1)
        if not epochs:
            continue

        # Loss curve
        if "loss" in hist and "val_loss" in hist:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            ax.plot(epochs, hist["loss"], label="Train Loss", color="steelblue")
            ax.plot(epochs, hist["val_loss"], label="Val Loss", color="tomato")
            for pb in phase_boundaries:
                ax.axvline(x=pb, linestyle="--", color="gray", alpha=0.7, label=f"Phase boundary @epoch {pb}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"InceptionResNetV2 Exp A — Fold {fold} Loss")
            ax.legend()
            plt.tight_layout()
            plt.savefig(curves_dir / f"fold_{fold}_loss.png", dpi=150)
            plt.close(fig)

        # Accuracy curve
        if "accuracy" in hist and "val_accuracy" in hist:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            ax.plot(epochs, hist["accuracy"], label="Train Accuracy", color="steelblue")
            ax.plot(epochs, hist["val_accuracy"], label="Val Accuracy", color="tomato")
            for pb in phase_boundaries:
                ax.axvline(x=pb, linestyle="--", color="gray", alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"InceptionResNetV2 Exp A — Fold {fold} Accuracy")
            ax.legend()
            plt.tight_layout()
            plt.savefig(curves_dir / f"fold_{fold}_accuracy.png", dpi=150)
            plt.close(fig)

    logger.info(f"Training curves generated in {curves_dir}")


def generate_inception_resnet_v2_screening_report(output_dir: Path) -> dict[str, Any]:
    """Generate all screening report artifacts from OOF prediction CSVs.

    Args:
        output_dir: Root output directory containing reports/inception_resnet_v2.

    Returns:
        Dictionary of overall metrics.

    Raises:
        FileNotFoundError: If required OOF CSV files are missing.
    """
    out_dir = Path(output_dir).resolve()
    oof_dir = out_dir / "reports" / ARCH_NAME / "predictions"

    # Load OOF predictions for screening folds 3, 0, 4
    fold_csvs = [oof_dir / f"fold_{f}_oof_predictions.csv" for f in SCREENING_FOLDS]
    missing = [str(f) for f in fold_csvs if not f.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing screening OOF prediction CSVs: {missing}"
        )

    oof_df = pd.concat([pd.read_csv(f) for f in fold_csvs], ignore_index=True)
    total_samples = len(oof_df)

    # Warn if not 259
    if total_samples != EXPECTED_TOTAL:
        logger.warning(
            f"Expected {EXPECTED_TOTAL} OOF samples for screening, found {total_samples}. "
            "Check fold CSV files."
        )

    class_names = load_class_mapping(out_dir)

    pres_dir = out_dir / "reports" / ARCH_NAME / "presentation_screening"
    pres_dir.mkdir(parents=True, exist_ok=True)

    y_true = oof_df["true_label"].values
    y_pred = oof_df["predicted_label"].values
    correct_samples = int(np.sum(y_true == y_pred))

    # 1. Confusion matrices
    cm_counts, cm_norm = generate_confusion_matrices(
        y_true, y_pred, class_names, pres_dir,
        screening_folds_label="Screening OOF folds 3, 0, 4",
    )

    # 2. Classification reports
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(
        pres_dir / "classification_report_screening.csv"
    )
    with open(pres_dir / "classification_report_screening.json", "w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)

    # 3. Top confusions
    labels = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm_counts[i, j] > 0:
                confusions.append({
                    "true_class_id": labels[i],
                    "true_class_name": class_names.get(labels[i], str(labels[i])),
                    "predicted_class_id": labels[j],
                    "predicted_class_name": class_names.get(labels[j], str(labels[j])),
                    "count": int(cm_counts[i, j]),
                })
    df_conf = pd.DataFrame(confusions).sort_values("count", ascending=False) if confusions else pd.DataFrame()
    df_conf.to_csv(pres_dir / "top_confusions_screening.csv", index=False)

    # 4. Overall metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    # Top-3 accuracy
    prob_cols = [f"prob_{c}" for c in range(22)]
    if all(c in oof_df.columns for c in prob_cols):
        probs = oof_df[prob_cols].values
        top3 = np.argsort(probs, axis=1)[:, -3:]
        top3_correct = sum(
            int(y_true[i]) in top3[i].tolist() for i in range(len(y_true))
        )
        top3_acc = top3_correct / len(y_true)
    else:
        top3_acc = None

    overall = {
        "model": "InceptionResNetV2",
        "experiment": "inception-resnet-v2-exp-a-high-performance",
        "scope": "Screening Folds 3, 0, 4 (259 images)",
        "scope_warning": (
            "259 images de screening — NE PAS comparer directement à "
            "une métrique sur 432 images sans indiquer les périmètres."
        ),
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
        "top_3_accuracy": round(top3_acc, 4) if top3_acc is not None else None,
        "reference_densenet121_screening_same_folds": {
            "mean_accuracy": 0.8610,
            "mean_macro_f1": 0.7889,
            "mean_weighted_f1": 0.8538,
            "note": "DenseNet121 Exp D sur folds 0, 3, 4 uniquement",
        },
    }

    with open(pres_dir / "overall_metrics_screening.json", "w", encoding="utf-8") as fp:
        json.dump(overall, fp, indent=2)

    # 5. Model statistics
    model_stats = {
        "architecture": "InceptionResNetV2",
        "input_shape": [299, 299, 3],
        "num_classes": 22,
        "preprocessing": "Rescaling(scale=1.0/127.5, offset=-1.0)",
        "input_range": "[0, 255] float32 → [-1, 1] float32",
        "backbone_bn_frozen": True,
        "training_phases": 3,
        "screening_folds": SCREENING_FOLDS,
        "total_screening_samples": total_samples,
    }
    with open(pres_dir / "model_statistics.json", "w", encoding="utf-8") as fp:
        json.dump(model_stats, fp, indent=2)

    # 6. Presentation summary markdown
    summary_md = f"""# InceptionResNetV2 Expérience A — Rapport de Screening (Folds 3, 0, 4)

> **Avertissement** : Ce système est à visée éducative uniquement.
> Il ne constitue en aucun cas un outil de diagnostic clinique.

## Résumé Général
- **Modèle** : InceptionResNetV2 (ImageNet pre-trained, résolution 299×299)
- **Périmètre** : {total_samples} images OOF de screening (folds 3, 0, 4)
- **⚠ Ce rapport couvre {total_samples} images sur 432 (screening uniquement)**
- **Échantillons correctement classés** : {correct_samples} / {total_samples}
- **Accuracy** : {acc:.4f}
- **Balanced Accuracy** : {bal_acc:.4f}
- **Cohen's Kappa** : {kappa:.4f}
- **Macro F1** : {macro_f1:.4f}
- **Weighted F1** : {weighted_f1:.4f}
{f"- **Top-3 Accuracy** : {top3_acc:.4f}" if top3_acc is not None else ""}

## Référence DenseNet121 Exp D (Mêmes Folds 0, 3, 4)
- **Mean Accuracy** : ~0.8610
- **Mean Macro F1** : ~0.7889
- **Mean Weighted F1** : ~0.8538

## Note de Portée
Les métriques de screening (259 images) ne sont pas comparables directement
aux métriques OOF complètes (432 images). La comparaison doit se faire
sur le même périmètre.

## Recommandation Screening
Analyser les métriques et la comparaison avec DenseNet121 avant de
décider d'entraîner les folds 1 et 2.
Ne pas lancer automatiquement les folds 1 et 2.
"""
    with open(pres_dir / "presentation_summary.md", "w", encoding="utf-8") as fp:
        fp.write(summary_md)

    # 7. Training curves
    generate_training_curves(out_dir)

    logger.info(
        f"InceptionResNetV2 screening report generated in {pres_dir} "
        f"({total_samples} samples, accuracy={acc:.4f}, macro_f1={macro_f1:.4f})"
    )
    return overall


def main() -> None:
    """CLI entrypoint for generating InceptionResNetV2 screening report."""
    parser = argparse.ArgumentParser(
        description="Generate InceptionResNetV2 Exp A screening report"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory containing models and reports",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    generate_inception_resnet_v2_screening_report(Path(args.output_dir))


if __name__ == "__main__":
    main()
