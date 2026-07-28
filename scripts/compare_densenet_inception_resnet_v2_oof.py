"""Complementarity and ensemble analysis between DenseNet121 Exp D and InceptionResNetV2 Exp A.

This script compares Out-Of-Fold (OOF) predictions between DenseNet121 Exp D
and InceptionResNetV2 Exp A, evaluating:
- Per-model metrics
- Error intersection and complementarity
- Disagreement rate
- Probability correlation
- Simple 50/50 ensemble metrics

Alignment: strict on image_id + fold + true_label.
Aborts on mismatches, duplicates, or missing data.

IMPORTANT: The 50/50 ensemble weights are NOT optimized on the 259 screening images.
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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

SCREENING_FOLDS = [3, 0, 4]
NUM_CLASSES = 22
PROB_COLS = [f"prob_{c}" for c in range(NUM_CLASSES)]


def load_and_validate_predictions(
    densenet_dir: Path,
    inception_resnet_v2_dir: Path,
    folds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and strictly align DenseNet121 and InceptionResNetV2 OOF predictions.

    Alignment key: image_id + fold + true_label.

    Args:
        densenet_dir: Directory containing DenseNet121 fold_X_oof_predictions.csv files.
        inception_resnet_v2_dir: Directory containing InceptionResNetV2 fold_X_oof_predictions.csv files.
        folds: Fold indices to compare.

    Returns:
        Tuple of (dense_aligned_df, irv2_aligned_df) with strict alignment.

    Raises:
        FileNotFoundError: If any required prediction file is missing.
        ValueError: If alignment fails due to mismatches, duplicates, or missing probabilities.
    """
    # Load DenseNet predictions
    dense_dfs = []
    for fold in folds:
        dense_file = densenet_dir / f"fold_{fold}_oof_predictions.csv"
        if not dense_file.is_file():
            raise FileNotFoundError(f"DenseNet121 OOF file not found: {dense_file}")
        dense_dfs.append(pd.read_csv(dense_file))
    dense_df = pd.concat(dense_dfs, ignore_index=True)

    # Load InceptionResNetV2 predictions
    irv2_dfs = []
    for fold in folds:
        irv2_file = inception_resnet_v2_dir / f"fold_{fold}_oof_predictions.csv"
        if not irv2_file.is_file():
            raise FileNotFoundError(f"InceptionResNetV2 OOF file not found: {irv2_file}")
        irv2_dfs.append(pd.read_csv(irv2_file))
    irv2_df = pd.concat(irv2_dfs, ignore_index=True)

    # Check for duplicates
    if dense_df["image_id"].duplicated().any():
        raise ValueError(
            f"Duplicates in DenseNet121 predictions: "
            f"{dense_df['image_id'][dense_df['image_id'].duplicated()].tolist()[:5]}"
        )
    if irv2_df["image_id"].duplicated().any():
        raise ValueError(
            f"Duplicates in InceptionResNetV2 predictions: "
            f"{irv2_df['image_id'][irv2_df['image_id'].duplicated()].tolist()[:5]}"
        )

    # Verify probability columns exist
    for col in PROB_COLS:
        if col not in dense_df.columns:
            raise ValueError(f"Missing column '{col}' in DenseNet121 OOF DataFrame.")
        if col not in irv2_df.columns:
            raise ValueError(f"Missing column '{col}' in InceptionResNetV2 OOF DataFrame.")

    # Align strictly on image_id + fold
    common_ids = set(dense_df["image_id"]).intersection(set(irv2_df["image_id"]))
    if not common_ids:
        raise ValueError("No matching image_id entries found between DenseNet121 and InceptionResNetV2 OOF predictions.")

    missing_dense = set(irv2_df["image_id"]) - set(dense_df["image_id"])
    missing_irv2 = set(dense_df["image_id"]) - set(irv2_df["image_id"])
    if missing_dense:
        raise ValueError(
            f"{len(missing_dense)} image_ids present in InceptionResNetV2 but missing from DenseNet121."
        )
    if missing_irv2:
        raise ValueError(
            f"{len(missing_irv2)} image_ids present in DenseNet121 but missing from InceptionResNetV2."
        )

    # Verify image count matches
    if len(dense_df) != len(irv2_df):
        raise ValueError(
            f"Image count mismatch: DenseNet121={len(dense_df)}, InceptionResNetV2={len(irv2_df)}"
        )

    # Align by image_id
    dense_aligned = dense_df.sort_values("image_id").reset_index(drop=True)
    irv2_aligned = irv2_df.sort_values("image_id").reset_index(drop=True)

    # Verify true_label consistency
    mismatched = dense_aligned["true_label"].values != irv2_aligned["true_label"].values
    if mismatched.any():
        mismatch_count = int(mismatched.sum())
        raise ValueError(
            f"true_label mismatch for {mismatch_count} images after alignment. "
            "Check class_mapping consistency between models."
        )

    logger.info(
        f"Aligned {len(dense_aligned)} images from folds {folds}. "
        "DenseNet121 and InceptionResNetV2 predictions consistent."
    )
    return dense_aligned, irv2_aligned


def compute_complementarity(
    dense_df: pd.DataFrame,
    irv2_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute full complementarity analysis and 50/50 ensemble.

    The 50/50 ensemble formula:
        p_ensemble = (p_densenet + p_inception_resnet_v2) / 2

    Weights are NOT optimized on the screening data.

    Args:
        dense_df: Aligned DenseNet121 dataframe.
        irv2_df: Aligned InceptionResNetV2 dataframe.

    Returns:
        Comprehensive analysis results dictionary.
    """
    total = len(dense_df)
    y_true = dense_df["true_label"].values

    dense_pred = dense_df["predicted_label"].values
    irv2_pred = irv2_df["predicted_label"].values

    dense_correct = dense_pred == y_true
    irv2_correct = irv2_pred == y_true

    both_correct = int(np.sum(dense_correct & irv2_correct))
    both_wrong = int(np.sum((~dense_correct) & (~irv2_correct)))
    dense_only_correct = int(np.sum(dense_correct & (~irv2_correct)))
    irv2_only_correct = int(np.sum((~dense_correct) & irv2_correct))

    agreement_count = int(np.sum(dense_pred == irv2_pred))
    agreement_rate = float(agreement_count / total)

    # Probability correlation
    dense_probs = dense_df[PROB_COLS].values
    irv2_probs = irv2_df[PROB_COLS].values
    flat_corr = float(np.corrcoef(dense_probs.flatten(), irv2_probs.flatten())[0, 1])

    # 50/50 Simple Ensemble (fixed weights, not optimized)
    ensemble_probs = (dense_probs + irv2_probs) / 2.0
    ensemble_pred = np.argmax(ensemble_probs, axis=1)

    # Per-model metrics
    dense_acc = float(accuracy_score(y_true, dense_pred))
    dense_macro_f1 = float(f1_score(y_true, dense_pred, average="macro", zero_division=0))
    dense_weighted_f1 = float(f1_score(y_true, dense_pred, average="weighted", zero_division=0))
    dense_macro_prec = float(precision_score(y_true, dense_pred, average="macro", zero_division=0))
    dense_macro_rec = float(recall_score(y_true, dense_pred, average="macro", zero_division=0))

    irv2_acc = float(accuracy_score(y_true, irv2_pred))
    irv2_macro_f1 = float(f1_score(y_true, irv2_pred, average="macro", zero_division=0))
    irv2_weighted_f1 = float(f1_score(y_true, irv2_pred, average="weighted", zero_division=0))
    irv2_macro_prec = float(precision_score(y_true, irv2_pred, average="macro", zero_division=0))
    irv2_macro_rec = float(recall_score(y_true, irv2_pred, average="macro", zero_division=0))

    # Ensemble metrics
    ens_acc = float(accuracy_score(y_true, ensemble_pred))
    ens_macro_f1 = float(f1_score(y_true, ensemble_pred, average="macro", zero_division=0))
    ens_weighted_f1 = float(f1_score(y_true, ensemble_pred, average="weighted", zero_division=0))

    # Qualification assessment
    ens_beats_dense_f1 = ens_macro_f1 > dense_macro_f1
    ens_not_lower_acc = ens_acc >= dense_acc

    return {
        "images_evaluated": total,
        "folds_compared": SCREENING_FOLDS,
        "scope_warning": (
            "Comparison on 259 screening images (folds 3, 0, 4) only. "
            "Not representative of full 432-image validation."
        ),
        "single_model_metrics": {
            "densenet121_exp_d": {
                "accuracy": round(dense_acc, 4),
                "macro_f1": round(dense_macro_f1, 4),
                "weighted_f1": round(dense_weighted_f1, 4),
                "macro_precision": round(dense_macro_prec, 4),
                "macro_recall": round(dense_macro_rec, 4),
            },
            "inception_resnet_v2_exp_a": {
                "accuracy": round(irv2_acc, 4),
                "macro_f1": round(irv2_macro_f1, 4),
                "weighted_f1": round(irv2_weighted_f1, 4),
                "macro_precision": round(irv2_macro_prec, 4),
                "macro_recall": round(irv2_macro_rec, 4),
            },
        },
        "complementarity_breakdown": {
            "both_correct": both_correct,
            "both_incorrect": both_wrong,
            "densenet_only_correct": dense_only_correct,
            "inception_resnet_v2_only_correct": irv2_only_correct,
            "agreement_count": agreement_count,
            "agreement_rate": round(agreement_rate, 4),
            "disagreement_rate": round(1.0 - agreement_rate, 4),
            "probability_pearson_correlation": round(flat_corr, 4),
        },
        "simple_ensemble_50_50": {
            "formula": "p_ensemble = (p_densenet + p_inception_resnet_v2) / 2",
            "weights_optimized_on_screening": False,
            "accuracy": round(ens_acc, 4),
            "macro_f1": round(ens_macro_f1, 4),
            "weighted_f1": round(ens_weighted_f1, 4),
            "accuracy_gain_vs_densenet": round(ens_acc - dense_acc, 4),
            "macro_f1_gain_vs_densenet": round(ens_macro_f1 - dense_macro_f1, 4),
            "accuracy_gain_vs_best_single": round(ens_acc - max(dense_acc, irv2_acc), 4),
            "macro_f1_gain_vs_best_single": round(ens_macro_f1 - max(dense_macro_f1, irv2_macro_f1), 4),
        },
        "qualification_by_complementarity": {
            "ensemble_macro_f1_beats_densenet": ens_beats_dense_f1,
            "ensemble_accuracy_not_lower_than_densenet": ens_not_lower_acc,
            "screening_qualified_ensemble_only": ens_beats_dense_f1 and ens_not_lower_acc,
        },
    }


def build_per_image_comparison(
    dense_df: pd.DataFrame,
    irv2_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-image comparison DataFrame.

    Args:
        dense_df: Aligned DenseNet121 dataframe.
        irv2_df: Aligned InceptionResNetV2 dataframe.

    Returns:
        DataFrame with per-image comparison columns.
    """
    y_true = dense_df["true_label"].values
    dense_pred = dense_df["predicted_label"].values
    irv2_pred = irv2_df["predicted_label"].values
    dense_probs = dense_df[PROB_COLS].values
    irv2_probs = irv2_df[PROB_COLS].values
    ensemble_probs = (dense_probs + irv2_probs) / 2.0
    ensemble_pred = np.argmax(ensemble_probs, axis=1)

    rows = []
    for i in range(len(dense_df)):
        rows.append({
            "image_id": dense_df["image_id"].iloc[i],
            "fold": dense_df["fold"].iloc[i],
            "true_label": int(y_true[i]),
            "densenet_pred": int(dense_pred[i]),
            "irv2_pred": int(irv2_pred[i]),
            "ensemble_pred": int(ensemble_pred[i]),
            "densenet_correct": bool(dense_pred[i] == y_true[i]),
            "irv2_correct": bool(irv2_pred[i] == y_true[i]),
            "ensemble_correct": bool(ensemble_pred[i] == y_true[i]),
            "agreement": bool(dense_pred[i] == irv2_pred[i]),
            "densenet_confidence": float(dense_probs[i, dense_pred[i]]),
            "irv2_confidence": float(irv2_probs[i, irv2_pred[i]]),
        })
    return pd.DataFrame(rows)


def generate_complementarity_summary_md(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a markdown complementarity summary.

    Args:
        results: Analysis results dictionary.
        output_path: Destination markdown file path.
    """
    dense_m = results["single_model_metrics"]["densenet121_exp_d"]
    irv2_m = results["single_model_metrics"]["inception_resnet_v2_exp_a"]
    ens = results["simple_ensemble_50_50"]
    comp = results["complementarity_breakdown"]
    qual = results["qualification_by_complementarity"]

    md = f"""# Comparaison DenseNet121 Exp D vs InceptionResNetV2 Exp A

## Périmètre
- **{results['images_evaluated']} images** (folds {results['folds_compared']})
- ⚠ {results['scope_warning']}

## Métriques par Modèle

| Métrique | DenseNet121 Exp D | InceptionResNetV2 Exp A |
|---|---|---|
| Accuracy | {dense_m['accuracy']:.4f} | {irv2_m['accuracy']:.4f} |
| Macro F1 | {dense_m['macro_f1']:.4f} | {irv2_m['macro_f1']:.4f} |
| Weighted F1 | {dense_m['weighted_f1']:.4f} | {irv2_m['weighted_f1']:.4f} |

## Complémentarité

| Catégorie | Images |
|---|---|
| Les deux corrects | {comp['both_correct']} |
| Les deux incorrects | {comp['both_incorrect']} |
| DenseNet seul correct | {comp['densenet_only_correct']} |
| InceptionResNetV2 seul correct | {comp['inception_resnet_v2_only_correct']} |
| Taux d'accord | {comp['agreement_rate']:.4f} |
| Corrélation des probabilités | {comp['probability_pearson_correlation']:.4f} |

## Ensemble 50/50 (poids fixes, non optimisés)

| Métrique | Valeur | Gain vs DenseNet |
|---|---|---|
| Accuracy | {ens['accuracy']:.4f} | {ens['accuracy_gain_vs_densenet']:+.4f} |
| Macro F1 | {ens['macro_f1']:.4f} | {ens['macro_f1_gain_vs_densenet']:+.4f} |
| Weighted F1 | {ens['weighted_f1']:.4f} | — |

## Qualification par Complémentarité

- Ensemble Macro F1 > DenseNet : **{'✓' if qual['ensemble_macro_f1_beats_densenet'] else '✗'}**
- Ensemble Accuracy ≥ DenseNet : **{'✓' if qual['ensemble_accuracy_not_lower_than_densenet'] else '✗'}**
- Statut : **{'screening_qualified_ensemble_only' if qual['screening_qualified_ensemble_only'] else 'non qualifié par complémentarité'}**

## Recommandation
Ne pas optimiser les poids d'ensemble sur les 259 images de screening.
Analyser avant de décider des folds 1 et 2.
"""
    output_path.write_text(md, encoding="utf-8")
    logger.info(f"Complementarity summary markdown saved to {output_path}")


def main() -> None:
    """CLI entrypoint for DenseNet vs InceptionResNetV2 OOF comparison."""
    parser = argparse.ArgumentParser(
        description="Compare DenseNet121 Exp D and InceptionResNetV2 Exp A OOF predictions"
    )
    parser.add_argument(
        "--densenet-dir",
        required=True,
        help="Directory containing DenseNet121 fold_X_oof_predictions.csv files",
    )
    parser.add_argument(
        "--inception-resnet-v2-dir",
        required=True,
        help="Directory containing InceptionResNetV2 fold_X_oof_predictions.csv files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for comparison reports",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=SCREENING_FOLDS,
        help=f"Folds to compare (default: {SCREENING_FOLDS})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    dense_dir = Path(args.densenet_dir)
    irv2_dir = Path(args.inception_resnet_v2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate predictions
    dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, args.folds)

    # Complementarity analysis
    results = compute_complementarity(dense_df, irv2_df)

    # Per-image comparison
    per_image_df = build_per_image_comparison(dense_df, irv2_df)

    # Ensemble metrics JSON
    ens_metrics_path = output_dir / "ensemble_50_50_metrics.json"
    with open(ens_metrics_path, "w", encoding="utf-8") as fp:
        json.dump(results["simple_ensemble_50_50"], fp, indent=2)

    # Summary JSON
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # Per-image CSV
    per_image_path = output_dir / "per_image_comparison.csv"
    per_image_df.to_csv(per_image_path, index=False)

    # Complementarity markdown
    md_path = output_dir / "complementarity_summary.md"
    generate_complementarity_summary_md(results, md_path)

    logger.info(f"Comparison reports saved to {output_dir}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
