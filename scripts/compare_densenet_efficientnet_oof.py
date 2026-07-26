"""Complementarity and ensemble analysis between DenseNet121 and EfficientNetV2B0.

This script compares Out-Of-Fold (OOF) predictions between DenseNet121 Exp D
and EfficientNetV2B0 Exp A, evaluating error intersection, disagreement rates,
probability correlations, and a simple 50/50 average probability ensemble.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


def load_predictions(densenet_csv: Path, efficientnet_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DenseNet121 OOF CSV and available EfficientNetV2B0 OOF CSVs.

    Args:
        densenet_csv: Path to DenseNet121 OOF CSV.
        efficientnet_dir: Directory containing EfficientNet fold OOF CSVs.

    Returns:
        Tuple of (densenet_df, efficientnet_df) aligned on matching image paths.

    Raises:
        FileNotFoundError: If prediction files cannot be found.
    """
    if not densenet_csv.is_file():
        raise FileNotFoundError(f"DenseNet121 OOF file not found: {densenet_csv}")

    dense_df = pd.read_csv(densenet_csv)

    # Collect all available fold CSVs for EfficientNet
    eff_files = sorted(list(efficientnet_dir.glob("fold_*_oof_predictions.csv")))
    if not eff_files:
        raise FileNotFoundError(f"No EfficientNet OOF prediction files found in {efficientnet_dir}")

    eff_df = pd.concat([pd.read_csv(fp) for fp in eff_files], ignore_index=True)

    # Align on image_path
    common_paths = set(dense_df["image_path"]).intersection(set(eff_df["image_path"]))
    if not common_paths:
        raise ValueError("No matching image_path entries found between DenseNet and EfficientNet OOF predictions.")

    dense_aligned = dense_df[dense_df["image_path"].isin(common_paths)].sort_values("image_path").reset_index(drop=True)
    eff_aligned = eff_df[eff_df["image_path"].isin(common_paths)].sort_values("image_path").reset_index(drop=True)

    logger.info(f"Loaded and aligned predictions for {len(common_paths)} images.")
    return dense_aligned, eff_aligned


def analyze_complementarity(dense_df: pd.DataFrame, eff_df: pd.DataFrame) -> dict[str, Any]:
    """Compute complementarity metrics and 50/50 ensemble performance.

    Args:
        dense_df: Aligned DenseNet121 dataframe.
        eff_df: Aligned EfficientNetV2B0 dataframe.

    Returns:
        Dictionary containing comprehensive analysis results.
    """
    total = len(dense_df)
    y_true = dense_df["true_label"].values

    dense_pred = dense_df["predicted_label"].values
    eff_pred = eff_df["predicted_label"].values

    dense_correct = (dense_pred == y_true)
    eff_correct = (eff_pred == y_true)

    both_correct = int(np.sum(dense_correct & eff_correct))
    both_wrong = int(np.sum((~dense_correct) & (~eff_correct)))
    dense_only_correct = int(np.sum(dense_correct & (~eff_correct)))
    eff_only_correct = int(np.sum((~dense_correct) & eff_correct))

    agreement_count = int(np.sum(dense_pred == eff_pred))
    agreement_rate = float(agreement_count / total)

    # Probability correlation
    prob_cols = [c for c in dense_df.columns if c.startswith("prob_")]
    dense_probs = dense_df[prob_cols].values
    eff_probs = eff_df[prob_cols].values

    # Flat Pearson correlation across all probabilities
    flat_corr = float(np.corrcoef(dense_probs.flatten(), eff_probs.flatten())[0, 1])

    # 50/50 Simple Ensemble
    ensemble_probs = (dense_probs + eff_probs) / 2.0
    ensemble_pred = np.argmax(ensemble_probs, axis=1)

    ens_acc = float(accuracy_score(y_true, ensemble_pred))
    ens_macro_f1 = float(f1_score(y_true, ensemble_pred, average="macro", zero_division=0))
    ens_weighted_f1 = float(f1_score(y_true, ensemble_pred, average="weighted", zero_division=0))

    dense_acc = float(accuracy_score(y_true, dense_pred))
    dense_macro_f1 = float(f1_score(y_true, dense_pred, average="macro", zero_division=0))

    eff_acc = float(accuracy_score(y_true, eff_pred))
    eff_macro_f1 = float(f1_score(y_true, eff_pred, average="macro", zero_division=0))

    results = {
        "images_evaluated": total,
        "single_model_metrics": {
            "densenet121": {"accuracy": round(dense_acc, 4), "macro_f1": round(dense_macro_f1, 4)},
            "efficientnetv2b0": {"accuracy": round(eff_acc, 4), "macro_f1": round(eff_macro_f1, 4)},
        },
        "complementarity_breakdown": {
            "both_correct": both_correct,
            "both_incorrect": both_wrong,
            "densenet_only_correct": dense_only_correct,
            "efficientnet_only_correct": eff_only_correct,
            "agreement_rate": round(agreement_rate, 4),
            "disagreement_rate": round(1.0 - agreement_rate, 4),
            "probability_pearson_correlation": round(flat_corr, 4),
        },
        "simple_ensemble_50_50": {
            "accuracy": round(ens_acc, 4),
            "macro_f1": round(ens_macro_f1, 4),
            "weighted_f1": round(ens_weighted_f1, 4),
            "accuracy_gain_vs_best_single": round(ens_acc - max(dense_acc, eff_acc), 4),
            "macro_f1_gain_vs_best_single": round(ens_macro_f1 - max(dense_macro_f1, eff_macro_f1), 4),
        },
    }

    return results


def main() -> None:
    """CLI entrypoint for OOF comparison and ensemble evaluation."""
    parser = argparse.ArgumentParser(description="Compare DenseNet and EfficientNet OOF predictions")
    parser.add_argument(
        "--densenet-oof",
        default="artifacts/models/densenet121_exp_d_v1/evaluation/oof_predictions_without_tta.csv",
        help="Path to DenseNet121 OOF CSV",
    )
    parser.add_argument(
        "--efficientnet-oof-dir",
        default="reports/efficientnetv2b0/predictions",
        help="Directory containing EfficientNet fold OOF CSVs",
    )
    parser.add_argument(
        "--output",
        default="reports/efficientnetv2b0/densenet_efficientnet_complementarity.json",
        help="Destination path for analysis JSON report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    dense_path = Path(args.densenet_oof)
    eff_dir = Path(args.efficientnet_oof_dir)
    out_path = Path(args.output)

    dense_df, eff_df = load_predictions(dense_path, eff_dir)
    results = analyze_complementarity(dense_df, eff_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    logger.info(f"Complementarity report saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
