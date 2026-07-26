"""Complementarity and ensemble analysis between DenseNet121 and InceptionV3.

This script compares Out-Of-Fold (OOF) predictions between DenseNet121 Exp D
and InceptionV3 Exp A, evaluating error intersection, disagreement rates,
probability correlations, and a simple 50/50 average probability ensemble.
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
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


def load_predictions(densenet_csv: Path, inception_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DenseNet121 OOF CSV and available InceptionV3 OOF CSVs.

    Args:
        densenet_csv: Path to DenseNet121 OOF CSV.
        inception_dir: Directory containing InceptionV3 fold OOF CSVs.

    Returns:
        Tuple of (densenet_df, inception_df) aligned on matching image paths/ids.

    Raises:
        FileNotFoundError: If prediction files cannot be found.
    """
    if not densenet_csv.is_file():
        raise FileNotFoundError(f"DenseNet121 OOF file not found: {densenet_csv}")

    dense_df = pd.read_csv(densenet_csv)

    inc_files = sorted(list(inception_dir.glob("fold_*_oof_predictions.csv")))
    if not inc_files:
        # Try screening OOF CSV if present
        screening_file = inception_dir / "inceptionv3_screening_oof_predictions.csv"
        if screening_file.is_file():
            inc_files = [screening_file]
        else:
            raise FileNotFoundError(f"No InceptionV3 OOF prediction files found in {inception_dir}")

    inc_df = pd.concat([pd.read_csv(fp) for fp in inc_files], ignore_index=True)

    # Match on image_path or image_id
    match_col = "image_path" if "image_path" in dense_df.columns and "image_path" in inc_df.columns else "image_id"
    common_keys = set(dense_df[match_col]).intersection(set(inc_df[match_col]))
    if not common_keys:
        raise ValueError(f"No matching {match_col} entries found between DenseNet and InceptionV3 OOF predictions.")

    dense_aligned = dense_df[dense_df[match_col].isin(common_keys)].sort_values(match_col).reset_index(drop=True)
    inc_aligned = inc_df[inc_df[match_col].isin(common_keys)].sort_values(match_col).reset_index(drop=True)

    logger.info(f"Loaded and aligned predictions for {len(common_keys)} images based on '{match_col}'.")
    return dense_aligned, inc_aligned


def analyze_complementarity(dense_df: pd.DataFrame, inc_df: pd.DataFrame) -> dict[str, Any]:
    """Compute complementarity metrics and 50/50 ensemble performance.

    Args:
        dense_df: Aligned DenseNet121 dataframe.
        inc_df: Aligned InceptionV3 dataframe.

    Returns:
        Dictionary containing comprehensive analysis results.
    """
    total = len(dense_df)
    y_true = dense_df["true_label"].values

    dense_pred = dense_df["predicted_label"].values
    inc_pred = inc_df["predicted_label"].values

    dense_correct = (dense_pred == y_true)
    inc_correct = (inc_pred == y_true)

    both_correct = int(np.sum(dense_correct & inc_correct))
    both_wrong = int(np.sum((~dense_correct) & (~inc_correct)))
    dense_only_correct = int(np.sum(dense_correct & (~inc_correct)))
    inception_only_correct = int(np.sum((~dense_correct) & inc_correct))

    agreement_count = int(np.sum(dense_pred == inc_pred))
    agreement_rate = float(agreement_count / total)

    # Probability correlation
    prob_cols = [c for c in dense_df.columns if c.startswith("prob_")]
    dense_probs = dense_df[prob_cols].values
    inc_probs = inc_df[prob_cols].values

    flat_corr = float(np.corrcoef(dense_probs.flatten(), inc_probs.flatten())[0, 1])

    # 50/50 Simple Ensemble
    ensemble_probs = (dense_probs + inc_probs) / 2.0
    ensemble_pred = np.argmax(ensemble_probs, axis=1)

    ens_acc = float(accuracy_score(y_true, ensemble_pred))
    ens_macro_f1 = float(f1_score(y_true, ensemble_pred, average="macro", zero_division=0))
    ens_weighted_f1 = float(f1_score(y_true, ensemble_pred, average="weighted", zero_division=0))

    dense_acc = float(accuracy_score(y_true, dense_pred))
    dense_macro_f1 = float(f1_score(y_true, dense_pred, average="macro", zero_division=0))

    inc_acc = float(accuracy_score(y_true, inc_pred))
    inc_macro_f1 = float(f1_score(y_true, inc_pred, average="macro", zero_division=0))

    results = {
        "images_evaluated": total,
        "single_model_metrics": {
            "densenet121": {"accuracy": round(dense_acc, 4), "macro_f1": round(dense_macro_f1, 4)},
            "inceptionv3": {"accuracy": round(inc_acc, 4), "macro_f1": round(inc_macro_f1, 4)},
        },
        "complementarity_breakdown": {
            "both_correct": both_correct,
            "both_incorrect": both_wrong,
            "densenet_only_correct": dense_only_correct,
            "inceptionv3_only_correct": inception_only_correct,
            "agreement_rate": round(agreement_rate, 4),
            "disagreement_rate": round(1.0 - agreement_rate, 4),
            "probability_pearson_correlation": round(flat_corr, 4),
        },
        "simple_ensemble_50_50": {
            "accuracy": round(ens_acc, 4),
            "macro_f1": round(ens_macro_f1, 4),
            "weighted_f1": round(ens_weighted_f1, 4),
            "accuracy_gain_vs_best_single": round(ens_acc - max(dense_acc, inc_acc), 4),
            "macro_f1_gain_vs_best_single": round(ens_macro_f1 - max(dense_macro_f1, inc_macro_f1), 4),
        },
    }

    return results


def main() -> None:
    """CLI entrypoint for OOF comparison and ensemble evaluation."""
    parser = argparse.ArgumentParser(description="Compare DenseNet and InceptionV3 OOF predictions")
    parser.add_argument(
        "--densenet-oof",
        default="artifacts/models/densenet121_exp_d_v1/evaluation/oof_predictions_without_tta.csv",
        help="Path to DenseNet121 OOF CSV",
    )
    parser.add_argument(
        "--inception-oof-dir",
        default="reports/inceptionv3/predictions",
        help="Directory containing InceptionV3 fold OOF CSVs",
    )
    parser.add_argument(
        "--output",
        default="reports/inceptionv3/densenet_inceptionv3_complementarity.json",
        help="Destination path for analysis JSON report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    dense_path = Path(args.densenet_oof)
    inc_dir = Path(args.inception_oof_dir)
    out_path = Path(args.output)

    dense_df, inc_df = load_predictions(dense_path, inc_dir)
    results = analyze_complementarity(dense_df, inc_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    logger.info(f"Complementarity report saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
