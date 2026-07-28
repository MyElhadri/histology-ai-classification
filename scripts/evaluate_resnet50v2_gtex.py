"""Evaluation script for ResNet50V2 on GTEx dataset.

Computes comprehensive patch-level and donor-level metrics.
Produces predictions CSV and metrics JSON.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.data.gtex_integrity import find_donor_column
from src.data.gtex_pipeline import create_gtex_dataset
from src.models.resnet50v2 import build_resnet50v2_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_top3_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    top3_preds = np.argsort(y_prob, axis=1)[:, -3:]
    correct = [1 if y_true[i] in top3_preds[i] else 0 for i in range(len(y_true))]
    return float(np.mean(correct))


def extract_top_confusions(cm: np.ndarray, class_names: list[str]) -> list[dict]:
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j])
                })
    return sorted(confusions, key=lambda x: x["count"], reverse=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ce = log_loss(y_true, y_prob, labels=range(len(class_names)))
    top3 = compute_top3_accuracy(y_true, y_prob)

    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    top_confusions = extract_top_confusions(cm, class_names)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "cohens_kappa": float(kappa),
        "cross_entropy": float(ce),
        "top3_accuracy": float(top3),
        "classification_report": cr,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.split == "test":
        logger.warning("ATTENTION: Executing TEST split evaluation. This should only be done once.")
        import hashlib
        with open(args.checkpoint, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            logger.info(f"Test checkpoint SHA256: {h}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Class Mapping
    map_path = args.dataset_dir / config["dataset"]["class_mapping"]
    from src.data.gtex_integrity import parse_gtex_class_mapping
    with open(map_path, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
        
    parsed = parse_gtex_class_mapping(raw_mapping)
    class_names = parsed["classes"]

    # Read Metadata for donor and paths
    csv_path = args.dataset_dir / "metadata" / f"{args.split}.csv"
    df = pd.read_csv(csv_path)
    donor_col = find_donor_column(df.columns.tolist())

    # Build Model
    model = build_resnet50v2_model(config, weights=None)
    model.load_weights(str(args.checkpoint))

    # Dataset
    batch_size = config["training"]["batch_size"]
    img_size = tuple(config["dataset"]["input_size"])
    ds = create_gtex_dataset(args.dataset_dir, args.split, is_training=False, batch_size=batch_size, image_size=img_size)

    logger.info("Running patch-level predictions...")
    y_prob = model.predict(ds, verbose=1)
    
    # Assertions
    assert not np.isnan(y_prob).any(), "NaN found in probabilities"
    sums = np.sum(y_prob, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), "Probabilities do not sum to 1"
    assert len(y_prob) == len(df), f"Expected {len(df)} predictions, got {len(y_prob)}"

    y_pred = np.argmax(y_prob, axis=1)
    y_true = df["class_id"].values

    # 1. Patch-Level Metrics
    metrics_patch = compute_metrics(y_true, y_pred, y_prob, class_names)
    metrics_patch["donor_level_available"] = bool(donor_col)
    
    with open(args.output_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_patch, f, indent=4)

    # 2. Patch-Level Predictions CSV
    pred_df = df.copy()
    pred_df["true_label"] = y_true
    pred_df["true_class"] = [class_names[t] for t in y_true]
    pred_df["predicted_label"] = y_pred
    pred_df["predicted_class"] = [class_names[p] for p in y_pred]
    pred_df["correct"] = y_true == y_pred

    for i in range(len(class_names)):
        pred_df[f"prob_{i}"] = y_prob[:, i]

    pred_df.to_csv(args.output_dir / f"{args.split}_predictions.csv", index=False)

    # 3. Donor-Level Metrics
    if donor_col:
        logger.info("Computing donor-level aggregation...")
        donor_groups = pred_df.groupby(donor_col)
        
        donor_true = []
        donor_pred = []
        donor_probs = []
        donor_ids = []
        
        for d_id, group in donor_groups:
            # Check coherence
            if group["true_label"].nunique() > 1:
                raise ValueError(f"Donor {d_id} has patches with different true classes!")
                
            d_true = group["true_label"].iloc[0]
            
            # Mean prob over patches
            prob_cols = [f"prob_{i}" for i in range(len(class_names))]
            d_prob = group[prob_cols].mean().values
            d_pred = np.argmax(d_prob)
            
            donor_ids.append(d_id)
            donor_true.append(d_true)
            donor_pred.append(d_pred)
            donor_probs.append(d_prob)
            
        donor_true = np.array(donor_true)
        donor_pred = np.array(donor_pred)
        donor_probs = np.array(donor_probs)
        
        metrics_donor = compute_metrics(donor_true, donor_pred, donor_probs, class_names)
        
        with open(args.output_dir / f"{args.split}_donor_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_donor, f, indent=4)
            
        # Donor CSV
        d_df = pd.DataFrame({
            "donor_id": donor_ids,
            "true_label": donor_true,
            "true_class": [class_names[t] for t in donor_true],
            "predicted_label": donor_pred,
            "predicted_class": [class_names[p] for p in donor_pred],
            "correct": donor_true == donor_pred
        })
        for i in range(len(class_names)):
            d_df[f"prob_{i}"] = donor_probs[:, i]
            
        d_df.to_csv(args.output_dir / f"{args.split}_donor_predictions.csv", index=False)
        logger.info(f"Donor-level evaluation complete for {len(donor_ids)} donors.")
    else:
        logger.info("No donor column found. Skipping donor-level evaluation.")

    logger.info("Evaluation script finished successfully.")


if __name__ == "__main__":
    main()
