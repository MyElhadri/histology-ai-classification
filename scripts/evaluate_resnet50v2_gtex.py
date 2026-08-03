"""Robust evaluation for ResNet50V2 on the GTEx 11-class dataset.

Primary scientific unit
-----------------------
Patch-level metrics are always the primary reported metrics.

Optional secondary aggregation
------------------------------
GTEx donors may contribute several tissue classes. Therefore, aggregating only
by donor_id is invalid for a multi-tissue classifier. This script selects, in
order, a slide identifier, a sample/specimen identifier, or the pair
(donor_id, true_label). The last option is reported explicitly as
"donor_tissue", not as donor-level evaluation.

Outputs
-------
For the requested split, the script writes:
- <split>_metrics.json
- <split>_predictions.csv
- <split>_classification_report.csv
- <split>_confusion_matrix_counts.csv/.png
- <split>_confusion_matrix_normalized.csv/.png
- <split>_top_confusions.csv
- optional <split>_<aggregation>_metrics.json and predictions.csv
- <split>_evaluation_manifest.json

The test split requires the explicit --confirm-test flag. A completed test
result in the same output directory is not overwritten unless
--allow-repeat-test is also supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
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

from src.data.gtex_integrity import (  # noqa: E402
    find_donor_column,
    parse_gtex_class_mapping,
    resolve_gtex_metadata_columns,
)
from src.data.gtex_pipeline import create_gtex_dataset  # noqa: E402
from src.models.resnet50v2 import build_resnet50v2_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


SLIDE_COLUMN_CANDIDATES = (
    "slide_id",
    "slide",
    "wsi_id",
    "whole_slide_id",
    "image_slide_id",
)
SAMPLE_COLUMN_CANDIDATES = (
    "sample_id",
    "sample",
    "specimen_id",
    "specimen",
    "tissue_sample_id",
    "aliquot_id",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 without loading a large model into RAM at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """JSON serializer for NumPy and Path values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the real column name matching one of the candidates."""
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def compute_topk_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 2-D, got {y_prob.shape}")
    k = min(k, y_prob.shape[1])
    topk = np.argpartition(y_prob, kth=y_prob.shape[1] - k, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == y_true[:, None], axis=1)))


def extract_top_confusions(cm: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    confusions: list[dict[str, Any]] = []
    for true_idx in range(len(cm)):
        for pred_idx in range(len(cm)):
            count = int(cm[true_idx, pred_idx])
            if true_idx != pred_idx and count > 0:
                confusions.append(
                    {
                        "true_class": class_names[true_idx],
                        "predicted_class": class_names[pred_idx],
                        "count": count,
                    }
                )
    return sorted(confusions, key=lambda item: item["count"], reverse=True)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute metrics with a fixed 0..C-1 class order."""
    labels = np.arange(len(class_names), dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "n_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "cross_entropy": float(log_loss(y_true, y_prob, labels=labels)),
        "top3_accuracy": compute_topk_accuracy(y_true, y_prob, k=3),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "top_confusions": extract_top_confusions(cm, class_names),
    }


def save_classification_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for name, values in report.items():
        if isinstance(values, dict):
            rows.append({"class_or_average": name, **values})
        else:
            rows.append({"class_or_average": name, "accuracy": values})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_confusion_matrix_artifacts(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    prefix: str,
) -> None:
    counts_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    counts_df.index.name = "true_class"
    counts_df.to_csv(output_dir / f"{prefix}_confusion_matrix_counts.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )
    norm_df = pd.DataFrame(normalized, index=class_names, columns=class_names)
    norm_df.index.name = "true_class"
    norm_df.to_csv(output_dir / f"{prefix}_confusion_matrix_normalized.csv")

    for matrix, suffix, value_format in (
        (cm, "counts", "d"),
        (normalized, "normalized", ".2f"),
    ):
        figure_size = max(9.0, len(class_names) * 0.85)
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        image = ax.imshow(matrix)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Classe prédite",
            ylabel="Classe réelle",
            title=f"Matrice de confusion — {prefix} ({suffix})",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        threshold = float(np.nanmax(matrix)) / 2.0 if matrix.size else 0.0
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text = format(int(value), value_format) if value_format == "d" else format(float(value), value_format)
                ax.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if float(value) > threshold else "black",
                    fontsize=7,
                )
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_confusion_matrix_{suffix}.png", dpi=180)
        plt.close(fig)


def validate_predictions(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    expected_rows: int,
    num_classes: int,
) -> None:
    if y_prob.shape != (expected_rows, num_classes):
        raise ValueError(
            f"Expected probabilities shape {(expected_rows, num_classes)}, got {y_prob.shape}"
        )
    if np.isnan(y_prob).any() or np.isinf(y_prob).any():
        raise ValueError("NaN or infinite values found in predicted probabilities")
    sums = np.sum(y_prob, axis=1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        raise ValueError(
            "Probabilities do not sum to 1 within atol=1e-3; "
            f"observed range: [{sums.min():.6f}, {sums.max():.6f}]"
        )
    if y_true.shape != (expected_rows,):
        raise ValueError(f"Expected labels shape {(expected_rows,)}, got {y_true.shape}")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise TypeError(f"Labels must be integers, got {y_true.dtype}")
    if y_true.min() < 0 or y_true.max() >= num_classes:
        raise ValueError(
            f"Label range [{y_true.min()}, {y_true.max()}] is incompatible with {num_classes} classes"
        )


def select_aggregation(
    pred_df: pd.DataFrame,
    donor_col: str | None,
    requested: str,
) -> tuple[str | None, list[str]]:
    """Select a scientifically valid secondary aggregation key."""
    if requested == "none":
        return None, []

    slide_col = find_column(pred_df.columns, SLIDE_COLUMN_CANDIDATES)
    sample_col = find_column(pred_df.columns, SAMPLE_COLUMN_CANDIDATES)

    if requested == "slide":
        if slide_col is None:
            raise ValueError(
                f"--aggregation slide requested, but no slide column was found. Columns: {list(pred_df.columns)}"
            )
        return "slide", [slide_col]

    if requested == "sample":
        if sample_col is None:
            raise ValueError(
                f"--aggregation sample requested, but no sample/specimen column was found. "
                f"Columns: {list(pred_df.columns)}"
            )
        return "sample", [sample_col]

    if requested == "donor_tissue":
        if donor_col is None:
            raise ValueError("--aggregation donor_tissue requested, but no donor column was found")
        return "donor_tissue", [donor_col, "true_label"]

    if requested != "auto":
        raise ValueError(f"Unknown aggregation mode: {requested}")

    if slide_col is not None:
        return "slide", [slide_col]
    if sample_col is not None:
        return "sample", [sample_col]
    if donor_col is not None:
        return "donor_tissue", [donor_col, "true_label"]
    return None, []


def aggregate_predictions(
    pred_df: pd.DataFrame,
    group_columns: list[str],
    class_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Average patch probabilities inside coherent slide/sample/donor-tissue groups."""
    probability_columns = [f"prob_{index}" for index in range(len(class_names))]
    records: list[dict[str, Any]] = []

    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_id, group in pred_df.groupby(grouper, sort=False, dropna=False):
        unique_labels = group["true_label"].dropna().unique()
        if len(unique_labels) != 1:
            raise ValueError(
                f"Aggregation group {group_id!r} contains multiple true labels: {unique_labels.tolist()}"
            )

        true_label = int(unique_labels[0])
        mean_probabilities = group[probability_columns].mean(axis=0).to_numpy(dtype=np.float64)
        predicted_label = int(np.argmax(mean_probabilities))

        if not isinstance(group_id, tuple):
            group_id = (group_id,)

        record: dict[str, Any] = {
            column: value for column, value in zip(group_columns, group_id, strict=True)
        }
        record.update(
            {
                "n_patches": int(len(group)),
                "true_label": true_label,
                "true_class": class_names[true_label],
                "predicted_label": predicted_label,
                "predicted_class": class_names[predicted_label],
                "correct": bool(true_label == predicted_label),
            }
        )
        for index, probability in enumerate(mean_probabilities):
            record[f"prob_{index}"] = float(probability)
        records.append(record)

    aggregated_df = pd.DataFrame(records)
    y_true = aggregated_df["true_label"].to_numpy(dtype=np.int64)
    y_pred = aggregated_df["predicted_label"].to_numpy(dtype=np.int64)
    y_prob = aggregated_df[probability_columns].to_numpy(dtype=np.float64)
    return aggregated_df, y_true, y_pred, y_prob


def save_metric_bundle(
    metrics: dict[str, Any],
    output_dir: Path,
    prefix: str,
    class_names: list[str],
) -> None:
    write_json(output_dir / f"{prefix}_metrics.json", metrics)
    save_classification_report(
        metrics["classification_report"],
        output_dir / f"{prefix}_classification_report.csv",
    )
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_matrix_artifacts(cm, class_names, output_dir, prefix)
    pd.DataFrame(metrics["top_confusions"]).to_csv(
        output_dir / f"{prefix}_top_confusions.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--aggregation",
        default="auto",
        choices=["auto", "none", "slide", "sample", "donor_tissue"],
        help=(
            "Secondary aggregation. 'auto' selects slide, then sample/specimen, then "
            "(donor_id, true_label). Patch-level metrics are always primary."
        ),
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required explicit confirmation before evaluating the held-out test split.",
    )
    parser.add_argument(
        "--allow-repeat-test",
        action="store_true",
        help="Allow overwriting/repeating an already completed test evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for required_path, description in (
        (args.config, "configuration"),
        (args.checkpoint, "checkpoint"),
        (args.dataset_dir, "dataset directory"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {description}: {required_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed_test_metrics = args.output_dir / "test_metrics.json"
    if args.split == "test":
        if not args.confirm_test:
            raise RuntimeError(
                "Refusing to evaluate the held-out TEST split without --confirm-test. "
                "Run validation first, freeze the model, then add --confirm-test exactly once."
            )
        if completed_test_metrics.exists() and not args.allow_repeat_test:
            raise RuntimeError(
                f"A completed test evaluation already exists at {completed_test_metrics}. "
                "Refusing to repeat it. Use --allow-repeat-test only for a documented technical rerun."
            )
        logger.warning("FINAL TEST EVALUATION CONFIRMED. Do not use these results for model selection.")

    checkpoint_hash = sha256_file(args.checkpoint)
    config_hash = sha256_file(args.config)
    logger.info("Checkpoint SHA256: %s", checkpoint_hash)

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    mapping_path = args.dataset_dir / config["dataset"]["class_mapping"]
    with mapping_path.open("r", encoding="utf-8") as handle:
        parsed_mapping = parse_gtex_class_mapping(json.load(handle))
    class_names = list(parsed_mapping["classes"])
    num_classes = len(class_names)

    expected_num_classes = int(config["dataset"]["num_classes"])
    if num_classes != expected_num_classes:
        raise ValueError(
            f"Class mapping contains {num_classes} classes but config expects {expected_num_classes}"
        )

    metadata_path = args.dataset_dir / "metadata" / f"{args.split}.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    df = pd.read_csv(metadata_path)
    donor_col = find_donor_column(df.columns.tolist())
    _, label_column = resolve_gtex_metadata_columns(df)
    y_true = df[label_column].to_numpy(dtype=np.int64)

    tf.keras.backend.clear_session()
    model = build_resnet50v2_model(config, weights=None)
    model.load_weights(str(args.checkpoint))

    batch_size = int(config["training"]["batch_size"])
    image_size = tuple(config["dataset"]["input_size"])
    dataset = create_gtex_dataset(
        args.dataset_dir,
        args.split,
        is_training=False,
        batch_size=batch_size,
        image_size=image_size,
    )

    logger.info("Running patch-level predictions for split=%s ...", args.split)
    y_prob = np.asarray(model.predict(dataset, verbose=1), dtype=np.float64)
    validate_predictions(y_prob, y_true, len(df), num_classes)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int64)

    pred_df = df.copy()
    pred_df["true_label"] = y_true
    pred_df["true_class"] = [class_names[index] for index in y_true]
    pred_df["predicted_label"] = y_pred
    pred_df["predicted_class"] = [class_names[index] for index in y_pred]
    pred_df["correct"] = y_true == y_pred
    for index in range(num_classes):
        pred_df[f"prob_{index}"] = y_prob[:, index]
    pred_df.to_csv(args.output_dir / f"{args.split}_predictions.csv", index=False)

    patch_metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    patch_metrics.update(
        {
            "split": args.split,
            "evaluation_unit": "patch",
            "class_names": class_names,
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": config_hash,
        }
    )
    save_metric_bundle(patch_metrics, args.output_dir, args.split, class_names)

    aggregation_name, group_columns = select_aggregation(pred_df, donor_col, args.aggregation)
    aggregation_summary: dict[str, Any] = {
        "requested": args.aggregation,
        "selected": aggregation_name,
        "group_columns": group_columns,
        "donor_column": donor_col,
    }

    if aggregation_name is not None:
        logger.info(
            "Computing secondary %s aggregation using columns %s ...",
            aggregation_name,
            group_columns,
        )
        group_df, group_true, group_pred, group_prob = aggregate_predictions(
            pred_df,
            group_columns,
            class_names,
        )
        group_prefix = f"{args.split}_{aggregation_name}"
        group_df.to_csv(args.output_dir / f"{group_prefix}_predictions.csv", index=False)

        group_metrics = compute_metrics(group_true, group_pred, group_prob, class_names)
        group_metrics.update(
            {
                "split": args.split,
                "evaluation_unit": aggregation_name,
                "group_columns": group_columns,
                "class_names": class_names,
                "checkpoint_sha256": checkpoint_hash,
                "config_sha256": config_hash,
            }
        )
        save_metric_bundle(group_metrics, args.output_dir, group_prefix, class_names)
        aggregation_summary["n_groups"] = int(len(group_df))
        logger.info("Secondary aggregation completed for %d groups.", len(group_df))
    else:
        logger.info("No valid secondary aggregation selected; patch-level evaluation remains complete.")

    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(args.config),
        "config_sha256": config_hash,
        "dataset_dir": str(args.dataset_dir),
        "metadata_csv": str(metadata_path),
        "n_patches": int(len(df)),
        "class_names": class_names,
        "primary_evaluation_unit": "patch",
        "secondary_aggregation": aggregation_summary,
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
    }
    write_json(args.output_dir / f"{args.split}_evaluation_manifest.json", manifest)

    logger.info("Evaluation finished successfully.")
    logger.info(
        "Primary patch metrics: accuracy=%.4f macro_f1=%.4f balanced_accuracy=%.4f top3=%.4f",
        patch_metrics["accuracy"],
        patch_metrics["macro_f1"],
        patch_metrics["balanced_accuracy"],
        patch_metrics["top3_accuracy"],
    )


if __name__ == "__main__":
    main()
