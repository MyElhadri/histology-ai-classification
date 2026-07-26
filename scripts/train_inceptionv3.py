"""Training and screening script for InceptionV3 Expérience A.

This script implements the fair comparison training pipeline for InceptionV3,
including fold completion checks, class weight export, dataset preprocessing validation,
two-phase fine-tuning with frozen BatchNormalization layers, OOF prediction export,
sample counter verification, and screening summary generation.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path before importing 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset
from src.evaluation.evaluate import evaluate_model
from src.models.inceptionv3 import (
    apply_inceptionv3_fine_tuning_strategy,
    build_inceptionv3,
    validate_dataset_preprocessing,
    validate_inceptionv3_strategy,
)
from src.training.train import _compute_fold_class_weights, _load_class_mapping
from src.utils.config import load_yaml
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def is_fold_complete(output_dir: Path, fold: int) -> bool:
    """Check if a fold training run is completely finished and all outputs exist.

    Args:
        output_dir: Root output directory.
        fold: Fold index.

    Returns:
        True if all required fold outputs exist and are non-empty, False otherwise.
    """
    out_dir = Path(output_dir).resolve()
    required_files = [
        out_dir / "models" / "inceptionv3" / "checkpoints" / f"fold_{fold}" / "best_model.keras",
        out_dir / "reports" / "inceptionv3" / "metrics" / f"fold_{fold}.json",
        out_dir / "reports" / "inceptionv3" / "predictions" / f"fold_{fold}_oof_predictions.csv",
        out_dir / "reports" / "inceptionv3" / "history" / f"fold_{fold}_history.json",
        out_dir / "reports" / "inceptionv3" / "history" / f"fold_{fold}_history.csv",
        out_dir / "reports" / "inceptionv3" / "class_weights" / f"fold_{fold}_class_weights.json",
    ]
    return all(f.is_file() and f.stat().st_size > 0 for f in required_files)


def verify_dataset_manifest_and_folds(config: dict[str, Any]) -> None:
    """Verify that dataset manifest contains 432 images and correct fold counts.

    Args:
        config: Configuration dictionary.

    Raises:
        ValueError: If total images or fold distributions do not match protocol.
    """
    manifest_path = Path(config["data"]["folds_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    df = pd.read_csv(manifest_path)
    total_images = len(df)
    if total_images != 432:
        raise ValueError(f"Expected exactly 432 images in manifest, found {total_images}")

    expected_folds = {0: 87, 1: 87, 2: 86, 3: 86, 4: 86}
    actual_folds = df["fold"].value_counts().to_dict()

    for fold_id, expected_count in expected_folds.items():
        actual_count = actual_folds.get(fold_id, 0)
        if actual_count != expected_count:
            raise ValueError(
                f"Fold {fold_id} expected {expected_count} validation images, found {actual_count}"
            )

    logger.info("Dataset manifest and fold distribution verified (432 total images).")


def export_fold_class_weights(config: dict[str, Any], fold: int, output_dir: Path) -> dict[int, float]:
    """Compute and export class weights for a given fold.

    Args:
        config: Configuration dictionary.
        fold: Fold index (0-4).
        output_dir: Root output directory.

    Returns:
        Dictionary mapping class_id (int) to weight (float).
    """
    computed_weights, _ = _compute_fold_class_weights(config, fold)

    cw_dir = output_dir / "reports" / "inceptionv3" / "class_weights"
    cw_dir.mkdir(parents=True, exist_ok=True)
    cw_file = cw_dir / f"fold_{fold}_class_weights.json"

    id_to_name = _load_class_mapping(config)
    export_dict = {
        "fold": fold,
        "class_weights": {
            str(cid): {
                "class_id": cid,
                "class_name": id_to_name.get(cid, str(cid)),
                "weight": float(w)
            }
            for cid, w in computed_weights.items()
        }
    }

    with open(cw_file, "w", encoding="utf-8") as fp:
        json.dump(export_dict, fp, indent=2)

    logger.info(f"Class weights for fold {fold} saved to {cw_file}")
    return computed_weights


def export_oof_predictions(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    config: dict[str, Any],
    fold: int,
    output_csv_path: Path
) -> pd.DataFrame:
    """Export Out-Of-Fold (OOF) predictions to CSV in mandatory schema.

    Args:
        model: Trained Keras Model.
        val_dataset: Validation dataset for fold.
        config: Configuration dictionary.
        fold: Fold index.
        output_csv_path: Destination CSV file path.

    Returns:
        DataFrame of generated OOF predictions.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    id_to_name = _load_class_mapping(config)
    num_classes = config["data"]["num_classes"]

    manifest_path = Path(config["data"]["folds_path"])
    df = pd.read_csv(manifest_path)
    val_df = df[df["fold"] == fold].copy()

    # Predict probabilities
    prob_matrix = model.predict(val_dataset, verbose=0)
    pred_class_ids = np.argmax(prob_matrix, axis=1)

    rows = []
    for idx, (_, row_data) in enumerate(val_df.iloc[:len(prob_matrix)].iterrows()):
        true_cid = int(row_data["class_id"])
        pred_cid = int(pred_class_ids[idx])
        probs = prob_matrix[idx]
        image_id = row_data.get("image_id", row_data.get("id", idx))

        row = {
            "image_path": row_data.get("image_path", f"fold_{fold}_img_{idx}"),
            "image_id": image_id,
            "fold": fold,
            "true_label": true_cid,
            "true_class": id_to_name.get(true_cid, str(true_cid)),
            "predicted_label": pred_cid,
            "predicted_class": id_to_name.get(pred_cid, str(pred_cid)),
            "correct": bool(true_cid == pred_cid),
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(probs[c])
        rows.append(row)

    fieldnames = [
        "image_path", "image_id", "fold", "true_label", "true_class",
        "predicted_label", "predicted_class", "correct"
    ] + [f"prob_{c}" for c in range(num_classes)]

    oof_df = pd.DataFrame(rows)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"OOF predictions for fold {fold} saved to {output_csv_path}")
    return oof_df


def generate_screening_summary(output_dir: Path, screening_folds: list[int] = [0, 3, 4]) -> bool:
    """Generate screening summary JSON if all requested screening fold metrics exist.

    Args:
        output_dir: Root output directory.
        screening_folds: List of fold indices comprising screening phase.

    Returns:
        True if summary was generated, False otherwise.
    """
    metrics_dir = output_dir / "reports" / "inceptionv3" / "metrics"
    pred_dir = output_dir / "reports" / "inceptionv3" / "predictions"
    fold_metrics = {}
    oof_images_list = []

    for f in screening_folds:
        metric_file = metrics_dir / f"fold_{f}.json"
        if not metric_file.is_file():
            logger.info(f"Screening fold {f} metrics missing ({metric_file}); summary deferred.")
            return False
        with open(metric_file, "r", encoding="utf-8") as fp:
            fold_metrics[f"fold_{f}"] = json.load(fp)

        pred_file = pred_dir / f"fold_{f}_oof_predictions.csv"
        if pred_file.is_file():
            pred_df = pd.read_csv(pred_file)
            oof_images_list.extend(pred_df["image_path"].tolist())

    accs = [m["accuracy"] for m in fold_metrics.values()]
    macro_prec = [m.get("macro_precision", 0.0) for m in fold_metrics.values()]
    macro_rec = [m.get("macro_recall", 0.0) for m in fold_metrics.values()]
    macro_f1s = [m["macro_f1"] for m in fold_metrics.values()]
    weighted_prec = [m.get("weighted_precision", 0.0) for m in fold_metrics.values()]
    weighted_rec = [m.get("weighted_recall", 0.0) for m in fold_metrics.values()]
    weighted_f1s = [m["weighted_f1"] for m in fold_metrics.values()]
    durations = [m.get("training_duration_seconds", 0.0) for m in fold_metrics.values()]

    summary = {
        "model": "InceptionV3",
        "experiment": "inceptionv3-exp-a-fair-comparison",
        "executed_folds": screening_folds,
        "metrics_by_fold": fold_metrics,
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy": round(float(np.std(accs)), 4),
        "mean_macro_precision": round(float(np.mean(macro_prec)), 4),
        "mean_macro_recall": round(float(np.mean(macro_rec)), 4),
        "mean_macro_f1": round(float(np.mean(macro_f1s)), 4),
        "std_macro_f1": round(float(np.std(macro_f1s)), 4),
        "mean_weighted_precision": round(float(np.mean(weighted_prec)), 4),
        "mean_weighted_recall": round(float(np.mean(weighted_rec)), 4),
        "mean_weighted_f1": round(float(np.mean(weighted_f1s)), 4),
        "std_weighted_f1": round(float(np.std(weighted_f1s)), 4),
        "total_oof_images": len(oof_images_list),
        "unique_images": len(set(oof_images_list)),
        "total_duration_seconds": round(float(sum(durations)), 2),
        "reference_densenet121_exp_d_screening_average": {
            "mean_accuracy": 0.8610,
            "mean_macro_f1": 0.7889,
            "mean_weighted_f1": 0.8538,
        },
        "conclusion": (
            "Screening baseline comparison against DenseNet121 D (folds 0, 3, 4). "
            "Do not automatically declare InceptionV3 superior."
        ),
    }

    summary_path = output_dir / "reports" / "inceptionv3" / "inceptionv3_screening_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    logger.info(f"Screening summary successfully generated at {summary_path}")
    return True


def train_fold_inceptionv3(config: dict[str, Any], fold: int, output_dir: Path) -> dict[str, Any]:
    """Train InceptionV3 on a single fold.

    Args:
        config: Configuration dictionary.
        fold: Fold index.
        output_dir: Root output directory.

    Returns:
        Evaluation metrics dictionary.
    """
    output_dir = Path(output_dir).resolve()
    start_time = time.time()
    seed = config["project"]["seed"]
    set_global_seed(seed)

    logger.info(f"\n==========================================")
    logger.info(f"Starting InceptionV3 Fold {fold} (seed={seed})")
    logger.info(f"==========================================")

    # 1. Export & compute class weights
    class_weights = export_fold_class_weights(config, fold, output_dir)

    # 2. Create datasets
    train_ds = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=True,
        batch_size=config["training"]["batch_size"],
        image_size=tuple(config["data"]["image_size"]),
        num_classes=config["data"]["num_classes"],
        augmentation_config=config.get("augmentation"),
    )
    val_ds = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=False,
        batch_size=config["training"]["batch_size"],
        image_size=tuple(config["data"]["image_size"]),
        num_classes=config["data"]["num_classes"],
    )

    # 3. Validate dataset preprocessing [0, 255] float32
    validate_dataset_preprocessing(train_ds, config)
    validate_dataset_preprocessing(val_ds, config)

    # 4. Build model
    head_cfg = config["model"]["classifier_head"]
    model = build_inceptionv3(
        num_classes=config["data"]["num_classes"],
        input_shape=tuple(config["data"]["image_size"]) + (3,),
        weights=config["model"]["weights"],
        dropout_rate=config["model"]["dropout_rate"],
        head_config=head_cfg,
    )

    ce_hard_metric = tf.keras.metrics.CategoricalCrossentropy(name="ce_hard", label_smoothing=0.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # 5. Phase 1: Head Training (Backbone frozen)
    logger.info("Phase 1: Training head only (backbone frozen)")
    backbone = next(l for l in model.layers if isinstance(l, tf.keras.Model) or "inception_v3" in l.name.lower())
    backbone.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["head_learning_rate"]),
        loss=loss_fn,
        metrics=["accuracy", ce_hard_metric],
    )

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["head_epochs"],
        class_weight=class_weights,
    )

    # 6. Phase 2: Representation Fine-Tuning (BatchNorm frozen, Re-compile)
    logger.info("Phase 2: Full backbone fine-tuning (BatchNormalization frozen)")
    ft_cfg = config.get("fine_tuning", {})
    strategy = ft_cfg.get("strategy", "full")
    keep_bn_frozen = ft_cfg.get("keep_batch_normalization_frozen", True)

    apply_inceptionv3_fine_tuning_strategy(
        model=model,
        strategy=strategy,
        keep_batch_normalization_frozen=keep_bn_frozen,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["fine_tuning_learning_rate"]),
        loss=loss_fn,
        metrics=["accuracy", ce_hard_metric],
    )
    validate_inceptionv3_strategy(model, config)

    fold_ckpt_dir = output_dir / "models" / "inceptionv3" / "checkpoints" / f"fold_{fold}"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = fold_ckpt_dir / "best_model.keras"

    cb_cfg = config.get("callbacks", {})
    monitor_metric = cb_cfg.get("monitor", "val_ce_hard")
    monitor_mode = cb_cfg.get("mode", "min")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            save_best_only=True,
            monitor=monitor_metric,
            mode=monitor_mode,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=cb_cfg.get("early_stopping_patience", 5),
            min_delta=cb_cfg.get("min_delta", 0.002),
            restore_best_weights=cb_cfg.get("restore_best_weights", True),
            start_from_epoch=cb_cfg.get("start_from_epoch", 5),
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode=monitor_mode,
            factor=cb_cfg.get("reduce_lr_factor", 0.2),
            patience=cb_cfg.get("reduce_lr_patience", 2),
            min_lr=cb_cfg.get("min_learning_rate", 1e-7),
        ),
    ]

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["fine_tuning_epochs"],
        callbacks=callbacks,
        class_weight=class_weights,
    )

    if best_model_path.is_file():
        model.load_weights(str(best_model_path))
        logger.info(f"Loaded best weights from {best_model_path}")

    elapsed = round(time.time() - start_time, 2)

    # 7. Export OOF Predictions first
    pred_dir = output_dir / "reports" / "inceptionv3" / "predictions"
    oof_df = export_oof_predictions(model, val_ds, config, fold, pred_dir / f"fold_{fold}_oof_predictions.csv")

    # 8. Evaluate and Save Metrics with exact OOF sample counters
    metrics = evaluate_model(model, val_ds)
    metrics["training_duration_seconds"] = elapsed
    metrics["total_samples"] = len(oof_df)
    metrics["correct_samples"] = int(oof_df["correct"].sum())
    metrics["accuracy"] = float(metrics["correct_samples"] / metrics["total_samples"])

    metrics_dir = output_dir / "reports" / "inceptionv3" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"fold_{fold}.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # 9. Save History (JSON and CSV)
    hist_dir = output_dir / "reports" / "inceptionv3" / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    combined_history = {}
    for k in history_p1.history:
        combined_history[k] = list(history_p1.history[k]) + list(history_p2.history.get(k, []))
    with open(hist_dir / f"fold_{fold}_history.json", "w", encoding="utf-8") as fp:
        json.dump(combined_history, fp, indent=2)

    pd.DataFrame(combined_history).to_csv(hist_dir / f"fold_{fold}_history.csv", index=False)

    return metrics


def main() -> None:
    """CLI entrypoint for training InceptionV3."""
    parser = argparse.ArgumentParser(description="Train InceptionV3 Exp A")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--dataset-dir", default=None, help="Optional override for dataset root")
    parser.add_argument("--output-dir", required=True, help="Root directory for models and reports")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 3, 4], help="Folds to train (e.g. 0 3 4)")
    parser.add_argument("--overwrite", action="store_true", help="Force re-training even if fold is complete")
    parser.add_argument("--skip-completed", action="store_true", help="Skip folds that are already completed")
    parser.add_argument(
        "--generate-summary-only",
        action="store_true",
        help="Generate screening summary JSON without running training loop",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config = load_yaml(args.config)
    if args.dataset_dir is not None:
        config["data"]["dataset_root"] = args.dataset_dir

    verify_dataset_manifest_and_folds(config)
    output_dir = Path(args.output_dir).resolve()

    if args.generate_summary_only:
        logger.info(f"Generating screening summary for folds: {args.folds}")
        generate_screening_summary(output_dir, screening_folds=args.folds)
        return

    for fold in args.folds:
        if is_fold_complete(output_dir, fold):
            if args.skip_completed or not args.overwrite:
                logger.info(f"Fold {fold} is already complete. Skipping.")
                continue

        train_fold_inceptionv3(config, fold, output_dir)

    generate_screening_summary(output_dir, screening_folds=args.folds)


if __name__ == "__main__":
    main()
