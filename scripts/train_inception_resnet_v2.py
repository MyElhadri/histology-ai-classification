"""Training and screening script for InceptionResNetV2 Expérience A — High Performance Histology.

This script implements three-phase fine-tuning for InceptionResNetV2:
  Phase 1: Head training (backbone frozen)
  Phase 2: Partial backbone fine-tuning (last 30%, BN frozen)
  Phase 3: Full backbone fine-tuning (all layers, BN frozen)

Features:
- Fold completion detection (--skip-completed)
- Partial fold detection with optional backup (--backup-partial)
- Dry-run mode (--dry-run): validates config/model without training
- Per-phase checkpoint saving
- OOF prediction export with mandatory schema
- Cumulative screening summary (reads all folds from disk)
- Class weight export per fold
- Training history export (JSON + CSV)
- Completion marker written last
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import Any

# ── project root must be on sys.path before any 'src' import ──────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset
from src.evaluation.evaluate import evaluate_model
from src.models.inception_resnet_v2 import (
    apply_full_inception_resnet_v2_fine_tuning,
    apply_partial_inception_resnet_v2_fine_tuning,
    build_inception_resnet_v2,
    freeze_inception_resnet_v2_backbone,
    validate_dataset_preprocessing,
    validate_inception_resnet_v2_trainability,
    validate_preprocessing_values,
)
from src.training.train import _compute_fold_class_weights, _load_class_mapping
from src.utils.config import load_yaml
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)

ARCH_NAME = "inception_resnet_v2"
EXPECTED_FOLD_SIZES = {0: 87, 1: 87, 2: 86, 3: 86, 4: 86}
SCREENING_FOLDS = [3, 0, 4]


# ─────────────────────────────────────────────────────────────────────────────
# Fold completion / validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fold_paths(output_dir: Path, fold: int) -> dict[str, Path]:
    """Return the canonical file paths for a fold."""
    ckpt_dir = output_dir / "models" / ARCH_NAME / "checkpoints" / f"fold_{fold}"
    rep_dir = output_dir / "reports" / ARCH_NAME
    return {
        "ckpt_dir": ckpt_dir,
        "phase_1_best": ckpt_dir / "phase_1_best.keras",
        "phase_2_best": ckpt_dir / "phase_2_best.keras",
        "phase_3_best": ckpt_dir / "phase_3_best.keras",
        "best_model": ckpt_dir / "best_model.keras",
        "selection_json": ckpt_dir / "selection.json",
        "metrics_json": rep_dir / "metrics" / f"fold_{fold}.json",
        "oof_csv": rep_dir / "predictions" / f"fold_{fold}_oof_predictions.csv",
        "history_json": rep_dir / "history" / f"fold_{fold}_history.json",
        "history_csv": rep_dir / "history" / f"fold_{fold}_history.csv",
        "class_weights_json": rep_dir / "class_weights" / f"fold_{fold}_class_weights.json",
        "model_info_json": rep_dir / "model_info" / f"fold_{fold}_model_info.json",
        "completion_json": rep_dir / "completion" / f"fold_{fold}_complete.json",
    }


def is_fold_complete(output_dir: Path, fold: int) -> bool:
    """Check if a fold is completely finished and all outputs are valid.

    A fold is complete only if:
    - The final checkpoint loads (existence check here; loading checked in validate_fold)
    - The metrics JSON is valid
    - The OOF CSV is valid (correct row count)
    - The history is valid
    - The class weights file exists
    - The completion marker is valid

    Args:
        output_dir: Root output directory.
        fold: Fold index.

    Returns:
        True if all required outputs exist and are non-empty, False otherwise.
    """
    paths = _fold_paths(output_dir, fold)
    required = [
        paths["best_model"],
        paths["metrics_json"],
        paths["oof_csv"],
        paths["history_json"],
        paths["history_csv"],
        paths["class_weights_json"],
        paths["completion_json"],
    ]
    if not all(f.is_file() and f.stat().st_size > 0 for f in required):
        return False

    # Validate completion marker
    try:
        with open(paths["completion_json"], "r", encoding="utf-8") as fp:
            marker = json.load(fp)
        if not marker.get("complete", False):
            return False
    except Exception:
        return False

    # Validate OOF CSV row count
    try:
        oof_df = pd.read_csv(paths["oof_csv"])
        expected = EXPECTED_FOLD_SIZES.get(fold)
        if expected and len(oof_df) != expected:
            return False
    except Exception:
        return False

    return True


def is_fold_partial(output_dir: Path, fold: int) -> bool:
    """Check if a fold has partial outputs (started but not complete)."""
    paths = _fold_paths(output_dir, fold)
    any_exists = any(
        p.is_file() for p in paths.values()
        if isinstance(p, Path)
    )
    return any_exists and not is_fold_complete(output_dir, fold)


def backup_partial_fold(output_dir: Path, fold: int) -> Path:
    """Rename partial fold checkpoint directory with timestamp.

    Args:
        output_dir: Root output directory.
        fold: Fold index.

    Returns:
        Path to the renamed backup directory.
    """
    ckpt_dir = output_dir / "models" / ARCH_NAME / "checkpoints" / f"fold_{fold}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ckpt_dir.parent / f"fold_{fold}_interrupted_{ts}"
    if ckpt_dir.exists():
        ckpt_dir.rename(backup_dir)
        logger.info(f"Partial fold {fold} checkpoint backed up to: {backup_dir}")
    return backup_dir


def verify_dataset_manifest_and_folds(config: dict[str, Any]) -> None:
    """Verify that dataset manifest contains 432 images and correct fold counts.

    Args:
        config: Configuration dictionary.

    Raises:
        FileNotFoundError: If manifest file is missing.
        ValueError: If total images or fold distributions do not match protocol.
    """
    manifest_path = Path(config["data"]["folds_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    df = pd.read_csv(manifest_path)
    total_images = len(df)
    if total_images != 432:
        raise ValueError(
            f"Expected exactly 432 images in manifest, found {total_images}"
        )

    for fold_id, expected_count in EXPECTED_FOLD_SIZES.items():
        actual_count = int((df["fold"] == fold_id).sum())
        if actual_count != expected_count:
            raise ValueError(
                f"Fold {fold_id}: expected {expected_count} validation images, "
                f"found {actual_count}"
            )

    logger.info("Dataset manifest and fold distribution verified (432 total images, 22 classes).")


# ─────────────────────────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────────────────────────

def export_fold_class_weights(
    config: dict[str, Any], fold: int, output_dir: Path
) -> dict[int, float]:
    """Compute and export class weights for a given fold.

    Args:
        config: Configuration dictionary.
        fold: Fold index (0-4).
        output_dir: Root output directory.

    Returns:
        Dictionary mapping class_id (int) to weight (float).
    """
    computed_weights, _ = _compute_fold_class_weights(config, fold)

    cw_dir = output_dir / "reports" / ARCH_NAME / "class_weights"
    cw_dir.mkdir(parents=True, exist_ok=True)
    cw_file = cw_dir / f"fold_{fold}_class_weights.json"

    id_to_name = _load_class_mapping(config)
    export_dict = {
        "fold": fold,
        "class_weights": {
            str(cid): {
                "class_id": cid,
                "class_name": id_to_name.get(cid, str(cid)),
                "weight": float(w),
            }
            for cid, w in computed_weights.items()
        },
    }

    with open(cw_file, "w", encoding="utf-8") as fp:
        json.dump(export_dict, fp, indent=2)

    logger.info(f"Class weights for fold {fold} saved to {cw_file}")
    return computed_weights


# ─────────────────────────────────────────────────────────────────────────────
# OOF predictions
# ─────────────────────────────────────────────────────────────────────────────

def export_oof_predictions(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    config: dict[str, Any],
    fold: int,
    output_csv_path: Path,
) -> pd.DataFrame:
    """Export Out-Of-Fold (OOF) predictions to CSV in mandatory schema.

    Schema columns:
        image_path, image_id, fold, true_label, true_class,
        predicted_label, predicted_class, correct, prob_0..prob_21

    Verifications:
        - One row per validation image
        - No duplicates
        - Probabilities sum ≈ 1.0
        - predicted_label == argmax(prob_0...prob_21)
        - correct == (true_label == predicted_label)
        - Counters: correct_samples, total_samples from CSV

    Args:
        model: Trained Keras Model.
        val_dataset: Validation dataset for fold (NO augmentation).
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
    val_df = df[df["fold"] == fold].copy().reset_index(drop=True)

    # Predict probabilities
    prob_matrix = model.predict(val_dataset, verbose=0)
    pred_class_ids = np.argmax(prob_matrix, axis=1)

    rows = []
    for idx in range(min(len(prob_matrix), len(val_df))):
        row_data = val_df.iloc[idx]
        true_cid = int(row_data["class_id"])
        pred_cid = int(pred_class_ids[idx])
        probs = prob_matrix[idx]
        image_id = row_data.get("image_id", row_data.get("id", idx))

        # Verify argmax consistency
        assert int(np.argmax(probs)) == pred_cid, (
            f"argmax mismatch at index {idx}: "
            f"argmax={np.argmax(probs)}, predicted_label={pred_cid}"
        )

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
        "predicted_label", "predicted_class", "correct",
    ] + [f"prob_{c}" for c in range(num_classes)]

    oof_df = pd.DataFrame(rows)

    # Verify no duplicates
    if oof_df["image_path"].duplicated().any():
        logger.warning(f"Fold {fold}: duplicate image_path entries detected in OOF CSV.")

    # Verify prob sums
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    prob_sums = oof_df[prob_cols].sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-4):
        logger.warning(
            f"Fold {fold}: some probability rows do not sum to 1.0 "
            f"(max deviation: {abs(prob_sums - 1.0).max():.6f})"
        )

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_samples = len(oof_df)
    correct_samples = int(oof_df["correct"].sum())
    logger.info(
        f"OOF predictions for fold {fold} saved to {output_csv_path}. "
        f"Total: {total_samples}, Correct: {correct_samples}, "
        f"Accuracy: {correct_samples / total_samples:.4f}"
    )
    return oof_df


# ─────────────────────────────────────────────────────────────────────────────
# Screening summary (cumulative — reads all completed folds from disk)
# ─────────────────────────────────────────────────────────────────────────────

def generate_screening_summary(
    output_dir: Path,
    screening_folds: list[int] | None = None,
) -> bool:
    """Generate cumulative screening summary JSON from all completed folds on disk.

    IMPORTANT: This function reads all fold data from disk (not from memory),
    so running it after fold 4 includes folds 3 and 0 already written.
    This prevents the bug observed in InceptionV3 where the summary was
    overwritten with only the last fold.

    Args:
        output_dir: Root output directory.
        screening_folds: List of fold indices comprising screening phase.
                         Defaults to [3, 0, 4].

    Returns:
        True if summary was generated (all screening folds present), False otherwise.
    """
    if screening_folds is None:
        screening_folds = SCREENING_FOLDS

    metrics_dir = output_dir / "reports" / ARCH_NAME / "metrics"
    pred_dir = output_dir / "reports" / ARCH_NAME / "predictions"
    fold_metrics: dict[str, dict] = {}
    oof_images_list: list[str] = []

    # Read ALL completed folds from disk (not just the current fold in memory)
    for f in screening_folds:
        metric_file = metrics_dir / f"fold_{f}.json"
        if not metric_file.is_file():
            logger.info(f"Screening fold {f} metrics missing; summary deferred.")
            return False
        with open(metric_file, "r", encoding="utf-8") as fp:
            fold_metrics[f"fold_{f}"] = json.load(fp)

        pred_file = pred_dir / f"fold_{f}_oof_predictions.csv"
        if pred_file.is_file():
            pred_df = pd.read_csv(pred_file)
            oof_images_list.extend(pred_df["image_path"].tolist())

    # Compute metrics from fold JSONs
    accs = [m["accuracy"] for m in fold_metrics.values()]
    macro_prec = [m.get("macro_precision", 0.0) for m in fold_metrics.values()]
    macro_rec = [m.get("macro_recall", 0.0) for m in fold_metrics.values()]
    macro_f1s = [m["macro_f1"] for m in fold_metrics.values()]
    weighted_prec = [m.get("weighted_precision", 0.0) for m in fold_metrics.values()]
    weighted_rec = [m.get("weighted_recall", 0.0) for m in fold_metrics.values()]
    weighted_f1s = [m["weighted_f1"] for m in fold_metrics.values()]
    durations = [m.get("training_duration_seconds", 0.0) for m in fold_metrics.values()]

    # Compute global OOF metrics (concat all predictions)
    global_oof_metrics: dict[str, Any] = {}
    all_pred_dfs = []
    for f in screening_folds:
        pred_file = pred_dir / f"fold_{f}_oof_predictions.csv"
        if pred_file.is_file():
            all_pred_dfs.append(pd.read_csv(pred_file))

    if all_pred_dfs:
        combined = pd.concat(all_pred_dfs, ignore_index=True)
        global_total = len(combined)
        global_correct = int(combined["correct"].sum())
        global_accuracy = global_correct / global_total if global_total > 0 else 0.0

        from sklearn.metrics import f1_score as sk_f1
        y_true_global = combined["true_label"].values
        y_pred_global = combined["predicted_label"].values
        global_macro_f1 = float(sk_f1(y_true_global, y_pred_global, average="macro", zero_division=0))
        global_weighted_f1 = float(sk_f1(y_true_global, y_pred_global, average="weighted", zero_division=0))

        global_oof_metrics = {
            "total_samples": global_total,
            "correct_samples": global_correct,
            "global_accuracy": round(global_accuracy, 4),
            "global_macro_f1": round(global_macro_f1, 4),
            "global_weighted_f1": round(global_weighted_f1, 4),
        }

    # Best and worst folds
    fold_acc_pairs = [(k, v["accuracy"]) for k, v in fold_metrics.items()]
    best_fold = max(fold_acc_pairs, key=lambda x: x[1])[0] if fold_acc_pairs else None
    worst_fold = min(fold_acc_pairs, key=lambda x: x[1])[0] if fold_acc_pairs else None

    # Determine screening status
    mean_acc = float(np.mean(accs))
    mean_mf1 = float(np.mean(macro_f1s))
    mean_wf1 = float(np.mean(weighted_f1s))
    global_acc = global_oof_metrics.get("global_accuracy", 0.0)
    global_mf1 = global_oof_metrics.get("global_macro_f1", 0.0)
    global_wf1 = global_oof_metrics.get("global_weighted_f1", 0.0)

    if len(fold_metrics) < len(screening_folds):
        screening_status = "screening_incomplete"
    elif (mean_acc >= 0.84 and mean_mf1 >= 0.76 and mean_wf1 >= 0.83):
        if global_acc >= 0.87 and global_mf1 >= 0.81 and global_wf1 >= 0.86:
            screening_status = "screening_qualified_individual"
        else:
            screening_status = "screening_qualified_individual"
    else:
        screening_status = "screening_rejected"

    summary = {
        "model": "InceptionResNetV2",
        "experiment": "inception-resnet-v2-exp-a-high-performance",
        "executed_folds": screening_folds,
        "total_oof_images": len(oof_images_list),
        "unique_images": len(set(oof_images_list)),
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
        "global_oof_metrics": global_oof_metrics,
        "best_fold": best_fold,
        "worst_fold": worst_fold,
        "total_duration_seconds": round(float(sum(durations)), 2),
        "screening_status": screening_status,
        "auto_proceed_to_full_training": False,  # NEVER automatic
        "reference_densenet121_exp_d_full_oof": {
            "accuracy": 0.8843,
            "macro_f1": 0.8280,
            "weighted_f1": 0.8794,
            "note": "DenseNet121 Exp D OOF complet (432 images, 5 folds)",
        },
        "reference_densenet121_exp_d_screening_folds_0_3_4": {
            "mean_accuracy": 0.8610,
            "mean_macro_f1": 0.7889,
            "mean_weighted_f1": 0.8538,
            "note": "DenseNet121 Exp D folds 0, 3, 4 uniquement — même périmètre",
        },
        "scope_warning": (
            "Les métriques sur 259 images (screening folds 3, 0, 4) ne sont pas "
            "comparables directement aux métriques sur 432 images (OOF complet)."
        ),
        "conclusion": (
            "Comparer avec DenseNet121 Exp D sur le même périmètre. "
            "Ne pas lancer automatiquement les folds 1 et 2 sans analyse du screening."
        ),
    }

    summary_path = output_dir / "reports" / ARCH_NAME / "inception_resnet_v2_screening_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    logger.info(
        f"Screening summary generated: {len(fold_metrics)}/{len(screening_folds)} folds, "
        f"status={screening_status}, path={summary_path}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Core fold training
# ─────────────────────────────────────────────────────────────────────────────

def train_fold_inception_resnet_v2(
    config: dict[str, Any],
    fold: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Train InceptionResNetV2 on a single fold with three-phase fine-tuning.

    Phase 1: Frozen backbone, head training
    Phase 2: Partial fine-tuning (last 30% of backbone, BN frozen)
    Phase 3: Full fine-tuning (all backbone layers, BN frozen)

    Args:
        config: Configuration dictionary.
        fold: Fold index.
        output_dir: Root output directory.

    Returns:
        Evaluation metrics dictionary.
    """
    output_dir = Path(output_dir).resolve()
    start_time = time.time()
    seed = config.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    logger.info(f"\n{'='*50}")
    logger.info(f"InceptionResNetV2 Exp A — Fold {fold} (seed={seed})")
    logger.info(f"{'='*50}")

    paths = _fold_paths(output_dir, fold)
    for path in paths.values():
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export class weights
    class_weights = export_fold_class_weights(config, fold, output_dir)

    # 2. Create datasets (train augmented, val original-only)
    image_size = tuple(config["data"]["image_size"])
    batch_size = config["training"]["batch_size"]
    num_classes = config["data"]["num_classes"]

    train_ds = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=True,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        augmentation_config=config.get("augmentation"),
    )
    val_ds = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=False,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
    )

    # 3. Validate dataset preprocessing
    validate_dataset_preprocessing(train_ds, config)
    validate_dataset_preprocessing(val_ds, config)
    validate_preprocessing_values()

    # 4. Mixed precision
    if config["training"].get("mixed_precision", True):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision: mixed_float16 enabled.")

    # 5. Build model (backbone called with training=False always)
    head_cfg = config["model"]["classifier_head"]
    model = build_inception_resnet_v2(
        num_classes=num_classes,
        input_shape=image_size + (3,),
        weights=config["model"]["weights"],
        head_config=head_cfg,
    )

    # Log model info
    total_params = model.count_params()
    layer_summary = {
        "total_params": total_params,
        "architecture": "InceptionResNetV2",
        "input_shape": list(image_size) + [3],
        "output_shape": list(model.output_shape),
        "head_type": head_cfg.get("type", "article_inspired"),
    }
    model_info_path = paths["model_info_json"]
    model_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_info_path, "w", encoding="utf-8") as fp:
        json.dump(layer_summary, fp, indent=2)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    ce_hard_metric = tf.keras.metrics.CategoricalCrossentropy(name="ce_hard", label_smoothing=0.0)

    cb_cfg = config.get("callbacks", {})
    monitor_metric = cb_cfg.get("monitor", "val_ce_hard")
    monitor_mode = cb_cfg.get("mode", "min")

    phase1_cfg = config["training"]["phase_1"]
    phase2_cfg = config["training"]["phase_2"]
    phase3_cfg = config["training"]["phase_3"]

    # ── Phase 1: Head training (backbone frozen) ──────────────────────────────
    logger.info(f"Phase 1: Head training — backbone frozen, LR={phase1_cfg['learning_rate']}")
    phase1_start = time.time()

    freeze_inception_resnet_v2_backbone(model)
    validate_inception_resnet_v2_trainability(model, phase=1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=phase1_cfg["learning_rate"]),
        loss=loss_fn,
        metrics=["accuracy", ce_hard_metric],
    )

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_cfg["epochs"],
        class_weight=class_weights,
    )

    # Save phase 1 best checkpoint
    p1_best = paths["phase_1_best"]
    model.save(str(p1_best))
    logger.info(f"Phase 1 checkpoint saved: {p1_best}")
    phase1_duration = time.time() - phase1_start

    # ── Phase 2: Partial fine-tuning (last 30%, BN frozen) ───────────────────
    logger.info(
        f"Phase 2: Partial fine-tuning — last {phase2_cfg['trainable_backbone_fraction']*100:.0f}% "
        f"backbone, BN frozen, LR={phase2_cfg['learning_rate']}"
    )
    phase2_start = time.time()

    p2_summary = apply_partial_inception_resnet_v2_fine_tuning(
        model=model,
        trainable_fraction=phase2_cfg["trainable_backbone_fraction"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=phase2_cfg["learning_rate"]),
        loss=loss_fn,
        metrics=["accuracy", ce_hard_metric],
    )
    validate_inception_resnet_v2_trainability(model, phase=2)

    p2_ckpt_path = paths["phase_2_best"]
    p2_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(p2_ckpt_path),
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
        epochs=phase2_cfg["epochs"],
        callbacks=p2_callbacks,
        class_weight=class_weights,
    )

    if p2_ckpt_path.is_file():
        model.load_weights(str(p2_ckpt_path))
        logger.info(f"Loaded best Phase 2 weights from {p2_ckpt_path}")
    phase2_duration = time.time() - phase2_start

    # ── Phase 3: Full fine-tuning (all backbone, BN frozen) ──────────────────
    logger.info(
        f"Phase 3: Full fine-tuning — all backbone, BN frozen, LR={phase3_cfg['learning_rate']}"
    )
    phase3_start = time.time()

    p3_summary = apply_full_inception_resnet_v2_fine_tuning(
        model=model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=phase3_cfg["learning_rate"]),
        loss=loss_fn,
        metrics=["accuracy", ce_hard_metric],
    )
    validate_inception_resnet_v2_trainability(model, phase=3)

    p3_ckpt_path = paths["phase_3_best"]
    p3_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(p3_ckpt_path),
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

    history_p3 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase3_cfg["epochs"],
        callbacks=p3_callbacks,
        class_weight=class_weights,
    )

    if p3_ckpt_path.is_file():
        model.load_weights(str(p3_ckpt_path))
        logger.info(f"Loaded best Phase 3 weights from {p3_ckpt_path}")
    phase3_duration = time.time() - phase3_start
    total_duration = time.time() - start_time

    # ── Selection: best phase checkpoint as final model ───────────────────────
    # Use the phase 3 best (lowest val_ce_hard), fallback to phase 2
    best_phase = "phase_3" if p3_ckpt_path.is_file() else "phase_2"
    best_source = p3_ckpt_path if p3_ckpt_path.is_file() else p2_ckpt_path

    # Save final model
    best_model_path = paths["best_model"]
    model.save(str(best_model_path))

    selection_info = {
        "selected_phase": best_phase,
        "source_checkpoint": str(best_source),
        "final_checkpoint": str(best_model_path),
        "selection_criterion": "phase_3_best_val_ce_hard",
    }
    with open(paths["selection_json"], "w", encoding="utf-8") as fp:
        json.dump(selection_info, fp, indent=2)

    # ── OOF Predictions (val_ds, no augmentation) ─────────────────────────────
    oof_csv_path = paths["oof_csv"]
    oof_df = export_oof_predictions(model, val_ds, config, fold, oof_csv_path)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = evaluate_model(model, val_ds)
    metrics["training_duration_seconds"] = round(total_duration, 2)
    metrics["phase_1_duration_seconds"] = round(phase1_duration, 2)
    metrics["phase_2_duration_seconds"] = round(phase2_duration, 2)
    metrics["phase_3_duration_seconds"] = round(phase3_duration, 2)
    metrics["total_params"] = total_params
    metrics["phase_2_trainable_summary"] = p2_summary
    metrics["phase_3_trainable_summary"] = p3_summary

    # Sample counters MUST come from OOF CSV, not from evaluate_model mini-batch
    total_samples = len(oof_df)
    correct_samples = int(oof_df["correct"].sum())
    metrics["total_samples"] = total_samples
    metrics["correct_samples"] = correct_samples
    metrics["accuracy"] = float(correct_samples / total_samples)

    metrics_path = paths["metrics_json"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in metrics.items()}, fp, indent=2)

    # ── History ────────────────────────────────────────────────────────────────
    combined_history: dict[str, list] = {}
    for k in history_p1.history:
        combined_history[k] = (
            list(history_p1.history[k])
            + list(history_p2.history.get(k, []))
            + list(history_p3.history.get(k, []))
        )
    combined_history["phase_boundaries"] = [
        phase1_cfg["epochs"],
        phase1_cfg["epochs"] + len(history_p2.history.get("loss", [])),
    ]

    hist_dir = paths["history_json"].parent
    hist_dir.mkdir(parents=True, exist_ok=True)
    with open(paths["history_json"], "w", encoding="utf-8") as fp:
        json.dump(combined_history, fp, indent=2)
    pd.DataFrame({k: v for k, v in combined_history.items() if isinstance(v, list) and k != "phase_boundaries"}).to_csv(
        paths["history_csv"], index=False
    )

    # ── Completion marker (written LAST) ──────────────────────────────────────
    completion_marker = {
        "fold": fold,
        "complete": True,
        "timestamp": datetime.now().isoformat(),
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "accuracy": float(correct_samples / total_samples),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "best_checkpoint": str(best_model_path),
        "training_duration_seconds": round(total_duration, 2),
    }
    completion_path = paths["completion_json"]
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(completion_path, "w", encoding="utf-8") as fp:
        json.dump(completion_marker, fp, indent=2)

    logger.info(
        f"Fold {fold} complete. "
        f"Accuracy={correct_samples}/{total_samples}={correct_samples/total_samples:.4f}, "
        f"MacroF1={metrics.get('macro_f1', 0):.4f}, "
        f"Duration={total_duration:.1f}s"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run
# ─────────────────────────────────────────────────────────────────────────────

def run_dry_run(config: dict[str, Any], output_dir: Path) -> None:
    """Dry-run: validate config, dataset, and model without any model.fit().

    Args:
        config: Configuration dictionary.
        output_dir: Root output directory (for manifest verification).
    """
    logger.info("DRY RUN — no training will be executed.")

    # Validate manifest
    verify_dataset_manifest_and_folds(config)

    # Validate preprocessing values
    validate_preprocessing_values()

    # Build model with weights=None (no ImageNet download)
    head_cfg = config["model"]["classifier_head"]
    image_size = tuple(config["data"]["image_size"])
    num_classes = config["data"]["num_classes"]

    model = build_inception_resnet_v2(
        num_classes=num_classes,
        input_shape=image_size + (3,),
        weights=None,
        head_config=head_cfg,
    )

    # Verify output shape
    assert model.output_shape == (None, 22), (
        f"Expected output shape (None, 22), got {model.output_shape}"
    )
    logger.info(f"Model output shape verified: {model.output_shape}")

    # Verify trainability strategies (no fit)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(1e-3)

    freeze_inception_resnet_v2_backbone(model)
    validate_inception_resnet_v2_trainability(model, phase=1)
    logger.info("Phase 1 trainability: OK")

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    apply_partial_inception_resnet_v2_fine_tuning(
        model=model,
        trainable_fraction=config["training"]["phase_2"]["trainable_backbone_fraction"],
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )
    validate_inception_resnet_v2_trainability(model, phase=2)
    logger.info("Phase 2 trainability (partial 30%): OK")

    apply_full_inception_resnet_v2_fine_tuning(
        model=model,
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )
    validate_inception_resnet_v2_trainability(model, phase=3)
    logger.info("Phase 3 trainability (full): OK")

    logger.info("DRY RUN PASSED — all validations successful. No model.fit() was called.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entrypoint for training InceptionResNetV2 Exp A."""
    parser = argparse.ArgumentParser(
        description=(
            "Train InceptionResNetV2 Exp A (High Performance Histology) "
            "with three-phase fine-tuning."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/experiments/inception_resnet_v2_exp_a_high_performance.yaml)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional override for dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for models and reports",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[3, 0, 4],
        help="Folds to train in order (default: 3 0 4)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip folds that are already complete",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-training even if fold is complete",
    )
    parser.add_argument(
        "--backup-partial",
        action="store_true",
        help="Rename partial fold checkpoint dir with timestamp before retraining",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config/dataset/model with weights=None. "
            "No training (model.fit) will be executed."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config = load_yaml(args.config)
    if args.dataset_dir is not None:
        config["data"]["dataset_root"] = args.dataset_dir

    output_dir = Path(args.output_dir).resolve()

    if args.dry_run:
        run_dry_run(config, output_dir)
        return

    verify_dataset_manifest_and_folds(config)

    for fold in args.folds:
        if is_fold_complete(output_dir, fold) and not args.overwrite:
            if args.skip_completed:
                logger.info(f"Fold {fold}: already complete. Skipping (--skip-completed).")
                continue
            else:
                logger.info(
                    f"Fold {fold}: already complete. "
                    "Use --overwrite to re-train or --skip-completed to skip."
                )
                continue

        if is_fold_partial(output_dir, fold):
            logger.warning(f"Fold {fold}: partial outputs detected.")
            if args.backup_partial:
                backup_partial_fold(output_dir, fold)
                logger.info(f"Fold {fold}: partial outputs backed up with timestamp.")
            else:
                logger.info(
                    f"Fold {fold}: partial outputs left in place. "
                    "Use --backup-partial to rename them."
                )

        try:
            train_fold_inception_resnet_v2(config, fold, output_dir)
        except Exception as exc:
            logger.error(f"Fold {fold} FAILED with error: {exc}")
            raise

    generate_screening_summary(output_dir, screening_folds=args.folds)
    logger.info("Screening summary written.")


if __name__ == "__main__":
    main()
