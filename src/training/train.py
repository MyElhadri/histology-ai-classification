"""Training orchestrator.

Implements the 5-fold cross-validation training loop for DenseNet121.
Includes the two-phase training strategy: head training followed by fine-tuning.

Supports optional class-weight balancing controlled by:
    training.use_class_weights   (bool)
    training.class_weight_method (str, currently only "balanced")
"""

import json
import logging
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset
from src.evaluation.evaluate import evaluate_model
from src.models.densenet121 import build_densenet121, set_trainable_layers
from src.training.class_weights import compute_balanced_class_weights
from src.utils.config import load_yaml
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class weight helpers
# ---------------------------------------------------------------------------

def _load_class_mapping(config: dict) -> dict[int, str]:
    """Load the class_id → class_name mapping from class_mapping.json.

    Falls back to extracting the mapping from the folds CSV if the
    JSON file is not found.
    """
    mapping_path = Path("data/manifests/class_mapping.json")
    if mapping_path.is_file():
        with open(mapping_path, "r", encoding="utf-8") as f:
            # class_mapping.json stores {class_name: class_id}
            name_to_id = json.load(f)
        return {int(v): k for k, v in name_to_id.items()}

    # Fallback: derive from the folds CSV
    df = pd.read_csv(config["data"]["folds_path"])
    mapping = df.drop_duplicates("class_id").set_index("class_id")["class_name"]
    return mapping.to_dict()


def _compute_fold_class_weights(
    config: dict,
    fold: int,
) -> tuple[dict[int, float] | None, dict | None]:
    """Compute class weights for a fold's training set.

    Returns:
        (class_weight_dict, info_dict)
        If class weights are disabled, returns (None, None).
    """
    use_weights = config.get("training", {}).get("use_class_weights", False)

    if not use_weights:
        logger.info("Class weights DISABLED — training without balancing.")
        return None, None

    method = config.get("training", {}).get("class_weight_method", "balanced")
    if method != "balanced":
        raise ValueError(
            f"Unsupported class_weight_method: '{method}'. "
            f"Only 'balanced' is currently supported."
        )

    num_classes = config["data"]["num_classes"]
    folds_path = config["data"]["folds_path"]

    # Read folds CSV and select training rows (fold != current fold)
    df = pd.read_csv(folds_path)
    train_df = df[df["fold"] != fold]

    train_labels = train_df["class_id"].tolist()

    # Compute weights
    weights = compute_balanced_class_weights(train_labels, num_classes)

    # Build readable info dict
    id_to_name = _load_class_mapping(config)

    train_class_dist = (
        train_df.groupby("class_name").size().sort_index().to_dict()
    )

    # Log distribution
    logger.info(f"  Fold {fold} — Train class distribution ({len(train_labels)} images):")
    for class_name in sorted(train_class_dist.keys()):
        count = train_class_dist[class_name]
        class_id = next(cid for cid, cn in id_to_name.items() if cn == class_name)
        w = weights[class_id]
        logger.info(f"    {class_name:20s} (id={class_id:2d}): {count:4d} images, weight={w:.4f}")

    info = {
        "fold": fold,
        "method": method,
        "enabled": True,
        "train_images": len(train_labels),
        "validation_images": int((df["fold"] == fold).sum()),
        "train_class_distribution": train_class_dist,
        "class_weights": {
            id_to_name.get(cid, str(cid)): {
                "class_id": cid,
                "weight": round(w, 6),
            }
            for cid, w in sorted(weights.items())
        },
    }

    return weights, info


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold(config: dict, fold: int, output_dir: Path) -> dict:
    """Train a model for a specific fold and evaluate it.

    Args:
        config: Configuration dictionary.
        fold: Fold index to use as validation.
        output_dir: Root directory for outputs.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"=== Starting Training for Fold {fold} ===")

    # 0. Compute class weights from the TRAINING set only
    class_weight_dict, class_weight_info = _compute_fold_class_weights(config, fold)

    # 1. Create datasets
    train_dataset = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=True,
        batch_size=config["training"]["batch_size"],
        image_size=tuple(config["data"]["image_size"]),
        num_classes=config["data"]["num_classes"]
    )

    val_dataset = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=False,
        batch_size=config["training"]["batch_size"],
        image_size=tuple(config["data"]["image_size"]),
        num_classes=config["data"]["num_classes"]
    )

    # 2. Build model
    model = build_densenet121(
        num_classes=config["data"]["num_classes"],
        input_shape=tuple(config["data"]["image_size"]) + (3,),
        weights=config["model"]["weights"],
        dropout_rate=config["model"]["dropout_rate"]
    )

    fold_checkpoint_dir = output_dir / "models" / "densenet121" / "checkpoints" / f"fold_{fold}"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3. Phase 1: Train head only
    logger.info("Phase 1: Training head only (base frozen)")
    set_trainable_layers(model, trainable=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["head_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["training"]["head_epochs"],
        class_weight=class_weight_dict,
    )

    # 4. Phase 2: Fine-tuning
    logger.info("Phase 2: Fine-tuning full model")
    set_trainable_layers(model, trainable=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["fine_tuning_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(fold_checkpoint_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            monitor="val_loss",
            restore_best_weights=True
        )
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["training"]["fine_tuning_epochs"],
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    # 5. Evaluate — NO class_weight here
    logger.info(f"Evaluating fold {fold}")
    metrics = evaluate_model(model, val_dataset)

    # 6. Save metrics
    metrics_dir = output_dir / "reports" / "densenet121" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"fold_{fold}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 7. Save class weight info
    if class_weight_info is not None:
        with open(metrics_dir / f"fold_{fold}_class_weights.json", "w") as f:
            json.dump(class_weight_info, f, indent=2, ensure_ascii=False)
        logger.info(f"  Class weight info saved to {metrics_dir / f'fold_{fold}_class_weights.json'}")

    return metrics


def run_cross_validation(config_path: Path | str, output_dir: Path | str) -> None:
    """Run the full 5-fold cross validation.

    Args:
        config_path: Path to densenet121.yaml.
        output_dir: Root directory for saving models and reports.
    """
    config = load_yaml(config_path)
    set_global_seed(config["project"]["seed"])
    output_dir = Path(output_dir).resolve()

    all_metrics = {}
    num_folds = config["validation"]["num_folds"]

    for fold in range(num_folds):
        metrics = train_fold(config, fold, output_dir)
        all_metrics[f"fold_{fold}"] = metrics

    # Compute summary
    avg_accuracy = sum(m["accuracy"] for m in all_metrics.values()) / num_folds
    avg_macro_f1 = sum(m["macro_f1"] for m in all_metrics.values()) / num_folds

    summary = {
        "folds": all_metrics,
        "average": {
            "accuracy": avg_accuracy,
            "macro_f1": avg_macro_f1
        }
    }

    metrics_dir = output_dir / "reports" / "densenet121" / "metrics"
    with open(metrics_dir / "cross_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Cross validation complete. Average Macro F1: {avg_macro_f1:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_cross_validation(args.config, args.output_dir)
