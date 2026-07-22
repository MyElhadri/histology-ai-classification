"""Training orchestrator.

Implements the 5-fold cross-validation training loop for DenseNet121.
Includes the two-phase training strategy: head training followed by fine-tuning.
"""

import json
import logging
from pathlib import Path

import tensorflow as tf

from src.data.pipeline import create_dataset
from src.evaluation.evaluate import evaluate_model
from src.models.densenet121 import build_densenet121, set_trainable_layers
from src.utils.config import load_yaml
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


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
        epochs=config["training"]["head_epochs"]
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
        callbacks=callbacks
    )
    
    # 5. Evaluate
    logger.info(f"Evaluating fold {fold}")
    metrics = evaluate_model(model, val_dataset)
    
    # Save metrics
    metrics_dir = output_dir / "reports" / "densenet121" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"fold_{fold}.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
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
