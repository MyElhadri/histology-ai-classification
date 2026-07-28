"""Training orchestrator for ResNet50V2 on GTEx dataset.

Usage:
    python scripts/train_resnet50v2_gtex.py --config configs/experiments/resnet50v2_gtex_11_exp_a.yaml --dataset-dir /path/to/GTEx --output-dir /path/to/output
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

# Ensure src is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import yaml

from src.data.gtex_integrity import audit_gtex_dataset
from src.data.gtex_pipeline import create_gtex_dataset, validate_batch
from src.models.resnet50v2 import (
    build_resnet50v2_model,
    apply_partial_resnet50v2_fine_tuning,
    apply_full_resnet50v2_fine_tuning,
    freeze_resnet50v2_backbone,
    validate_resnet50v2_trainability,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_callbacks(
    output_dir: Path, phase: int, patience_es: int, patience_lr: int, factor_lr: float
) -> list[tf.keras.callbacks.Callback]:
    phase_dir = output_dir / "checkpoints"
    phase_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = phase_dir / f"phase_{phase}_best.keras"
    csv_path = history_dir / f"phase_{phase}_training.csv"
    
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=factor_lr,
            patience=patience_lr,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path)),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def run_phase(
    model: tf.keras.Model,
    phase: int,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: dict,
    output_dir: Path,
    smoke_test: bool = False
) -> None:
    phase_config = config["training"][f"phase_{phase}"]
    epochs = 1 if smoke_test else phase_config["epochs"]
    lr = float(phase_config["learning_rate"])
    
    logger.info(f"=== Starting Phase {phase}: {phase_config['name']} ===")
    
    if phase == 1:
        freeze_resnet50v2_backbone(model)
    elif phase == 2:
        apply_partial_resnet50v2_fine_tuning(model, fraction=phase_config["trainable_backbone_fraction"])
    elif phase == 3:
        apply_full_resnet50v2_fine_tuning(model)
        
    validate_resnet50v2_trainability(model, phase)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=config["training"].get("loss", "sparse_categorical_crossentropy"),
        metrics=["accuracy"]
    )
    
    callbacks = setup_callbacks(
        output_dir,
        phase,
        patience_es=config["training"].get("early_stopping_patience", 5),
        patience_lr=config["training"].get("reduce_lr_patience", 2),
        factor_lr=config["training"].get("reduce_lr_factor", 0.5)
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Mark completion
    completion_dir = output_dir / "completion"
    completion_dir.mkdir(parents=True, exist_ok=True)
    with open(completion_dir / f"phase_{phase}.done", "w") as f:
        f.write("DONE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--run-test-evaluation", action="store_true")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        args.output_dir = args.output_dir / "smoke_test"
        
    if args.output_dir.exists() and not args.overwrite and not args.resume and not args.dry_run:
        raise ValueError(f"Output dir {args.output_dir} exists. Use --overwrite or --resume.")
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(args.config)
    
    if config["training"].get("mixed_precision", False) and not args.dry_run:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        
    # Dataset Audit
    if not args.dry_run:
        logger.info("Auditing GTEx dataset...")
        audit_gtex_dataset(args.dataset_dir, args.output_dir / "dataset_integrity.json")
    
    # Create datasets
    batch_size = config["training"]["batch_size"]
    img_size = tuple(config["dataset"]["input_size"])
    
    train_ds = create_gtex_dataset(
        args.dataset_dir, "train", is_training=True, 
        batch_size=batch_size, image_size=img_size, 
        augmentation_config=config.get("augmentation")
    )
    val_ds = create_gtex_dataset(
        args.dataset_dir, "validation", is_training=False, 
        batch_size=batch_size, image_size=img_size
    )
    
    if args.smoke_test:
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(2)
        
    if args.dry_run:
        validate_batch(train_ds)
        
    # Model Setup
    weights = None if args.dry_run else config["model"]["weights"]
    model = build_resnet50v2_model(config, weights=weights)
    
    # Training Phases
    for phase in [1, 2, 3]:
        completion_marker = args.output_dir / "completion" / f"phase_{phase}.done"
        if args.resume and completion_marker.exists():
            logger.info(f"Phase {phase} already completed. Resuming from checkpoint...")
            ckpt_path = args.output_dir / "checkpoints" / f"phase_{phase}_best.keras"
            model.load_weights(str(ckpt_path))
            continue
            
        if args.dry_run:
            logger.info(f"[DRY-RUN] Validating phase {phase} logic...")
            if phase == 1: freeze_resnet50v2_backbone(model)
            elif phase == 2: apply_partial_resnet50v2_fine_tuning(model, 0.3)
            elif phase == 3: apply_full_resnet50v2_fine_tuning(model)
            validate_resnet50v2_trainability(model, phase)
            continue
            
        run_phase(model, phase, train_ds, val_ds, config, args.output_dir, args.smoke_test)
        
    # Finalize
    if not args.dry_run:
        best_ckpt = args.output_dir / "checkpoints" / "phase_3_best.keras"
        final_dest = args.output_dir / "checkpoints" / "best_model.keras"
        if best_ckpt.exists():
            import shutil
            shutil.copy(str(best_ckpt), str(final_dest))
            logger.info("Saved final best model.")

if __name__ == "__main__":
    main()
