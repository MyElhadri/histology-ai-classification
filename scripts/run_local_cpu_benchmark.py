"""Local CPU Benchmark Runner for DenseNet121 (Model D).

Measures training time for a single fold (Fold 0) on local CPU with reduced epochs
and estimates total training time for full folds (10 head / 20-40 fine-tuning epochs).
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Force CPU execution before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to sys.path so 'src' can be imported when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.seed import set_global_seed
from src.data.pipeline import create_dataset
from src.models.densenet121 import (
    build_densenet121,
    set_trainable_layers,
    apply_fine_tuning_strategy,
    validate_model_matches_config,
)
from src.training.train import _compute_fold_class_weights

# Ensure Windows stdout handles UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Callback to record wall-clock execution time of each epoch."""

    def __init__(self):
        super().__init__()
        self.epoch_durations: list[float] = []
        self._epoch_start_time: float = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.perf_counter() - self._epoch_start_time
        self.epoch_durations.append(round(duration, 4))


def validate_dataset(dataset_dir: str | Path) -> int:
    """Verify dataset directory existence and count image files.

    If directory is missing, prints required error and raises FileNotFoundError.
    """
    path = Path(dataset_dir)
    if not path.exists() or not path.is_dir():
        print(f"Dataset not found:\n{dataset_dir}\n\nDo not start training.", file=sys.stderr)
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = [f for f in path.rglob("*") if f.suffix.lower() in valid_extensions]
    return len(images)


def calculate_estimations(avg_head: float, avg_ft: float) -> dict[str, float]:
    """Calculate estimated training durations for full folds (10 head epochs)."""
    return {
        "estimated_full_fold_20_epochs": round(10 * avg_head + 20 * avg_ft, 2),
        "estimated_full_fold_30_epochs": round(10 * avg_head + 30 * avg_ft, 2),
        "estimated_full_fold_40_epochs": round(10 * avg_head + 40 * avg_ft, 2),
    }


def run_benchmark(config_path: str | Path = "configs/experiments/densenet121_local_cpu_benchmark.yaml") -> dict:
    """Execute the local CPU benchmark training fold 0 and compute timing estimates."""
    config = load_yaml(config_path)
    dataset_root = config["data"]["dataset_root"]

    # 1. Validate dataset existence and count images
    try:
        img_count = validate_dataset(dataset_root)
    except FileNotFoundError:
        sys.exit(1)

    tf_version = tf.__version__
    gpus = tf.config.list_physical_devices("GPU")
    fold = config.get("validation", {}).get("target_fold", 0)
    batch_size = config["training"]["batch_size"]
    head_epochs = config["training"]["head_epochs"]
    ft_epochs = config["training"]["fine_tuning_epochs"]

    # 2. Display exact benchmark banner
    banner = (
        f"LOCAL CPU BENCHMARK\n"
        f"TensorFlow: {tf_version}\n"
        f"GPU detected: {gpus}\n"
        f"Device: CPU\n"
        f"Fold: {fold}\n"
        f"Batch size: {batch_size}\n"
        f"Head epochs: {head_epochs}\n"
        f"Fine-tuning epochs: {ft_epochs}\n"
        f"Dataset images: {img_count}"
    )
    print(banner)

    seed = config["project"]["seed"]
    set_global_seed(seed)

    # 3. Prepare datasets
    train_dataset = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=True,
        batch_size=batch_size,
        image_size=tuple(config["data"]["image_size"]),
        num_classes=config["data"]["num_classes"],
        augmentation_config=config.get("augmentation"),
    )

    val_dataset = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=False,
        batch_size=batch_size,
        image_size=tuple(config["data"]["image_size"]),
        num_classes=config["data"]["num_classes"],
    )

    # 4. Compute class weights
    class_weight_dict, _ = _compute_fold_class_weights(config, fold)

    # 5. Build model
    head_config = config.get("model", {}).get("classifier_head")
    model = build_densenet121(
        num_classes=config["data"]["num_classes"],
        input_shape=tuple(config["data"]["image_size"]) + (3,),
        weights=config["model"]["weights"],
        dropout_rate=config["model"]["dropout_rate"],
        head_config=head_config,
    )
    validate_model_matches_config(model, config)

    overall_start_time = time.perf_counter()

    # --- Phase 1: Train head only ---
    logger.info("Starting Benchmark Phase 1: Classifier Head Training (%d epochs)", head_epochs)
    set_trainable_layers(model, trainable=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["head_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    head_timer = EpochTimingCallback()
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=head_epochs,
        class_weight=class_weight_dict,
        callbacks=[head_timer],
    )

    # --- Phase 2: Fine-tuning ---
    logger.info("Starting Benchmark Phase 2: Full Fine-Tuning (%d epochs)", ft_epochs)
    ft_config = config.get("fine_tuning", {})
    apply_fine_tuning_strategy(
        model,
        strategy=ft_config.get("strategy", "full"),
        trainable_layer_prefixes=ft_config.get("trainable_layer_prefixes", ["conv5_"]),
        keep_batch_normalization_frozen=ft_config.get("keep_batch_normalization_frozen", True),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["fine_tuning_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ft_timer = EpochTimingCallback()
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=ft_epochs,
        class_weight=class_weight_dict,
        callbacks=[ft_timer],
    )

    total_benchmark_seconds = round(time.perf_counter() - overall_start_time, 4)

    head_epoch_durations = head_timer.epoch_durations
    ft_epoch_durations = ft_timer.epoch_durations
    all_epoch_durations = head_epoch_durations + ft_epoch_durations

    avg_head = round(sum(head_epoch_durations) / len(head_epoch_durations), 4) if head_epoch_durations else 0.0
    avg_ft = round(sum(ft_epoch_durations) / len(ft_epoch_durations), 4) if ft_epoch_durations else 0.0

    estimations = calculate_estimations(avg_head, avg_ft)

    summary = {
        "cpu_only": True,
        "tensorflow_version": tf_version,
        "batch_size": batch_size,
        "fold": fold,
        "head_epochs_executed": head_epochs,
        "fine_tuning_epochs_executed": ft_epochs,
        "epoch_durations": all_epoch_durations,
        "average_head_epoch_seconds": avg_head,
        "average_finetuning_epoch_seconds": avg_ft,
        "total_benchmark_seconds": total_benchmark_seconds,

        **estimations,
    }

    # 6. Save results to results/local_cpu_benchmark/benchmark_summary.json
    results_dir = Path("results/local_cpu_benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "benchmark_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Benchmark complete. Summary saved to %s", summary_path)
    return summary


if __name__ == "__main__":
    run_benchmark()
