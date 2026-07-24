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

from src.data.pipeline import create_dataset, create_square_root_dataset
from src.evaluation.evaluate import evaluate_model
from src.models.densenet121 import (
    apply_fine_tuning_strategy,
    build_densenet121,
    set_trainable_layers,
    validate_fine_tuning_strategy,
    validate_model_matches_config,
)
from src.training.class_weights import compute_balanced_class_weights
from src.utils.config import load_yaml
from src.utils.seed import set_global_seed
from src.utils.serialization import convert_numpy_to_python

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

def _train_fold_e1(
    config: dict,
    fold: int,
    output_dir: Path,
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    fold_checkpoint_dir: Path,
    actual_head_type: str,
    aug_policy: str,
    req_head_type: str,
    req_ft_strategy: str,
    total_params: int,
    seed: int,
) -> dict:
    """Specialized training path for E1 experiments (3-phase pipeline)."""
    import numpy as np
    num_classes = config["data"]["num_classes"]
    manifest_path = config["data"]["folds_path"]

    df = pd.read_csv(manifest_path)
    train_df = df[df["fold"] != fold]
    train_counts = train_df["class_id"].value_counts().to_dict()
    all_train_counts = {c: int(train_counts.get(c, 0)) for c in range(num_classes)}
    Q1 = float(np.percentile(list(all_train_counts.values()), 25))
    minority_class_ids = [c for c, count in all_train_counts.items() if count <= Q1]
    id_to_name = _load_class_mapping(config)
    minority_class_names = [id_to_name.get(c, str(c)) for c in minority_class_ids]
    minority_support = {c: all_train_counts[c] for c in minority_class_ids}

    label_smoothing = config.get("loss", {}).get("label_smoothing", 0.05)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    ce_hard_metric = tf.keras.metrics.CategoricalCrossentropy(name="ce_hard", label_smoothing=0.0)

    # --- Phase 1: Head Training ---
    set_trainable_layers(model, trainable=False)
    head_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    logger.info("Phase: head_training")
    logger.info(f"Classifier head: {actual_head_type}")
    logger.info(f"Augmentation train: {aug_policy}")
    logger.info("Augmentation validation: disabled")
    logger.info("Sampling strategy: natural")
    logger.info("Class weights enabled: false")
    logger.info(f"Label smoothing: {label_smoothing}")
    logger.info("Backbone trainable layers: 0")
    logger.info(f"Trainable parameters: {head_trainable_params}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["head_learning_rate"]),
        loss=loss_fn,
        metrics=["accuracy", ce_hard_metric],
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["training"]["head_epochs"],
        class_weight=None,
    )

    # --- Phase 2: Representation Fine-Tuning ---
    logger.info("Phase: representation_fine_tuning")
    ft_config = config.get("fine_tuning", {})
    prefixes = ft_config.get("trainable_layer_prefixes", ["conv5_"])
    keep_bn_frozen = ft_config.get("keep_batch_normalization_frozen", True)

    apply_fine_tuning_strategy(
        model,
        strategy=req_ft_strategy,
        trainable_layer_prefixes=prefixes,
        keep_batch_normalization_frozen=keep_bn_frozen,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["fine_tuning_learning_rate"]),
        loss=loss_fn,
        metrics=["accuracy", ce_hard_metric],
    )

    validate_fine_tuning_strategy(model, config)

    trainable_backbone_layers = [
        l for l in model.layers
        if not l.name.startswith(("global_average_pooling", "classifier_", "predictions"))
        and l.trainable
    ]
    trainable_backbone_params = sum(
        tf.keras.backend.count_params(w)
        for l in trainable_backbone_layers for w in l.trainable_weights
    )
    total_trainable_params_ft = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    logger.info(f"Fine-tuning strategy: {req_ft_strategy}")
    logger.info("Sampling strategy: natural")
    logger.info("Class weights enabled: false")
    logger.info(f"Label smoothing: {label_smoothing}")
    logger.info(f"Keep backbone BatchNormalization frozen: {keep_bn_frozen}")
    logger.info(f"Trainable backbone layers: {len(trainable_backbone_layers)}")
    logger.info(f"Trainable backbone parameters: {trainable_backbone_params}")
    logger.info(f"Total trainable parameters: {total_trainable_params_ft}")
    logger.info("Checkpoint monitor: val_ce_hard")

    rep_ckpt_path = fold_checkpoint_dir / "best_representation.weights.h5"
    phase2_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(rep_ckpt_path),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_ce_hard",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ce_hard",
            mode="min",
            patience=5,
            min_delta=0.002,
            restore_best_weights=True,
            start_from_epoch=5,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_ce_hard",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["training"]["fine_tuning_epochs"],
        callbacks=phase2_callbacks,
        class_weight=None,
    )

    if rep_ckpt_path.is_file():
        model.load_weights(str(rep_ckpt_path))

    pre_crt_metrics = evaluate_model(
        model,
        val_dataset,
        minority_class_ids=minority_class_ids,
        num_classes=num_classes,
    )
    logger.info(
        f"Pre-cRT Metrics — Accuracy: {pre_crt_metrics['accuracy']:.4f}, "
        f"Macro F1: {pre_crt_metrics['macro_f1']:.4f}, "
        f"val_ce_hard: {pre_crt_metrics['val_ce_hard']:.4f}"
    )

    # --- Phase 3: Classifier Retraining (cRT) ---
    logger.info("Phase: classifier_retraining")
    crt_config = config.get("classifier_retraining", {})

    for layer in model.layers:
        if layer.name == "predictions":
            layer.trainable = True
        else:
            layer.trainable = False

    pred_layer = next(l for l in model.layers if l.name == "predictions")
    if hasattr(pred_layer, "kernel_initializer") and pred_layer.kernel is not None:
        new_k = pred_layer.kernel_initializer(shape=pred_layer.kernel.shape, dtype=pred_layer.kernel.dtype)
        pred_layer.kernel.assign(new_k)
    if hasattr(pred_layer, "bias_initializer") and pred_layer.bias is not None:
        new_b = pred_layer.bias_initializer(shape=pred_layer.bias.shape, dtype=pred_layer.bias.dtype)
        pred_layer.bias.assign(new_b)

    crt_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    if crt_trainable_params != 2838:
        raise ValueError(f"Expected 2838 trainable parameters during cRT, found {crt_trainable_params}")

    crt_sampler_cfg = crt_config.get("sampler", {})
    freq_power = crt_sampler_cfg.get("frequency_power", -0.5)
    crt_train_ds, N_train, class_probs_dict, orig_counts_dict = create_square_root_dataset(
        manifest_path=manifest_path,
        fold=fold,
        batch_size=config["training"]["batch_size"],
        image_size=tuple(config["data"]["image_size"]),
        num_classes=num_classes,
        augmentation_config=config.get("augmentation"),
        seed=seed,
        frequency_power=freq_power,
    )

    crt_lr = crt_config.get("optimizer", {}).get("learning_rate", 0.0003)
    crt_loss_smoothing = crt_config.get("loss", {}).get("label_smoothing", 0.05)
    crt_loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=crt_loss_smoothing)
    crt_ce_metric = tf.keras.metrics.CategoricalCrossentropy(name="ce_hard", label_smoothing=0.0)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=crt_lr),
        loss=crt_loss_fn,
        metrics=["accuracy", crt_ce_metric],
    )

    crt_ckpt_path = fold_checkpoint_dir / "best_crt.weights.h5"
    crt_callbacks_cfg = crt_config.get("callbacks", {})
    phase3_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(crt_ckpt_path),
            save_best_only=True,
            save_weights_only=True,
            monitor=crt_callbacks_cfg.get("monitor", "val_ce_hard"),
            mode=crt_callbacks_cfg.get("mode", "min"),
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=crt_callbacks_cfg.get("monitor", "val_ce_hard"),
            mode=crt_callbacks_cfg.get("mode", "min"),
            patience=crt_callbacks_cfg.get("early_stopping_patience", 3),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=crt_callbacks_cfg.get("monitor", "val_ce_hard"),
            mode=crt_callbacks_cfg.get("mode", "min"),
            factor=crt_callbacks_cfg.get("reduce_lr_factor", 0.2),
            patience=crt_callbacks_cfg.get("reduce_lr_patience", 2),
            min_lr=crt_callbacks_cfg.get("min_learning_rate", 1e-7),
        ),
    ]

    logger.info("Phase: classifier_retraining")
    logger.info(f"Sampling strategy: {crt_sampler_cfg.get('strategy', 'square_root')}")
    logger.info(f"Sampling frequency power: {freq_power}")
    logger.info(f"Original train epoch size: {N_train}")
    logger.info(f"cRT epoch size: {N_train}")
    logger.info("Class weights enabled: false")
    logger.info("Predictions reinitialized: true")
    logger.info("Trainable layers: ['predictions']")
    logger.info(f"Trainable parameters: {crt_trainable_params}")
    logger.info("Validation sampling: natural")
    logger.info("Validation augmentation: disabled")

    model.fit(
        crt_train_ds,
        validation_data=val_dataset,
        epochs=crt_config.get("epochs", 12),
        callbacks=phase3_callbacks,
        class_weight=None,
    )

    if crt_ckpt_path.is_file():
        model.load_weights(str(crt_ckpt_path))

    post_crt_metrics = evaluate_model(
        model,
        val_dataset,
        minority_class_ids=minority_class_ids,
        num_classes=num_classes,
    )

    crt_delta = {
        "accuracy": float(post_crt_metrics["accuracy"] - pre_crt_metrics["accuracy"]),
        "macro_f1": float(post_crt_metrics["macro_f1"] - pre_crt_metrics["macro_f1"]),
        "weighted_f1": float(post_crt_metrics["weighted_f1"] - pre_crt_metrics["weighted_f1"]),
        "minority_macro_f1": float(post_crt_metrics["minority_macro_f1"] - pre_crt_metrics["minority_macro_f1"]),
        "minority_macro_recall": float(post_crt_metrics["minority_macro_recall"] - pre_crt_metrics["minority_macro_recall"]),
        "zero_f1_class_count": int(post_crt_metrics["zero_f1_class_count"] - pre_crt_metrics["zero_f1_class_count"]),
    }

    metrics_dir = output_dir / "reports" / "densenet121" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / f"pre_crt_fold_{fold}.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy_to_python(pre_crt_metrics), f, indent=2)

    with open(metrics_dir / f"post_crt_fold_{fold}.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy_to_python(post_crt_metrics), f, indent=2)

    fold_summary = {
        "fold": fold,
        "pre_crt_metrics": convert_numpy_to_python(pre_crt_metrics),
        "post_crt_metrics": convert_numpy_to_python(post_crt_metrics),
        "crt_delta": convert_numpy_to_python(crt_delta),
        "minority_class_ids": minority_class_ids,
        "minority_class_names": minority_class_names,
        "minority_threshold": Q1,
        "minority_support_by_class": minority_support,
        "class_probabilities": class_probs_dict,
        "original_counts": orig_counts_dict,
    }

    with open(metrics_dir / f"fold_{fold}.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy_to_python(post_crt_metrics), f, indent=2)

    return post_crt_metrics

def train_fold(
    config: dict,
    fold: int,
    output_dir: Path | str,
) -> dict:
    """Train a single fold.

    Args:
        config: Configuration dictionary.
        fold: Fold index (0-4).
        output_dir: Root output directory.

    Returns:
        Evaluation metrics dictionary.
    """
    output_dir = Path(output_dir).resolve()
    seed = config.get("_seed_override") if config.get("_seed_override") is not None else config["project"]["seed"]
    set_global_seed(seed)

    logger.info(f"\n==========================================")
    logger.info(f"Starting Fold {fold} (seed={seed})")
    logger.info(f"==========================================")

    # 1. Compute class weights if enabled
    class_weight_dict, class_weight_info = _compute_fold_class_weights(config, fold)

    # 2. Create datasets
    train_dataset = create_dataset(
        manifest_path=config["data"]["folds_path"],
        dataset_root=config["data"]["dataset_root"],
        fold=fold,
        is_training=True,
        batch_size=config["training"]["batch_size"],
        image_size=tuple(config["data"]["image_size"]),
        num_classes=config["data"]["num_classes"],
        augmentation_config=config.get("augmentation")
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

    # 3. Build model
    head_config = config.get("model", {}).get("classifier_head") or config.get("classifier_head", {"type": "baseline"})
    model = build_densenet121(
        num_classes=config["data"]["num_classes"],
        input_shape=tuple(config["data"]["image_size"]) + (3,),
        weights=config["model"]["weights"],
        dropout_rate=config["model"]["dropout_rate"],
        head_config=head_config
    )
    
    # Validate model architecture matches config BEFORE training
    validate_model_matches_config(model, config)
    
    # Determine actual head type by inspecting layer names
    layer_names = [l.name for l in model.layers]
    if "classifier_dense_512" in layer_names and "classifier_dense_128" in layer_names:
        actual_head_type = "article_inspired"
    elif "predictions" in layer_names and "classifier_dense_512" not in layer_names:
        actual_head_type = "baseline"
    else:
        actual_head_type = "unknown"

    req_head_type = head_config.get("type", "baseline")
    aug_policy = config.get("augmentation", {}).get("policy", "baseline") if config.get("augmentation", {}).get("enabled", False) else "baseline"
    total_params = model.count_params()
    classifier_layers = [l.name for l in model.layers if l.name.startswith(("global_average_pooling", "classifier_", "predictions"))]

    ft_config = config.get("fine_tuning", {})
    req_ft_strategy = ft_config.get("strategy", "full")

    fold_checkpoint_dir = output_dir / "models" / "densenet121" / "checkpoints" / f"fold_{fold}"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    is_e1 = (
        config.get("experiment_name") == "densenet121_exp_e1_regularized_crt"
        or config.get("classifier_retraining", {}).get("enabled", False)
    )

    if is_e1:
        return _train_fold_e1(
            config=config,
            fold=fold,
            output_dir=output_dir,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            fold_checkpoint_dir=fold_checkpoint_dir,
            actual_head_type=actual_head_type,
            aug_policy=aug_policy,
            req_head_type=req_head_type,
            req_ft_strategy=req_ft_strategy,
            total_params=total_params,
            seed=seed,
        )

    logger.info(f"Config path: {config.get('_config_path', 'N/A')}")
    logger.info(f"Requested head type: {req_head_type}")
    logger.info(f"Actual head type: {actual_head_type}")
    logger.info(f"Augmentation policy: {aug_policy}")
    logger.info(f"Fine-tuning strategy requested: {req_ft_strategy}")
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Classifier layers: {classifier_layers}")

    fold_checkpoint_dir = output_dir / "models" / "densenet121" / "checkpoints" / f"fold_{fold}"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3. Phase 1: Train head only
    logger.info("Phase 1: Training head only (base frozen)")
    set_trainable_layers(model, trainable=False)
    
    backbone_layers = [l for l in model.layers if not l.name.startswith(("global_average_pooling", "classifier_", "predictions"))]
    trainable_backbone_head_phase = [l for l in backbone_layers if l.trainable]
    logger.info(f"Backbone trainable layers during head phase: {len(trainable_backbone_head_phase)}")

    head_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    logger.info(f"Trainable parameters (head phase): {head_trainable_params}")

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
    logger.info("Phase 2: Fine-tuning")
    prefixes = ft_config.get("trainable_layer_prefixes", ["conv5_"])
    keep_bn_frozen = ft_config.get("keep_batch_normalization_frozen", True)

    apply_fine_tuning_strategy(
        model,
        strategy=req_ft_strategy,
        trainable_layer_prefixes=prefixes,
        keep_batch_normalization_frozen=keep_bn_frozen
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["fine_tuning_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    validate_fine_tuning_strategy(model, config)

    earlier_trainable = [
        l.name for l in model.layers
        if l.name.startswith(("conv1_", "conv2_", "conv3_", "conv4_"))
        and not isinstance(l, tf.keras.layers.BatchNormalization)
        and l.trainable
    ]
    actual_ft_strategy = "full" if earlier_trainable else "last_dense_block"

    trainable_backbone_layers = [
        l for l in model.layers
        if not l.name.startswith(("global_average_pooling", "classifier_", "predictions"))
        and l.trainable
    ]
    frozen_backbone_layers = [
        l for l in model.layers
        if not l.name.startswith(("global_average_pooling", "classifier_", "predictions"))
        and not l.trainable
    ]

    trainable_backbone_params = sum(
        tf.keras.backend.count_params(w)
        for l in trainable_backbone_layers for w in l.trainable_weights
    )
    total_trainable_params_ft = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    first_trainable = trainable_backbone_layers[0].name if trainable_backbone_layers else "None"
    last_trainable = trainable_backbone_layers[-1].name if trainable_backbone_layers else "None"

    logger.info(f"Fine-tuning strategy actual: {actual_ft_strategy}")
    logger.info(f"Keep BatchNormalization frozen: {keep_bn_frozen}")
    logger.info(f"Trainable backbone layer count: {len(trainable_backbone_layers)}")
    logger.info(f"Frozen backbone layer count: {len(frozen_backbone_layers)}")
    logger.info(f"Trainable backbone parameters: {trainable_backbone_params}")
    logger.info(f"Total trainable parameters: {total_trainable_params_ft}")
    logger.info(f"First trainable backbone layer: {first_trainable}")
    logger.info(f"Last trainable backbone layer: {last_trainable}")
    logger.info(f"Trainable backbone layer names: {[l.name for l in trainable_backbone_layers]}")

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
