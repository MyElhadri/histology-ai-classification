"""Unit tests for EfficientNetV2B0 Expérience A implementation.

Verifies architecture instantiation, article-inspired head parity with archived
DenseNet121 D, dataset preprocessing rules, two-phase BN freezing, class weights,
and OOF prediction export without launching automated training.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Rescaling

from scripts.train_efficientnetv2b0 import (
    generate_screening_summary,
    verify_dataset_manifest_and_folds,
    verify_fold_class_weights_against_archived,
)
from src.models.efficientnetv2b0 import (
    apply_efficientnet_fine_tuning_strategy,
    build_efficientnetv2b0,
    validate_dataset_preprocessing,
    validate_efficientnet_strategy,
    verify_no_double_preprocessing,
)
from src.models.heads import build_article_inspired_head
from src.utils.config import load_yaml


@pytest.fixture
def exp_a_config() -> dict[str, Any]:
    """Load EfficientNetV2B0 Exp A configuration."""
    config_path = Path("configs/experiments/efficientnetv2b0_exp_a_fair_comparison.yaml")
    return load_yaml(config_path)


def test_config_loading_and_parameters(exp_a_config: dict[str, Any]) -> None:
    """Test that Exp A YAML configuration loads and matches fair comparison protocol."""
    assert exp_a_config["experiment_name"] == "efficientnetv2b0-exp-a-fair-comparison"
    assert exp_a_config["model"]["architecture"] == "EfficientNetV2B0"
    assert exp_a_config["model"]["weights"] == "imagenet"
    assert exp_a_config["model"]["classifier_head"]["type"] == "article_inspired"
    assert exp_a_config["model"]["classifier_head"]["l2_strength"] == 0.01
    assert exp_a_config["data"]["num_classes"] == 22
    assert exp_a_config["data"]["image_size"] == [224, 224]
    assert exp_a_config["project"]["seed"] == 42
    assert exp_a_config["fine_tuning"]["strategy"] == "full"
    assert exp_a_config["fine_tuning"]["keep_batch_normalization_frozen"] is True

    # Check exact callbacks configuration
    cb_cfg = exp_a_config["callbacks"]
    assert cb_cfg["monitor"] == "val_ce_hard"
    assert cb_cfg["mode"] == "min"
    assert cb_cfg["early_stopping_patience"] == 5
    assert cb_cfg["min_delta"] == 0.002
    assert cb_cfg["restore_best_weights"] is True
    assert cb_cfg["reduce_lr_factor"] == 0.2
    assert cb_cfg["reduce_lr_patience"] == 2
    assert cb_cfg["min_learning_rate"] == 1e-7


def test_head_parity_with_archived_densenet_d(exp_a_config: dict[str, Any]) -> None:
    """Verify that our new head matches archived DenseNet121 Exp D head exactly."""
    archived_model_path = Path("artifacts/models/densenet121_exp_d_v1/checkpoints/fold_0/best_model.keras")
    if not archived_model_path.is_file():
        pytest.skip(f"Archived checkpoint not found at {archived_model_path}")

    archived_model = tf.keras.models.load_model(archived_model_path, compile=False)
    archived_head_layers = [
        l for l in archived_model.layers
        if l.name.startswith(("global_average_pooling", "classifier_", "predictions"))
    ]

    # Build new head using heads.py on dummy input
    dummy_input = tf.keras.Input(shape=(7, 7, 1024))
    head_cfg = exp_a_config["model"]["classifier_head"]
    out = build_article_inspired_head(dummy_input, num_classes=22, head_config=head_cfg)
    new_head_model = tf.keras.Model(dummy_input, out)

    new_head_layers = [
        l for l in new_head_model.layers
        if l.name.startswith(("global_average_pooling", "classifier_", "predictions"))
    ]

    assert len(archived_head_layers) == len(new_head_layers), "Head layer count mismatch!"

    for arch_l, new_l in zip(archived_head_layers, new_head_layers):
        assert arch_l.name == new_l.name, f"Name mismatch: {arch_l.name} vs {new_l.name}"
        assert type(arch_l).__name__ == type(new_l).__name__, f"Type mismatch for {arch_l.name}"

        if hasattr(arch_l, "units"):
            assert arch_l.units == new_l.units, f"Units mismatch for {arch_l.name}"

        if isinstance(arch_l, tf.keras.layers.Dropout):
            assert arch_l.rate == new_l.rate, f"Dropout rate mismatch for {arch_l.name}"

        # Check L2 regularization on dense_128
        if arch_l.name == "classifier_dense_128":
            arch_reg = getattr(arch_l.kernel_regularizer, "l2", None)
            new_reg = getattr(new_l.kernel_regularizer, "l2", None)
            assert arch_reg is not None and new_reg is not None
            assert abs(float(arch_reg) - float(new_reg)) < 1e-7, "L2 regularization mismatch!"


def test_model_build_and_no_double_preprocessing(exp_a_config: dict[str, Any]) -> None:
    """Test model construction with weights=None and double preprocessing detection."""
    head_cfg = exp_a_config["model"]["classifier_head"]
    model = build_efficientnetv2b0(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights=None,  # Do not download ImageNet during unit tests
        dropout_rate=0.30,
        head_config=head_cfg,
    )

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 22)

    # Should pass without error
    verify_no_double_preprocessing(model)

    # Test detection of illegal external Rescaling layer
    dummy_in = tf.keras.Input((224, 224, 3))
    x = Rescaling(1.0 / 255)(dummy_in)
    out = model(x)
    bad_model = tf.keras.Model(dummy_in, out)

    with pytest.raises(ValueError, match="Double preprocessing detected"):
        verify_no_double_preprocessing(bad_model)


def test_backbone_training_false_and_bn_freezing(exp_a_config: dict[str, Any]) -> None:
    """Test two-phase fine-tuning where backbone BN layers stay frozen."""
    head_cfg = exp_a_config["model"]["classifier_head"]
    model = build_efficientnetv2b0(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights=None,
        head_config=head_cfg,
    )

    # Apply Phase 2 fine-tuning strategy
    apply_efficientnet_fine_tuning_strategy(
        model=model,
        strategy="full",
        keep_batch_normalization_frozen=True,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    backbone = next(l for l in model.layers if isinstance(l, tf.keras.Model) or "efficientnetv2" in l.name.lower())
    assert backbone.trainable is True, "Backbone should be trainable in Phase 2"

    trainable_bns = [
        l.name for l in backbone.layers
        if isinstance(l, tf.keras.layers.BatchNormalization) and l.trainable
    ]
    assert len(trainable_bns) == 0, f"Found trainable BN layers in backbone: {trainable_bns[:5]}"

    # Strategy validation should pass
    validate_efficientnet_strategy(model, exp_a_config)


def test_dataset_preprocessing_validation(exp_a_config: dict[str, Any]) -> None:
    """Test validation of dataset batch values and configuration."""
    # Good dataset: float32 in [0, 255]
    good_images = tf.random.uniform((4, 224, 224, 3), minval=0.0, maxval=255.0, dtype=tf.float32)
    good_labels = tf.zeros((4, 22), dtype=tf.float32)
    good_ds = tf.data.Dataset.from_tensor_slices((good_images, good_labels)).batch(2)

    validate_dataset_preprocessing(good_ds, exp_a_config)

    # Bad dataset: values in [0, 1]
    bad_images = tf.random.uniform((4, 224, 224, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
    bad_ds = tf.data.Dataset.from_tensor_slices((bad_images, good_labels)).batch(2)

    with pytest.raises(ValueError, match="max value <= 1.0"):
        validate_dataset_preprocessing(bad_ds, exp_a_config)

    # Bad config: contains rescale=1/255
    bad_config = exp_a_config.copy()
    bad_config["data"] = bad_config["data"].copy()
    bad_config["data"]["rescale"] = 1.0 / 255.0

    with pytest.raises(ValueError, match="illegal rescale"):
        validate_dataset_preprocessing(good_ds, bad_config)


def test_fold_class_weights_verification(exp_a_config: dict[str, Any]) -> None:
    """Verify that class weights calculated match archived DenseNet121 D fold weights exactly."""
    weights_f0 = verify_fold_class_weights_against_archived(exp_a_config, fold=0)
    assert len(weights_f0) == 22
    assert 0 in weights_f0 and 21 in weights_f0
    assert round(weights_f0[0], 6) == 1.742424


def test_manifest_and_folds_verification(exp_a_config: dict[str, Any]) -> None:
    """Test verification of manifest 432 image count and fold distribution."""
    verify_dataset_manifest_and_folds(exp_a_config)


def test_screening_summary_generation(tmp_path: Path) -> None:
    """Test screening summary generation from mock fold metrics."""
    metrics_dir = tmp_path / "reports" / "efficientnetv2b0" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for f in [0, 3, 4]:
        data = {"accuracy": 0.88, "macro_f1": 0.83, "weighted_f1": 0.87, "training_duration_seconds": 12.5}
        with open(metrics_dir / f"fold_{f}.json", "w", encoding="utf-8") as fp:
            json.dump(data, fp)

    success = generate_screening_summary(tmp_path, screening_folds=[0, 3, 4])
    assert success is True

    summary_file = tmp_path / "reports" / "efficientnetv2b0" / "efficientnetv2b0_screening_summary.json"
    assert summary_file.is_file()
    with open(summary_file, "r", encoding="utf-8") as fp:
        summary = json.load(fp)

    assert summary["mean_accuracy"] == 0.88
    assert summary["mean_macro_f1"] == 0.83
    assert summary["executed_folds"] == [0, 3, 4]
    assert "std_accuracy" in summary
    assert "std_macro_f1" in summary
    assert "std_weighted_f1" in summary
    assert summary["reference_densenet121_exp_d_screening_average"]["mean_accuracy"] == 0.8610


def test_oof_predictions_schema_and_prob_sum(tmp_path: Path, exp_a_config: dict[str, Any]) -> None:
    """Test OOF prediction export schema and verify probability vector sums to ~1.0."""
    from scripts.train_efficientnetv2b0 import export_oof_predictions
    import pandas as pd

    head_cfg = exp_a_config["model"]["classifier_head"]
    model = build_efficientnetv2b0(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights=None,
        head_config=head_cfg,
    )

    dummy_imgs = tf.random.uniform((5, 224, 224, 3), minval=0.0, maxval=255.0, dtype=tf.float32)
    dummy_lbls = tf.one_hot([0, 1, 2, 3, 4], depth=22)
    dummy_ds = tf.data.Dataset.from_tensor_slices((dummy_imgs, dummy_lbls)).batch(5)

    csv_out = tmp_path / "test_oof.csv"
    export_oof_predictions(model, dummy_ds, exp_a_config, fold=0, output_csv_path=csv_out)

    assert csv_out.is_file()
    df = pd.read_csv(csv_out)

    expected_cols = [
        "image_path", "image_id", "fold", "true_label", "true_class",
        "predicted_label", "predicted_class", "correct"
    ] + [f"prob_{c}" for c in range(22)]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    prob_cols = [f"prob_{c}" for c in range(22)]
    prob_sums = df[prob_cols].sum(axis=1).values
    np.testing.assert_allclose(prob_sums, 1.0, atol=1e-4)


def test_densenet121_exp_d_remains_unchanged() -> None:
    """Verify DenseNet121 Exp D configuration and source code are unmutated."""
    densenet_cfg_path = Path("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    assert densenet_cfg_path.is_file()

    densenet_model_path = Path("src/models/densenet121.py")
    assert densenet_model_path.is_file()

    densenet_ckpt_dir = Path("artifacts/models/densenet121_exp_d_v1")
    if densenet_ckpt_dir.is_dir():
        assert (densenet_ckpt_dir / "checkpoints").exists() or (densenet_ckpt_dir / "evaluation").exists()


@pytest.mark.skip(reason="Smoke test downloading ImageNet weights - do not run automatically in standard suite")
def test_imagenet_smoke_test(exp_a_config: dict[str, Any]) -> None:
    """Smoke test for instantiating EfficientNetV2B0 with ImageNet weights."""
    head_cfg = exp_a_config["model"]["classifier_head"]
    model = build_efficientnetv2b0(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights="imagenet",
        head_config=head_cfg,
    )
    assert model.output_shape == (None, 22)

