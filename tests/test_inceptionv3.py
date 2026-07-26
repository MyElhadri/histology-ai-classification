"""Unit tests for InceptionV3 model architecture, preprocessing, fine-tuning, and screening logic."""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Rescaling

from src.models.heads import build_classification_head
from src.models.inceptionv3 import (
    apply_inceptionv3_fine_tuning_strategy,
    build_inceptionv3,
    freeze_inceptionv3_backbone,
    validate_dataset_preprocessing,
    validate_inceptionv3_strategy,
    verify_no_double_preprocessing,
)
from src.utils.config import load_yaml


@pytest.fixture
def exp_a_config() -> dict[str, Any]:
    """Fixture providing loaded InceptionV3 Exp A config dictionary."""
    config_path = Path("configs/experiments/inceptionv3_exp_a_fair_comparison.yaml")
    assert config_path.is_file(), f"Config file not found at {config_path}"
    return load_yaml(config_path)


def test_config_loading_and_parameters(exp_a_config: dict[str, Any]) -> None:
    """Test YAML config loading and key protocol requirements."""
    assert exp_a_config["experiment"]["name"] == "inceptionv3-exp-a-fair-comparison"
    assert exp_a_config["model"]["architecture"] == "InceptionV3"
    assert exp_a_config["model"]["num_classes"] == 22
    assert exp_a_config["model"]["input_shape"] == [224, 224, 3]
    assert exp_a_config["preprocessing"]["input_dtype"] == "float32"
    assert exp_a_config["preprocessing"]["input_range"] == [0, 255]
    assert exp_a_config["preprocessing"]["model_output_range"] == [-1, 1]
    assert exp_a_config["preprocessing"]["method"] == "inceptionv3_rescaling"
    assert exp_a_config["preprocessing"]["external_divide_by_255"] is False
    assert exp_a_config["preprocessing"]["external_preprocess_input"] is False
    assert exp_a_config["screening"]["folds"] == [0, 3, 4]


def test_head_parity_with_archived_densenet_d(exp_a_config: dict[str, Any]) -> None:
    """Test classifier head structure parity with DenseNet121 D."""
    head_cfg = exp_a_config["model"]["classifier_head"]
    assert head_cfg["type"] == "article_inspired"
    assert head_cfg["pooling"] == "global_average"
    assert head_cfg["dense_1_units"] == 512
    assert head_cfg["dense_1_activation"] == "elu"
    assert head_cfg["batch_normalization"] is True
    assert head_cfg["dropout_rate"] == 0.30
    assert head_cfg["dense_2_units"] == 128
    assert head_cfg["dense_2_activation"] == "elu"
    assert head_cfg["l2_strength"] == 0.01
    assert head_cfg["output_activation"] == "softmax"


def test_model_build_and_rescaling_equivalence(exp_a_config: dict[str, Any]) -> None:
    """Test building InceptionV3 with weights=None and verify numerical rescaling equivalence."""
    head_cfg = exp_a_config["model"]["classifier_head"]
    model = build_inceptionv3(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights=None,
        head_config=head_cfg,
    )
    assert model.output_shape == (None, 22)
    verify_no_double_preprocessing(model)

    # Verify numerical equivalence between Rescaling(1/127.5, -1.0) and tf.keras.applications.inception_v3.preprocess_input
    sample_images = np.random.uniform(0.0, 255.0, size=(4, 224, 224, 3)).astype(np.float32)
    rescaling_layer = Rescaling(scale=1.0 / 127.5, offset=-1.0)
    rescaled_output = rescaling_layer(sample_images).numpy()
    official_output = tf.keras.applications.inception_v3.preprocess_input(sample_images.copy())

    np.testing.assert_allclose(rescaled_output, official_output, rtol=1e-5, atol=1e-5)


def test_backbone_training_false_and_bn_freezing(exp_a_config: dict[str, Any]) -> None:
    """Test two-phase fine-tuning trainability rules."""
    head_cfg = exp_a_config["model"]["classifier_head"]
    model = build_inceptionv3(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights=None,
        head_config=head_cfg,
    )

    # Phase 1: Freeze backbone
    freeze_inceptionv3_backbone(model)
    backbone = next(l for l in model.layers if isinstance(l, tf.keras.Model) or "inception_v3" in l.name.lower())
    assert backbone.trainable is False

    # Phase 2: Unfreeze backbone with frozen BatchNorms
    apply_inceptionv3_fine_tuning_strategy(
        model=model,
        strategy="full",
        keep_batch_normalization_frozen=True,
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
    )
    assert backbone.trainable is True

    # Verify all BN layers inside backbone remain frozen
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            assert layer.trainable is False

    validate_inceptionv3_strategy(model, exp_a_config)


def test_dataset_preprocessing_validation(exp_a_config: dict[str, Any]) -> None:
    """Test validation of dataset batch preprocessing range [0, 255] float32."""
    valid_batch = np.random.uniform(0.0, 255.0, size=(2, 224, 224, 3)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((valid_batch, np.zeros((2, 22)))).batch(2)
    validate_dataset_preprocessing(ds, exp_a_config)

    # Test rejection of pre-scaled [0, 1] inputs
    invalid_batch = np.random.uniform(0.0, 1.0, size=(2, 224, 224, 3)).astype(np.float32)
    ds_invalid = tf.data.Dataset.from_tensor_slices((invalid_batch, np.zeros((2, 22)))).batch(2)
    with pytest.raises(ValueError, match="Dataset images have max value <= 1.0"):
        validate_dataset_preprocessing(ds_invalid, exp_a_config)


def test_manifest_and_folds_verification(exp_a_config: dict[str, Any]) -> None:
    """Test manifest file presence and exact fold counts."""
    manifest_path = Path(exp_a_config["data"]["folds_path"])
    assert manifest_path.is_file()

    df = pd.read_csv(manifest_path)
    assert len(df) == 432
    assert df["image_id"].nunique() == 432

    fold_counts = df["fold"].value_counts().to_dict()
    assert fold_counts[0] == 87
    assert fold_counts[1] == 87
    assert fold_counts[2] == 86
    assert fold_counts[3] == 86
    assert fold_counts[4] == 86


def test_oof_predictions_schema_and_sample_counters(tmp_path: Path) -> None:
    """Test OOF CSV prediction schema and accuracy == correct_samples / total_samples equality."""
    from scripts.train_inceptionv3 import export_oof_predictions

    num_classes = 22
    val_images = np.random.uniform(0.0, 255.0, size=(10, 224, 224, 3)).astype(np.float32)
    val_labels = np.zeros((10, num_classes), dtype=np.float32)
    for i in range(10):
        val_labels[i, i % num_classes] = 1.0

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(10)

    # Build dummy model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    config = {
        "data": {
            "num_classes": 22,
            "folds_path": "data/manifests/densenet121_folds.csv",
        }
    }

    output_csv = tmp_path / "test_oof.csv"
    oof_df = export_oof_predictions(model, val_ds, config, fold=0, output_csv_path=output_csv)

    assert output_csv.is_file()
    assert len(oof_df) == 10

    expected_cols = [
        "image_path", "image_id", "fold", "true_label", "true_class",
        "predicted_label", "predicted_class", "correct"
    ] + [f"prob_{c}" for c in range(22)]
    assert list(oof_df.columns) == expected_cols

    # Verify probability sums ~ 1.0
    prob_cols = [f"prob_{c}" for c in range(22)]
    sums = oof_df[prob_cols].sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(10), rtol=1e-5, atol=1e-5)

    # Verify sample counter equality rule
    total_samples = len(oof_df)
    correct_samples = int(oof_df["correct"].sum())
    calculated_acc = correct_samples / total_samples
    assert calculated_acc == oof_df["correct"].mean()


def test_densenet121_exp_d_remains_unchanged() -> None:
    """Test that DenseNet121 Exp D config and artifacts remain untouched."""
    densenet_config_path = Path("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    assert densenet_config_path.is_file()

    densenet_cfg = load_yaml(densenet_config_path)
    assert densenet_cfg["model"]["architecture"] == "DenseNet121"
    assert densenet_cfg["project"]["name"] == "histology-ai-classification"

    # Verify DenseNet checkpoint directory exists if local artifacts present
    densenet_ckpt_dir = Path("artifacts/models/densenet121_exp_d_v1")
    if densenet_ckpt_dir.exists():
        assert (densenet_ckpt_dir / "checkpoints").exists() or (densenet_ckpt_dir / "evaluation").exists()


def test_script_execution_without_pythonpath(tmp_path: Path) -> None:
    """Test that train_inceptionv3.py runs with --help from outside repo without PYTHONPATH."""
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "train_inceptionv3.py"

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    res = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert res.returncode == 0, f"Script failed with exit code {res.returncode}. Output:\n{res.stderr}"
    assert "ModuleNotFoundError" not in res.stderr
    assert "ModuleNotFoundError" not in res.stdout
    assert "Train InceptionV3" in res.stdout or "--help" in res.stdout
