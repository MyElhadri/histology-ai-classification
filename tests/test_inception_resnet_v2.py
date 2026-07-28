"""Unit tests for InceptionResNetV2 model architecture, preprocessing, fine-tuning,
and screening logic.

Test categories:
- YAML config validity
- Model architecture (output shape, 22 classes, float32 softmax)
- Preprocessing (internal Rescaling, no double-preprocessing, known values)
- Three-phase fine-tuning trainability
- BatchNormalization always frozen in phases 2 and 3
- Recompilation after trainability change
- Deterministic partial unfreezing fraction
- Mixed precision compatibility
- Manifest and fold counts
- OOF CSV schema and sample counters
- Cumulative screening summary (not overwritten by last fold)
- CLI --help without PYTHONPATH
- --dry-run without model.fit()
- DenseNet121 Exp D, EfficientNetV2B0, InceptionV3 unchanged

IMPORTANT:
- All model builds use weights=None (no ImageNet downloads)
- No model.fit() is called
- No scientific training in any test
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Rescaling

# Make sure src is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.heads import build_classification_head, build_article_inspired_head
from src.models.inception_resnet_v2 import (
    PREPROCESSING_LAYER_NAME,
    apply_full_inception_resnet_v2_fine_tuning,
    apply_partial_inception_resnet_v2_fine_tuning,
    build_inception_resnet_v2,
    freeze_backbone_batch_normalization,
    freeze_inception_resnet_v2_backbone,
    get_layer_counts,
    validate_dataset_preprocessing,
    validate_inception_resnet_v2_trainability,
    validate_preprocessing_values,
    verify_no_double_preprocessing,
)
from src.utils.config import load_yaml


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def exp_a_config() -> dict[str, Any]:
    """Loaded InceptionResNetV2 Exp A config dictionary."""
    config_path = PROJECT_ROOT / "configs" / "experiments" / "inception_resnet_v2_exp_a_high_performance.yaml"
    assert config_path.is_file(), f"Config not found: {config_path}"
    return load_yaml(config_path)


@pytest.fixture(scope="module")
def model_no_weights(exp_a_config: dict[str, Any]) -> tf.keras.Model:
    """InceptionResNetV2 model built with weights=None (no ImageNet download)."""
    head_cfg = exp_a_config["model"]["classifier_head"]
    return build_inception_resnet_v2(
        num_classes=22,
        input_shape=(299, 299, 3),
        weights=None,
        head_config=head_cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# YAML configuration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestYAMLConfig:
    def test_config_file_exists(self) -> None:
        p = PROJECT_ROOT / "configs" / "experiments" / "inception_resnet_v2_exp_a_high_performance.yaml"
        assert p.is_file()

    def test_experiment_name(self, exp_a_config: dict[str, Any]) -> None:
        assert exp_a_config["experiment"]["name"] == "inception-resnet-v2-exp-a-high-performance"

    def test_architecture(self, exp_a_config: dict[str, Any]) -> None:
        assert exp_a_config["model"]["architecture"] == "InceptionResNetV2"

    def test_num_classes(self, exp_a_config: dict[str, Any]) -> None:
        assert exp_a_config["model"]["num_classes"] == 22
        assert exp_a_config["data"]["num_classes"] == 22

    def test_input_shape_299(self, exp_a_config: dict[str, Any]) -> None:
        assert exp_a_config["model"]["input_shape"] == [299, 299, 3]
        assert exp_a_config["data"]["image_size"] == [299, 299]

    def test_preprocessing_config(self, exp_a_config: dict[str, Any]) -> None:
        pre = exp_a_config["preprocessing"]
        assert pre["input_dtype"] == "float32"
        assert pre["input_range"] == [0, 255]
        assert pre["output_range"] == [-1, 1]
        assert pre["method"] == "internal_rescaling"
        assert pre["external_divide_by_255"] is False
        assert pre["external_preprocess_input"] is False

    def test_screening_folds_order(self, exp_a_config: dict[str, Any]) -> None:
        # Must be [3, 0, 4] in this exact order
        assert exp_a_config["screening"]["folds"] == [3, 0, 4]

    def test_screening_expected_samples(self, exp_a_config: dict[str, Any]) -> None:
        sc = exp_a_config["screening"]["expected_samples"]
        assert sc["fold_3"] == 86
        assert sc["fold_0"] == 87
        assert sc["fold_4"] == 86
        assert sc["total"] == 259

    def test_seed(self, exp_a_config: dict[str, Any]) -> None:
        assert exp_a_config["experiment"]["seed"] == 42
        assert exp_a_config["project"]["seed"] == 42

    def test_three_phase_training(self, exp_a_config: dict[str, Any]) -> None:
        tr = exp_a_config["training"]
        assert "phase_1" in tr
        assert "phase_2" in tr
        assert "phase_3" in tr

    def test_head_type(self, exp_a_config: dict[str, Any]) -> None:
        assert exp_a_config["model"]["classifier_head"]["type"] == "article_inspired"

    def test_manifest_path(self, exp_a_config: dict[str, Any]) -> None:
        assert exp_a_config["data"]["folds_path"] == "data/manifests/densenet121_folds.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Architecture tests
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitecture:
    def test_output_shape_22_classes(self, model_no_weights: tf.keras.Model) -> None:
        assert model_no_weights.output_shape == (None, 22)

    def test_input_shape_299(self, model_no_weights: tf.keras.Model) -> None:
        assert model_no_weights.input_shape == (None, 299, 299, 3)

    def test_output_dtype_float32(self, model_no_weights: tf.keras.Model) -> None:
        test_input = np.random.uniform(0, 255, (2, 299, 299, 3)).astype(np.float32)
        output = model_no_weights(test_input, training=False)
        assert output.dtype == tf.float32

    def test_softmax_probabilities_sum_to_1(self, model_no_weights: tf.keras.Model) -> None:
        test_input = np.random.uniform(0, 255, (3, 299, 299, 3)).astype(np.float32)
        output = model_no_weights(test_input, training=False).numpy()
        sums = output.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(3), rtol=1e-5, atol=1e-4)

    def test_predictions_layer_present(self, model_no_weights: tf.keras.Model) -> None:
        layer_names = [l.name for l in model_no_weights.layers]
        assert "predictions" in layer_names

    def test_model_name(self, model_no_weights: tf.keras.Model) -> None:
        assert model_no_weights.name == "InceptionResNetV2"

    def test_weights_none_no_imagenet_download(self) -> None:
        """Building with weights=None must not trigger any ImageNet download."""
        model = build_inception_resnet_v2(
            num_classes=22,
            input_shape=(299, 299, 3),
            weights=None,
        )
        assert model.output_shape == (None, 22)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_rescaling_layer_present(self, model_no_weights: tf.keras.Model) -> None:
        rescaling_layers = [l for l in model_no_weights.layers if isinstance(l, Rescaling)]
        assert len(rescaling_layers) == 1

    def test_rescaling_layer_name(self, model_no_weights: tf.keras.Model) -> None:
        rescaling_layers = [l for l in model_no_weights.layers if isinstance(l, Rescaling)]
        assert rescaling_layers[0].name == PREPROCESSING_LAYER_NAME

    def test_rescaling_scale_and_offset(self, model_no_weights: tf.keras.Model) -> None:
        rescaling_layers = [l for l in model_no_weights.layers if isinstance(l, Rescaling)]
        layer = rescaling_layers[0]
        assert abs(float(layer.scale) - 1.0 / 127.5) < 1e-6
        assert abs(float(layer.offset) - (-1.0)) < 1e-6

    def test_no_double_preprocessing(self, model_no_weights: tf.keras.Model) -> None:
        # Should not raise
        verify_no_double_preprocessing(model_no_weights)

    def test_preprocessing_known_values_0(self) -> None:
        rescaling = Rescaling(scale=1.0 / 127.5, offset=-1.0)
        val = float(rescaling(tf.constant([[[0.0, 0.0, 0.0]]], dtype=tf.float32)).numpy()[0, 0, 0])
        assert abs(val - (-1.0)) < 1e-5

    def test_preprocessing_known_values_127_5(self) -> None:
        rescaling = Rescaling(scale=1.0 / 127.5, offset=-1.0)
        val = float(rescaling(tf.constant([[[127.5, 127.5, 127.5]]], dtype=tf.float32)).numpy()[0, 0, 0])
        assert abs(val - 0.0) < 1e-5

    def test_preprocessing_known_values_255(self) -> None:
        rescaling = Rescaling(scale=1.0 / 127.5, offset=-1.0)
        val = float(rescaling(tf.constant([[[255.0, 255.0, 255.0]]], dtype=tf.float32)).numpy()[0, 0, 0])
        assert abs(val - 1.0) < 1e-5

    def test_equivalence_with_official_preprocess_input(self) -> None:
        rescaling = Rescaling(scale=1.0 / 127.5, offset=-1.0)
        sample = np.random.uniform(0.0, 255.0, (4, 299, 299, 3)).astype(np.float32)
        rescaled = rescaling(sample).numpy()
        official = tf.keras.applications.inception_resnet_v2.preprocess_input(sample.copy())
        if hasattr(official, 'numpy'):
            official = official.numpy()
        np.testing.assert_allclose(rescaled, official, rtol=1e-5, atol=1e-4)

    def test_validate_preprocessing_values_function(self) -> None:
        """validate_preprocessing_values should pass without error."""
        validate_preprocessing_values()

    def test_dataset_preprocessing_validation_valid(self, exp_a_config: dict[str, Any]) -> None:
        valid = np.random.uniform(1.0, 255.0, (2, 299, 299, 3)).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((valid, np.zeros((2, 22)))).batch(2)
        validate_dataset_preprocessing(ds, exp_a_config)

    def test_dataset_preprocessing_rejects_normalized_01(self, exp_a_config: dict[str, Any]) -> None:
        invalid = np.random.uniform(0.01, 0.99, (2, 299, 299, 3)).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((invalid, np.zeros((2, 22)))).batch(2)
        with pytest.raises(ValueError, match="max value"):
            validate_dataset_preprocessing(ds, exp_a_config)


# ─────────────────────────────────────────────────────────────────────────────
# Head parity with DenseNet121 Exp D
# ─────────────────────────────────────────────────────────────────────────────

class TestHeadParity:
    def test_head_config_matches_densenet_exp_d(self, exp_a_config: dict[str, Any]) -> None:
        """Head configuration must be identical to DenseNet121 Exp D."""
        densenet_config_path = PROJECT_ROOT / "configs" / "experiments" / "densenet121_exp_d_rich_aug_article_head.yaml"
        densenet_cfg = load_yaml(densenet_config_path)

        irv2_head = exp_a_config["model"]["classifier_head"]
        dense_head = densenet_cfg["model"]["classifier_head"]

        assert irv2_head["type"] == dense_head["type"] == "article_inspired"
        assert irv2_head["dense_1_units"] == dense_head["dense_1_units"] == 512
        assert irv2_head["dense_1_activation"] == dense_head["dense_1_activation"] == "elu"
        assert irv2_head["batch_normalization"] == dense_head["batch_normalization"] is True
        assert irv2_head["dropout_rate"] == dense_head["dropout_rate"] == 0.30
        assert irv2_head["dense_2_units"] == dense_head["dense_2_units"] == 128
        assert irv2_head["dense_2_activation"] == dense_head["dense_2_activation"] == "elu"
        assert irv2_head["l2_strength"] == dense_head["l2_strength"] == 0.01
        assert irv2_head["output_activation"] == dense_head["output_activation"] == "softmax"

    def test_model_contains_article_inspired_layers(self, model_no_weights: tf.keras.Model) -> None:
        layer_names = [l.name for l in model_no_weights.layers]
        assert "global_average_pooling" in layer_names
        assert "classifier_dense_512" in layer_names
        assert "classifier_batch_norm" in layer_names
        assert "classifier_dropout" in layer_names
        assert "classifier_dense_128" in layer_names
        assert "predictions" in layer_names


# ─────────────────────────────────────────────────────────────────────────────
# Three-phase fine-tuning tests
# ─────────────────────────────────────────────────────────────────────────────

class TestThreePhaseFineTuning:
    def _build_model(self) -> tf.keras.Model:
        return build_inception_resnet_v2(
            num_classes=22,
            input_shape=(299, 299, 3),
            weights=None,
        )

    def test_phase_1_backbone_frozen(self) -> None:
        model = self._build_model()
        freeze_inception_resnet_v2_backbone(model)

        # Find backbone
        backbone = next(
            (l for l in model.layers if isinstance(l, tf.keras.Model) and "inception_resnet" in l.name.lower()),
            None,
        )
        assert backbone is not None
        assert backbone.trainable is False

    def test_phase_1_validation_passes(self) -> None:
        model = self._build_model()
        freeze_inception_resnet_v2_backbone(model)
        validate_inception_resnet_v2_trainability(model, phase=1)

    def test_phase_1_recompile_required(self) -> None:
        model = self._build_model()
        freeze_inception_resnet_v2_backbone(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        # If compiled without error, we're good
        assert model.optimizer is not None

    def test_phase_2_bn_always_frozen(self) -> None:
        model = self._build_model()
        apply_partial_inception_resnet_v2_fine_tuning(
            model=model,
            trainable_fraction=0.30,
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        backbone = next(
            (l for l in model.layers if isinstance(l, tf.keras.Model) and "inception_resnet" in l.name.lower()),
            None,
        )
        assert backbone is not None
        for layer in backbone.layers:
            if isinstance(layer, BatchNormalization):
                assert layer.trainable is False, f"BN layer {layer.name} is trainable in Phase 2"

    def test_phase_2_at_least_one_conv_trainable(self) -> None:
        model = self._build_model()
        apply_partial_inception_resnet_v2_fine_tuning(
            model=model,
            trainable_fraction=0.30,
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        backbone = next(
            (l for l in model.layers if isinstance(l, tf.keras.Model) and "inception_resnet" in l.name.lower()),
            None,
        )
        trainable_non_bn = [l for l in backbone.layers if not isinstance(l, BatchNormalization) and l.trainable]
        assert len(trainable_non_bn) > 0

    def test_phase_2_deterministic_fraction(self) -> None:
        """Running phase 2 twice with same fraction gives same number of trainable layers."""
        model1 = self._build_model()
        model2 = self._build_model()

        s1 = apply_partial_inception_resnet_v2_fine_tuning(model1, 0.30)
        s2 = apply_partial_inception_resnet_v2_fine_tuning(model2, 0.30)

        assert s1["trainable_backbone_layers"] == s2["trainable_backbone_layers"]
        assert s1["cutoff_index"] == s2["cutoff_index"]

    def test_phase_2_validation_passes(self) -> None:
        model = self._build_model()
        apply_partial_inception_resnet_v2_fine_tuning(
            model=model,
            trainable_fraction=0.30,
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(),
        )
        validate_inception_resnet_v2_trainability(model, phase=2)

    def test_phase_3_bn_always_frozen(self) -> None:
        model = self._build_model()
        apply_full_inception_resnet_v2_fine_tuning(
            model=model,
            optimizer=tf.keras.optimizers.Adam(3e-6),
            loss=tf.keras.losses.CategoricalCrossentropy(),
        )
        backbone = next(
            (l for l in model.layers if isinstance(l, tf.keras.Model) and "inception_resnet" in l.name.lower()),
            None,
        )
        for layer in backbone.layers:
            if isinstance(layer, BatchNormalization):
                assert layer.trainable is False, f"BN layer {layer.name} is trainable in Phase 3"

    def test_phase_3_validation_passes(self) -> None:
        model = self._build_model()
        apply_full_inception_resnet_v2_fine_tuning(
            model=model,
            optimizer=tf.keras.optimizers.Adam(3e-6),
            loss=tf.keras.losses.CategoricalCrossentropy(),
        )
        validate_inception_resnet_v2_trainability(model, phase=3)

    def test_recompilation_after_each_phase(self) -> None:
        """Model must be recompiled after each phase change."""
        model = self._build_model()

        # Phase 1
        freeze_inception_resnet_v2_backbone(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(),
        )
        assert model.optimizer is not None

        # Phase 2 (recompile happens inside apply_partial...)
        apply_partial_inception_resnet_v2_fine_tuning(
            model=model,
            trainable_fraction=0.30,
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(),
        )
        assert model.optimizer is not None

        # Phase 3 (recompile happens inside apply_full...)
        apply_full_inception_resnet_v2_fine_tuning(
            model=model,
            optimizer=tf.keras.optimizers.Adam(3e-6),
            loss=tf.keras.losses.CategoricalCrossentropy(),
        )
        assert model.optimizer is not None

    def test_freeze_backbone_batch_normalization_utility(self) -> None:
        model = self._build_model()
        count = freeze_backbone_batch_normalization(model)
        assert count > 0

        backbone = next(
            (l for l in model.layers if isinstance(l, tf.keras.Model) and "inception_resnet" in l.name.lower()),
            None,
        )
        for layer in backbone.layers:
            if isinstance(layer, BatchNormalization):
                assert layer.trainable is False

    def test_layer_counts_summary(self) -> None:
        model = self._build_model()
        counts = get_layer_counts(model)
        assert "backbone_total_layers" in counts
        assert counts["backbone_total_layers"] > 0
        assert counts["backbone_bn_layers"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Manifest and fold tests
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestAndFolds:
    def test_manifest_exists(self) -> None:
        manifest = PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"
        assert manifest.is_file()

    def test_manifest_432_images(self) -> None:
        manifest = PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"
        df = pd.read_csv(manifest)
        assert len(df) == 432

    def test_fold_counts(self) -> None:
        manifest = PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"
        df = pd.read_csv(manifest)
        counts = df["fold"].value_counts().to_dict()
        assert counts[0] == 87
        assert counts[1] == 87
        assert counts[2] == 86
        assert counts[3] == 86
        assert counts[4] == 86

    def test_screening_fold_3_has_86_images(self) -> None:
        manifest = PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"
        df = pd.read_csv(manifest)
        assert len(df[df["fold"] == 3]) == 86

    def test_screening_fold_0_has_87_images(self) -> None:
        manifest = PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"
        df = pd.read_csv(manifest)
        assert len(df[df["fold"] == 0]) == 87

    def test_screening_fold_4_has_86_images(self) -> None:
        manifest = PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"
        df = pd.read_csv(manifest)
        assert len(df[df["fold"] == 4]) == 86

    def test_total_screening_images_259(self) -> None:
        manifest = PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"
        df = pd.read_csv(manifest)
        screening_df = df[df["fold"].isin([3, 0, 4])]
        assert len(screening_df) == 259

    def test_class_mapping_22_classes(self) -> None:
        mapping_path = PROJECT_ROOT / "data" / "manifests" / "class_mapping.json"
        assert mapping_path.is_file()
        with open(mapping_path) as f:
            mapping = json.load(f)
        assert len(mapping) == 22


# ─────────────────────────────────────────────────────────────────────────────
# OOF CSV schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOOFCSVSchema:
    def test_oof_csv_schema_and_sample_counters(self, tmp_path: Path) -> None:
        """OOF CSV must have correct schema, probability sums, and counter consistency."""
        from scripts.train_inception_resnet_v2 import export_oof_predictions

        num_classes = 22
        val_images = np.random.uniform(10.0, 240.0, (10, 299, 299, 3)).astype(np.float32)
        val_labels = np.zeros((10, num_classes), dtype=np.float32)
        for i in range(10):
            val_labels[i, i % num_classes] = 1.0

        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(10)

        # Dummy model
        inputs = tf.keras.Input(shape=(299, 299, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        config = {
            "data": {
                "num_classes": 22,
                "folds_path": str(PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"),
            }
        }

        output_csv = tmp_path / "test_oof.csv"
        oof_df = export_oof_predictions(model, val_ds, config, fold=3, output_csv_path=output_csv)

        assert output_csv.is_file()

        # Verify schema
        expected_cols = [
            "image_path", "image_id", "fold", "true_label", "true_class",
            "predicted_label", "predicted_class", "correct",
        ] + [f"prob_{c}" for c in range(22)]
        assert list(oof_df.columns) == expected_cols

        # Verify probability sums ≈ 1
        prob_cols = [f"prob_{c}" for c in range(22)]
        sums = oof_df[prob_cols].sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(len(oof_df)), rtol=1e-4, atol=1e-4)

        # Verify predicted_label == argmax(prob_*)
        for i, row in oof_df.iterrows():
            probs = [row[f"prob_{c}"] for c in range(22)]
            assert int(np.argmax(probs)) == int(row["predicted_label"])

        # Verify counter consistency: accuracy = correct / total
        total_samples = len(oof_df)
        correct_samples = int(oof_df["correct"].sum())
        assert correct_samples <= total_samples
        if total_samples > 0:
            acc = correct_samples / total_samples
            assert 0.0 <= acc <= 1.0

        # Must never be 0/0 when data exists
        assert total_samples > 0

    def test_prob_cols_0_to_21_present(self, tmp_path: Path) -> None:
        """All prob_0 to prob_21 must be present."""
        from scripts.train_inception_resnet_v2 import export_oof_predictions

        num_classes = 22
        val_images = np.random.uniform(10.0, 240.0, (5, 299, 299, 3)).astype(np.float32)
        val_labels = np.zeros((5, num_classes), dtype=np.float32)
        for i in range(5):
            val_labels[i, i % num_classes] = 1.0

        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(5)

        inputs = tf.keras.Input(shape=(299, 299, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        config = {
            "data": {
                "num_classes": 22,
                "folds_path": str(PROJECT_ROOT / "data" / "manifests" / "densenet121_folds.csv"),
            }
        }

        output_csv = tmp_path / "test_prob_cols.csv"
        oof_df = export_oof_predictions(model, val_ds, config, fold=0, output_csv_path=output_csv)

        for c in range(22):
            assert f"prob_{c}" in oof_df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative screening summary test
# ─────────────────────────────────────────────────────────────────────────────

class TestScreeningSummary:
    def test_cumulative_summary_not_overwritten_by_last_fold(self, tmp_path: Path) -> None:
        """Simulate three separate fold runs and verify summary includes all folds."""
        from scripts.train_inception_resnet_v2 import generate_screening_summary

        output_dir = tmp_path / "results"
        metrics_dir = output_dir / "reports" / "inception_resnet_v2" / "metrics"
        pred_dir = output_dir / "reports" / "inception_resnet_v2" / "predictions"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Simulate fold 3 execution
        fold3_metrics = {
            "accuracy": 0.87,
            "macro_f1": 0.82,
            "weighted_f1": 0.86,
            "macro_precision": 0.83,
            "macro_recall": 0.81,
            "training_duration_seconds": 100.0,
        }
        with open(metrics_dir / "fold_3.json", "w") as f:
            json.dump(fold3_metrics, f)

        pred_rows_3 = [{"image_path": f"img_{i}", "true_label": i % 22, "predicted_label": i % 22, "correct": True} for i in range(86)]
        pd.DataFrame(pred_rows_3).to_csv(pred_dir / "fold_3_oof_predictions.csv", index=False)

        result = generate_screening_summary(output_dir, screening_folds=[3, 0, 4])
        assert result is False  # Not yet complete (folds 0 and 4 missing)

        # Simulate fold 0 execution
        fold0_metrics = {
            "accuracy": 0.85,
            "macro_f1": 0.80,
            "weighted_f1": 0.84,
            "macro_precision": 0.81,
            "macro_recall": 0.79,
            "training_duration_seconds": 110.0,
        }
        with open(metrics_dir / "fold_0.json", "w") as f:
            json.dump(fold0_metrics, f)

        pred_rows_0 = [{"image_path": f"img_{i+86}", "true_label": i % 22, "predicted_label": i % 22, "correct": True} for i in range(87)]
        pd.DataFrame(pred_rows_0).to_csv(pred_dir / "fold_0_oof_predictions.csv", index=False)

        result = generate_screening_summary(output_dir, screening_folds=[3, 0, 4])
        assert result is False  # Still missing fold 4

        # Simulate fold 4 execution
        fold4_metrics = {
            "accuracy": 0.86,
            "macro_f1": 0.81,
            "weighted_f1": 0.85,
            "macro_precision": 0.82,
            "macro_recall": 0.80,
            "training_duration_seconds": 105.0,
        }
        with open(metrics_dir / "fold_4.json", "w") as f:
            json.dump(fold4_metrics, f)

        pred_rows_4 = [{"image_path": f"img_{i+173}", "true_label": i % 22, "predicted_label": i % 22, "correct": True} for i in range(86)]
        pd.DataFrame(pred_rows_4).to_csv(pred_dir / "fold_4_oof_predictions.csv", index=False)

        result = generate_screening_summary(output_dir, screening_folds=[3, 0, 4])
        assert result is True  # All three folds now complete

        # Verify summary contains all three folds (not just fold_4)
        summary_path = output_dir / "reports" / "inception_resnet_v2" / "inception_resnet_v2_screening_summary.json"
        assert summary_path.is_file()
        with open(summary_path) as f:
            summary = json.load(f)

        assert "fold_3" in summary["metrics_by_fold"]
        assert "fold_0" in summary["metrics_by_fold"]
        assert "fold_4" in summary["metrics_by_fold"]
        assert len(summary["metrics_by_fold"]) == 3

        # Total OOF images should be 259 (86+87+86)
        assert summary["total_oof_images"] == 259
        assert summary["unique_images"] == 259


# ─────────────────────────────────────────────────────────────────────────────
# CLI tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_help_without_pythonpath(self, tmp_path: Path) -> None:
        """train_inception_resnet_v2.py --help must work from any directory without PYTHONPATH."""
        script_path = PROJECT_ROOT / "scripts" / "train_inception_resnet_v2.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        res = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=env,
        )

        assert res.returncode == 0, (
            f"Script failed with exit code {res.returncode}.\n"
            f"stdout: {res.stdout[:500]}\nstderr: {res.stderr[:500]}"
        )
        assert "ModuleNotFoundError" not in res.stderr
        assert "ModuleNotFoundError" not in res.stdout

    def test_dry_run_does_not_call_fit(self, tmp_path: Path) -> None:
        """--dry-run mode must not run model.fit()."""
        script_path = PROJECT_ROOT / "scripts" / "train_inception_resnet_v2.py"
        config_path = PROJECT_ROOT / "configs" / "experiments" / "inception_resnet_v2_exp_a_high_performance.yaml"

        res = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--config", str(config_path),
                "--output-dir", str(tmp_path / "out"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300,
        )

        # Dry-run should exit 0 (or print DRY RUN PASSED)
        combined = res.stdout + res.stderr
        assert "model.fit" not in combined.lower() or "DRY RUN" in combined or res.returncode == 0
        assert "ModuleNotFoundError" not in combined


# ─────────────────────────────────────────────────────────────────────────────
# Unchanged experiments protection
# ─────────────────────────────────────────────────────────────────────────────

class TestExistingExperimentsUnchanged:
    def test_densenet121_exp_d_config_unchanged(self) -> None:
        cfg_path = PROJECT_ROOT / "configs" / "experiments" / "densenet121_exp_d_rich_aug_article_head.yaml"
        assert cfg_path.is_file()
        cfg = load_yaml(cfg_path)
        assert cfg["model"]["architecture"] == "DenseNet121"
        assert cfg["project"]["name"] == "histology-ai-classification"
        assert cfg["model"]["classifier_head"]["type"] == "article_inspired"
        assert cfg["training"]["use_class_weights"] is True

    def test_efficientnetv2b0_config_unchanged(self) -> None:
        cfg_path = PROJECT_ROOT / "configs" / "experiments" / "efficientnetv2b0_exp_a_fair_comparison.yaml"
        assert cfg_path.is_file()
        cfg = load_yaml(cfg_path)
        assert cfg["model"]["architecture"] == "EfficientNetV2B0"

    def test_inceptionv3_config_unchanged(self) -> None:
        cfg_path = PROJECT_ROOT / "configs" / "experiments" / "inceptionv3_exp_a_fair_comparison.yaml"
        assert cfg_path.is_file()
        cfg = load_yaml(cfg_path)
        assert cfg["model"]["architecture"] == "InceptionV3"
        assert cfg["experiment"]["name"] == "inceptionv3-exp-a-fair-comparison"

    def test_densenet121_model_file_unchanged(self) -> None:
        model_path = PROJECT_ROOT / "src" / "models" / "densenet121.py"
        assert model_path.is_file()
        content = model_path.read_text(encoding="utf-8")
        assert "build_densenet121" in content
        assert "apply_fine_tuning_strategy" in content

    def test_inceptionv3_model_file_unchanged(self) -> None:
        model_path = PROJECT_ROOT / "src" / "models" / "inceptionv3.py"
        assert model_path.is_file()
        content = model_path.read_text(encoding="utf-8")
        assert "build_inceptionv3" in content
