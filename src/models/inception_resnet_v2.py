"""InceptionResNetV2 model architecture, preprocessing, and three-phase fine-tuning utilities.

This module implements ImageNet pre-trained InceptionResNetV2 for histology classification,
including:
- Built-in [0, 255] float32 to [-1, 1] preprocessing via a serializable Rescaling layer
- Explicit inference-mode BatchNormalization (backbone called with training=False)
- Three-phase fine-tuning: head training, partial unfreezing (30%), full unfreezing
- All BatchNormalization layers kept frozen during fine-tuning phases 2 and 3
- Reusable article-inspired classification head from DenseNet121 Exp D

Preprocessing contract (STRICT):
    External pipeline must supply: float32, values in [0, 255]
    Internal model applies: Rescaling(scale=1.0/127.5, offset=-1.0) → [-1, 1]
    No external divide-by-255, no external preprocess_input, no double normalization.
"""

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Rescaling
from tensorflow.keras.models import Model

from src.models.heads import build_classification_head

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preprocessing validation
# ---------------------------------------------------------------------------

PREPROCESSING_LAYER_NAME = "inception_resnet_v2_preprocessing"


def verify_no_double_preprocessing(model: Model) -> None:
    """Verify that model uses internal InceptionResNetV2 Rescaling and has no external double rescaling.

    InceptionResNetV2 expects inputs in range [-1, 1]. We include internal
    Rescaling(1/127.5, -1.0) inside the model graph. Adding external rescaling
    or preprocess_input would distort features.

    Args:
        model: Keras Model instance to inspect.

    Raises:
        ValueError: If internal rescaling is missing or external rescaling is detected.
    """
    rescaling_layers = [
        layer for layer in model.layers
        if isinstance(layer, Rescaling)
    ]
    if not rescaling_layers:
        raise ValueError(
            "InceptionResNetV2 model is missing internal "
            f"Rescaling(scale=1.0/127.5, offset=-1.0) layer named '{PREPROCESSING_LAYER_NAME}'."
        )

    if len(rescaling_layers) > 1:
        raise ValueError(
            f"Double preprocessing detected! Found {len(rescaling_layers)} "
            "Rescaling layers in InceptionResNetV2 model graph."
        )

    inc_rescaling = rescaling_layers[0]
    expected_scale = 1.0 / 127.5
    if (
        abs(float(inc_rescaling.scale) - expected_scale) > 1e-6
        or abs(float(inc_rescaling.offset) - (-1.0)) > 1e-6
    ):
        raise ValueError(
            f"InceptionResNetV2 Rescaling layer has invalid "
            f"scale={inc_rescaling.scale} or offset={inc_rescaling.offset}. "
            "Expected scale=1.0/127.5 and offset=-1.0."
        )

    logger.info(
        "Verified: InceptionResNetV2 internal Rescaling layer present; "
        "no external double preprocessing found."
    )


def validate_preprocessing_values() -> None:
    """Verify preprocessing transformation on known values.

    Checks:
        0.0     → -1.0
        127.5   → 0.0
        255.0   → +1.0

    Also verifies numerical equivalence with tf.keras.applications.inception_resnet_v2.preprocess_input.

    Raises:
        AssertionError: If any known-value check fails.
    """
    import numpy as np

    rescaling = Rescaling(scale=1.0 / 127.5, offset=-1.0)

    test_values = [0.0, 127.5, 255.0]
    expected_values = [-1.0, 0.0, 1.0]

    for val, expected in zip(test_values, expected_values):
        tensor = tf.constant([[[val, val, val]]], dtype=tf.float32)
        result = float(rescaling(tensor).numpy()[0, 0, 0])
        assert abs(result - expected) < 1e-5, (
            f"Preprocessing check failed for input {val}: "
            f"expected {expected}, got {result:.6f}"
        )

    # Verify numerical equivalence with official preprocess_input
    sample = np.random.uniform(0.0, 255.0, size=(4, 299, 299, 3)).astype(np.float32)
    rescaled = rescaling(sample).numpy()
    official = tf.keras.applications.inception_resnet_v2.preprocess_input(sample.copy())
    if hasattr(official, 'numpy'):
        official = official.numpy()
    max_diff = float(np.max(np.abs(rescaled - official)))
    assert max_diff < 1e-4, (
        f"InceptionResNetV2 Rescaling numerical divergence from preprocess_input: max_diff={max_diff:.6f}"
    )

    logger.info(
        "Preprocessing validation passed: 0→-1, 127.5→0, 255→1; "
        "numerical equivalence with preprocess_input confirmed."
    )


def validate_dataset_preprocessing(dataset: tf.data.Dataset, config: dict[str, Any]) -> None:
    """Validate real dataset batches to ensure float32 [0, 255] input tensors entering the model.

    Args:
        dataset: tf.data.Dataset instance to inspect.
        config: Configuration dictionary to inspect for forbidden rescaling settings.

    Raises:
        ValueError: If input tensor dtype, range, or config violates [0, 255] float32 requirements.
    """
    # Check config for forbidden preprocessing parameters
    for section_name in ["data", "augmentation", "preprocessing"]:
        section = config.get(section_name, {})
        if not isinstance(section, dict):
            continue
        if section.get("external_divide_by_255") is True:
            raise ValueError(
                f"Illegal external_divide_by_255=True in '{section_name}' section. "
                "InceptionResNetV2 expects native [0, 255] float32 inputs."
            )
        if section.get("external_preprocess_input") is True:
            raise ValueError(
                f"Illegal external_preprocess_input=True in '{section_name}' section. "
                "InceptionResNetV2 model handles preprocessing internally."
            )
        rescale_val = section.get("rescale")
        if rescale_val is not None and (rescale_val == 1 / 255 or rescale_val <= 0.01):
            raise ValueError(
                f"Configuration section '{section_name}' contains illegal rescale={rescale_val}. "
                "InceptionResNetV2 model expects native [0, 255] float32 inputs."
            )

    try:
        batch = next(iter(dataset))
    except StopIteration:
        raise ValueError("Dataset is empty; cannot validate preprocessing.")

    images = batch[0] if isinstance(batch, (tuple, list)) else batch

    if images.dtype != tf.float32:
        raise ValueError(
            f"Expected input images dtype tf.float32, got {images.dtype}"
        )

    min_val = float(tf.reduce_min(images))
    max_val = float(tf.reduce_max(images))

    if min_val < -1e-3 or max_val > 255.001:
        raise ValueError(
            f"Dataset images out of expected [0, 255] range: "
            f"min={min_val:.4f}, max={max_val:.4f}. "
            "Stop: double preprocessing or incorrect interval detected."
        )

    if max_val <= 1.0 and max_val > 0.0:
        raise ValueError(
            f"Dataset images have max value <= 1.0 (max={max_val:.4f}). "
            "This indicates double preprocessing or unexpected [0, 1] scaling! "
            "InceptionResNetV2 expects native [0, 255] float32 inputs. "
            "Stop: incorrect interval detected."
        )

    # Verify internal output range by running a mini-batch through rescaling
    rescaling_layer = Rescaling(scale=1.0 / 127.5, offset=-1.0)
    internal_out = rescaling_layer(images[:1]).numpy()
    internal_min = float(internal_out.min())
    internal_max = float(internal_out.max())
    if internal_min < -1.01 or internal_max > 1.01:
        raise ValueError(
            f"Internal rescaling output out of expected [-1.01, 1.01] range: "
            f"min={internal_min:.4f}, max={internal_max:.4f}. "
            "Stop: double preprocessing or incorrect interval detected."
        )

    logger.info(
        f"InceptionResNetV2 dataset preprocessing validated: "
        f"dtype={images.dtype}, min={min_val:.2f}, max={max_val:.2f}, "
        f"internal_range=[{internal_min:.3f}, {internal_max:.3f}]"
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_inception_resnet_v2(
    num_classes: int = 22,
    input_shape: tuple[int, int, int] = (299, 299, 3),
    weights: str | None = "imagenet",
    dropout_rate: float = 0.30,
    head_config: dict[str, Any] | None = None,
) -> Model:
    """Build InceptionResNetV2 classification model with built-in [-1, 1] rescaling.

    Architecture:
        Input (299×299×3, float32, [0,255])
        → Rescaling(1/127.5, -1.0)  [internal, [-1,1]]
        → InceptionResNetV2 backbone (training=False for BN inference mode)
        → article_inspired head (GAP → Dense(512)+ELU → BN → Dropout → Dense(128,L2)+ELU → Dense(22,softmax))

    Args:
        num_classes: Number of target classification classes (22 for this project).
        input_shape: Image input dimensions (H, W, C). Should be (299, 299, 3).
        weights: Pre-trained weights to load ('imagenet' or None). Use None for unit tests only.
        dropout_rate: Classifier dropout rate if baseline head is used.
        head_config: Configuration dictionary for classifier head architecture.
                     Should be article_inspired for Exp A.

    Returns:
        Keras Model instance.

    Raises:
        ValueError: If preprocessing verification fails.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Internal Rescaling layer: [0, 255] float32 → [-1, 1] float32
    # This is the ONLY preprocessing. External pipeline must not divide by 255.
    x = Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name=PREPROCESSING_LAYER_NAME,
    )(inputs)

    backbone = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    backbone._name = "inception_resnet_v2"

    # Call backbone explicitly with training=False to maintain BN in inference mode
    # This applies during ALL phases (head training, partial fine-tuning, full fine-tuning).
    features = backbone(x, training=False)

    outputs = build_classification_head(
        x=features,
        num_classes=num_classes,
        head_config=head_config,
        dropout_rate=dropout_rate,
    )

    model = Model(inputs=inputs, outputs=outputs, name="InceptionResNetV2")

    verify_no_double_preprocessing(model)
    return model


# ---------------------------------------------------------------------------
# Phase 1: Freeze backbone
# ---------------------------------------------------------------------------

def freeze_inception_resnet_v2_backbone(model: Model) -> int:
    """Freeze all layers of the InceptionResNetV2 backbone for Phase 1 head training.

    Head layers (global_average_pooling, classifier_*, predictions) remain trainable.

    Args:
        model: InceptionResNetV2 Keras Model instance.

    Returns:
        Number of trainable parameters in head (Phase 1).
    """
    backbone = _find_backbone(model)
    if backbone is not None:
        backbone.trainable = False
        logger.info(
            f"InceptionResNetV2 backbone frozen for Phase 1 head training. "
            f"Backbone layers: {len(backbone.layers)}"
        )
    else:
        logger.warning("Could not locate InceptionResNetV2 backbone layer to freeze.")

    head_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    logger.info(f"Phase 1 trainable parameters (head only): {head_params}")
    return head_params


# ---------------------------------------------------------------------------
# Phase 2: Partial fine-tuning (last 30% of backbone layers)
# ---------------------------------------------------------------------------

def apply_partial_inception_resnet_v2_fine_tuning(
    model: Model,
    trainable_fraction: float = 0.30,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    loss: Any | None = None,
    metrics: list[Any] | None = None,
) -> dict[str, Any]:
    """Unfreeze only the last trainable_fraction of backbone layers for Phase 2.

    - First (1 - trainable_fraction) of backbone layers remain frozen
    - Last trainable_fraction of backbone layers are unfrozen
    - ALL BatchNormalization layers in backbone remain FROZEN
    - Model is recompiled after trainability change

    The fraction is computed on non-BN backbone layers only, to avoid counting
    BN layers in the fraction (which are always frozen anyway).

    Args:
        model: InceptionResNetV2 Keras Model instance.
        trainable_fraction: Fraction of backbone layers to unfreeze (default 0.30 = 30%).
        optimizer: Optimizer for recompilation.
        loss: Loss function for recompilation.
        metrics: Metrics list for recompilation.

    Returns:
        Dictionary with trainability summary.

    Raises:
        ValueError: If BN layers are found trainable after application.
    """
    backbone = _find_backbone(model)
    if backbone is None:
        raise ValueError("Could not locate InceptionResNetV2 backbone in model.")

    # Ensure backbone is trainable at the top level so individual layer control works
    backbone.trainable = True

    all_backbone_layers = backbone.layers
    total_backbone = len(all_backbone_layers)

    # Compute cutoff: freeze first (1 - fraction), unfreeze last fraction
    # This is deterministic given the same backbone
    cutoff = int(total_backbone * (1.0 - trainable_fraction))

    for i, layer in enumerate(all_backbone_layers):
        is_bn = isinstance(layer, BatchNormalization)
        if is_bn:
            # BatchNormalization always frozen in phases 2 and 3
            layer.trainable = False
        elif i < cutoff:
            # Frozen: first (1 - fraction) of layers
            layer.trainable = False
        else:
            # Trainable: last fraction of layers
            layer.trainable = True

    # Verify: no BN should be trainable
    trainable_bn = [
        layer.name for layer in backbone.layers
        if isinstance(layer, BatchNormalization) and layer.trainable
    ]
    if trainable_bn:
        raise ValueError(
            f"Phase 2 violation: Found {len(trainable_bn)} trainable BatchNormalization "
            f"layers in backbone: {trainable_bn[:5]}"
        )

    # Verify: at least one deep conv layer is trainable
    trainable_non_bn = [
        layer.name for layer in backbone.layers
        if not isinstance(layer, BatchNormalization) and layer.trainable
    ]
    if not trainable_non_bn:
        raise ValueError(
            "Phase 2: No non-BatchNormalization backbone layers are trainable. "
            f"trainable_fraction={trainable_fraction}, cutoff={cutoff}/{total_backbone}"
        )

    total_bn = sum(1 for l in backbone.layers if isinstance(l, BatchNormalization))
    frozen_count = sum(1 for l in backbone.layers if not l.trainable)
    trainable_count = sum(1 for l in backbone.layers if l.trainable)

    summary = {
        "phase": 2,
        "trainable_fraction": trainable_fraction,
        "total_backbone_layers": total_backbone,
        "cutoff_index": cutoff,
        "frozen_backbone_layers": frozen_count,
        "trainable_backbone_layers": trainable_count,
        "total_bn_layers": total_bn,
        "trainable_bn_layers": 0,
        "first_trainable_layer": trainable_non_bn[0] if trainable_non_bn else "None",
    }

    logger.info(
        f"Phase 2 partial fine-tuning: {trainable_count}/{total_backbone} backbone layers trainable "
        f"(last {trainable_fraction*100:.0f}%), {total_bn} BN layers frozen."
    )

    if optimizer is not None and loss is not None:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info("Model recompiled for Phase 2.")

    return summary


# ---------------------------------------------------------------------------
# Phase 3: Full fine-tuning
# ---------------------------------------------------------------------------

def apply_full_inception_resnet_v2_fine_tuning(
    model: Model,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    loss: Any | None = None,
    metrics: list[Any] | None = None,
) -> dict[str, Any]:
    """Unfreeze all compatible backbone layers for Phase 3 full fine-tuning.

    - ALL non-BN backbone layers are made trainable
    - ALL BatchNormalization layers in backbone remain FROZEN
    - Model is recompiled after trainability change

    Args:
        model: InceptionResNetV2 Keras Model instance.
        optimizer: Optimizer for recompilation (use very low LR, e.g. 3e-6).
        loss: Loss function for recompilation.
        metrics: Metrics list for recompilation.

    Returns:
        Dictionary with trainability summary.

    Raises:
        ValueError: If BN layers are found trainable after application.
    """
    backbone = _find_backbone(model)
    if backbone is None:
        raise ValueError("Could not locate InceptionResNetV2 backbone in model.")

    backbone.trainable = True

    # Freeze all BN layers in backbone
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    # Verify: no BN should be trainable
    trainable_bn = [
        layer.name for layer in backbone.layers
        if isinstance(layer, BatchNormalization) and layer.trainable
    ]
    if trainable_bn:
        raise ValueError(
            f"Phase 3 violation: Found {len(trainable_bn)} trainable BatchNormalization "
            f"layers in backbone: {trainable_bn[:5]}"
        )

    total_bn = sum(1 for l in backbone.layers if isinstance(l, BatchNormalization))
    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    total_backbone = len(backbone.layers)

    summary = {
        "phase": 3,
        "total_backbone_layers": total_backbone,
        "trainable_backbone_layers": trainable_count,
        "frozen_bn_layers": total_bn,
        "trainable_bn_layers": 0,
    }

    logger.info(
        f"Phase 3 full fine-tuning: {trainable_count}/{total_backbone} backbone layers trainable, "
        f"{total_bn} BN layers frozen."
    )

    if optimizer is not None and loss is not None:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info("Model recompiled for Phase 3.")

    return summary


# ---------------------------------------------------------------------------
# BatchNormalization freezing helper
# ---------------------------------------------------------------------------

def freeze_backbone_batch_normalization(model: Model) -> int:
    """Freeze all BatchNormalization layers in the InceptionResNetV2 backbone.

    Args:
        model: InceptionResNetV2 Keras Model instance.

    Returns:
        Number of BatchNormalization layers frozen.
    """
    backbone = _find_backbone(model)
    if backbone is None:
        logger.warning("Could not locate backbone to freeze BatchNormalization layers.")
        return 0

    count = 0
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            count += 1

    logger.info(f"Frozen {count} BatchNormalization layers in InceptionResNetV2 backbone.")
    return count


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inception_resnet_v2_trainability(model: Model, phase: int) -> dict[str, Any]:
    """Validate that model trainability state is consistent for the given phase.

    Args:
        model: InceptionResNetV2 Keras Model instance.
        phase: Training phase (1, 2, or 3).

    Returns:
        Validation summary dictionary.

    Raises:
        ValueError: If trainability state violates phase requirements.
    """
    backbone = _find_backbone(model)
    if backbone is None:
        raise ValueError("Could not locate InceptionResNetV2 backbone in model.")

    trainable_bn = [
        layer.name for layer in backbone.layers
        if isinstance(layer, BatchNormalization) and layer.trainable
    ]
    total_bn = sum(1 for l in backbone.layers if isinstance(l, BatchNormalization))

    if phase in (2, 3) and trainable_bn:
        raise ValueError(
            f"Phase {phase} validation failed: Found {len(trainable_bn)} trainable "
            f"BatchNormalization layers: {trainable_bn[:5]}"
        )

    trainable_non_bn = [
        layer for layer in backbone.layers
        if not isinstance(layer, BatchNormalization) and layer.trainable
    ]

    if phase == 1 and trainable_non_bn:
        layer_names = [l.name for l in trainable_non_bn]
        raise ValueError(
            f"Phase 1 validation failed: {len(trainable_non_bn)} backbone non-BN layers "
            f"are trainable (should be 0): {layer_names[:5]}"
        )

    if phase in (2, 3) and not trainable_non_bn:
        raise ValueError(
            f"Phase {phase} validation failed: No backbone non-BN layers are trainable."
        )

    trainable_total = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    summary = {
        "phase": phase,
        "total_bn_layers": total_bn,
        "trainable_bn_layers": len(trainable_bn),
        "trainable_non_bn_backbone_layers": len(trainable_non_bn),
        "total_trainable_params": trainable_total,
        "validation_passed": True,
    }

    logger.info(
        f"Phase {phase} trainability validation passed: "
        f"{len(trainable_non_bn)} non-BN backbone trainable, "
        f"0/{total_bn} BN trainable, "
        f"{trainable_total} total trainable params."
    )
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_backbone(model: Model) -> Model | None:
    """Find the InceptionResNetV2 backbone submodel."""
    for layer in model.layers:
        if isinstance(layer, Model) and "inception_resnet" in layer.name.lower():
            return layer
    # Fallback: any Model submodel
    for layer in model.layers:
        if isinstance(layer, Model) and layer is not model:
            return layer
    return None


def get_layer_counts(model: Model) -> dict[str, int]:
    """Return a summary of layer counts for documentation purposes.

    Args:
        model: Built InceptionResNetV2 model.

    Returns:
        Dictionary with total, BN, and head layer counts.
    """
    backbone = _find_backbone(model)
    if backbone is None:
        return {}

    total = len(backbone.layers)
    bn_count = sum(1 for l in backbone.layers if isinstance(l, BatchNormalization))
    non_bn = total - bn_count

    return {
        "backbone_total_layers": total,
        "backbone_bn_layers": bn_count,
        "backbone_non_bn_layers": non_bn,
        "phase_1_trainable_backbone_layers": 0,
        "phase_2_trainable_backbone_layers_approx": int(non_bn * 0.30),
        "phase_3_trainable_backbone_layers": non_bn,
        "always_frozen_bn_layers": bn_count,
    }
