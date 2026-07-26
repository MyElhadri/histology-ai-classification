"""InceptionV3 model architecture, preprocessing, and fine-tuning utilities.

This module implements ImageNet pre-trained InceptionV3 for histology classification,
including built-in [0, 255] float32 to [-1, 1] preprocessing via a serializable
tf.keras.layers.Rescaling layer, explicit inference-mode BatchNormalization,
and two-phase fine-tuning support.
"""

import logging
from typing import Any
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Rescaling
from tensorflow.keras.models import Model

from src.models.heads import build_classification_head

logger = logging.getLogger(__name__)


def verify_no_double_preprocessing(model: Model) -> None:
    """Verify that model uses internal InceptionV3 Rescaling and has no external double rescaling.

    InceptionV3 expects inputs in range [-1, 1]. We include internal Rescaling(1/127.5, -1.0)
    inside the model graph. Adding external rescaling or preprocess_input would distort features.

    Args:
        model: Keras Model instance to inspect.

    Raises:
        ValueError: If internal rescaling is missing or external rescaling is detected.
    """
    rescaling_layers = [l for l in model.layers if isinstance(l, Rescaling)]
    if not rescaling_layers:
        raise ValueError(
            "InceptionV3 model is missing internal Rescaling(scale=1.0/127.5, offset=-1.0) layer."
        )

    # Check for duplicate rescaling layers
    if len(rescaling_layers) > 1:
        raise ValueError(
            f"Double preprocessing detected! Found {len(rescaling_layers)} Rescaling layers in model graph."
        )

    # Verify scale and offset of the internal rescaling layer
    inc_rescaling = rescaling_layers[0]
    if abs(float(inc_rescaling.scale) - 1.0 / 127.5) > 1e-6 or abs(float(inc_rescaling.offset) - (-1.0)) > 1e-6:
        raise ValueError(
            f"InceptionV3 Rescaling layer has invalid scale={inc_rescaling.scale} or offset={inc_rescaling.offset}. "
            "Expected scale=1.0/127.5 and offset=-1.0."
        )

    logger.info("Verified: InceptionV3 internal Rescaling layer present; no external double preprocessing found.")


def validate_dataset_preprocessing(dataset: tf.data.Dataset, config: dict[str, Any]) -> None:
    """Validate real dataset batches to ensure float32 [0, 255] input tensors entering the model.

    Args:
        dataset: tf.data.Dataset instance to inspect.
        config: Configuration dictionary to inspect for forbidden rescaling settings.

    Raises:
        ValueError: If input tensor dtype, range, or config violates [0, 255] float32 requirements.
    """
    for section_name in ["data", "augmentation", "preprocessing"]:
        section = config.get(section_name, {})
        if not isinstance(section, dict):
            continue
        if section.get("external_divide_by_255") is True:
            raise ValueError(f"Illegal external_divide_by_255=True in '{section_name}' section.")
        if section.get("external_preprocess_input") is True:
            raise ValueError(f"Illegal external_preprocess_input=True in '{section_name}' section.")
        rescale_val = section.get("rescale")
        if rescale_val is not None and (rescale_val == 1 / 255 or rescale_val <= 0.01):
            raise ValueError(
                f"Configuration section '{section_name}' contains illegal rescale={rescale_val}. "
                "InceptionV3 model expects native [0, 255] float32 inputs."
            )

    try:
        batch = next(iter(dataset))
    except StopIteration:
        raise ValueError("Dataset is empty; cannot validate preprocessing.")

    images = batch[0] if isinstance(batch, (tuple, list)) else batch

    if images.dtype != tf.float32:
        raise ValueError(f"Expected input images dtype tf.float32, got {images.dtype}")

    min_val = float(tf.reduce_min(images))
    max_val = float(tf.reduce_max(images))

    if min_val < -1e-3 or max_val > 255.001:
        raise ValueError(
            f"Dataset images out of expected [0, 255] range: min={min_val:.4f}, max={max_val:.4f}"
        )

    if max_val <= 1.0 and max_val > 0.0:
        raise ValueError(
            f"Dataset images have max value <= 1.0 (max={max_val:.4f}). "
            "This indicates double preprocessing or unexpected [0, 1] scaling! "
            "InceptionV3 expects native [0, 255] float32 inputs."
        )

    logger.info(
        f"Dataset preprocessing successfully validated: dtype={images.dtype}, min={min_val:.2f}, max={max_val:.2f}"
    )


def build_inceptionv3(
    num_classes: int = 22,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
    dropout_rate: float = 0.30,
    head_config: dict[str, Any] | None = None,
) -> Model:
    """Build InceptionV3 classification model with built-in [-1, 1] rescaling.

    Args:
        num_classes: Number of target classification classes.
        input_shape: Image input dimensions (H, W, C).
        weights: Pre-trained weights to load ('imagenet' or None).
        dropout_rate: Classifier dropout rate if baseline head is used.
        head_config: Configuration dictionary for classifier head architecture.

    Returns:
        Compiled Keras Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Internal Rescaling layer converting [0, 255] float32 to [-1, 1] float32
    x = Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="inceptionv3_preprocessing"
    )(inputs)

    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )

    # Call backbone explicitly with training=False to enforce inference mode on BN layers
    features = backbone(x, training=False)

    outputs = build_classification_head(
        x=features,
        num_classes=num_classes,
        head_config=head_config,
        dropout_rate=dropout_rate
    )

    model = Model(inputs=inputs, outputs=outputs, name="InceptionV3")
    verify_no_double_preprocessing(model)
    return model


def freeze_inceptionv3_backbone(model: Model) -> None:
    """Freeze all layers of the InceptionV3 backbone for Phase 1 head training.

    Args:
        model: InceptionV3 Keras Model instance.
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, Model) or "inception_v3" in layer.name.lower():
            backbone = layer
            break

    if backbone is not None:
        backbone.trainable = False
        logger.info("InceptionV3 backbone frozen for Phase 1 head training.")
    else:
        logger.warning("Could not locate InceptionV3 backbone layer to freeze.")


def apply_inceptionv3_fine_tuning_strategy(
    model: Model,
    strategy: str = "full",
    keep_batch_normalization_frozen: bool = True,
    optimizer: tf.keras.optimizers.Optimizer | str | None = None,
    loss: Any | None = None,
    metrics: list[Any] | None = None,
) -> None:
    """Apply fine-tuning strategy to InceptionV3 and recompile model.

    Unfreezes the backbone for representation fine-tuning while keeping all
    backbone BatchNormalization layers non-trainable. Recompiles the model
    after modifying trainability.

    Args:
        model: InceptionV3 Keras Model instance.
        strategy: Fine-tuning strategy (must be 'full').
        keep_batch_normalization_frozen: Whether to keep backbone BN layers frozen.
        optimizer: Optimizer instance or name for recompilation.
        loss: Loss function for recompilation.
        metrics: List of metrics for recompilation.

    Raises:
        ValueError: If strategy is unsupported or BN layers remain trainable when
            keep_batch_normalization_frozen is True.
    """
    if strategy != "full":
        raise ValueError(f"Unsupported fine-tuning strategy for InceptionV3: '{strategy}'. Only 'full' is supported.")

    backbone = None
    for layer in model.layers:
        if isinstance(layer, Model) or "inception_v3" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Could not locate InceptionV3 backbone in model.")

    # Unfreeze backbone
    backbone.trainable = True

    # Maintain all backbone BatchNormalization layers non-trainable
    if keep_batch_normalization_frozen:
        for layer in backbone.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False

    # Verify BN layer trainability
    trainable_bn_count = sum(
        1 for l in backbone.layers if isinstance(l, BatchNormalization) and l.trainable
    )
    if keep_batch_normalization_frozen and trainable_bn_count > 0:
        raise ValueError(
            f"Violation: Found {trainable_bn_count} trainable BatchNormalization layers "
            "in backbone despite keep_batch_normalization_frozen=True."
        )

    # Recompile model after modifying trainability
    if optimizer is not None and loss is not None:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info("Model recompiled after applying InceptionV3 fine-tuning strategy.")


def validate_inceptionv3_strategy(model: Model, config: dict[str, Any]) -> None:
    """Validate that model trainability matches configuration.

    Args:
        model: InceptionV3 Keras Model instance.
        config: Configuration dictionary.
    """
    ft_config = config.get("fine_tuning", {})
    keep_bn_frozen = ft_config.get("keep_batch_normalization_frozen", True)

    backbone = next((l for l in model.layers if isinstance(l, Model) or "inception_v3" in l.name.lower()), None)
    if backbone is not None and keep_bn_frozen:
        trainable_bn = [l.name for l in backbone.layers if isinstance(l, BatchNormalization) and l.trainable]
        if trainable_bn:
            raise ValueError(f"Found trainable backbone BatchNormalization layers: {trainable_bn[:5]}")

    logger.info("InceptionV3 strategy validation passed.")
