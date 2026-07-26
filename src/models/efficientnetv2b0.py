"""EfficientNetV2B0 model architecture and training utilities.

This module implements the ImageNet pre-trained EfficientNetV2B0 model
for histology classification, with built-in [0, 255] float32 preprocessing
and support for two-phase fine-tuning with frozen BatchNormalization layers.
"""

import logging
from typing import Any
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Rescaling
from tensorflow.keras.models import Model

from src.models.heads import build_classification_head

logger = logging.getLogger(__name__)


def verify_no_double_preprocessing(model: Model) -> None:
    """Verify that model uses internal preprocessing and has no external rescaling.

    EfficientNetV2B0(include_preprocessing=True) includes internal rescaling
    and normalization layers. Adding external rescaling would cause double
    preprocessing and degrade performance.

    Args:
        model: Keras Model instance to inspect.

    Raises:
        ValueError: If external rescaling/preprocessing is detected or internal
            preprocessing is missing.
    """
    def _find_true_backbone(obj):
        if hasattr(obj, "layers"):
            names = [l.name.lower() for l in obj.layers]
            if any("rescaling" in n for n in names) and any("normalization" in n for n in names):
                return obj
            for sub in obj.layers:
                res = _find_true_backbone(sub)
                if res is not None:
                    return res
        return None

    backbone = _find_true_backbone(model)
    if backbone is None:
        raise ValueError(
            "Could not locate EfficientNetV2B0 backbone or missing internal rescaling/normalization layers. "
            "Ensure include_preprocessing=True."
        )

    # Check internal preprocessing exists in backbone
    backbone_layer_names = [l.name for l in backbone.layers]
    has_rescaling = any("rescaling" in name.lower() for name in backbone_layer_names)
    has_norm = any("normalization" in name.lower() for name in backbone_layer_names)
    if not (has_rescaling and has_norm):
        raise ValueError(
            "EfficientNetV2B0 backbone is missing internal rescaling/normalization layers. "
            "Ensure include_preprocessing=True."
        )

    # Check against external preprocessing in outer model layers recursively
    def _check_external_layers(layers, ignore_obj):
        for layer in layers:
            if layer is ignore_obj:
                continue
            if isinstance(layer, Rescaling):
                raise ValueError(
                    f"Double preprocessing detected! Found external Rescaling layer '{layer.name}' "
                    f"outside EfficientNetV2B0 backbone."
                )
            if any(term in layer.name.lower() for term in ["rescale", "preprocess", "divide"]):
                raise ValueError(
                    f"Suspicious external preprocessing layer detected: '{layer.name}'. "
                    f"EfficientNetV2B0 expects float32 pixels in range [0, 255] directly."
                )
            if hasattr(layer, "layers") and layer is not ignore_obj:
                _check_external_layers(layer.layers, ignore_obj)

    _check_external_layers(model.layers, backbone)
    logger.info("Verified: EfficientNetV2B0 internal preprocessing present; no external double preprocessing found.")


def validate_dataset_preprocessing(dataset: tf.data.Dataset, config: dict[str, Any]) -> None:
    """Validate real dataset batches to ensure float32 [0, 255] input tensors.

    Args:
        dataset: tf.data.Dataset instance to inspect.
        config: Configuration dictionary to inspect for forbidden rescaling settings.

    Raises:
        ValueError: If input tensor dtype, range, or config violates [0, 255] float32 requirements.
    """
    # Check config for forbidden rescaling parameters
    for section_name in ["data", "augmentation", "preprocessing"]:
        section = config.get(section_name, {})
        if not isinstance(section, dict):
            continue
        rescale_val = section.get("rescale")
        if rescale_val is not None and (rescale_val == 1/255 or rescale_val <= 0.01):
            raise ValueError(
                f"Configuration section '{section_name}' contains illegal rescale={rescale_val}. "
                f"EfficientNetV2B0 expects [0, 255] inputs."
            )
        norm_val = section.get("normalization")
        if norm_val == [0, 1] or norm_val == "[0, 1]":
            raise ValueError(
                f"Configuration section '{section_name}' contains illegal normalization={norm_val}."
            )

    # Inspect a real batch from the dataset
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
            f"This indicates double preprocessing or unexpected [0, 1] scaling! "
            f"EfficientNetV2B0 expects native [0, 255] float32 inputs."
        )

    logger.info(
        f"Dataset preprocessing successfully validated: dtype={images.dtype}, "
        f"min={min_val:.2f}, max={max_val:.2f}"
    )


def build_efficientnetv2b0(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
    dropout_rate: float = 0.30,
    head_config: dict[str, Any] | None = None
) -> Model:
    """Build EfficientNetV2B0 classification model.

    Constructs the model using explicit training=False on the backbone call
    to ensure BatchNormalization layers operate in inference mode during both
    head training and representation fine-tuning.

    Args:
        num_classes: Number of target classification classes.
        input_shape: Image input dimensions (H, W, C).
        weights: Pre-trained weights to load (e.g., 'imagenet' or None).
        dropout_rate: Classifier dropout rate if baseline head is used.
        head_config: Configuration dictionary for classifier head architecture.

    Returns:
        Compiled Keras Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        include_preprocessing=True
    )

    # Per mandatory requirement: call backbone explicitly with training=False
    x = backbone(inputs, training=False)

    outputs = build_classification_head(
        x=x,
        num_classes=num_classes,
        head_config=head_config,
        dropout_rate=dropout_rate
    )

    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetV2B0")

    verify_no_double_preprocessing(model)
    return model


def apply_efficientnet_fine_tuning_strategy(
    model: Model,
    strategy: str = "full",
    keep_batch_normalization_frozen: bool = True,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    loss: Any | None = None,
    metrics: list[Any] | None = None
) -> None:
    """Apply fine-tuning strategy to EfficientNetV2B0 and recompile model.

    Unfreezes the backbone for representation fine-tuning while keeping all
    backbone BatchNormalization layers non-trainable. Recompiles the model
    after modifying trainability as required by Keras and user specifications.

    Args:
        model: EfficientNetV2B0 Keras Model instance.
        strategy: Fine-tuning strategy (must be 'full').
        keep_batch_normalization_frozen: Whether to keep backbone BN layers frozen.
        optimizer: Optimizer instance for recompilation.
        loss: Loss function for recompilation.
        metrics: List of metrics for recompilation.

    Raises:
        ValueError: If strategy is unsupported or BN layers remain trainable when
            keep_batch_normalization_frozen is True.
    """
    if strategy != "full":
        raise ValueError(f"Unsupported fine-tuning strategy for EfficientNetV2B0: '{strategy}'. Only 'full' is supported.")

    backbone = None
    for layer in model.layers:
        if isinstance(layer, Model) or "efficientnetv2" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Could not locate EfficientNetV2B0 backbone in model.")

    # Set backbone trainable
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

    # Recompile model after modifying trainability per mandatory requirement
    if optimizer is not None and loss is not None:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info("Model recompiled after applying fine-tuning strategy.")


def validate_efficientnet_strategy(model: Model, config: dict[str, Any]) -> None:
    """Validate that model trainability matches configuration.

    Args:
        model: EfficientNetV2B0 Keras Model instance.
        config: Configuration dictionary.
    """
    ft_config = config.get("fine_tuning", {})
    keep_bn_frozen = ft_config.get("keep_batch_normalization_frozen", True)

    backbone = next((l for l in model.layers if isinstance(l, Model) or "efficientnetv2" in l.name.lower()), None)
    if backbone is not None and keep_bn_frozen:
        trainable_bn = [l.name for l in backbone.layers if isinstance(l, BatchNormalization) and l.trainable]
        if trainable_bn:
            raise ValueError(f"Found trainable backbone BatchNormalization layers: {trainable_bn[:5]}")
    logger.info("EfficientNetV2B0 strategy validation passed.")
