"""ResNet50V2 model architecture, preprocessing, and three-phase fine-tuning utilities.

This module implements ImageNet pre-trained ResNet50V2 for histology classification,
including:
- Built-in [0, 255] float32 to [-1, 1] preprocessing via a serializable Rescaling layer
- Explicit inference-mode BatchNormalization (backbone called with training=False)
- Three-phase fine-tuning: head training, partial unfreezing (30%), full unfreezing
- All BatchNormalization layers kept frozen during fine-tuning phases 2 and 3
- Reusable article-inspired classification head
"""

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Rescaling
from tensorflow.keras.models import Model

from src.models.heads import build_classification_head

logger = logging.getLogger(__name__)

PREPROCESSING_LAYER_NAME = "resnet50v2_preprocessing"


def verify_no_double_preprocessing(model: Model) -> None:
    rescaling_layers = [
        layer for layer in model.layers
        if isinstance(layer, Rescaling)
    ]
    if not rescaling_layers:
        raise ValueError(
            "ResNet50V2 model is missing internal "
            f"Rescaling(scale=1.0/127.5, offset=-1.0) layer named '{PREPROCESSING_LAYER_NAME}'."
        )
    if len(rescaling_layers) > 1:
        raise ValueError("Double preprocessing detected!")
        
    rescaling = rescaling_layers[0]
    expected_scale = 1.0 / 127.5
    if abs(float(rescaling.scale) - expected_scale) > 1e-6 or abs(float(rescaling.offset) - (-1.0)) > 1e-6:
        raise ValueError("Rescaling parameters are incorrect. Expected scale=1.0/127.5, offset=-1.0")


def validate_preprocessing_values() -> None:
    rescaling = Rescaling(scale=1.0 / 127.5, offset=-1.0)
    test_values = [0.0, 127.5, 255.0]
    expected_values = [-1.0, 0.0, 1.0]

    for val, expected in zip(test_values, expected_values):
        tensor = tf.constant([[[val, val, val]]], dtype=tf.float32)
        result = float(rescaling(tensor).numpy()[0, 0, 0])
        assert abs(result - expected) < 1e-5


def validate_dataset_preprocessing(dataset: tf.data.Dataset, config: dict[str, Any]) -> None:
    for section_name in ["data", "augmentation", "preprocessing"]:
        section = config.get(section_name, {})
        if not isinstance(section, dict):
            continue
        if section.get("external_divide_by_255") is True:
            raise ValueError(f"Illegal external_divide_by_255=True in {section_name}")
        if section.get("external_preprocess_input") is True:
            raise ValueError(f"Illegal external_preprocess_input=True in {section_name}")

    for batch, _ in dataset.take(1):
        if batch.dtype != tf.float32:
            raise ValueError(f"Expected float32 inputs, got {batch.dtype}")
        batch_min = tf.reduce_min(batch).numpy()
        batch_max = tf.reduce_max(batch).numpy()
        if batch_min < 0.0 or batch_max > 255.0:
            raise ValueError(f"Inputs must be in [0, 255]. Got min={batch_min}, max={batch_max}")


def build_resnet50v2_model(
    config: dict[str, Any],
    weights: str | None = "imagenet",
    input_shape: tuple[int, int, int] = (224, 224, 3)
) -> Model:
    num_classes = config.get("dataset", {}).get("num_classes", 11)
    
    inputs = tf.keras.Input(shape=input_shape, name="images_input")
    x = Rescaling(scale=1.0 / 127.5, offset=-1.0, name=PREPROCESSING_LAYER_NAME)(inputs)
    
    backbone = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    
    # Freeze backbone initially
    backbone.trainable = False
    
    # Always call backbone with training=False to freeze BN during Phase 1
    features = backbone(x, training=False)
    
    head_config = config.get("model", {}).get("head_config")
    
    outputs = build_classification_head(
        x=features,
        num_classes=num_classes,
        head_config=head_config,
        dropout_rate=0.30
    )
    
    model = Model(inputs=inputs, outputs=outputs, name="ResNet50V2_GTEx")
    
    # Validation checks
    verify_no_double_preprocessing(model)
    validate_preprocessing_values()
    
    return model


def _find_backbone(model: Model) -> Model:
    for layer in model.layers:
        if isinstance(layer, Model) and "resnet50v2" in layer.name.lower():
            return layer
    raise ValueError("ResNet50V2 backbone not found in model.")


def freeze_resnet50v2_batch_normalization(model: Model) -> None:
    backbone = _find_backbone(model)
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def freeze_resnet50v2_backbone(model: Model) -> None:
    backbone = _find_backbone(model)
    backbone.trainable = False
    logger.info("Phase 1: Backbone totally frozen.")


def apply_partial_resnet50v2_fine_tuning(model: Model, fraction: float = 0.30) -> None:
    backbone = _find_backbone(model)
    backbone.trainable = True
    freeze_resnet50v2_batch_normalization(model)
    
    trainable_layers = [layer for layer in backbone.layers if not isinstance(layer, BatchNormalization)]
    num_trainable = int(len(trainable_layers) * fraction)
    
    for layer in trainable_layers[:-num_trainable]:
        layer.trainable = False
        
    for layer in trainable_layers[-num_trainable:]:
        layer.trainable = True
        
    logger.info(f"Phase 2: Unfrozen last {fraction*100}% of non-BN layers ({num_trainable} layers).")


def apply_full_resnet50v2_fine_tuning(model: Model) -> None:
    backbone = _find_backbone(model)
    backbone.trainable = True
    
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            
    logger.info("Phase 3: Unfrozen ALL non-BN layers.")


def validate_resnet50v2_trainability(model: Model, phase: int) -> int:
    backbone = _find_backbone(model)
    
    # 1. Verify BN is always frozen
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            assert not layer.trainable, f"BatchNormalization layer {layer.name} is trainable!"
            
    # Count trainable weights in backbone
    trainable_weights_count = sum(tf.size(w).numpy() for w in backbone.trainable_weights)
    
    if phase == 1:
        assert trainable_weights_count == 0, "Phase 1 expects 0 trainable backbone weights."
    elif phase == 2:
        assert trainable_weights_count > 0, "Phase 2 expects >0 trainable backbone weights."
    elif phase == 3:
        assert trainable_weights_count > 0, "Phase 3 expects >0 trainable backbone weights."
        
    return int(trainable_weights_count)
