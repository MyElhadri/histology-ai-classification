"""DenseNet121 architecture for tissue classification.

Builds a DenseNet121 model with transfer learning (ImageNet weights)
and a custom classification head.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_densenet121(
    num_classes: int = 22,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
    dropout_rate: float = 0.30,
    head_config: dict | None = None
) -> Model:
    """Build a DenseNet121 model.
    
    Args:
        num_classes: Number of output classes.
        input_shape: Input image dimensions.
        weights: Pre-trained weights to load ("imagenet" or None).
        dropout_rate: Dropout rate before the final dense layer (used if baseline).
        head_config: Dictionary containing the head configuration.
        
    Returns:
        A Keras Model.
    """
    # Define input tensor
    inputs = tf.keras.Input(shape=input_shape)
    
    # Preprocess inputs for DenseNet
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    
    # Load base model
    base_model = DenseNet121(
        include_top=False,
        weights=weights,
        input_tensor=x,
        pooling=None
    )
    
    # Add classification head
    x = base_model.output
    
    if head_config is None or head_config.get("type", "baseline") == "baseline":
        x = GlobalAveragePooling2D(name="global_average_pooling")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name="classifier_dropout")(x)
        outputs = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    elif head_config.get("type") == "article_inspired":
        if head_config.get("dense_1_units", 0) <= 0 or head_config.get("dense_2_units", 0) <= 0:
            raise ValueError("units strictly positive required")
        dr = head_config.get("dropout_rate", 0.30)
        if not (0 <= dr <= 1):
            raise ValueError("dropout strictly between 0 and 1 required")
        l2_str = head_config.get("l2_strength", 0.0)
        if l2_str < 0:
            raise ValueError("L2 strictly positive or zero required")
            
        x = GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = Dense(head_config["dense_1_units"], activation=head_config.get("dense_1_activation", "elu"), name=f"classifier_dense_{head_config['dense_1_units']}")(x)
        if head_config.get("batch_normalization", False):
            x = tf.keras.layers.BatchNormalization(name="classifier_batch_norm")(x)
        if dr > 0:
            x = Dropout(dr, name="classifier_dropout")(x)
        
        reg = tf.keras.regularizers.l2(l2_str) if l2_str > 0 else None
        x = Dense(head_config["dense_2_units"], activation=head_config.get("dense_2_activation", "elu"), kernel_regularizer=reg, name=f"classifier_dense_{head_config['dense_2_units']}")(x)
        outputs = Dense(num_classes, activation=head_config.get("output_activation", "softmax"), dtype="float32", name="predictions")(x)
    else:
        raise ValueError(f"Unknown classifier_head type: {head_config.get('type')}")
    
    model = Model(inputs=base_model.input, outputs=outputs, name="densenet121")
    return model


def set_trainable_layers(model: Model, trainable: bool = False) -> None:
    """Freeze or unfreeze the base model layers.
    
    Args:
        model: The Keras model.
        trainable: True to unfreeze the base, False to freeze it (keeping only the head trainable).
    """
    # Find the base model (all layers except the head)
    # The last few layers are GlobalAveragePooling, Dropout, Dense
    # We want to freeze/unfreeze the DenseNet121 layers.
    for layer in model.layers:
        # Skip custom head layers based on their type
        if isinstance(layer, (GlobalAveragePooling2D, Dropout, Dense, tf.keras.layers.BatchNormalization)) and layer.name.startswith(("global_average_pooling", "classifier_", "predictions")):
            # Head layers are always trainable during phase 1, except maybe BatchNorm should be treated carefully
            # Actually, standard head training unfreezes head layers.
            layer.trainable = True
        else:
            # BatchNormalization should generally remain frozen during fine-tuning
            # to prevent destabilizing the weights.
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = trainable
