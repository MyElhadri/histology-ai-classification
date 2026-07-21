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
    dropout_rate: float = 0.30
) -> Model:
    """Build a DenseNet121 model.
    
    Args:
        num_classes: Number of output classes.
        input_shape: Input image dimensions.
        weights: Pre-trained weights to load ("imagenet" or None).
        dropout_rate: Dropout rate before the final dense layer.
        
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
    x = GlobalAveragePooling2D()(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
        
    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)
    
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
        if isinstance(layer, (GlobalAveragePooling2D, Dropout, Dense)):
            layer.trainable = True
        else:
            # BatchNormalization should generally remain frozen during fine-tuning
            # to prevent destabilizing the weights.
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = trainable
