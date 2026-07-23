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
        d1_units = head_config.get("dense_1_units", 512)
        if d1_units <= 0:
            raise ValueError("units strictly positive required")
        d2_units = head_config.get("dense_2_units", 128)
        if d2_units <= 0:
            raise ValueError("units strictly positive required")
        dr = head_config.get("dropout_rate", 0.30)
        if not (0 <= dr <= 1):
            raise ValueError("dropout strictly between 0 and 1 required")
        l2_str = head_config.get("l2_strength", 0.01)
        if l2_str < 0:
            raise ValueError("L2 strictly positive or zero required")
            
        x = GlobalAveragePooling2D(name="global_average_pooling")(x)
        
        x = Dense(d1_units, name=f"classifier_dense_{d1_units}")(x)
        d1_act = head_config.get("dense_1_activation", "elu")
        if d1_act == "elu":
            x = tf.keras.layers.ELU(name=f"classifier_elu_{d1_units}")(x)
        else:
            x = tf.keras.layers.Activation(d1_act, name=f"classifier_{d1_act}_{d1_units}")(x)
            
        if head_config.get("batch_normalization", False):
            x = tf.keras.layers.BatchNormalization(name="classifier_batch_norm")(x)
            
        if dr > 0:
            x = Dropout(dr, name="classifier_dropout")(x)
            
        reg = tf.keras.regularizers.l2(l2_str) if l2_str > 0 else None
        x = Dense(d2_units, kernel_regularizer=reg, name=f"classifier_dense_{d2_units}")(x)
        d2_act = head_config.get("dense_2_activation", "elu")
        if d2_act == "elu":
            x = tf.keras.layers.ELU(name=f"classifier_elu_{d2_units}")(x)
        else:
            x = tf.keras.layers.Activation(d2_act, name=f"classifier_{d2_act}_{d2_units}")(x)
            
        outputs = Dense(num_classes, activation=head_config.get("output_activation", "softmax"), dtype="float32", name="predictions")(x)
    else:
        raise ValueError(f"Unknown classifier_head type: {head_config.get('type')}")
    
    model = Model(inputs=base_model.input, outputs=outputs, name="densenet121")
    return model


def validate_model_matches_config(model: Model, config: dict) -> None:
    """Validate that the built model's architecture matches the configuration.
    
    Args:
        model: Keras model to validate.
        config: Configuration dictionary.
        
    Raises:
        ValueError: If model architecture does not match the configured head.
    """
    head_config = config.get("model", {}).get("classifier_head") or config.get("classifier_head", {"type": "baseline"})
    req_type = head_config.get("type", "baseline")
    
    layer_names = [l.name for l in model.layers]
    
    # Check for Flatten
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Flatten):
            raise ValueError(f"Configured head '{req_type}' does not match constructed model: Flatten layer detected.")
            
    if req_type == "article_inspired":
        d1_units = head_config.get("dense_1_units", 512)
        d1_layer = next((l for l in model.layers if l.name == f"classifier_dense_{d1_units}"), None)
        if d1_layer is None or not isinstance(d1_layer, Dense) or d1_layer.units != d1_units:
            raise ValueError(f"Configured head '{req_type}' does not match constructed model. Missing or invalid classifier_dense_{d1_units}.")
            
        if head_config.get("batch_normalization", False):
            bn_layer = next((l for l in model.layers if l.name == "classifier_batch_norm"), None)
            if bn_layer is None or not isinstance(bn_layer, tf.keras.layers.BatchNormalization):
                raise ValueError(f"Configured head '{req_type}' does not match constructed model. Missing classifier_batch_norm.")
                
        dr = head_config.get("dropout_rate", 0.30)
        if dr > 0:
            drop_layer = next((l for l in model.layers if l.name == "classifier_dropout"), None)
            if drop_layer is None or not isinstance(drop_layer, Dropout) or abs(drop_layer.rate - dr) > 1e-5:
                raise ValueError(f"Configured head '{req_type}' does not match constructed model. Missing or invalid classifier_dropout (expected rate {dr}).")
                
        d2_units = head_config.get("dense_2_units", 128)
        d2_layer = next((l for l in model.layers if l.name == f"classifier_dense_{d2_units}"), None)
        if d2_layer is None or not isinstance(d2_layer, Dense) or d2_layer.units != d2_units:
            raise ValueError(f"Configured head '{req_type}' does not match constructed model. Missing or invalid classifier_dense_{d2_units}.")
            
        l2_str = head_config.get("l2_strength", 0.01)
        if l2_str > 0:
            reg = d2_layer.kernel_regularizer
            if reg is None or not hasattr(reg, "l2") or abs(float(reg.l2) - l2_str) > 1e-5:
                raise ValueError(f"Configured head '{req_type}' does not match constructed model. classifier_dense_{d2_units} missing expected L2 strength {l2_str}.")
                
        pred_units = config.get("data", {}).get("num_classes", 22)
        pred_layer = next((l for l in model.layers if l.name == "predictions"), None)
        if pred_layer is None or not isinstance(pred_layer, Dense) or pred_layer.units != pred_units:
            raise ValueError(f"Configured head '{req_type}' does not match constructed model. Missing or invalid predictions layer.")

    elif req_type == "baseline":
        forbidden_names = ["classifier_dense_512", "classifier_batch_norm", "classifier_dense_128"]
        found_forbidden = [n for n in forbidden_names if n in layer_names]
        if found_forbidden:
            raise ValueError(f"Configured head '{req_type}' does not match constructed model. Found unexpected layers: {found_forbidden}.")
    else:
        raise ValueError(f"Unknown classifier_head type: {req_type}")



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
        if isinstance(layer, (GlobalAveragePooling2D, Dropout, Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.ELU, tf.keras.layers.Activation)) and layer.name.startswith(("global_average_pooling", "classifier_", "predictions")):
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
