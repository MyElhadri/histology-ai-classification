"""Classification head builders for Deep Learning histology models.

This module provides reusable classification head architectures such as
the baseline GlobalAveragePooling head and the article-inspired multi-layer head.
Note: This module is used by new models (e.g. EfficientNetV2B0). DenseNet121
Expérience D retains its original internal implementation in densenet121.py.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D


def build_baseline_head(
    x: tf.Tensor,
    num_classes: int,
    dropout_rate: float = 0.30
) -> tf.Tensor:
    """Build baseline classification head.

    Args:
        x: Input feature tensor from backbone.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability.

    Returns:
        Output predictions tensor.
    """
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions"
    )(x)
    return outputs


def build_article_inspired_head(
    x: tf.Tensor,
    num_classes: int,
    head_config: dict
) -> tf.Tensor:
    """Build article-inspired classification head.

    Args:
        x: Input feature tensor from backbone.
        num_classes: Number of output classes.
        head_config: Dictionary containing head parameters.

    Returns:
        Output predictions tensor.
    """
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

    if head_config.get("batch_normalization", True):
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

    outputs = Dense(
        num_classes,
        activation=head_config.get("output_activation", "softmax"),
        dtype="float32",
        name="predictions"
    )(x)
    return outputs


def build_classification_head(
    x: tf.Tensor,
    num_classes: int,
    head_config: dict | None = None,
    dropout_rate: float = 0.30
) -> tf.Tensor:
    """Dispatch to appropriate classification head builder based on config.

    Args:
        x: Input feature tensor from backbone.
        num_classes: Number of output classes.
        head_config: Configuration dictionary for classifier head.
        dropout_rate: Default dropout rate if head_config is minimal.

    Returns:
        Output predictions tensor.
    """
    if head_config is None or head_config.get("type", "baseline") == "baseline":
        return build_baseline_head(x, num_classes, dropout_rate=dropout_rate)
    elif head_config.get("type") == "article_inspired":
        return build_article_inspired_head(x, num_classes, head_config=head_config)
    else:
        raise ValueError(f"Unknown classifier_head type: {head_config.get('type')}")
