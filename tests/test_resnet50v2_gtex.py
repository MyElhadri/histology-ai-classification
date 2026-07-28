import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.models.resnet50v2 import (
    build_resnet50v2_model,
    freeze_resnet50v2_backbone,
    apply_partial_resnet50v2_fine_tuning,
    apply_full_resnet50v2_fine_tuning,
    validate_resnet50v2_trainability
)
from src.data.gtex_pipeline import parse_image, GTExAugmentation


@pytest.fixture
def mock_config():
    return {
        "dataset": {"num_classes": 11, "input_size": [224, 224]},
        "model": {
            "head_config": {
                "type": "article_inspired",
                "dense_1_units": 512,
                "dense_2_units": 128,
                "dropout_rate": 0.30,
                "batch_normalization": True,
                "l2_strength": 0.01
            }
        },
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": True
        }
    }


def test_resnet50v2_architecture_and_output(mock_config):
    model = build_resnet50v2_model(mock_config, weights=None, input_shape=(224, 224, 3))
    
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 11)
    
    # Check Softmax float32
    assert model.layers[-1].activation.__name__ == "softmax"
    assert model.layers[-1].dtype == "float32"


def test_resnet50v2_three_phases(mock_config):
    model = build_resnet50v2_model(mock_config, weights=None)
    
    # Phase 1
    freeze_resnet50v2_backbone(model)
    p1 = validate_resnet50v2_trainability(model, 1)
    
    # Phase 2
    apply_partial_resnet50v2_fine_tuning(model, fraction=0.30)
    p2 = validate_resnet50v2_trainability(model, 2)
    
    # Phase 3
    apply_full_resnet50v2_fine_tuning(model)
    p3 = validate_resnet50v2_trainability(model, 3)
    
    assert p1 == 0
    assert p1 < p2 < p3


def test_gtex_augmentation_output_range(mock_config):
    augmenter = GTExAugmentation(mock_config["augmentation"])
    
    # Create fake image
    img = tf.random.uniform((224, 224, 3), minval=0.0, maxval=255.0, dtype=tf.float32)
    label = tf.constant(1)
    
    aug_img, _ = augmenter(img, label)
    
    assert tf.reduce_min(aug_img) >= 0.0
    assert tf.reduce_max(aug_img) <= 255.0


def test_shared_head_11_classes(mock_config):
    from src.models.heads import build_classification_head
    inputs = tf.keras.Input(shape=(7, 7, 2048))
    outputs = build_classification_head(
        inputs,
        num_classes=11,
        head_config=mock_config["model"]["head_config"]
    )
    model = tf.keras.Model(inputs, outputs)
    assert model.output_shape == (None, 11)
