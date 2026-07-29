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

def test_sparse_labels_train_on_batch(mock_config):
    model = build_resnet50v2_model(mock_config, weights=None, input_shape=(224, 224, 3))
    
    # Emulate the Kaggle categorical error
    model_categorical = build_resnet50v2_model(mock_config, weights=None, input_shape=(224, 224, 3))
    model_categorical.compile(
        optimizer="adam",
        loss="categorical_crossentropy"
    )
    
    x = tf.random.normal((2, 224, 224, 3))
    y = tf.constant([0, 10], dtype=tf.int32)
    
    with pytest.raises(ValueError, match="Arguments `target` and `output` must have the same rank"):
        model_categorical.train_on_batch(x, y)
        
    # Test the correct sparse approach
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy")
        ]
    )
    
    # Train on batch
    loss, acc, top3 = model.train_on_batch(x, y)
    
    assert not tf.math.is_nan(loss)
    assert not tf.math.is_nan(acc)
    assert not tf.math.is_nan(top3)

def test_sparse_config():
    import yaml
    config_path = Path("configs/experiments/resnet50v2_gtex_11_exp_a.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    assert config["training"]["loss"] == "sparse_categorical_crossentropy"
    assert config["dataset"]["num_classes"] == 11

def test_pipeline_labels_shape(mock_config, tmp_path):
    from src.data.gtex_pipeline import create_gtex_dataset
    import pandas as pd
    
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    
    # Create fake mapping
    class_mapping = {
        "label_to_index": {"bladder": 0, "brain": 1, "testis": 10},
        "index_to_label": {"0": "bladder", "1": "brain", "10": "testis"},
        "class_weights": {"bladder": 1.0, "brain": 1.0, "testis": 1.0}
    }
    with open(metadata_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f)
        
    # Create fake images and CSV
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    
    import numpy as np
    from PIL import Image
    for i, cls in enumerate(["bladder", "brain", "testis"]):
        img_path = train_dir / f"{cls}_0.png"
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(img_path)
        
    df = pd.DataFrame([
        {"image_path": "train/bladder_0.png", "label": "bladder", "label_index": 0},
        {"image_path": "train/brain_0.png", "label": "brain", "label_index": 1},
        {"image_path": "train/testis_0.png", "label": "testis", "label_index": 10},
    ])
    df.to_csv(metadata_dir / "train.csv", index=False)
    
    ds = create_gtex_dataset(tmp_path, "train", is_training=False, batch_size=2)
    for images, labels in ds.take(1):
        assert labels.ndim == 1
        assert labels.shape[0] == 2
        assert labels.dtype in [tf.int32, tf.int64]
        assert tf.reduce_min(labels) >= 0
        assert tf.reduce_max(labels) <= 10
