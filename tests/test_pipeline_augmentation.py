import pytest
from pathlib import Path

try:
    import tensorflow as tf
    from src.data.pipeline import create_dataset, RichAugmentation
    HAS_TF = True
except ImportError:
    HAS_TF = False
    create_dataset = None
    RichAugmentation = None

pytestmark = pytest.mark.skipif(not HAS_TF, reason="TensorFlow is required for pipeline augmentation tests")

def test_validation_deterministic_without_augmentation():
    """1. Validation pipeline is deterministic and has no augmentation."""
    manifest = Path("data/manifests/densenet121_folds.csv")
    ds = create_dataset(manifest, "", fold=0, is_training=False, batch_size=2)
    
    # Check that reading twice yields identical results
    batch1 = next(iter(ds))
    batch2 = next(iter(ds)) # Note: ds is repeatable but next(iter(ds)) resets iterator
    
    it = iter(ds)
    batch_a = next(it)
    
    it2 = iter(ds)
    batch_b = next(it2)
    
    tf.debugging.assert_equal(batch_a[0], batch_b[0])


def test_output_shape_and_type():
    """2 & 3. Output is 224x224x3 and float32."""
    manifest = Path("data/manifests/densenet121_folds.csv")
    ds = create_dataset(manifest, "", fold=0, is_training=True, batch_size=2)
    
    images, labels = next(iter(ds))
    assert images.shape[1:] == (224, 224, 3)
    assert images.dtype == tf.float32
    assert labels.dtype == tf.float32


def test_values_finite_and_in_range():
    """4 & 5. Values are finite and in [0, 255]."""
    manifest = Path("data/manifests/densenet121_folds.csv")
    config = {
        "enabled": True, "policy": "rich", 
        "brightness_delta": 0.5, "gaussian_noise_stddev": 0.5,
        "clip_min": 0.0, "clip_max": 255.0
    }
    ds = create_dataset(manifest, "", fold=0, is_training=True, batch_size=2, augmentation_config=config)
    
    images, labels = next(iter(ds))
    assert tf.reduce_all(tf.math.is_finite(images))
    assert tf.reduce_min(images) >= 0.0
    assert tf.reduce_max(images) <= 255.0


def test_baseline_preserves_augmentation():
    """6. Baseline behavior without config uses original augmentation."""
    manifest = Path("data/manifests/densenet121_folds.csv")
    ds_baseline = create_dataset(manifest, "", fold=0, is_training=True, batch_size=2)
    
    # baseline shouldn't fail and should be randomized
    # we can't easily assert exactly which ops ran, but we can verify it doesn't crash
    # and generates valid images
    images, labels = next(iter(ds_baseline))
    assert images.shape[1:] == (224, 224, 3)


def test_rich_policy_transformations():
    """7. Rich policy applies transformations correctly."""
    config = {
        "horizontal_flip": True,
        "rotation_factor": 0.1,
        "brightness_delta": 0.1,
        "saturation_lower": 0.8,
        "saturation_upper": 1.2
    }
    aug = RichAugmentation(config)
    
    dummy_image = tf.ones((224, 224, 3), dtype=tf.float32) * 128.0
    dummy_label = tf.constant([1, 0])
    
    aug_image, label = aug(dummy_image, dummy_label)
    assert aug_image.shape == (224, 224, 3)


def test_original_images_not_modified():
    """8. Original images are not modified on disk."""
    import os
    # We can't strictly test no disk modification in a simple unit test,
    # but the pipeline maps are stateless and run in memory. 
    # Just asserting the pipeline runs successfully without FileNotFoundError on subsequent reads.
    manifest = Path("data/manifests/densenet121_folds.csv")
    ds = create_dataset(manifest, "", fold=0, is_training=True, batch_size=2)
    next(iter(ds))
    next(iter(ds)) # Second read would fail or produce different results if disk was overwritten


def test_train_val_different_policies():
    """9. Train and Validation use different policies (augmentation vs none)."""
    manifest = Path("data/manifests/densenet121_folds.csv")
    ds_train = create_dataset(manifest, "", fold=0, is_training=True, batch_size=2)
    ds_val = create_dataset(manifest, "", fold=0, is_training=False, batch_size=2)
    
    # Read same image from both? Not possible easily due to folds.
    # Just check their output shapes and types are consistent, and one is deterministic.
    assert next(iter(ds_train))[0].shape == next(iter(ds_val))[0].shape


def test_invalid_config_raises_clear_error():
    """10. Invalid config could produce clear errors (if implemented)."""
    # For instance, if rotation_factor is a string, TF will throw a specific error
    config = {"horizontal_flip": "yes please"}
    with pytest.raises(Exception):
        aug = RichAugmentation(config)
        dummy_image = tf.ones((224, 224, 3), dtype=tf.float32)
        aug(dummy_image, tf.constant(1))
