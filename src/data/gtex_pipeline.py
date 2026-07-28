"""GTEx Data Pipeline for TensorFlow/Keras.

Prepares a tf.data.Dataset from the GTEx metadata CSVs.
Applies resizing and (optionally) data augmentation.
Outputs raw [0, 255] float32 tensors. Rescaling is expected inside the model.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import tensorflow as tf


def parse_image(
    filename: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read an image, convert to RGB, and resize."""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label


class GTExAugmentation:
    """GTEx Data Augmentation policy."""
    
    def __init__(self, config: dict):
        self.config = config
        layers = []
        
        if config.get("horizontal_flip") and config.get("vertical_flip"):
            layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
        elif config.get("horizontal_flip"):
            layers.append(tf.keras.layers.RandomFlip("horizontal"))
        elif config.get("vertical_flip"):
            layers.append(tf.keras.layers.RandomFlip("vertical"))
            
        if "rotation_factor" in config:
            layers.append(tf.keras.layers.RandomRotation(config["rotation_factor"]))
            
        if "zoom_factor" in config:
            layers.append(tf.keras.layers.RandomZoom(config["zoom_factor"]))
            
        if "contrast_factor" in config:
            layers.append(tf.keras.layers.RandomContrast(config["contrast_factor"]))
            
        height_factor = config.get("translation_height_factor", 0.0)
        width_factor = config.get("translation_width_factor", 0.0)
        if height_factor > 0.0 or width_factor > 0.0:
            layers.append(tf.keras.layers.RandomTranslation(height_factor, width_factor))
            
        self.augmenter = tf.keras.Sequential(layers) if layers else None
        
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.augmenter is not None:
            image = self.augmenter(image, training=True)
            
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image, label


def create_gtex_dataset(
    dataset_dir: Path | str,
    split: str,
    is_training: bool,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: dict | None = None
) -> tf.data.Dataset:
    """Create a tf.data.Dataset for GTEx.
    
    Args:
        dataset_dir: Path to the GTEx dataset root (containing 'metadata').
        split: 'train', 'validation', or 'test'.
        is_training: Whether to shuffle and apply augmentations.
        batch_size: Batch size.
        image_size: Target image dimensions.
        augmentation_config: Dictionary of augmentations (if is_training=True).
        
    Returns:
        tf.data.Dataset producing (images, labels) batches.
    """
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "metadata" / f"{split}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"GTEx metadata CSV not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Resolve absolute paths
    # Assuming image_path in CSV is relative to dataset_dir (e.g., 'train/bladder_0.png')
    df["abs_path"] = df["image_path"].apply(lambda x: str((dataset_dir / x).resolve()))
    
    # Validate a few paths
    for p in df["abs_path"].head(3):
        if not Path(p).exists():
            raise FileNotFoundError(f"Image not found at expected path: {p}")
            
    filepaths = df["abs_path"].values
    labels = df["class_id"].values
    
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)
        
    dataset = dataset.map(
        lambda x, y: parse_image(x, y, image_size), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if is_training and augmentation_config:
        augmenter = GTExAugmentation(augmentation_config)
        dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def validate_batch(dataset: tf.data.Dataset) -> None:
    """Validate that the dataset produces correctly formatted batches."""
    for images, labels in dataset.take(1):
        assert images.dtype == tf.float32, f"Images must be float32, got {images.dtype}"
        assert len(images.shape) == 4, f"Images must be 4D (batch, h, w, c), got {images.shape}"
        assert images.shape[-1] == 3, f"Images must have 3 channels, got {images.shape[-1]}"
        
        batch_min = tf.reduce_min(images).numpy()
        batch_max = tf.reduce_max(images).numpy()
        assert batch_min >= 0.0, f"Image min value < 0: {batch_min}"
        assert batch_max <= 255.0, f"Image max value > 255: {batch_max}"
        
        label_min = tf.reduce_min(labels).numpy()
        label_max = tf.reduce_max(labels).numpy()
        assert label_min >= 0, f"Label min value < 0: {label_min}"
        assert label_max <= 10, f"Label max value > 10: {label_max}"
