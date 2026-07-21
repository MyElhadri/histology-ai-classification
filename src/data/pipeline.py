"""Data pipeline for TensorFlow/Keras.

Prepares a tf.data.Dataset from the folds manifest.
Applies resizing, normalization, and (optionally) data augmentation.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import tensorflow as tf


def parse_image(
    filename: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read an image from a file and resize it."""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    # DenseNet expects inputs to be scaled using its own preprocess_input,
    # or typically normalized to [0, 1] if not using the built-in function.
    # We will normalize to [0, 1] here for simplicity, or we can rely on
    # tf.keras.applications.densenet.preprocess_input inside the model.
    # We'll output [0, 255] float32 here and let the model handle preprocessing.
    return image, label


def data_augmentation(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply random data augmentations to the image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


def create_dataset(
    manifest_path: Path | str,
    dataset_root: Path | str,
    fold: int,
    is_training: bool,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (224, 224),
    num_classes: int = 22
) -> tf.data.Dataset:
    """Create a tf.data.Dataset for a specific fold.
    
    Args:
        manifest_path: Path to densenet121_folds.csv.
        dataset_root: Root directory of the dataset.
        fold: The fold number (0-4) to use for validation.
        is_training: If True, uses all folds EXCEPT the specified fold, and applies augmentation.
                     If False, uses ONLY the specified fold, without augmentation.
        batch_size: Batch size.
        image_size: Target image dimensions.
        num_classes: Number of classes (for one-hot encoding).
        
    Returns:
        A tf.data.Dataset instance.
    """
    df = pd.read_csv(manifest_path)
    
    if is_training:
        # Use all folds except the validation fold
        subset_df = df[df["fold"] != fold]
    else:
        # Use only the validation fold
        subset_df = df[df["fold"] == fold]
        
    # Construct absolute paths
    root_path = Path(dataset_root).resolve()
    file_paths = [str(root_path / path) for path in subset_df["image_path"].values]
    
    # Extract labels and one-hot encode
    labels = subset_df["class_id"].values
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_onehot))
    
    # Shuffle only for training
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)
        
    # Read and resize images
    dataset = dataset.map(
        lambda x, y: parse_image(x, y, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Augment only for training
    if is_training:
        dataset = dataset.map(
            data_augmentation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
