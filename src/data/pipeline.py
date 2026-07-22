"""Data pipeline for TensorFlow/Keras.

Prepares a tf.data.Dataset from the folds manifest.
Applies resizing, normalization, and (optionally) data augmentation.

Path convention:
    image_path in the manifest is relative to the project root, e.g.:
        data/raw/nuinsseg_human_22_original/human_tissue_image-brain/img.png

    The project root is detected as the parent of the 'src/' directory.
    If image_path is absolute, it is used directly.

Note on DenseNet121 preprocessing:
    densenet.preprocess_input is already applied INSIDE densenet121.py
    (as an Input layer in the Keras functional graph).
    Therefore, this pipeline outputs raw [0, 255] float32 tensors
    and must NOT apply preprocess_input again.
"""

from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
# src/data/pipeline.py lives at  <project_root>/src/data/pipeline.py
# Therefore .parent.parent.parent gives the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_image_path(image_path: str | Path, project_root: Path = _PROJECT_ROOT) -> Path:
    """Resolve an image path to an absolute filesystem path.

    Args:
        image_path: Either an absolute path or a path relative to *project_root*.
        project_root: Absolute path of the project root directory.

    Returns:
        Absolute Path to the image file.

    Raises:
        FileNotFoundError: If the resolved path does not exist on disk.
    """
    p = Path(image_path)
    if p.is_absolute():
        resolved = p
    else:
        resolved = (project_root / p).resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Image not found: {resolved}\n"
            f"  image_path supplied : {image_path}\n"
            f"  project_root used   : {project_root}"
        )
    return resolved


import pandas as pd
import tensorflow as tf


def parse_image(
    filename: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read an image, convert to RGB, and resize.

    Handles both RGB (3-channel) and RGBA (4-channel) images gracefully
    by forcing decode to 3 channels. Pixel values are kept as float32
    in [0, 255] — DenseNet121's preprocess_input is applied inside the
    model graph and must NOT be repeated here.
    """
    image_string = tf.io.read_file(filename)
    # channels=3 forces RGB decode; RGBA alpha channel is discarded automatically
    image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label


def data_augmentation(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentations — used ONLY during training."""
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
    num_classes: int = 22,
    project_root: Path = _PROJECT_ROOT,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset for a specific fold.

    Args:
        manifest_path: Path to densenet121_folds.csv.
        dataset_root: Unused — kept for API compatibility.
            Image paths are resolved relative to *project_root* instead.
        fold: The fold number (0-4) to use for validation.
        is_training: If True, uses all folds EXCEPT the specified fold,
                     and applies augmentation.
                     If False, uses ONLY the specified fold, no augmentation.
        batch_size: Batch size.
        image_size: Target image dimensions (height, width).
        num_classes: Number of classes (for one-hot encoding).
        project_root: Root of the project — used for path resolution.

    Returns:
        A tf.data.Dataset instance.
    """
    df = pd.read_csv(manifest_path)

    if is_training:
        subset_df = df[df["fold"] != fold]
    else:
        subset_df = df[df["fold"] == fold]

    # Resolve each image path relative to the project root.
    # Raises FileNotFoundError early if any path is invalid.
    file_paths = []
    for raw_path in subset_df["image_path"].values:
        resolved = resolve_image_path(raw_path, project_root)
        file_paths.append(str(resolved))

    # Extract labels and one-hot encode
    labels = subset_df["class_id"].values
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    # Build dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_onehot))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda x, y: parse_image(x, y, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training:
        dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
