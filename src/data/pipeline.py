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


class RichAugmentation:
    """Rich Data Augmentation policy for Experience A."""
    
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
            
        self.spatial_and_contrast = tf.keras.Sequential(layers) if layers else None
        
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.spatial_and_contrast is not None:
            image = self.spatial_and_contrast(image, training=True)
            
        if "brightness_delta" in self.config:
            max_delta = self.config["brightness_delta"] * 255.0
            image = tf.image.random_brightness(image, max_delta=max_delta)
            
        if "saturation_lower" in self.config and "saturation_upper" in self.config:
            image = tf.image.random_saturation(
                image, 
                lower=self.config["saturation_lower"], 
                upper=self.config["saturation_upper"]
            )
            
        if "gaussian_noise_stddev" in self.config:
            stddev = self.config["gaussian_noise_stddev"] * 255.0
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
            image = image + noise
            
        clip_min = self.config.get("clip_min", 0.0)
        clip_max = self.config.get("clip_max", 255.0)
        image = tf.clip_by_value(image, clip_min, clip_max)
        
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
    augmentation_config: dict | None = None,
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
        if augmentation_config and augmentation_config.get("enabled") and augmentation_config.get("policy") == "rich":
            aug_fn = RichAugmentation(augmentation_config)
            dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_square_root_dataset(
    manifest_path: Path | str,
    fold: int,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (224, 224),
    num_classes: int = 22,
    project_root: Path = _PROJECT_ROOT,
    augmentation_config: dict | None = None,
    seed: int | None = 42,
    frequency_power: float = -0.5,
) -> Tuple[tf.data.Dataset, int, dict[int, float], dict[int, int]]:
    """Create a square-root sampled tf.data.Dataset for Phase 3 (cRT).

    Samples classes with probability proportional to n_c ** (1 + frequency_power),
    where for frequency_power = -0.5, sampling probability is proportional to sqrt(n_c).
    Total epoch size is constrained to the original number of training images N.

    Returns:
        (dataset, original_total_images, class_probabilities_dict, original_counts_dict)
    """
    df = pd.read_csv(manifest_path)
    train_df = df[df["fold"] != fold]

    class_counts = train_df["class_id"].value_counts().to_dict()
    original_counts = {c: int(class_counts.get(c, 0)) for c in range(num_classes)}
    N = len(train_df)

    raw_weights = []
    for c in range(num_classes):
        cnt = original_counts[c]
        if cnt > 0:
            w = float(cnt ** (1.0 + frequency_power))
        else:
            w = 0.0
        raw_weights.append(w)

    total_w = sum(raw_weights)
    class_probs = [w / total_w if total_w > 0 else 0.0 for w in raw_weights]
    class_probs_dict = {c: class_probs[c] for c in range(num_classes)}

    class_datasets = []
    for c in range(num_classes):
        c_df = train_df[train_df["class_id"] == c]
        if len(c_df) > 0:
            c_paths = [str(resolve_image_path(p, project_root)) for p in c_df["image_path"].values]
            c_labels = tf.keras.utils.to_categorical([c] * len(c_paths), num_classes=num_classes)
            ds_c = tf.data.Dataset.from_tensor_slices((c_paths, c_labels))
            ds_c = ds_c.shuffle(buffer_size=len(c_paths), seed=seed)
            ds_c = ds_c.repeat()
        else:
            dummy_path = [""]
            dummy_label = tf.zeros((1, num_classes))
            ds_c = tf.data.Dataset.from_tensor_slices((dummy_path, dummy_label)).repeat()
        class_datasets.append(ds_c)

    dataset = tf.data.Dataset.sample_from_datasets(class_datasets, weights=class_probs, seed=seed)
    dataset = dataset.take(N)

    dataset = dataset.map(
        lambda x, y: parse_image(x, y, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augmentation_config and augmentation_config.get("enabled") and augmentation_config.get("policy") == "rich":
        aug_fn = RichAugmentation(augmentation_config)
        dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, N, class_probs_dict, original_counts

