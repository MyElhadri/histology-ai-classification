"""Build a dataset manifest from a directory-structured image dataset.

Scans a root directory where each subdirectory represents a class and
contains image files. Produces:
  - dataset_manifest.csv — one row per image with metadata
  - class_mapping.json   — deterministic class_name → class_id mapping

The manifest records relative paths (relative to dataset_root) so it
remains portable across machines and environments (local, Colab, etc.).

Usage (as a module):
    from src.data.build_manifest import build_manifest
    manifest_df, class_mapping = build_manifest(dataset_root, output_dir)

Author: Yassine
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# Image extensions considered valid (lowercase).
VALID_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _compute_file_hash(filepath: Path, algorithm: str = "md5") -> str:
    """Compute a hex-digest hash of a file's content.

    Args:
        filepath: Absolute path to the file.
        algorithm: Hash algorithm name (default ``md5``).

    Returns:
        Hex-digest string.
    """
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_image_metadata(filepath: Path) -> dict[str, Any] | None:
    """Read basic metadata from an image file without modifying it.

    Returns ``None`` if the image cannot be opened (corrupted / unreadable).
    """
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            # Number of channels: RGB=3, L=1, RGBA=4, etc.
            channels = len(img.getbands())
        return {"width": width, "height": height, "channels": channels}
    except Exception as exc:  # noqa: BRD001
        logger.warning("Cannot read image %s: %s", filepath, exc)
        return None


def _discover_classes(dataset_root: Path) -> list[str]:
    """Return a sorted list of subdirectory names (class labels).

    Sorting ensures a deterministic class order regardless of OS.
    """
    classes = sorted(
        d.name for d in dataset_root.iterdir() if d.is_dir()
    )
    if not classes:
        raise FileNotFoundError(
            f"No subdirectories found in {dataset_root}. "
            "Expected one directory per class."
        )
    return classes


def _build_class_mapping(class_names: list[str]) -> dict[str, int]:
    """Create a deterministic class_name → class_id mapping.

    The mapping is based on alphabetical sorting so that both
    DenseNet121 and ResNet50V2 share the exact same IDs.
    """
    return {name: idx for idx, name in enumerate(class_names)}


def build_manifest(
    dataset_root: Path,
    output_dir: Path,
    *,
    source_label: str = "original",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Scan *dataset_root* and produce the dataset manifest and class mapping.

    Args:
        dataset_root: Path to the directory containing one sub-folder per class.
        output_dir: Directory where ``dataset_manifest.csv`` and
            ``class_mapping.json`` will be saved.
        source_label: Value written in the ``source`` column (default
            ``"original"``).

    Returns:
        A tuple of ``(manifest_dataframe, class_mapping_dict)``.

    Raises:
        FileNotFoundError: If *dataset_root* does not exist or contains no
            class subdirectories.
    """
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # --- Discover classes ------------------------------------------------
    class_names = _discover_classes(dataset_root)
    class_mapping = _build_class_mapping(class_names)
    logger.info("Found %d classes: %s", len(class_names), class_names)

    # --- Scan images -----------------------------------------------------
    records: list[dict[str, Any]] = []
    unreadable_files: list[str] = []
    image_counter = 0

    for class_name in class_names:
        class_dir = dataset_root / class_name
        image_files = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )

        for img_path in image_files:
            meta = _read_image_metadata(img_path)
            if meta is None:
                unreadable_files.append(str(img_path))
                continue

            # Relative path from dataset_root for portability.
            rel_path = img_path.relative_to(dataset_root)

            record = {
                "image_id": f"img_{image_counter:06d}",
                "image_path": str(rel_path).replace("\\", "/"),
                "class_name": class_name,
                "class_id": class_mapping[class_name],
                "width": meta["width"],
                "height": meta["height"],
                "channels": meta["channels"],
                "file_size": img_path.stat().st_size,
                "file_hash": _compute_file_hash(img_path),
                "source": source_label,
            }
            records.append(record)
            image_counter += 1

    if not records:
        raise RuntimeError(
            f"No valid images found in {dataset_root}. "
            "Check that the directory contains image files with extensions: "
            f"{VALID_IMAGE_EXTENSIONS}"
        )

    # --- Build DataFrame -------------------------------------------------
    manifest_df = pd.DataFrame(records)

    # --- Save outputs ----------------------------------------------------
    manifest_path = output_dir / "dataset_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("Manifest saved: %s (%d images)", manifest_path, len(manifest_df))

    mapping_path = output_dir / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    logger.info("Class mapping saved: %s", mapping_path)

    # --- Report unreadable files -----------------------------------------
    if unreadable_files:
        logger.warning(
            "%d unreadable image(s) were skipped:\n  %s",
            len(unreadable_files),
            "\n  ".join(unreadable_files),
        )

    return manifest_df, class_mapping
