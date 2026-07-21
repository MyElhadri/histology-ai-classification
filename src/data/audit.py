"""Dataset audit — quality checks on the dataset manifest.

Analyses the manifest produced by :mod:`build_manifest` and generates:
  - reports/metrics/dataset_audit.json  — structured audit results
  - reports/figures/class_distribution.png — bar chart of class balance

The audit checks for:
  - Unreadable / corrupted images (already flagged during manifest build)
  - Exact duplicate images (by file hash)
  - Dimension anomalies
  - Class distribution balance
  - Extension summary
  - Total class count verification

This module never modifies the original images.

Author: Yassine
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for Colab & headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def audit_manifest(
    manifest_df: pd.DataFrame,
    output_dir: Path,
    *,
    expected_num_classes: int = 22,
) -> dict[str, Any]:
    """Run quality checks on the dataset manifest.

    Args:
        manifest_df: DataFrame produced by
            :func:`build_manifest.build_manifest`.
        output_dir: Root directory for report outputs.  Audit JSON goes to
            ``<output_dir>/reports/metrics/`` and the distribution plot to
            ``<output_dir>/reports/figures/``.
        expected_num_classes: Expected number of classes to validate against.

    Returns:
        Dictionary containing the full audit report.
    """
    report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Basic counts
    # ------------------------------------------------------------------
    total_images = len(manifest_df)
    class_counts = manifest_df["class_name"].value_counts().to_dict()
    num_classes = manifest_df["class_name"].nunique()

    report["total_images"] = total_images
    report["num_classes"] = num_classes
    report["expected_num_classes"] = expected_num_classes
    report["class_count_match"] = num_classes == expected_num_classes
    report["images_per_class"] = {
        k: int(v) for k, v in class_counts.items()
    }

    counts_array = np.array(list(class_counts.values()))
    report["min_images_per_class"] = int(counts_array.min())
    report["max_images_per_class"] = int(counts_array.max())
    report["mean_images_per_class"] = round(float(counts_array.mean()), 2)
    report["std_images_per_class"] = round(float(counts_array.std()), 2)

    imbalance_ratio = float(counts_array.max() / counts_array.min())
    report["imbalance_ratio"] = round(imbalance_ratio, 2)

    logger.info(
        "Dataset: %d images, %d classes, imbalance ratio %.2f",
        total_images, num_classes, imbalance_ratio,
    )

    # ------------------------------------------------------------------
    # 2. Duplicate detection (by file hash)
    # ------------------------------------------------------------------
    hash_counts = manifest_df["file_hash"].value_counts()
    duplicates = hash_counts[hash_counts > 1]
    duplicate_groups: dict[str, list[str]] = {}

    for file_hash in duplicates.index:
        paths = manifest_df.loc[
            manifest_df["file_hash"] == file_hash, "image_path"
        ].tolist()
        duplicate_groups[file_hash] = paths

    report["exact_duplicates"] = {
        "count": int(duplicates.sum() - len(duplicates)),
        "groups": duplicate_groups,
    }

    if duplicate_groups:
        logger.warning(
            "Found %d duplicate groups (%d redundant images)",
            len(duplicate_groups),
            int(duplicates.sum() - len(duplicates)),
        )
    else:
        logger.info("No exact duplicate images found.")

    # ------------------------------------------------------------------
    # 3. Dimension analysis
    # ------------------------------------------------------------------
    unique_dims = manifest_df.groupby(["width", "height"]).size().to_dict()
    report["unique_dimensions"] = {
        f"{w}x{h}": int(c) for (w, h), c in unique_dims.items()
    }
    report["uniform_dimensions"] = len(unique_dims) == 1

    # ------------------------------------------------------------------
    # 4. Extension summary
    # ------------------------------------------------------------------
    extensions = (
        manifest_df["image_path"]
        .apply(lambda p: Path(p).suffix.lower())
        .value_counts()
        .to_dict()
    )
    report["extensions"] = {k: int(v) for k, v in extensions.items()}

    # ------------------------------------------------------------------
    # 5. Channel summary
    # ------------------------------------------------------------------
    channel_counts = manifest_df["channels"].value_counts().to_dict()
    report["channels"] = {str(k): int(v) for k, v in channel_counts.items()}

    # ------------------------------------------------------------------
    # Save audit JSON
    # ------------------------------------------------------------------
    metrics_dir = Path(output_dir) / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    audit_path = metrics_dir / "dataset_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Audit report saved: %s", audit_path)

    # ------------------------------------------------------------------
    # Save class distribution plot
    # ------------------------------------------------------------------
    _plot_class_distribution(class_counts, output_dir)

    return report


def _plot_class_distribution(
    class_counts: dict[str, int],
    output_dir: Path,
) -> None:
    """Create and save a horizontal bar chart of class distribution."""
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1])
    names = [n.replace("human_tissue_image-", "") for n, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(counts)))
    bars = ax.barh(names, counts, color=colors, edgecolor="white")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )

    mean_count = float(np.mean(counts))
    ax.axvline(
        x=mean_count, color="red", linestyle="--", linewidth=1.5,
        label=f"Mean: {mean_count:.1f}",
    )

    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution — Images per Tissue Type",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    figures_dir = Path(output_dir) / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Class distribution plot saved: %s", plot_path)
