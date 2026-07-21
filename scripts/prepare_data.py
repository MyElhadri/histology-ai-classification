#!/usr/bin/env python3
"""Prepare data pipeline — CLI entry point.

Orchestrates the full dataset preparation workflow:
    1. Build dataset manifest and class mapping
    2. Run quality audit
    3. Create shared test split + DenseNet121 fold assignments

Usage:
    python scripts/prepare_data.py \\
        --dataset-root /path/to/dataset \\
        --output-root  /path/to/output

The script accepts paths as arguments and does NOT hardcode any
personal Google Drive path or local machine path.

Author: Yassine
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``src.*`` imports work when
# this script is invoked directly (e.g., from Colab or CLI).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.audit import audit_manifest          # noqa: E402
from src.data.build_manifest import build_manifest  # noqa: E402
from src.data.split_dataset import create_splits    # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset: manifest, audit, and splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset root directory (one sub-folder per class).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help=(
            "Root directory for all outputs.  Manifests go to "
            "<output-root>/data/manifests/, reports to "
            "<output-root>/reports/."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of images reserved for the shared test set (default: 0.20).",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for DenseNet121 (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full prepare-data pipeline."""
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifests_dir = output_root / "data" / "manifests"

    logger.info("=" * 60)
    logger.info("DATASET PREPARATION PIPELINE")
    logger.info("=" * 60)
    logger.info("Dataset root : %s", dataset_root)
    logger.info("Output root  : %s", output_root)
    logger.info("Test size    : %.0f%%", args.test_size * 100)
    logger.info("Num folds    : %d (DenseNet121 only)", args.num_folds)
    logger.info("Seed         : %d", args.seed)
    logger.info("=" * 60)

    # Step 1 — Build manifest + class mapping
    logger.info("[1/3] Building dataset manifest...")
    manifest_df, class_mapping = build_manifest(
        dataset_root=dataset_root,
        output_dir=manifests_dir,
    )

    # Step 2 — Audit
    logger.info("[2/3] Running dataset audit...")
    audit_report = audit_manifest(
        manifest_df=manifest_df,
        output_dir=output_root,
    )

    # Step 3 — Splits
    logger.info("[3/3] Creating splits...")
    test_df, folds_df = create_splits(
        manifest_df=manifest_df,
        output_dir=manifests_dir,
        test_size=args.test_size,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total images       : %d", len(manifest_df))
    logger.info("Classes            : %d", len(class_mapping))
    logger.info("Test set           : %d images (shared)", len(test_df))
    logger.info("Development set    : %d images", len(folds_df))
    logger.info("DenseNet121 folds  : %d", args.num_folds)
    logger.info("Duplicates found   : %d", audit_report["exact_duplicates"]["count"])
    logger.info("Outputs saved to   : %s", output_root)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
