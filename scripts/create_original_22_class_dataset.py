"""Script to extract 22 human histology classes from NuInsSeg.

Copies images without modification, generates a manifest, class mapping,
and an analysis report.
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import shutil

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Valid image extensions to copy
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Exact mapping from NuInsSeg to target
FOLDER_MAPPING = {
    "human bladder": "human_tissue_image-bladder",
    "human brain": "human_tissue_image-brain",
    "human cardia": "human_tissue_image-cardia",
    "human cerebellum": "human_tissue_image-cerebellum",
    "human epiglottis": "human_tissue_image-epiglottis",
    "human salivory gland": "human_tissue_image-gland",
    "human jejunum": "human_tissue_image-jejunum",
    "human kidney": "human_tissue_image-kidney",
    "human liver": "human_tissue_image-liver",
    "human lung": "human_tissue_image-lung",
    "human melanoma": "human_tissue_image-melanoma",
    "human muscle": "human_tissue_image-muscle",
    "human oesophagus": "human_tissue_image-oesophagus",
    "human pancreas": "human_tissue_image-pancreas",
    "human peritoneum": "human_tissue_image-peritoneum",
    "human pylorus": "human_tissue_image-pylorus",
    "human rectum": "human_tissue_image-rectum",
    "human spleen": "human_tissue_image-spleen",
    "human testis": "human_tissue_image-testis",
    "human tongue": "human_tissue_image-tongue",
    "human tonsile": "human_tissue_image-tonsile",
    "human umbilical cord": "human_tissue_image-umbilical-cord",
}

EXCLUDED_FOLDERS = {"human placenta"}


def calculate_hash(image_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    hasher = sha256()
    with image_path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_image_info(image_path: Path) -> tuple[int, int, str]:
    """Get image width, height, and color mode."""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height, img.mode
    except Exception:
        return -1, -1, "ERROR"


def create_dataset(source_root: Path, dest_root: Path, overwrite: bool) -> None:
    """Main execution function to copy data and generate artifacts."""
    source_root = source_root.resolve()
    dest_root = dest_root.resolve()
    project_root = Path(__file__).resolve().parent.parent

    # 1. Verify source
    if not source_root.is_dir():
        logger.error(f"Source directory does not exist: {source_root}")
        sys.exit(1)

    # 2. Verify all 22 source folders exist
    missing_sources = []
    for src_name in FOLDER_MAPPING.keys():
        if not (source_root / src_name).is_dir():
            missing_sources.append(src_name)
    
    if missing_sources:
        logger.error(f"Missing mandatory source folders: {missing_sources}")
        sys.exit(1)

    # Handle destination
    if dest_root.exists():
        if any(dest_root.iterdir()):
            if not overwrite:
                logger.error(f"Destination directory is not empty: {dest_root}")
                logger.error("Use --overwrite to proceed.")
                sys.exit(1)
            else:
                logger.warning(f"Overwriting destination directory: {dest_root}")
                shutil.rmtree(dest_root)
    
    dest_root.mkdir(parents=True, exist_ok=True)

    # Setup metrics collections
    class_distribution = defaultdict(int)
    dimensions_dist = defaultdict(int)
    color_modes_dist = defaultdict(int)
    extensions_dist = defaultdict(int)
    corrupted_images = []
    hash_map: dict[str, list[str]] = defaultdict(list)
    manifest_records = []
    
    # Define Class Mapping deterministically
    sorted_short_names = sorted([target.split("-", 1)[1] for target in FOLDER_MAPPING.values()])
    class_mapping = {name: idx for idx, name in enumerate(sorted_short_names)}

    total_images = 0

    # 4 & 5 & 6. Create folders and copy files
    for src_name, target_name in FOLDER_MAPPING.items():
        short_class_name = target_name.split("-", 1)[1]
        class_id = class_mapping[short_class_name]
        
        src_dir = source_root / src_name / "tissue images"
        dest_dir = dest_root / target_name
        dest_dir.mkdir()

        if not src_dir.is_dir():
            logger.warning(f"Tissue images directory not found: {src_dir}")
            continue

        for file_path in src_dir.iterdir():
            if not file_path.is_file():
                continue
                
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            # Copy file with metadata
            dest_file = dest_dir / file_path.name
            shutil.copy2(file_path, dest_file)
            
            # Analyze copied file
            total_images += 1
            class_distribution[short_class_name] += 1
            extensions_dist[file_path.suffix.lower()] += 1
            
            w, h, mode = get_image_info(dest_file)
            if mode == "ERROR":
                corrupted_images.append(str(dest_file.relative_to(project_root).as_posix()))
                logger.error(f"Corrupted image detected: {dest_file}")
            else:
                dimensions_dist[f"{w}x{h}"] += 1
                color_modes_dist[mode] += 1

            file_hash = calculate_hash(dest_file)
            rel_path = dest_file.relative_to(project_root).as_posix()
            hash_map[file_hash].append(rel_path)

            manifest_records.append({
                "image_id": f"img_nuinsseg_{total_images:06d}",
                "image_path": rel_path,
                "class_name": short_class_name,
                "class_id": class_id,
                "original_filename": file_path.name,
                "width": w,
                "height": h,
                "channels": 3 if mode == "RGB" else (1 if mode in ("L", "1") else 4),
                "file_hash": file_hash
            })

    # Find exact duplicate groups
    exact_duplicate_groups = [paths for paths in hash_map.values() if len(paths) > 1]

    # Display Warnings if not exactly 432
    if total_images != 432:
        logger.warning(f"Expected 432 images, but found {total_images}!")
        logger.warning("Class distribution:")
        for cls, count in class_distribution.items():
            logger.warning(f"  {cls}: {count}")
    else:
        logger.info("Successfully verified total image count: 432.")

    empty_classes = [c for c in class_mapping.keys() if class_distribution[c] == 0]
    if empty_classes:
        logger.error(f"The following classes are empty: {empty_classes}")

    # Generate Manifest
    manifest_path = project_root / "data" / "manifests" / "original_22_dataset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "image_path", "class_name", "class_id", 
            "original_filename", "width", "height", "channels", "file_hash"
        ])
        writer.writeheader()
        writer.writerows(manifest_records)
    logger.info(f"Manifest saved to {manifest_path.relative_to(project_root).as_posix()}")

    # Generate Class Mapping JSON
    mapping_path = project_root / "data" / "manifests" / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    logger.info(f"Class mapping saved to {mapping_path.relative_to(project_root).as_posix()}")

    # Generate Report JSON
    report_dict = {
        "source_root": str(source_root),
        "destination_root": str(dest_root),
        "number_of_classes": len(class_mapping),
        "total_images": total_images,
        "class_distribution": dict(class_distribution),
        "dimensions": dict(dimensions_dist),
        "color_modes": dict(color_modes_dist),
        "extensions": dict(extensions_dist),
        "corrupted_images": corrupted_images,
        "exact_duplicate_groups": exact_duplicate_groups
    }

    report_dir = project_root / "reports" / "densenet121" / "metrics"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "original_22_dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Report saved to {report_path.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 22 human classes from NuInsSeg.")
    parser.add_argument("--source-root", type=Path, required=True, help="Path to original NuInsSeg tissue images.")
    parser.add_argument("--destination-root", type=Path, required=True, help="Path to destination dataset directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists.")
    
    args = parser.parse_args()
    create_dataset(args.source_root, args.destination_root, args.overwrite)
