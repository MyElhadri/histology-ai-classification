"""Build a simple dataset manifest.

Scans the raw dataset directory and produces `dataset_manifest.csv`.
"""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Accepted image extensions
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_manifest(dataset_root: Path | str, output_path: Path | str) -> None:
    """Build dataset_manifest.csv from the dataset root.
    
    Args:
        dataset_root: Path to the dataset directory containing class subfolders.
        output_path: Path where the CSV will be saved.
    """
    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path).resolve()
    
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        
    # Discover classes and sort alphabetically for consistent mapping
    class_names = sorted(d.name for d in dataset_root.iterdir() if d.is_dir())
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    records = []
    image_counter = 0
    
    for class_name in class_names:
        class_dir = dataset_root / class_name
        class_id = class_mapping[class_name]
        
        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                # Use relative paths for portability
                rel_path = file_path.relative_to(dataset_root).as_posix()
                
                records.append({
                    "image_id": f"img_{image_counter:06d}",
                    "image_path": rel_path,
                    "class_name": class_name,
                    "class_id": class_id
                })
                image_counter += 1
                
    if not records:
        raise ValueError(f"No valid images found in {dataset_root}")
        
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "class_name", "class_id"])
        writer.writeheader()
        writer.writerows(records)
        
    logger.info(f"Manifest saved to {output_path} ({len(records)} images).")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    build_manifest(args.dataset_root, args.output_path)
