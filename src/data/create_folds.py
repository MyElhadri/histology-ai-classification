"""Create stratified 5-fold cross-validation splits.

Reads dataset_manifest.csv and assigns a fold (0-4) to each image,
ensuring class distribution is maintained across folds.
Produces densenet121_folds.csv.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def create_folds(
    manifest_path: Path | str,
    output_path: Path | str,
    num_folds: int = 5,
    seed: int = 42
) -> None:
    """Create stratified folds and save to CSV.
    
    Args:
        manifest_path: Path to dataset_manifest.csv.
        output_path: Path to save densenet121_folds.csv.
        num_folds: Number of cross-validation folds.
        seed: Random seed for reproducibility.
    """
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    
    # Initialize fold column
    df["fold"] = -1
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["class_id"])):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx
        
    # Ensure all images have a valid fold
    if (df["fold"] == -1).any():
        raise RuntimeError("Some images were not assigned a fold.")
        
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save folds
    df.to_csv(output_path, index=False)
    
    logger.info(f"Folds saved to {output_path}")
    for fold_idx in range(num_folds):
        count = (df["fold"] == fold_idx).sum()
        logger.info(f"  Fold {fold_idx}: {count} images")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_folds(args.manifest_path, args.output_path, args.num_folds, args.seed)
