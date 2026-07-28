"""GTEx Dataset Integrity Auditor.

Validates exact GTEx 11 classes counts and strict donor isolation across splits.
"""

import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COUNTS = {
    "train": 40424,
    "validation": 8114,
    "test": 8204,
    "total": 56742,
}

EXPECTED_CLASS_COUNTS = {
    "bladder": {"train": 3846, "validation": 1000, "test": 800},
    "brain": {"train": 2301, "validation": 500, "test": 500},
    "cerebellum": {"train": 2020, "validation": 220, "test": 310},
    "kidney": {"train": 4249, "validation": 1000, "test": 709},
    "liver": {"train": 4330, "validation": 730, "test": 999},
    "lung": {"train": 4286, "validation": 1005, "test": 684},
    "muscle": {"train": 4339, "validation": 619, "test": 1093},
    "oesophagus": {"train": 4346, "validation": 710, "test": 1000},
    "pancreas": {"train": 3938, "validation": 1110, "test": 1009},
    "spleen": {"train": 4640, "validation": 820, "test": 600},
    "testis": {"train": 2129, "validation": 400, "test": 500},
}


def find_donor_column(columns: list[str]) -> str | None:
    """Identify the donor/patient ID column from a list of column names."""
    candidates = ["donor_id", "donor", "subject_id", "subject", "participant_id", "case_id"]
    for col in columns:
        if col.lower() in candidates:
            return col
    return None


def audit_gtex_dataset(dataset_dir: Path | str, output_report: Path | str | None = None) -> dict:
    """Audit the GTEx dataset for exact counts and donor leakage.

    Args:
        dataset_dir: Path to the GTEx_11_classes directory.
        output_report: Optional path to save the JSON report.

    Returns:
        Dictionary containing the audit report.

    Raises:
        ValueError: If integrity checks fail (counts mismatch, leakage, etc.)
    """
    dataset_dir = Path(dataset_dir)
    metadata_dir = dataset_dir / "metadata"

    splits = ["train", "validation", "test"]
    dfs = {}
    donor_col = None

    for split in splits:
        csv_path = metadata_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Ensure no duplicates
        if df.duplicated().any():
            raise ValueError(f"Duplicate rows found in {split}.csv")
            
        dfs[split] = df

    # Find donor column
    donor_col = find_donor_column(dfs["train"].columns.tolist())
    
    # Class mapping
    mapping_path = metadata_dir / "class_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing class mapping: {mapping_path}")
        
    with open(mapping_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)
        
    if len(class_mapping) != 11:
        raise ValueError(f"Expected 11 classes, got {len(class_mapping)} in class_mapping.json")
        
    for k, v in class_mapping.items():
        if int(v) not in range(11):
            raise ValueError(f"Class index out of bounds (0-10): {k} -> {v}")

    report = {
        "status": "PASS",
        "total_expected": EXPECTED_COUNTS["total"],
        "total_actual": 0,
        "splits": {},
        "donor_level_available": bool(donor_col),
        "donor_column": donor_col,
        "leakage_checks": {}
    }

    # Verify counts
    total_actual = 0
    for split in splits:
        df = dfs[split]
        count = len(df)
        total_actual += count
        
        # Verify class counts
        class_counts = df["class"].value_counts().to_dict()
        for cls_name, splits_counts in EXPECTED_CLASS_COUNTS.items():
            expected = splits_counts[split]
            actual = class_counts.get(cls_name, 0)
            if actual != expected:
                raise ValueError(f"Class count mismatch for {cls_name} in {split}. Expected {expected}, got {actual}.")
                
        report["splits"][split] = {
            "expected": EXPECTED_COUNTS[split],
            "actual": count,
            "match": count == EXPECTED_COUNTS[split]
        }
        if count != EXPECTED_COUNTS[split]:
            raise ValueError(f"Split {split} count mismatch. Expected {EXPECTED_COUNTS[split]}, got {count}")
            
    report["total_actual"] = total_actual
    if total_actual != EXPECTED_COUNTS["total"]:
        raise ValueError(f"Total count mismatch. Expected {EXPECTED_COUNTS['total']}, got {total_actual}")

    # Cross-split image uniqueness
    all_images = pd.concat([dfs[s]["image_path"] for s in splits])
    if all_images.duplicated().any():
        raise ValueError("Image paths are duplicated across splits! Severe data leakage.")

    # Donor isolation
    if donor_col:
        train_donors = set(dfs["train"][donor_col].dropna().unique())
        val_donors = set(dfs["validation"][donor_col].dropna().unique())
        test_donors = set(dfs["test"][donor_col].dropna().unique())

        leak_train_val = train_donors.intersection(val_donors)
        leak_train_test = train_donors.intersection(test_donors)
        leak_val_test = val_donors.intersection(test_donors)

        report["leakage_checks"] = {
            "train_val_overlap": len(leak_train_val),
            "train_test_overlap": len(leak_train_test),
            "val_test_overlap": len(leak_val_test)
        }

        if leak_train_val or leak_train_test or leak_val_test:
            msg = "DONOR LEAKAGE DETECTED across splits:\n"
            if leak_train_val:
                msg += f"Train/Val: {len(leak_train_val)} donors overlap.\n"
            if leak_train_test:
                msg += f"Train/Test: {len(leak_train_test)} donors overlap.\n"
            if leak_val_test:
                msg += f"Val/Test: {len(leak_val_test)} donors overlap.\n"
            raise ValueError(msg)
    else:
        logger.warning("No valid donor column found in metadata. Donor-level evaluation will be disabled. Proceeding with patch-level only.")

    if output_report:
        output_report = Path(output_report)
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
    return report
