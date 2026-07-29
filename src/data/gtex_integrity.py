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


def _parse_counts(counts_dict: dict, class_to_idx: dict, section_name: str) -> dict:
    parsed = {}
    for k, v in counts_dict.items():
        if k not in class_to_idx:
            continue
        try:
            val = float(v)
        except ValueError:
            raise ValueError(f"Count for {k} in {section_name} must be a number.")
        if not val.is_integer() or val < 0:
            raise ValueError(f"Count for {k} in {section_name} must be a non-negative integer.")
        parsed[k] = int(val)
    if len(parsed) != 11:
        raise ValueError(f"{section_name} must contain exactly 11 classes, got {len(parsed)}.")
    return parsed

def parse_gtex_class_mapping(mapping: dict) -> dict:
    """Parses and normalizes the GTEx class_mapping.json document."""
    
    mapping_source_path = ""
    detected_schema = "unknown"
    
    if "label_to_index" in mapping:
        class_to_idx = mapping["label_to_index"]
        if "index_to_label" in mapping:
            detected_schema = "label_to_index/index_to_label"
        else:
            detected_schema = "label_to_index"
        mapping_source_path = "$.label_to_index"
    elif "class_to_idx" in mapping:
        class_to_idx = mapping["class_to_idx"]
        detected_schema = "class_to_idx"
        mapping_source_path = "$.class_to_idx"
    elif "classes" in mapping and isinstance(mapping["classes"], dict):
        class_to_idx = mapping["classes"]
        detected_schema = "classes"
        mapping_source_path = "$.classes"
    else:
        class_to_idx = {}
        for k, v in mapping.items():
            if isinstance(v, int):
                class_to_idx[k] = v
            elif isinstance(v, str) and v.isdigit():
                class_to_idx[k] = int(v)
        detected_schema = "flat"
        mapping_source_path = "$"
                
    if not class_to_idx:
        raise ValueError(
            "Could not extract a valid class mapping from the document.\n"
            f"Root keys found: {list(mapping.keys())}\n"
            "Supported mapping keys: 'label_to_index', 'class_to_idx', 'classes' or flat integers."
        )

    try:
        class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
    except ValueError:
        raise ValueError("Some class indices could not be converted to integers.")

    expected_classes = {
        "bladder", "brain", "cerebellum", "kidney", "liver", "lung", 
        "muscle", "oesophagus", "pancreas", "spleen", "testis"
    }
    
    found_classes = set(class_to_idx.keys())
    if found_classes != expected_classes:
        missing = expected_classes - found_classes
        extra = found_classes - expected_classes
        raise ValueError(
            f"Class mismatch in {mapping_source_path}.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )
        
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    if len(idx_to_class) != 11:
        raise ValueError("Duplicate indices found in class mapping.")
        
    expected_indices = set(range(11))
    if set(idx_to_class.keys()) != expected_indices:
        missing_idx = expected_indices - set(idx_to_class.keys())
        extra_idx = set(idx_to_class.keys()) - expected_indices
        raise ValueError(
            f"Indices must be exactly 0 to 10.\n"
            f"Missing indices: {missing_idx}\n"
            f"Extra indices: {extra_idx}"
        )
        
    if "index_to_label" in mapping:
        for idx_str, class_name in mapping["index_to_label"].items():
            if not str(idx_str).isdigit():
                continue
            idx = int(idx_str)
            if idx not in idx_to_class or idx_to_class[idx] != class_name:
                raise ValueError(f"Incohérence entre {mapping_source_path} et index_to_label pour l'index {idx}")

    class_weights = {}
    if "class_weights" in mapping:
        import math
        cw = mapping["class_weights"]
        for k, v in cw.items():
            if k in class_to_idx:
                idx = class_to_idx[k]
            elif str(k).isdigit() and int(k) in idx_to_class:
                idx = int(k)
            else:
                continue
            try:
                weight = float(v)
            except ValueError:
                raise ValueError(f"Weight for {k} must be numeric.")
            if weight <= 0 or not math.isfinite(weight) or math.isnan(weight):
                raise ValueError(f"Invalid weight {weight} for {k} (must be > 0 and finite).")
            class_weights[idx] = weight
        if len(class_weights) != 11:
            raise ValueError(f"class_weights must contain exactly 11 classes, got {len(class_weights)}.")

    train_counts = {}
    if "train_counts" in mapping:
        train_counts = _parse_counts(mapping["train_counts"], class_to_idx, "train_counts")
    
    validation_counts = {}
    if "validation_counts" in mapping:
        validation_counts = _parse_counts(mapping["validation_counts"], class_to_idx, "validation_counts")
        
    test_counts = {}
    if "test_counts" in mapping:
        test_counts = _parse_counts(mapping["test_counts"], class_to_idx, "test_counts")

    print(f"Detected GTEx mapping schema:\n{detected_schema}\n")
    print(f"Mapping source:\n{mapping_source_path}\n")

    return {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "classes": [idx_to_class[i] for i in range(11)],
        "class_weights": class_weights,
        "num_classes": 11,
        "train_counts": train_counts,
        "validation_counts": validation_counts,
        "test_counts": test_counts,
        "detected_schema": detected_schema,
        "mapping_source_path": mapping_source_path
    }



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
        raw_mapping = json.load(f)
        
    parsed_mapping = parse_gtex_class_mapping(raw_mapping)
    class_mapping = parsed_mapping["class_to_idx"]

    report = {
        "status": "PASS",
        "total_expected": EXPECTED_COUNTS["total"],
        "total_actual": 0,
        "splits": {},
        "donor_level_available": bool(donor_col),
        "donor_column": donor_col,
        "leakage_checks": {},
        "detected_schema": parsed_mapping.get("detected_schema"),
        "mapping_source_path": parsed_mapping.get("mapping_source_path"),
        "class_to_idx": parsed_mapping.get("class_to_idx"),
        "idx_to_class": {str(k): v for k, v in parsed_mapping.get("idx_to_class", {}).items()},
        "classes": parsed_mapping.get("classes"),
        "class_weights_by_index": {str(k): v for k, v in parsed_mapping.get("class_weights", {}).items()},
        "train_counts_expected": parsed_mapping.get("train_counts"),
        "validation_counts_expected": parsed_mapping.get("validation_counts"),
        "test_counts_expected": parsed_mapping.get("test_counts")
    }

    # Verify counts
    total_actual = 0
    for split in splits:
        df = dfs[split]
        count = len(df)
        total_actual += count
        
        # Verify class counts
        class_col = "class" if "class" in df.columns else "label"
        class_counts = df[class_col].value_counts().to_dict()
        for cls_name, splits_counts in EXPECTED_CLASS_COUNTS.items():
            expected = splits_counts[split]
            actual = class_counts.get(cls_name, 0)
            if actual != expected:
                raise ValueError(f"Class count mismatch for {cls_name} in {split}. Expected {expected}, got {actual}.")
                
        report["splits"][split] = {
            "expected": EXPECTED_COUNTS[split],
            "actual": count,
            "match": count == EXPECTED_COUNTS[split],
            "expected_class_counts": {k: v for k, v in EXPECTED_CLASS_COUNTS.items()},
            "actual_class_counts": class_counts,
            "differences": {k: class_counts.get(k, 0) - EXPECTED_CLASS_COUNTS[k][split] for k in EXPECTED_CLASS_COUNTS}
        }
        if count != EXPECTED_COUNTS[split]:
            raise ValueError(f"Split {split} count mismatch. Expected {EXPECTED_COUNTS[split]}, got {count}")
            
    report["total_actual"] = total_actual
    report["total_expected"] = EXPECTED_COUNTS["total"]
    report["total_difference"] = total_actual - EXPECTED_COUNTS["total"]
    
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
