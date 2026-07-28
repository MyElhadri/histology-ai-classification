import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.gtex_integrity import audit_gtex_dataset, EXPECTED_COUNTS, EXPECTED_CLASS_COUNTS


@pytest.fixture
def mock_gtex_dataset(tmp_path: Path):
    """Create a minimal synthetic GTEx dataset matching expected counts."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    
    # Create class mapping with 6 root keys to simulate the error condition
    class_mapping = {cls: idx for idx, cls in enumerate(EXPECTED_CLASS_COUNTS.keys())}
    complex_mapping = {
        "class_to_idx": class_mapping,
        "idx_to_class": {v: k for k, v in class_mapping.items()},
        "classes": list(class_mapping.keys()),
        "class_weights": {str(v): 1.0 for v in class_mapping.values()},
        "num_classes": 11,
        "metadata": {"version": "1.0"}
    }
    with open(metadata_dir / "class_mapping.json", "w") as f:
        json.dump(complex_mapping, f)
        
    donor_counter = 1
    
    for split in ["train", "validation", "test"]:
        rows = []
        for cls_name, counts_dict in EXPECTED_CLASS_COUNTS.items():
            count = counts_dict[split]
            cls_idx = class_mapping[cls_name]
            for i in range(count):
                rows.append({
                    "image_path": f"{split}/{cls_name}_{i}.png",
                    "class": cls_name,
                    "class_id": cls_idx,
                    "donor_id": f"donor_{donor_counter}"
                })
                # Increment donor every 10 images to ensure multiple donors per split
                if i % 10 == 0:
                    donor_counter += 1
        
        # Advance donor counter by 1000 between splits to ensure NO OVERLAP
        donor_counter += 1000
        
        df = pd.DataFrame(rows)
        df.to_csv(metadata_dir / f"{split}.csv", index=False)
        
    return tmp_path


def test_audit_gtex_dataset_success(mock_gtex_dataset: Path):
    report = audit_gtex_dataset(mock_gtex_dataset)
    assert report["status"] == "PASS"
    assert report["total_actual"] == EXPECTED_COUNTS["total"]
    assert report["donor_level_available"] is True
    assert report["leakage_checks"]["train_val_overlap"] == 0
    assert report["leakage_checks"]["train_test_overlap"] == 0
    assert report["leakage_checks"]["val_test_overlap"] == 0


def test_audit_gtex_dataset_detects_donor_leakage(mock_gtex_dataset: Path):
    # Introduce leakage
    val_csv = mock_gtex_dataset / "metadata" / "validation.csv"
    val_df = pd.read_csv(val_csv)
    
    train_csv = mock_gtex_dataset / "metadata" / "train.csv"
    train_df = pd.read_csv(train_csv)
    
    leaked_donor = train_df.iloc[0]["donor_id"]
    val_df.loc[0, "donor_id"] = leaked_donor
    val_df.to_csv(val_csv, index=False)
    
    with pytest.raises(ValueError, match="DONOR LEAKAGE DETECTED across splits"):
        audit_gtex_dataset(mock_gtex_dataset)


def test_audit_gtex_dataset_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing metadata file"):
        audit_gtex_dataset(tmp_path)


def test_audit_gtex_dataset_detects_duplicates(mock_gtex_dataset: Path):
    train_csv = mock_gtex_dataset / "metadata" / "train.csv"
    train_df = pd.read_csv(train_csv)
    
    # Duplicate first row
    train_df = pd.concat([train_df, train_df.iloc[[0]]])
    train_df.to_csv(train_csv, index=False)
    
    with pytest.raises(ValueError, match="Duplicate rows found in train.csv"):
        audit_gtex_dataset(mock_gtex_dataset)


def test_audit_gtex_dataset_detects_image_overlap(mock_gtex_dataset: Path):
    # Introduce image path leakage
    val_csv = mock_gtex_dataset / "metadata" / "validation.csv"
    val_df = pd.read_csv(val_csv)
    
    train_csv = mock_gtex_dataset / "metadata" / "train.csv"
    train_df = pd.read_csv(train_csv)
    
    leaked_image = train_df.iloc[0]["image_path"]
    val_df.loc[0, "image_path"] = leaked_image
    val_df.to_csv(val_csv, index=False)
    
    with pytest.raises(ValueError, match="Image paths are duplicated across splits!"):
        audit_gtex_dataset(mock_gtex_dataset)


def test_audit_gtex_dataset_count_mismatch(mock_gtex_dataset: Path):
    train_csv = mock_gtex_dataset / "metadata" / "train.csv"
    train_df = pd.read_csv(train_csv)
    
    # Drop one row
    train_df = train_df.iloc[1:]
    train_df.to_csv(train_csv, index=False)
    
    with pytest.raises(ValueError, match="Class count mismatch"):
        audit_gtex_dataset(mock_gtex_dataset)


def test_parse_gtex_class_mapping():
    from src.data.gtex_integrity import parse_gtex_class_mapping
    from src.data.gtex_integrity import EXPECTED_CLASS_COUNTS
    
    class_mapping = {cls: idx for idx, cls in enumerate(EXPECTED_CLASS_COUNTS.keys())}
    complex_mapping = {
        "class_to_idx": class_mapping,
        "idx_to_class": {v: k for k, v in class_mapping.items()},
        "classes": list(class_mapping.keys()),
        "class_weights": {str(v): 1.5 for v in class_mapping.values()},
        "num_classes": 11,
        "metadata": {"version": "1.0"}
    }
    
    parsed = parse_gtex_class_mapping(complex_mapping)
    assert parsed["num_classes"] == 11
    assert parsed["class_to_idx"] == class_mapping
    
    # Negative: missing class
    bad_mapping = class_mapping.copy()
    del bad_mapping["bladder"]
    with pytest.raises(ValueError, match="Class mismatch"):
        parse_gtex_class_mapping({"class_to_idx": bad_mapping})
        
    # Negative: extra class
    bad_mapping = class_mapping.copy()
    bad_mapping["extra"] = 11
    with pytest.raises(ValueError, match="Class mismatch"):
        parse_gtex_class_mapping({"class_to_idx": bad_mapping})
        
    # Negative: invalid weights
    bad_weight_mapping = complex_mapping.copy()
    bad_weight_mapping["class_weights"] = {"0": -1.0}
    with pytest.raises(ValueError, match="Invalid weight"):
        parse_gtex_class_mapping(bad_weight_mapping)
        
    # Test string integers flat
    string_int_mapping = {k: str(v) for k, v in class_mapping.items()}
    parsed_str = parse_gtex_class_mapping(string_int_mapping)
    assert parsed_str["class_to_idx"] == class_mapping
    
    # Test string integers inside class_to_idx
    str_idx = {"class_to_idx": {k: str(v) for k, v in class_mapping.items()}}
    parsed_str2 = parse_gtex_class_mapping(str_idx)
    assert parsed_str2["class_to_idx"] == class_mapping
