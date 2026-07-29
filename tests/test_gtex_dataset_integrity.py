import json
import copy
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

@pytest.fixture
def real_gtex_schema_json():
    return {
        "label_to_index": {
            "bladder": 0, "brain": 1, "cerebellum": 2, "kidney": 3,
            "liver": 4, "lung": 5, "muscle": 6, "oesophagus": 7,
            "pancreas": 8, "spleen": 9, "testis": 10
        },
        "index_to_label": {
            "0": "bladder", "1": "brain", "2": "cerebellum", "3": "kidney",
            "4": "liver", "5": "lung", "6": "muscle", "7": "oesophagus",
            "8": "pancreas", "9": "spleen", "10": "testis"
        },
        "class_weights": {
            "bladder": 0.9555, "brain": 1.5971, "cerebellum": 1.8193, "kidney": 0.8649,
            "liver": 0.8487, "lung": 0.8574, "muscle": 0.8469, "oesophagus": 0.8456,
            "pancreas": 0.9332, "spleen": 0.792, "testis": 1.7261
        },
        "train_counts": {
            "spleen": 4640, "brain": 2301, "pancreas": 3938, "oesophagus": 4346,
            "lung": 4286, "liver": 4330, "kidney": 4249, "testis": 2129,
            "muscle": 4339, "cerebellum": 2020, "bladder": 3846
        },
        "validation_counts": {
            "lung": 1005, "kidney": 1000, "pancreas": 1110, "liver": 730,
            "cerebellum": 220, "brain": 500, "muscle": 619, "spleen": 820,
            "oesophagus": 710, "testis": 400, "bladder": 1000
        },
        "test_counts": {
            "oesophagus": 1000, "pancreas": 1009, "brain": 500, "lung": 684,
            "spleen": 600, "cerebellum": 310, "muscle": 1093, "liver": 999,
            "testis": 500, "kidney": 709, "bladder": 800
        }
    }

def test_parse_real_gtex_label_to_index_schema(real_gtex_schema_json):
    from src.data.gtex_integrity import parse_gtex_class_mapping
    
    raw_document = real_gtex_schema_json
    assert len(raw_document) == 6
    
    parsed = parse_gtex_class_mapping(raw_document)
    
    assert parsed["num_classes"] == 11
    assert parsed["detected_schema"] == "label_to_index/index_to_label"
    assert parsed["mapping_source_path"] == "$.label_to_index"
    
    assert parsed["class_to_idx"]["bladder"] == 0
    assert parsed["class_to_idx"]["testis"] == 10
    assert parsed["idx_to_class"][0] == "bladder"
    assert parsed["idx_to_class"][10] == "testis"
    
    expected_classes_order = ["bladder", "brain", "cerebellum", "kidney", "liver", "lung", "muscle", "oesophagus", "pancreas", "spleen", "testis"]
    assert parsed["classes"] == expected_classes_order
    
    assert parsed["class_weights"][0] == 0.9555
    assert parsed["class_weights"][10] == 1.7261
    
    assert sum(parsed["train_counts"].values()) == 40424
    assert sum(parsed["validation_counts"].values()) == 8114
    assert sum(parsed["test_counts"].values()) == 8204

def test_parse_real_gtex_negative_cases(real_gtex_schema_json):
    from src.data.gtex_integrity import parse_gtex_class_mapping
    
    # 1. label_to_index absent but index_to_label present -> unsupported (flat fallback)
    # wait, if label_to_index absent, it falls back to empty flat and raises ValueError
    bad1 = real_gtex_schema_json.copy()
    del bad1["label_to_index"]
    with pytest.raises(ValueError):
        parse_gtex_class_mapping(bad1)
        
    # 2. index_to_label incoherence
    bad2 = copy.deepcopy(real_gtex_schema_json)
    bad2["index_to_label"]["0"] = "brain"
    with pytest.raises(ValueError, match="Incohérence"):
        parse_gtex_class_mapping(bad2)
        
    # 3. classe manquante
    bad3 = copy.deepcopy(real_gtex_schema_json)
    del bad3["label_to_index"]["bladder"]
    with pytest.raises(ValueError, match="Class mismatch"):
        parse_gtex_class_mapping(bad3)
        
    # 4. classe supplémentaire
    bad4 = copy.deepcopy(real_gtex_schema_json)
    bad4["label_to_index"]["extra"] = 11
    with pytest.raises(ValueError, match="Class mismatch"):
        parse_gtex_class_mapping(bad4)
        
    # 5. index dupliqué
    bad5 = copy.deepcopy(real_gtex_schema_json)
    bad5["label_to_index"]["testis"] = 0
    with pytest.raises(ValueError, match="Duplicate indices found"):
        parse_gtex_class_mapping(bad5)
        
    # 6. index hors plage
    bad6 = copy.deepcopy(real_gtex_schema_json)
    bad6["label_to_index"]["bladder"] = 11
    with pytest.raises(ValueError, match="Indices must be exactly"):
        parse_gtex_class_mapping(bad6)
        
    # 7. poids manquant
    bad7 = copy.deepcopy(real_gtex_schema_json)
    del bad7["class_weights"]["bladder"]
    with pytest.raises(ValueError, match="exactly 11 classes"):
        parse_gtex_class_mapping(bad7)
        
    # 8. poids nul
    bad8 = copy.deepcopy(real_gtex_schema_json)
    bad8["class_weights"]["bladder"] = 0
    with pytest.raises(ValueError, match="Invalid weight"):
        parse_gtex_class_mapping(bad8)
        
    # 9. poids négatif
    bad9 = copy.deepcopy(real_gtex_schema_json)
    bad9["class_weights"]["bladder"] = -1.0
    with pytest.raises(ValueError, match="Invalid weight"):
        parse_gtex_class_mapping(bad9)
        
    # 10. poids NaN
    bad10 = copy.deepcopy(real_gtex_schema_json)
    bad10["class_weights"]["bladder"] = float('nan')
    with pytest.raises(ValueError, match="Invalid weight"):
        parse_gtex_class_mapping(bad10)
        
    # 11. poids infini
    bad11 = copy.deepcopy(real_gtex_schema_json)
    bad11["class_weights"]["bladder"] = float('inf')
    with pytest.raises(ValueError, match="Invalid weight"):
        parse_gtex_class_mapping(bad11)
        
    # 12. compte de classe manquant
    bad12 = copy.deepcopy(real_gtex_schema_json)
    del bad12["train_counts"]["bladder"]
    with pytest.raises(ValueError, match="exactly 11 classes"):
        parse_gtex_class_mapping(bad12)
        
    # 13. compte négatif
    bad13 = copy.deepcopy(real_gtex_schema_json)
    bad13["train_counts"]["bladder"] = -5
    with pytest.raises(ValueError, match="non-negative integer"):
        parse_gtex_class_mapping(bad13)

from unittest.mock import patch

@pytest.fixture
def tiny_dataset(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    
    # Class mapping for testing (must be valid for parser)
    import json
    class_mapping = {
        "label_to_index": {
            "bladder": 0, "brain": 1, "cerebellum": 2, "kidney": 3,
            "liver": 4, "lung": 5, "muscle": 6, "oesophagus": 7,
            "pancreas": 8, "spleen": 9, "testis": 10
        },
        "index_to_label": {
            "0": "bladder", "1": "brain", "2": "cerebellum", "3": "kidney",
            "4": "liver", "5": "lung", "6": "muscle", "7": "oesophagus",
            "8": "pancreas", "9": "spleen", "10": "testis"
        },
        "class_weights": {
            "bladder": 1.0, "brain": 1.0, "cerebellum": 1.0, "kidney": 1.0,
            "liver": 1.0, "lung": 1.0, "muscle": 1.0, "oesophagus": 1.0,
            "pancreas": 1.0, "spleen": 1.0, "testis": 1.0
        },
        "train_counts": {
            "bladder": 1, "brain": 0, "cerebellum": 0, "kidney": 0,
            "liver": 0, "lung": 0, "muscle": 0, "oesophagus": 0,
            "pancreas": 0, "spleen": 0, "testis": 0
        },
        "validation_counts": {
            "bladder": 1, "brain": 0, "cerebellum": 0, "kidney": 0,
            "liver": 0, "lung": 0, "muscle": 0, "oesophagus": 0,
            "pancreas": 0, "spleen": 0, "testis": 0
        },
        "test_counts": {
            "bladder": 1, "brain": 0, "cerebellum": 0, "kidney": 0,
            "liver": 0, "lung": 0, "muscle": 0, "oesophagus": 0,
            "pancreas": 0, "spleen": 0, "testis": 0
        }
    }
    with open(metadata_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f)
        
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame([{
            "image_path": f"{split}/bladder_0.png",
            "label": "bladder",
            "label_index": 0,
            "donor_id": f"donor_{split}"
        }])
        df.to_csv(metadata_dir / f"{split}.csv", index=False)
        
    return tmp_path

@pytest.fixture
def patch_counts():
    with patch("src.data.gtex_integrity.EXPECTED_COUNTS", {"train": 1, "validation": 1, "test": 1, "total": 3}), \
         patch("src.data.gtex_integrity.EXPECTED_CLASS_COUNTS", {"bladder": {"train": 1, "validation": 1, "test": 1}}):
        yield

def test_A_label_label_index_valid(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    report = audit_gtex_dataset(tiny_dataset)
    assert report["status"] == "PASS"
    
    # Test pipeline on this valid dataset
    from src.data.gtex_pipeline import create_gtex_dataset, validate_batch
    # create fake images to avoid FileNotFoundError in pipeline
    for split in ["train", "validation", "test"]:
        img_dir = tiny_dataset / split
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / "bladder_0.png"
        import numpy as np
        from PIL import Image
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(img_path)
        
    ds = create_gtex_dataset(tiny_dataset, "train", is_training=False, batch_size=1)
    validate_batch(ds)
    for images, labels in ds.take(1):
        assert labels.numpy()[0] == 0  # index of bladder

def test_B_legacy_class_class_id(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    for split in ["train", "validation", "test"]:
        csv_path = tiny_dataset / "metadata" / f"{split}.csv"
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"label": "class", "label_index": "class_id"})
        df.to_csv(csv_path, index=False)
        
    report = audit_gtex_dataset(tiny_dataset)
    assert report["status"] == "PASS"

def test_C_incorrect_label(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "label"] = "unknown_class"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Unknown label"):
        audit_gtex_dataset(tiny_dataset)

def test_D_incorrect_label_index(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "label_index"] = 5
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Label/index mismatch"):
        audit_gtex_dataset(tiny_dataset)

def test_E_label_class_identical(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df["class"] = df["label"]
    df.to_csv(csv_path, index=False)
    report = audit_gtex_dataset(tiny_dataset)
    assert report["status"] == "PASS"

def test_F_label_class_different(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df["class"] = "brain"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Both 'class' and 'label' columns exist but they are not identical"):
        audit_gtex_dataset(tiny_dataset)

def test_G_label_index_class_id_identical(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df["class_id"] = df["label_index"]
    df.to_csv(csv_path, index=False)
    report = audit_gtex_dataset(tiny_dataset)
    assert report["status"] == "PASS"

def test_H_label_index_class_id_different(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df["class_id"] = 1
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Both 'class_id' and 'label_index' columns exist but they are not identical"):
        audit_gtex_dataset(tiny_dataset)

def test_I_no_class_column(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["label"])
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Neither 'class' nor 'label' column found"):
        audit_gtex_dataset(tiny_dataset)

def test_J_no_index_column(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["label_index"])
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Neither 'class_id' nor 'label_index' column found"):
        audit_gtex_dataset(tiny_dataset)

def test_K_nan_label(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "label"] = pd.NA
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="NaN or empty string"):
        audit_gtex_dataset(tiny_dataset)

def test_L_out_of_bounds_index(tiny_dataset, patch_counts):
    from src.data.gtex_integrity import audit_gtex_dataset
    csv_path = tiny_dataset / "metadata" / "train.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "label_index"] = 15
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Index out of range"):
        audit_gtex_dataset(tiny_dataset)
