# Data Directory

## Structure

```
data/
├── raw/                         # Original dataset (gitignored)
├── processed/                   # Preprocessed data (gitignored)
├── external/                    # External reference data (gitignored)
│   └── smartphone_microscope/   # Future: test images from phone cameras
├── manifests/                   # Metadata files (versioned)
│   ├── dataset_manifest.csv     # Full image inventory
│   ├── class_mapping.json       # Shared class → ID mapping
│   ├── test_manifest.csv        # 20% shared test set
│   └── densenet121_folds.csv    # 5-fold CV (Yassine only)
└── README.md
```

## Important Rules

### Raw Data Is Not Versioned

The raw dataset (`data/raw/`) is excluded from Git via `.gitignore`.
It must be downloaded separately and placed in the correct location.

### Original Images Are Never Modified

No pipeline step modifies the original image files. All transformations
(resizing, normalization, augmentation) are applied **on-the-fly** during
training, not saved to disk.

### Manifests Can Be Versioned

The files in `data/manifests/` are small metadata files (CSV, JSON)
that **should be committed to Git**. They contain:
- Relative paths (no personal absolute paths)
- Class mappings
- Split assignments

This allows both team members to use the exact same test set and
class mapping without re-generating them.

### Augmentations Are Online Only

Data augmentation is performed **at training time** (online), not
pre-computed and saved to disk. This avoids:
- Disk space waste
- Data leakage across splits
- Difficulty tracking which augmentations were applied

### Shared vs. Individual Manifests

| File | Scope | Description |
|---|---|---|
| `dataset_manifest.csv` | Shared | Full dataset inventory |
| `class_mapping.json` | Shared | Class name → class ID |
| `test_manifest.csv` | Shared | 20% test set for both models |
| `densenet121_folds.csv` | Yassine | 5-fold CV for DenseNet121 |


