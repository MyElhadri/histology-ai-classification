# Data Directory

## Structure

```
data/
├── raw/
│   ├── Human_Histopathological_H_E_Stained_Nuclei_Images/ # Legacy full dataset
│   └── nuinsseg_human_22_original/ # Cleaned 22-class dataset
├── manifests/                   # Metadata files (versioned)
│   ├── original_22_dataset_manifest.csv # New 22-class inventory
│   ├── class_mapping.json       # Deterministic class mapping
│   ├── dataset_manifest.csv     # Legacy inventory
│   └── densenet121_folds.csv    # 5-fold CV assignments
└── README.md
```

## Dataset Origin and Selection

- **NuInsSeg** is conserved as the original source dataset.
- `nuinsseg_human_22_original` is a carefully selected subset of the NuInsSeg dataset.
- **Exclusions:** The class `placenta` and all murine (mouse) tissues have been strictly excluded from this experiment.
- **No Modifications:** The original images are never modified. No resizing, formatting, or augmentations have been applied to the files on disk. 

## Important Rules

### Raw Data Is Not Versioned
The raw datasets (`data/raw/`) are excluded from Git via `.gitignore`.
They must be downloaded and extracted separately.

### Original Images Are Never Modified
No pipeline step modifies the original image files. All transformations (resizing, normalization, augmentation) are applied **on-the-fly** during training.

### Manifests Can Be Versioned
The files in `data/manifests/` are small metadata files (CSV, JSON) that **should be committed to Git**. They contain relative paths (no personal absolute paths) to ensure reproducibility across different environments.
