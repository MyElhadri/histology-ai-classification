# Project Guide — Histology AI Classification

> Educational deep learning project for histopathological tissue classification.
> **Not a medical diagnostic tool.**

---

## 1. Educational Objective

This project trains a DenseNet121 model to classify 22 types of human tissue from H&E-stained microscope images. It is a paired academic project (binôme) — each student implements their own model and they will later be compared or combined.

---

## 2. Role of Each Folder

```
histology-ai-classification/
├── configs/         → YAML configuration files for each model
├── data/
│   ├── manifests/   → Shared metadata: manifest CSV, class mapping, fold assignments
│   └── raw/         → Raw images (gitignored, never modified)
├── docs/            → Technical documentation
├── models/          → Saved checkpoints (gitignored)
├── notebooks/colab/ → Google Colab notebooks for execution
├── reports/         → Generated metrics and figures
├── scripts/         → CLI scripts for dataset preparation
├── src/
│   ├── data/        → EDA, manifest building, fold creation, tf.data pipeline
│   ├── evaluation/  → Metrics computation (accuracy, F1, confusion matrix)
│   ├── models/      → DenseNet121 architecture
│   ├── training/    → 5-fold cross-validation orchestrator
│   └── utils/       → Config loader, seed setter, runtime detection
└── tests/           → Pytest unit tests
```

---

## 3. How to Reconstruct the Dataset from NuInsSeg

The dataset used in this project (`data/raw/nuinsseg_human_22_original`) is a **clean subset** of the original NuInsSeg dataset. It is **not published on GitHub** because:
- Raw image data is large (432 images × 512×512px).
- Researchers should always obtain data directly from its official source to preserve provenance.

**Step 1 — Download NuInsSeg:**
The original dataset is available at:
> https://doi.org/10.5281/zenodo.10518852

**Step 2 — Extract 22 human classes:**
```bash
python scripts/create_original_22_class_dataset.py \
    --source-root "/path/to/NuInsSeg" \
    --destination-root "data/raw/nuinsseg_human_22_original"
```

This script:
- Copies only the 22 human tissue classes (excluded: mouse tissues, placenta).
- Preserves original filenames and pixel values exactly (no modification, no resizing).
- Generates `data/manifests/original_22_dataset_manifest.csv` and `data/manifests/class_mapping.json`.

---

## 4. How to Use `original_22_dataset_manifest.csv`

This CSV contains one row per image with the following columns:

| Column | Description |
|---|---|
| `image_id` | Unique identifier (`img_nuinsseg_XXXXXX`) |
| `image_path` | Path relative to project root |
| `class_name` | Short class name (e.g. `brain`, `liver`) |
| `class_id` | Integer label (0-21, alphabetically ordered) |
| `original_filename` | Name in original NuInsSeg |
| `width`, `height` | 512, 512 |
| `channels` | 3 (RGB) or 4 (RGBA — 1 image) |
| `file_hash` | SHA-256 of the copied file |

The manifest can be used by **both** Yassine (DenseNet121) and the teammate (ResNet50V2 or other), providing a shared ground truth for class labels.

---

## 5. How the 5 Folds for DenseNet121 Work

Yassine uses **Stratified 5-Fold Cross-Validation** from scikit-learn:
- 432 images are split into 5 folds with approximately equal class distributions.
- For fold *k*, validation uses images where `fold == k`, training uses `fold != k`.
- A new model is initialized fresh for every fold.
- Results are averaged across all 5 folds.

The fold assignments are stored in `data/manifests/densenet121_folds.csv`.

This file is generated once with:
```bash
python -m src.data.create_folds \
    --manifest-path data/manifests/original_22_dataset_manifest.csv \
    --output-path data/manifests/densenet121_folds.csv \
    --num-folds 5 \
    --seed 42
```

---

## 6. How `pipeline.py` Works

`src/data/pipeline.py` creates a `tf.data.Dataset` for training or validation:

1. Reads `densenet121_folds.csv`.
2. Filters rows by fold number and `is_training` flag.
3. Resolves each `image_path` **relative to the project root** (does NOT duplicate the path prefix).
4. Loads images with `tf.io.read_file` + `tf.image.decode_image(channels=3)` — RGBA images are automatically converted to RGB.
5. Resizes to **224×224** (from 512×512).
6. Applies augmentations (flips, brightness) **only when `is_training=True`**.
7. Batches and prefetches.

Note: `densenet.preprocess_input` is applied **inside the model graph** (in `densenet121.py`), not in the pipeline.

---

## 7. How to Run in Google Colab

1. Open `notebooks/colab/01_dataset_preparation.ipynb` in Colab.
2. Configure `DRIVE_BASE`, `REPO_URL`, and `NUINSSEG_SOURCE` in the first cells.
3. Run all cells to: mount Drive, clone the repo, reconstruct the dataset, run EDA, and create folds.
4. Open `notebooks/colab/02_train_densenet121.ipynb`.
5. Run all cells to: verify GPU, load config, and train.

---

## 8. Shared Resources for the Binôme

These files are tracked in Git and are meant to be used by **both team members**:

| File | Purpose |
|---|---|
| `data/manifests/class_mapping.json` | Shared class → integer label mapping |
| `data/manifests/original_22_dataset_manifest.csv` | Shared image inventory |
| `scripts/create_original_22_class_dataset.py` | Dataset reconstruction script |
| `src/evaluation/evaluate.py` | Shared evaluation metrics |

---

## 9. How the Teammate Can Add a Model Without Breaking DenseNet121

1. Add a new YAML config in `configs/` (e.g. `resnet50v2.yaml`).
2. Create a new model file in `src/models/` (e.g. `resnet50v2.py`).
3. Create a new training script or folder in `src/training/`.
4. Use the same `class_mapping.json` and `original_22_dataset_manifest.csv`.
5. Choose your own validation strategy — it does **not** have to match DenseNet121's 5-fold CV.
6. Save results in a separate subdirectory (e.g. `reports/resnet50v2/`).
7. Never modify `configs/densenet121.yaml` or `src/models/densenet121.py`.

---

## 10. Class Imbalance Strategy

The dataset is highly imbalanced (e.g., 9 `muscle` images vs 47 `oesophagus` images).

- **Data Augmentation vs. Balancing:** Augmentation (flips, brightness) creates variations of existing images to prevent overfitting, but it does NOT change the ratio between classes.
- **Why Class Weights?** We assign a higher penalty (weight) to errors made on minority classes. This forces the model to pay equal attention to all classes during training.
- **Why Per-Fold?** Since each fold has a slightly different training set, the exact class counts change. Weights must be recalculated based *only* on the training images for that specific fold.
- **Why No Oversampling Yet?** Oversampling duplicates minority class images. While effective, it increases training time and can cause overfitting. We start with class weights as a lighter, standard approach.
- **Validation Remains Unchanged:** The validation set is never balanced or augmented. It must reflect the real, unmodified data distribution to provide an honest evaluation of the model.
