# Histology AI Classification

> AI-Powered Learning Assistant for Medical Students Studying Histology

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚠️ Disclaimer

This application is intended **exclusively** for educational use. It is **not** a medical diagnostic device.

---

## Objective

A deep learning system to classify 22 human histopathological tissue types from H&E-stained microscope slides. Designed as an educational tool to help medical students identify tissue types.

---

## Current Status

| Component | Status |
|---|---|
| Dataset (432 images, 22 classes) | ✅ Ready |
| Class mapping (alphabetical, 0–21) | ✅ Ready |
| Manifest (`original_22_dataset_manifest.csv`) | ✅ Ready |
| 5-Fold CV manifest (`densenet121_folds.csv`) | ✅ Ready |
| Data pipeline (`pipeline.py`) | ✅ Fixed |
| DenseNet121 architecture | ✅ Ready |
| Training orchestrator (5-fold) | ✅ Ready |
| Evaluation metrics | ✅ Ready |
| Google Colab notebooks | ✅ Ready |
| Full training on Colab GPU | 🔜 Next step |

---

## Project Structure

```
histology-ai-classification/
├── configs/             # YAML configuration files
│   └── densenet121.yaml
├── data/
│   ├── manifests/       # Shared metadata (versioned in Git)
│   │   ├── class_mapping.json
│   │   ├── original_22_dataset_manifest.csv
│   │   └── densenet121_folds.csv
│   └── raw/             # Raw images (gitignored)
├── docs/
│   └── project_guide.md # Full technical guide
├── notebooks/colab/     # Google Colab execution notebooks
│   ├── 01_dataset_preparation.ipynb
│   └── 02_train_densenet121.ipynb
├── scripts/
│   └── create_original_22_class_dataset.py
├── src/
│   ├── data/            # EDA, manifest, folds, pipeline
│   ├── evaluation/      # Accuracy, F1, confusion matrix
│   ├── models/          # DenseNet121 architecture
│   ├── training/        # 5-fold cross-validation loop
│   └── utils/           # Config loader, seed, runtime
├── reports/             # Generated metrics and figures
├── models/              # Saved model checkpoints (gitignored)
└── tests/               # Automated tests (pytest)
```

---

## Team

| Role | Person | Architecture | Validation Strategy |
|---|---|---|---|
| DenseNet121 developer | Yassine | DenseNet121 | Stratified 5-Fold CV |
| Second architecture | Teammate | ResNet50V2 (or other) | Teammate's choice |

---

## Installation (Local)

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

---

## Dataset Reconstruction

The raw dataset is **not included in Git**. Obtain it from the official source:
> **NuInsSeg** — https://doi.org/10.5281/zenodo.10518852

Then reconstruct the 22-class subset:
```bash
python scripts/create_original_22_class_dataset.py \
    --source-root "path/to/NuInsSeg" \
    --destination-root "data/raw/nuinsseg_human_22_original"
```

---

## Run Exploratory Data Analysis

```bash
python -m src.data.explore_dataset
# or with explicit path:
python -m src.data.explore_dataset --data-dir data/raw/nuinsseg_human_22_original
```

---

## Create 5-Fold Manifest

```bash
python -m src.data.create_folds \
    --manifest-path data/manifests/original_22_dataset_manifest.csv \
    --output-path data/manifests/densenet121_folds.csv \
    --num-folds 5 \
    --seed 42
```

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Training on Google Colab

1. Open `notebooks/colab/01_dataset_preparation.ipynb` to prepare data and folds.
2. Open `notebooks/colab/02_train_densenet121.ipynb` to train DenseNet121.

Both notebooks call `src/` for all logic — no duplicate code.

---

## Future Enhancements

- Grad-CAM visual explanations
- Ensemble (DenseNet121 + ResNet50V2)
- REST API backend
- Web / Mobile frontend
