# Histology AI Classification

> AI-Powered Learning Assistant for Medical Students Studying Histology

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This research project develops a deep learning system for automatic classification of human histopathological tissue types from H&E-stained microscope slide images. The system is designed to serve as an educational tool that helps medical students identify tissue types and learn about their characteristics.

### System Architecture

```
Medical Student
        ↓
Take a picture of a microscope slide
        ↓
Deep Learning Model
        ↓
Predict tissue type
        ↓
Return confidence score + Top 3 predictions
        ↓
Display Grad-CAM explanation + educational information
```

### Models

Two models are developed independently and later combined:

| Model | Developer | Validation Strategy | Status |
|---|---|---|---|
| DenseNet121 | Yassine | 5-fold stratified CV | In Progress |
| ResNet50V2 | Mabena | Mabena's choice | In Progress |
| Ensemble | Collaborative | — | Planned |

Both models share the **same test set** (20%), **same class mapping** (22 classes), and **same evaluation metrics**. Each developer is free to choose their own validation strategy and training hyperparameters.

## Dataset

**Human Histopathological H&E Stained Nuclei Images** from Kaggle — 22 tissue classes.

> The raw dataset should be placed in `data/raw/` and is excluded from version control.

## Project Structure

```
histology-ai-classification/
│
├── configs/                    # Configuration files (YAML)
│   ├── common.yaml             # Shared parameters (test split, metrics, classes)
│   ├── densenet121.yaml        # DenseNet121 config (Yassine)
│   ├── resnet50v2.yaml         # ResNet50V2 config (Mabena)
│   ├── config.example.yaml     # Reference template
│   └── logging.yaml            # Logging configuration
│
├── data/                       # All data
│   ├── raw/                    # Original unmodified dataset (gitignored)
│   ├── processed/              # Preprocessed data (gitignored)
│   ├── external/               # External reference data (gitignored)
│   │   └── smartphone_microscope/
│   ├── manifests/              # Metadata files (versioned)
│   │   ├── dataset_manifest.csv
│   │   ├── class_mapping.json
│   │   ├── test_manifest.csv
│   │   └── densenet121_folds.csv
│   └── README.md
│
├── docs/                       # Project documentation
│   ├── architecture_guide.md   # Repository guide and workflows
│   ├── dataset.md              # Dataset description
│   ├── methodology.md          # Evaluation protocol
│   └── colab_training.md       # Google Colab setup guide
│
├── experiments/                # Experiment logs and tracking
│   ├── densenet121/
│   └── resnet50v2/
│
├── models/                     # Saved models and checkpoints (gitignored)
│   ├── checkpoints/
│   ├── saved/
│   └── exports/
│
├── notebooks/                  # Jupyter notebooks
│   ├── exploration/            # Local exploration
│   └── colab/                  # Google Colab notebooks
│       └── 00_setup_and_dataset_audit.ipynb
│
├── reports/                    # Generated reports and figures (gitignored)
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── scripts/                    # CLI entry points
│   └── prepare_data.py         # Dataset audit and split pipeline
│
├── src/                        # Source code
│   ├── data/                   # Data loading, manifest, audit, splitting
│   ├── models/                 # Model architectures
│   ├── training/               # Training pipelines
│   ├── evaluation/             # Evaluation and metrics
│   ├── explainability/         # Grad-CAM and interpretability
│   ├── callbacks/              # Custom training callbacks
│   ├── visualization/          # Plotting utilities
│   └── utils/                  # Shared utilities (config, seed, runtime)
│
├── tests/                      # Unit and structural tests
├── api/                        # Future backend API
├── mobile/                     # Future mobile application
│
├── requirements.txt            # Python dependencies
├── requirements-colab.txt      # Additional Colab dependencies
├── pyproject.toml              # Project metadata and build config
├── LICENSE                     # MIT License
└── .gitignore                  # Git exclusions
```

## Installation

```bash
# Clone the repository
git clone https://github.com/MyElhadri/histology-ai-classification.git
cd histology-ai-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The project uses a layered configuration system:

```bash
# Common parameters are in configs/common.yaml (shared)
# Model-specific parameters are in configs/densenet121.yaml or configs/resnet50v2.yaml
# For local overrides, copy the example:
cp configs/config.example.yaml configs/config.yaml
```

## Usage

### Dataset Audit

```bash
python scripts/prepare_data.py \
    --dataset-root data/raw/Human_Histopathological_H_E_Stained_Nuclei_Images \
    --output-root .
```

This produces the dataset manifest, class mapping, shared test set, and DenseNet121 fold assignments.

### Google Colab

See `docs/colab_training.md` for detailed Colab setup instructions, or open `notebooks/colab/00_setup_and_dataset_audit.ipynb` directly in Colab.

### Tests

```bash
pytest tests/ -v
```

## Training Environment

Local development is performed on CPU because the development computer does not have a CUDA-compatible dedicated GPU and only provides an integrated GPU.

DenseNet121 training and fine-tuning are performed on Google Colab using a GPU runtime.

The source code is stored on GitHub. Google Drive is used for persistent storage of the dataset, checkpoints, saved models, metrics, and generated reports.

## Reproducibility

This project follows strict reproducibility guidelines:

- **Seed:** 42 — set globally across Python, NumPy, and TensorFlow
- **Splits:** Generated deterministically and saved as CSV manifests
- **Test set:** Shared and locked — never used for hyperparameter selection
- **Augmentations:** Applied online during training (not saved to disk)

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV, Pillow |
| Numerical | NumPy |
| Data Handling | Pandas |
| Visualization | Matplotlib |
| ML Utilities | Scikit-learn |

## Educational Disclaimer

This application is intended exclusively for educational use during histology practical sessions. It is not a medical diagnostic device and must not be used for clinical decision-making.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Authors

- **Yassine** — DenseNet121 Architecture (5-fold stratified CV)
- **Mabena** — ResNet50V2 Architecture (validation strategy: her choice)
