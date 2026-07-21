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
Return confidence score
        ↓
Display educational information about the tissue
```

### Models

Two models are developed independently and later combined:

| Model | Architect | Status |
|---|---|---|
| DenseNet121 | Yassine | In Progress |
| ResNet50V2 | Teammate | In Progress |
| Ensemble | Collaborative | Planned |

## Dataset

**Human Histopathological H&E Stained Nuclei Images** from Kaggle.

> The raw dataset should be placed in `data/raw/` and is excluded from version control.

## Project Structure

```
histology-ai-classification/
│
├── configs/                    # Configuration files (YAML)
├── data/                       # All data (excluded from git)
│   ├── raw/                    # Original unmodified dataset
│   ├── processed/              # Cleaned and preprocessed data
│   ├── augmented/              # Augmented training data
│   └── external/               # External reference data
│
├── docs/                       # Project documentation
├── experiments/                # Experiment logs and tracking
├── models/                     # Saved models and checkpoints
│   ├── saved/                  # Final trained models
│   ├── checkpoints/            # Training checkpoints
│   └── exports/                # Exported models (TFLite, ONNX)
│
├── notebooks/                  # Jupyter notebooks (exploration only)
├── reports/                    # Generated reports and figures
│   ├── figures/                # Plots and visualizations
│   └── metrics/                # Evaluation metrics
│
├── src/                        # Source code
│   ├── data/                   # Data loading and processing
│   ├── models/                 # Model architectures
│   ├── training/               # Training pipelines
│   ├── evaluation/             # Evaluation and metrics
│   ├── visualization/          # Plotting and visual analysis
│   ├── callbacks/              # Custom training callbacks
│   └── utils/                  # Shared utilities
│
├── tests/                      # Unit and integration tests
├── api/                        # Future backend API
├── mobile/                     # Future mobile application
│
├── configs/                    # Configuration files
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata and build config
├── LICENSE                     # MIT License
└── .gitignore                  # Git exclusions
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<org>/histology-ai-classification.git
cd histology-ai-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example configuration and adjust for your environment:

```bash
cp configs/config.example.yaml configs/config.yaml
```

## Usage

> **Note:** This section will be updated as training pipelines are implemented.

## Reproducibility

This project follows strict reproducibility guidelines:

- All random seeds are configurable via `configs/config.yaml`
- Experiment parameters are logged in `experiments/`
- Data processing pipelines are deterministic when seeded
- Model checkpoints are versioned

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Numerical | NumPy |
| Data Handling | Pandas |
| Visualization | Matplotlib |
| ML Utilities | Scikit-learn |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Authors

- **Yassine** — DenseNet121 Architecture
- **Teammate** — ResNet50V2 Architecture
