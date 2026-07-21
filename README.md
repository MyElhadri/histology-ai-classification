# Histology AI Classification (DenseNet121)

> AI-Powered Learning Assistant for Medical Students Studying Histology

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

---

## Educational Goal

This project develops a deep learning system to classify human histopathological tissue types from H&E-stained microscope slides. It is designed as an educational tool to help medical students identify tissue types and learn their characteristics.

**⚠️ Disclaimer:** This application is intended *exclusively* for educational use. It is not a medical diagnostic device.

## Architecture

Currently, the project implements a **DenseNet121** architecture developed by **Yassine**. 
It utilizes a **5-fold stratified cross-validation** strategy to ensure robust evaluation across the dataset.

## Training Environment

Due to hardware constraints, the models are trained on **Google Colab** using GPU runtimes. 
Data, manifests, and model checkpoints are saved to **Google Drive** for persistence.

## Repository Structure

```
histology-ai-classification/
├── configs/             # YAML configuration (densenet121.yaml)
├── data/
│   ├── manifests/       # Generated CSV metadata and folds
│   └── raw/             # Raw dataset (gitignored)
├── notebooks/colab/     # Google Colab notebooks for execution
├── src/
│   ├── data/            # Dataset exploration, manifests, pipeline
│   ├── evaluation/      # Metrics and scoring
│   ├── models/          # DenseNet121 definition
│   └── training/        # 5-fold CV training loop
├── models/              # Saved model checkpoints
├── reports/             # Generated metrics
└── tests/               # Unit tests
```

## How to Run

1. Place the dataset in `data/raw/Human_Histopathological_H_E_Stained_Nuclei_Images`.
2. Open the Colab notebooks in `notebooks/colab/`:
   - `01_dataset_preparation.ipynb` to explore the data and generate folds.
   - `02_train_densenet121.ipynb` to execute the 5-fold training loop.

## Future Work

The following features are planned for future development and are *not* currently implemented in this simplified branch:
- **ResNet50V2** (to be developed by the teammate).
- **Grad-CAM** interpretability for visual explanations.
- **Ensemble** model combining both architectures.
- **REST API** backend for serving predictions.
- **Web Frontend** and **Mobile App** interfaces.
