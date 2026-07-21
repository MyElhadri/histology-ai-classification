# Histology AI Classification

> AI-Powered Learning Assistant for Medical Students Studying Histology

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

---

## Educational Goal

This project develops a deep learning system to classify human histopathological tissue types from H&E-stained microscope slides. It is designed as an educational tool to help medical students identify tissue types and learn their characteristics.

**⚠️ Disclaimer:** This application is intended *exclusively* for educational use. It is not a medical diagnostic device.

## Architecture & Models

The repository is designed to be modular and scalable, allowing the integration of multiple deep learning architectures for histology classification. The training pipeline utilizes a **5-fold stratified cross-validation** strategy to ensure robust evaluation across the dataset.

Key components of the architecture include:
- **Data Pipeline:** Standardized manifest generation and cross-validation fold assignments.
- **Model Training:** Automated workflows for training various architectures (e.g., DenseNet, ResNet).
- **Evaluation:** Unified metrics computation (Accuracy, Macro F1, etc.) for fair model comparison.

## Training Environment

The models are configured to be trained on **Google Colab** using GPU runtimes to accelerate the deep learning processes. 
Data, manifests, and model checkpoints are saved to **Google Drive** for persistence and collaboration.

## Repository Structure

```
histology-ai-classification/
├── configs/             # YAML configuration files for models
├── data/
│   ├── manifests/       # Generated CSV metadata and folds
│   └── raw/             # Raw dataset (gitignored)
├── notebooks/colab/     # Google Colab notebooks for execution
├── src/
│   ├── data/            # Dataset exploration, manifests, pipeline
│   ├── evaluation/      # Metrics and scoring
│   ├── models/          # Model architecture definitions
│   └── training/        # 5-fold CV training loop orchestrators
├── models/              # Saved model checkpoints
├── reports/             # Generated metrics and summaries
└── tests/               # Unit tests
```

## How to Run

1. Place the dataset in `data/raw/Human_Histopathological_H_E_Stained_Nuclei_Images`.
2. Open the Colab notebooks in `notebooks/colab/`:
   - `01_dataset_preparation.ipynb` to explore the data and generate cross-validation folds.
   - Run the relevant model training notebook (e.g., `02_train_densenet121.ipynb`) to execute the training loop.

## Future Enhancements

The following features are planned for integration as the project evolves:
- **Additional Architectures** for comparative analysis and benchmarking.
- **Grad-CAM Interpretability** for visual explanations of model predictions.
- **Ensemble Strategies** combining multiple architectures for improved robustness.
- **REST API Backend** for serving predictions in real-time.
- **Web Frontend** and **Mobile App** interfaces for end-user interaction.
