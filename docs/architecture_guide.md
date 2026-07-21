# Architecture Guide

> Complete guide to the repository structure, workflows, and design
> decisions for the histology-ai-classification project.

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [How to Run the Project](#how-to-run-the-project)
3. [Google Colab Workflow](#google-colab-workflow)
4. [DenseNet121 Workflow (Yassine)](#densenet121-workflow-yassine)
5. [ResNet50V2 Freedom (Mabena)](#resnet50v2-freedom-mabena)
6. [Why DenseNet121 for Histology](#why-densenet121-for-histology)
7. [Modular Architecture](#modular-architecture)
8. [Adding a New Model](#adding-a-new-model)
9. [Conditions for Comparison and Ensemble](#conditions-for-comparison-and-ensemble)

---

## Repository Structure

### Role of Each Directory

| Directory | Purpose |
|---|---|
| `configs/` | YAML configuration files. `common.yaml` is shared; each model has its own file. |
| `data/raw/` | Original unmodified dataset (gitignored). |
| `data/processed/` | Preprocessed data if needed (gitignored). |
| `data/external/` | External reference data (e.g., smartphone microscope test images). |
| `data/manifests/` | CSV/JSON metadata files (versioned). Shared test set, class mapping, DenseNet121 folds. |
| `docs/` | Project documentation: methodology, dataset, Colab guide, this architecture guide. |
| `experiments/` | Per-model experiment directories for logs and tracking. |
| `models/` | Model artifacts: checkpoints, saved models, exported formats (gitignored). |
| `notebooks/colab/` | Google Colab notebooks for GPU-based workflows. |
| `notebooks/exploration/` | Local exploration notebooks. |
| `reports/` | Generated outputs: figures, metrics, predictions (gitignored). |
| `scripts/` | CLI entry points (e.g., `prepare_data.py`). |
| `src/data/` | Data loading, manifest building, audit, splitting. |
| `src/models/` | Model architecture definitions (DenseNet121, ResNet50V2, ensemble). |
| `src/training/` | Training pipeline orchestration. |
| `src/evaluation/` | Evaluation and metrics computation. |
| `src/explainability/` | Grad-CAM and interpretability tools. |
| `src/utils/` | Shared utilities: config loader, seed, runtime detection. |
| `tests/` | Unit tests and structural tests. |
| `api/` | Future REST API backend (not developed yet). |
| `mobile/` | Future mobile application (not developed yet). |

### Configuration Architecture

```
configs/
├── common.yaml          ← Shared: seed, test_size, class mapping, metrics
├── densenet121.yaml     ← Yassine: model params + 5-fold CV validation
├── resnet50v2.yaml      ← Mabena: model params + her validation strategy
├── config.example.yaml  ← Reference template (original)
└── logging.yaml         ← Python logging configuration
```

The config loader (`src/utils/config.py`) merges `common.yaml` with the
model-specific file. Shared parameters are defined once; model-specific
parameters override or extend them.

### Data Manifest Architecture

```
data/manifests/
├── dataset_manifest.csv    ← Full dataset inventory (all images)
├── class_mapping.json      ← Shared class_name → class_id mapping
├── test_manifest.csv       ← 20% test set (shared by both models)
└── densenet121_folds.csv   ← 5-fold CV assignments (Yassine only)
```

**Mabena does NOT use `densenet121_folds.csv`.** She defines her own
validation split for ResNet50V2.

---

## How to Run the Project

### Local Development (CPU)

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the dataset audit (requires dataset)
python scripts/prepare_data.py \
    --dataset-root data/raw/Human_Histopathological_H_E_Stained_Nuclei_Images \
    --output-root .

# Run tests
pytest tests/ -v
```

### Compilation Check

```bash
python -m compileall src scripts
```

---

## Google Colab Workflow

1. **Open** `notebooks/colab/00_setup_and_dataset_audit.ipynb` in Colab
2. **Activate GPU** runtime (Runtime → Change runtime type → GPU)
3. **Mount Drive** to access the dataset and save outputs
4. **Configure paths** in the configuration cell
5. **Run all cells** — the notebook clones the repo, installs deps, runs audit

See `docs/colab_training.md` for detailed step-by-step instructions.

### Why Google Colab?

Yassine's machine has only an integrated GPU (no CUDA). DenseNet121
training requires a GPU, so Colab provides free GPU access. Source
code stays on GitHub; data and outputs persist on Google Drive.

---

## DenseNet121 Workflow (Yassine)

### Two-Phase Training

1. **Head training** — freeze DenseNet121 base, train only the
   classification head (10 epochs, LR=0.001)
2. **Fine-tuning** — unfreeze the full network, train with a low
   learning rate (40 epochs, LR=0.00001)

### 5-Fold Stratified Cross-Validation

The development set (80%) is divided into 5 stratified folds:

```
Development Set (80%)
├── Fold 0: ~16% validation, ~64% training
├── Fold 1: ~16% validation, ~64% training
├── Fold 2: ~16% validation, ~64% training
├── Fold 3: ~16% validation, ~64% training
└── Fold 4: ~16% validation, ~64% training
```

For each fold:
- 4 folds are used for training
- 1 fold is used for validation
- The model is trained from scratch each time
- Metrics are averaged across all 5 runs

This provides a robust estimate of model performance and reduces
the risk of lucky/unlucky validation splits.

### Final Evaluation

After cross-validation, the best hyperparameters are used to train
a final model on the **entire development set**. This model is
evaluated on the **shared test set** (20%).

---

## ResNet50V2 Freedom (Mabena)

Mabena develops ResNet50V2 independently. She:

- **Must use** the same test set (`test_manifest.csv`)
- **Must use** the same class mapping (`class_mapping.json`)
- **Must report** the same evaluation metrics

She is **free to choose**:

- Her validation strategy (holdout, k-fold, leave-one-out, etc.)
- Her training hyperparameters
- Her fine-tuning approach
- Her augmentation pipeline
- Her training environment (local GPU, Colab, cloud, etc.)

Her model-specific config is in `configs/resnet50v2.yaml`, marked
with `TODO` comments for her to customize.

---

## Why DenseNet121 for Histology

DenseNet121 is well-suited for histological image classification for
several reasons:

### Feature Reuse Through Dense Connections

DenseNet connects each layer to every subsequent layer (dense blocks).
This creates:
- **Efficient feature propagation** — gradients flow directly to early
  layers, reducing vanishing gradient problems
- **Feature reuse** — the network reuses low-level features (edges,
  textures) alongside high-level patterns

### Relevance to Histology

H&E stained tissue images have:
- **Fine-grained textures** (cell morphology, nuclear staining patterns)
- **Hierarchical structures** (nuclei → cells → tissue architecture)
- **Subtle inter-class differences** (some tissues look very similar)

DenseNet's dense connections help the model simultaneously leverage:
- Low-level features (staining intensity, nuclear shape)
- Mid-level features (cell arrangements, gland structures)
- High-level features (tissue architecture patterns)

### Transfer Learning Advantage

DenseNet121 pre-trained on ImageNet provides strong initial features
for medical imaging despite the domain difference. The two-phase
training (freeze → fine-tune) adapts these features efficiently.

### Parameter Efficiency

DenseNet121 has ~8M parameters (vs. ~25M for ResNet50V2), making it:
- Faster to train on Colab's limited GPU time
- Less prone to overfitting on our medium-sized dataset

---

## Modular Architecture

The repository is designed as a **modular pipeline** where each
component can be developed, tested, and swapped independently:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data Layer  │───▶│ Model Layer  │───▶│  Eval Layer  │
│  src/data/   │    │ src/models/  │    │ src/eval/     │
└──────────────┘    └──────────────┘    └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Manifests   │    │  Checkpoints │    │   Reports    │
│ data/manif./ │    │   models/    │    │  reports/    │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Why Modular?

1. **Independent development** — Yassine and Mabena work on separate
   model files without conflicts
2. **Shared infrastructure** — data loading, evaluation, and utilities
   are written once and used by both
3. **Testability** — each module has its own tests
4. **Extensibility** — new models can be added without touching existing code

---

## Adding a New Model

To add a third architecture (e.g., EfficientNetB3):

### 1. Create the model file

```python
# src/models/efficientnetb3.py
def build_efficientnetb3(num_classes, input_shape=(224, 224, 3), ...):
    ...
```

### 2. Create the config file

```yaml
# configs/efficientnetb3.yaml
model:
  architecture: EfficientNetB3
  weights: imagenet
  ...
training:
  ...
validation:
  strategy: ...  # your choice
```

### 3. Create the experiment directory

```
experiments/efficientnetb3/.gitkeep
```

### 4. Use the shared infrastructure

- Use `data/manifests/test_manifest.csv` for evaluation
- Use `data/manifests/class_mapping.json` for class IDs
- Use `src/utils/config.py` to load `common.yaml` + your config
- Use `src/utils/seed.py` for reproducibility

### 5. Report the same metrics

Evaluate on the shared test set using the metrics defined in
`configs/common.yaml`.

---

## Conditions for Comparison and Ensemble

### Comparison Requirements

For a fair comparison between DenseNet121 and ResNet50V2:

| Requirement | Status |
|---|---|
| Same test set | ✓ `test_manifest.csv` |
| Same class mapping | ✓ `class_mapping.json` |
| Same evaluation metrics | ✓ Defined in `common.yaml` |
| Same original dataset | ✓ Not modified |
| No test set leakage | ✓ Test never used for tuning |

### Ensemble Prerequisites

Before building the ensemble (DenseNet121 + ResNet50V2):

1. Both individual models must be **fully trained**
2. Both must be **evaluated on the shared test set**
3. Both must produce **class probability vectors** (softmax outputs)
4. Performance of each model must be **documented**

### Ensemble Strategies (Planned)

- **Simple averaging** — average the softmax outputs
- **Weighted averaging** — weight by individual model performance
- **Stacking** — train a meta-learner on the individual predictions

The ensemble will be developed collaboratively after both models
are individually validated.
