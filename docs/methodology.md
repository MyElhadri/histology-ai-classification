# Methodology

> Evaluation protocol and experimental methodology for the histology
> classification project.

## Team Responsibilities

| Model | Developer | Validation Strategy |
|---|---|---|
| DenseNet121 | Yassine | 5-fold stratified cross-validation |
| ResNet50V2 | Mabena | Mabena's choice (see below) |
| Ensemble | Collaborative | After individual evaluation |

## What Is Shared

The following elements are **identical** for both models to ensure
fair comparison:

1. **Dataset** — same original images, same class mapping
2. **Test set** — 20% stratified holdout (`test_manifest.csv`)
3. **Class mapping** — deterministic alphabetical order (`class_mapping.json`)
4. **Evaluation metrics** — see below

## What Is Independent

Each team member is free to choose:

- **Validation strategy** — Yassine uses 5-fold stratified CV;
  Mabena can use holdout, k-fold, or any other approach
- **Training hyperparameters** — batch size, learning rate, optimizer, etc.
- **Fine-tuning strategy** — which layers to freeze/unfreeze
- **Augmentation strategy** — types and intensity of augmentations

## Split Protocol

### Shared Test Set (20%)

```
Full dataset (100%)
    ├── Development set (80%) — for training and validation
    └── Test set (20%)        — shared, locked, never used for tuning
```

The test set is created using `sklearn.model_selection.train_test_split`
with `stratify` and `random_state=42`.  The result is saved in
`data/manifests/test_manifest.csv`.

**Rule:** The test set must NEVER be used to select hyperparameters,
choose architectures, or make any training decision.

### DenseNet121 Validation (Yassine)

The 80% development set is split into 5 stratified folds using
`sklearn.model_selection.StratifiedKFold` with `random_state=42`.
The fold assignments are saved in `data/manifests/densenet121_folds.csv`.

### ResNet50V2 Validation (Mabena)

Mabena is free to define her own validation strategy on the 80%
development set.  She may create her own split file or use a
different approach entirely.  The only constraint is that the
final evaluation is performed on the **shared test set**.

## Evaluation Metrics

All models are evaluated using these metrics (computed on the shared
test set):

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions / total |
| Balanced Accuracy | Mean per-class recall |
| Macro F1 | F1 averaged across classes (unweighted) — **primary** |
| Weighted F1 | F1 weighted by class support |
| Precision | Macro-averaged precision |
| Recall | Macro-averaged recall |
| Top-3 Accuracy | Correct class in top 3 predictions |

**Primary metric:** `macro_f1` — chosen because it handles class
imbalance better than raw accuracy.

## Conditions for Model Comparison

To compare DenseNet121 and ResNet50V2 fairly:

1. Both must be evaluated on the **same test set**
2. Both must use the **same class mapping**
3. Both must report the **same metrics**

The ensemble model will be developed only after both individual
models are trained and evaluated.

## Reproducibility

- **Seed:** 42 (set globally via `src/utils/seed.py`)
- **Splits:** Generated deterministically and saved as CSV
- **Augmentations:** Applied online (on-the-fly) during training
